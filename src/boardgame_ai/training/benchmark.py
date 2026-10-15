"""Repeated, synchronized stage timings against an explicitly saved PPO source."""

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from unittest.mock import patch

import torch

from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.training import rollout as rollout_module
from boardgame_ai.training.checkpoint import save_ppo_checkpoint
from boardgame_ai.training.ppo import PPOConfig, PPOTrainer, prepare_ppo_batch
from boardgame_ai.training.rollout import PlayerTrajectory, collect_tic_tac_toe_rollout
from boardgame_ai.training.run import _initial_model, _seed


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def compare_trainers(first: PPOTrainer, second: PPOTrainer) -> None:
    for key, tensor in first.model.state_dict().items():
        torch.testing.assert_close(
            tensor, second.model.state_dict()[key], rtol=0, atol=0
        )
    for key, state in first.optimizer.state_dict()["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(
                tensor,
                second.optimizer.state_dict()["state"][key][name],
                rtol=0,
                atol=0,
            )
    assert torch.equal(first.generator.get_state(), second.generator.get_state())
    assert first.optimizer_steps == second.optimizer_steps


def _run(
    trainer: PPOTrainer,
    output: Path,
    *,
    warmup: int,
    measured: int,
    games: int,
    seed: int,
) -> list[dict[str, Any]]:
    device = str(trainer.device)
    records = []
    for update in range(warmup + measured):
        inference = 0.0
        original = rollout_module._predict

        def timed_predict(*args: Any, _predict: Any = original, **kwargs: Any) -> Any:
            nonlocal inference
            start = perf_counter()
            result = _predict(*args, **kwargs)
            inference += perf_counter() - start
            return result

        _sync(device)
        begin = perf_counter()
        policy = deepcopy(trainer.model).cpu().eval()
        segments: list[PlayerTrajectory] = []
        with patch.object(rollout_module, "_predict", timed_predict):
            for local in range(games):
                index = update * games + local
                rollout = collect_tic_tac_toe_rollout(
                    TicTacToeEngine(),
                    (policy, policy),
                    max_steps=9,
                    engine_seed=_seed(seed, index, "engine"),
                    sampling_seeds=(
                        _seed(seed, index, "seat:0"),
                        _seed(seed, index, "seat:1"),
                    ),
                )
                segments.extend(rollout.players)
        _sync(device)
        collection_end = perf_counter()
        batch = prepare_ppo_batch(tuple(segments), trainer.config)
        _sync(device)
        batch_end = perf_counter()
        report = trainer.update(batch)
        _sync(device)
        optimization_end = perf_counter()
        save_ppo_checkpoint(trainer, output)
        _sync(device)
        end = perf_counter()
        if update >= warmup:
            records.append(
                {
                    "update": update,
                    "collection_seconds": collection_end - begin,
                    "inference_seconds": inference,
                    "batch_seconds": batch_end - collection_end,
                    "optimization_seconds": optimization_end - batch_end,
                    "checkpoint_seconds": end - optimization_end,
                    "total_seconds": end - begin,
                    "games": games,
                    "decisions": report.num_samples,
                    "optimizer_steps": report.optimizer_steps,
                    "games_per_second": games / (end - begin),
                    "decisions_per_second": report.num_samples / (end - begin),
                }
            )
    return records


def inference_probe() -> dict[str, Any]:
    """Fixed player-visible corpus only; never changes the training collector."""
    model = _initial_model(73).eval()
    segments: list[PlayerTrajectory] = []
    for index in range(16):
        result = collect_tic_tac_toe_rollout(
            TicTacToeEngine(),
            (model, model),
            max_steps=9,
            sampling_seeds=(index, index + 100),
        )
        segments.extend(result.players)
    observations = torch.cat([s.observations for s in segments])
    masks = torch.cat([s.legal_masks for s in segments])
    from boardgame_ai.models.masking import mask_action_logits

    with torch.no_grad():
        # Warm up both paths, then report repeated timings and numerical differences.
        model(observations)
        model(observations[0])
        timings = []
        for _ in range(5):
            start = perf_counter()
            separate = [model(row) for row in observations]
            middle = perf_counter()
            logits, values = model(observations)
            finish = perf_counter()
            timings.append(
                {"separate_seconds": middle - start, "batched_seconds": finish - middle}
            )
        single_logits = torch.stack([item[0] for item in separate])
        single_values = torch.stack([item[1] for item in separate])
        p = mask_action_logits(single_logits, masks).softmax(-1)
        q = mask_action_logits(logits, masks).softmax(-1)
        torch.testing.assert_close(logits, single_logits, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(values, single_values, rtol=2e-5, atol=2e-6)
    return {
        "positions": len(observations),
        "timings": timings,
        "max_logit_difference": float((logits - single_logits).abs().max()),
        "max_value_difference": float((values - single_values).abs().max()),
        "max_probability_difference": float((p - q).abs().max()),
        "adopted": False,
        "limitation": (
            "Fixed-corpus inference probe; no end-to-end batched "
            "collection or sampling-equivalence guarantee"
        ),
    }


def benchmark(
    reference: Path,
    output: Path,
    *,
    warmup: int = 2,
    measured: int = 5,
    repeats: int = 3,
    games: int = 64,
    threads: int = 1,
) -> dict[str, Any]:
    if any(
        type(v) is not int or v < 1 for v in (warmup, measured, repeats, games, threads)
    ):
        raise ValueError("Benchmark budgets must be positive integers")
    if output.exists():
        raise ValueError("Benchmark output must be new")
    spec = importlib.util.spec_from_file_location(
        "boardgame_ai_reference_ppo", reference
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load saved reference PPO source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    output.mkdir(parents=True)
    rows = []
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    try:
        for repeat in range(repeats):
            for device in devices[:: 1 if repeat % 2 == 0 else -1]:
                trainers: dict[str, PPOTrainer] = {}
                for version in (
                    ["reference", "optimized"]
                    if repeat % 2 == 0
                    else ["optimized", "reference"]
                ):
                    model = _initial_model(_seed(73, 0, "initialization")).to(device)
                    constructor = (
                        module.PPOTrainer if version == "reference" else PPOTrainer
                    )
                    trainer = cast(
                        PPOTrainer,
                        constructor(
                            model, PPOConfig(), shuffle_seed=_seed(73, 0, "shuffle")
                        ),
                    )
                    trainers[version] = trainer
                    records = _run(
                        trainer,
                        output / f"{repeat}-{device}-{version}.pt",
                        warmup=warmup,
                        measured=measured,
                        games=games,
                        seed=73,
                    )
                    rows.append(
                        {
                            "repeat": repeat,
                            "device": device,
                            "version": version,
                            "measurements": records,
                        }
                    )
                compare_trainers(trainers["reference"], trainers["optimized"])
        result = {
            "schema_version": 1,
            "torch": str(torch.__version__),
            "cpu_threads": threads,
            "cuda_name": torch.cuda.get_device_name(0) if "cuda" in devices else None,
            "reference_sha256": sha256(reference.read_bytes()).hexdigest(),
            "config": asdict(PPOConfig()),
            "warmup": warmup,
            "measured": measured,
            "repeats": repeats,
            "games_per_update": games,
            "seed": 73,
            "exact_equivalence_passed": True,
            "runs": rows,
            "inference_probe": inference_probe(),
        }
        (output / "benchmark.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ppo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.reference_ppo, args.output)
    print(
        json.dumps(
            {
                "exact_equivalence_passed": result["exact_equivalence_passed"],
                "runs": len(result["runs"]),
            }
        )
    )


if __name__ == "__main__":
    main()
