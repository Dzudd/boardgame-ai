"""Predeclared six-run entropy comparison; latest-policy self-play only."""

import argparse
import json
from dataclasses import asdict, replace
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Any

import torch

from boardgame_ai.policy_evaluation import evaluate_policy_checkpoint
from boardgame_ai.training.checkpoint import load_ppo_checkpoint
from boardgame_ai.training.monitoring import (
    MonitoringConfig,
    atomic_json,
    policy_diagnostics,
)
from boardgame_ai.training.ppo import PPOConfig
from boardgame_ai.training.run import TrainingConfig, run_self_play_training


def paired_interval(differences: list[float]) -> dict[str, Any]:
    """Predeclared Student-t(2), three paired training seeds only."""
    if len(differences) != 3:
        raise ValueError("Require three paired training-seed differences")
    mean = sum(differences) / 3
    se = sqrt(sum((d - mean) ** 2 for d in differences) / 6)
    return {
        "differences": differences,
        "mean": mean,
        "se": se,
        "approximate_95_interval": [mean - 4.303 * se, mean + 4.303 * se],
        "method": "paired training seeds, Student-t df=2, multiplier 4.303",
    }


def run_experiments(output: Path, *, resume: bool = False) -> dict[str, Any]:
    protocol_path = Path("PHASE45_PROTOCOL.md")
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "protocol_sha256": sha256(protocol_path.read_bytes()).hexdigest(),
        "seeds": [42, 43, 44],
        "updates": 200,
        "games_per_update": 64,
        "variants": {"baseline": 0.01, "entropy": 0.05},
        "monitoring": asdict(MonitoringConfig()),
        "final_seed": 20261029,
        "final_games_per_seat": 512,
        "device": "cpu",
        "threads": 1,
    }
    if resume:
        if (
            json.loads((output / "experiment.json").read_text(encoding="utf-8"))
            != protocol
        ):
            raise ValueError("Experiment resume protocol mismatch")
    else:
        if output.exists():
            raise ValueError("Experiment output must be absent")
        output.mkdir(parents=True)
        atomic_json(output / "experiment.json", protocol)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for index, seed in enumerate(protocol["seeds"]):
            variants = list(protocol["variants"].items())
            for variant, entropy in variants[:: 1 if index % 2 == 0 else -1]:
                directory = output / f"{variant}-{seed}"
                config = TrainingConfig(
                    updates=200,
                    games_per_update=64,
                    seed=seed,
                    device="cpu",
                    ppo=replace(PPOConfig(), entropy_coefficient=entropy),
                )

                def progress(
                    record: dict[str, Any], seed: int = seed, variant: str = variant
                ) -> None:
                    if record["update"] % 25 == 0:
                        print(
                            f"{variant} seed={seed}: {record['update']}/200", flush=True
                        )

                run_self_play_training(
                    directory,
                    config,
                    resume=directory.exists(),
                    monitoring=MonitoringConfig(),
                    on_update=progress,
                )
        # Freeze every development-only selection before opening any final results.
        selected = []
        for seed in protocol["seeds"]:
            for variant in protocol["variants"]:
                directory = output / f"{variant}-{seed}"
                best = json.loads((directory / "best.json").read_text(encoding="utf-8"))
                selected.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "directory": directory.name,
                        **best,
                    }
                )
        selection_path = output / "selections.json"
        if selection_path.exists():
            if json.loads(selection_path.read_text(encoding="utf-8")) != selected:
                raise ValueError("Frozen development selections changed")
        else:
            atomic_json(selection_path, selected)
        results = []
        for selection in selected:
            directory = output / selection["directory"]
            checkpoint = directory / selection["checkpoint"]
            if (
                sha256(checkpoint.read_bytes()).hexdigest()
                != selection["checkpoint_sha256"]
            ):
                raise ValueError("Selected checkpoint hash mismatch")
            destination = (
                output / f"final-{selection['variant']}-{selection['seed']}.json"
            )
            if destination.exists():
                report = json.loads(destination.read_text(encoding="utf-8"))
                if report["checkpoint_sha256"] != selection["checkpoint_sha256"]:
                    raise ValueError("Final report checkpoint mismatch")
            else:
                evaluation = evaluate_policy_checkpoint(
                    checkpoint, games_per_seat=512, seed=20261029
                )
                cases: list[dict[str, Any]] = []
                for case in evaluation.cases:
                    scores = [
                        r.outcome.scores[case.learner_seat]
                        for r in case.report.records
                        if r.outcome is not None and r.error is None
                    ]
                    if len(scores) != 512:
                        raise ValueError("Incomplete final evaluation")
                    mean = sum(scores) / 512
                    se = sqrt(sum((v - mean) ** 2 for v in scores) / (512 * 511))
                    cases.append(
                        {
                            "seat": case.learner_seat,
                            "opponent": case.opponent,
                            "mean_score": mean,
                            "approximate_95_interval": [
                                mean - 1.96 * se,
                                mean + 1.96 * se,
                            ],
                            "summary": asdict(case.report.summary),
                        }
                    )
                logs = [
                    json.loads(line)
                    for line in (directory / "updates.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                report = {
                    **selection,
                    "checkpoint_sha256": evaluation.checkpoint_sha256,
                    "final_seed": 20261029,
                    "cases": cases,
                    "evaluation": asdict(evaluation),
                    "balanced_score": sum(c["mean_score"] for c in cases) / 4,
                    "total_training_games": 12800,
                    "total_training_decisions": sum(
                        row["ppo"]["num_samples"] for row in logs
                    ),
                    "total_optimizer_steps": sum(
                        row["ppo"]["optimizer_steps"] for row in logs
                    ),
                    "diagnostics": policy_diagnostics(
                        load_ppo_checkpoint(checkpoint), games=64, seed=20261029
                    ),
                }
                atomic_json(destination, report)
            results.append(report)
            print(
                f"Final {selection['variant']} seed={selection['seed']}: "
                f"{report['balanced_score']:.4f}",
                flush=True,
            )
        by_key = {(r["seed"], r["variant"]): r for r in results}
        paired = paired_interval(
            [
                by_key[(seed, "entropy")]["balanced_score"]
                - by_key[(seed, "baseline")]["balanced_score"]
                for seed in protocol["seeds"]
            ]
        )
        cases_paired = []
        for index in range(4):
            cases_paired.append(
                {
                    "seat": results[0]["cases"][index]["seat"],
                    "opponent": results[0]["cases"][index]["opponent"],
                    **paired_interval(
                        [
                            by_key[(seed, "entropy")]["cases"][index]["mean_score"]
                            - by_key[(seed, "baseline")]["cases"][index]["mean_score"]
                            for seed in protocol["seeds"]
                        ]
                    ),
                }
            )
        result = {
            "protocol": protocol,
            "runs": results,
            "paired_balanced_effect": paired,
            "paired_case_effects": cases_paired,
            "improvement_supported": paired["approximate_95_interval"][0] > 0,
        }
        atomic_json(output / "results.json", result)
        return result
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_experiments(args.output, resume=args.resume)


if __name__ == "__main__":
    main()
