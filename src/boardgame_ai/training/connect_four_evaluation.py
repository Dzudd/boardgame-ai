"""Isolated fixed-snapshot Connect Four evaluation; never returns PPO samples."""

import io
import json
import math
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
from time import monotonic
from typing import Any

import torch

from boardgame_ai.agents.random_agent import RandomAgent
from boardgame_ai.games.connect_four import ConnectFourEngine
from boardgame_ai.games.connect_four.observation import ConnectFourObservation
from boardgame_ai.models.connect_four import ConnectFourActorCritic
from boardgame_ai.models.encoders import (
    decode_connect_four_action,
    encode_connect_four_legal_actions,
    encode_connect_four_observation,
)
from boardgame_ai.models.masking import mask_action_logits
from boardgame_ai.training.connect_four_checkpoint import (
    trainer_from_connect_four_payload,
)
from boardgame_ai.training.connect_four_run import digest, read_settings, validate_run
from boardgame_ai.training.monitoring import (
    _immutable_snapshot,
    atomic_json,
    isolated_rng,
)


class EvaluationTruncationError(RuntimeError):
    """A live game at the maximum decision count is a failed evaluation."""


def evaluation_seed(root: int, index: int, stream: str) -> int:
    text = f"boardgame-ai/connect-four/evaluation/v1:{root}:{index}:{stream}"
    return int.from_bytes(sha256(text.encode()).digest()[:8], "big")


def _game(
    model: ConnectFourActorCritic | None,
    seat: int,
    seeds: dict[str, int],
    deadline: float,
) -> dict[str, Any]:
    game = ConnectFourEngine()
    game.reset(seeds["engine"])
    opponent: RandomAgent[ConnectFourObservation] = RandomAgent()
    opponent.reset(seeds["opponent"])
    control: RandomAgent[ConnectFourObservation] = RandomAgent()
    control.reset(seeds["policy"])
    generator = torch.Generator(device="cpu").manual_seed(seeds["policy"])
    for step in range(42):
        if monotonic() >= deadline:
            raise TimeoutError("Evaluation deadline reached")
        actor = game.current_player()
        if actor is None:
            raise RuntimeError("Unexpected terminal game")
        view, legal = game.decision_for(actor)
        if actor != seat:
            action = opponent.select_action(view, legal)
        elif model is None:
            action = control.select_action(view, legal)
        else:
            with torch.no_grad():
                logits, _ = model(encode_connect_four_observation(view))
                probabilities = mask_action_logits(
                    logits, encode_connect_four_legal_actions(legal)
                ).softmax(-1)
                index = int(
                    torch.multinomial(probabilities, 1, generator=generator).item()
                )
            action = decode_connect_four_action(index, legal)
        if monotonic() >= deadline:
            raise TimeoutError("Evaluation deadline reached")
        result = game.step(action)
        if result.terminated:
            reward = result.rewards[seat]
            return {
                "outcome": "win" if reward > 0 else "loss" if reward < 0 else "draw",
                "score": (reward + 1) / 2,
                "length": step + 1,
                "errors": 0,
                "truncations": 0,
            }
    raise EvaluationTruncationError("Evaluation truncated")


def _effect(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    seats = []
    for seat in (0, 1):
        left = [r for r in a if r["seat"] == seat]
        right = [r for r in b if r["seat"] == seat]
        if len(left) != len(right) or not left:
            return {"available": False}
        differences = [
            x["score"] - y["score"] for x, y in zip(left, right, strict=True)
        ]
        se = (
            stdev(differences) / math.sqrt(len(differences))
            if len(differences) > 1
            else None
        )
        center = mean(differences)
        seats.append(
            {
                "seat": seat,
                "mean": center,
                "standard_error": se,
                "interval": [center - 1.96 * se, center + 1.96 * se]
                if se is not None
                else None,
            }
        )
    center = mean(s["mean"] for s in seats)
    se = (
        0.5 * math.sqrt(sum(s["standard_error"] ** 2 for s in seats))
        if all(s["standard_error"] is not None for s in seats)
        else None
    )
    return {
        "available": True,
        "seats": seats,
        "balanced_mean": center,
        "balanced_interval": [center - 1.96 * se, center + 1.96 * se]
        if se is not None
        else None,
        "method": (
            "approximate paired normal interval; game-level only, "
            "no training-seed uncertainty"
        ),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seats = []
    for seat in (0, 1):
        cases = [r for r in rows if r["seat"] == seat]
        seats.append(
            {
                "seat": seat,
                "games": len(cases),
                **{
                    name: sum(r["outcome"] == name for r in cases)
                    for name in ("win", "draw", "loss")
                },
                "score": mean(r["score"] for r in cases) if cases else None,
                "mean_length": mean(r["length"] for r in cases) if cases else None,
                "errors": 0,
                "truncations": 0,
            }
        )
    return {
        "seats": seats,
        "balanced_score": mean(s["score"] for s in seats)
        if all(s["score"] is not None for s in seats)
        else None,
    }


def evaluate_connect_four_run(
    output: str | Path, *, settings_path: str | Path
) -> dict[str, Any]:
    """Exclusive report, cumulative separate cap, no training state mutations.

    An interrupted reservation is conservatively spent. A partial report records
    complete cases and the stop reason; it cannot be overwritten or rerun.
    """
    start = monotonic()
    directory = Path(output)
    raw, settings = read_settings(settings_path)
    trainer, _, ledger = validate_run(directory, raw, settings)
    if trainer.update_count != settings["training"]["updates"]:
        raise ValueError("Evaluation requires the declared final boundary")
    if not (directory / "final.pt").is_file():
        raise ValueError("Final snapshot missing; resume the saved boundary first")
    destination = directory / "evaluation.json"
    if destination.exists():
        raise ValueError("Evaluation output already exists")
    e = settings["evaluation"]
    used = ledger["evaluation"]
    if used >= e["wall_time_cap_seconds"]:
        raise ValueError("Cumulative evaluation cap exhausted")
    deadline = start + e["wall_time_cap_seconds"] - used
    ledger["evaluation"] = e["wall_time_cap_seconds"]
    atomic_json(directory / "budget.json", ledger)
    report: dict[str, Any] = {
        "schema_version": 1,
        "settings_sha256": digest(raw),
        "cases": {"initial": [], "final": [], "control": []},
        "checkpoint_sha256": {},
        "complete": False,
        "stop_reason": None,
        "failure": None,
    }
    old_threads = torch.get_num_threads()
    with isolated_rng():
        torch.set_num_threads(1)
        stage, seat, i = "initial", 0, 0
        try:
            for stage in ("initial", "final", "control"):
                if monotonic() >= deadline:
                    raise TimeoutError("Evaluation deadline before checkpoint load")
                model = None
                if stage != "control":
                    snapshot = (directory / f"{stage}.pt").read_bytes()
                    report["checkpoint_sha256"][stage] = digest(snapshot)
                    frozen = trainer_from_connect_four_payload(
                        torch.load(
                            io.BytesIO(snapshot), map_location="cpu", weights_only=True
                        )
                    )
                    model = frozen.model.eval()
                for seat in (0, 1):
                    for i in range(e["games_per_seat"]):
                        index = seat * e["games_per_seat"] + i
                        seeds = {
                            stream: evaluation_seed(
                                e["control_seed"]
                                if stage == "control" and stream == "policy"
                                else e["seed"],
                                index,
                                stream,
                            )
                            for stream in ("engine", "policy", "opponent")
                        }
                        result = _game(model, seat, seeds, deadline)
                        report["cases"][stage].append(
                            {"seat": seat, "index": i, "seeds": seeds, **result}
                        )
            report["complete"] = True
        except Exception as error:
            report["stop_reason"] = type(error).__name__
            report["failure"] = {
                "stage": stage,
                "seat": seat,
                "index": i,
                "category": type(error).__name__,
                "errors": int(
                    not isinstance(error, (TimeoutError, EvaluationTruncationError))
                ),
                "truncations": int(isinstance(error, EvaluationTruncationError)),
            }
        finally:
            torch.set_num_threads(old_threads)
            ledger["evaluation"] = used + monotonic() - start
            report["elapsed_seconds"] = ledger["evaluation"]
            report["cap_overrun_seconds"] = max(0.0, monotonic() - deadline)
            report["summaries"] = {k: _summary(v) for k, v in report["cases"].items()}
            if report["failure"] is not None:
                failure = report["failure"]
                summary = report["summaries"][failure["stage"]]["seats"][
                    failure["seat"]
                ]
                summary["errors"] += failure["errors"]
                summary["truncations"] += failure["truncations"]
            report["effects"] = (
                {
                    "final_minus_initial": _effect(
                        report["cases"]["final"], report["cases"]["initial"]
                    ),
                    "final_minus_control": _effect(
                        report["cases"]["final"], report["cases"]["control"]
                    ),
                }
                if report["complete"]
                else {}
            )
            _immutable_snapshot(
                destination,
                (json.dumps(report, indent=2, allow_nan=False) + "\n").encode(),
            )
            ledger["evaluation"] = used + monotonic() - start
            atomic_json(directory / "budget.json", ledger)
    return report
