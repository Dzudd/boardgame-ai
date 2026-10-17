"""Read-only Phase 6 cost/strength/transfer/forgetting analysis.

Missing, partial and censored results are explicit. Training seeds, rather than
evaluation games, are the replication unit for cross-arm intervals.
"""

import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

from boardgame_ai.models.multigame import DEVELOPMENT_GAMES, TRANSFER_GAME
from boardgame_ai.training.connect_four_run import digest
from boardgame_ai.training.metrics_log import recover_metrics_log
from boardgame_ai.training.multigame_run import jobs, recover_environment, validate_run


def interval(values: list[float], multiplier: float) -> dict[str, Any]:
    average = mean(values)
    half = multiplier * stdev(values) / sqrt(len(values)) if len(values) > 1 else None
    return {
        "mean": average,
        "seeds": len(values),
        "approximate_95_seed_interval": [average - half, average + half]
        if half is not None
        else None,
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    seats = []
    for seat in (0, 1):
        records = [c for c in cases if c["seat"] == seat]
        seats.append(
            {
                "seat": seat,
                "games": len(records),
                "wins": sum(c["score"] == 1 for c in records),
                "draws": sum(c["score"] == 0.5 for c in records),
                "losses": sum(c["score"] == 0 for c in records),
                "score": mean(c["score"] for c in records) if records else None,
                "mean_raw_return": mean(c["raw_return"] for c in records)
                if records
                else None,
            }
        )
    return {
        "seats": seats,
        "balanced_score": mean(s["score"] for s in seats)
        if all(s["score"] is not None for s in seats)
        else None,
    }


def report_suite(root: Path, raw: bytes, settings: dict[str, Any]) -> dict[str, Any]:
    schedule = jobs(settings)
    report: dict[str, Any] = {
        "protocol_sha256": digest(raw),
        "jobs": {},
        "comparisons": {},
        "acceptance": "awaiting operator results",
        "predeclared_success_criteria": settings["success"],
    }
    valid: dict[str, Any] = {}
    for name, job in schedule.items():
        directory = root / name
        if not directory.exists():
            report["jobs"][name] = {"state": "missing"}
            continue
        try:
            trainer, payload, ledger = validate_run(
                directory, raw, settings, job, torch.device("cpu")
            )
            result = {
                "state": "training complete" if payload["complete"] else "incomplete",
                "training_seconds": ledger["training"],
                "evaluation_seconds": ledger["evaluation"],
                "per_game": payload["per_game"],
                "optimizer_steps": trainer.optimizer_steps,
                "curves": payload["curves"],
                "scores": {},
                "timing_uncertain": (directory / "finalization-recovery.json").exists(),
            }
            attempts_path = directory / "environment.jsonl"
            recover_environment(attempts_path)
            attempted = {game: 0 for game in job["games"]}
            if attempts_path.exists():
                for line in attempts_path.read_text(encoding="utf-8").splitlines():
                    try:
                        attempt = json.loads(line)
                    except json.JSONDecodeError:
                        # Validated torn tail: completed samples are a lower bound.
                        break
                    attempted[attempt["game"]] += attempt["ppo"]["num_samples"]
            result["attempted_complete_game_decisions"] = attempted
            result["discarded_or_unsaved_known_decisions"] = {
                g: attempted[g] - payload["per_game"][g]["decisions"] for g in attempted
            }
            result["partial_game_decisions"] = "unknown if collection was interrupted"
            result["metrics_complete"] = (
                recover_metrics_log(
                    directory / "updates.jsonl",
                    trainer.updates,
                    repair=False,
                )
                is not None
            )
            manifest = json.loads((directory / "training.json").read_bytes())
            result["parameters"] = manifest["parameters"]
            result["hardware"] = manifest["device"]
            path = directory / "evaluation.json"
            if path.exists():
                evaluation = json.loads(path.read_bytes())
                if (
                    evaluation["identity"] != payload["identity"]
                    or evaluation["curves"] != payload["curves"]
                    or evaluation["job"] != job
                ):
                    raise ValueError("Evaluation identity mismatch")
                result["evaluation_sha256"] = digest(path.read_bytes())
                result["state"] = (
                    "evaluated" if evaluation["complete"] else "partial evaluation"
                )
                result["scores"] = {
                    label: {g: summarize_cases(cases) for g, cases in games.items()}
                    for label, games in evaluation["cases"].items()
                }
                if evaluation["complete"]:
                    if evaluation["errors"] or evaluation["partial"]:
                        raise ValueError("Evaluation has errors despite complete flag")
                    expected_games = (
                        list((*DEVELOPMENT_GAMES, TRANSFER_GAME))
                        if job["arm"].startswith("adapt-")
                        else job["games"]
                    )
                    for label in (*payload["curves"], "control"):
                        for game in expected_games:
                            cases = evaluation["cases"][label][game]
                            if (
                                len(cases) != 2 * settings["evaluation_games_per_seat"]
                                or {(c["seat"], c["index"]) for c in cases}
                                != {
                                    (s, i)
                                    for s in (0, 1)
                                    for i in range(
                                        settings["evaluation_games_per_seat"]
                                    )
                                }
                                or any(c["score"] not in (0, 0.5, 1) for c in cases)
                            ):
                                raise ValueError("Incomplete/corrupt evaluation cases")
                    valid[name] = result
            report["jobs"][name] = result
        except (ValueError, OSError, KeyError, RuntimeError) as error:
            report["jobs"][name] = {"state": "rejected", "error": str(error)}
    success = settings["success"]
    for regime in settings["regimes"]:
        comparisons: dict[str, Any] = {}
        for game in DEVELOPMENT_GAMES:
            effects, forgetting = [], []
            costs = []
            for seed in settings["training_seeds"]:
                prefix = f"{regime}-{seed}"
                shared, separate, adapted = (
                    valid.get(f"{prefix}-{arm}")
                    for arm in ("shared", f"separate-{game}", "adapt-pretrained")
                )
                if shared and separate:
                    effects.append(
                        shared["scores"]["p100"][game]["balanced_score"]
                        - separate["scores"]["p100"][game]["balanced_score"]
                    )
                    costs.append(
                        {
                            "seed": seed,
                            "shared_total_seconds": shared["training_seconds"],
                            "separate_game_seconds": separate["training_seconds"],
                            "shared_game": shared["per_game"][game],
                            "separate_game": separate["per_game"][game],
                            "shared_attempted_decisions": shared[
                                "attempted_complete_game_decisions"
                            ][game],
                            "separate_attempted_decisions": separate[
                                "attempted_complete_game_decisions"
                            ][game],
                        }
                    )
                if adapted:
                    forgetting.append(
                        adapted["scores"]["p100"][game]["balanced_score"]
                        - adapted["scores"]["p0"][game]["balanced_score"]
                    )
            comparisons[game] = {"costs": costs}
            for key, effects_list, margin in (
                ("shared_minus_separate", effects, success["negative_transfer_margin"]),
                (
                    "after_minus_before_adaptation",
                    forgetting,
                    success["forgetting_margin"],
                ),
            ):
                effect = (
                    interval(effects_list, success["seed_interval_multiplier"])
                    if effects_list
                    else None
                )
                comparisons[game][key] = effect
                if effect:
                    bounds = effect["approximate_95_seed_interval"]
                    effect["harmful_point_estimate"] = effect["mean"] < -margin
                    effect["harm_supported_by_interval"] = bool(
                        bounds and bounds[1] < -margin
                    )
                    effect["noninferiority_supported_by_interval"] = bool(
                        len(effects_list) == len(settings["training_seeds"])
                        and bounds
                        and bounds[0] > -margin
                    )
        adaptation = []
        for seed in settings["training_seeds"]:
            arms: dict[str, Any] = {}
            for arm in ("adapt-pretrained", "adapt-fresh"):
                data = valid.get(f"{regime}-{seed}-{arm}")
                if not data:
                    continue
                curve = []
                for label in ("p0", "p25", "p50", "p75", "p100"):
                    if label not in data["scores"]:
                        continue
                    metadata = data["curves"][label]
                    curve.append(
                        {
                            "point": label,
                            "seconds": metadata["elapsed"],
                            "decisions": metadata["per_game"][TRANSFER_GAME][
                                "decisions"
                            ],
                            "score": data["scores"][label][TRANSFER_GAME][
                                "balanced_score"
                            ],
                        }
                    )
                crossing = next(
                    (
                        p
                        for p in curve
                        if p["score"] >= success["adaptation_score_target"]
                    ),
                    None,
                )
                arms[arm] = {
                    "curve": curve,
                    "first_observed_target": crossing,
                    "censored": crossing is None,
                }
            speedups = {}
            if len(arms) == 2:
                pretrained = arms["adapt-pretrained"]["first_observed_target"]
                fresh = arms["adapt-fresh"]["first_observed_target"]
                for axis in ("seconds", "decisions"):
                    ratio = (
                        fresh[axis] / pretrained[axis]
                        if fresh and pretrained and pretrained[axis] > 0
                        else None
                    )
                    speedups[axis] = {
                        "fresh_over_pretrained": ratio,
                        "meets_predeclared_target": ratio is not None
                        and ratio >= success["minimum_adaptation_speedup"],
                        "censoring_or_zero_origin": ratio is None,
                    }
            source = valid.get(f"{regime}-{seed}-shared")
            adaptation.append(
                {
                    "seed": seed,
                    "arms": arms,
                    "speedup": speedups,
                    "pretraining_seconds": source["training_seconds"]
                    if source
                    else None,
                    "pretraining_cost_excluded_from_adaptation_axis": True,
                }
            )
        comparisons["adaptation"] = adaptation
        # Compare total separate cost to shared, as well as per-game costs.
        comparisons["aggregate_costs"] = []
        for seed in settings["training_seeds"]:
            shared = valid.get(f"{regime}-{seed}-shared")
            separate_models = [
                valid.get(f"{regime}-{seed}-separate-{g}") for g in DEVELOPMENT_GAMES
            ]
            if shared and all(separate_models):
                separate_seconds = sum(
                    m["training_seconds"] for m in separate_models if m is not None
                )
                comparisons["aggregate_costs"].append(
                    {
                        "seed": seed,
                        "shared_seconds": shared["training_seconds"],
                        "separate_total_seconds": separate_seconds,
                        "wall_difference_seconds": shared["training_seconds"]
                        - separate_seconds,
                        "wall_ratio": shared["training_seconds"] / separate_seconds,
                        "within_five_percent_wall_match": not (
                            shared["timing_uncertain"]
                            or any(m["timing_uncertain"] for m in separate_models if m)
                        )
                        and abs(shared["training_seconds"] / separate_seconds - 1)
                        <= 0.05,
                    }
                )
        report["comparisons"][regime] = comparisons
        comparisons["strength"] = {}
        for arm in ("shared", *(f"separate-{g}" for g in DEVELOPMENT_GAMES)):
            games_for_arm = (
                DEVELOPMENT_GAMES
                if arm == "shared"
                else (arm.removeprefix("separate-"),)
            )
            for game in games_for_arm:
                scores = [
                    valid[f"{regime}-{seed}-{arm}"]["scores"]["p100"][game][
                        "balanced_score"
                    ]
                    for seed in settings["training_seeds"]
                    if f"{regime}-{seed}-{arm}" in valid
                ]
                effect = (
                    interval(scores, success["seed_interval_multiplier"])
                    if scores
                    else None
                )
                if effect:
                    bounds = effect["approximate_95_seed_interval"]
                    effect["strength_threshold_supported"] = bool(
                        len(scores) == len(settings["training_seeds"])
                        and bounds
                        and bounds[0] > success["minimum_balanced_score"]
                    )
                comparisons["strength"][f"{arm}/{game}"] = effect
    report["evaluated_jobs"] = len(valid)
    report["cost_uncertain_jobs"] = [
        name for name, value in report["jobs"].items() if value.get("timing_uncertain")
    ]
    report["expected_jobs"] = len(schedule)
    if len(valid) == len(schedule):
        report["acceptance"] = (
            "all scheduled evidence available; inspect effects, cost mismatch "
            "and censored transfer before claiming success"
        )
    return report
