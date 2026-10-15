"""Optional isolated development evaluation and bounded immutable snapshots."""

import json
import os
import random
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.training.checkpoint import load_ppo_checkpoint
from boardgame_ai.training.ppo import PPOTrainer
from boardgame_ai.training.rollout import collect_tic_tac_toe_rollout


@dataclass(frozen=True, slots=True, kw_only=True)
class MonitoringConfig:
    interval: int = 50
    retain: int = 3
    games_per_seat: int = 128
    self_play_games: int = 64
    seed: int = 20261022

    def __post_init__(self) -> None:
        for value in (
            self.interval,
            self.retain,
            self.games_per_seat,
            self.self_play_games,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("Monitoring counts must be positive integers")
        if type(self.seed) is not int:
            raise ValueError("Monitoring seed must be an integer")


@contextmanager
def isolated_rng() -> Iterator[None]:
    """Restore even on failure; evaluation never owns training RNG state."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None  # type: ignore[no-untyped-call]
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def atomic_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=".monitor-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _immutable_snapshot(path: Path, data: bytes) -> None:
    """Publish complete bytes exclusively; interrupted writes cannot poison resume."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Immutable checkpoint mismatch")
        return
    fd, name = tempfile.mkstemp(prefix=".snapshot-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(
                name, path
            )  # Exclusive publication; never replace an existing file.
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("Immutable checkpoint mismatch") from None
    finally:
        Path(name).unlink(missing_ok=True)


def policy_diagnostics(trainer: PPOTrainer, *, games: int, seed: int) -> dict[str, Any]:
    """Measurement only: no returned trajectory is offered to an optimizer."""
    from boardgame_ai.policy_audit import (
        audit_decision,
        reachable_decisions,
        summarize,
        trace_rollout,
    )

    census = [
        audit_decision(trainer.model, view, legal)
        for view, legal in reachable_decisions()
    ]
    visits: list[dict[str, Any]] = []
    outcomes = [0, 0, 0]
    for index in range(games):

        def derived(stream: str, index: int = index) -> int:
            return int.from_bytes(
                sha256(
                    f"boardgame-ai/monitor/v1:{seed}:{index}:{stream}".encode()
                ).digest()[:8],
                "big",
            )

        rollout = collect_tic_tac_toe_rollout(
            TicTacToeEngine(),
            (trainer.model, trainer.model),
            max_steps=9,
            engine_seed=derived("engine"),
            sampling_seeds=(derived("seat:0"), derived("seat:1")),
        )
        _, rows = trace_rollout(rollout, trainer)
        visits.extend(rows)
        outcomes[
            0
            if rollout.total_rewards[0] > 0
            else 1
            if rollout.total_rewards[1] > 0
            else 2
        ] += 1
    seats = []
    for seat in (0, 1):
        rows = [row for row in visits if row["seat"] == seat]
        entropy = [
            -sum(p * np.log(p) for p in row["probabilities"] if p > 0) for row in rows
        ]
        positions = {tuple(row["board"]) for row in rows}
        position_actions = {(tuple(row["board"]), row["action"]) for row in rows}
        seats.append(
            {
                "seat": seat,
                "decisions": len(rows),
                "distinct_positions": len(positions),
                "distinct_position_actions": len(position_actions),
                "mean_actions_per_position": len(position_actions) / len(positions),
                "mean_entropy": float(sum(entropy) / len(entropy)),
            }
        )
    return {
        "census_tactics": summarize(census),
        "visited_tactics": summarize(visits),
        "self_play_outcomes": outcomes,
        "seats": seats,
        "self_play_games": games,
        "seed": seed,
        "seed_schedule": "boardgame-ai/monitor/v1",
    }


def development_evaluation(
    checkpoint: Path, config: MonitoringConfig
) -> dict[str, Any]:
    from boardgame_ai.policy_evaluation import evaluate_policy_checkpoint

    with isolated_rng():
        evaluation = evaluate_policy_checkpoint(
            checkpoint, games_per_seat=config.games_per_seat, seed=config.seed
        )
        cases: list[dict[str, Any]] = []
        for case in evaluation.cases:
            records = case.report.records
            if any(record.error or record.outcome is None for record in records):
                raise ValueError(
                    "Development evaluation requires complete error-free games"
                )
            scores = [
                record.outcome.scores[case.learner_seat]
                for record in records
                if record.outcome is not None
            ]
            mean = sum(scores) / len(scores)
            se = (
                sqrt(
                    sum((score - mean) ** 2 for score in scores)
                    / (len(scores) * (len(scores) - 1))
                )
                if len(scores) > 1
                else None
            )
            cases.append(
                {
                    "opponent": case.opponent,
                    "seat": case.learner_seat,
                    "mean_score": mean,
                    "se": se,
                    "approximate_95_interval": [mean - 1.96 * se, mean + 1.96 * se]
                    if se is not None
                    else None,
                    "summary": asdict(case.report.summary),
                }
            )
        trainer = load_ppo_checkpoint(checkpoint)
        return {
            "metric": "equal-case mean score; earliest tie wins",
            "score": sum(c["mean_score"] for c in cases) / 4,
            "cases": cases,
            "evaluation": asdict(evaluation),
            "diagnostics": policy_diagnostics(
                trainer, games=config.self_play_games, seed=config.seed
            ),
        }


def configure_monitoring(
    directory: Path, config: MonitoringConfig | None, *, resume: bool
) -> None:
    path = directory / "monitoring.json"
    manifest = {"schema_version": 1, "config": asdict(config)} if config else None
    if resume:
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if saved != manifest:
            raise ValueError("Resume monitoring configuration must match")
    elif config is not None:
        atomic_json(path, manifest)


def monitor_checkpoint(
    directory: Path,
    config: MonitoringConfig,
    *,
    games_seen: int,
    samples_seen: int | None,
) -> dict[str, Any] | None:
    """Retry missing current-boundary evaluation on resume; never rewrite snapshots.

    Only owned snapshot files in snapshots/ are removed by declared retention.
    Best is pinned in addition to the newest retain snapshots. Report files remain.
    """
    source = directory / "latest.pt"
    snapshot_bytes = source.read_bytes()
    with isolated_rng():
        trainer = load_ppo_checkpoint(source)
    update = trainer.update_count
    if not update or update % config.interval:
        return None
    digest = sha256(snapshot_bytes).hexdigest()
    folder = directory / "snapshots"
    folder.mkdir(exist_ok=True)
    name = f"update-{update:08d}-{digest}.pt"
    snapshot = folder / name
    _immutable_snapshot(snapshot, snapshot_bytes)
    report_path = directory / f"development-{update:08d}.json"
    if report_path.exists():
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        if report["checkpoint_sha256"] != digest:
            raise ValueError("Development report checkpoint mismatch")
    else:
        report = {
            "schema_version": 1,
            "checkpoint": f"snapshots/{name}",
            "checkpoint_sha256": digest,
            "update": update,
            "optimizer_steps": trainer.optimizer_steps,
            "games_seen": games_seen,
            "samples_seen": samples_seen,
            "config": asdict(config),
            **development_evaluation(snapshot, config),
        }
        atomic_json(report_path, report)
    best_path = directory / "best.json"
    best = (
        json.loads(best_path.read_text(encoding="utf-8"))
        if best_path.exists()
        else None
    )
    if best is None or report["score"] > best["score"]:
        best = {
            key: report[key]
            for key in ("checkpoint", "checkpoint_sha256", "update", "score", "metric")
        }
        atomic_json(best_path, best)
    keep = {path.name for path in sorted(folder.glob("update-*.pt"))[-config.retain :]}
    keep.add(Path(best["checkpoint"]).name)
    for path in folder.glob("update-*.pt"):
        if path.name not in keep:
            path.unlink()
    return report
