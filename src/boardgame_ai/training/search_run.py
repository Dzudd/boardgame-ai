"""Explicit-budget CPU search-target training. No evaluation or experiment campaign."""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from boardgame_ai.search.puct import SearchConfig
from boardgame_ai.training.monitoring import _immutable_snapshot
from boardgame_ai.training.search_checkpoint import (
    load_search_checkpoint,
    save_search_checkpoint,
)
from boardgame_ai.training.search_learning import (
    LEARNER_ID,
    SearchTrainer,
    SearchTrainingConfig,
)


def _publish_report(directory: Path, report: dict[str, Any]) -> None:
    _immutable_snapshot(
        directory / f"update-{report['update']:08d}.json",
        (json.dumps(report, indent=2, allow_nan=False) + "\n").encode(),
    )


def run_search_training(
    output: str | Path,
    config: SearchTrainingConfig,
    *,
    updates: int,
    resume: bool = False,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> SearchTrainer:
    """Save before metrics/callback; retry the current saved report on resume.

    Only CPU and update-boundary resume are supported. Total updates may grow;
    all collection, replay, optimizer and seed settings remain immutable.
    """
    if type(updates) is not int or updates < 0:
        raise ValueError("Total updates must be a nonnegative integer")
    directory = Path(output)
    manifest = {"schema_version": 1, "learner_id": LEARNER_ID, "config": asdict(config)}
    checkpoint = directory / "latest.pt"
    if resume:
        if (
            json.loads((directory / "training.json").read_text(encoding="utf-8"))
            != manifest
        ):
            raise ValueError("Search training resume configuration must match")
        trainer, report = load_search_checkpoint(checkpoint)
        if trainer.config != config:
            raise ValueError("Search checkpoint and manifest configurations disagree")
        if trainer.update_count > updates:
            raise ValueError("Requested updates precede the search checkpoint")
        if report is not None:
            _publish_report(directory, report)
    else:
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError("New search output must be absent or empty")
        trainer = SearchTrainer(config)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "training.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        save_search_checkpoint(trainer, checkpoint)
    for _ in range(trainer.update_count, updates):
        report = trainer.update()
        save_search_checkpoint(trainer, checkpoint, last_report=report)
        _publish_report(directory, report)
        if on_update is not None:
            on_update(report)
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--updates", type=int, required=True, help="Explicit total update budget"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--games-per-update", type=int, default=64)
    parser.add_argument("--optimizer-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--simulations", type=int, default=32)
    args = parser.parse_args()
    config = SearchTrainingConfig(
        seed=args.seed,
        games_per_update=args.games_per_update,
        optimizer_steps_per_update=args.optimizer_steps,
        batch_size=args.batch_size,
        replay_capacity_games=args.replay_capacity,
        search=SearchConfig(simulations=args.simulations, exploration=True),
    )
    run_search_training(
        args.output,
        config,
        updates=args.updates,
        resume=args.resume,
        on_update=lambda report: print(json.dumps(report), flush=True),
    )


if __name__ == "__main__":
    main()
