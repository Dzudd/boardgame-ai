"""User-launched, bounded Tic-Tac-Toe self-play training and resume."""

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.training.checkpoint import load_ppo_checkpoint, save_ppo_checkpoint
from boardgame_ai.training.device import resolve_training_device
from boardgame_ai.training.monitoring import (
    MonitoringConfig,
    configure_monitoring,
    monitor_checkpoint,
)
from boardgame_ai.training.ppo import PPOConfig, PPOTrainer, prepare_ppo_batch
from boardgame_ai.training.rollout import PlayerTrajectory, collect_tic_tac_toe_rollout


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingConfig:
    """Total update budget may grow on resume; rollout schedule is immutable."""

    updates: int = 1000
    games_per_update: int = 64
    seed: int = 42
    device: str = "cuda"
    ppo: PPOConfig = PPOConfig()

    def __post_init__(self) -> None:
        if type(self.updates) is not int or self.updates < 0:
            raise ValueError("updates must be a nonnegative integer")
        if type(self.games_per_update) is not int or self.games_per_update <= 0:
            raise ValueError("games_per_update must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("Training seed must be an integer")


def _seed(seed: int, index: int, stream: str) -> int:
    text = f"boardgame-ai/training/v1:{seed}:{index}:{stream}"
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:8], "big")


def _initial_model(seed: int) -> TicTacToeActorCritic:
    # Seed only CPU initialization, preserving all global CPU/CUDA generators.
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.random.set_rng_state(generator.get_state())
        return TicTacToeActorCritic()


def run_self_play_training(
    output: str | Path,
    config: TrainingConfig,
    *,
    resume: bool = False,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    monitoring: MonitoringConfig | None = None,
) -> PPOTrainer:
    """Run requested updates, saving each completed update before log/callback.

    CPU engines and a frozen CPU copy of current policy collect complete games.
    Shared-policy self-play makes both seat segments on-policy. PPO forward,
    backward, Adam, and batch tensors run on the explicit CPU/CUDA device.
    Save portable latest.pt at every boundary; resume derives future rollout
    seeds from absolute update/game indices. Only checkpoints are authoritative:
    a crash after save can leave a missing metrics row. Existing history is not
    replayed. Optional isolated development monitoring never supplies training data.
    """
    device = resolve_training_device(config.device)
    directory = Path(output)
    manifest = {
        "schema_version": 1,
        "game_id": "tic_tac_toe",
        "mode": "self-play",
        "seed": config.seed,
        "games_per_update": config.games_per_update,
        "ppo": asdict(config.ppo),
    }
    checkpoint = directory / "latest.pt"
    if resume:
        try:
            saved_manifest = json.loads(
                (directory / "training.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ValueError("Cannot read training resume manifest") from error
        if saved_manifest != manifest:
            raise ValueError(
                "Resume seed, games_per_update, and PPO config must match "
                "the original run"
            )
        trainer = load_ppo_checkpoint(checkpoint, device=device)
        if trainer.config != config.ppo:
            raise ValueError("Checkpoint and training manifest configurations disagree")
        if trainer.update_count > config.updates:
            raise ValueError("Requested total updates precede the checkpoint")
        configure_monitoring(directory, monitoring, resume=True)
    else:
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError(
                "New training output must be absent or empty; "
                "use --resume for an existing run"
            )
        model = _initial_model(_seed(config.seed, 0, "initialization")).to(device)
        trainer = PPOTrainer(
            model, config.ppo, shuffle_seed=_seed(config.seed, 0, "shuffle")
        )
        directory.mkdir(parents=True, exist_ok=True)
        configure_monitoring(directory, monitoring, resume=False)
        (directory / "training.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        save_ppo_checkpoint(trainer, checkpoint)

    samples_seen: int | None = 0
    log_path = directory / "updates.jsonl"
    if trainer.update_count:
        rows = (
            [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            if log_path.exists()
            else []
        )
        samples_seen = (
            sum(row["ppo"]["num_samples"] for row in rows)
            if [row["update"] for row in rows]
            == list(range(1, trainer.update_count + 1))
            else None
        )
    if monitoring is not None and trainer.update_count % monitoring.interval == 0:
        monitor_checkpoint(
            directory,
            monitoring,
            games_seen=trainer.update_count * config.games_per_update,
            samples_seen=samples_seen,
        )
    game = TicTacToeEngine()
    for update_index in range(trainer.update_count, config.updates):
        # One frozen policy for all collection in this update. No CUDA inference
        # calls per board move; large training batches use the GPU afterward.
        policy = deepcopy(trainer.model).cpu().eval()
        segments: list[PlayerTrajectory] = []
        outcomes = [0, 0, 0]
        for local_game in range(config.games_per_update):
            game_index = update_index * config.games_per_update + local_game
            rollout = collect_tic_tac_toe_rollout(
                game,
                (policy, policy),
                max_steps=9,
                engine_seed=_seed(config.seed, game_index, "engine"),
                sampling_seeds=(
                    _seed(config.seed, game_index, "seat:0"),
                    _seed(config.seed, game_index, "seat:1"),
                ),
            )
            if not rollout.terminated:
                raise RuntimeError("Training requires complete Tic-Tac-Toe games")
            segments.extend(rollout.players)
            result = (
                0
                if rollout.total_rewards[0] > 0
                else 1
                if rollout.total_rewards[1] > 0
                else 2
            )
            outcomes[result] += 1
        batch = prepare_ppo_batch(tuple(segments), config.ppo)
        report = trainer.update(batch)
        save_ppo_checkpoint(trainer, checkpoint)
        record: dict[str, Any] = {
            "update": trainer.update_count,
            "device": str(device),
            "games": config.games_per_update,
            "x_wins": outcomes[0],
            "o_wins": outcomes[1],
            "draws": outcomes[2],
            "ppo": asdict(report),
        }
        with (directory / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if samples_seen is not None:
            samples_seen += report.num_samples
        if monitoring is not None and trainer.update_count % monitoring.interval == 0:
            monitor_checkpoint(
                directory,
                monitoring,
                games_seen=trainer.update_count * config.games_per_update,
                samples_seen=samples_seen,
            )
        if on_update is not None:
            on_update(record)
    return trainer
