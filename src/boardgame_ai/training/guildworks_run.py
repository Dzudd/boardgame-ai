"""Bounded recurrent self-play training, portable resume and isolated evaluation.

Usage: python -m boardgame_ai.training.guildworks_run NEW_DIRECTORY --device cpu
"""

import argparse
import json
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from boardgame_ai.agents import Agent, RandomAgent
from boardgame_ai.games.guildworks import GuildworksEngine, GuildworksObservation
from boardgame_ai.games.guildworks.evaluation import guildworks_outcome
from boardgame_ai.models.guildworks import (
    ENCODING_ID,
    GuildworksRecurrentPolicy,
    RecurrentGuildworksAgent,
)
from boardgame_ai.simulation import evaluate_batch
from boardgame_ai.training.device import resolve_training_device
from boardgame_ai.training.guildworks import (
    DEFAULT_CONFIG,
    RecurrentPPOTrainer,
    SeatEpisode,
    collect_guildworks,
)
from boardgame_ai.training.ppo import PPOConfig

MODEL_ID = "guildworks-gru32-actions8-v1"
_KEYS = {
    "schema_version",
    "game_id",
    "model_id",
    "encoding_id",
    "config",
    "model_state",
    "optimizer_state",
    "shuffle_rng_state",
    "update_count",
    "optimizer_steps",
    "model_training",
}


def derive_seed(seed: int, index: int, stream: str) -> int:
    text = f"boardgame-ai/guildworks/v1:{seed}:{index}:{stream}"
    return int.from_bytes(sha256(text.encode()).digest()[:8], "big")


def initial_model(seed: int) -> GuildworksRecurrentPolicy:
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator().manual_seed(seed)
        torch.random.set_rng_state(generator.get_state())
        return GuildworksRecurrentPolicy()


def _validate_checkpoint(payload: Any, trainer: RecurrentPPOTrainer) -> None:
    if not isinstance(payload, dict) or set(payload) != _KEYS:
        raise ValueError("Invalid recurrent checkpoint fields")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["game_id"] != "guildworks"
        or payload["model_id"] != MODEL_ID
        or payload["encoding_id"] != ENCODING_ID
        or payload["config"] != asdict(trainer.config)
        or type(payload["model_training"]) is not bool
    ):
        raise ValueError("Incompatible recurrent checkpoint")
    for key in ("update_count", "optimizer_steps"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError("Invalid checkpoint counters")
    if payload["update_count"] > payload["optimizer_steps"]:
        raise ValueError("Inconsistent checkpoint counters")
    state = payload["model_state"]
    references = trainer.model.state_dict()
    if not isinstance(state, dict) or set(state) != set(references):
        raise ValueError("Invalid recurrent model fields")
    for key, reference in references.items():
        tensor = state[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != reference.shape
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or not torch.isfinite(tensor).all().item()
        ):
            raise ValueError("Invalid recurrent model tensor")
    optimizer = payload["optimizer_state"]
    default = torch.optim.Adam(
        trainer.model.parameters(), lr=trainer.config.learning_rate
    ).state_dict()
    if (
        not isinstance(optimizer, dict)
        or set(optimizer) != {"state", "param_groups"}
        or optimizer["param_groups"] != default["param_groups"]
        or not isinstance(optimizer["state"], dict)
    ):
        raise ValueError("Invalid recurrent optimizer configuration")
    parameters = list(trainer.model.parameters())
    expected = set(range(len(parameters))) if payload["optimizer_steps"] else set()
    if set(optimizer["state"]) != expected:
        raise ValueError("Incomplete Adam state")
    for index, moments in optimizer["state"].items():
        if (
            type(index) is not int
            or not isinstance(moments, dict)
            or set(moments) != {"step", "exp_avg", "exp_avg_sq"}
        ):
            raise ValueError("Invalid Adam fields")
        for key, tensor in moments.items():
            shape = () if key == "step" else parameters[index].shape
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != shape
                or tensor.dtype != torch.float32
                or tensor.device.type != "cpu"
                or not torch.isfinite(tensor).all().item()
            ):
                raise ValueError("Invalid Adam tensor")
        step = float(moments["step"])
        if (
            step != int(step)
            or not 1 <= step <= payload["optimizer_steps"]
            or (moments["exp_avg_sq"] < 0).any().item()
        ):
            raise ValueError("Invalid Adam moments")
    rng = payload["shuffle_rng_state"]
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
        raise ValueError("Invalid shuffle RNG")
    try:
        torch.Generator().set_state(rng)
    except RuntimeError as error:
        raise ValueError("Invalid shuffle RNG") from error


def save_checkpoint(trainer: RecurrentPPOTrainer, path: str | Path) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "game_id": "guildworks",
        "model_id": MODEL_ID,
        "encoding_id": ENCODING_ID,
        "config": asdict(trainer.config),
        "model_state": {
            k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()
        },
        "optimizer_state": deepcopy(trainer.optimizer.state_dict()),
        "shuffle_rng_state": trainer.generator.get_state(),
        "update_count": trainer.update_count,
        "optimizer_steps": trainer.optimizer_steps,
        "model_training": trainer.model.training,
    }
    for moments in payload["optimizer_state"]["state"].values():
        for key, tensor in moments.items():
            moments[key] = tensor.detach().cpu().clone()
    _validate_checkpoint(payload, trainer)
    destination = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".guildworks-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> RecurrentPPOTrainer:
    target = resolve_training_device(device)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != _KEYS
        or not isinstance(payload["config"], dict)
        or set(payload["config"]) != set(asdict(DEFAULT_CONFIG))
    ):
        raise ValueError("Invalid recurrent checkpoint/configuration")
    try:
        config = PPOConfig(**payload["config"])
    except TypeError as error:
        raise ValueError("Invalid checkpoint configuration") from error
    trainer = RecurrentPPOTrainer(initial_model(0), config, shuffle_seed=0)
    _validate_checkpoint(payload, trainer)
    trainer.model.load_state_dict(payload["model_state"], strict=True)
    if target.type != "cpu":
        trainer = RecurrentPPOTrainer(trainer.model.to(target), config, shuffle_seed=0)
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.generator.set_state(payload["shuffle_rng_state"])
    trainer.update_count = payload["update_count"]
    trainer.optimizer_steps = payload["optimizer_steps"]
    trainer.model.train(payload["model_training"])
    return trainer


@dataclass(frozen=True, slots=True, kw_only=True)
class RecurrentRunConfig:
    updates: int = 16
    games_per_update: int = 16
    seed: int = 20261016
    device: str = "cpu"
    ppo: PPOConfig = DEFAULT_CONFIG

    def __post_init__(self) -> None:
        if (
            type(self.updates) is not int
            or self.updates < 0
            or type(self.games_per_update) is not int
            or self.games_per_update <= 0
            or type(self.seed) is not int
        ):
            raise ValueError("Invalid recurrent training budget or seed")


def run_training(
    output: str | Path, config: RecurrentRunConfig, *, resume: bool = False
) -> RecurrentPPOTrainer:
    device = resolve_training_device(config.device)
    directory = Path(output)
    manifest = {
        "schema_version": 1,
        "game_id": "guildworks",
        "model_id": MODEL_ID,
        "encoding_id": ENCODING_ID,
        "seed": config.seed,
        "games_per_update": config.games_per_update,
        "ppo": asdict(config.ppo),
    }
    if resume:
        if (
            json.loads((directory / "training.json").read_text(encoding="utf-8"))
            != manifest
        ):
            raise ValueError("Resume requires the same seed, schedule and PPO config")
        trainer = load_checkpoint(directory / "latest.pt", device=str(device))
        if trainer.config != config.ppo or trainer.update_count > config.updates:
            raise ValueError("Checkpoint disagrees with requested training")
    else:
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError("New training output must be absent or empty")
        trainer = RecurrentPPOTrainer(
            initial_model(derive_seed(config.seed, 0, "initialization")).to(device),
            config.ppo,
            shuffle_seed=derive_seed(config.seed, 0, "shuffle"),
        )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "training.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        save_checkpoint(trainer, directory / "latest.pt")
    for update in range(trainer.update_count, config.updates):
        policy = deepcopy(trainer.model).cpu().eval()
        episodes: list[SeatEpisode] = []
        scores = [0.0, 0.0]
        for local in range(config.games_per_update):
            index = update * config.games_per_update + local
            pair = collect_guildworks(
                GuildworksEngine(),
                policy,
                engine_seed=derive_seed(config.seed, index, "engine"),
                sampling_seeds=(
                    derive_seed(config.seed, index, "seat:0"),
                    derive_seed(config.seed, index, "seat:1"),
                ),
            )
            episodes.extend(pair)
            for seat in (0, 1):
                scores[seat] += float(pair[seat].rewards.sum())
        report = trainer.update(tuple(episodes))
        save_checkpoint(trainer, directory / "latest.pt")
        row = {
            "update": trainer.update_count,
            "games": config.games_per_update,
            "mean_scores": [s / config.games_per_update for s in scores],
            "ppo": asdict(report),
        }
        with (directory / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return trainer


def evaluate_model(
    model: GuildworksRecurrentPolicy, *, games_per_seat: int, seed: int
) -> list[dict[str, Any]]:
    if type(games_per_seat) is not int or games_per_seat <= 0 or type(seed) is not int:
        raise ValueError("Invalid evaluation budget or seed")
    summaries = []
    for seat in (0, 1):
        policy = RecurrentGuildworksAgent(model)
        random_agent = RandomAgent[GuildworksObservation]()
        agents: tuple[Agent[GuildworksObservation], ...] = (
            (policy, random_agent) if seat == 0 else (random_agent, policy)
        )
        report = evaluate_batch(
            GuildworksEngine(),
            agents,
            outcome_for=guildworks_outcome,
            agent_labels=("policy", "random") if seat == 0 else ("random", "policy"),
            num_matches=games_per_seat,
            seed=derive_seed(seed, seat, "evaluation"),
            max_steps=32,
        )
        if report.summary.errors or report.summary.truncated:
            raise RuntimeError("Evaluation did not complete")
        summaries.append({"policy_seat": seat, **asdict(report.summary)})
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--games-per-update", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-games-per-seat", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.threads <= 0 or args.eval_games_per_seat <= 0:
        parser.error("Thread and evaluation counts must be positive")
    torch.set_num_threads(args.threads)
    try:
        config = RecurrentRunConfig(
            updates=args.updates,
            games_per_update=args.games_per_update,
            seed=args.seed,
            device=args.device,
        )
        trainer = run_training(args.output, config, resume=args.resume)
        evaluation = {
            "protocol": "guildworks-functional-v1",
            "evaluation_seed": args.eval_seed,
            "updates": trainer.update_count,
            "initial": evaluate_model(
                initial_model(derive_seed(args.seed, 0, "initialization")),
                games_per_seat=args.eval_games_per_seat,
                seed=args.eval_seed,
            ),
            "trained": evaluate_model(
                trainer.model,
                games_per_seat=args.eval_games_per_seat,
                seed=args.eval_seed,
            ),
        }
        (Path(args.output) / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(evaluation, indent=2))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
