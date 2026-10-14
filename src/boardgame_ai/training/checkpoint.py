"""Versioned CPU model/Adam checkpoints without engine or rollout data."""

import os
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.training.device import resolve_training_device
from boardgame_ai.training.ppo import PPOConfig, PPOTrainer

_KEYS = {
    "schema_version",
    "game_id",
    "model_id",
    "config",
    "model_state",
    "optimizer_state",
    "shuffle_rng_state",
    "update_count",
    "optimizer_steps",
    "model_training",
}


def _validate_payload(payload: Any, trainer: PPOTrainer) -> None:
    if not isinstance(payload, dict) or set(payload) != _KEYS:
        raise ValueError("Unsupported PPO checkpoint fields")
    if not isinstance(payload["config"], dict) or payload["config"] != asdict(
        trainer.config
    ):
        raise ValueError("Checkpoint configuration does not match")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["game_id"] != "tic_tac_toe"
        or payload["model_id"] != "ttt-actor-critic-obs6-actions9-v1"
    ):
        raise ValueError("Unsupported PPO checkpoint version/game/model")
    if type(payload["model_training"]) is not bool:
        raise ValueError("Checkpoint model mode must be boolean")
    for name in ("update_count", "optimizer_steps"):
        if type(payload[name]) is not int or payload[name] < 0:
            raise ValueError("Checkpoint counters must be nonnegative integers")
    if payload["update_count"] > payload["optimizer_steps"]:
        raise ValueError("Checkpoint counters disagree")
    expected = trainer.model.state_dict()
    state = payload["model_state"]
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Checkpoint model fields do not match")
    for name, reference in expected.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device.type != "cpu"
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError("Checkpoint model tensors are incompatible")
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
        raise ValueError("Checkpoint optimizer configuration does not match")
    parameters = list(trainer.model.parameters())
    for index, moments in optimizer["state"].items():
        if (
            type(index) is not int
            or not 0 <= index < len(parameters)
            or not isinstance(moments, dict)
            or set(moments) != {"step", "exp_avg", "exp_avg_sq"}
        ):
            raise ValueError("Checkpoint Adam state fields are incompatible")
        for name, tensor in moments.items():
            shape = () if name == "step" else parameters[index].shape
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != shape
                or (
                    tensor.dtype not in (torch.float32, torch.float64)
                    if name == "step"
                    else tensor.dtype != torch.float32
                )
                or tensor.device.type != "cpu"
                or not torch.isfinite(tensor).all().item()
            ):
                raise ValueError("Checkpoint Adam tensors are incompatible")
        step = float(moments["step"].item())
        if (
            step < 1
            or step != int(step)
            or step > payload["optimizer_steps"]
            or (moments["exp_avg_sq"] < 0).any().item()
        ):
            raise ValueError("Checkpoint Adam moments/steps are invalid")
    if payload["optimizer_steps"] == 0 and optimizer["state"]:
        raise ValueError("Unused optimizer must have no moments")
    if payload["optimizer_steps"] > 0 and set(optimizer["state"]) != set(
        range(len(parameters))
    ):
        raise ValueError("Checkpoint must contain all Adam parameter states")
    rng = payload["shuffle_rng_state"]
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
        raise ValueError("Checkpoint shuffle RNG state is incompatible")
    try:
        torch.Generator(device="cpu").set_state(rng)
    except RuntimeError as error:
        raise ValueError("Checkpoint shuffle RNG state is invalid") from error


def save_ppo_checkpoint(trainer: PPOTrainer, path: str | Path) -> None:
    """Validate then atomically replace a file; parent folder must exist.

    Stores model, Adam, configuration, update/step counters, mode, and private
    shuffle RNG at an update boundary. No games, observations, sampling seeds,
    gradients, or global RNG state. This is not a mid-update/episode checkpoint.
    """
    payload: dict[str, Any] = {
        "schema_version": 1,
        "game_id": "tic_tac_toe",
        "model_id": "ttt-actor-critic-obs6-actions9-v1",
        "config": asdict(trainer.config),
        "model_state": {
            k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()
        },
        "optimizer_state": deepcopy(trainer.optimizer.state_dict()),
        "shuffle_rng_state": trainer.generator.get_state().clone(),
        "update_count": trainer.update_count,
        "optimizer_steps": trainer.optimizer_steps,
        "model_training": trainer.model.training,
    }
    for moments in payload["optimizer_state"]["state"].values():
        for name, value in moments.items():
            if isinstance(value, torch.Tensor):
                moments[name] = value.detach().cpu().clone()
    _validate_payload(payload, trainer)
    destination = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".ppo-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_ppo_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> PPOTrainer:
    """Restore a trainer on an explicit device with restricted loading.

    Validate schema/config/tensors/Adam/RNG before loading into the fresh trainer.
    Stored tensors are portable CPU tensors. Adam moments follow model parameters.
    Existing trainers are never modified; global Torch RNG is preserved during
    model construction. Exact continuation requires the same next batch/runtime.
    """
    target_device = resolve_training_device(device)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != _KEYS
        or not isinstance(payload["config"], dict)
    ):
        raise ValueError("Unsupported PPO checkpoint fields/configuration")
    if set(payload["config"]) != set(asdict(PPOConfig())):
        raise ValueError("Checkpoint configuration fields do not match")
    try:
        config = PPOConfig(**payload["config"])
    except TypeError as error:
        raise ValueError("Checkpoint configuration fields do not match") from error
    with torch.random.fork_rng(devices=[]):
        trainer = PPOTrainer(TicTacToeActorCritic(), config, shuffle_seed=0)
    _validate_payload(payload, trainer)
    trainer.model.load_state_dict(payload["model_state"], strict=True)
    if target_device.type != "cpu":
        model = trainer.model.to(target_device)
        trainer = PPOTrainer(model, config, shuffle_seed=0)
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.generator.set_state(payload["shuffle_rng_state"])
    trainer.update_count = payload["update_count"]
    trainer.optimizer_steps = payload["optimizer_steps"]
    trainer.model.train(payload["model_training"])
    return trainer
