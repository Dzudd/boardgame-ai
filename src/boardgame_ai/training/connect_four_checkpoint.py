"""Portable update-boundary Connect Four PPO checkpoints, separate from runs."""

import os
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from boardgame_ai.models.connect_four import (
    ENCODING_ID,
    MODEL_ID,
    initial_connect_four_model,
)
from boardgame_ai.training.checkpoint import _KEYS as _COMMON_KEYS
from boardgame_ai.training.checkpoint import _validate_payload
from boardgame_ai.training.connect_four_ppo import LEARNER_ID, ConnectFourPPOTrainer
from boardgame_ai.training.device import resolve_training_device
from boardgame_ai.training.ppo import PPOConfig

_KEYS = _COMMON_KEYS | {"encoding_id", "learner_id", "samples_seen"}


def _same_types_and_values(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(
            _same_types_and_values(a[k], b[k]) for k in a
        )
    if isinstance(a, list):
        return len(a) == len(b) and all(
            _same_types_and_values(x, y) for x, y in zip(a, b, strict=True)
        )
    return bool(a == b)


def _validate(payload: Any, trainer: ConnectFourPPOTrainer) -> None:
    if (
        not isinstance(payload, dict)
        or set(payload) != _KEYS
        or payload["encoding_id"] != ENCODING_ID
        or payload["learner_id"] != LEARNER_ID
        or type(payload["samples_seen"]) is not int
        or payload["samples_seen"] < 0
    ):
        raise ValueError("Invalid Connect Four checkpoint fields or identity")
    _validate_payload(
        {key: payload[key] for key in _COMMON_KEYS},
        trainer,
        game_id="connect_four",
        model_id=MODEL_ID,
    )
    updates = payload["update_count"]
    steps = payload["optimizer_steps"]
    samples = payload["samples_seen"]
    epochs = trainer.config.epochs
    size = trainer.config.minibatch_size
    if (
        (updates == 0 and (samples != 0 or steps != 0))
        or (updates > 0 and samples < updates)
        or steps % epochs
        or steps < epochs * ((samples + size - 1) // size)
        or steps > epochs * ((samples + updates * (size - 1)) // size)
    ):
        raise ValueError("Connect Four checkpoint sample/optimizer counters disagree")
    optimizer = payload["optimizer_state"]
    default_groups = torch.optim.Adam(
        trainer.model.parameters(), lr=trainer.config.learning_rate
    ).state_dict()["param_groups"]
    if not _same_types_and_values(optimizer["param_groups"], default_groups):
        raise ValueError("Connect Four Adam configuration types do not match")
    if any(float(m["step"].item()) != steps for m in optimizer["state"].values()):
        raise ValueError("Connect Four Adam steps must match completed optimizer work")
    if payload["shuffle_rng_state"].device.type != "cpu":
        raise ValueError("Connect Four shuffle state must be on CPU")


def connect_four_payload(trainer: ConnectFourPPOTrainer) -> dict[str, Any]:
    """Validated portable trainer payload for atomic run envelopes."""
    if not trainer.at_update_boundary:
        raise ValueError("Expected a successful PPO boundary")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "game_id": "connect_four",
        "model_id": MODEL_ID,
        "encoding_id": ENCODING_ID,
        "learner_id": LEARNER_ID,
        "config": asdict(trainer.config),
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in trainer.model.state_dict().items()
        },
        "optimizer_state": deepcopy(trainer.optimizer.state_dict()),
        "shuffle_rng_state": trainer.generator.get_state().clone(),
        "update_count": trainer.update_count,
        "optimizer_steps": trainer.optimizer_steps,
        "samples_seen": trainer.samples_seen,
        "model_training": trainer.model.training,
    }
    for moments in payload["optimizer_state"]["state"].values():
        for name, value in moments.items():
            if isinstance(value, torch.Tensor):
                moments[name] = value.detach().cpu().clone()
    _validate(payload, trainer)
    return payload


def save_connect_four_checkpoint(
    trainer: ConnectFourPPOTrainer, path: str | Path
) -> None:
    """Validate before atomically publishing a successful optimizer boundary.

    Existing destinations must be compatible Connect Four checkpoints. Failed
    updates cannot be saved. No run seeds/manifests, trajectories, global RNG or
    gradients are stored; this is trainer restore, not a complete run scheduler.
    Parent must exist. Like training, publication assumes a single writer.
    """
    if not isinstance(trainer, ConnectFourPPOTrainer) or not trainer.at_update_boundary:
        raise ValueError("Expected a successful Connect Four PPO update boundary")
    payload = connect_four_payload(trainer)
    destination = Path(path)
    if destination.exists():
        existing = torch.load(destination, map_location="cpu", weights_only=True)
        _validate(existing, trainer)
        if existing["update_count"] > trainer.update_count:
            raise ValueError("Cannot replace a newer Connect Four checkpoint")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".connect-four-ppo-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_connect_four_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_config: PPOConfig | None = None,
) -> ConnectFourPPOTrainer:
    """Restricted load into a fresh trainer, preserving global RNG on failure.

    Exact CPU continuation requires the same next collected batches/runtime.
    Games, scheduling seeds, protocol identity and elapsed caps belong to the
    future run layer. Cross-device numerical identity is not promised.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return trainer_from_connect_four_payload(
        payload, device=device, expected_config=expected_config
    )


def trainer_from_connect_four_payload(
    payload: Any,
    *,
    device: str | torch.device = "cpu",
    expected_config: PPOConfig | None = None,
) -> ConnectFourPPOTrainer:
    """Validate an embedded payload before constructing a fresh trainer."""
    target = resolve_training_device(device)
    if (
        not isinstance(payload, dict)
        or set(payload) != _KEYS
        or not isinstance(payload["config"], dict)
        or set(payload["config"]) != set(asdict(PPOConfig()))
    ):
        raise ValueError("Invalid Connect Four checkpoint fields/configuration")
    config = PPOConfig(**payload["config"])
    if expected_config is not None and config != expected_config:
        raise ValueError("Connect Four checkpoint configuration does not match")
    trainer = ConnectFourPPOTrainer(initial_connect_four_model(0), config)
    _validate(payload, trainer)
    trainer.model.load_state_dict(payload["model_state"], strict=True)
    if target.type != "cpu":
        trainer = ConnectFourPPOTrainer(trainer.model.to(target), config)
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.generator.set_state(payload["shuffle_rng_state"])
    trainer.update_count = payload["update_count"]
    trainer.optimizer_steps = payload["optimizer_steps"]
    trainer.samples_seen = payload["samples_seen"]
    trainer.model.train(payload["model_training"])
    return trainer
