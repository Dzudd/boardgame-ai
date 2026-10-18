"""Restricted, update-boundary checkpoints for search-target learning and replay."""

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Any

import torch

from boardgame_ai.search.puct import SearchConfig
from boardgame_ai.training.search_learning import (
    LEARNER_ID,
    MODEL_ID,
    SearchTrainer,
    SearchTrainingConfig,
    game_from_payload,
    game_payload,
    validate_game,
)

_KEYS = {
    "schema_version",
    "game_id",
    "learner_id",
    "model_id",
    "config",
    "model_state",
    "optimizer_state",
    "replay_rng_state",
    "replay",
    "update_count",
    "optimizer_steps",
    "games_seen",
    "decisions_seen",
    "last_report",
    "model_training",
}


def config_from_payload(payload: dict[str, Any]) -> SearchTrainingConfig:
    if (
        not isinstance(payload, dict)
        or set(payload) != set(SearchTrainingConfig.__dataclass_fields__)
        or not isinstance(payload["search"], dict)
        or set(payload["search"]) != set(SearchConfig.__dataclass_fields__)
    ):
        raise ValueError("Invalid search training configuration fields")
    return SearchTrainingConfig(
        **{**payload, "search": SearchConfig(**payload["search"])}
    )


def _validate(payload: Any, trainer: SearchTrainer) -> None:
    if (
        not isinstance(payload, dict)
        or set(payload) != _KEYS
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["learner_id"] != LEARNER_ID
        or payload["model_id"] != MODEL_ID
        or payload["game_id"] != "tic_tac_toe"
        or payload["config"] != asdict(trainer.config)
        or type(payload["model_training"]) is not bool
    ):
        raise ValueError("Invalid search checkpoint identity or fields")
    if any(
        type(payload[name]) is not int or payload[name] < 0
        for name in ("update_count", "optimizer_steps", "games_seen", "decisions_seen")
    ):
        raise ValueError("Invalid search checkpoint counters")
    config = trainer.config
    if (
        payload["optimizer_steps"]
        != payload["update_count"] * config.optimizer_steps_per_update
        or payload["games_seen"] != payload["update_count"] * config.games_per_update
        or not 5 * payload["games_seen"]
        <= payload["decisions_seen"]
        <= 9 * payload["games_seen"]
    ):
        raise ValueError("Search checkpoint counters disagree")
    state = payload["model_state"]
    if not isinstance(state, dict) or set(state) != set(trainer.model.state_dict()):
        raise ValueError("Invalid search model state")
    for name, reference in trainer.model.state_dict().items():
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != reference.dtype
            or tensor.shape != reference.shape
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("Invalid search model tensors")
    optimizer = payload["optimizer_state"]
    default = trainer.optimizer.state_dict()
    if (
        not isinstance(optimizer, dict)
        or set(optimizer) != {"state", "param_groups"}
        or optimizer["param_groups"] != default["param_groups"]
        or not isinstance(optimizer["state"], dict)
    ):
        raise ValueError("Invalid search Adam configuration")
    parameters = list(trainer.model.parameters())
    expected = set(range(len(parameters))) if payload["optimizer_steps"] else set()
    if set(optimizer["state"]) != expected:
        raise ValueError("Incomplete search Adam state")
    for index, moments in optimizer["state"].items():
        if (
            type(index) is not int
            or not isinstance(moments, dict)
            or set(moments) != {"step", "exp_avg", "exp_avg_sq"}
        ):
            raise ValueError("Invalid search Adam fields")
        for name, tensor in moments.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.shape != (() if name == "step" else parameters[index].shape)
                or tensor.dtype != torch.float32
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError("Invalid search Adam tensors")
        if float(moments["step"]) != payload["optimizer_steps"] or bool(
            (moments["exp_avg_sq"] < 0).any()
        ):
            raise ValueError("Invalid search Adam moments or step")
    rng = payload["replay_rng_state"]
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
        raise ValueError("Invalid search replay RNG")
    try:
        torch.Generator().set_state(rng)
    except RuntimeError as error:
        raise ValueError("Invalid search replay RNG") from error
    replay = payload["replay"]
    if not isinstance(replay, list) or len(replay) != min(
        config.replay_capacity_games, payload["games_seen"]
    ):
        raise ValueError("Invalid search replay capacity/history")
    hashes: dict[int, str] = {}
    retained_decisions = 0
    for index, data in enumerate(replay, start=payload["games_seen"] - len(replay)):
        game = game_from_payload(data)
        validate_game(game)
        retained_decisions += len(game.samples)
        if (
            game.game_index != index
            or game.search != config.search
            or game.policy_version != index // config.games_per_update
            or hashes.setdefault(game.policy_version, game.policy_sha256)
            != game.policy_sha256
        ):
            raise ValueError("Invalid search replay schedule or policy version")
    evicted = payload["games_seen"] - len(replay)
    if not 5 * evicted <= payload["decisions_seen"] - retained_decisions <= 9 * evicted:
        raise ValueError("Search replay and decision counters disagree")
    report = payload["last_report"]
    if payload["update_count"] == 0:
        if report is not None:
            raise ValueError("Unused search trainer must have no report")
    elif (
        not isinstance(report, dict)
        or any(
            report.get(name) != payload[name]
            for name in ("optimizer_steps", "games_seen", "decisions_seen")
        )
        or report.get("update") != payload["update_count"]
    ):
        raise ValueError("Search boundary report counters disagree")
    if report is not None:
        fields = {
            "update",
            "games_seen",
            "decisions_seen",
            "optimizer_steps",
            "collected_games",
            "collected_decisions",
            "search_simulations",
            "evaluator_calls",
            "policy_loss",
            "value_loss",
            "sampled_positions",
            "replay_games",
            "replay_age_min",
            "replay_age_max",
            "replay_age_mean",
        }
        if (
            set(report) != fields
            or any(
                type(value) not in (int, float) or not isfinite(value)
                for value in report.values()
            )
            or any(
                type(report[name]) is not int
                for name in fields - {"policy_loss", "value_loss", "replay_age_mean"}
            )
            or report["collected_games"] != config.games_per_update
            or not 5 * config.games_per_update
            <= report["collected_decisions"]
            <= 9 * config.games_per_update
            or report["search_simulations"]
            != report["collected_decisions"] * config.search.simulations
            or not report["collected_decisions"]
            <= report["evaluator_calls"]
            <= report["collected_decisions"] * (config.search.simulations + 1)
            or report["sampled_positions"]
            != config.batch_size * config.optimizer_steps_per_update
            or report["replay_games"] != len(replay)
            or not 0
            <= report["replay_age_min"]
            <= report["replay_age_mean"]
            <= report["replay_age_max"]
            < payload["update_count"]
            or report["value_loss"] < 0
            or report["policy_loss"] < 0
        ):
            raise ValueError("Invalid search boundary report")
        json.dumps(report, allow_nan=False)


def save_search_checkpoint(
    trainer: SearchTrainer,
    path: str | Path,
    *,
    last_report: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "game_id": "tic_tac_toe",
        "learner_id": LEARNER_ID,
        "model_id": MODEL_ID,
        "config": asdict(trainer.config),
        "model_state": {
            k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()
        },
        "optimizer_state": deepcopy(trainer.optimizer.state_dict()),
        "replay_rng_state": trainer.generator.get_state().clone(),
        "replay": [game_payload(game) for game in trainer.replay],
        "update_count": trainer.update_count,
        "optimizer_steps": trainer.optimizer_steps,
        "games_seen": trainer.games_seen,
        "decisions_seen": trainer.decisions_seen,
        "model_training": trainer.model.training,
        "last_report": last_report,
    }
    _validate(payload, trainer)
    destination = Path(path)
    fd, name = tempfile.mkstemp(
        prefix=".search-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def load_search_checkpoint(
    path: str | Path,
) -> tuple[SearchTrainer, dict[str, Any] | None]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != _KEYS:
        raise ValueError(
            "Invalid search checkpoint fields; PPO checkpoints are incompatible"
        )
    config = config_from_payload(payload["config"])
    trainer = SearchTrainer(config)
    _validate(payload, trainer)
    trainer.model.load_state_dict(payload["model_state"], strict=True)
    trainer.model.train(payload["model_training"])
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.generator.set_state(payload["replay_rng_state"])
    trainer.replay = [game_from_payload(game) for game in payload["replay"]]
    for name in ("update_count", "optimizer_steps", "games_seen", "decisions_seen"):
        setattr(trainer, name, payload[name])
    return trainer, payload["last_report"]
