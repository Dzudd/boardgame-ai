"""Checkpoint validation, atomic writes, and exact next-update restoration."""

from pathlib import Path
from typing import Any

import pytest
import torch

from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.training import (
    PPOBatch,
    PPOConfig,
    PPOTrainer,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
)


@pytest.fixture
def trainer() -> PPOTrainer:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = TicTacToeActorCritic()
    return PPOTrainer(model, PPOConfig(epochs=1, minibatch_size=2), shuffle_seed=123)


def batch() -> PPOBatch:
    return PPOBatch(
        observations=torch.ones(3, 6, 3, 3),
        legal_masks=torch.ones(3, 9, dtype=torch.bool),
        actions=torch.tensor([0, 1, 2]),
        old_log_probs=torch.full((3,), -2.2),
        advantages=torch.tensor([1.0, -1.0, 0.5]),
        returns=torch.tensor([1.0, -1.0, 0.5]),
    )


def assert_nested_equal(first: Any, second: Any) -> None:
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_nested_equal(a, b)
    else:
        assert first == second


def test_checkpoint_exact_next_update_and_global_rng(
    trainer: PPOTrainer, tmp_path: Path
) -> None:
    trainer.update(batch())
    trainer.model.eval()
    path = tmp_path / "ppo.pt"
    save_ppo_checkpoint(trainer, path)
    rng = torch.random.get_rng_state().clone()
    restored = load_ppo_checkpoint(path)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert not restored.model.training
    assert restored.config == trainer.config
    assert restored.update_count == trainer.update_count == 1
    assert_nested_equal(trainer.model.state_dict(), restored.model.state_dict())
    assert_nested_equal(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
    assert torch.equal(trainer.generator.get_state(), restored.generator.get_state())
    original_report = trainer.update(batch())
    restored_report = restored.update(batch())
    assert original_report == restored_report
    assert_nested_equal(trainer.model.state_dict(), restored.model.state_dict())
    assert_nested_equal(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
    assert restored.optimizer_steps == trainer.optimizer_steps == 4


def test_unused_optimizer_and_cpu_tensor_storage(
    trainer: PPOTrainer, tmp_path: Path
) -> None:
    path = tmp_path / "ppo.pt"
    save_ppo_checkpoint(trainer, path)
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {
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
    assert all(t.device.type == "cpu" for t in payload["model_state"].values())
    assert load_ppo_checkpoint(path).optimizer.state == {}
    assert len(list(tmp_path.iterdir())) == 1  # temporary sibling removed


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "game",
        "model",
        "extra",
        "missing",
        "config",
        "config_missing",
        "weight_shape",
        "weight_nan",
        "weight_dtype",
        "counter",
        "mode",
        "rng",
        "optimizer_lr",
        "optimizer_shape",
        "optimizer_moment",
        "optimizer_step",
    ],
)
def test_corrupt_checkpoint_rejects(
    trainer: PPOTrainer, tmp_path: Path, change: str
) -> None:
    trainer.update(batch())
    path = tmp_path / "ppo.pt"
    save_ppo_checkpoint(trainer, path)
    payload = torch.load(path, weights_only=True)
    if change == "version":
        payload["schema_version"] = 2
    elif change == "game":
        payload["game_id"] = "other"
    elif change == "model":
        payload["model_id"] = "other"
    elif change == "extra":
        payload["engine"] = {}
    elif change == "missing":
        del payload["shuffle_rng_state"]
    elif change == "config":
        payload["config"]["clip_ratio"] = -1.0
    elif change == "config_missing":
        del payload["config"]["epochs"]
    elif change == "weight_shape":
        payload["model_state"]["policy_head.weight"] = torch.zeros(2)
    elif change == "weight_nan":
        payload["model_state"]["policy_head.weight"].fill_(float("nan"))
    elif change == "weight_dtype":
        payload["model_state"]["policy_head.weight"] = payload["model_state"][
            "policy_head.weight"
        ].double()
    elif change == "counter":
        payload["optimizer_steps"] = -1
    elif change == "mode":
        payload["model_training"] = 1
    elif change == "rng":
        payload["shuffle_rng_state"] = torch.empty(0, dtype=torch.uint8)
    elif change == "optimizer_lr":
        payload["optimizer_state"]["param_groups"][0]["lr"] = 1.0
    elif change == "optimizer_shape":
        payload["optimizer_state"]["state"][0]["exp_avg"] = torch.zeros(1)
    elif change == "optimizer_moment":
        payload["optimizer_state"]["state"][0]["exp_avg_sq"].fill_(-1.0)
    elif change == "optimizer_step":
        payload["optimizer_state"]["state"][0]["step"].fill_(999.0)
    torch.save(payload, path)
    before = trainer.model.policy_head.weight.detach().clone()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        load_ppo_checkpoint(path)
    assert torch.equal(trainer.model.policy_head.weight, before)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_invalid_save_preserves_existing_file(
    trainer: PPOTrainer, tmp_path: Path
) -> None:
    path = tmp_path / "existing.pt"
    path.write_bytes(b"keep me")
    with torch.no_grad():
        trainer.model.policy_head.weight.fill_(float("nan"))
    with pytest.raises(ValueError):
        save_ppo_checkpoint(trainer, path)
    assert path.read_bytes() == b"keep me"
    assert len(list(tmp_path.iterdir())) == 1


def test_modified_optimizer_settings_reject_on_save(
    trainer: PPOTrainer, tmp_path: Path
) -> None:
    trainer.optimizer.param_groups[0]["lr"] = 1.0
    with pytest.raises(ValueError, match="configuration"):
        save_ppo_checkpoint(trainer, tmp_path / "invalid.pt")
    assert not list(tmp_path.iterdir())
