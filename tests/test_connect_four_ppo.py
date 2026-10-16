"""Connect Four PPO oracles, save safety and exact next-update restoration."""

import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from boardgame_ai.games.connect_four import ConnectFourEngine
from boardgame_ai.models.connect_four import initial_connect_four_model
from boardgame_ai.training.checkpoint import load_ppo_checkpoint, save_ppo_checkpoint
from boardgame_ai.training.connect_four import (
    collect_connect_four_self_play,
    derive_connect_four_seed,
)
from boardgame_ai.training.connect_four_checkpoint import (
    load_connect_four_checkpoint as load,
)
from boardgame_ai.training.connect_four_checkpoint import (
    save_connect_four_checkpoint as save,
)
from boardgame_ai.training.connect_four_ppo import (
    DEFAULT_CONFIG,
    ConnectFourPPOTrainer,
    connect_four_ppo_loss,
    prepare_connect_four_batch,
    validate_connect_four_batch,
)
from boardgame_ai.training.ppo import PPOBatch, PPOConfig, validate_ppo_batch
from tests.test_ppo_checkpoint import assert_nested_equal


def trainer(*, device: str = "cpu") -> ConnectFourPPOTrainer:
    return ConnectFourPPOTrainer(
        initial_connect_four_model(91).to(device),
        PPOConfig(gamma=1.0, epochs=2, minibatch_size=4),
        shuffle_seed=19,
    )


def batch(n: int = 5) -> PPOBatch:
    return PPOBatch(
        observations=torch.ones(n, 6, 6, 7),
        legal_masks=torch.tensor([[True, True, False, False, False, False, False]])
        .expand(n, 7)
        .clone(),
        actions=torch.arange(n) % 2,
        old_log_probs=torch.full((n,), -math.log(2)),
        advantages=torch.linspace(-1, 1, n),
        returns=torch.linspace(1, -1, n),
    )


def test_defaults_match_predeclared_numeric_protocol() -> None:
    settings = json.loads(
        Path("docs/phase6/connect-four-functional-v1.json").read_text()
    )
    assert asdict(DEFAULT_CONFIG) == settings["training"]["ppo"]


def test_loss_matches_independent_hand_calculation_and_illegal_gradients() -> None:
    b = replace(
        batch(2),
        advantages=torch.tensor([1.0, -1.0]),
        returns=torch.tensor([1.0, -1.0]),
    )
    logits = torch.tensor(
        [
            [math.log(3), 0, 100, 100, 100, 100, 100],
            [math.log(3), 0, 100, 100, 100, 100, 100],
        ],
        requires_grad=True,
    )
    values = torch.tensor([0.0, 0.5], requires_grad=True)
    loss = connect_four_ppo_loss(logits, values, b)
    # New selected probabilities .75/.25 versus old .5/.5: ratios 1.5/.5.
    # Surrogates min(1.5,1.2)=1.2 and min(-.5,-.8)=-.8.
    entropy = -0.75 * math.log(0.75) - 0.25 * math.log(0.25)
    assert loss.policy.item() == pytest.approx(-0.2, abs=1e-6)
    assert loss.value.item() == pytest.approx((1 + 2.25) / 4)
    assert loss.entropy.item() == pytest.approx(entropy)
    assert loss.clip_fraction.item() == 1
    assert loss.total.item() == pytest.approx(
        -0.2 + 0.5 * (3.25 / 4) - 0.01 * entropy, abs=1e-6
    )
    loss.total.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None and not logits.grad[:, 2:].any()
    assert values.grad is not None and torch.equal(
        values.grad, torch.tensor([-0.25, 0.375])
    )


def test_trajectory_builder_keeps_own_seat_gae_and_owned_storage() -> None:
    model = initial_connect_four_model(91)
    rollout = collect_connect_four_self_play(
        ConnectFourEngine(), model, sampling_seeds=(17, 18)
    )
    b = prepare_connect_four_batch(
        rollout.players, PPOConfig(gamma=1.0, gae_lambda=1.0)
    )
    assert len(b.actions) == rollout.num_steps
    offset = 0
    for seat, segment in enumerate(rollout.players):
        count = len(segment.actions)
        # At lambda=gamma=1 each own decision has the same terminal return.
        assert torch.allclose(
            b.returns[offset : offset + count],
            torch.full((count,), rollout.total_rewards[seat]),
            atol=1e-6,
        )
        assert torch.allclose(
            b.advantages[offset : offset + count],
            b.returns[offset : offset + count] - segment.values,
        )
        offset += count
    saved = b.observations.clone()
    rollout.players[0].observations.zero_()
    assert torch.equal(saved, b.observations)
    with pytest.raises(ValueError):
        validate_ppo_batch(b)
    with pytest.raises(ValueError):
        validate_connect_four_batch(
            replace(b, observations=torch.zeros(len(b.actions), 6, 3, 3))
        )


@pytest.mark.parametrize("change", ["bootstrap", "seat", "steps", "values", "empty"])
def test_bad_trajectory_rejects(change: str) -> None:
    rollout = collect_connect_four_self_play(
        ConnectFourEngine(), initial_connect_four_model(91), sampling_seeds=(17, 18)
    )
    segment = rollout.players[0]
    changes: dict[str, Any] = {
        "bootstrap": {"bootstrap_value": 1.0},
        "seat": {"player_id": True},
        "steps": {"step_indices": torch.zeros_like(segment.step_indices)},
        "values": {"values": torch.full_like(segment.values, float("nan"))},
        "empty": {"actions": torch.empty(0, dtype=torch.int64)},
    }
    with pytest.raises(ValueError):
        prepare_connect_four_batch((replace(segment, **changes[change]),))


@pytest.mark.parametrize(
    "change", ["action", "illegal", "nan", "log-prob", "mask-shape", "dtype", "empty"]
)
def test_bad_batch_rejects_without_optimizer_or_shuffle_changes(change: str) -> None:
    t = trainer()
    b = batch()
    changes: dict[str, Any] = {
        "action": {"actions": torch.full((5,), 7, dtype=torch.int64)},
        "illegal": {"actions": torch.full((5,), 2, dtype=torch.int64)},
        "nan": {"returns": torch.full((5,), float("nan"))},
        "log-prob": {"old_log_probs": torch.ones(5)},
        "mask-shape": {"legal_masks": torch.ones(5, 9, dtype=torch.bool)},
        "dtype": {"actions": torch.zeros(5)},
        "empty": {"actions": torch.empty(0, dtype=torch.int64)},
    }
    weights = deepcopy(t.model.state_dict())
    rng = t.generator.get_state().clone()
    with pytest.raises(ValueError):
        t.update(replace(b, **changes[change]))
    assert (
        t.at_update_boundary
        and t.update_count == t.optimizer_steps == t.samples_seen == 0
    )
    assert_nested_equal(weights, t.model.state_dict())
    assert torch.equal(rng, t.generator.get_state())


def test_update_changes_parameters_uses_every_sample_and_preserves_inputs_rng() -> None:
    t = trainer()
    b = batch()
    before = deepcopy(t.model.state_dict())
    inputs = deepcopy(b)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    rng = torch.random.get_rng_state().clone()
    report = t.update(b)
    assert (
        report.optimizer_steps == t.optimizer_steps == 4
    )  # 5 samples, batch 4, two epochs
    assert t.update_count == 1 and t.samples_seen == report.num_samples == 5
    assert all(math.isfinite(v) for v in asdict(report).values())
    assert any(not torch.equal(v, before[k]) for k, v in t.model.state_dict().items())
    for name in b.__dataclass_fields__:
        assert torch.equal(getattr(inputs, name), getattr(b, name))
    assert random.getstate() == python_rng
    after = np.random.get_state()
    assert (
        after[0] == numpy_rng[0]
        and np.array_equal(after[1], numpy_rng[1])
        and after[2:] == numpy_rng[2:]
    )
    assert torch.equal(rng, torch.random.get_rng_state())


def test_checkpoint_roundtrip_exact_next_update(tmp_path: Path) -> None:
    t = trainer()
    t.update(batch())
    t.model.eval()
    path = tmp_path / "latest.pt"
    save(t, path)
    original = path.read_bytes()
    rng = torch.random.get_rng_state().clone()
    restored = load(path, expected_config=t.config)
    assert path.read_bytes() == original and not restored.model.training
    assert torch.equal(rng, torch.random.get_rng_state())
    assert restored.samples_seen == t.samples_seen == 5
    assert t.update(batch()) == restored.update(batch())
    assert_nested_equal(t.model.state_dict(), restored.model.state_dict())
    assert_nested_equal(t.optimizer.state_dict(), restored.optimizer.state_dict())
    assert torch.equal(t.generator.get_state(), restored.generator.get_state())
    assert restored.samples_seen == 10


def test_two_plus_two_self_play_updates_resume_exactly(tmp_path: Path) -> None:
    t = trainer()

    def update_one(value: ConnectFourPPOTrainer, index: int) -> Any:
        game = collect_connect_four_self_play(
            ConnectFourEngine(),
            value.model,
            engine_seed=derive_connect_four_seed(999, index, "engine"),
            sampling_seeds=(
                derive_connect_four_seed(999, index, "seat:0"),
                derive_connect_four_seed(999, index, "seat:1"),
            ),
        )
        return value.update(prepare_connect_four_batch(game.players, value.config))

    for i in range(2):
        update_one(t, i)
    path = tmp_path / "latest.pt"
    save(t, path)
    restored = load(path)
    for i in range(2, 4):
        assert update_one(t, i) == update_one(restored, i)
    assert_nested_equal(t.model.state_dict(), restored.model.state_dict())
    assert_nested_equal(t.optimizer.state_dict(), restored.optimizer.state_dict())
    assert torch.equal(t.generator.get_state(), restored.generator.get_state())
    assert restored.update_count == t.update_count == 4
    assert restored.samples_seen == t.samples_seen


def test_initial_checkpoint_excludes_training_records_and_has_cpu_tensors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "initial.pt"
    save(trainer(), path)
    payload = torch.load(path, weights_only=True)
    assert payload["samples_seen"] == payload["update_count"] == 0
    assert set(payload) == {
        "schema_version",
        "game_id",
        "model_id",
        "encoding_id",
        "learner_id",
        "config",
        "model_state",
        "optimizer_state",
        "shuffle_rng_state",
        "update_count",
        "optimizer_steps",
        "samples_seen",
        "model_training",
    }
    assert all(x.device.type == "cpu" for x in payload["model_state"].values())
    assert load(path).optimizer.state_dict()["state"] == {}


@pytest.mark.parametrize(
    "corruption",
    [
        "schema",
        "game",
        "model",
        "encoding",
        "learner",
        "weights",
        "weight-shape",
        "config",
        "extra",
        "counter",
        "bool-counter",
        "samples",
        "mode",
        "rng",
        "adam-missing",
        "adam-nan",
        "adam-negative",
        "adam-step",
        "adam-bool-index",
    ],
)
def test_corrupt_checkpoint_rejects_without_source_or_rng_changes(
    tmp_path: Path, corruption: str
) -> None:
    t = trainer()
    t.update(batch())
    path = tmp_path / "bad.pt"
    save(t, path)
    p = torch.load(path, weights_only=True)
    if corruption in ("game", "model", "encoding", "learner"):
        p[f"{corruption}_id"] = "unsupported"
    elif corruption == "schema":
        p["schema_version"] = True
    elif corruption == "weights":
        p["model_state"]["body.0.weight"][0, 0] = float("nan")
    elif corruption == "weight-shape":
        p["model_state"]["body.0.weight"] = torch.zeros(1)
    elif corruption == "config":
        p["config"]["epochs"] = True
    elif corruption == "extra":
        p["observations"] = torch.zeros(1)
    elif corruption == "counter":
        p["optimizer_steps"] = 3
    elif corruption == "bool-counter":
        p["update_count"] = True
    elif corruption == "samples":
        p["samples_seen"] = -1
    elif corruption == "mode":
        p["model_training"] = 1
    elif corruption == "rng":
        p["shuffle_rng_state"] = torch.zeros(5, dtype=torch.uint8)
    elif corruption == "adam-missing":
        p["optimizer_state"]["state"].pop(0)
    elif corruption == "adam-nan":
        p["optimizer_state"]["state"][0]["exp_avg"].fill_(float("nan"))
    elif corruption == "adam-negative":
        p["optimizer_state"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif corruption == "adam-step":
        p["optimizer_state"]["state"][0]["step"].fill_(1)
    elif corruption == "adam-bool-index":
        p["optimizer_state"]["param_groups"][0]["params"][0] = False
    torch.save(p, path)
    original = path.read_bytes()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        load(path)
    assert path.read_bytes() == original and torch.equal(
        rng, torch.random.get_rng_state()
    )


def test_config_mismatch_and_cross_format_rejection(tmp_path: Path) -> None:
    from boardgame_ai.models import TicTacToeActorCritic
    from boardgame_ai.training.ppo import PPOTrainer

    path = tmp_path / "cf.pt"
    save(trainer(), path)
    with pytest.raises(ValueError):
        load(path, expected_config=PPOConfig())
    with pytest.raises(ValueError):
        load_ppo_checkpoint(path)
    other = tmp_path / "ttt.pt"
    save_ppo_checkpoint(PPOTrainer(TicTacToeActorCritic()), other)
    original = other.read_bytes()
    with pytest.raises(ValueError):
        load(other)
    with pytest.raises(ValueError):
        save(trainer(), other)
    assert other.read_bytes() == original
    with pytest.raises(ValueError):
        save_ppo_checkpoint(cast_any(trainer()), other)
    assert other.read_bytes() == original


def cast_any(value: Any) -> Any:
    return value


def test_failed_update_cannot_replace_boundary_and_reload_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = trainer()
    path = tmp_path / "latest.pt"
    save(t, path)
    original = path.read_bytes()
    step = t.optimizer.step
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected optimizer failure")
        return step(*args, **kwargs)

    monkeypatch.setattr(t.optimizer, "step", fail_second)
    with pytest.raises(RuntimeError):
        t.update(batch())
    assert not t.at_update_boundary and t.optimizer_steps == 1
    with pytest.raises(ValueError):
        save(t, path)
    with pytest.raises(ValueError):
        t.update(batch())
    assert path.read_bytes() == original
    restored = load(path)
    assert restored.update(batch()) == trainer().update(batch())


def test_atomic_replace_failure_preserves_previous_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = trainer()
    path = tmp_path / "latest.pt"
    save(t, path)
    original = path.read_bytes()
    t.update(batch())

    def fail(*args: Any) -> None:
        raise OSError("injected replacement failure")

    monkeypatch.setattr(
        "boardgame_ai.training.connect_four_checkpoint.os.replace", fail
    )
    with pytest.raises(OSError):
        save(t, path)
    assert path.read_bytes() == original and list(tmp_path.iterdir()) == [path]


def test_deadline_before_update_preserves_boundary() -> None:
    t = trainer()
    rng = t.generator.get_state().clone()
    with pytest.raises(TimeoutError):
        t.update(batch(), deadline=-1)
    assert t.at_update_boundary and t.optimizer_steps == t.update_count == 0
    assert torch.equal(rng, t.generator.get_state())


def test_mid_update_deadline_requires_reload_and_preserves_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = trainer()
    path = tmp_path / "latest.pt"
    save(t, path)
    original = path.read_bytes()
    calls = 0

    def fake_clock() -> float:
        nonlocal calls
        calls += 1
        return 1.0 if calls >= 5 else 0.0

    monkeypatch.setattr("boardgame_ai.training.ppo.monotonic", fake_clock)
    with pytest.raises(TimeoutError):
        t.update(batch(), deadline=0.5)
    assert t.optimizer_steps == 1 and not t.at_update_boundary
    with pytest.raises(ValueError):
        save(t, path)
    assert path.read_bytes() == original


def test_invalid_save_and_stale_checkpoint_preserve_destination(tmp_path: Path) -> None:
    t = trainer()
    path = tmp_path / "latest.pt"
    save(t, path)
    original = path.read_bytes()
    with torch.no_grad():
        t.model.value_head.bias.fill_(float("nan"))
    with pytest.raises(ValueError):
        save(t, path)
    assert path.read_bytes() == original and list(tmp_path.iterdir()) == [path]
    t = trainer()
    t.update(batch())
    save(t, path)
    newer = path.read_bytes()
    with pytest.raises(ValueError):
        save(trainer(), path)
    assert path.read_bytes() == newer


@pytest.mark.parametrize("kind", ["guildworks", "search"])
def test_other_learner_checkpoint_formats_reject(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "other.pt"
    if kind == "guildworks":
        from boardgame_ai.training.guildworks import RecurrentPPOTrainer
        from boardgame_ai.training.guildworks_run import initial_model, save_checkpoint

        save_checkpoint(RecurrentPPOTrainer(initial_model(9)), path)
    else:
        from boardgame_ai.training.search_checkpoint import save_search_checkpoint
        from boardgame_ai.training.search_learning import SearchTrainer

        save_search_checkpoint(SearchTrainer(), path)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        load(path)
    with pytest.raises(ValueError):
        save(trainer(), path)
    assert path.read_bytes() == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_update_portable_save_and_same_device_resume(tmp_path: Path) -> None:
    t = trainer(device="cuda")
    t.update(batch())
    assert all(
        p.is_cuda and p.grad is not None and p.grad.is_cuda
        for p in t.model.parameters()
    )
    path = tmp_path / "cuda.pt"
    save(t, path)
    payload = torch.load(path, weights_only=True)
    assert all(x.device.type == "cpu" for x in payload["model_state"].values())
    assert all(
        x.device.type == "cpu"
        for m in payload["optimizer_state"]["state"].values()
        for x in m.values()
    )
    rng = torch.cuda.get_rng_state_all()
    restored = load(path, device="cuda")
    cpu = load(path)
    assert all(p.device.type == "cpu" for p in cpu.model.parameters())
    assert all(m["exp_avg"].is_cuda for m in restored.optimizer.state.values())
    assert t.update(batch()) == restored.update(batch())
    assert_nested_equal(t.optimizer.state_dict(), restored.optimizer.state_dict())
    assert all(
        torch.equal(a, b)
        for a, b in zip(rng, torch.cuda.get_rng_state_all(), strict=True)
    )
