"""Monitoring must not change learning, resume, sources, or RNG streams."""

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from boardgame_ai.training import PPOConfig, TrainingConfig, run_self_play_training
from boardgame_ai.training.monitoring import MonitoringConfig, isolated_rng
from boardgame_ai.training.ppo import PPOTrainer


def equal_trainers(first: PPOTrainer, second: PPOTrainer) -> None:
    for key, tensor in first.model.state_dict().items():
        assert torch.equal(tensor, second.model.state_dict()[key])
    for key, state in first.optimizer.state_dict()["state"].items():
        for name, tensor in state.items():
            assert torch.equal(
                tensor, second.optimizer.state_dict()["state"][key][name]
            )
    assert first.update_count == second.update_count
    assert first.optimizer_steps == second.optimizer_steps
    assert torch.equal(first.generator.get_state(), second.generator.get_state())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_monitoring_resume_and_unmonitored_identical(
    tmp_path: Path, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = TrainingConfig(
        updates=3,
        games_per_update=2,
        seed=12,
        device=device,
        ppo=PPOConfig(epochs=1, minibatch_size=32),
    )
    monitor = MonitoringConfig(
        interval=1, retain=1, games_per_seat=2, self_play_games=1
    )
    plain = run_self_play_training(tmp_path / "plain", config)
    monitored = run_self_play_training(
        tmp_path / "monitored", config, monitoring=monitor
    )
    run_self_play_training(
        tmp_path / "resumed", replace(config, updates=1), monitoring=monitor
    )
    resumed = run_self_play_training(
        tmp_path / "resumed", config, monitoring=monitor, resume=True
    )
    equal_trainers(plain, monitored)
    equal_trainers(plain, resumed)
    assert (tmp_path / "plain" / "updates.jsonl").read_bytes() == (
        tmp_path / "monitored" / "updates.jsonl"
    ).read_bytes()
    assert len(list((tmp_path / "monitored" / "snapshots").glob("*.pt"))) <= 2
    best = json.loads((tmp_path / "monitored" / "best.json").read_text())
    assert (tmp_path / "monitored" / best["checkpoint"]).exists()
    assert len(list((tmp_path / "monitored").glob("development-*.json"))) == 3
    report = json.loads(
        (tmp_path / "monitored" / "development-00000003.json").read_text()
    )
    assert report["samples_seen"] > 0 and report["games_seen"] == 6
    assert report["optimizer_steps"] == monitored.optimizer_steps
    before = {
        p: p.read_bytes() for p in (tmp_path / "monitored").rglob("*") if p.is_file()
    }
    run_self_play_training(
        tmp_path / "monitored", config, monitoring=monitor, resume=True
    )
    assert before == {p: p.read_bytes() for p in before}
    with pytest.raises(ValueError, match="monitoring"):
        run_self_play_training(tmp_path / "monitored", config, resume=True)


def test_rng_restore_even_when_evaluation_fails() -> None:
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.random.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    with pytest.raises(RuntimeError):
        with isolated_rng():
            random.random()
            np.random.random()
            torch.rand(4)
            if cuda:
                torch.rand(4, device="cuda")
            raise RuntimeError("test")
    assert python == random.getstate()
    assert np.array_equal(numpy[1], np.random.get_state()[1])
    assert torch.equal(cpu, torch.random.get_rng_state())
    for expected, actual in zip(
        cuda, torch.cuda.get_rng_state_all() if cuda else [], strict=True
    ):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    "change",
    [
        {"interval": 0},
        {"retain": True},
        {"games_per_seat": -1},
        {"self_play_games": 0},
        {"seed": False},
    ],
)
def test_invalid_monitoring_config(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MonitoringConfig(**change)


def test_best_is_pinned_ties_earliest_and_missing_evaluation_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import boardgame_ai.training.monitoring as monitoring

    calls = []

    def evaluate(checkpoint: Path, config: MonitoringConfig) -> dict[str, Any]:
        calls.append(checkpoint.name)
        return {"score": 0.5 if len(calls) < 3 else 0.1, "metric": "test metric"}

    monkeypatch.setattr(monitoring, "development_evaluation", evaluate)
    config = TrainingConfig(
        updates=3, games_per_update=1, device="cpu", ppo=PPOConfig(epochs=1)
    )
    monitor = MonitoringConfig(
        interval=1, retain=1, games_per_seat=1, self_play_games=1
    )
    run_self_play_training(tmp_path, config, monitoring=monitor)
    best = json.loads((tmp_path / "best.json").read_text())
    assert best["update"] == 1  # update 2 ties; update 3 is worse.
    snapshots = list((tmp_path / "snapshots").glob("*.pt"))
    assert len(snapshots) == 2
    assert (tmp_path / best["checkpoint"]).exists()
    assert not any("00000002" in path.name for path in snapshots)
    pinned = (tmp_path / best["checkpoint"]).read_bytes()
    (tmp_path / "development-00000003.json").unlink()
    run_self_play_training(tmp_path, config, resume=True, monitoring=monitor)
    assert len(calls) == 4
    assert (tmp_path / "development-00000003.json").exists()
    assert (tmp_path / best["checkpoint"]).read_bytes() == pinned


def test_real_evaluation_failure_restores_global_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from boardgame_ai import policy_evaluation
    from boardgame_ai.training.monitoring import development_evaluation

    def fail(*args: Any, **kwargs: Any) -> Any:
        random.random()
        np.random.random()
        torch.rand(3)
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(policy_evaluation, "evaluate_policy_checkpoint", fail)
    before = torch.random.get_rng_state().clone()
    python = random.getstate()
    numpy = np.random.get_state()
    with pytest.raises(RuntimeError):
        development_evaluation(tmp_path / "unused.pt", MonitoringConfig())
    assert torch.equal(before, torch.random.get_rng_state())
    assert python == random.getstate()
    assert np.array_equal(numpy[1], np.random.get_state()[1])


def test_interrupted_snapshot_publication_is_retryable_and_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from boardgame_ai.training import monitoring

    destination = tmp_path / "snapshot.pt"

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("interrupted publication")

    with monkeypatch.context() as context:
        context.setattr("boardgame_ai.training.monitoring.os.link", fail)
        with pytest.raises(OSError):
            monitoring._immutable_snapshot(destination, b"complete checkpoint")
    assert not destination.exists()
    monitoring._immutable_snapshot(destination, b"complete checkpoint")
    monitoring._immutable_snapshot(destination, b"complete checkpoint")
    with pytest.raises(ValueError, match="Immutable"):
        monitoring._immutable_snapshot(destination, b"wrong checkpoint")
    assert destination.read_bytes() == b"complete checkpoint"
