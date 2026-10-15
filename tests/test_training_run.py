"""Tiny orchestration checks; no full training or strength experiments."""

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from boardgame_ai.cli import main
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.training import (
    PPOConfig,
    PPOTrainer,
    TrainingConfig,
    collect_tic_tac_toe_rollout,
    load_ppo_checkpoint,
    prepare_ppo_batch,
    resolve_training_device,
    run_self_play_training,
    save_ppo_checkpoint,
)


def small_config(updates: int = 1) -> TrainingConfig:
    return TrainingConfig(
        updates=updates,
        games_per_update=2,
        seed=123,
        device="cpu",
        ppo=PPOConfig(epochs=1, minibatch_size=32),
    )


def test_cpu_run_writes_checkpoint_manifest_and_diagnostics(tmp_path: Path) -> None:
    records: list[dict[str, Any]] = []
    rng = torch.random.get_rng_state().clone()
    trainer = run_self_play_training(
        tmp_path / "run", small_config(), on_update=records.append
    )
    assert trainer.update_count == 1 and trainer.device.type == "cpu"
    assert torch.equal(torch.random.get_rng_state(), rng)
    directory = tmp_path / "run"
    manifest = json.loads((directory / "training.json").read_text())
    assert manifest["mode"] == "self-play" and manifest["games_per_update"] == 2
    lines = [
        json.loads(line)
        for line in (directory / "updates.jsonl").read_text().splitlines()
    ]
    assert lines == records and records[0]["update"] == 1
    assert sum(records[0][key] for key in ("x_wins", "o_wins", "draws")) == 2
    assert 10 <= records[0]["ppo"]["num_samples"] <= 18
    restored = load_ppo_checkpoint(directory / "latest.pt")
    assert all(
        torch.equal(t, restored.model.state_dict()[k])
        for k, t in trainer.model.state_dict().items()
    )


def test_resume_matches_uninterrupted_run_and_future_seeds(tmp_path: Path) -> None:
    complete = run_self_play_training(tmp_path / "complete", small_config(2))
    run_self_play_training(tmp_path / "resume", small_config(1))
    resumed = run_self_play_training(tmp_path / "resume", small_config(2), resume=True)
    assert complete.update_count == resumed.update_count == 2
    assert complete.optimizer_steps == resumed.optimizer_steps
    assert all(
        torch.equal(t, resumed.model.state_dict()[k])
        for k, t in complete.model.state_dict().items()
    )
    assert torch.equal(complete.generator.get_state(), resumed.generator.get_state())
    assert (tmp_path / "complete" / "updates.jsonl").read_bytes() == (
        tmp_path / "resume" / "updates.jsonl"
    ).read_bytes()
    for index, state in complete.optimizer.state_dict()["state"].items():
        for name, tensor in state.items():
            assert torch.equal(
                tensor, resumed.optimizer.state_dict()["state"][index][name]
            )


def test_new_run_rejects_existing_output_without_overwriting(tmp_path: Path) -> None:
    existing = tmp_path / "keep.txt"
    existing.write_text("keep")
    with pytest.raises(ValueError, match="absent or empty"):
        run_self_play_training(tmp_path, small_config())
    assert existing.read_text() == "keep" and len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 999},
        {"games_per_update": 3},
        {"ppo": PPOConfig(epochs=2)},
        {"updates": 0},
    ],
)
def test_resume_rejects_schedule_changes(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    run_self_play_training(tmp_path, small_config())
    checkpoint = (tmp_path / "latest.pt").read_bytes()
    with pytest.raises(ValueError):
        run_self_play_training(tmp_path, replace(small_config(), **change), resume=True)
    assert (tmp_path / "latest.pt").read_bytes() == checkpoint


def test_zero_budget_initializes_only_and_equal_budget_resume_is_noop(
    tmp_path: Path,
) -> None:
    trainer = run_self_play_training(tmp_path, small_config(0))
    assert trainer.update_count == 0 and not (tmp_path / "updates.jsonl").exists()
    restored = run_self_play_training(tmp_path, small_config(0), resume=True)
    assert restored.update_count == 0


def test_cuda_unavailable_fails_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "cuda"
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        run_self_play_training(output, replace(small_config(), device="cuda"))
    assert not output.exists()


@pytest.mark.parametrize("device", ["mps", "meta", "cpu:1", "cuda:-1", "garbage"])
def test_invalid_device_rejects(device: str) -> None:
    with pytest.raises(ValueError):
        resolve_training_device(device)


def test_cuda_index_rejection_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="index"):
        resolve_training_device("cuda:1")


@pytest.mark.parametrize(
    "change",
    [
        {"updates": -1},
        {"updates": True},
        {"games_per_update": 0},
        {"seed": False},
    ],
)
def test_invalid_training_config(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(small_config(), **change)


def test_cli_device_check_has_no_output_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "unused"
    main(["train", "--device", "cpu", "--check-device", "--output", str(output)])
    assert not output.exists()
    assert "no training" in capsys.readouterr().out


def test_cli_default_cuda_no_silent_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as error:
        main(["train", "--check-device"])
    assert error.value.code == 2
    assert "CUDA requested but unavailable" in capsys.readouterr().err


def test_cli_help_and_tiny_explicit_cpu_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["train", "--help"])
    assert error.value.code == 0
    assert "--resume" in capsys.readouterr().out
    main(
        [
            "train",
            "--device",
            "cpu",
            "--updates",
            "1",
            "--games-per-update",
            "1",
            "--epochs",
            "1",
            "--output",
            str(tmp_path),
        ]
    )
    assert "Update 1/1" in capsys.readouterr().out
    assert load_ppo_checkpoint(tmp_path / "latest.pt").update_count == 1


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-enabled PyTorch/GPU unavailable"
)
def test_cuda_update_and_portable_checkpoint_continuation(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        model = TicTacToeActorCritic()
    rollout = collect_tic_tac_toe_rollout(
        TicTacToeEngine(), (model, model), max_steps=9, sampling_seeds=(1, 2)
    )
    config = PPOConfig(epochs=1, minibatch_size=32)
    batch = prepare_ppo_batch(rollout.players, config)
    assert all(getattr(batch, f.name).device.type == "cpu" for f in fields(batch))
    trainer = PPOTrainer(model.cuda(), config, shuffle_seed=1)
    gpu_batch = batch.to(trainer.device)
    assert all(
        getattr(gpu_batch, f.name).device == trainer.device for f in fields(gpu_batch)
    )
    devices: list[torch.device] = []
    handle = trainer.model.body.register_forward_pre_hook(
        lambda module, args: devices.append(args[0].device)
    )
    before = model.policy_head.weight.detach().clone()
    trainer.update(batch)
    handle.remove()
    assert all(device.type == "cuda" for device in devices) and devices
    assert not torch.equal(model.policy_head.weight, before)
    assert all(p.device == trainer.device for p in model.parameters())
    assert all(
        p.grad is not None
        and p.grad.device == trainer.device
        and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )
    assert all(
        trainer.optimizer.state[p][name].device == trainer.device
        for p in model.parameters()
        for name in ("exp_avg", "exp_avg_sq")
    )
    save_ppo_checkpoint(trainer, tmp_path / "cuda.pt")
    payload = torch.load(tmp_path / "cuda.pt", weights_only=True)
    assert all(t.device.type == "cpu" for t in payload["model_state"].values())
    assert all(
        t.device.type == "cpu"
        for state in payload["optimizer_state"]["state"].values()
        for t in state.values()
    )
    restored = load_ppo_checkpoint(tmp_path / "cuda.pt", device="cuda")
    assert all(
        restored.optimizer.state[p][name].device == restored.device
        for p in restored.model.parameters()
        for name in ("exp_avg", "exp_avg_sq")
    )
    original_report = trainer.update(batch)
    restored_report = restored.update(batch)
    assert original_report == restored_report
    assert all(
        torch.equal(t, restored.model.state_dict()[k])
        for k, t in trainer.model.state_dict().items()
    )
    cpu_restored = load_ppo_checkpoint(tmp_path / "cuda.pt")
    assert cpu_restored.model.policy_head.weight.device.type == "cpu"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-enabled PyTorch/GPU unavailable"
)
def test_cuda_runner_selects_gpu_and_saves_portable_output(tmp_path: Path) -> None:
    config = TrainingConfig(
        updates=1,
        games_per_update=1,
        device="cuda",
        ppo=PPOConfig(epochs=1, minibatch_size=32),
    )
    trainer = run_self_play_training(tmp_path, config)
    assert trainer.device.type == "cuda"
    assert all(p.device.type == "cuda" for p in trainer.model.parameters())
    record = json.loads((tmp_path / "updates.jsonl").read_text())
    assert record["device"] == "cuda:0"
    assert load_ppo_checkpoint(tmp_path / "latest.pt").device.type == "cpu"
    resumed = run_self_play_training(tmp_path, replace(config, updates=2), resume=True)
    complete = run_self_play_training(tmp_path / "complete", replace(config, updates=2))
    assert resumed.update_count == complete.update_count == 2
    assert resumed.optimizer_steps == complete.optimizer_steps
    assert all(
        torch.equal(t, complete.model.state_dict()[k])
        for k, t in resumed.model.state_dict().items()
    )
    assert torch.equal(resumed.generator.get_state(), complete.generator.get_state())
    for index, state in complete.optimizer.state_dict()["state"].items():
        for name, tensor in state.items():
            assert torch.equal(
                tensor, resumed.optimizer.state_dict()["state"][index][name]
            )
