"""Tiny disposable device/run checks; never execute the campaign schedule."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from boardgame_ai.training import connect_four_evaluation as evaluation
from boardgame_ai.training import connect_four_run as run
from boardgame_ai.training.connect_four_checkpoint import (
    connect_four_payload,
    load_connect_four_checkpoint,
)
from boardgame_ai.training.connect_four_ppo import ConnectFourPPOTrainer
from tests.test_connect_four_run import files, rows, stable, train
from tests.test_ppo_checkpoint import assert_nested_equal

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture
def settings(tmp_path: Path) -> Path:
    data = json.loads(Path("docs/phase6/connect-four-cuda-100k-v1.json").read_bytes())
    data["protocol_id"] = "disposable-device-run-v1"
    data["training"].update(updates=2, games_per_update=1, wall_time_cap_seconds=120)
    data["evaluation"].update(
        games_per_seat=1, control_games_per_seat=1, checkpoints=["initial", "update-2"]
    )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_campaign_settings_and_frozen_identity(tmp_path: Path) -> None:
    path = Path("docs/phase6/connect-four-cuda-100k-v1.json")
    raw, data = run.read_settings(path)
    assert (
        raw
        == Path(
            "src/boardgame_ai/training/protocols/connect-four-cuda-100k-v1.json"
        ).read_bytes()
    )
    assert data["training"]["updates"] * data["training"]["games_per_update"] == 100_000
    assert data["training"]["wall_time_cap_seconds"] == 21600
    assert data["evaluation"]["wall_time_cap_seconds"] == 600
    original = json.loads(
        Path("docs/phase6/connect-four-functional-v1.json").read_bytes()
    )
    assert {data[s]["seed"] for s in ("training", "evaluation")}.isdisjoint(
        {original[s]["seed"] for s in ("training", "evaluation")}
    )
    data["training"]["updates"] -= 1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="new protocol version"):
        run.read_settings(changed)


@pytest.mark.parametrize("operation", ["check", "train", "resume"])
def test_unavailable_cuda_rejects_without_outputs(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "absent"
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        if operation == "check":
            run.check_connect_four_run(
                output, settings_path=settings, protocol_path="missing"
            )
        else:
            train(output, settings, resume=operation == "resume")
    assert not output.exists()


@pytest.mark.parametrize("device", ["cuda:-1", "cuda:no", "cpu:0", "mps", 0, True])
def test_invalid_device_settings(settings: Path, device: Any) -> None:
    data = json.loads(settings.read_bytes())
    data["training"]["device"] = device
    settings.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        run.read_settings(settings)


@CUDA
def test_cuda_preflight_no_model_or_output(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("preflight constructed model")

    monkeypatch.setattr(run, "initial_connect_four_model", forbidden)
    output = tmp_path / "absent"
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    report = run.check_connect_four_run(
        output,
        settings_path=settings,
        protocol_path="docs/PHASE6_CONNECT_FOUR_PROTOCOL.md",
    )
    assert report["ready"] and report["execution_device"] == "cuda:0"
    assert report["collection_device"] == report["evaluation_device"] == "cpu"
    assert report["device_name"]
    assert not output.exists()
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert_nested_equal(cuda_rng, torch.cuda.get_rng_state_all())


@CUDA
def test_cuda_exact_resume_portability_and_isolated_cpu_evaluation(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole, split = tmp_path / "whole", tmp_path / "split"
    rng = torch.cuda.get_rng_state_all()
    expected = train(whole, settings)
    first = train(split, settings, stop_after=1)
    assert first.device == torch.device("cuda:0")
    report = run.check_connect_four_run(
        split, settings_path=settings, protocol_path="missing", resume=True
    )
    assert (
        report["completed_updates"] == 1
        and report["remaining_training_games_to_target"] == 1
    )
    actual = train(split, settings, resume=True)
    assert_nested_equal(connect_four_payload(expected), connect_four_payload(actual))
    assert [stable(r) for r in rows(whole)] == [stable(r) for r in rows(split)]
    before = files(split)
    train(split, settings, resume=True)
    assert files(split) == before
    manifest = json.loads((split / "training.json").read_bytes())
    assert manifest["runtime"]["device"] == "cuda:0"
    assert manifest["runtime"]["resolved_device"] == "cuda:0"
    cpu = load_connect_four_checkpoint(split / "final.pt", device="cpu")
    assert_nested_equal(connect_four_payload(cpu), connect_four_payload(actual))
    gpu = load_connect_four_checkpoint(split / "initial.pt", device="cuda:0")
    assert gpu.device.type == "cuda"
    assert all(v["exp_avg"].is_cuda for v in actual.optimizer.state.values())
    # Evaluation and portable reads remain usable on CPU-only hosts.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checked = run.check_connect_four_run(
        split, settings_path=settings, protocol_path="missing", evaluate=True
    )
    assert checked["execution_device"] == "cpu"
    report = evaluation.evaluate_connect_four_run(split, settings_path=settings)
    assert report["complete"] and report["device"] == "cpu"
    for name, raw in before.items():
        if name != "budget.json":
            assert (split / name).read_bytes() == raw
    assert_nested_equal(rng, torch.cuda.get_rng_state_all())


@CUDA
@pytest.mark.parametrize("interrupt", [False, True])
def test_cuda_partial_optimizer_preserves_boundary_and_charges_attempt(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    output = tmp_path / "run"
    train(output, settings, stop_after=1)
    before = files(output)
    original = torch.optim.Adam.step

    def fail(self: Any, *args: Any, **kwargs: Any) -> Any:
        original(self, *args, **kwargs)
        if interrupt:
            raise KeyboardInterrupt
        raise TimeoutError("disposable optimizer interruption")

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", fail)
        if interrupt:
            with pytest.raises(KeyboardInterrupt):
                train(output, settings, resume=True)
        else:
            restored = train(output, settings, resume=True)
            assert restored.device.type == "cuda" and restored.update_count == 1
    assert (output / "latest.pt").read_bytes() == before["latest.pt"]
    assert (output / "updates.jsonl").read_bytes() == before["updates.jsonl"]
    ledger = json.loads((output / "budget.json").read_bytes())
    assert ledger["training"] > json.loads(before["budget.json"])["training"]
    status = json.loads((output / "training-status.json").read_bytes())
    assert not status["complete"] and status["updates"] == 1
    actual = train(output, settings, resume=True)
    expected = train(tmp_path / "reference", settings)
    assert_nested_equal(connect_four_payload(actual), connect_four_payload(expected))


@CUDA
def test_cuda_synchronization_overrun_rejects_publication(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    train(output, settings, stop_after=1)
    before = files(output)
    clock = [1000.0]
    monkeypatch.setattr(run, "monotonic", lambda: clock[0])
    original = ConnectFourPPOTrainer.update
    original_sync = run.synchronize_device
    updated = [False]

    def update(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        updated[0] = True
        return result

    def sync(device: torch.device) -> None:
        original_sync(device)
        if updated[0]:
            clock[0] = 1300.0

    monkeypatch.setattr(ConnectFourPPOTrainer, "update", update)
    monkeypatch.setattr(run, "synchronize_device", sync)
    monkeypatch.setattr("boardgame_ai.training.ppo.monotonic", lambda: 1000.0)
    monkeypatch.setattr(
        "boardgame_ai.training.connect_four.monotonic", lambda: clock[0]
    )
    train(output, settings, resume=True)
    assert (output / "latest.pt").read_bytes() == before["latest.pt"]
    status = json.loads((output / "training-status.json").read_bytes())
    assert status["stop_reason"] == "time_cap" and status["cap_overrun_seconds"] > 0
    assert status["discarded_complete_games"] == 1


@CUDA
def test_cuda_metrics_recovery_and_exhausted_cap(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    train(output, settings, stop_after=1)
    log = output / "updates.jsonl"
    torn = log.read_bytes() + b'{"update":2,"ppo":'
    log.write_bytes(torn)
    before = files(output)
    run.check_connect_four_run(
        output, settings_path=settings, protocol_path="missing", resume=True
    )
    assert files(output) == before
    actual = train(output, settings, resume=True)
    assert actual.update_count == 2 and len(rows(output)) == 2
    assert next(output.glob("updates.jsonl.recovery-*.bak")).read_bytes() == torn
    budget = json.loads((output / "budget.json").read_bytes())
    budget["training"] = 120
    (output / "budget.json").write_text(json.dumps(budget), encoding="utf-8")
    before = files(output)
    train(output, settings, resume=True)
    assert files(output) == before


@CUDA
def test_cpu_run_restore_on_cuda_does_not_change_artifacts(
    tmp_path: Path, settings: Path
) -> None:
    data = json.loads(settings.read_bytes())
    data["training"]["device"] = "cpu"
    settings.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "run"
    expected = train(output, settings, stop_after=1)
    before = files(output)
    raw, data = run.read_settings(settings)
    restored, _ = run.load_run(output, run.digest(raw), data, device="cuda:0")
    assert restored.device.type == "cuda"
    assert all(m["exp_avg"].is_cuda for m in restored.optimizer.state.values())
    assert_nested_equal(connect_four_payload(expected), connect_four_payload(restored))
    assert files(output) == before


def test_unavailable_cuda_index_rejects_before_output(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="index is unavailable"):
        train(tmp_path / "absent", settings)
    assert not (tmp_path / "absent").exists()


@CUDA
def test_cuda_minibatch_sync_observes_deadline_before_next_step(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    train(output, settings, stop_after=1)
    before = files(output)
    clock = [1000.0]
    steps = [0]
    step = torch.optim.Adam.step
    sync = torch.cuda.synchronize

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = step(self, *args, **kwargs)
        steps[0] += 1
        return result

    def synchronized(*args: Any, **kwargs: Any) -> None:
        sync(*args, **kwargs)
        if steps[0]:
            clock[0] = 1300.0

    monkeypatch.setattr(torch.optim.Adam, "step", counted)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized)
    monkeypatch.setattr(run, "monotonic", lambda: clock[0])
    monkeypatch.setattr("boardgame_ai.training.ppo.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "boardgame_ai.training.connect_four.monotonic", lambda: clock[0]
    )
    actual = train(output, settings, resume=True)
    assert actual.update_count == 1 and steps[0] == 1
    assert (output / "latest.pt").read_bytes() == before["latest.pt"]


@CUDA
def test_cuda_incomplete_cap_never_resets(tmp_path: Path, settings: Path) -> None:
    output = tmp_path / "run"
    train(output, settings, stop_after=1)
    budget = json.loads((output / "budget.json").read_bytes())
    budget["training"] = 120
    (output / "budget.json").write_text(json.dumps(budget), encoding="utf-8")
    before = files(output)
    report = run.check_connect_four_run(
        output, settings_path=settings, protocol_path="missing", resume=True
    )
    assert not report["ready"] and report["training_seconds_remaining"] == 0
    actual = train(output, settings, resume=True)
    assert actual.update_count == 1 and files(output) == before
