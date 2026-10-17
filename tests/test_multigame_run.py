"""Tiny disposable operator-layer checks, with explicitly reduced test budgets."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from boardgame_ai.models.multigame import initial_multigame_model
from boardgame_ai.training import multigame as learning_module
from boardgame_ai.training import multigame_evaluation as evaluation_module
from boardgame_ai.training import multigame_run as run_module
from boardgame_ai.training.connect_four_run import _atomic_bytes, _encode, digest
from boardgame_ai.training.multigame import MultiGameTrainer, collect_game
from boardgame_ai.training.multigame_evaluation import evaluate_job
from boardgame_ai.training.multigame_report import (
    interval,
    report_suite,
    summarize_cases,
)
from boardgame_ai.training.multigame_run import (
    _load,
    jobs,
    read_protocol,
    restore_trainer,
    run_job,
    trainer_payload,
    validate_run,
)


def tiny_settings() -> dict[str, Any]:
    _, settings = read_protocol(Path("docs/phase6/multigame-v1.json"))
    settings["protocol_id"] = "disposable-test-only"
    settings["device"] = "cpu"
    settings["training_seeds"] = [119]
    settings["evaluation_roots"] = [120]
    settings["evaluation_games_per_seat"] = 2
    settings["regimes"] = {
        "samples": {
            "pretrain_shared_seconds": 60,
            "pretrain_separate_seconds": 60,
            "pretrain_decisions_per_game": 80,
            "adapt_seconds": 60,
            "adapt_decisions": 80,
        }
    }
    return settings


def assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_frozen_protocol_and_job_dependencies(tmp_path: Path) -> None:
    path = Path("docs/phase6/multigame-v1.json")
    raw, settings = read_protocol(path)
    assert len(jobs(settings)) == 36
    modified = json.loads(raw)
    modified["transfer_game"] = "connect_four"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        read_protocol(changed)
    assert (
        raw
        == Path("src/boardgame_ai/training/protocols/multigame-v1.json").read_bytes()
    )


def test_atomic_resume_exact_and_metrics_tail(tmp_path: Path) -> None:
    settings = tiny_settings()
    raw = json.dumps(settings).encode()
    job = jobs(settings)["samples-119-separate-tic_tac_toe"]
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Disposable test", encoding="utf-8")
    first, second = tmp_path / "first", tmp_path / "second"
    assert run_job(first, raw, settings, job, protocol, check_only=True)["ready"]
    assert not first.exists()
    assert run_job(first, raw, settings, job, protocol, quiet=True)["complete"]
    paused = run_job(second, raw, settings, job, protocol, stop_after=2, quiet=True)
    assert paused["paused"]
    directory = second / job["id"]
    with (directory / "updates.jsonl").open("ab") as stream:
        stream.write(b'{"update":3')
    run_job(second, raw, settings, job, protocol, resume=True, quiet=True)
    assert list(directory.glob("*.bak"))
    a = _load(first / job["id"] / "latest.pt")
    b = _load(directory / "latest.pt")
    assert_nested_equal(a["trainer"], b["trainer"])
    assert a["per_game"] == b["per_game"]
    before = {
        p.name: digest(p.read_bytes()) for p in directory.iterdir() if p.is_file()
    }
    run_job(second, raw, settings, job, protocol, resume=True, quiet=True)
    after = {p.name: digest(p.read_bytes()) for p in directory.iterdir() if p.is_file()}
    assert before == after
    bad = deepcopy(settings)
    bad["device"] = "cpu"
    with pytest.raises(ValueError, match="provenance"):
        validate_run(directory, raw + b" ", bad, job, torch.device("cpu"))


def test_transfer_adapters_and_isolated_evaluation(tmp_path: Path) -> None:
    settings = tiny_settings()
    raw = json.dumps(settings).encode()
    schedule = jobs(settings)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Disposable test", encoding="utf-8")
    source = schedule["samples-119-shared"]
    adapted = schedule["samples-119-adapt-pretrained"]
    fresh = schedule["samples-119-adapt-fresh"]
    with pytest.raises(FileNotFoundError):
        run_job(tmp_path, raw, settings, adapted, protocol, check_only=True)
    assert run_job(tmp_path, raw, settings, source, protocol, quiet=True)["complete"]
    assert run_job(tmp_path, raw, settings, adapted, protocol, quiet=True)["complete"]
    assert run_job(tmp_path, raw, settings, fresh, protocol, quiet=True)["complete"]
    initial_a = _load(tmp_path / adapted["id"] / "initial.pt")
    initial_f = _load(tmp_path / fresh["id"] / "initial.pt")
    for key in initial_a["model"]:
        if "breakthrough" in key:
            assert torch.equal(initial_a["model"][key], initial_f["model"][key])
    assert not initial_a["optimizer"]["state"]
    assert not initial_f["optimizer"]["state"]
    directory = tmp_path / adapted["id"]
    before = digest((directory / "latest.pt").read_bytes())
    state = torch.random.get_rng_state().clone()
    evaluated = evaluate_job(tmp_path, raw, settings, adapted)
    assert evaluated["complete"]
    assert before == digest((directory / "latest.pt").read_bytes())
    assert torch.equal(state, torch.random.get_rng_state())
    with pytest.raises(ValueError, match="already exists"):
        evaluate_job(tmp_path, raw, settings, adapted)
    report = report_suite(tmp_path, raw, settings)
    assert report["evaluated_jobs"] == 1
    assert report["acceptance"] == "awaiting operator results"
    assert (
        report["comparisons"]["samples"]["tic_tac_toe"]["after_minus_before_adaptation"]
        is not None
    )


def test_corrupt_trainer_and_failed_save_boundary() -> None:
    trainer = MultiGameTrainer(initial_multigame_model(1))
    payload = trainer_payload(trainer)
    bad = deepcopy(payload)
    bad["model"]["recurrent.weight_ih_l0"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="tensor"):
        restore_trainer(bad, torch.device("cpu"))
    rollout = collect_game(trainer.model, "tic_tac_toe", root=3, index=0)
    with pytest.raises(TimeoutError):
        trainer.update((rollout,), deadline=0)
    with pytest.raises(ValueError, match="incomplete"):
        trainer_payload(trainer)
    restored = restore_trainer(payload, torch.device("cpu"))
    assert _encode(trainer_payload(restored)) == _encode(payload)


def test_failed_update_publication_preserves_checkpoint_and_charges_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = tiny_settings()
    raw = json.dumps(settings).encode()
    job = jobs(settings)["samples-119-separate-tic_tac_toe"]
    protocol = tmp_path / "test.md"
    protocol.write_text("Disposable", encoding="utf-8")
    run_job(tmp_path, raw, settings, job, protocol, stop_after=0, quiet=True)
    directory = tmp_path / job["id"]
    before = (directory / "latest.pt").read_bytes()
    original = _atomic_bytes

    def fail_latest(path: Path, data: bytes) -> None:
        if path.name == "latest.pt":
            raise OSError("Disposable publication fault")
        original(path, data)

    with monkeypatch.context() as context:
        context.setattr(run_module, "_atomic_bytes", fail_latest)
        with pytest.raises(OSError, match="publication"):
            run_job(tmp_path, raw, settings, job, protocol, resume=True, quiet=True)
    assert (directory / "latest.pt").read_bytes() == before
    assert json.loads((directory / "budget.json").read_bytes())["training"] > 0
    assert (directory / "environment.jsonl").stat().st_size > 0
    assert run_job(tmp_path, raw, settings, job, protocol, resume=True, quiet=True)[
        "complete"
    ]


def test_exhausted_hard_kill_reservation_is_read_only(tmp_path: Path) -> None:
    settings = tiny_settings()
    raw = json.dumps(settings).encode()
    job = jobs(settings)["samples-119-separate-tic_tac_toe"]
    protocol = tmp_path / "test.md"
    protocol.write_text("Disposable", encoding="utf-8")
    run_job(tmp_path, raw, settings, job, protocol, stop_after=0, quiet=True)
    directory = tmp_path / job["id"]
    ledger = json.loads((directory / "budget.json").read_bytes())
    ledger["training"] = job["cap_seconds"]
    (directory / "budget.json").write_text(json.dumps(ledger), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    result = run_job(
        tmp_path, raw, settings, job, protocol, resume=True, check_only=True
    )
    assert not result["ready"]
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_reporting_numeric_seats_and_training_seed_interval() -> None:
    summary = summarize_cases(
        [
            {"seat": 0, "score": 1.0, "raw_return": 1},
            {"seat": 0, "score": 0.5, "raw_return": 0},
            {"seat": 1, "score": 0.0, "raw_return": -1},
            {"seat": 1, "score": 0.5, "raw_return": 0},
        ]
    )
    assert summary["balanced_score"] == 0.5
    assert summary["seats"][0]["score"] == 0.75
    effect = interval([0.0, 0.1, 0.2], 4.303)
    assert effect["mean"] == pytest.approx(0.1)
    assert effect["approximate_95_seed_interval"] == pytest.approx(
        [0.1 - 4.303 * 0.1 / 3**0.5, 0.1 + 4.303 * 0.1 / 3**0.5],
    )
    assert interval([0.2], 4.303)["approximate_95_seed_interval"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_multigame_cuda_portable_resume() -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer = MultiGameTrainer(initial_multigame_model(71).cuda())
        rollout = collect_game(trainer.model, "tic_tac_toe", root=72, index=0)
        trainer.update((rollout,))
        payload = trainer_payload(trainer)
        restored = restore_trainer(payload, torch.device("cuda:0"))
        portable = restore_trainer(payload, torch.device("cpu"))
        assert all(p.device.type == "cpu" for p in portable.model.parameters())
        rollout = collect_game(trainer.model, "tic_tac_toe", root=72, index=1)
        trainer.update((rollout,))
        restored.update((rollout,))
        assert_nested_equal(trainer_payload(trainer), trainer_payload(restored))
    finally:
        torch.set_num_threads(previous)


def test_interrupt_preserves_boundary_and_finishes_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = tiny_settings()
    raw = json.dumps(settings).encode()
    job = jobs(settings)["samples-119-separate-tic_tac_toe"]
    protocol = tmp_path / "test.md"
    protocol.write_text("Disposable", encoding="utf-8")
    run_job(tmp_path, raw, settings, job, protocol, stop_after=0, quiet=True)
    directory = tmp_path / job["id"]
    before = (directory / "latest.pt").read_bytes()
    threads = torch.get_num_threads()

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(run_module, "collect_game", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_job(tmp_path, raw, settings, job, protocol, resume=True)
    assert (directory / "latest.pt").read_bytes() == before
    assert torch.get_num_threads() == threads
    assert json.loads((directory / "status.json").read_bytes())["saved_updates"] == 0
    assert "Stopped: KeyboardInterrupt" in capsys.readouterr().err


def test_evaluation_write_failure_restores_threads_and_charges_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = tiny_settings()
    raw = json.dumps(settings).encode()
    job = jobs(settings)["samples-119-separate-tic_tac_toe"]
    protocol = tmp_path / "test.md"
    protocol.write_text("Disposable", encoding="utf-8")
    run_job(tmp_path, raw, settings, job, protocol, quiet=True)
    directory = tmp_path / job["id"]
    before = (directory / "latest.pt").read_bytes()
    threads = torch.get_num_threads()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("Disposable report publication fault")

    monkeypatch.setattr(evaluation_module, "_immutable_snapshot", fail)
    with pytest.raises(OSError, match="publication"):
        evaluate_job(tmp_path, raw, settings, job)
    assert torch.get_num_threads() == threads
    assert (directory / "latest.pt").read_bytes() == before
    assert json.loads((directory / "budget.json").read_bytes())["evaluation"] > 0


def test_wall_deadline_reloads_durable_boundary_and_reports_unsaved_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = tiny_settings()
    raw = json.dumps(settings).encode()
    job = deepcopy(jobs(settings)["samples-119-separate-tic_tac_toe"])
    job["regime"], job["cap_seconds"] = "wall", 10
    protocol = tmp_path / "test.md"
    protocol.write_text("Disposable", encoding="utf-8")
    clock = [1000.0]
    monkeypatch.setattr(run_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(learning_module, "monotonic", lambda: clock[0])
    run_job(tmp_path, raw, settings, job, protocol, stop_after=0, quiet=True)

    def late_game(*args: Any, **kwargs: Any) -> Any:
        rollout = collect_game(*args, **kwargs)
        clock[0] += 11
        return rollout

    monkeypatch.setattr(run_module, "collect_game", late_game)
    result = run_job(tmp_path, raw, settings, job, protocol, resume=True, quiet=True)
    assert result["complete"]
    assert result["saved_updates"] == 0
    assert result["overrun_seconds"] == 1
    assert result["per_game"]["tic_tac_toe"]["decisions"] == 0
    assert (tmp_path / job["id"] / "environment.jsonl").stat().st_size > 0
