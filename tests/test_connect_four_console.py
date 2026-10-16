"""Read-only console readiness and exit codes; these tests never train/play."""

import json
import platform
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from boardgame_ai.models.connect_four import initial_connect_four_model
from boardgame_ai.training import connect_four_run as run
from boardgame_ai.training.connect_four_checkpoint import connect_four_payload
from boardgame_ai.training.connect_four_ppo import ConnectFourPPOTrainer
from boardgame_ai.training.ppo import PPOConfig

SETTINGS = Path("docs/phase6/connect-four-functional-v1.json")
PROTOCOL = Path("docs/PHASE6_CONNECT_FOUR_PROTOCOL.md")


def check(directory: Path, **kwargs: Any) -> dict[str, Any]:
    return run.check_connect_four_run(
        directory, settings_path=SETTINGS, protocol_path=PROTOCOL, **kwargs
    )


def snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }


def cli(monkeypatch: pytest.MonkeyPatch, directory: Path, *extra: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "connect-four-run",
            str(directory),
            "--settings",
            str(SETTINGS),
            "--protocol",
            str(PROTOCOL),
            *extra,
        ],
    )
    run.main()


@pytest.fixture
def saved_boundary(tmp_path: Path) -> Path:
    """An initialized boundary only: zero games and zero optimizer steps."""
    directory = tmp_path / "existing"
    directory.mkdir()
    raw, settings = run.read_settings(SETTINGS)
    identity = run.digest(raw)
    trainer = ConnectFourPPOTrainer(
        initial_connect_four_model(0), PPOConfig(**settings["training"]["ppo"])
    )
    initial = run._encode(connect_four_payload(trainer))
    (directory / "initial.pt").write_bytes(initial)
    (directory / "settings.json").write_bytes(raw)
    (directory / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    (directory / "training.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "settings_sha256": identity,
                "protocol_sha256": run.digest(PROTOCOL.read_bytes()),
                "initial_sha256": run.digest(initial),
                "code": run._code_identity(),
                "runtime": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "device": "cpu",
                    "threads": 1,
                },
                "parameter_count": 20872,
            }
        ),
        encoding="utf-8",
    )
    (directory / "budget.json").write_text(
        json.dumps({"settings_sha256": identity, "training": 3.0, "evaluation": 0.0}),
        encoding="utf-8",
    )
    run._save_boundary(directory, trainer, identity, 0, 1.0)
    return directory


def test_canonical_check_never_constructs_model_or_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Read-only check must not start a trainer or game")

    for name in (
        "initial_connect_four_model",
        "ConnectFourPPOTrainer",
        "collect_connect_four_self_play",
        "run_connect_four",
        "_save_boundary",
    ):
        monkeypatch.setattr(run, name, forbidden)
    python, numpy = random.getstate(), np.random.get_state()
    cpu, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    before = snapshot(tmp_path)
    report = check(tmp_path / "nested" / "new-output")
    assert report["check_only"] and report["ready"]
    assert report["scheduled_training_games"] == 32
    assert report["scheduled_evaluation_games"] == 192
    assert (
        report["training_seconds_remaining"]
        == report["evaluation_seconds_remaining"]
        == 120
    )
    assert report["completed_updates"] == report["next_game_index"] == 0
    assert report["device"] == "cpu" and report["threads"] == 1
    assert report["settings_sha256"] == run.digest(SETTINGS.read_bytes())
    assert snapshot(tmp_path) == before
    assert random.getstate() == python
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), cpu)
    assert torch.get_num_threads() == threads


def test_resume_check_does_not_repair_tail_or_charge_time(saved_boundary: Path) -> None:
    (saved_boundary / "updates.jsonl").write_bytes(b'{"update":')
    before = snapshot(saved_boundary)
    cpu = torch.get_rng_state().clone()
    report = check(saved_boundary, resume=True)
    assert report["ready"] and report["training_seconds_remaining"] == 117
    assert report["remaining_training_games_to_target"] == 32
    assert snapshot(saved_boundary) == before
    assert torch.equal(torch.get_rng_state(), cpu)


def test_resume_check_uses_saved_prose(saved_boundary: Path) -> None:
    report = run.check_connect_four_run(
        saved_boundary,
        settings_path=SETTINGS,
        protocol_path=saved_boundary / "missing.md",
        resume=True,
    )
    assert report["protocol_sha256"] == run.digest(PROTOCOL.read_bytes())


def test_resume_check_spent_cap_no_writes(saved_boundary: Path) -> None:
    ledger = json.loads((saved_boundary / "budget.json").read_bytes())
    ledger["training"] = 120
    (saved_boundary / "budget.json").write_text(json.dumps(ledger), encoding="utf-8")
    before = snapshot(saved_boundary)
    report = check(saved_boundary, resume=True)
    assert not report["ready"] and report["reason"] == "cumulative_time_cap_exhausted"
    assert snapshot(saved_boundary) == before


def test_resume_check_corrupt_metrics_no_writes(saved_boundary: Path) -> None:
    (saved_boundary / "updates.jsonl").write_bytes(b"garbage\n")
    before = snapshot(saved_boundary)
    with pytest.raises(ValueError):
        check(saved_boundary, resume=True)
    assert snapshot(saved_boundary) == before


@pytest.mark.parametrize(
    "kind", ["nonempty", "file", "parent-file", "missing-protocol"]
)
def test_new_check_rejects_before_creating_output(tmp_path: Path, kind: str) -> None:
    directory, protocol = tmp_path / "run", PROTOCOL
    if kind == "nonempty":
        directory.mkdir()
        (directory / "preserve").write_bytes(b"keep")
    elif kind in ("file", "parent-file"):
        directory.write_bytes(b"keep")
        if kind == "parent-file":
            directory = directory / "child"
    else:
        protocol = tmp_path / "missing.md"
    before = snapshot(tmp_path)
    with pytest.raises((OSError, ValueError)):
        run.check_connect_four_run(
            directory, settings_path=SETTINGS, protocol_path=protocol
        )
    assert snapshot(tmp_path) == before


def test_cli_check_only_cannot_dispatch_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Never dispatch training in check-only mode")

    monkeypatch.setattr(run, "run_connect_four", forbidden)
    cli(monkeypatch, tmp_path / "new", "--check-only")
    report = json.loads(capsys.readouterr().out)
    assert report["check_only"] and report["ready"]
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize(
    "extra", [("--resume", "--evaluate"), ("--evaluate", "--stop-after", "2")]
)
def test_cli_conflicting_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...]
) -> None:
    with pytest.raises(SystemExit) as error:
        cli(monkeypatch, tmp_path / "new", "--check-only", *extra)
    assert error.value.code == 2
    assert not (tmp_path / "new").exists()


def test_cli_spent_cap_exit_one(
    saved_boundary: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = json.loads((saved_boundary / "budget.json").read_bytes())
    ledger["training"] = 120
    (saved_boundary / "budget.json").write_text(json.dumps(ledger), encoding="utf-8")
    before = snapshot(saved_boundary)
    with pytest.raises(SystemExit) as error:
        cli(monkeypatch, saved_boundary, "--check-only", "--resume")
    assert error.value.code == 1
    assert not json.loads(capsys.readouterr().out)["ready"]
    assert snapshot(saved_boundary) == before


def test_cli_incomplete_exit_one_without_training(
    saved_boundary: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        run,
        "run_connect_four",
        lambda *args, **kwargs: SimpleNamespace(update_count=0, samples_seen=0),
    )
    with pytest.raises(SystemExit) as error:
        cli(monkeypatch, saved_boundary, "--resume")
    assert error.value.code == 1
    assert not json.loads(capsys.readouterr().out)["target_reached"]


def test_cli_pause_success_without_training(
    saved_boundary: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        run,
        "run_connect_four",
        lambda *args, **kwargs: SimpleNamespace(update_count=2, samples_seen=100),
    )
    cli(monkeypatch, saved_boundary, "--resume", "--stop-after", "2")
    report = json.loads(capsys.readouterr().out)
    assert report["target_reached"] and not report["complete"]


@pytest.mark.parametrize("complete,exit_code", [(False, 1), (True, 0)])
def test_cli_evaluation_exit_codes_without_games(
    saved_boundary: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    complete: bool,
    exit_code: int,
) -> None:
    from boardgame_ai.training import connect_four_evaluation

    monkeypatch.setattr(
        connect_four_evaluation,
        "evaluate_connect_four_run",
        lambda *args, **kwargs: {"complete": complete},
    )
    if exit_code:
        with pytest.raises(SystemExit) as error:
            cli(monkeypatch, saved_boundary, "--evaluate")
        assert error.value.code == exit_code
    else:
        cli(monkeypatch, saved_boundary, "--evaluate")
    assert json.loads(capsys.readouterr().out)["complete"] == complete


def test_interrupt_reports_saved_boundary_without_training(
    saved_boundary: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(run, "collect_connect_four_self_play", interrupt)
    original = (saved_boundary / "latest.pt").read_bytes()
    with pytest.raises(SystemExit) as error:
        cli(monkeypatch, saved_boundary, "--resume")
    assert error.value.code == 130
    assert "Interrupted" in capsys.readouterr().err
    assert (saved_boundary / "latest.pt").read_bytes() == original
    status = json.loads((saved_boundary / "training-status.json").read_bytes())
    assert status["stop_reason"] == "KeyboardInterrupt"
    assert status["updates"] == 0 and status["decisions"] == 0


def test_evaluation_preflight_rejects_unfinished_run_without_writes(
    saved_boundary: Path,
) -> None:
    before = snapshot(saved_boundary)
    with pytest.raises(ValueError, match="final boundary"):
        check(saved_boundary, evaluate=True)
    assert snapshot(saved_boundary) == before


@pytest.mark.parametrize(
    "state", ["ready", "missing-final", "existing-report", "spent"]
)
def test_evaluation_preflight_schedule_and_readiness(
    saved_boundary: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    # Isolate readiness branches; validate_run's real snapshot validation has
    # separate corruption coverage. No games or optimization are involved here.
    ledger = {"training": 5.0, "evaluation": 120.0 if state == "spent" else 10.0}

    def validated(*args: Any, **kwargs: Any) -> Any:
        return (
            SimpleNamespace(update_count=4, samples_seen=224, optimizer_steps=14),
            {},
            ledger,
        )

    monkeypatch.setattr(run, "validate_run", validated)
    if state != "missing-final":
        (saved_boundary / "final.pt").write_bytes(
            b"not loaded in this readiness-only unit test"
        )
    if state == "existing-report":
        (saved_boundary / "evaluation.json").write_text("{}", encoding="utf-8")
    before = snapshot(saved_boundary)
    if state in ("missing-final", "existing-report"):
        with pytest.raises(ValueError):
            check(saved_boundary, evaluate=True)
    else:
        report = check(saved_boundary, evaluate=True)
        assert report["ready"] == (state == "ready")
        assert report["operation"] == "evaluation"
        assert report["scheduled_evaluation_games"] == 192
        assert report["remaining_training_games_to_target"] == 0
        assert report["evaluation_seconds_remaining"] == (
            110 if state == "ready" else 0
        )
    assert snapshot(saved_boundary) == before


@pytest.mark.parametrize("check_only", [True, False])
@pytest.mark.parametrize("failure", ["missing-git", "not-a-checkout"])
def test_git_preflight_failure_creates_no_run_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    check_only: bool,
    failure: str,
) -> None:
    import subprocess

    def git_failure(*args: Any, **kwargs: Any) -> Any:
        if failure == "missing-git":
            raise FileNotFoundError(2, "git executable missing")
        return subprocess.CompletedProcess(
            args=[], returncode=128, stdout=b"", stderr=b"not a git repository"
        )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Git must be checked before any model/game work")

    monkeypatch.setattr(subprocess, "run", git_failure)
    monkeypatch.setattr(run, "initial_connect_four_model", forbidden)
    monkeypatch.setattr(run, "collect_connect_four_self_play", forbidden)
    directory = tmp_path / "new"
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli(monkeypatch, directory, *(["--check-only"] if check_only else []))
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert ("PATH" if failure == "missing-git" else "repository root") in message
    assert not directory.exists()
    assert snapshot(tmp_path) == before
