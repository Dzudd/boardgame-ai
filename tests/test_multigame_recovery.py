"""Disposable Windows publication faults and explicit no-training recovery."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from boardgame_ai.training import multigame_io as publication
from boardgame_ai.training import multigame_run as run
from boardgame_ai.training.connect_four_run import _atomic_bytes, _encode
from boardgame_ai.training.monitoring import atomic_json
from boardgame_ai.training.multigame_report import report_suite
from boardgame_ai.training.multigame_run import (
    _load,
    finalize_capped_job,
    jobs,
    run_job,
    validate_run,
)
from tests.test_multigame_run import assert_nested_equal, tiny_settings


def files(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


def fixture_run(
    tmp_path: Path, *, updates: int = 1
) -> tuple[bytes, dict[str, Any], dict[str, Any], Path]:
    settings = tiny_settings()
    raw = json.dumps(settings).encode()
    job = deepcopy(jobs(settings)["samples-119-separate-tic_tac_toe"])
    job["regime"] = "wall"
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Disposable recovery test", encoding="utf-8")
    run_job(tmp_path, raw, settings, job, protocol, stop_after=updates, quiet=True)
    directory = tmp_path / job["id"]
    payload = _load(directory / "latest.pt")
    payload["elapsed"] = job["cap_seconds"] - 0.1
    _atomic_bytes(directory / "latest.pt", _encode(payload))
    ledger = json.loads((directory / "budget.json").read_bytes())
    ledger["training"] = job["cap_seconds"]
    atomic_json(directory / "budget.json", ledger)
    return raw, settings, job, directory


def test_recover_near_cap_without_training_or_budget_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, settings, job, directory = fixture_run(tmp_path)
    original = files(directory)
    trainer_before = _load(directory / "latest.pt")["trainer"]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Recovery must never collect or optimize")

    monkeypatch.setattr(run, "collect_game", forbidden)
    assert finalize_capped_job(tmp_path, raw, settings, job, check_only=True)["ready"]
    assert files(directory) == original
    result = finalize_capped_job(tmp_path, raw, settings, job)
    assert result["complete"] and result["timing_uncertain"]
    assert result["additional_training"] is False
    assert_nested_equal(trainer_before, _load(directory / "latest.pt")["trainer"])
    for name in (
        "budget.json",
        "initial.pt",
        "training.json",
        "settings.json",
        "protocol.md",
        "updates.jsonl",
        "environment.jsonl",
    ):
        assert (directory / name).read_bytes() == original[name]
    receipt = json.loads((directory / "finalization-recovery.json").read_bytes())
    assert (directory / receipt["backup"]).read_bytes() == original["latest.pt"]
    validate_run(directory, raw, settings, job, torch.device("cpu"))
    recovered = files(directory)
    assert finalize_capped_job(tmp_path, raw, settings, job)["preserved"]
    assert files(directory) == recovered
    # The report exposes timing uncertainty even before evaluation exists.
    # Use the original schedule ID with matching test job metadata for this test.
    monkeypatch.setattr(
        "boardgame_ai.training.multigame_report.jobs", lambda _: {job["id"]: job}
    )
    report = report_suite(tmp_path, raw, settings)
    assert report["cost_uncertain_jobs"] == [job["id"]]


@pytest.mark.parametrize("condition", ["samples", "early", "unspent", "zero"])
def test_recovery_rejects_ineligible_boundaries(tmp_path: Path, condition: str) -> None:
    raw, settings, job, directory = fixture_run(
        tmp_path, updates=0 if condition == "zero" else 1
    )
    if condition == "samples":
        job["regime"] = "samples"
        manifest = json.loads((directory / "training.json").read_bytes())
        manifest["job"] = job
        atomic_json(directory / "training.json", manifest)
    if condition == "early":
        payload = _load(directory / "latest.pt")
        payload["elapsed"] = 10.0
        _atomic_bytes(directory / "latest.pt", _encode(payload))
    if condition == "unspent":
        ledger = json.loads((directory / "budget.json").read_bytes())
        ledger["training"] -= 0.05
        atomic_json(directory / "budget.json", ledger)
    original = files(directory)
    with pytest.raises(ValueError, match="Finalization requires"):
        finalize_capped_job(tmp_path, raw, settings, job)
    assert files(directory) == original


def test_recovery_retries_after_failed_checkpoint_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, settings, job, directory = fixture_run(tmp_path)
    original = (directory / "latest.pt").read_bytes()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("Disposable finalization failure")

    with monkeypatch.context() as context:
        context.setattr(run, "_atomic_bytes", fail)
        with pytest.raises(OSError, match="finalization"):
            finalize_capped_job(tmp_path, raw, settings, job)
    assert (directory / "latest.pt").read_bytes() == original
    assert finalize_capped_job(tmp_path, raw, settings, job)["complete"]


@pytest.mark.parametrize("winerror", [5, 32, 33, None])
def test_windows_retry_is_bounded_and_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winerror: int | None
) -> None:
    destination = tmp_path / "budget.json"
    atomic_json(destination, {"training": 60})
    original = destination.read_bytes()
    calls = []
    monkeypatch.setattr(publication, "sleep", lambda _: None)

    def deny(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        assert destination.read_bytes() == original
        error = PermissionError("Disposable lock")
        if winerror is not None:
            monkeypatch.setattr(error, "winerror", winerror, raising=False)
        raise error

    monkeypatch.setattr(publication, "_publish_json", deny)
    with pytest.raises(PermissionError):
        publication.atomic_json(destination, {"training": 61})
    assert len(calls) == (6 if winerror else 1)
    assert destination.read_bytes() == original


def test_windows_transient_lock_eventually_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "budget.json"
    atomic_json(destination, {"training": 60})
    calls = []
    monkeypatch.setattr(publication, "sleep", lambda _: None)

    def transient(path: Path, payload: Any) -> None:
        calls.append(1)
        if len(calls) < 3:
            error = PermissionError("Disposable lock")
            monkeypatch.setattr(error, "winerror", 5, raising=False)
            raise error
        atomic_json(path, payload)

    monkeypatch.setattr(publication, "_publish_json", transient)
    publication.atomic_json(destination, {"training": 61})
    assert len(calls) == 3
    assert json.loads(destination.read_bytes()) == {"training": 61}


def test_exact_source_compatibility_rejects_unknown_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert run._compatible_source(
        "ba082fa1c154b7007c871e83c9a05f42a2c2fee42a757cce434c3347ab5b20f4"
    )
    assert not run._compatible_source("unknown-saved-source")
    monkeypatch.setattr(run, "_source_digest", lambda: "unknown-current-source")
    assert not run._compatible_source(
        "ba082fa1c154b7007c871e83c9a05f42a2c2fee42a757cce434c3347ab5b20f4"
    )
