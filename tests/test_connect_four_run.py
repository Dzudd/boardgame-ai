"""Disposable run schedules, provenance, recovery, caps and isolated evaluation."""

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from boardgame_ai.training import connect_four_evaluation as evaluation
from boardgame_ai.training import connect_four_run as run
from boardgame_ai.training.connect_four import (
    collect_connect_four_self_play,
    derive_connect_four_seed,
)
from boardgame_ai.training.connect_four_checkpoint import connect_four_payload
from tests.test_ppo_checkpoint import assert_nested_equal


@pytest.fixture
def settings(tmp_path: Path) -> Path:
    data = json.loads(Path("docs/phase6/connect-four-functional-v1.json").read_bytes())
    data["protocol_id"] = "disposable-run-test-v1"
    data["training"]["games_per_update"] = 1
    data["evaluation"]["games_per_seat"] = 2
    data["evaluation"]["control_games_per_seat"] = 2
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data))
    return path


def train(output: Path, settings: Path, **kwargs: Any) -> Any:
    return run.run_connect_four(
        output,
        settings_path=settings,
        protocol_path="docs/PHASE6_CONNECT_FOUR_PROTOCOL.md",
        **kwargs,
    )


def files(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


def rows(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (directory / "updates.jsonl").read_text().splitlines()
    ]


def stable(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.endswith("_seconds")}


def test_exact_two_plus_two_resume_and_noop(tmp_path: Path, settings: Path) -> None:
    whole, split = tmp_path / "whole", tmp_path / "split"
    expected = train(whole, settings)
    first = train(split, settings, stop_after=2)
    assert first.update_count == 2
    before_elapsed = json.loads((split / "budget.json").read_bytes())["training"]
    actual = train(split, settings, resume=True)
    assert_nested_equal(connect_four_payload(expected), connect_four_payload(actual))
    assert [stable(r) for r in rows(whole)] == [stable(r) for r in rows(split)]
    assert json.loads((split / "budget.json").read_bytes())["training"] > before_elapsed
    before = files(split)
    train(split, settings, resume=True)
    assert files(split) == before
    assert actual.samples_seen == sum(r["ppo"]["num_samples"] for r in rows(split))
    assert any(
        not torch.equal(
            actual.model.state_dict()[k],
            torch.load(split / "initial.pt", weights_only=True)["model_state"][k],
        )
        for k in actual.model.state_dict()
    )


def test_protocol_copy_not_current_prose_is_resume_provenance(
    tmp_path: Path, settings: Path
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=1)
    run.run_connect_four(
        directory,
        settings_path=settings,
        protocol_path=tmp_path / "missing.md",
        resume=True,
        stop_after=2,
    )
    assert (directory / "protocol.md").read_bytes() == Path(
        "docs/PHASE6_CONNECT_FOUR_PROTOCOL.md"
    ).read_bytes()


@pytest.mark.parametrize("name", ["settings.json", "protocol.md", "initial.pt"])
def test_corrupt_provenance_rejects_without_writes(
    tmp_path: Path, settings: Path, name: str
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=1)
    with (directory / name).open("ab") as stream:
        stream.write(b"corrupt")
    before = files(directory)
    with pytest.raises(ValueError):
        train(directory, settings, resume=True)
    assert files(directory) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("games", 999),
        ("elapsed_seconds", -1),
        ("settings_sha256", "wrong"),
        ("schema_version", True),
    ],
)
def test_corrupt_run_checkpoint_rejects_without_writes(
    tmp_path: Path, settings: Path, field: str, value: Any
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=1)
    payload = torch.load(directory / "latest.pt", weights_only=True)
    payload[field] = value
    torch.save(payload, directory / "latest.pt")
    before = files(directory)
    with pytest.raises(ValueError):
        train(directory, settings, resume=True)
    assert files(directory) == before


def test_trainer_only_checkpoint_rejected(tmp_path: Path, settings: Path) -> None:
    directory = tmp_path / "run"
    trainer = train(directory, settings, stop_after=1)
    torch.save(connect_four_payload(trainer), directory / "latest.pt")
    before = files(directory)
    with pytest.raises(ValueError):
        train(directory, settings, resume=True)
    assert files(directory) == before


def test_metrics_torn_tail_backup_and_gap_never_invented(
    tmp_path: Path, settings: Path
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=2)
    first = (directory / "updates.jsonl").read_bytes().splitlines(keepends=True)[0]
    torn = first + b'{"update":2,"ppo":'
    (directory / "updates.jsonl").write_bytes(torn)
    train(directory, settings, resume=True)
    assert [r["update"] for r in rows(directory)] == [1, 3, 4]
    backups = list(directory.glob("updates.jsonl.recovery-*.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == torn


@pytest.mark.parametrize(
    "bad", [b"garbage", b'{"update":99,"ppo":{"num_samples":1}}\n', b"{}\n", b"{bad\n"]
)
def test_metrics_corruption_rejects_without_writes(
    tmp_path: Path, settings: Path, bad: bytes
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=1)
    (directory / "updates.jsonl").write_bytes(bad)
    before = files(directory)
    with pytest.raises(ValueError):
        train(directory, settings, resume=True)
    assert files(directory) == before


def test_caps_do_not_reset_or_recollect(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=1)
    budget = json.loads((directory / "budget.json").read_bytes())
    budget["training"] = 120.0
    (directory / "budget.json").write_text(json.dumps(budget))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Collection must not start")

    monkeypatch.setattr(run, "collect_connect_four_self_play", forbidden)
    before = files(directory)
    trainer = train(directory, settings, resume=True)
    assert trainer.update_count == 1
    assert files(directory) == before


def test_failed_collection_charged_and_absolute_schedule_retried(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=1)
    old = files(directory)
    seeds = []
    original = collect_connect_four_self_play

    def fail(*args: Any, **kwargs: Any) -> Any:
        seeds.append(kwargs["engine_seed"])
        raise TimeoutError("disposable interruption")

    monkeypatch.setattr(run, "collect_connect_four_self_play", fail)
    trainer = train(directory, settings, resume=True)
    assert trainer.update_count == 1
    assert (directory / "latest.pt").read_bytes() == old["latest.pt"]
    assert (
        json.loads((directory / "budget.json").read_bytes())["training"]
        > json.loads(old["budget.json"])["training"]
    )
    monkeypatch.setattr(run, "collect_connect_four_self_play", original)
    train(directory, settings, resume=True, stop_after=2)
    assert seeds == [derive_connect_four_seed(630601, 1, "engine")]


def test_atomic_publication_failure_keeps_boundary(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=1)
    old = (directory / "latest.pt").read_bytes()
    original = os.replace

    def fail(source: Any, destination: Any) -> None:
        if Path(destination).name == "latest.pt":
            raise OSError("disposable save failure")
        original(source, destination)

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        train(directory, settings, resume=True)
    assert (directory / "latest.pt").read_bytes() == old
    assert not list(directory.glob(".connect-four-run-*"))


def test_training_and_evaluation_preserve_rng_threads_and_training_bytes(
    tmp_path: Path, settings: Path
) -> None:
    directory = tmp_path / "run"
    python = random.getstate()
    numpy = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    train(directory, settings)
    before = files(directory)
    report = evaluation.evaluate_connect_four_run(directory, settings_path=settings)
    assert report["complete"]
    assert all(len(cases) == 4 for cases in report["cases"].values())
    for name, data in before.items():
        if name != "budget.json":
            assert (directory / name).read_bytes() == data
    assert random.getstate() == python
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.get_num_threads() == threads
    initial, final, control = (
        report["cases"][s] for s in ("initial", "final", "control")
    )
    for a, b, c in zip(initial, final, control, strict=True):
        assert a["seeds"] == b["seeds"]
        assert a["seeds"]["engine"] == c["seeds"]["engine"]
        assert a["seeds"]["opponent"] == c["seeds"]["opponent"]
        assert a["seeds"]["policy"] != c["seeds"]["policy"]
    before = files(directory)
    with pytest.raises(ValueError, match="already exists"):
        evaluation.evaluate_connect_four_run(directory, settings_path=settings)
    assert files(directory) == before


def test_evaluation_timeout_records_partial_and_charges_budget(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    train(directory, settings)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise TimeoutError("stop")

    monkeypatch.setattr(evaluation, "_game", fail)
    report = evaluation.evaluate_connect_four_run(directory, settings_path=settings)
    assert not report["complete"] and report["stop_reason"] == "TimeoutError"
    assert report["cases"] == {"initial": [], "final": [], "control": []}
    assert json.loads((directory / "budget.json").read_bytes())["evaluation"] > 0


def test_paired_effect_hand_calculation() -> None:
    a = [{"seat": s, "score": score} for s in (0, 1) for score in (1.0, 0.0)]
    b = [{"seat": s, "score": 0.0} for s in (0, 1) for _ in range(2)]
    effect = evaluation._effect(a, b)
    assert effect["balanced_mean"] == 0.5
    assert effect["seats"][0]["standard_error"] == pytest.approx(0.5)
    assert effect["balanced_interval"] == pytest.approx(
        [0.5 - 1.96 * 0.5 / 2**0.5, 0.5 + 1.96 * 0.5 / 2**0.5]
    )


def test_protocol_settings_canonical_copy_and_version(
    tmp_path: Path, settings: Path
) -> None:
    assert (
        Path("docs/phase6/connect-four-functional-v1.json").read_bytes()
        == Path(
            "src/boardgame_ai/training/protocols/connect-four-functional-v1.json"
        ).read_bytes()
    )
    data = json.loads(settings.read_bytes())
    data["protocol_id"] = "connect-four-functional-v1"
    settings.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="new protocol version"):
        run.read_settings(settings)
    assert not (tmp_path / "run").exists()


def test_new_output_must_be_empty(tmp_path: Path, settings: Path) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "existing").write_bytes(b"keep")
    with pytest.raises(ValueError, match="absent or empty"):
        train(directory, settings)
    assert files(directory) == {"existing": b"keep"}


def test_remaining_deadline_uses_cumulative_budget(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=1)
    budget = json.loads((directory / "budget.json").read_bytes())
    budget["training"] = 119.0
    (directory / "budget.json").write_text(json.dumps(budget))
    monkeypatch.setattr(run, "monotonic", lambda: 1000.0)
    deadlines: list[float] = []

    def fail(*args: Any, **kwargs: Any) -> Any:
        deadlines.append(kwargs["deadline"])
        raise TimeoutError("deadline")

    monkeypatch.setattr(run, "collect_connect_four_self_play", fail)
    train(directory, settings, resume=True)
    assert deadlines == [1001.0]
    assert json.loads((directory / "budget.json").read_bytes())["training"] == 119.0


def test_metrics_sample_mismatch_does_not_repair_tail(
    tmp_path: Path, settings: Path
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=1)
    row = rows(directory)[0]
    row["ppo"]["num_samples"] += 1
    (directory / "updates.jsonl").write_text(json.dumps(row))
    before = files(directory)
    with pytest.raises(ValueError, match="samples disagree"):
        train(directory, settings, resume=True)
    assert files(directory) == before


def test_evaluation_requires_final_and_respects_spent_cap(
    tmp_path: Path, settings: Path
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=1)
    before = files(directory)
    with pytest.raises(ValueError, match="final boundary"):
        evaluation.evaluate_connect_four_run(directory, settings_path=settings)
    assert files(directory) == before
    train(directory, settings, resume=True)
    budget = json.loads((directory / "budget.json").read_bytes())
    budget["evaluation"] = 120
    (directory / "budget.json").write_text(json.dumps(budget))
    before = files(directory)
    with pytest.raises(ValueError, match="cap exhausted"):
        evaluation.evaluate_connect_four_run(directory, settings_path=settings)
    assert files(directory) == before


def test_saved_code_identity_rejects_without_metrics_repair(
    tmp_path: Path, settings: Path
) -> None:
    directory = tmp_path / "run"
    train(directory, settings, stop_after=1)
    manifest = json.loads((directory / "training.json").read_bytes())
    manifest["code"]["source_sha256"] = "wrong"
    (directory / "training.json").write_text(json.dumps(manifest))
    with (directory / "updates.jsonl").open("ab") as stream:
        stream.write(b'{"update":')
    before = files(directory)
    with pytest.raises(ValueError):
        train(directory, settings, resume=True)
    assert files(directory) == before


def test_missing_final_snapshot_recovered_without_new_training(
    tmp_path: Path, settings: Path
) -> None:
    directory = tmp_path / "run"
    train(directory, settings)
    final = (directory / "final.pt").read_bytes()
    (directory / "final.pt").unlink()
    before = files(directory)
    with pytest.raises(ValueError, match="Final snapshot missing"):
        evaluation.evaluate_connect_four_run(directory, settings_path=settings)
    assert files(directory) == before
    train(directory, settings, resume=True)
    assert (directory / "final.pt").read_bytes() == final
    assert files(directory) == {**before, "final.pt": final}


@pytest.mark.parametrize(
    "field,value",
    [("action_count", 9), ("schema_version", True), ("observation_shape", [6, 3, 3])],
)
def test_unsupported_protocol_fails_before_output(
    tmp_path: Path, settings: Path, field: str, value: Any
) -> None:
    data = json.loads(settings.read_bytes())
    data[field] = value
    settings.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        train(tmp_path / "run", settings)
    assert not (tmp_path / "run").exists()


def test_timeout_after_optimizer_work_returns_saved_boundary(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    boundary = train(directory, settings, stop_after=1)
    expected = connect_four_payload(boundary)
    old_update = type(boundary).update

    def interrupted(self: Any, *args: Any, **kwargs: Any) -> Any:
        old_update(self, *args, **kwargs)
        raise TimeoutError("completed optimizer work was not published")

    monkeypatch.setattr(type(boundary), "update", interrupted)
    actual = train(directory, settings, resume=True)
    assert_nested_equal(expected, connect_four_payload(actual))
    status = json.loads((directory / "training-status.json").read_bytes())
    assert status["updates"] == 1 and status["stop_reason"] == "time_cap"
    assert status["discarded_complete_games"] == 1
    assert [r["update"] for r in rows(directory)] == [1]


@pytest.mark.parametrize(
    "failure,errors,truncations",
    [(ValueError, 1, 0), (evaluation.EvaluationTruncationError, 0, 1)],
)
def test_evaluation_failure_counts_are_separate(
    tmp_path: Path,
    settings: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[Exception],
    errors: int,
    truncations: int,
) -> None:
    directory = tmp_path / "run"
    train(directory, settings)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise failure("private exception text")

    monkeypatch.setattr(evaluation, "_game", fail)
    report = evaluation.evaluate_connect_four_run(directory, settings_path=settings)
    assert not report["complete"]
    assert report["summaries"]["initial"]["seats"][0]["errors"] == errors
    assert report["summaries"]["initial"]["seats"][0]["truncations"] == truncations
    assert "private exception text" not in (directory / "evaluation.json").read_text()
