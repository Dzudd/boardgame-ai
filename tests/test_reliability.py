"""Fault paths use disposable artifacts and tiny runs, never recorded experiments."""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from boardgame_ai.artifact_safety import validate_evaluation_destination
from boardgame_ai.policy_evaluation import (
    evaluate_policy_checkpoint,
    save_policy_evaluation,
)
from boardgame_ai.training import PPOConfig, TrainingConfig, run_self_play_training
from boardgame_ai.training.experiment_protocol import (
    LEGACY_SHA256,
    create_experiment_protocol,
    legacy_manifest,
    validate_experiment_protocol,
)
from boardgame_ai.training.metrics_log import _unfinished_json, recover_metrics_log


@pytest.mark.parametrize(
    "name",
    [
        "latest.pt",
        "retained.pt",
        "training.json",
        "updates.jsonl",
        "monitoring.json",
        "best.json",
        "development-00000001.json",
        "snapshots/report.json",
    ],
)
def test_managed_evaluation_destinations_reject_before_creation(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / name
    path.parent.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="training artifacts"):
        validate_evaluation_destination(path)
    assert not path.exists()


def test_report_writer_preserves_every_existing_identity(tmp_path: Path) -> None:
    config = TrainingConfig(updates=0, device="cpu")
    run_self_play_training(tmp_path / "run", config)
    source = tmp_path / "run/latest.pt"
    result = evaluate_policy_checkpoint(source, games_per_seat=1)
    for path in (
        source,
        tmp_path / "monitoring.json",
        tmp_path / "best.json",
        tmp_path / "development-00000001.json",
        tmp_path / "ordinary.json",
    ):
        if path != source:
            path.write_bytes(b"original evidence")
        before = path.read_bytes()
        with pytest.raises((ValueError, FileExistsError)):
            save_policy_evaluation(result, path)
        assert path.read_bytes() == before
    alias = tmp_path / "alias.json"
    os.link(source, alias)
    before = source.read_bytes()
    with pytest.raises(FileExistsError):
        save_policy_evaluation(result, alias)
    assert source.read_bytes() == alias.read_bytes() == before
    report = tmp_path / "new.json"
    save_policy_evaluation(result, report)
    with pytest.raises(FileExistsError):
        save_policy_evaluation(result, report.parent / "run/../new.json")


def test_symlink_evaluation_alias_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "monitoring.json"
    source.write_bytes(b"original")
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(source)
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer mode or privilege")
    with pytest.raises(ValueError, match="training artifacts"):
        validate_evaluation_destination(alias)
    assert alias.is_symlink() and source.read_bytes() == b"original"


def test_exclusive_report_creation_checks_at_write_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from boardgame_ai import policy_evaluation

    run_self_play_training(tmp_path / "run", TrainingConfig(updates=0, device="cpu"))
    result = evaluate_policy_checkpoint(tmp_path / "run/latest.pt", games_per_seat=1)
    destination = tmp_path / "report.json"

    def racing_writer(path: str | Path) -> Path:
        validated = validate_evaluation_destination(path)
        validated.write_bytes(b"another writer")
        return validated

    monkeypatch.setattr(
        policy_evaluation, "validate_evaluation_destination", racing_writer
    )
    with pytest.raises(FileExistsError):
        save_policy_evaluation(result, destination)
    assert destination.read_bytes() == b"another writer"


def row(update: int) -> bytes:
    return json.dumps({"update": update, "ppo": {"num_samples": 4}}).encode() + b"\n"


@pytest.mark.parametrize(
    "tail",
    [
        b'{"update":',
        b'{"update": 2, "ppo": {"num_samples": 4',
        b'{"x": "abc',
        b'{"x": 1e-',
        b'{"x": tru',
        b'{"x": "\\u12',
    ],
)
def test_torn_final_record_is_backed_up_and_recovery_is_idempotent(
    tmp_path: Path, tail: bytes
) -> None:
    path = tmp_path / "updates.jsonl"
    original = row(1) + tail
    path.write_bytes(original)
    assert recover_metrics_log(path, 2) is None
    assert path.read_bytes() == row(1)
    backups = list(tmp_path.glob("*.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert recover_metrics_log(path, 2) is None
    assert list(tmp_path.glob("*.bak")) == backups


@pytest.mark.parametrize(
    "data",
    [
        b'{"update":\n',
        b"bad\n" + row(1),
        row(1) + b'{"x": broken}',
        row(1) + b'{"x": 1.e',
        row(1) + b'{"x": t r',
        row(1) + b"\xff",
        row(1) + row(1),
        row(3),
        b"{}\n",
        b"[]",
    ],
)
def test_corruption_rejects_without_writes(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "updates.jsonl"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        recover_metrics_log(path, 2)
    assert path.read_bytes() == data
    assert list(tmp_path.iterdir()) == [path]


def test_all_truncations_of_emitted_json_are_valid_prefixes() -> None:
    text = json.dumps(
        {
            "update": 2,
            "ppo": {"num_samples": 14, "kl": -1.2e-10, "ok": True, "text": "a\n\u2603"},
        }
    )
    assert all(_unfinished_json(text[:end]) for end in range(1, len(text)))
    assert not _unfinished_json(text)


def test_valid_final_record_without_newline_and_gapped_counts(tmp_path: Path) -> None:
    path = tmp_path / "updates.jsonl"
    path.write_bytes(row(1).rstrip(b"\n"))
    assert recover_metrics_log(path, 1) == 4
    assert path.read_bytes() == row(1)
    assert next(tmp_path.glob("*.bak")).read_bytes() == row(1).rstrip(b"\n")
    path.write_bytes(row(2))
    assert recover_metrics_log(path, 2) is None


@pytest.mark.parametrize("failure", ["backup", "replace"])
def test_recovery_publication_failure_preserves_original_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from boardgame_ai.training import metrics_log

    path = tmp_path / "updates.jsonl"
    damaged = row(1) + b'{"update":'
    path.write_bytes(damaged)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected publication failure")

    with monkeypatch.context() as patch:
        if failure == "backup":
            patch.setattr(metrics_log, "_immutable_snapshot", fail)
        else:
            patch.setattr(os, "replace", fail)
        with pytest.raises(OSError, match="injected"):
            recover_metrics_log(path, 2)
    assert path.read_bytes() == damaged
    assert recover_metrics_log(path, 2) is None
    assert path.read_bytes() == row(1)
    assert next(tmp_path.glob("*.bak")).read_bytes() == damaged


def test_recovered_missing_row_keeps_monitoring_sample_count_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from boardgame_ai.training import run
    from boardgame_ai.training.monitoring import MonitoringConfig

    config = TrainingConfig(
        updates=1,
        games_per_update=1,
        device="cpu",
        ppo=PPOConfig(epochs=1, minibatch_size=32),
    )
    monitoring = MonitoringConfig(interval=1)
    counts: list[int | None] = []

    def observe(*args: Any, **kwargs: Any) -> None:
        counts.append(kwargs["samples_seen"])

    monkeypatch.setattr(run, "monitor_checkpoint", observe)
    run_self_play_training(tmp_path, config, monitoring=monitoring)
    (tmp_path / "updates.jsonl").write_bytes(b'{"update":')
    counts.clear()
    run_self_play_training(
        tmp_path, replace(config, updates=2), resume=True, monitoring=monitoring
    )
    assert counts == [None, None]
    assert [
        json.loads(line)["update"]
        for line in (tmp_path / "updates.jsonl").read_text().splitlines()
    ] == [2]


def test_recovered_resume_matches_model_adam_shuffle_and_future_rows(
    tmp_path: Path,
) -> None:
    config = TrainingConfig(
        updates=3,
        games_per_update=2,
        device="cpu",
        seed=123,
        ppo=PPOConfig(epochs=1, minibatch_size=32),
    )
    full = tmp_path / "full"
    interrupted = tmp_path / "interrupted"
    expected = run_self_play_training(full, config)

    def interrupt(record: dict[str, Any]) -> None:
        if record["update"] == 2:
            raise InterruptedError

    with pytest.raises(InterruptedError):
        run_self_play_training(interrupted, config, on_update=interrupt)
    log = interrupted / "updates.jsonl"
    lines = log.read_bytes().splitlines(keepends=True)
    # Inject interruption after checkpoint publication and during row-2 append.
    damaged = lines[0] + lines[1][: len(lines[1]) // 2]
    log.write_bytes(damaged)
    checkpoint = (interrupted / "latest.pt").read_bytes()
    assert recover_metrics_log(log, 2) is None
    assert (interrupted / "latest.pt").read_bytes() == checkpoint
    assert next(interrupted.glob("*.bak")).read_bytes() == damaged
    # Reintroduce the fault so the actual resume entry point exercises recovery.
    log.write_bytes(damaged)
    actual = run_self_play_training(interrupted, config, resume=True)
    assert actual.update_count == expected.update_count == 3
    assert actual.optimizer_steps == expected.optimizer_steps
    assert torch.equal(actual.generator.get_state(), expected.generator.get_state())
    for key, value in expected.model.state_dict().items():
        assert torch.equal(value, actual.model.state_dict()[key])
    for key, state in expected.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            assert torch.equal(value, actual.optimizer.state_dict()["state"][key][name])
    assert (
        log.read_bytes()
        == lines[0] + (full / "updates.jsonl").read_bytes().splitlines(keepends=True)[2]
    )


def test_protocol_prose_is_editable_but_saved_provenance_is_checked(
    tmp_path: Path,
) -> None:
    prose = tmp_path / "explanation.md"
    prose.write_text("original explanation")
    output = tmp_path / "experiment"
    create_experiment_protocol(output, prose)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    prose.write_text("editorial changes")
    validated = validate_experiment_protocol(output)
    assert validated["compatibility"] == "native-v2"
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    (output / "protocol.md").write_text("damaged snapshot")
    with pytest.raises(ValueError, match="mismatch"):
        validate_experiment_protocol(output)
    (output / "protocol.md").write_bytes(before["protocol.md"])
    (output / "settings.json").write_text("{}")
    with pytest.raises(ValueError, match="mismatch"):
        validate_experiment_protocol(output)


def test_historical_compatibility_requires_exact_known_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "experiment.json"
    saved = legacy_manifest()
    manifest.write_text(json.dumps(saved))
    before = manifest.read_bytes()
    result = validate_experiment_protocol(tmp_path)
    assert result["compatibility"] == "verified-v1"
    assert result["provenance"]["protocol_sha256"] == LEGACY_SHA256
    assert manifest.read_bytes() == before and len(list(tmp_path.iterdir())) == 1
    for field, value in (
        ("protocol_sha256", "unknown"),
        ("updates", 201),
        ("threads", True),
    ):
        manifest.write_text(json.dumps({**saved, field: value}))
        with pytest.raises(ValueError, match="Unverified historical protocol"):
            validate_experiment_protocol(tmp_path)


def test_readonly_protocol_cli_cannot_launch_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from boardgame_ai.training import experiments

    (tmp_path / "experiment.json").write_text(json.dumps(legacy_manifest()))

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Read-only validation launched an experiment")

    monkeypatch.setattr(experiments, "run_experiments", forbidden)
    monkeypatch.setattr(
        "sys.argv", ["experiments", "--output", str(tmp_path), "--validate-only"]
    )
    experiments.main()
    assert json.loads(capsys.readouterr().out)["compatibility"] == "verified-v1"
    assert len(list(tmp_path.iterdir())) == 1


def test_native_resume_uses_frozen_settings_after_prose_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from boardgame_ai.training import experiments
    from boardgame_ai.training.experiment_protocol import settings

    prose = tmp_path / "prose.md"
    prose.write_text("original")
    output = tmp_path / "experiment"
    create_experiment_protocol(output, prose)
    prose.write_text("changed")
    before = {p.name: p.read_bytes() for p in output.iterdir()}

    def stop_before_training(path: Path, config: TrainingConfig, **kwargs: Any) -> None:
        frozen = settings()
        assert config.ppo == PPOConfig(**frozen["ppo"])
        assert config.updates == frozen["updates"]
        assert config.games_per_update == frozen["games_per_update"]
        assert config.seed == frozen["seeds"][0]
        assert kwargs["monitoring"].seed == frozen["monitoring"]["seed"]
        raise InterruptedError

    threads = torch.get_num_threads()
    monkeypatch.setattr(experiments, "run_self_play_training", stop_before_training)
    with pytest.raises(InterruptedError):
        experiments.run_experiments(output, resume=True)
    assert torch.get_num_threads() == threads
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_recorded_phase45_copy_validates_without_writing(tmp_path: Path) -> None:
    original = Path("runs/phase45/entropy-comparison")
    if not original.exists():
        pytest.skip("Local recorded artifacts are not part of a clone")
    # Include the recorded runs and checkpoints, without loading or resuming them.
    copied = tmp_path / "recorded-copy"
    shutil.copytree(original, copied)
    before = {p: p.read_bytes() for p in copied.rglob("*") if p.is_file()}
    assert validate_experiment_protocol(copied)["compatibility"] == "verified-v1"
    assert {p: p.read_bytes() for p in copied.rglob("*") if p.is_file()} == before
