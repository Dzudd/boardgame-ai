"""Live console feedback on disposable runs and narrow source compatibility."""

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from boardgame_ai.training import connect_four_run as run
from boardgame_ai.training.connect_four_checkpoint import connect_four_payload
from tests.test_connect_four_run import files, rows, stable, train
from tests.test_ppo_checkpoint import assert_nested_equal

PROTOCOL = "docs/PHASE6_CONNECT_FOUR_PROTOCOL.md"


@pytest.fixture
def settings(tmp_path: Path) -> Path:
    data = json.loads(Path("docs/phase6/connect-four-functional-v1.json").read_bytes())
    data["protocol_id"] = "disposable-live-progress-v1"
    data["training"].update(updates=2, games_per_update=1)
    data["evaluation"]["checkpoints"] = ["initial", "update-2"]
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("option", ["normal", "--quiet", "--check-only"])
def test_cli_live_progress_and_stdout_json(
    tmp_path: Path,
    settings: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: str,
) -> None:
    output = tmp_path / "run"
    args = [
        "connect-four-run",
        str(output),
        "--settings",
        str(settings),
        "--protocol",
        PROTOCOL,
    ]
    if option != "normal":
        args.append(option)
    monkeypatch.setattr(sys, "argv", args)
    run.main()
    captured = capsys.readouterr()
    assert isinstance(json.loads(captured.out), dict)
    if option != "normal":
        assert captured.err == ""
    else:
        assert "[------------]" in captured.err
        assert "[############]" in captured.err
        assert "100.0%" in captured.err and "2/2" in captured.err
        assert "Stopped: boundary" in captured.err
        assert len(captured.err.splitlines()) == 2
    if option == "--check-only":
        assert not output.exists()


def test_terminal_bar_overwrites_throttles_and_finishes_one_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Terminal(io.StringIO):
        flushes = 0

        def isatty(self) -> bool:
            return True

        def flush(self) -> None:
            self.flushes += 1

    terminal = Terminal()
    clock = [0.0]
    monkeypatch.setattr(sys, "stderr", terminal)
    monkeypatch.setattr(run, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "boardgame_ai.training.connect_four_run.shutil.get_terminal_size",
        lambda: os.terminal_size((60, 24)),
    )
    bar = run._ConsoleProgress()
    bar("Starting test on cpu; target update 10/10, scheduled games 80.")
    clock[0] = 0.1
    bar("Saved update 1/10; games 8/80; remaining 100s")
    assert terminal.getvalue().count("\r") == 1
    clock[0] = 0.3
    bar("Saved update 2/10; games 16/80; remaining 99s")
    bar("Stopped: boundary; saved update 2/10, games 16; complete=False")
    text = terminal.getvalue()
    assert text.count("\r") == 3 and text.count("\n") == 1
    assert "20.0%" in text and "2/10" in text
    assert all(len(line.rstrip("\n")) <= 59 for line in text.split("\r"))
    assert terminal.flushes == 3


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_progress_preserves_learning_and_resume(
    tmp_path: Path, settings: Path, device: str
) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data = json.loads(settings.read_bytes())
    data["training"]["device"] = device
    settings.write_text(json.dumps(data), encoding="utf-8")
    whole, split = tmp_path / "silent", tmp_path / "live"
    expected = train(whole, settings)
    messages: list[str] = []
    train(split, settings, stop_after=1, progress=messages.append)
    actual = train(split, settings, resume=True, progress=messages.append)
    assert_nested_equal(connect_four_payload(expected), connect_four_payload(actual))
    assert [stable(r) for r in rows(whole)] == [stable(r) for r in rows(split)]
    assert any("Resuming" in m for m in messages)
    assert any("Saved update 2/2" in m for m in messages)
    before = files(split)
    train(split, settings, resume=True, progress=messages.append)
    assert "Already at target" in messages[-1]
    assert files(split) == before


def test_progress_timeout_never_claims_unsaved_update(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages: list[str] = []

    def timeout(*args: Any, **kwargs: Any) -> Any:
        raise TimeoutError("disposable collection stop")

    monkeypatch.setattr(run, "collect_connect_four_self_play", timeout)
    actual = train(tmp_path / "run", settings, progress=messages.append)
    assert actual.update_count == 0
    assert not any("Saved update 1/" in m for m in messages)
    assert "Stopped: time_cap; saved update 0/2" in messages[-1]
    assert "complete=False" in messages[-1]


@pytest.mark.parametrize("mismatch", ["none", "source", "current-source", "runtime"])
def test_compatibility_is_one_exact_source_transition_without_rewrites(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    output = tmp_path / "run"
    train(output, settings, stop_after=0)
    mapping = json.loads(
        Path(
            "src/boardgame_ai/training/protocols/connect-four-console-compat-v1.json"
        ).read_bytes()
    )
    assert mapping["source_sha256"] == run._source_digest()
    path = output / "training.json"
    manifest = json.loads(path.read_bytes())
    manifest["code"]["source_sha256"] = mapping["previous_source_sha256"]
    if mismatch == "source":
        manifest["code"]["source_sha256"] = "unreviewed-source"
    elif mismatch == "current-source":
        monkeypatch.setattr(run, "_source_digest", lambda: "unreviewed-current-source")
    elif mismatch == "runtime":
        manifest["runtime"]["torch"] = "different-runtime"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = files(output)
    if mismatch == "none":
        report = run.check_connect_four_run(
            output, settings_path=settings, protocol_path=PROTOCOL, resume=True
        )
        assert report["ready"]
        train(output, settings, resume=True, stop_after=0)
    else:
        with pytest.raises(ValueError, match="provenance mismatch"):
            run.check_connect_four_run(
                output, settings_path=settings, protocol_path=PROTOCOL, resume=True
            )
    assert files(output) == before


def test_slow_console_output_cannot_publish_past_deadline(
    tmp_path: Path, settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    train(output, settings, stop_after=1)
    before = files(output)
    clock = [1000.0]
    monkeypatch.setattr(run, "monotonic", lambda: clock[0])
    monkeypatch.setattr("boardgame_ai.training.ppo.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "boardgame_ai.training.connect_four.monotonic", lambda: clock[0]
    )

    def slow_output(message: str) -> None:
        if "publishing checkpoint" in message:
            clock[0] = 1300.0

    actual = train(output, settings, resume=True, progress=slow_output)
    assert actual.update_count == 1
    assert (output / "latest.pt").read_bytes() == before["latest.pt"]
    status = json.loads((output / "training-status.json").read_bytes())
    assert status["stop_reason"] == "time_cap" and status["cap_overrun_seconds"] > 0
