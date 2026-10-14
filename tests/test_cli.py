"""Check the command-line interface's public behavior."""

import pytest

from boardgame_ai.cli import main


def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    main(["info"])
    output = capsys.readouterr().out
    for label in ("Python:", "NumPy:", "PyTorch:", "CUDA available:"):
        assert label in output


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "info" in capsys.readouterr().out


def test_unknown_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["unknown"])
    assert exc.value.code == 2
