"""Public replay round trips, engine reconstruction, validation, and CLI logs."""

import json
from pathlib import Path
from typing import Any

import pytest

from boardgame_ai.agents import RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.cli import main
from boardgame_ai.core import Action, IllegalActionError
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.games.tic_tac_toe.replay import (
    ReplayMove,
    TicTacToeRecorder,
    TicTacToeReplay,
    load_replay,
    replay_positions,
    save_replay,
)
from boardgame_ai.simulation import play_game, run_match
from tests.test_cli_play import input_sequence


@pytest.mark.parametrize("starter", [0, 1])
@pytest.mark.parametrize(
    "moves",
    [
        (0, 3, 1, 4, 2),
        (0, 3, 1, 4, 8, 5),
        (0, 1, 2, 4, 3, 5, 7, 6, 8),
    ],
)
def test_full_replay_roundtrip_and_positions(
    tmp_path: Path,
    starter: int,
    moves: tuple[int, ...],
) -> None:
    game = TicTacToeEngine(starting_player=starter)
    game.reset(42)
    recorder = TicTacToeRecorder()
    expected = [game.observe(0)]
    for cell in moves:
        actor = game.current_player()
        assert actor is not None
        action = Action(kind="place", parameters=(("cell", cell),))
        game.step(action)
        recorder.record(actor, action)
        expected.append(game.observe(0))
    replay = recorder.finish(starter, "completed")
    path = tmp_path / "game.json"
    save_replay(replay, path)
    assert load_replay(path) == replay
    assert replay_positions(replay) == tuple(expected)
    data = json.loads(path.read_text())
    assert set(data) == {
        "schema_version",
        "game_id",
        "starting_player",
        "moves",
        "status",
    }
    assert "seed" not in path.read_text() and "events" not in path.read_text()


def test_runner_callback_records_only_successful_steps_and_resume(
    tmp_path: Path,
) -> None:
    game = TicTacToeEngine()
    agents = (TicTacToeHeuristicAgent(), RandomAgent[TicTacToeObservation](42))
    recorder = TicTacToeRecorder()
    result = run_match(
        game, agents, max_steps=2, agent_seeds=(None, 42), on_step=recorder.record
    )
    assert result.truncated
    replay = recorder.finish(0, "truncated")
    assert len(replay.moves) == 2
    assert replay_positions(replay)[-1] == game.observe(0)
    result = play_game(game, agents, max_steps=9, on_step=recorder.record)
    assert result.terminated
    assert replay_positions(recorder.finish(0, "completed"))[-1] == game.observe(0)


@pytest.mark.parametrize("status", ["cancelled", "truncated"])
def test_zero_move_incomplete_replays(status: Any, tmp_path: Path) -> None:
    replay = TicTacToeRecorder().finish(0, status)
    save_replay(replay, tmp_path / "empty.json")
    assert len(replay_positions(load_replay(tmp_path / "empty.json"))) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"game_id": "hidden_test"},
        {"starting_player": True},
        {"status": "completed"},
        {"status": "unknown"},
        {"moves": [{"player": 1, "cell": 0}]},
        {"moves": [{"player": 0, "cell": True}]},
        {"moves": [{"player": 0, "cell": 9}]},
        {"moves": [{"player": 0, "cell": 0}, {"player": 1, "cell": 0}]},
        {"moves": [{"player": 0, "cell": 0}] * 10},
        {"moves": {}},
        {"private_hand": [999]},
    ],
)
def test_invalid_logs_rejected(change: dict[str, Any], tmp_path: Path) -> None:
    data: dict[str, Any] = dict(
        schema_version=1,
        game_id="tic_tac_toe",
        starting_player=0,
        moves=[],
        status="cancelled",
    )
    data.update(change)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid Tic-Tac-Toe replay"):
        load_replay(path)


def test_moves_after_termination_and_invalid_status_do_not_overwrite(
    tmp_path: Path,
) -> None:
    moves = tuple(
        ReplayMove(player=index % 2, cell=cell)
        for index, cell in enumerate((0, 3, 1, 4, 2, 5))
    )
    path = tmp_path / "preserved.json"
    path.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError):
        save_replay(
            TicTacToeReplay(starting_player=0, moves=moves, status="completed"), path
        )
    assert path.read_text() == "preserve"


def test_rejected_engine_action_is_not_recorded() -> None:
    from tests.test_interfaces import DummyGame
    from tests.test_simulation import IllegalAgent, RecordingAgent

    calls: list[tuple[int, Action]] = []
    with pytest.raises(IllegalActionError):
        run_match(
            DummyGame(),
            (IllegalAgent(), RecordingAgent[int]()),
            max_steps=3,
            on_step=lambda actor, action: calls.append((actor, action)),
        )
    assert calls == []


@pytest.mark.parametrize("starts", ["human", "ai"])
def test_cli_save_completed_then_inspect(
    starts: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "played.json"
    input_sequence(monkeypatch, iter(str(cell) for cell in list(range(9)) * 6))
    main(
        [
            "play",
            "--game",
            "tic-tac-toe",
            "--starts",
            starts,
            "--save-replay",
            str(path),
        ]
    )
    assert "Replay saved:" in capsys.readouterr().out
    replay = load_replay(path)
    assert replay.status == "completed"
    main(["replay", str(path)])
    output = capsys.readouterr().out
    assert "Replay: Tic-Tac-Toe" in output and "Status: completed" in output
    assert output.count("places at cell") == len(replay.moves)
    assert "wins." in output or "Draw." in output


def test_cancelled_cli_log_retains_only_played_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "cancelled.json"
    input_sequence(monkeypatch, iter(["bad", "quit"]))
    main(["play", "--starts", "ai", "--save-replay", str(path)])
    replay = load_replay(path)
    assert replay.status == "cancelled" and len(replay.moves) == 1
    capsys.readouterr()
    main(["replay", str(path)])
    output = capsys.readouterr().out
    assert "Status: cancelled" in output
    assert "wins." not in output and "Draw." not in output


def test_cli_invalid_replay_has_no_partial_display(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["replay", str(path)])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "Cannot read or validate replay" in output.err
