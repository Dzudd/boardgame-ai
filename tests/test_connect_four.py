"""Gravity, outcomes, lifecycle, public inputs and human column selection."""

import random

import pytest

from boardgame_ai.agents import RandomAgent
from boardgame_ai.cli import main
from boardgame_ai.cli_connect_four import HumanConnectFourAgent
from boardgame_ai.core import Action, GameOverError
from boardgame_ai.games.connect_four import ConnectFourEngine, ConnectFourObservation
from boardgame_ai.games.connect_four.engine import winner
from boardgame_ai.games.connect_four.evaluation import connect_four_outcome
from boardgame_ai.simulation import run_match
from tests.contracts import assert_engine_contract, assert_rejection_preserves_game


def drop(column: int) -> Action:
    return Action(kind="drop", parameters=(("column", column),))


def play(columns: list[int], starter: int = 0) -> ConnectFourEngine:
    game = ConnectFourEngine(starting_player=starter)
    game.reset(42)
    for column in columns:
        game.step(drop(column))
    return game


def test_shared_contract() -> None:
    assert_engine_contract(ConnectFourEngine, max_steps=42)


@pytest.mark.parametrize("starter", [0, 1])
@pytest.mark.parametrize(
    "columns",
    [
        [0, 6, 1, 6, 2, 5, 3],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 2, 4, 2, 2, 3, 4, 3, 5, 3, 3],
        [6, 5, 5, 4, 2, 4, 4, 3, 2, 3, 1, 3, 3],
    ],
)
def test_all_win_directions(columns: list[int], starter: int) -> None:
    game = play(columns, starter)
    assert game.is_terminal()
    assert winner(game.observe(0).data.board) == starter
    assert game.get_rewards() == tuple(1.0 if p == starter else -1.0 for p in (0, 1))
    assert connect_four_outcome(game).winners == (starter,)
    assert game.observe(0).phase == "finished"
    assert game.legal_actions(0) == game.legal_actions(1) == ()
    with pytest.raises(GameOverError):
        game.step(drop(4))


def test_draw_on_last_cell() -> None:
    columns = [
        3,
        3,
        0,
        2,
        5,
        4,
        5,
        6,
        4,
        2,
        2,
        3,
        3,
        5,
        5,
        2,
        2,
        4,
        3,
        3,
        1,
        1,
        2,
        1,
        5,
        5,
        0,
        4,
        0,
        4,
        4,
        6,
        6,
        6,
        6,
        0,
        1,
        0,
        6,
        0,
        1,
        1,
    ]
    game = play(columns[:-1])
    assert not game.is_terminal()
    result = game.step(drop(columns[-1]))
    assert result.terminated and result.rewards == (0.0, 0.0)
    assert connect_four_outcome(game).draw
    assert None not in game.observe(0).data.board


def test_gravity_public_history_and_stable_views() -> None:
    game = play([2, 2])
    observation = game.observe(1)
    assert observation.data.board[5 * 7 + 2] == 0
    assert observation.data.board[4 * 7 + 2] == 1
    assert observation.data.history == (2, 2)
    assert observation.data == game.observe(0).data
    assert game.legal_actions(1) == ()
    game.step(drop(4))
    game.reset()
    assert observation.data.history == (2, 2)
    assert game.get_rewards() == (0.0, 0.0)


@pytest.mark.parametrize(
    "action",
    [
        drop(-1),
        drop(7),
        Action(kind="place"),
        Action(kind="drop", parameters=(("column", True),)),
        Action(kind="drop", parameters=(("column", 1.0),)),
        Action(kind="drop", parameters=(("column", "1"),)),
        Action(kind="drop", parameters=(("column", 1), ("extra", 0))),
        Action(kind="drop", parameters=(("cell", 1),)),
    ],
)
def test_atomic_rejection(action: Action) -> None:
    assert_rejection_preserves_game(play([3]), action, max_steps=42)


def test_full_column_clone_reset_and_seed_rejection() -> None:
    game = play([0] * 6)
    assert drop(0) not in game.legal_actions(0)
    assert_rejection_preserves_game(game, drop(0), max_steps=42)
    game = play([0] * 6)
    before = game.observe(0)
    clone = game.clone()
    clone.step(drop(1))
    assert game.observe(0) == before
    with pytest.raises(ValueError):
        game.reset(True)
    assert game.observe(0) == before
    with pytest.raises(ValueError):
        connect_four_outcome(game)


@pytest.mark.parametrize("starter", [-1, 2, True, 0.0])
def test_invalid_starter(starter: int) -> None:
    with pytest.raises(ValueError):
        ConnectFourEngine(starting_player=starter)


def test_runner_reproducibility_and_global_rng() -> None:
    before = random.getstate()

    def match() -> object:
        game = ConnectFourEngine()
        result = run_match(
            game,
            (
                RandomAgent[ConnectFourObservation](),
                RandomAgent[ConnectFourObservation](),
            ),
            engine_seed=9,
            agent_seeds=(10, 11),
            max_steps=42,
        )
        assert result.terminated and not result.truncated and result.num_steps <= 42
        return result, game.observe(0), connect_four_outcome(game)

    assert match() == match()
    assert random.getstate() == before


def test_render_layout() -> None:
    assert play([1, 3]).render(0) == "\n".join(
        ["| . . . . . . . |"] * 5
        + ["| . X . O . . . |", "-----------------", "| 1 2 3 4 5 6 7 |"]
    )


def test_win_detection_does_not_wrap_rows() -> None:
    board: list[int | None] = [None] * 42
    for cell in (5, 6, 7, 8):
        board[cell] = 0
    assert winner(tuple(board)) is None


def test_quit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: " quit ")
    main(["play", "--game", "connect-four"])
    output = capsys.readouterr().out
    assert "Game stopped." in output and "Final position:" not in output


def test_human_retries_full_column_and_returns_supplied_action(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    game = play([0] * 6)
    values = iter(["bad", "0", "8", "1", " 2 "])
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))
    observation, legal = game.decision_for(0)
    assert HumanConnectFourAgent().select_action(observation, legal) is legal[0]
    assert legal[0] == drop(1)
    output = capsys.readouterr().out
    assert "1 2 3 4 5 6 7" in output and "full or unavailable" in output
    assert game.observe(0) == observation
    with pytest.raises(ValueError):
        HumanConnectFourAgent().select_action(observation, ())


@pytest.mark.parametrize("starts", ["human", "ai"])
def test_complete_cli_game(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    starts: str,
) -> None:
    values = iter([str(c) for _ in range(42) for c in range(1, 8)])
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))
    main(["play", "--game", "connect-four", "--starts", starts, "--agent-seed", "42"])
    output = capsys.readouterr().out
    assert "Final position:" in output
    assert any(text in output for text in ("You win!", "AI wins.", "Draw."))


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_cli_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: type[BaseException],
) -> None:
    def stop(prompt: str) -> str:
        raise error

    monkeypatch.setattr("builtins.input", stop)
    main(["play", "--game", "connect-four"])
    assert "Game stopped." in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra", [["--opponent", "heuristic"], ["--save-replay", "x.json"]]
)
def test_unsupported_cli_options(extra: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["play", "--game", "connect-four", *extra])
    assert error.value.code == 2
