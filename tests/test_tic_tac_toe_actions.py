"""Legal choices and validation, using trusted setup before transitions exist."""

import pytest

from boardgame_ai.core import (
    Action,
    GameNotReadyError,
    GameOverError,
    GameState,
    IllegalActionError,
    InvalidPlayerError,
)
from boardgame_ai.games.tic_tac_toe import TicTacToeState
from tests.test_tic_tac_toe_reset import ResetHarness


def place(cell: int) -> Action:
    return Action(kind="place", parameters=(("cell", cell),))


def test_actions_require_reset() -> None:
    game = ResetHarness()
    with pytest.raises(GameNotReadyError):
        game.legal_actions(0)
    with pytest.raises(GameNotReadyError):
        game._validate_action(place(0))


def test_initial_choices_and_validated_cells() -> None:
    game = ResetHarness()
    game.reset(42)
    assert game.legal_actions(0) == tuple(place(cell) for cell in range(9))
    assert game.legal_actions(1) == ()
    for cell in range(9):
        assert game._validate_action(place(cell)) == cell


def test_choices_follow_actor_and_skip_both_marks() -> None:
    game = ResetHarness()
    game.reset(42)
    game._state = GameState(
        current_player=1,
        phase="play",
        data=TicTacToeState(board=(0, None, None, None, 1, None, None, None, 0)),
    )
    assert game.legal_actions(0) == ()
    assert game.legal_actions(1) == tuple(place(cell) for cell in (1, 2, 3, 5, 6, 7))
    for action in game.legal_actions(1):
        game._validate_action(action)
    for cell in (0, 4, 8):
        with pytest.raises(IllegalActionError):
            game._validate_action(place(cell))


@pytest.mark.parametrize("player", [-1, 2, True, False, 0.0, "0", None])
def test_invalid_seats_are_rejected(player: object) -> None:
    game = ResetHarness()
    game.reset()
    with pytest.raises(InvalidPlayerError):
        game.legal_actions(player)  # type: ignore[arg-type]
    game.dirty_state()
    with pytest.raises(InvalidPlayerError):
        game.legal_actions(player)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "action",
    [
        Action(kind="pass"),
        Action(kind="PLACE", parameters=(("cell", 0),)),
        Action(kind="place"),
        Action(kind="place", parameters=(("row", 0),)),
        Action(kind="place", parameters=(("cell", 0), ("extra", 1))),
        *(
            Action(kind="place", parameters=(("cell", cell),))
            for cell in (-1, 9, 100, True, False, 0.0, 1.5, "0", None)
        ),
    ],
)
def test_invalid_actions_preserve_state_rewards_and_rng(action: Action) -> None:
    game = ResetHarness()
    game.reset(42)
    state, rewards, rng = game.state, game.get_rewards(), game._rng
    assert rng is not None
    rng_state = rng.getstate()
    with pytest.raises(IllegalActionError, match="Action is not legal"):
        game._validate_action(action)
    assert game.state is state
    assert game.get_rewards() == rewards
    assert game._rng is rng
    assert rng.getstate() == rng_state


def test_terminal_choices_and_rejection_even_with_empty_cells() -> None:
    game = ResetHarness()
    game.reset()
    game._state = GameState(
        current_player=None,
        phase="finished",
        data=TicTacToeState(board=(0, 0, 0, 1, 1, None, None, None, None)),
    )
    assert game.legal_actions(0) == game.legal_actions(1) == ()
    for action in (place(5), Action(kind="pass")):
        with pytest.raises(GameOverError):
            game._validate_action(action)


def test_queries_and_successful_validation_do_not_change_state_or_rng() -> None:
    game = ResetHarness()
    game.reset(42)
    state, rng = game.state, game._rng
    assert rng is not None
    rng_state = rng.getstate()
    for _ in range(2):
        for action in game.legal_actions(0):
            game._validate_action(action)
        assert game.legal_actions(1) == ()
    assert game.state is state
    assert game.get_rewards() == (0.0, 0.0)
    assert rng.getstate() == rng_state
