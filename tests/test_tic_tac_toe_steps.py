"""Nonterminal transitions and atomic step rejection for the partial engine."""

import pytest

from boardgame_ai.core import (
    Action,
    GameNotReadyError,
    GameOverError,
    IllegalActionError,
)
from tests.test_tic_tac_toe_actions import place
from tests.test_tic_tac_toe_reset import ResetHarness


@pytest.mark.parametrize("cell", range(9))
def test_first_move_places_x_in_only_the_chosen_cell(cell: int) -> None:
    game = ResetHarness()
    game.reset(42)
    previous = game.state
    result = game.step(place(cell))
    assert game.state.data.board == tuple(0 if i == cell else None for i in range(9))
    assert previous.data.board == (None,) * 9
    assert game.state is not previous
    assert game.current_player() == result.next_player == 1
    assert game.state.phase == "play"
    assert result.terminated is False
    assert result.rewards == game.get_rewards() == (0.0, 0.0)
    assert result.events == ()
    assert game.legal_actions(0) == ()
    assert game.legal_actions(1) == tuple(place(i) for i in range(9) if i != cell)


def test_sequence_preserves_marks_and_alternates_actors_without_rng_draws() -> None:
    game = ResetHarness()
    game.reset(42)
    rng = game._rng
    assert rng is not None
    rng_state = rng.getstate()
    expected: list[int | None] = [None] * 9
    for turn, cell in enumerate((4, 0, 8, 2)):
        assert game.current_player() == turn % 2
        expected[cell] = turn % 2
        result = game.step(place(cell))
        assert game.state.data.board == tuple(expected)
        assert game.current_player() == result.next_player == 1 - turn % 2
        assert result.rewards == game.get_rewards() == (0.0, 0.0)
        assert rng.getstate() == rng_state
    game.reset(42)
    assert game.state.data.board == (None,) * 9
    assert game.current_player() == 0


@pytest.mark.parametrize(
    "action",
    [
        place(4),  # Occupied by X.
        place(0),  # Occupied by O.
        place(-1),
        place(9),
        Action(kind="pass"),
        Action(kind="place"),
        Action(kind="place", parameters=(("row", 1),)),
        Action(kind="place", parameters=(("cell", 1), ("player", 0))),
        *(
            Action(kind="place", parameters=(("cell", value),))
            for value in (True, False, 1.0, "1", None)
        ),
    ],
)
def test_rejected_step_preserves_state_actor_rewards_and_rng(action: Action) -> None:
    game = ResetHarness()
    game.reset(42)
    game.step(place(4))
    game.step(place(0))
    game._rewards = (0.5, -0.5)  # Verify rejection retains existing rewards.
    state, rewards, rng = game.state, game.get_rewards(), game._rng
    assert rng is not None
    rng_state = rng.getstate()
    with pytest.raises(IllegalActionError):
        game.step(action)
    assert game.state is state
    assert game.current_player() == 0
    assert game.get_rewards() == rewards
    assert game._rng is rng
    assert rng.getstate() == rng_state
    assert place(1) in game.legal_actions(0)


def test_step_requires_reset_and_rejects_stored_terminal_state() -> None:
    game = ResetHarness()
    with pytest.raises(GameNotReadyError):
        game.step(place(0))
    game.reset(42)
    game.dirty_state()
    state, rewards, rng = game.state, game.get_rewards(), game._rng
    assert rng is not None
    rng_state = rng.getstate()
    with pytest.raises(GameOverError):
        game.step(place(0))
    assert game.state is state
    assert game.get_rewards() == rewards
    assert rng.getstate() == rng_state


def test_successful_step_replaces_latest_rewards() -> None:
    game = ResetHarness()
    game.reset()
    game._rewards = (0.5, -0.5)
    result = game.step(place(0))
    assert result.rewards == game.get_rewards() == (0.0, 0.0)
