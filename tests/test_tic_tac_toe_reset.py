"""Reset tests with trusted state setup for lifecycle edge cases."""

from inspect import isabstract
from random import getstate

import pytest

from boardgame_ai.core import (
    GameNotReadyError,
    GameState,
)
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeState


class ResetHarness(TicTacToeEngine):
    """Expose trusted state setup for tests; inherit all gameplay methods."""

    @property
    def state(self) -> GameState[TicTacToeState]:
        return self._require_state()

    def dirty_state(self) -> None:
        self._state = GameState(
            current_player=None,
            phase="finished",
            data=TicTacToeState(board=(0, 1, 0, 1, 0, 1, 0, 1, 0)),
        )
        self._rewards = (1.0, -1.0)


def test_engine_is_concrete_and_metadata_is_available() -> None:
    assert not isabstract(TicTacToeEngine)
    game = ResetHarness()
    assert game.spec.game_id == "tic_tac_toe"
    assert game.spec.min_players == game.spec.max_players == game.num_players == 2
    for query in (game.current_player, game.get_rewards, lambda: game.state):
        with pytest.raises(GameNotReadyError):
            query()


@pytest.mark.parametrize("seed", [None, 0, 42, -7, 2**80])
def test_reset_initializes_empty_board_and_x_first(seed: int | None) -> None:
    game = ResetHarness()
    global_rng = getstate()
    game.reset(seed)
    assert game.state.data.board == (None,) * 9
    assert game.state.phase == "play"
    assert game.current_player() == 0
    assert game.get_rewards() == (0.0, 0.0)
    assert getstate() == global_rng


def test_reset_replaces_previous_state_and_clears_rewards() -> None:
    game = ResetHarness()
    game.reset(42)
    initial = game.state
    game.dirty_state()
    previous = game.state
    game.reset(42)
    assert game.state == initial
    assert game.state is not initial
    assert game.state is not previous
    assert game.state.data is not initial.data
    assert previous.data.board[0] == 0
    assert game.get_rewards() == (0.0, 0.0)


@pytest.mark.parametrize("seed", [True, False, 1.5, "42"])
def test_invalid_seed_preserves_state_rewards_and_rng(seed: object) -> None:
    game = ResetHarness()
    game.reset(42)
    game.dirty_state()
    state, rewards, rng = game.state, game.get_rewards(), game._rng
    assert rng is not None
    rng_state = rng.getstate()
    with pytest.raises(ValueError, match="Seed"):
        game.reset(seed)  # type: ignore[arg-type]
    assert game.state is state
    assert game.get_rewards() == rewards
    assert game._rng is rng
    assert rng.getstate() == rng_state


def test_instances_reset_independently() -> None:
    first, second = ResetHarness(), ResetHarness()
    first.reset(42)
    second.reset(42)
    assert first.state == second.state
    assert first.state is not second.state
    first.dirty_state()
    first.reset(9)
    assert second.state.data.board == (None,) * 9
    assert first._rng is not second._rng
