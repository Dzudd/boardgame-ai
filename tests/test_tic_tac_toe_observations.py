"""Public views, immutable snapshots, and text rendering."""

from dataclasses import FrozenInstanceError, asdict

import pytest

from boardgame_ai.core import GameNotReadyError, InvalidPlayerError
from boardgame_ai.games.tic_tac_toe import render_board
from tests.test_tic_tac_toe_actions import place
from tests.test_tic_tac_toe_reset import ResetHarness


def test_observation_lifecycle_and_public_schema() -> None:
    game = ResetHarness()
    with pytest.raises(GameNotReadyError):
        game.observe(0)
    with pytest.raises(GameNotReadyError):
        game.render(0)
    game.reset(42)
    for player in (0, 1):
        observation = game.observe(player)
        assert observation.player_id == player
        assert observation.current_player == 0
        assert observation.phase == "play"
        assert observation.data.board == (None,) * 9
        assert set(asdict(observation)) == {
            "player_id",
            "current_player",
            "phase",
            "data",
        }
        assert set(asdict(observation.data)) == {"board"}


@pytest.mark.parametrize("player", [-1, 2, True, False, 0.0, "0", None])
def test_observe_and_render_reject_invalid_seats(player: object) -> None:
    game = ResetHarness()
    game.reset()
    for terminal in (False, True):
        if terminal:
            game.dirty_state()
        with pytest.raises(InvalidPlayerError):
            game.observe(player)  # type: ignore[arg-type]
        with pytest.raises(InvalidPlayerError):
            game.render(player)  # type: ignore[arg-type]


def test_views_are_fresh_immutable_and_survive_moves_and_reset() -> None:
    game = ResetHarness()
    game.reset()
    previous = game.observe(1)
    fresh = game.observe(1)
    assert previous == fresh
    assert previous is not fresh
    assert previous.data is not fresh.data
    with pytest.raises(FrozenInstanceError):
        previous.data.board = (None,) * 9  # type: ignore[misc]
    with pytest.raises(TypeError):
        previous.data.board[0] = 1  # type: ignore[index]
    game.step(place(4))
    current = game.observe(0)
    assert current.current_player == 1
    assert current.data.board[4] == 0
    assert previous.data.board == (None,) * 9
    game.reset()
    assert current.data.board[4] == 0
    assert game.observe(0).data.board == (None,) * 9


@pytest.mark.parametrize(
    "moves",
    [(0, 3, 1, 4, 2), (0, 1, 2, 4, 3, 5, 7, 6, 8)],
)
def test_terminal_observations_and_decisions_for_both_seats(
    moves: tuple[int, ...],
) -> None:
    game = ResetHarness()
    game.reset()
    for cell in moves:
        game.step(place(cell))
    for player in (0, 1):
        observation, actions = game.decision_for(player)
        assert observation == game.observe(player)
        assert observation.current_player is None
        assert observation.phase == "finished"
        assert observation.data.board == game.state.data.board
        assert actions == ()


def test_rendering_and_decision_queries_are_pure_and_seed_independent() -> None:
    game, other = ResetHarness(), ResetHarness()
    game.reset(42)
    other.reset(99)
    empty = ". | . | .\n---------\n. | . | .\n---------\n. | . | ."
    assert game.render(0) == game.render(1) == empty
    for cell in (4, 0, 8):
        game.step(place(cell))
        other.step(place(cell))
    state, rewards, rng = game.state, game.get_rewards(), game._rng
    assert rng is not None
    rng_state = rng.getstate()
    for player in (0, 1):
        observation, actions = game.decision_for(player)
        assert (observation, actions) == other.decision_for(player)
        assert actions == game.legal_actions(player)
        expected = "O | . | .\n---------\n. | X | .\n---------\n. | . | X"
        assert game.render(player) == render_board(observation) == expected
    assert game.state is state
    assert game.get_rewards() == rewards
    assert rng.getstate() == rng_state
