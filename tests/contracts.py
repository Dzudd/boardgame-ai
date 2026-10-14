"""Reusable bounded conformance checks; future games supply their own factory.

These exercise a deterministic first-legal-action path, not all reachable states.
Game-specific correctness and visibility tests are still required.
"""

from collections.abc import Callable

import pytest

from boardgame_ai.core import (
    Action,
    GameEngine,
    GameNotReadyError,
    GameOverError,
    IllegalActionError,
    InvalidPlayerError,
)


def assert_engine_contract[Data](
    factory: Callable[[], GameEngine[Data]], *, max_steps: int
) -> None:
    game = factory()
    assert game.spec.min_players <= game.num_players <= game.spec.max_players
    for query in (
        game.current_player,
        game.is_terminal,
        game.get_rewards,
        game.clone,
        lambda: game.observe(0),
        lambda: game.legal_actions(0),
        lambda: game.step(Action(kind="unavailable")),
    ):
        with pytest.raises(GameNotReadyError):
            query()

    game.reset(seed=123)
    assert game.get_rewards() == (0.0,) * game.num_players
    for invalid_player in (-1, game.num_players, True):
        with pytest.raises(InvalidPlayerError):
            game.observe(invalid_player)
        with pytest.raises(InvalidPlayerError):
            game.legal_actions(invalid_player)

    twin = factory()
    twin.reset(seed=123)
    for _ in range(max_steps):
        for seat in range(game.num_players):
            assert game.observe(seat) == twin.observe(seat)
            assert game.legal_actions(seat) == twin.legal_actions(seat)
        assert game.current_player() == twin.current_player()
        assert game.is_terminal() == twin.is_terminal()
        if game.is_terminal():
            break
        player = game.current_player()
        assert player is not None
        observation, legal = game.decision_for(player)
        assert observation.player_id == observation.current_player == player
        assert legal
        for seat in range(game.num_players):
            if seat != player:
                assert game.legal_actions(seat) == ()
        cloned = game.clone()
        before = tuple(game.observe(seat) for seat in range(game.num_players))
        clone_result = cloned.step(legal[0])
        assert tuple(game.observe(seat) for seat in range(game.num_players)) == before
        result = game.step(legal[0])
        assert result == clone_result == twin.step(legal[0])
        assert result.next_player == game.current_player()
        assert result.terminated == game.is_terminal()
        assert result.rewards == game.get_rewards()
        assert len(result.rewards) == game.num_players
    assert game.is_terminal(), "Game exceeded the test's step bound"
    assert game.current_player() is None
    for seat in range(game.num_players):
        assert game.legal_actions(seat) == ()
    with pytest.raises(GameOverError):
        game.step(Action(kind="unavailable"))
    game.reset(seed=123)
    fresh = factory()
    fresh.reset(seed=123)
    assert game.get_rewards() == (0.0,) * game.num_players
    for seat in range(game.num_players):
        assert game.observe(seat) == fresh.observe(seat)


def assert_rejection_preserves_game[Data](
    game: GameEngine[Data], invalid_action: Action, *, max_steps: int
) -> None:
    """Compare future outcomes with a clone to catch consumed RNG on rejection."""
    before = game.clone()
    rewards = game.get_rewards()
    with pytest.raises(IllegalActionError):
        game.step(invalid_action)
    assert game.get_rewards() == rewards
    for _ in range(max_steps):
        for seat in range(game.num_players):
            assert game.observe(seat) == before.observe(seat)
            assert game.legal_actions(seat) == before.legal_actions(seat)
        assert game.is_terminal() == before.is_terminal()
        player = game.current_player()
        assert player == before.current_player()
        if player is None:
            return
        action = game.legal_actions(player)[0]
        assert game.step(action) == before.step(action)
    assert game.is_terminal(), "Game exceeded the test's step bound"
