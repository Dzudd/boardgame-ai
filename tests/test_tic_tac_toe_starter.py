"""Configured O-first gameplay, reset persistence, and independent clones."""

from typing import Any

import pytest

from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from tests.test_tic_tac_toe_actions import place


@pytest.mark.parametrize(
    "moves, rewards",
    [
        ((0, 3, 1, 4, 2), (-1.0, 1.0)),
        ((0, 3, 1, 4, 8, 5), (1.0, -1.0)),
        ((0, 1, 2, 4, 3, 5, 7, 6, 8), (0.0, 0.0)),
    ],
)
def test_o_first_complete_games(
    moves: tuple[int, ...],
    rewards: tuple[float, ...],
) -> None:
    game = TicTacToeEngine(starting_player=1)
    game.reset(42)
    for index, cell in enumerate(moves):
        actor = 1 - index % 2
        assert game.current_player() == actor
        observation, legal = game.decision_for(actor)
        assert observation.current_player == actor and place(cell) in legal
        game.step(place(cell))
        assert game.observe(actor).data.board[cell] == actor
    assert game.is_terminal()
    assert game.get_rewards() == rewards
    assert game.current_player() is None
    assert game.legal_actions(0) == game.legal_actions(1) == ()


def test_clone_and_reset_preserve_starter_configuration() -> None:
    game = TicTacToeEngine(starting_player=1)
    game.reset(42)
    game.step(place(4))
    cloned = game.clone()
    assert cloned.observe(0) == game.observe(0)
    cloned.reset(99)
    assert cloned.current_player() == 1
    assert game.observe(0).data.board[4] == 1
    game.reset(99)
    assert game.observe(0) == cloned.observe(0)
    assert game.get_rewards() == (0.0, 0.0)


@pytest.mark.parametrize("starter", [True, False, -1, 2, 1.0, "O", None])
def test_invalid_starting_player(starter: Any) -> None:
    with pytest.raises(ValueError, match="Starting player"):
        TicTacToeEngine(starting_player=starter)
