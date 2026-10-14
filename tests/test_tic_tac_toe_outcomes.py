"""Game-ending trajectories, rewards, and terminal lifecycle behavior."""

import pytest

from boardgame_ai.core import Action, GameNotReadyError, GameOverError
from tests.test_tic_tac_toe_actions import place
from tests.test_tic_tac_toe_reset import ResetHarness


@pytest.mark.parametrize("winner", [0, 1])
@pytest.mark.parametrize(
    "line",
    [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ],
)
def test_each_winning_line_for_both_players(
    line: tuple[int, int, int], winner: int
) -> None:
    outside = [cell for cell in range(9) if cell not in line]
    moves: tuple[int, ...]
    if winner == 0:
        moves = (line[0], outside[0], line[1], outside[1], line[2])
    else:
        moves = (outside[0], line[0], outside[1], line[1], outside[-1], line[2])
    game = ResetHarness()
    game.reset(42)
    for turn, cell in enumerate(moves):
        assert not game.is_terminal()
        assert game.current_player() == turn % 2
        result = game.step(place(cell))
        assert result.terminated == game.is_terminal() == (turn == len(moves) - 1)
        if not result.terminated:
            assert result.rewards == game.get_rewards() == (0.0, 0.0)
            assert result.next_player is not None
            assert game.legal_actions(result.next_player)
    assert result.rewards == ((1.0, -1.0) if winner == 0 else (-1.0, 1.0))
    assert result.rewards == game.get_rewards()
    assert result.next_player is game.current_player() is None
    assert game.state.phase == "finished"
    assert game.legal_actions(0) == game.legal_actions(1) == ()


@pytest.mark.parametrize(
    ("moves", "rewards"),
    [
        ((0, 1, 2, 4, 3, 5, 7, 6, 8), (0.0, 0.0)),  # Full-board draw.
        ((0, 1, 2, 3, 4, 5, 7, 6, 8), (1.0, -1.0)),  # Ninth-move diagonal win.
    ],
)
def test_full_board_outcomes(
    moves: tuple[int, ...], rewards: tuple[float, float]
) -> None:
    game = ResetHarness()
    game.reset()
    for turn, cell in enumerate(moves):
        result = game.step(place(cell))
        assert result.terminated == (turn == 8)
    assert all(cell is not None for cell in game.state.data.board)
    assert game.is_terminal()
    assert game.current_player() is None
    assert game.state.phase == "finished"
    assert result.rewards == game.get_rewards() == rewards
    assert game.legal_actions(0) == game.legal_actions(1) == ()


def test_terminal_queries_rejection_and_reset_preserve_reward_semantics() -> None:
    game = ResetHarness()
    with pytest.raises(GameNotReadyError):
        game.is_terminal()
    game.reset(42)
    rng = game._rng
    assert rng is not None
    rng_state = rng.getstate()
    for cell in (0, 3, 1, 4, 2):
        game.step(place(cell))
    terminal_state = game.state
    for _ in range(2):
        assert game.is_terminal()
        assert game.current_player() is None
        assert game.get_rewards() == (1.0, -1.0)
        for action in (place(8), place(0), Action(kind="pass")):
            with pytest.raises(GameOverError):
                game.step(action)
        assert game.state is terminal_state
        assert game.get_rewards() == (1.0, -1.0)
        assert rng.getstate() == rng_state
    game.reset(42)
    assert not game.is_terminal()
    assert game.current_player() == 0
    assert game.state.phase == "play"
    assert game.state.data.board == (None,) * 9
    assert game.get_rewards() == (0.0, 0.0)
    assert len(game.legal_actions(0)) == 9
