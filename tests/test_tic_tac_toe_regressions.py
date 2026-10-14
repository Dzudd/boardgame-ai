"""Reachable-state rule audit and deterministic public-API replay regressions."""

from collections import Counter

import pytest

from boardgame_ai.core import Action, GameOverError, IllegalActionError
from tests.test_tic_tac_toe_actions import place
from tests.test_tic_tac_toe_reset import ResetHarness


def outcome(board: tuple[int | None, ...]) -> int | None:
    """Independent coordinate-based oracle; -1 denotes a draw."""
    for player in (0, 1):
        occupied = {
            (index // 3, index % 3)
            for index, value in enumerate(board)
            if value == player
        }
        if (
            any(
                sum(row == coordinate for row, _ in occupied) == 3
                for coordinate in range(3)
            )
            or any(
                sum(column == coordinate for _, column in occupied) == 3
                for coordinate in range(3)
            )
            or sum(row == column for row, column in occupied) == 3
            or sum(row + column == 2 for row, column in occupied) == 3
        ):
            return player
    return -1 if None not in board else None


def test_all_distinct_reachable_positions_match_independent_rules() -> None:
    visited: set[tuple[int | None, ...]] = set()
    endings: Counter[int] = Counter()

    def visit(moves: tuple[int, ...]) -> None:
        game = ResetHarness()
        game.reset(42)
        for cell in moves:
            result = game.step(place(cell))
            assert result.next_player == game.current_player()
            assert result.terminated == game.is_terminal()
            assert result.rewards == game.get_rewards()
        board = game.observe(0).data.board
        if board in visited:
            return
        visited.add(board)
        expected = outcome(board)
        terminal = expected is not None
        assert game.is_terminal() == terminal
        assert board.count(0) == (len(moves) + 1) // 2
        assert board.count(1) == len(moves) // 2
        rewards = (
            (1.0, -1.0)
            if expected == 0
            else ((-1.0, 1.0) if expected == 1 else (0.0, 0.0))
        )
        assert game.get_rewards() == rewards
        for seat in (0, 1):
            observation, legal = game.decision_for(seat)
            assert observation.data.board == board
            assert observation.phase == ("finished" if terminal else "play")
            assert observation.current_player == (None if terminal else len(moves) % 2)
            cells = tuple(index for index, value in enumerate(board) if value is None)
            assert legal == (
                tuple(place(cell) for cell in cells)
                if not terminal and seat == len(moves) % 2
                else ()
            )
        if terminal:
            assert expected is not None
            endings[expected] += 1
            with pytest.raises(GameOverError):
                game.step(Action(kind="pass"))
            return
        assert len(moves) < 9
        for cell, value in enumerate(board):
            if value is None:
                visit((*moves, cell))

    visit(())
    # Counts of distinct boards, not move sequences or symmetry classes.
    assert len(visited) == 5478
    assert endings == {0: 626, 1: 316, -1: 16}


@pytest.mark.parametrize("seed", [None, 0, 42, -7])
@pytest.mark.parametrize(
    "moves",
    [(0, 3, 1, 4, 2), (0, 3, 1, 4, 8, 5), (0, 1, 2, 4, 3, 5, 7, 6, 8)],
)
def test_complete_replays_after_rejection_and_reset(
    seed: int | None, moves: tuple[int, ...]
) -> None:
    game, reference = ResetHarness(), ResetHarness()
    game.reset(seed)
    reference.reset(seed)
    for replay in range(2):
        if replay:
            game.reset(seed)
            reference.reset(seed)
        for turn, cell in enumerate(moves):
            for action in (Action(kind="pass"), place(-1), place(moves[0])):
                if turn == 0 and action == place(moves[0]):
                    continue  # This cell is occupied only after the first move.
                before = tuple(game.decision_for(seat) for seat in (0, 1))
                rewards = game.get_rewards()
                with pytest.raises(IllegalActionError):
                    game.step(action)
                assert tuple(game.decision_for(seat) for seat in (0, 1)) == before
                assert game.get_rewards() == rewards
            assert game.step(place(cell)) == reference.step(place(cell))
            for seat in (0, 1):
                assert game.decision_for(seat) == reference.decision_for(seat)
                assert game.render(seat) == reference.render(seat)
        assert game.is_terminal() and reference.is_terminal()
        assert game.get_rewards() == reference.get_rewards()
