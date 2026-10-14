"""Clone independence and shared conformance for the production engine."""

import pytest

from boardgame_ai.core import Action, GameNotReadyError, GameOverError
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from tests.contracts import assert_engine_contract, assert_rejection_preserves_game
from tests.test_tic_tac_toe_actions import place


def test_production_engine_contract() -> None:
    assert_engine_contract(TicTacToeEngine, max_steps=9)


@pytest.mark.parametrize("moves", [(), (4,), (4, 0)])
def test_rejection_conformance(moves: tuple[int, ...]) -> None:
    game = TicTacToeEngine()
    game.reset(42)
    for cell in moves:
        game.step(place(cell))
    invalid = place(moves[0]) if moves else Action(kind="pass")
    assert_rejection_preserves_game(game, invalid, max_steps=9)


def test_clone_requires_reset() -> None:
    with pytest.raises(GameNotReadyError):
        TicTacToeEngine().clone()


def test_divergent_moves_and_resets_are_independent() -> None:
    game = TicTacToeEngine()
    game.reset(42)
    game.step(place(4))
    cloned = game.clone()
    assert type(cloned) is TicTacToeEngine
    assert cloned is not game
    original = game.observe(0)
    cloned.step(place(0))
    assert game.observe(0) == original
    game.step(place(8))
    assert game.observe(0).data.board[0] is None
    assert cloned.observe(0).data.board[8] is None
    assert game.current_player() == cloned.current_player() == 0
    clone_view = cloned.observe(0)
    game.reset(99)
    assert cloned.observe(0) == clone_view
    game.step(place(2))
    original = game.observe(0)
    cloned.reset(7)
    assert game.observe(0) == original


@pytest.mark.parametrize(
    "moves",
    [(0, 3, 1, 4, 2), (0, 3, 1, 4, 8, 5), (0, 1, 2, 4, 3, 5, 7, 6, 8)],
)
def test_clones_follow_identical_continuations_and_preserve_terminal_results(
    moves: tuple[int, ...],
) -> None:
    game = TicTacToeEngine()
    game.reset(42)
    for cell in moves:
        cloned = game.clone()
        before = game.observe(0)
        clone_result = cloned.step(place(cell))
        assert game.observe(0) == before
        assert game.step(place(cell)) == clone_result
        for seat in (0, 1):
            assert game.decision_for(seat) == cloned.decision_for(seat)
            assert game.render(seat) == cloned.render(seat)
    terminal = game.clone()
    assert terminal.is_terminal()
    assert terminal.current_player() is None
    assert terminal.get_rewards() == game.get_rewards()
    with pytest.raises(GameOverError):
        terminal.step(Action(kind="pass"))
    terminal_view, rewards = terminal.observe(0), terminal.get_rewards()
    game.reset()
    assert terminal.observe(0) == terminal_view
    assert terminal.get_rewards() == rewards


def test_clone_rng_is_independent_preserves_sequence_and_concrete_subclass() -> None:
    class TaggedEngine(TicTacToeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.tags = ["original"]

    game = TaggedEngine()
    game.reset(42)
    rng = game._rng
    assert rng is not None
    rng.random()  # Clone a progressed RNG, not just its initial seed.
    before = rng.getstate()
    cloned = game.clone()
    assert type(cloned) is TaggedEngine
    assert cloned._rng is not None
    assert cloned._rng is not rng
    assert cloned._rng.getstate() == rng.getstate() == before
    cloned.tags.append("clone")
    assert game.tags == ["original"]
    value = cloned._rng.random()
    assert rng.getstate() == before
    assert rng.random() == value
    assert cloned._rng.getstate() == rng.getstate()
