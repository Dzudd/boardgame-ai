"""Terminal outcome adapter from Tic-Tac-Toe's public board."""

from boardgame_ai.core import GameEngine
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation
from boardgame_ai.simulation.evaluation import GameOutcome


def tic_tac_toe_outcome(game: GameEngine[TicTacToeObservation]) -> GameOutcome:
    """Use +1 winner/-1 loser, zero draw as Tic-Tac-Toe result scores."""
    if game.spec.game_id != "tic_tac_toe" or game.num_players != 2:
        raise ValueError("Tic-Tac-Toe outcome requires a two-seat Tic-Tac-Toe game")
    if not game.is_terminal():
        raise ValueError("Outcome requires a terminal game")
    board = game.observe(0).data.board
    lines = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )
    for seat in (0, 1):
        if any(all(board[cell] == seat for cell in line) for line in lines):
            return GameOutcome(
                scores=(1.0, -1.0) if seat == 0 else (-1.0, 1.0), winners=(seat,)
            )
    if None in board:
        raise ValueError("Terminal board has no result")
    return GameOutcome(scores=(0.0, 0.0), draw=True)
