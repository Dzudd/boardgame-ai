"""Terminal public-board outcome adapter."""

from boardgame_ai.core import GameEngine
from boardgame_ai.games.connect_four.engine import winner
from boardgame_ai.games.connect_four.observation import ConnectFourObservation
from boardgame_ai.simulation.evaluation import GameOutcome


def connect_four_outcome(game: GameEngine[ConnectFourObservation]) -> GameOutcome:
    if game.spec.game_id != "connect_four" or game.num_players != 2:
        raise ValueError("Connect Four outcome requires a two-seat Connect Four game")
    if not game.is_terminal():
        raise ValueError("Outcome requires a terminal game")
    board = game.observe(0).data.board
    seat = winner(board)
    if seat is not None:
        return GameOutcome(
            scores=tuple(1.0 if p == seat else -1.0 for p in (0, 1)),
            winners=(seat,),
        )
    if None in board:
        raise ValueError("Terminal board has no result")
    return GameOutcome(scores=(0.0, 0.0), draw=True)
