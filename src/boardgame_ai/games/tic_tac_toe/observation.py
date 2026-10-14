"""Public player views, separate from authoritative game state."""

from dataclasses import dataclass

from boardgame_ai.games.tic_tac_toe.state import Board


@dataclass(frozen=True, slots=True, kw_only=True)
class TicTacToeObservation:
    """Public board: row-major cells, None empty, seat 0 X, seat 1 O."""

    board: Board
