"""Immutable authoritative board data for Tic-Tac-Toe."""

from dataclasses import dataclass
from typing import Literal

type Cell = Literal[0, 1] | None
type Board = tuple[Cell, Cell, Cell, Cell, Cell, Cell, Cell, Cell, Cell]


@dataclass(frozen=True, slots=True, kw_only=True)
class TicTacToeState:
    """Nine row-major cells: None is empty, seat 0 is X, and seat 1 is O.

    Actor and phase belong to the enclosing GameState, avoiding duplicate data.
    This authoritative payload is not an agent observation.
    """

    board: Board = (None, None, None, None, None, None, None, None, None)
