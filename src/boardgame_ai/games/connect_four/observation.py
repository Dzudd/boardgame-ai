"""Framework-free public Connect Four records; rows run top to bottom."""

from dataclasses import dataclass

ROWS = 6
COLUMNS = 7
MAX_STEPS = ROWS * COLUMNS
type Board = tuple[int | None, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ConnectFourObservation:
    board: Board
    # Zero-based columns in execution order. All moves are public.
    history: tuple[int, ...]
