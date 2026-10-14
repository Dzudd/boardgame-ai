"""Text rendering from public observations only."""

from boardgame_ai.core import Observation
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation


def render_board(observation: Observation[TicTacToeObservation]) -> str:
    """Return three rows separated by rules; dots represent empty cells.

    This pure formatter does not print or inspect an engine's private state.
    """
    symbols = {None: ".", 0: "X", 1: "O"}
    cells = [symbols[cell] for cell in observation.data.board]
    return "\n---------\n".join(
        " | ".join(cells[start : start + 3]) for start in (0, 3, 6)
    )
