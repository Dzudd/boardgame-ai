"""Human board display; column labels match terminal input."""

from boardgame_ai.core import Observation
from boardgame_ai.games.connect_four.observation import (
    COLUMNS,
    ROWS,
    ConnectFourObservation,
)


def render_board(observation: Observation[ConnectFourObservation]) -> str:
    symbols = {None: ".", 0: "X", 1: "O"}
    return "\n".join(
        [
            "| "
            + " ".join(
                symbols[cell]
                for cell in observation.data.board[r * COLUMNS : (r + 1) * COLUMNS]
            )
            + " |"
            for r in range(ROWS)
        ]
        + ["-----------------", "| 1 2 3 4 5 6 7 |"]
    )
