"""Simple Tic-Tac-Toe tactics derived exclusively from the public board."""

from boardgame_ai.agents.base import Agent
from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation

_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)
_PREFERENCE = (4, 0, 2, 6, 8, 1, 3, 5, 7)


class TicTacToeHeuristicAgent(Agent[TicTacToeObservation]):
    """Win, block, then prefer center, corners, and edges in row-major order.

    Deterministic and stateless: inherited reset is a no-op, regardless of seed.
    This tactical baseline does not handle forks or guarantee optimal play.
    """

    def select_action(
        self,
        observation: Observation[TicTacToeObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        """Return a supplied placement for the observing actor's public board."""
        if not legal_actions:
            raise ValueError("TicTacToeHeuristicAgent requires legal actions")
        player = observation.player_id
        if type(player) is not int or player not in (0, 1):
            raise ValueError("TicTacToeHeuristicAgent requires seat 0 or 1")
        if observation.current_player != player:
            raise ValueError("TicTacToeHeuristicAgent requires the actor's observation")

        board = observation.data.board
        candidates: list[tuple[int, Action]] = []
        for action in legal_actions:
            if (
                action.kind != "place"
                or len(action.parameters) != 1
                or action.parameters[0][0] != "cell"
            ):
                raise ValueError("TicTacToeHeuristicAgent requires place/cell actions")
            cell = action.parameters[0][1]
            if type(cell) is not int or not 0 <= cell < 9 or board[cell] is not None:
                raise ValueError("TicTacToeHeuristicAgent requires empty board cells")
            candidates.append((cell, action))
        candidates.sort(key=lambda candidate: candidate[0])

        for mark in (player, 1 - player):
            for cell, action in candidates:
                if any(
                    cell in line
                    and all(index == cell or board[index] == mark for index in line)
                    for line in _LINES
                ):
                    return action
        return min(candidates, key=lambda candidate: _PREFERENCE.index(candidate[0]))[1]
