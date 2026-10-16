"""Learning components, separate from framework-independent game engines."""

from boardgame_ai.models.connect_four import ConnectFourActorCritic
from boardgame_ai.models.tic_tac_toe import TicTacToeActorCritic

__all__ = ["ConnectFourActorCritic", "TicTacToeActorCritic"]
