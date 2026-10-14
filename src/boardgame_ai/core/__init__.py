"""Shared game contracts with no numerical framework dependencies."""

from boardgame_ai.core.errors import (
    GameNotReadyError,
    GameOverError,
    IllegalActionError,
    InvalidPlayerError,
)
from boardgame_ai.core.game import GameEngine
from boardgame_ai.core.randomness import create_rng
from boardgame_ai.core.types import (
    Action,
    GameEvent,
    GameSpec,
    GameState,
    Observation,
    Parameters,
    PlayerId,
    Scalar,
    StepResult,
)

__all__ = [
    "Action",
    "GameEvent",
    "GameNotReadyError",
    "GameOverError",
    "GameEngine",
    "GameSpec",
    "GameState",
    "IllegalActionError",
    "InvalidPlayerError",
    "Observation",
    "Parameters",
    "PlayerId",
    "Scalar",
    "StepResult",
    "create_rng",
]
