"""Stable engine errors; messages must never include private game data."""


class IllegalActionError(ValueError):
    """A well-formed action is not legal at the current decision point."""


class GameOverError(IllegalActionError):
    """An action was attempted after termination."""


class InvalidPlayerError(ValueError):
    """A player ID is not a seat in this game."""


class GameNotReadyError(RuntimeError):
    """Gameplay was queried before reset initialized the engine."""
