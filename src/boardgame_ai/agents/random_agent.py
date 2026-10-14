"""Uniform random baseline using only the supplied legal choices."""

from boardgame_ai.agents.base import Agent
from boardgame_ai.core import Action, Observation, create_rng


class RandomAgent[ObservationData](Agent[ObservationData]):
    """Choose uniformly among tuple entries with independently owned randomness.

    Reset before each game. Reproduction requires the same seed, ordered legal
    choices, selection calls, and Python version. Observations are not inspected.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = create_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """Replace the private generator after validating the seed."""
        self._rng = create_rng(seed)

    def select_action(
        self,
        observation: Observation[ObservationData],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        """Return a supplied action, or reject empty choices without an RNG draw."""
        if not legal_actions:
            raise ValueError("RandomAgent requires at least one legal action")
        return self._rng.choice(legal_actions)
