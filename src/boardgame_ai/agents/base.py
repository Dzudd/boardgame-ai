"""Agent contract without access to authoritative engine state."""

from abc import ABC, abstractmethod

from boardgame_ai.core.types import Action, Observation


class Agent[ObservationData](ABC):
    """Choose actions using only a player view and its legal choices.

    Use one instance per player seat so future recurrent memory stays private.
    A runner calls reset before each game and select_action only for the actor.
    """

    def reset(self, seed: int | None = None) -> None:
        """Clear per-game memory and optionally seed agent-owned randomness.

        Stateless agents may keep this no-op implementation. Stateful agents
        override it; the engine's random generator is never shared with agents.
        """
        return None

    @abstractmethod
    def select_action(
        self,
        observation: Observation[ObservationData],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        """Choose a supplied legal action; the caller provides nonempty choices.

        The caller must give this player's observation at its decision point.
        Engines still validate the returned action before changing state.
        """
