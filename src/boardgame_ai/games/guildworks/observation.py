"""Immutable filtered Guildworks records, independent of engine implementation."""

from dataclasses import dataclass

from boardgame_ai.core import GameEvent


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildworksObservation:
    """Same visibility for active, inactive and terminal observers."""

    round: int
    placements: tuple[int | None, ...]
    workers: tuple[int, ...]
    ink: tuple[int, ...]
    scores: tuple[int, ...]
    own_hand: tuple[int, ...]
    hand_counts: tuple[int, ...]
    deck_count: int
    discard: tuple[int, ...]
    history: tuple[GameEvent, ...]
