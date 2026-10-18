"""Framework-free public transfer-game observation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class BreakthroughObservation:
    board: tuple[int | None, ...]
