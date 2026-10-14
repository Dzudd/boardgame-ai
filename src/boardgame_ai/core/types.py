"""Framework-independent records shared by game engines and agents.

Player IDs are zero-based seat indices. Frozen records prevent field reassignment,
but game-specific payloads must be copied or made immutable by their engine.
"""

from dataclasses import dataclass
from math import isfinite

type PlayerId = int
type Scalar = str | int | float | bool | None
type Parameters = tuple[tuple[str, Scalar], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class GameState[StateData]:
    """Authoritative engine snapshot, including any private game data.

    Never pass this record to an agent. RNG state and cloning are engine concerns.
    """

    current_player: PlayerId | None
    phase: str
    data: StateData


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation[ObservationData]:
    """A player's filtered view, including only legitimately visible history.

    The observer need not be the current player. No reference to GameState is
    included; the engine must build an independent, safe payload.
    """

    player_id: PlayerId
    current_player: PlayerId | None
    phase: str
    data: ObservationData


@dataclass(frozen=True, slots=True, kw_only=True)
class Action:
    """One atomic choice, without unresolved chance outcomes.

    Named scalar parameters keep the initial format small and JSON-compatible.
    Legality, parameter schemas, and the acting player are validated by the engine.
    """

    kind: str
    parameters: Parameters = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("Action kind must not be empty")
        names = [name for name, _ in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Action parameter names must be unique")
        for _, value in self.parameters:
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError("Action parameters must be scalar values")
            if isinstance(value, float) and not isfinite(value):
                raise ValueError("Action parameters must be finite")


@dataclass(frozen=True, slots=True, kw_only=True)
class GameEvent:
    """A game-defined event; None recipients means public, a tuple means private.

    Empty recipients exposes the event to no player. Payloads use the same scalar
    format as actions. Full event lists are engine-facing, not agent-facing.
    """

    kind: str
    parameters: Parameters = ()
    recipients: tuple[PlayerId, ...] | None = None

    def visible_to(self, player_id: PlayerId) -> bool:
        """Whether this event is intended for the specified player."""
        return self.recipients is None or player_id in self.recipients


@dataclass(frozen=True, slots=True, kw_only=True)
class StepResult:
    """Engine-facing outcome of one decision after automatic chance resolution.

    Rewards are incremental for this step, indexed by player seat. Engines must
    supply one reward per player. No zero-sum or two-player assumption is made.
    A nonterminal result always identifies the next decision-maker, which can be
    the player who just acted. Filter events before delivering them to agents.
    """

    next_player: PlayerId | None
    terminated: bool
    rewards: tuple[float, ...]
    events: tuple[GameEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.terminated != (self.next_player is None):
            raise ValueError("Only terminal results must have no next player")
        if not self.rewards or not all(isfinite(reward) for reward in self.rewards):
            raise ValueError("Rewards must be nonempty and finite")
        if self.next_player is not None and not 0 <= self.next_player < len(
            self.rewards
        ):
            raise ValueError("Next player must index the rewards tuple")

    def events_for(self, player_id: PlayerId) -> tuple[GameEvent, ...]:
        """Return only public events and private events for this player."""
        if not 0 <= player_id < len(self.rewards):
            raise ValueError("Player must index the rewards tuple")
        return tuple(event for event in self.events if event.visible_to(player_id))


@dataclass(frozen=True, slots=True, kw_only=True)
class GameSpec:
    """Stable game identifier and supported player counts, without ML details."""

    game_id: str
    name: str
    min_players: int
    max_players: int

    def __post_init__(self) -> None:
        if not self.game_id or not self.name:
            raise ValueError("Game identifier and name must not be empty")
        if not 1 <= self.min_players <= self.max_players:
            raise ValueError("Player counts must satisfy 1 <= min <= max")
