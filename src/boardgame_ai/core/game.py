"""Abstract contract for authoritative game engines."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Self

from boardgame_ai.core.errors import InvalidPlayerError
from boardgame_ai.core.types import (
    Action,
    GameSpec,
    Observation,
    PlayerId,
    StepResult,
)


class GameEngine[ObservationData](ABC):
    """Execute one atomic decision at a time, resolving chance internally.

    Call reset before gameplay. State representation and storage belong to each
    concrete engine; no full-state accessor is part of this contract.
    Before reset, gameplay queries and clone raise GameNotReadyError; metadata
    remains available. Invalid seats raise InvalidPlayerError after reset.
    """

    @property
    @abstractmethod
    def spec(self) -> GameSpec:
        """Game metadata describing supported player counts."""

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Actual player count for this instance, within spec's supported range."""

    def validate_player(self, player_id: PlayerId) -> None:
        """Reject IDs outside this instance's seats, including boolean IDs."""
        if (
            isinstance(player_id, bool)
            or not isinstance(player_id, int)
            or not 0 <= player_id < self.num_players
        ):
            raise InvalidPlayerError("Invalid player seat")

    def decision_for(
        self, player_id: PlayerId
    ) -> tuple[Observation[ObservationData], tuple[Action, ...]]:
        """Detach the agent inputs from engine-owned mutable objects.

        Callers pass these two values to select_action, never the engine or a
        full StepResult. Copying prevents aliasing, not incorrectly included
        secrets: observe and legal_actions must already respect visibility.
        """
        self.validate_player(player_id)
        return deepcopy((self.observe(player_id), self.legal_actions(player_id)))

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Start a fresh game, clearing rewards and resolving setup chance.

        An explicit seed supports reproducible setup and subsequent chance.
        Use an engine-owned generator; None creates fresh system-seeded randomness.
        Validate seed/configuration before replacing any existing game state.
        Reset may produce a terminal game; otherwise a next actor must exist.
        """

    @abstractmethod
    def observe(self, player_id: PlayerId) -> Observation[ObservationData]:
        """Return an independent view containing only this player's information.

        Valid for any seat, including inactive players and after termination.
        Reading observations must not advance the game or consume randomness.
        """

    @abstractmethod
    def legal_actions(self, player_id: PlayerId) -> tuple[Action, ...]:
        """Return legal atomic choices, without advancing state or chance.

        Return an empty tuple for inactive players and terminal games. A live
        decision must offer at least one action, including pass when appropriate.
        """

    @abstractmethod
    def current_player(self) -> PlayerId | None:
        """Next decision-maker, or None precisely when the game is terminal."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Validate and execute one current-player choice and automatic effects.

        The result's next actor and termination must agree with engine queries.
        Include one incremental reward per seat. The same player may act again.
        Full results may contain private events and must not be passed to agents.
        Reject terminal steps with GameOverError, other illegal choices with
        IllegalActionError. Validate before mutation or RNG draws: rejection must
        preserve state, phase, rewards, events, and RNG state. Error messages must
        not disclose private state. Structurally invalid Action construction
        raises ValueError before an engine call.
        """

    @abstractmethod
    def is_terminal(self) -> bool:
        """Whether no further player decisions remain."""

    @abstractmethod
    def get_rewards(self) -> tuple[float, ...]:
        """Latest step's incremental rewards, or one zero per seat after reset.

        This is a read-only query, not cumulative scores or a reward accumulator.
        Repeated calls do not consume or reissue rewards.
        """

    @abstractmethod
    def clone(self) -> Self:
        """Copy the game independently, including mutable state and RNG state.

        Identical subsequent choices must have identical outcomes until the
        original and clone diverge. Cloning must not advance either RNG.
        """
