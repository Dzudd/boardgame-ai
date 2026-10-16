"""Standard-library rules; authoritative secrets never enter player inputs."""

from copy import deepcopy
from typing import Self

from boardgame_ai.core import (
    Action,
    GameEngine,
    GameEvent,
    GameNotReadyError,
    GameOverError,
    GameSpec,
    IllegalActionError,
    Observation,
    StepResult,
    create_rng,
)
from boardgame_ai.games.guildworks.observation import GuildworksObservation

SPACES = ("inkwell", "archive", "festival", "lottery")
ROUNDS = 4
MAX_STEPS = 32


class GuildworksEngine(GameEngine[GuildworksObservation]):
    """Four rounds, four placements per round, then one craft/pass per placement.

    Each round starts with seat (round-1)%2 and alternates placements. Two
    workers each guarantees a free space whenever a placement decision exists.
    Card values are interchangeable; action ordering never depends on deck order.
    """

    def __init__(self) -> None:
        self._ready = False

    @property
    def spec(self) -> GameSpec:
        return GameSpec(
            game_id="guildworks", name="Guildworks", min_players=2, max_players=2
        )

    @property
    def num_players(self) -> int:
        return 2

    def _require_ready(self) -> None:
        if not self._ready:
            raise GameNotReadyError("Call reset before gameplay")

    def reset(self, seed: int | None = None) -> None:
        rng = create_rng(seed)
        deck = [1, 2, 3] * 4
        rng.shuffle(deck)
        self._rng = rng
        self._deck = deck
        self._hands: list[list[int]] = [[], []]
        self._history: list[GameEvent] = []
        self._discard: list[int] = []
        self._round = 1
        self._actor: int | None = 0
        self._phase = "place"
        self._placements: list[int | None] = [None] * 4
        self._workers = [2, 2]
        self._ink = [2, 2]
        self._scores = [0, 0]
        self._rewards: tuple[float, ...] = (0.0, 0.0)
        for seat in (0, 1):
            self._draw(seat, 2)
        self._ready = True

    def _event(self, kind: str, seat: int, value: int | str) -> None:
        self._history.append(
            GameEvent(kind=kind, parameters=(("player", seat), ("value", value)))
        )

    def _draw(self, seat: int, count: int) -> None:
        count = min(count, len(self._deck))
        self._event("draw", seat, count)
        for _ in range(count):
            card = self._deck.pop()
            self._hands[seat].append(card)
            self._history.append(
                GameEvent(kind="card", parameters=(("card", card),), recipients=(seat,))
            )

    def current_player(self) -> int | None:
        self._require_ready()
        return self._actor

    def is_terminal(self) -> bool:
        return self.current_player() is None

    def observe(self, player_id: int) -> Observation[GuildworksObservation]:
        self._require_ready()
        self.validate_player(player_id)
        return Observation(
            player_id=player_id,
            current_player=self._actor,
            phase=self._phase,
            data=GuildworksObservation(
                round=self._round,
                placements=tuple(self._placements),
                workers=tuple(self._workers),
                ink=tuple(self._ink),
                scores=tuple(self._scores),
                own_hand=tuple(sorted(self._hands[player_id])),
                hand_counts=tuple(map(len, self._hands)),
                deck_count=len(self._deck),
                discard=tuple(self._discard),
                history=tuple(e for e in self._history if e.visible_to(player_id)),
            ),
        )

    def legal_actions(self, player_id: int) -> tuple[Action, ...]:
        self._require_ready()
        self.validate_player(player_id)
        if player_id != self._actor:
            return ()
        if self._phase == "place":
            return tuple(
                Action(kind="place", parameters=(("space", space),))
                for space, owner in zip(SPACES, self._placements, strict=True)
                if owner is None
            )
        return (
            *(
                Action(kind="craft", parameters=(("card", card),))
                for card in sorted(set(self._hands[player_id]))
                if card <= self._ink[player_id]
            ),
            Action(kind="pass"),
        )

    def step(self, action: Action) -> StepResult:
        seat = self.current_player()
        if seat is None:
            raise GameOverError("Game is over")
        # Exact scalar types matter: bool/float must not alias integer cards.
        if action not in self.legal_actions(seat) or any(
            type(value) not in (int, str) for _, value in action.parameters
        ):
            raise IllegalActionError("Action is not legal")
        start = len(self._history)
        before = self._scores.copy()
        if self._phase == "place":
            space = str(action.parameters[0][1])
            self._placements[SPACES.index(space)] = seat
            self._workers[seat] -= 1
            self._event("place", seat, space)
            if space == "archive":
                self._draw(seat, 2)
            elif space == "festival":
                self._scores[seat] += 1
                self._event("points", seat, 1)
            else:
                gain = 2 if space == "inkwell" else self._rng.randint(1, 3)
                self._ink[seat] += gain
                self._event("ink", seat, gain)
            self._phase = "craft"
        else:
            if action.kind == "craft":
                card = int(str(action.parameters[0][1]))
                self._hands[seat].remove(card)
                self._discard.append(card)
                self._ink[seat] -= card
                self._scores[seat] += card + 1
                self._event("craft", seat, card)
            else:
                self._event("pass", seat, 0)
            self._phase = "place"
            if sum(self._workers) == 0:
                if self._round == ROUNDS:
                    self._actor = None
                    self._phase = "finished"
                else:
                    self._round += 1
                    self._workers = [2, 2]
                    self._placements = [None] * 4
                    self._actor = (self._round - 1) % 2
                    self._event("round", self._actor, self._round)
            else:
                self._actor = 1 - seat
        self._rewards = tuple(
            float(a - b) for a, b in zip(self._scores, before, strict=True)
        )
        return StepResult(
            next_player=self._actor,
            terminated=self.is_terminal(),
            rewards=self._rewards,
            events=tuple(self._history[start:]),
        )

    def get_rewards(self) -> tuple[float, ...]:
        self._require_ready()
        return self._rewards

    def clone(self) -> Self:
        self._require_ready()
        return deepcopy(self)

    def render(self, player_id: int) -> str:
        return render_guildworks(self.observe(player_id))


def render_guildworks(observation: Observation[GuildworksObservation]) -> str:
    """Render only an already filtered view; never read authoritative state."""
    view = observation.data
    spaces = ", ".join(
        f"{name}={owner if owner is not None else '-'}"
        for name, owner in zip(SPACES, view.placements, strict=True)
    )
    return (
        f"Guildworks round {view.round}/{ROUNDS} | {observation.phase} | "
        f"actor {observation.current_player}\n{spaces}\n"
        f"Workers {view.workers} | Ink {view.ink} | Scores {view.scores}\n"
        f"Seat {observation.player_id} hand {view.own_hand} | "
        f"Hand counts {view.hand_counts} | Deck {view.deck_count}\n"
        f"Discard {view.discard}"
    )
