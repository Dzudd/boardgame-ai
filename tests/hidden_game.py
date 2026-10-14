"""Test-only chance game: private hands and deck, with actors 0, 0, 1."""

from copy import deepcopy
from dataclasses import dataclass
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


@dataclass
class CardView:
    own_hand: list[int]
    hand_sizes: tuple[int, ...]
    decisions: int


class HiddenGame(GameEngine[CardView]):
    """Minimal fixture, not the Phase 5 card game or a production engine."""

    def __init__(self) -> None:
        self._ready = False

    @property
    def spec(self) -> GameSpec:
        return GameSpec(
            game_id="hidden_test", name="Hidden test", min_players=2, max_players=2
        )

    @property
    def num_players(self) -> int:
        return 2

    def _require_ready(self) -> None:
        if not self._ready:
            raise GameNotReadyError("Call reset before gameplay")

    def reset(self, seed: int | None = None) -> None:
        rng = create_rng(seed)
        deck = list(range(20))
        rng.shuffle(deck)
        self._rng = rng
        self._deck = deck
        self._hands = [[deck.pop()], [deck.pop()]]
        self._decisions = 0
        self._rewards = (0.0, 0.0)
        self._ready = True

    def observe(self, player_id: int) -> Observation[CardView]:
        self._require_ready()
        self.validate_player(player_id)
        return Observation(
            player_id=player_id,
            current_player=self.current_player(),
            phase="finished" if self.is_terminal() else "draw",
            data=CardView(
                own_hand=self._hands[player_id].copy(),
                hand_sizes=tuple(len(hand) for hand in self._hands),
                decisions=self._decisions,
            ),
        )

    def legal_actions(self, player_id: int) -> tuple[Action, ...]:
        self._require_ready()
        self.validate_player(player_id)
        return (Action(kind="draw"),) if player_id == self.current_player() else ()

    def current_player(self) -> int | None:
        self._require_ready()
        return None if self.is_terminal() else (0 if self._decisions < 2 else 1)

    def step(self, action: Action) -> StepResult:
        player = self.current_player()
        if player is None:
            raise GameOverError("Game is over")
        if action not in self.legal_actions(player):
            raise IllegalActionError("Action is not legal")
        # Draw a random remaining card so clone/rejection tests exercise RNG state.
        card = self._deck.pop(self._rng.randrange(len(self._deck)))
        self._hands[player].append(card)
        self._decisions += 1
        self._rewards = (1.0, -1.0) if self.is_terminal() else (0.0, 0.0)
        return StepResult(
            next_player=self.current_player(),
            terminated=self.is_terminal(),
            rewards=self._rewards,
            events=(
                GameEvent(kind="drew", parameters=(("player", player),)),
                GameEvent(
                    kind="card", parameters=(("card", card),), recipients=(player,)
                ),
            ),
        )

    def is_terminal(self) -> bool:
        self._require_ready()
        return self._decisions == 3

    def get_rewards(self) -> tuple[float, ...]:
        self._require_ready()
        return self._rewards

    def clone(self) -> Self:
        self._require_ready()
        return deepcopy(self)
