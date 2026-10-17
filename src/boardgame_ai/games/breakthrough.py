"""Frozen Phase 6 transfer game: public, deterministic 5x5 Breakthrough.

Two occupied home ranks per seat. Move one rank forward into an empty square,
or diagonally into an empty/enemy square. Reach the far rank, remove every enemy,
or leave the enemy without a move to win. No draws, chance, or shaping rewards.
"""

from copy import deepcopy
from typing import Self

from boardgame_ai.core import (
    Action,
    GameEngine,
    GameNotReadyError,
    GameOverError,
    GameSpec,
    IllegalActionError,
    Observation,
    StepResult,
    create_rng,
)
from boardgame_ai.games.breakthrough_observation import BreakthroughObservation


class BreakthroughEngine(GameEngine[BreakthroughObservation]):
    def __init__(self, *, starting_player: int = 0) -> None:
        if type(starting_player) is not int or starting_player not in (0, 1):
            raise ValueError("Invalid starter")
        self._starter = starting_player
        self._ready = False

    @property
    def spec(self) -> GameSpec:
        return GameSpec(
            game_id="breakthrough",
            name="Breakthrough 5x5 v1",
            min_players=2,
            max_players=2,
        )

    @property
    def num_players(self) -> int:
        return 2

    def _require_ready(self) -> None:
        if not self._ready:
            raise GameNotReadyError("Call reset before gameplay")

    def reset(self, seed: int | None = None) -> None:
        create_rng(seed)
        self._board: tuple[int | None, ...] = (1,) * 10 + (None,) * 5 + (0,) * 10
        self._actor: int | None = self._starter
        self._rewards: tuple[float, ...] = (0.0, 0.0)
        self._ready = True

    def current_player(self) -> int | None:
        self._require_ready()
        return self._actor

    def is_terminal(self) -> bool:
        return self.current_player() is None

    def observe(self, player_id: int) -> Observation[BreakthroughObservation]:
        self._require_ready()
        self.validate_player(player_id)
        return Observation(
            player_id=player_id,
            current_player=self._actor,
            phase="finished" if self._actor is None else "play",
            data=BreakthroughObservation(board=self._board),
        )

    def _moves(self, seat: int) -> tuple[Action, ...]:
        moves = []
        for source, owner in enumerate(self._board):
            if owner != seat:
                continue
            row, column = divmod(source, 5)
            next_row = row + (-1 if seat == 0 else 1)
            if not 0 <= next_row < 5:
                continue
            for dc in (-1, 0, 1):
                target_column = column + dc
                if not 0 <= target_column < 5:
                    continue
                target = 5 * next_row + target_column
                occupant = self._board[target]
                if occupant is None or dc != 0 and occupant != seat:
                    moves.append(
                        Action(
                            kind="move",
                            parameters=(
                                ("source", source),
                                ("target", target),
                            ),
                        )
                    )
        return tuple(moves)

    def legal_actions(self, player_id: int) -> tuple[Action, ...]:
        self._require_ready()
        self.validate_player(player_id)
        return self._moves(player_id) if self._actor == player_id else ()

    def step(self, action: Action) -> StepResult:
        seat = self.current_player()
        if seat is None:
            raise GameOverError("Game is over")
        if action not in self._moves(seat) or any(
            type(value) is not int for _, value in action.parameters
        ):
            raise IllegalActionError("Action is not legal")
        source, target = (value for _, value in action.parameters)
        assert type(source) is int and type(target) is int
        board = list(self._board)
        board[source], board[target] = None, seat
        self._board = tuple(board)
        won = target // 5 == (0 if seat == 0 else 4) or not self._moves(1 - seat)
        self._actor = None if won else 1 - seat
        self._rewards = tuple(
            (1.0 if player == seat else -1.0) if won else 0.0 for player in (0, 1)
        )
        return StepResult(
            next_player=self._actor,
            terminated=won,
            rewards=self._rewards,
        )

    def get_rewards(self) -> tuple[float, ...]:
        self._require_ready()
        return self._rewards

    def clone(self) -> Self:
        self._require_ready()
        return deepcopy(self)
