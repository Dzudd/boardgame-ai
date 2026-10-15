"""Immutable public-only Tic-Tac-Toe reconstruction."""

from dataclasses import dataclass
from functools import lru_cache
from typing import cast

from boardgame_ai.core import Action, GameOverError, IllegalActionError, Observation
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation
from boardgame_ai.games.tic_tac_toe.state import Board
from boardgame_ai.models.encoders import (
    encode_tic_tac_toe_legal_actions,
    encode_tic_tac_toe_observation,
)

LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def winner(board: Board) -> int | None:
    return next(
        (s for s in (0, 1) if any(all(board[i] == s for i in line) for line in LINES)),
        None,
    )


@lru_cache(maxsize=1)
def reachable() -> frozenset[tuple[Board, int | None]]:
    seen: set[tuple[Board, int | None]] = set()

    def visit(board: Board, actor: int) -> None:
        terminal = winner(board) is not None or None not in board
        key = (board, None if terminal else actor)
        if key in seen:
            return
        seen.add(key)
        if not terminal:
            for i, cell in enumerate(board):
                if cell is None:
                    visit(
                        cast(
                            Board,
                            tuple(actor if j == i else v for j, v in enumerate(board)),
                        ),
                        1 - actor,
                    )

    for starter in (0, 1):
        visit(cast(Board, (None,) * 9), starter)
    return frozenset(seen)


@dataclass(frozen=True, slots=True)
class PublicTicTacToeSimulator:
    board: Board
    actor: int | None

    @classmethod
    def from_observation(
        cls,
        observation: Observation[TicTacToeObservation],
        legal_actions: tuple[Action, ...],
    ) -> "PublicTicTacToeSimulator":
        encode_tic_tac_toe_observation(observation)
        encode_tic_tac_toe_legal_actions(legal_actions)
        state = cls(
            cast(Board, tuple(observation.data.board)), observation.current_player
        )
        if (state.board, state.actor) not in reachable():
            raise ValueError("Unreachable public position")
        if state.actor is not None and observation.player_id != state.actor:
            raise ValueError("Requires acting-seat observation")
        expected = state.legal_actions()
        if len(set(legal_actions)) != len(legal_actions) or set(legal_actions) != set(
            expected
        ):
            raise ValueError("Root choices disagree with public rules")
        return state

    def legal_actions(self) -> tuple[Action, ...]:
        return (
            tuple(
                Action(kind="place", parameters=(("cell", i),))
                for i, v in enumerate(self.board)
                if v is None
            )
            if self.actor is not None
            else ()
        )

    def observe(self) -> Observation[TicTacToeObservation]:
        return Observation(
            player_id=self.actor if self.actor is not None else 0,
            current_player=self.actor,
            phase="play" if self.actor is not None else "finished",
            data=TicTacToeObservation(board=self.board),
        )

    def step(
        self, action: Action
    ) -> tuple["PublicTicTacToeSimulator", tuple[float, float]]:
        if self.actor is None:
            raise GameOverError("Game is over")
        if (
            action not in self.legal_actions()
            or type(action.parameters[0][1]) is not int
        ):
            raise IllegalActionError("Action is not legal")
        cell = action.parameters[0][1]
        board = cast(
            Board,
            tuple(self.actor if i == cell else v for i, v in enumerate(self.board)),
        )
        won = winner(board)
        terminal = won is not None or None not in board
        rewards = (
            (0.0, 0.0) if won is None else ((1.0, -1.0) if won == 0 else (-1.0, 1.0))
        )
        return PublicTicTacToeSimulator(
            board, None if terminal else 1 - self.actor
        ), rewards
