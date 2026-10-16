"""Deterministic gravity-based Connect Four rules, independent of ML tools."""

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
from boardgame_ai.games.connect_four.observation import (
    COLUMNS,
    MAX_STEPS,
    ROWS,
    Board,
    ConnectFourObservation,
)
from boardgame_ai.games.connect_four.rendering import render_board


def winner(board: Board) -> int | None:
    """Find a four-disc line without wrapping across board edges."""
    for row in range(ROWS):
        for column in range(COLUMNS):
            seat = board[row * COLUMNS + column]
            if seat is None:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= row + 3 * dr < ROWS and 0 <= column + 3 * dc < COLUMNS:
                    if all(
                        board[(row + i * dr) * COLUMNS + column + i * dc] == seat
                        for i in range(1, 4)
                    ):
                        return seat
    return None


class ConnectFourEngine(GameEngine[ConnectFourObservation]):
    """Seven columns, six rows, alternating drops; first four in a row wins."""

    def __init__(self, *, starting_player: int = 0) -> None:
        if type(starting_player) is not int or starting_player not in (0, 1):
            raise ValueError("Starting player must be seat 0 or 1")
        self._starting_player = starting_player
        self._ready = False

    @property
    def spec(self) -> GameSpec:
        return GameSpec(
            game_id="connect_four", name="Connect Four", min_players=2, max_players=2
        )

    @property
    def num_players(self) -> int:
        return 2

    def _require_ready(self) -> None:
        if not self._ready:
            raise GameNotReadyError("Call reset before gameplay")

    def reset(self, seed: int | None = None) -> None:
        # Validate seed before mutation. There is no setup or gameplay chance.
        create_rng(seed)
        self._board: Board = (None,) * MAX_STEPS
        self._history: tuple[int, ...] = ()
        self._actor: int | None = self._starting_player
        self._rewards: tuple[float, ...] = (0.0, 0.0)
        self._ready = True

    def current_player(self) -> int | None:
        self._require_ready()
        return self._actor

    def is_terminal(self) -> bool:
        return self.current_player() is None

    def observe(self, player_id: int) -> Observation[ConnectFourObservation]:
        self._require_ready()
        self.validate_player(player_id)
        return Observation(
            player_id=player_id,
            current_player=self._actor,
            phase="finished" if self._actor is None else "play",
            data=ConnectFourObservation(board=self._board, history=self._history),
        )

    def legal_actions(self, player_id: int) -> tuple[Action, ...]:
        self._require_ready()
        self.validate_player(player_id)
        if self._actor is None or self._actor != player_id:
            return ()
        return tuple(
            Action(kind="drop", parameters=(("column", column),))
            for column in range(COLUMNS)
            if self._board[column] is None
        )

    def step(self, action: Action) -> StepResult:
        seat = self.current_player()
        if seat is None:
            raise GameOverError("Game is over")
        if (
            action not in self.legal_actions(seat)
            or len(action.parameters) != 1
            or type(action.parameters[0][1]) is not int
        ):
            raise IllegalActionError("Action is not legal")
        column = action.parameters[0][1]
        assert type(column) is int
        row = next(
            r
            for r in range(ROWS - 1, -1, -1)
            if self._board[r * COLUMNS + column] is None
        )
        board = list(self._board)
        board[row * COLUMNS + column] = seat
        self._board = tuple(board)
        self._history += (column,)
        won = winner(self._board) is not None
        self._actor = None if won or None not in self._board else 1 - seat
        self._rewards = (
            tuple(1.0 if p == seat else -1.0 for p in (0, 1)) if won else (0.0, 0.0)
        )
        return StepResult(
            next_player=self._actor,
            terminated=self._actor is None,
            rewards=self._rewards,
        )

    def get_rewards(self) -> tuple[float, ...]:
        self._require_ready()
        return self._rewards

    def clone(self) -> Self:
        self._require_ready()
        return deepcopy(self)

    def render(self, player_id: int) -> str:
        return render_board(self.observe(player_id))
