"""Authoritative deterministic Tic-Tac-Toe engine."""

from copy import deepcopy
from random import Random
from typing import Self, cast

from boardgame_ai.core import (
    Action,
    GameEngine,
    GameNotReadyError,
    GameOverError,
    GameSpec,
    GameState,
    IllegalActionError,
    Observation,
    StepResult,
    create_rng,
)
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation
from boardgame_ai.games.tic_tac_toe.rendering import render_board
from boardgame_ai.games.tic_tac_toe.state import Board, Cell, TicTacToeState

_WINNING_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


class TicTacToeEngine(GameEngine[TicTacToeObservation]):
    """Complete two-seat engine with public views and no full-state accessor."""

    def __init__(self, *, starting_player: int = 0) -> None:
        if type(starting_player) is not int or starting_player not in (0, 1):
            raise ValueError("Starting player must be seat 0 or 1")
        self._starting_player = starting_player
        self._state: GameState[TicTacToeState] | None = None
        self._rng: Random | None = None
        self._rewards: tuple[float, ...] = (0.0, 0.0)

    @property
    def spec(self) -> GameSpec:
        return GameSpec(
            game_id="tic_tac_toe", name="Tic-Tac-Toe", min_players=2, max_players=2
        )

    @property
    def num_players(self) -> int:
        return 2

    def reset(self, seed: int | None = None) -> None:
        """Clear the board and rewards; configured starting seat acts first."""
        rng = create_rng(seed)
        state = GameState(
            current_player=self._starting_player, phase="play", data=TicTacToeState()
        )
        self._rng = rng
        self._state = state
        self._rewards = (0.0, 0.0)

    def _require_state(self) -> GameState[TicTacToeState]:
        if self._state is None:
            raise GameNotReadyError("Call reset before gameplay")
        return self._state

    def current_player(self) -> int | None:
        return self._require_state().current_player

    def observe(self, player_id: int) -> Observation[TicTacToeObservation]:
        """Return a fresh public view for either valid seat, even after the game."""
        state = self._require_state()
        self.validate_player(player_id)
        return Observation(
            player_id=player_id,
            current_player=state.current_player,
            phase=state.phase,
            data=TicTacToeObservation(board=state.data.board),
        )

    def render(self, player_id: int) -> str:
        """Format this player's public board without printing or changing state."""
        return render_board(self.observe(player_id))

    def legal_actions(self, player_id: int) -> tuple[Action, ...]:
        """List empty cells in row-major order for the current actor only."""
        state = self._require_state()
        self.validate_player(player_id)
        if state.current_player is None or player_id != state.current_player:
            return ()
        return tuple(
            Action(kind="place", parameters=(("cell", cell),))
            for cell, value in enumerate(state.data.board)
            if value is None
        )

    def _validate_action(self, action: Action) -> int:
        """Return a legal cell index without changing state or randomness.

        Step calls this before applying a move.
        Explicit types avoid Python's boolean/float equality with integer cells.
        """
        state = self._require_state()
        if state.current_player is None:
            raise GameOverError("Game is over")
        if action.kind != "place" or len(action.parameters) != 1:
            raise IllegalActionError("Action is not legal")
        name, cell = action.parameters[0]
        if (
            name != "cell"
            or type(cell) is not int
            or not 0 <= cell < 9
            or state.data.board[cell] is not None
        ):
            raise IllegalActionError("Action is not legal")
        return cell

    def step(self, action: Action) -> StepResult:
        """Apply a validated placement, resolve wins/draws, and set step rewards."""
        cell = self._validate_action(action)
        state = self._require_state()
        mark: Cell = 0 if state.current_player == 0 else 1
        board = cast(
            Board,
            tuple(
                mark if index == cell else value
                for index, value in enumerate(state.data.board)
            ),
        )
        won = any(
            board[a] == board[b] == board[c] == mark for a, b, c in _WINNING_LINES
        )
        terminated = won or all(value is not None for value in board)
        next_player = None if terminated else (1 if mark == 0 else 0)
        rewards = ((1.0, -1.0) if mark == 0 else (-1.0, 1.0)) if won else (0.0, 0.0)
        next_state = GameState(
            current_player=next_player,
            phase="finished" if terminated else "play",
            data=TicTacToeState(board=board),
        )
        result = StepResult(
            next_player=next_player, terminated=terminated, rewards=rewards
        )
        self._state = next_state
        self._rewards = result.rewards
        return result

    def is_terminal(self) -> bool:
        """Whether the stored outcome has no remaining decision-maker."""
        return self._require_state().current_player is None

    def get_rewards(self) -> tuple[float, ...]:
        self._require_state()
        return self._rewards

    def clone(self) -> Self:
        """Copy initialized state and RNG independently without consuming chance."""
        self._require_state()
        return deepcopy(self)
