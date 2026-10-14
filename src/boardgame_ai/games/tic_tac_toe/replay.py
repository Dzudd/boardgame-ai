"""Versioned public placement logs; no private state, events, or RNG seeds."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe.engine import TicTacToeEngine
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation

type ReplayStatus = Literal["completed", "cancelled", "truncated"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayMove:
    player: int
    cell: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TicTacToeReplay:
    starting_player: int
    moves: tuple[ReplayMove, ...]
    status: ReplayStatus


class TicTacToeRecorder:
    """Observe validated successful placements in trusted orchestration."""

    def __init__(self) -> None:
        self._moves: list[ReplayMove] = []

    def record(self, player: int, action: Action) -> None:
        if action.kind != "place" or len(action.parameters) != 1:
            raise ValueError("Replay requires a placement")
        name, cell = action.parameters[0]
        if name != "cell" or type(cell) is not int:
            raise ValueError("Replay requires an integer cell")
        self._moves.append(ReplayMove(player=player, cell=cell))

    def finish(self, starting_player: int, status: ReplayStatus) -> TicTacToeReplay:
        return TicTacToeReplay(
            starting_player=starting_player, moves=tuple(self._moves), status=status
        )


def replay_positions(
    replay: TicTacToeReplay,
) -> tuple[Observation[TicTacToeObservation], ...]:
    """Validate the entire log through a fresh engine, returning public boards."""
    if replay.status not in ("completed", "cancelled", "truncated"):
        raise ValueError("Invalid replay status")
    if len(replay.moves) > 9:
        raise ValueError("Replay exceeds Tic-Tac-Toe decision limit")
    game = TicTacToeEngine(starting_player=replay.starting_player)
    game.reset(0)  # Setup is deterministic; no engine/agent seed is needed.
    positions = [game.observe(0)]
    for move in replay.moves:
        if type(move.player) is not int or move.player != game.current_player():
            raise ValueError("Replay actor disagrees with engine")
        if type(move.cell) is not int:
            raise ValueError("Replay cell must be an integer")
        game.step(Action(kind="place", parameters=(("cell", move.cell),)))
        positions.append(game.observe(0))
    if (replay.status == "completed") != game.is_terminal():
        raise ValueError("Replay ending status disagrees with engine")
    return tuple(positions)


def save_replay(replay: TicTacToeReplay, path: str | Path) -> None:
    """Validate then save a public Tic-Tac-Toe log, overwriting the chosen path."""
    replay_positions(replay)
    payload = {"schema_version": 1, "game_id": "tic_tac_toe", **asdict(replay)}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_replay(path: str | Path) -> TicTacToeReplay:
    """Read strict version-1 JSON, then verify actors, moves, and ending status."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {
            "schema_version",
            "game_id",
            "starting_player",
            "moves",
            "status",
        }:
            raise ValueError
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError
        if data["game_id"] != "tic_tac_toe" or not isinstance(data["moves"], list):
            raise ValueError
        if len(data["moves"]) > 9:
            raise ValueError
        moves = []
        for move in data["moves"]:
            if not isinstance(move, dict) or set(move) != {"player", "cell"}:
                raise ValueError
            moves.append(ReplayMove(player=move["player"], cell=move["cell"]))
        replay = TicTacToeReplay(
            starting_player=data["starting_player"],
            moves=tuple(moves),
            status=data["status"],
        )
        replay_positions(replay)
        return replay
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError("Invalid Tic-Tac-Toe replay") from error
