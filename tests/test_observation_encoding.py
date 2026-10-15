"""Tensor semantics and isolation at the filtered-observation boundary."""

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.games.tic_tac_toe.state import Board
from boardgame_ai.models.encoders import encode_tic_tac_toe_observation as encode


def test_standalone_observation_orientation_and_perspective() -> None:
    view = Observation(
        player_id=0,
        current_player=1,
        phase="play",
        data=TicTacToeObservation(board=(0, None, 1, None, 0, None, 1, None, None)),
    )
    tensor = encode(view)
    expected = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 0]] * 3,
            [[1, 1, 1]] * 3,
            [[0, 0, 0]] * 3,
        ],
        dtype=torch.float32,
    )
    assert torch.equal(tensor, expected)
    other = encode(replace(view, player_id=1))
    assert torch.equal(other, expected[[1, 0, 2, 4, 3, 5]])
    assert tensor.shape == (6, 3, 3)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert tensor.is_contiguous() and not tensor.requires_grad


@pytest.mark.parametrize("starter", [0, 1])
def test_initial_views_and_player_visible_decision_inputs(starter: int) -> None:
    game = TicTacToeEngine(starting_player=starter)
    game.reset(seed=42)
    view, _ = game.decision_for(starter)
    tensor = encode(view)
    assert torch.count_nonzero(tensor[:2]).item() == 0
    assert torch.all(tensor[2:4] == 1).item()
    assert torch.count_nonzero(tensor[4:]).item() == 0
    inactive = encode(game.observe(1 - starter))
    assert torch.all(inactive[4] == 1).item()
    assert torch.count_nonzero(inactive[3]).item() == 0


@pytest.mark.parametrize(
    "moves", [(0, 3, 1, 4, 2), (0, 3, 1, 4, 8, 5), (0, 1, 2, 4, 3, 5, 7, 6, 8)]
)
@pytest.mark.parametrize("starter", [0, 1])
def test_terminal_wins_and_draws_retain_board(
    moves: tuple[int, ...], starter: int
) -> None:
    game = TicTacToeEngine(starting_player=starter)
    game.reset()
    for cell in moves:
        game.step(Action(kind="place", parameters=(("cell", cell),)))
    assert game.is_terminal()
    for player in (0, 1):
        view = game.observe(player)
        tensor = encode(view)
        assert torch.count_nonzero(tensor[3:5]).item() == 0
        assert torch.all(tensor[5] == 1).item()
        assert torch.all(tensor[:3].sum(dim=0) == 1).item()
        for cell, mark in enumerate(view.data.board):
            channel = 2 if mark is None else (0 if mark == player else 1)
            assert tensor[channel, cell // 3, cell % 3].item() == 1


def test_encoding_is_deterministic_and_storage_is_independent() -> None:
    game = TicTacToeEngine()
    game.reset(seed=1)
    view = game.observe(0)
    rng_before = torch.random.get_rng_state().clone()
    first = encode(view)
    second = encode(view)
    assert torch.equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    first.zero_()
    assert torch.all(second[2:4] == 1).item()
    assert view.data.board == (None,) * 9
    game.step(Action(kind="place", parameters=(("cell", 4),)))
    game.reset(seed=999)
    assert torch.equal(encode(view), second)
    assert torch.equal(encode(game.observe(0)), second)
    assert torch.equal(torch.stack([second, encode(view)])[0], second)


@pytest.mark.parametrize(
    "changes",
    [
        {"player_id": True},
        {"player_id": -1},
        {"player_id": 2},
        {"current_player": False},
        {"current_player": 1.0},
        {"current_player": 2},
        {"phase": "unknown"},
        {"phase": "finished"},
        {"current_player": None},
        {"data": object()},
        {"data": TicTacToeObservation(board=cast(Any, (None,) * 8))},
        {"data": TicTacToeObservation(board=cast(Any, [None] * 9))},
        {"data": TicTacToeObservation(board=cast(Any, (True,) + (None,) * 8))},
        {"data": TicTacToeObservation(board=cast(Any, (1.0,) + (None,) * 8))},
        {"data": TicTacToeObservation(board=cast(Any, (2,) + (None,) * 8))},
    ],
)
def test_incompatible_observation_schema_rejected(changes: dict[str, Any]) -> None:
    view = Observation(
        player_id=0,
        current_player=0,
        phase="play",
        data=TicTacToeObservation(board=(None,) * 9),
    )
    with pytest.raises(ValueError):
        encode(replace(view, **changes))


def test_authoritative_records_are_not_encoder_inputs() -> None:
    from boardgame_ai.core import GameState
    from boardgame_ai.games.tic_tac_toe import TicTacToeState

    state = GameState(current_player=0, phase="play", data=TicTacToeState())
    with pytest.raises(ValueError, match="observation"):
        encode(cast(Any, state))


def test_all_reachable_boards_preserve_visible_information_and_seat_symmetry() -> None:
    """Audit public views independently of engine state and move generation."""
    from tests.test_tic_tac_toe_regressions import outcome

    visited: set[tuple[int | None, ...]] = set()
    terminal_count = 0

    def visit(board: tuple[int | None, ...], actor: int) -> None:
        nonlocal terminal_count
        if board in visited:
            return
        visited.add(board)
        terminal = outcome(board) is not None
        terminal_count += int(terminal)
        for seat in (0, 1):
            view = Observation(
                player_id=seat,
                current_player=None if terminal else actor,
                phase="finished" if terminal else "play",
                data=TicTacToeObservation(board=cast(Board, board)),
            )
            tensor = encode(view)
            assert tensor.shape == (6, 3, 3)
            assert tensor.dtype == torch.float32 and tensor.device.type == "cpu"
            assert tensor.is_contiguous() and not tensor.requires_grad
            assert torch.all((tensor == 0) | (tensor == 1)).item()
            assert torch.all(tensor[:3].sum(dim=0) == 1).item()
            # Recover the complete visible board, preserving spatial coordinates.
            recovered = tuple(
                None
                if tensor[2, row, col]
                else seat
                if tensor[0, row, col]
                else 1 - seat
                for row in range(3)
                for col in range(3)
            )
            assert recovered == board
            assert torch.all(tensor[3] == int(not terminal and actor == seat)).item()
            assert torch.all(tensor[4] == int(not terminal and actor != seat)).item()
            assert torch.all(tensor[5] == int(terminal)).item()
            relabeled = replace(
                view,
                player_id=1 - seat,
                current_player=None if terminal else 1 - actor,
                data=TicTacToeObservation(
                    board=cast(
                        Board,
                        tuple(None if mark is None else 1 - mark for mark in board),
                    )
                ),
            )
            assert torch.equal(tensor, encode(relabeled))
        if not terminal:
            for cell, mark in enumerate(board):
                if mark is None:
                    visit((*board[:cell], actor, *board[cell + 1 :]), 1 - actor)

    visit((None,) * 9, 0)
    assert len(visited) == 5478
    assert terminal_count == 958
