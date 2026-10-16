"""Connect Four tensor perspective and supplied-action boundary regressions."""

import random
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from boardgame_ai.core import Action, GameState
from boardgame_ai.games.connect_four import ConnectFourEngine, ConnectFourObservation
from boardgame_ai.models.encoders import (
    decode_connect_four_action as decode,
)
from boardgame_ai.models.encoders import (
    encode_connect_four_action as encode_action,
)
from boardgame_ai.models.encoders import (
    encode_connect_four_legal_actions as encode_legal,
)
from boardgame_ai.models.encoders import (
    encode_connect_four_observation as encode,
)
from boardgame_ai.models.masking import mask_action_logits


def drop(column: int) -> Action:
    return Action(kind="drop", parameters=(("column", column),))


def test_orientation_perspective_and_markov_features() -> None:
    game = ConnectFourEngine()
    game.reset()
    for column in (0, 6, 0):
        game.step(drop(column))
    view = game.observe(0)
    tensor = encode(view)
    assert tensor.shape == (6, 6, 7)
    assert tensor.dtype == torch.float32 and tensor.device.type == "cpu"
    assert tensor.is_contiguous() and not tensor.requires_grad
    assert tensor[0, 5, 0] == tensor[0, 4, 0] == tensor[1, 5, 6] == 1
    assert tensor[0].sum() == 2 and tensor[1].sum() == 1
    assert torch.all(tensor[:3].sum(0) == 1)
    assert torch.all(tensor[3] == 0) and torch.all(tensor[4] == 1)
    assert torch.all(tensor[5] == 0)
    assert torch.equal(encode(game.observe(1)), tensor[[1, 0, 2, 4, 3, 5]])
    # Visible history is redundant for this board/actor policy representation.
    assert torch.equal(
        encode(replace(view, data=replace(view.data, history=()))), tensor
    )
    relabeled = replace(
        view,
        player_id=1,
        current_player=0,
        data=replace(
            view.data,
            board=tuple(None if c is None else 1 - c for c in view.data.board),
        ),
    )
    assert torch.equal(encode(relabeled), tensor)


@pytest.mark.parametrize("starter", [0, 1])
def test_full_columns_inactive_and_terminal_views(starter: int) -> None:
    game = ConnectFourEngine(starting_player=starter)
    game.reset()
    for _ in range(6):
        game.step(drop(0))
    view, legal = game.decision_for(starter)
    assert encode(view)[2, :, 0].count_nonzero() == 0
    assert encode_legal(legal).tolist() == [False] + [True] * 6
    assert not encode_legal(game.legal_actions(1 - starter)).any()
    with pytest.raises(ValueError):
        decode(0, legal)
    logits = torch.tensor([1000.0, 1, 2, 3, 4, 5, 6])
    assert mask_action_logits(logits, encode_legal(legal)).softmax(0)[0] == 0
    # A win with empty columns must still expose no terminal choices.
    for column in (1, 6, 2, 6, 3):
        game.step(drop(column))
    assert game.is_terminal()
    for seat in (0, 1):
        encoded = encode(game.observe(seat))
        assert not encoded[3:5].any() and torch.all(encoded[5] == 1)
        assert encoded[2].sum() > 0
        assert not encode_legal(game.legal_actions(seat)).any()
        with pytest.raises(ValueError):
            decode(1, game.legal_actions(seat))


@pytest.mark.parametrize("column", range(7))
def test_column_roundtrip(column: int) -> None:
    action = drop(column)
    assert encode_action(action) == column
    assert decode(column, (action,)) is action
    assert encode_legal((action,)).tolist() == [i == column for i in range(7)]


def test_draw_encodes_terminal_without_empty_cells() -> None:
    game = ConnectFourEngine()
    game.reset()
    for column in (
        3,
        3,
        0,
        2,
        5,
        4,
        5,
        6,
        4,
        2,
        2,
        3,
        3,
        5,
        5,
        2,
        2,
        4,
        3,
        3,
        1,
        1,
        2,
        1,
        5,
        5,
        0,
        4,
        0,
        4,
        4,
        6,
        6,
        6,
        6,
        0,
        1,
        0,
        6,
        0,
        1,
        1,
    ):
        game.step(drop(column))
    assert game.is_terminal() and game.get_rewards() == (0.0, 0.0)
    for seat in (0, 1):
        tensor = encode(game.observe(seat))
        assert not tensor[2:5].any() and torch.all(tensor[5] == 1)
        assert not encode_legal(game.legal_actions(seat)).any()


@pytest.mark.parametrize("starter", [0, 1])
def test_complete_games_recover_boards_and_use_supplied_actions(starter: int) -> None:
    rng = random.Random(714)
    for _ in range(12):
        game = ConnectFourEngine(starting_player=starter)
        game.reset()
        steps = 0
        while not game.is_terminal():
            actor = game.current_player()
            assert actor is not None
            view, legal = game.decision_for(actor)
            tensor = encode(view)
            recovered = tuple(
                None if empty else actor if own else 1 - actor
                for own, empty in zip(
                    tensor[0].flatten().tolist(),
                    tensor[2].flatten().tolist(),
                    strict=True,
                )
            )
            assert recovered == view.data.board
            mask = encode_legal(legal)
            assert mask.tolist() == [cell is None for cell in view.data.board[:7]]
            action = rng.choice(legal)
            assert decode(encode_action(action), tuple(reversed(legal))) is action
            game.step(action)
            steps += 1
        assert 7 <= steps <= 42
        assert torch.all(encode(game.observe(starter))[5] == 1)


def test_storage_determinism_and_global_rng_isolation() -> None:
    game = ConnectFourEngine()
    game.reset()
    view, legal = game.decision_for(0)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    tensor, mask = encode(view), encode_legal(legal)
    other_tensor, other_mask = encode(view), encode_legal(tuple(reversed(legal)))
    assert decode(0, legal) is legal[0]
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0]
    assert np.array_equal(after_numpy[1], numpy_rng[1])
    assert after_numpy[2:] == numpy_rng[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    tensor.zero_()
    mask.zero_()
    assert torch.all(other_tensor[2:4] == 1) and other_mask.all()
    game.step(drop(0))
    game.reset()
    assert torch.equal(encode(view), other_tensor)
    assert view.data.board == (None,) * 42
    assert other_mask.shape == (7,) and other_mask.dtype == torch.bool
    assert other_mask.device.type == "cpu" and other_mask.is_contiguous()


@pytest.mark.parametrize(
    "changes",
    [
        {"player_id": True},
        {"player_id": 2},
        {"current_player": False},
        {"current_player": 1.0},
        {"current_player": 2},
        {"current_player": None},
        {"phase": "finished"},
        {"phase": "unknown"},
        {"data": object()},
        *[
            {"data": ConnectFourObservation(board=cast(Any, board), history=())}
            for board in (
                (None,) * 41,
                [None] * 42,
                (True,) + (None,) * 41,
                (1.0,) + (None,) * 41,
                (2,) + (None,) * 41,
            )
        ],
    ],
)
def test_invalid_observation_schema_rejects(changes: dict[str, Any]) -> None:
    game = ConnectFourEngine()
    game.reset()
    with pytest.raises(ValueError):
        encode(replace(game.observe(0), **changes))


def test_authoritative_state_is_not_an_observation() -> None:
    state = GameState(current_player=0, phase="play", data=object())
    with pytest.raises(ValueError, match="observation"):
        encode(cast(Any, state))


@pytest.mark.parametrize(
    "action",
    [
        Action(kind="place", parameters=(("column", 0),)),
        Action(kind="drop"),
        Action(kind="drop", parameters=(("cell", 0),)),
        Action(kind="drop", parameters=(("column", 0), ("extra", 1))),
        *[drop(cast(Any, column)) for column in (-1, 7, True, 1.0)],
        cast(Any, object()),
    ],
)
def test_invalid_action_schema_rejects(action: Action) -> None:
    with pytest.raises(ValueError):
        encode_action(action)
    with pytest.raises(ValueError):
        encode_legal((action,))


@pytest.mark.parametrize("index", [-1, 7, True, 1.0])
def test_invalid_indices_reject(index: Any) -> None:
    with pytest.raises(ValueError):
        decode(index, (drop(1),))


def test_duplicate_unavailable_and_non_tuple_choices_reject() -> None:
    for actions in ((drop(1), drop(1)), cast(Any, [drop(1)])):
        with pytest.raises(ValueError):
            encode_legal(actions)
        with pytest.raises(ValueError):
            decode(1, actions)
    with pytest.raises(ValueError):
        decode(0, (drop(1),))
    with pytest.raises(ValueError):
        mask_action_logits(torch.zeros(7), encode_legal(()))
