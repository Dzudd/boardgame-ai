"""Fixed action mapping, player-input boundaries, and probability masking."""

from typing import Any, cast

import pytest
import torch

from boardgame_ai.core import Action
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.models.encoders import (
    decode_tic_tac_toe_action as decode,
)
from boardgame_ai.models.encoders import (
    encode_tic_tac_toe_action as encode,
)
from boardgame_ai.models.encoders import (
    encode_tic_tac_toe_legal_actions as encode_legal,
)
from boardgame_ai.models.masking import mask_action_logits


def place(cell: int) -> Action:
    return Action(kind="place", parameters=(("cell", cell),))


@pytest.mark.parametrize("cell", range(9))
def test_cell_roundtrip_returns_supplied_action(cell: int) -> None:
    action = place(cell)
    assert encode(action) == cell
    assert decode(cell, (action,)) is action
    assert encode_legal((action,)).tolist() == [i == cell for i in range(9)]


def test_legal_mask_order_independence_and_storage() -> None:
    actions = (place(8), place(0), place(4))
    rng_before = torch.random.get_rng_state().clone()
    mask = encode_legal(actions)
    other = encode_legal(tuple(reversed(actions)))
    assert mask.dtype == torch.bool and mask.device.type == "cpu"
    assert mask.shape == (9,) and mask.is_contiguous()
    assert mask.tolist() == [True, False, False, False, True, False, False, False, True]
    assert torch.equal(mask, other)
    mask.zero_()
    assert other.sum().item() == 3
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert actions == (place(8), place(0), place(4))


@pytest.mark.parametrize("starter", [0, 1])
def test_complete_games_use_only_detached_player_choices(starter: int) -> None:
    game = TicTacToeEngine(starting_player=starter)
    game.reset(seed=42)
    moves = (0, 3, 1, 4, 2)
    for cell in moves:
        actor = game.current_player()
        assert actor is not None
        _, legal = game.decision_for(actor)
        mask = encode_legal(legal)
        assert mask[cell].item()
        assert encode_legal(game.legal_actions(1 - actor)).sum().item() == 0
        action = decode(cell, legal)
        assert any(action is candidate for candidate in legal)
        game.step(action)
    assert game.is_terminal()
    # Terminal wins can have empty cells; availability comes from supplied choices.
    for seat in (0, 1):
        mask = encode_legal(game.legal_actions(seat))
        assert mask.sum().item() == 0
        with pytest.raises(ValueError):
            mask_action_logits(torch.zeros(9), mask)
        with pytest.raises(ValueError):
            decode(8, ())


@pytest.mark.parametrize(
    "action",
    [
        Action(kind="draw"),
        Action(kind="place"),
        Action(kind="place", parameters=(("row", 0),)),
        Action(kind="place", parameters=(("cell", 0), ("extra", 1))),
        place(-1),
        place(9),
        place(cast(Any, True)),
        place(cast(Any, 1.0)),
        cast(Any, object()),
    ],
)
def test_incompatible_actions_reject(action: Action) -> None:
    with pytest.raises(ValueError):
        encode(action)
    with pytest.raises(ValueError):
        encode_legal((action,))


@pytest.mark.parametrize("index", [-1, 9, True, 1.0])
def test_invalid_indices_reject(index: Any) -> None:
    with pytest.raises(ValueError):
        decode(index, (place(1),))


def test_duplicates_and_unavailable_indices_reject() -> None:
    with pytest.raises(ValueError):
        encode_legal((place(1), place(1)))
    with pytest.raises(ValueError):
        decode(1, (place(1), place(1)))
    with pytest.raises(ValueError):
        decode(0, (place(1),))
    with pytest.raises(ValueError):
        encode_legal(cast(Any, [place(1)]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_masked_probabilities_and_gradients(dtype: torch.dtype) -> None:
    logits = torch.tensor([[1, 100, 2], [100, 3, 100]], dtype=dtype, requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    original = logits.detach().clone()
    rng_before = torch.random.get_rng_state().clone()
    result = mask_action_logits(logits, mask)
    assert result.dtype == dtype and result.device == logits.device
    assert torch.equal(result[mask], logits[mask])
    assert torch.isneginf(result[~mask]).all().item()
    probabilities = result.softmax(dim=-1)
    assert probabilities[~mask].count_nonzero().item() == 0
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2, dtype=dtype))
    assert probabilities[1, 1].item() == 1
    result[mask].sum().backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert torch.equal(logits.grad, mask.to(dtype))
    assert torch.equal(logits.detach(), original)
    assert torch.equal(mask, torch.tensor([[True, False, True], [False, True, False]]))
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_single_and_multiple_batch_dimensions() -> None:
    for shape in ((9,), (2, 3, 9)):
        logits = torch.zeros(shape)
        mask = torch.ones(shape, dtype=torch.bool)
        result = mask_action_logits(logits, mask)
        assert result.shape == logits.shape
        result.fill_(5)
        assert logits.count_nonzero().item() == 0


@pytest.mark.parametrize(
    "logits,mask",
    [
        (torch.zeros(9), torch.ones(1, 9, dtype=torch.bool)),
        (torch.zeros(2, 9), torch.ones(9, dtype=torch.bool)),
        (torch.tensor(0.0), torch.tensor(True)),
        (torch.empty(0, 9), torch.empty(0, 9, dtype=torch.bool)),
        (torch.empty(0), torch.empty(0, dtype=torch.bool)),
        (torch.zeros(9, dtype=torch.int64), torch.ones(9, dtype=torch.bool)),
        (torch.zeros(9, dtype=torch.complex64), torch.ones(9, dtype=torch.bool)),
        (torch.zeros(9), torch.ones(9)),
        (torch.zeros(9), torch.zeros(9, dtype=torch.bool)),
        (
            torch.zeros(2, 3),
            torch.tensor([[True, False, False], [False, False, False]]),
        ),
        (torch.tensor([float("nan"), 0.0]), torch.tensor([False, True])),
        (torch.tensor([float("inf"), 0.0]), torch.tensor([True, True])),
        (torch.tensor([float("-inf"), 0.0]), torch.tensor([False, True])),
    ],
)
def test_invalid_masking_inputs_reject(
    logits: torch.Tensor, mask: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        mask_action_logits(logits, mask)


def test_device_mismatch_rejects_before_computation() -> None:
    with pytest.raises(ValueError, match="device"):
        mask_action_logits(
            torch.zeros(9), torch.ones(9, dtype=torch.bool, device="meta")
        )
