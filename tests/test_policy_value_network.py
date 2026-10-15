"""Policy/value semantics, differentiation, and player-input isolation."""

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.models.encoders import (
    encode_tic_tac_toe_legal_actions,
    encode_tic_tac_toe_observation,
)
from boardgame_ai.models.masking import mask_action_logits


@pytest.fixture
def model() -> TicTacToeActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return TicTacToeActorCritic()


def visible_tensor() -> torch.Tensor:
    return encode_tic_tac_toe_observation(
        Observation(
            player_id=0,
            current_player=1,
            phase="play",
            data=TicTacToeObservation(board=(0, None, 1, None, 0, None, 1, None, None)),
        )
    )


@pytest.mark.parametrize("batch_shape", [(), (1,), (4,), (2, 3)])
def test_single_and_batched_outputs_match_individual_predictions(
    model: TicTacToeActorCritic, batch_shape: tuple[int, ...]
) -> None:
    single = visible_tensor()
    batch = single.expand(*batch_shape, 6, 3, 3)
    logits, values = model.forward(batch)
    expected_logits, expected_value = model.forward(single)
    assert logits.shape == (*batch_shape, 9)
    assert values.shape == batch_shape
    assert logits.dtype == values.dtype == torch.float32
    assert logits.device.type == values.device.type == "cpu"
    assert torch.isfinite(logits).all().item() and torch.isfinite(values).all().item()
    assert torch.allclose(logits, expected_logits.expand_as(logits), atol=1e-7)
    assert torch.allclose(values, expected_value.expand_as(values), atol=1e-7)


def test_heads_are_raw_logits_and_unbounded_scalar_values(
    model: TicTacToeActorCritic,
) -> None:
    with torch.no_grad():
        model.policy_head.weight.zero_()
        model.policy_head.bias.copy_(torch.arange(9, dtype=torch.float32) - 4)
        model.value_head.weight.zero_()
        model.value_head.bias.fill_(3.5)
    logits, value = model.forward(visible_tensor())
    assert torch.equal(logits, torch.arange(9, dtype=torch.float32) - 4)
    assert value.item() == 3.5


def test_both_heads_reach_shared_parameters_and_masking(
    model: TicTacToeActorCritic,
) -> None:
    tensor = visible_tensor().requires_grad_()
    logits, value = model.forward(tensor)
    mask = encode_tic_tac_toe_legal_actions(
        (
            Action(kind="place", parameters=(("cell", 1),)),
            Action(kind="place", parameters=(("cell", 8),)),
        )
    )
    masked = mask_action_logits(logits, mask)
    assert masked.softmax(dim=-1)[~mask].count_nonzero().item() == 0
    masked[mask].sum().backward(retain_graph=True)  # type: ignore[no-untyped-call]
    policy_grad = model.policy_head.weight.grad
    assert policy_grad is not None
    assert policy_grad[~mask].count_nonzero().item() == 0
    assert policy_grad[mask].abs().sum().item() > 0
    assert model.value_head.weight.grad is None
    for parameter in model.body.parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum().item() > 0
    model.zero_grad(set_to_none=True)
    tensor.grad = None
    value.backward()  # type: ignore[no-untyped-call]
    assert model.policy_head.weight.grad is None
    for parameter in (*model.body.parameters(), *model.value_head.parameters()):
        assert (
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        )
        assert parameter.grad.abs().sum().item() > 0
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item()


def test_forward_preserves_rng_inputs_parameters_and_mode(
    model: TicTacToeActorCritic,
) -> None:
    tensor = visible_tensor()
    original = tensor.clone()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    training_logits, training_value = model.forward(tensor)
    model.eval()
    with torch.inference_mode():
        logits, value = model(tensor)
    assert torch.equal(logits, training_logits) and torch.equal(value, training_value)
    assert not logits.requires_grad and not value.requires_grad
    assert torch.equal(tensor, original)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(
        torch.equal(value, state[key]) for key, value in model.state_dict().items()
    )
    assert not model.training


def test_initialization_reproducible_with_caller_seed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = TicTacToeActorCritic()
        torch.manual_seed(123)
        second = TicTacToeActorCritic()
    assert all(
        torch.equal(value, second.state_dict()[key])
        for key, value in first.state_dict().items()
    )


def test_default_dtype_and_explicit_model_conversion(
    model: TicTacToeActorCritic,
) -> None:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        other = TicTacToeActorCritic()
        assert all(
            p.dtype == torch.float32 and p.device.type == "cpu"
            for p in other.parameters()
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    model.double()
    logits, value = model.forward(visible_tensor().double())
    assert logits.dtype == value.dtype == torch.float64
    with pytest.raises(ValueError, match="dtype"):
        model.forward(visible_tensor())


def test_equivalent_player_views_have_identical_predictions(
    model: TicTacToeActorCritic,
) -> None:
    # Relabel seats and observer together: permitted relative information is identical.
    view = Observation(
        player_id=0,
        current_player=1,
        phase="play",
        data=TicTacToeObservation(board=(0, None, 1, None, 0, None, 1, None, None)),
    )
    relabeled = replace(
        view,
        player_id=1,
        current_player=0,
        data=TicTacToeObservation(board=(1, None, 0, None, 1, None, 0, None, None)),
    )
    first = encode_tic_tac_toe_observation(view)
    second = encode_tic_tac_toe_observation(relabeled)
    assert torch.equal(first, second)
    first_logits, first_value = model.forward(first)
    other_logits, other_value = model.forward(second)
    assert torch.equal(first_logits, other_logits) and torch.equal(
        first_value, other_value
    )


@pytest.mark.parametrize("starter", [0, 1])
def test_engine_decisions_and_terminal_views_integrate(
    model: TicTacToeActorCritic, starter: int
) -> None:
    game = TicTacToeEngine(starting_player=starter)
    game.reset(seed=123)
    for cell in (0, 3, 1, 4, 2):
        actor = game.current_player()
        assert actor is not None
        view, legal = game.decision_for(actor)
        logits, value = model.forward(encode_tic_tac_toe_observation(view))
        mask = encode_tic_tac_toe_legal_actions(legal)
        probabilities = mask_action_logits(logits, mask).softmax(dim=-1)
        assert probabilities[~mask].count_nonzero().item() == 0
        assert torch.isfinite(value).item()
        game.step(Action(kind="place", parameters=(("cell", cell),)))
    terminal_inputs = torch.stack(
        [encode_tic_tac_toe_observation(game.observe(p)) for p in (0, 1)]
    )
    logits, values = model.forward(terminal_inputs)
    assert logits.shape == (2, 9) and values.shape == (2,)
    assert torch.isfinite(values).all().item()
    with pytest.raises(ValueError, match="legal action"):
        mask_action_logits(logits, torch.zeros(2, 9, dtype=torch.bool))


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros(54),
        torch.zeros(3, 3),
        torch.zeros(5, 3, 3),
        torch.zeros(6, 9, 1),
        torch.zeros(0, 6, 3, 3),
        torch.zeros(6, 3, 3, dtype=torch.int64),
        torch.zeros(6, 3, 3, dtype=torch.float64),
        torch.zeros(6, 3, 3, dtype=torch.complex64),
        torch.full((6, 3, 3), float("nan")),
        torch.full((6, 3, 3), float("inf")),
        torch.full((6, 3, 3), float("-inf")),
        torch.empty(6, 3, 3, device="meta"),
        cast(Any, object()),
    ],
)
def test_incompatible_inputs_reject(
    model: TicTacToeActorCritic, tensor: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        model.forward(tensor)
