"""Independent PPO objective checks and bounded update integration."""

from dataclasses import replace
from math import log
from typing import Any

import pytest
import torch

from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.training import (
    PPOBatch,
    PPOConfig,
    PPOTrainer,
    collect_tic_tac_toe_rollout,
    ppo_loss,
    prepare_ppo_batch,
)


@pytest.fixture
def model() -> TicTacToeActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return TicTacToeActorCritic()


def sample_batch() -> PPOBatch:
    return PPOBatch(
        observations=torch.zeros(2, 6, 3, 3),
        legal_masks=torch.tensor([[True, True] + [False] * 7] * 2),
        actions=torch.tensor([0, 0]),
        old_log_probs=torch.tensor([log(0.5)] * 2),
        advantages=torch.tensor([2.0, -3.0]),
        returns=torch.tensor([1.0, -1.0]),
    )


def test_policy_clipping_both_advantage_signs_and_detached_targets() -> None:
    batch = sample_batch()
    batch = replace(
        batch,
        old_log_probs=batch.old_log_probs.requires_grad_(),
        advantages=batch.advantages.requires_grad_(),
        returns=batch.returns.requires_grad_(),
    )
    logits = torch.tensor(
        [[log(0.8), log(0.2)] + [100.0] * 7, [log(0.2), log(0.8)] + [100.0] * 7],
        requires_grad=True,
    )
    values = torch.tensor([0.5, 0.5], requires_grad=True)
    loss = ppo_loss(
        logits, values, batch, PPOConfig(value_coefficient=0.5, entropy_coefficient=0.1)
    )
    # ratios [1.6,.4]; min surrogate [1.2*2, .8*(-3)] = [2.4,-2.4].
    expected_entropy = -(0.8 * log(0.8) + 0.2 * log(0.2))
    assert loss.policy.item() == pytest.approx(0.0, abs=1e-6)
    assert loss.value.item() == pytest.approx(0.625)
    assert loss.entropy.item() == pytest.approx(expected_entropy)
    assert loss.total.item() == pytest.approx(
        0.5 * 0.625 - 0.1 * expected_entropy, abs=1e-6
    )
    assert loss.clip_fraction.item() == 1.0
    loss.total.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None and logits.grad[:, 2:].count_nonzero().item() == 0
    assert values.grad is not None and torch.isfinite(values.grad).all().item()
    assert (
        batch.old_log_probs.grad is None
        and batch.advantages.grad is None
        and batch.returns.grad is None
    )


def test_unclipped_branch_and_single_legal_entropy() -> None:
    batch = sample_batch()
    logits = torch.zeros(2, 9, requires_grad=True)
    loss = ppo_loss(logits, torch.zeros(2), batch)
    assert loss.policy.item() == pytest.approx(0.5)
    assert loss.approximate_kl.item() == pytest.approx(0.0)
    assert loss.clip_fraction.item() == 0.0
    singleton = replace(
        batch,
        legal_masks=torch.tensor([[True] + [False] * 8] * 2),
        old_log_probs=torch.zeros(2),
    )
    single_loss = ppo_loss(logits, torch.zeros(2), singleton)
    assert single_loss.entropy.item() == 0.0
    single_loss.total.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None and torch.isfinite(logits.grad).all().item()


def test_prepare_runs_gae_before_combining_seats_and_owns_storage(
    model: TicTacToeActorCritic,
) -> None:
    rollout = collect_tic_tac_toe_rollout(
        TicTacToeEngine(), (model, model), max_steps=9, sampling_seeds=(1, 2)
    )
    config = PPOConfig(gamma=1.0, gae_lambda=1.0)
    batch = prepare_ppo_batch(rollout.players, config)
    for seat, trajectory in enumerate(rollout.players):
        offset = 0 if seat == 0 else len(rollout.players[0].actions)
        length = len(trajectory.actions)
        assert torch.allclose(
            batch.returns[offset : offset + length],
            torch.full((length,), rollout.total_rewards[seat]),
            atol=1e-6,
        )
    original = batch.observations.clone()
    rollout.players[0].observations.zero_()
    assert torch.equal(batch.observations, original)
    assert not batch.advantages.requires_grad and not batch.returns.requires_grad


def test_update_changes_both_heads_without_mutating_batch_or_global_rng(
    model: TicTacToeActorCritic,
) -> None:
    config = PPOConfig(
        epochs=2, minibatch_size=2, normalize_advantages=False, max_grad_norm=0.1
    )
    rollout = collect_tic_tac_toe_rollout(
        TicTacToeEngine(), (model, model), max_steps=9, sampling_seeds=(3, 4)
    )
    batch = prepare_ppo_batch(rollout.players, config)
    tensors = [
        batch.observations.clone(),
        batch.advantages.clone(),
        batch.returns.clone(),
    ]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    trainer = PPOTrainer(model, config, shuffle_seed=123)
    rng = torch.random.get_rng_state().clone()
    report = trainer.update(batch)
    assert trainer.update_count == 1
    assert report.num_samples == rollout.num_steps
    assert report.optimizer_steps == 2 * ((rollout.num_steps + 1) // 2)
    assert trainer.optimizer_steps == report.optimizer_steps
    assert not torch.equal(model.policy_head.weight, before["policy_head.weight"])
    assert not torch.equal(model.value_head.weight, before["value_head.weight"])
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(batch.observations, tensors[0])
    assert torch.equal(batch.advantages, tensors[1]) and torch.equal(
        batch.returns, tensors[2]
    )
    grad_norm = torch.sqrt(
        torch.stack(
            [p.grad.square().sum() for p in model.parameters() if p.grad is not None]
        ).sum()
    )
    assert grad_norm.item() <= 0.10001


def test_normalization_singleton_and_constant_advantages(
    model: TicTacToeActorCritic,
) -> None:
    config = PPOConfig(epochs=1, minibatch_size=1)
    batch = sample_batch().subset(torch.tensor([0]))
    trainer = PPOTrainer(model, config, shuffle_seed=1)
    report = trainer.update(batch)
    assert report.policy_loss == 0.0
    assert report.num_samples == 1 and report.optimizer_steps == 1
    constant = replace(sample_batch(), advantages=torch.full((2,), 5.0))
    assert trainer.update(constant).policy_loss == 0.0


@pytest.mark.parametrize(
    "changes",
    [
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"gamma": -1.0},
        {"gae_lambda": 2.0},
        {"clip_ratio": 0.0},
        {"clip_ratio": 1.0},
        {"value_coefficient": -1.0},
        {"entropy_coefficient": -1.0},
        {"epochs": 0},
        {"epochs": True},
        {"minibatch_size": 1.5},
        {"max_grad_norm": 0.0},
        {"normalize_advantages": 1},
    ],
)
def test_invalid_config_rejects(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PPOConfig(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"actions": torch.tensor([0, 9])},
        {"actions": torch.tensor([0.0, 0.0])},
        {"actions": torch.tensor([2, 0])},
        {"old_log_probs": torch.ones(2)},
        {"old_log_probs": torch.tensor([float("-inf"), 0.0])},
        {"returns": torch.ones(3)},
        {"advantages": torch.tensor([float("nan"), 1.0])},
        {"legal_masks": torch.zeros(2, 9, dtype=torch.bool)},
        {"observations": torch.zeros(2, 54)},
        {"returns": torch.zeros(2, dtype=torch.float64)},
    ],
)
def test_invalid_batch_rejects_before_update(
    model: TicTacToeActorCritic, changes: dict[str, Any]
) -> None:
    trainer = PPOTrainer(model, shuffle_seed=1)
    before = model.policy_head.weight.detach().clone()
    rng = trainer.generator.get_state().clone()
    with pytest.raises(ValueError):
        trainer.update(replace(sample_batch(), **changes))
    assert torch.equal(model.policy_head.weight, before)
    assert torch.equal(trainer.generator.get_state(), rng)
    assert trainer.optimizer_steps == trainer.update_count == 0
    assert not trainer.optimizer.state


def test_empty_batch_rejects_and_empty_segments_are_skipped(
    model: TicTacToeActorCritic,
) -> None:
    empty = sample_batch().subset(torch.empty(0, dtype=torch.int64))
    with pytest.raises(ValueError):
        PPOTrainer(model).update(empty)
    rollout = collect_tic_tac_toe_rollout(
        TicTacToeEngine(), (model, model), max_steps=1
    )
    assert len(prepare_ppo_batch(rollout.players).actions) == 1
    with pytest.raises(ValueError):
        prepare_ppo_batch(())


def test_nonfinite_ratio_and_value_reject() -> None:
    batch = replace(sample_batch(), old_log_probs=torch.full((2,), -1e30))
    with pytest.raises(ValueError, match="finite"):
        ppo_loss(torch.zeros(2, 9), torch.zeros(2), batch)
    with pytest.raises(ValueError, match="finite"):
        ppo_loss(torch.zeros(2, 9), torch.full((2,), float("nan")), sample_batch())


def test_trainer_cpu_float32_contract(model: TicTacToeActorCritic) -> None:
    with pytest.raises(ValueError, match="CPU float32"):
        PPOTrainer(model.double())


def test_policy_clipped_gradient_is_zero_for_saturated_samples() -> None:
    logits = torch.tensor(
        [[1.38629436, 0.0] + [100.0] * 7, [0.0, 1.38629436] + [100.0] * 7],
        requires_grad=True,
    )
    loss = ppo_loss(
        logits,
        torch.zeros(2),
        sample_batch(),
        PPOConfig(value_coefficient=0.0, entropy_coefficient=0.0),
    )
    loss.total.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None and logits.grad.count_nonzero().item() == 0


def test_update_rejects_normalization_overflow_before_rng_change(
    model: TicTacToeActorCritic,
) -> None:
    trainer = PPOTrainer(model, shuffle_seed=1)
    rng = trainer.generator.get_state().clone()
    extreme = replace(sample_batch(), advantages=torch.full((2,), 3e38))
    with pytest.raises(ValueError, match="finite"):
        trainer.update(extreme)
    assert torch.equal(trainer.generator.get_state(), rng)
    assert trainer.optimizer_steps == 0
