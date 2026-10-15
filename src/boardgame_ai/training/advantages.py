"""GAE over one player's decision sequence, never across seats or episodes."""

from math import isfinite

import torch


def generalized_advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    bootstrap_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached advantages and lambda-return targets for one segment.

    rewards/values are finite 1-D floating tensors with identical shape, dtype,
    and device. Rewards sum intervening engine increments for this player's
    latest decision; gamma and lambda advance per player decision, not per
    engine step. bootstrap_value must be zero at true termination, or the
    player's value at a live cutoff. Never concatenate episodes or player seats.

    delta[t] = reward[t] + gamma * next_value - value[t]
    advantage[t] = delta[t] + gamma * lambda * advantage[t+1]
    return[t] = advantage[t] + value[t]. Empty segments return empty tensors.
    """
    for coefficient in (gamma, gae_lambda):
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, (int, float))
            or not isfinite(coefficient)
            or not 0 <= coefficient <= 1
        ):
            raise ValueError("GAE coefficients must be finite numbers in [0, 1]")
    if (
        isinstance(bootstrap_value, bool)
        or not isinstance(bootstrap_value, (int, float))
        or not isfinite(bootstrap_value)
    ):
        raise ValueError("Bootstrap value must be finite")
    if (
        rewards.ndim != 1
        or rewards.shape != values.shape
        or rewards.dtype != values.dtype
        or rewards.device != values.device
        or not rewards.is_floating_point()
    ):
        raise ValueError("Rewards/values require identical floating vectors")
    if (
        not torch.isfinite(rewards).all().item()
        or not torch.isfinite(values).all().item()
    ):
        raise ValueError("Rewards/values must be finite")
    with torch.no_grad():
        next_value = values.new_tensor(bootstrap_value)
        if not torch.isfinite(next_value).item():
            raise ValueError("Bootstrap value must fit the tensor dtype")
        advantages = torch.empty_like(values)
        running = values.new_zeros(())
        for index in range(len(values) - 1, -1, -1):
            delta = rewards[index] + gamma * next_value - values[index]
            running = delta + gamma * gae_lambda * running
            advantages[index] = running
            next_value = values[index]
        returns = advantages + values
        if (
            not torch.isfinite(advantages).all().item()
            or not torch.isfinite(returns).all().item()
        ):
            raise ValueError("GAE targets must remain finite")
        return advantages, returns
