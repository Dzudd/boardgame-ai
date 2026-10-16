"""Connect Four adapter to the existing PPO arithmetic and private optimizer."""

from dataclasses import fields

import torch

from boardgame_ai.models.connect_four import ConnectFourActorCritic
from boardgame_ai.training.advantages import generalized_advantage_estimation
from boardgame_ai.training.connect_four import ConnectFourTrajectory
from boardgame_ai.training.ppo import (
    PPOBatch,
    PPOConfig,
    PPOLoss,
    PPOUpdate,
    _check_deadline,
    _PPOTrainer,
    _validate_ppo_batch,
    _validated_ppo_loss,
)

DEFAULT_CONFIG = PPOConfig(gamma=1.0, epochs=2, minibatch_size=32)
LEARNER_ID = "connect-four-ppo-v1"


def validate_connect_four_batch(batch: PPOBatch) -> None:
    """Validate explicit Connect Four dimensions; old PPO remains Tic-Tac-Toe."""
    _validate_ppo_batch(batch, (6, 6, 7), 7)


def prepare_connect_four_batch(
    segments: tuple[ConnectFourTrajectory, ...], config: PPOConfig = DEFAULT_CONFIG
) -> PPOBatch:
    """Compute complete own-seat GAE before combining independently owned samples.

    Provenance is the caller's responsibility: use only current-policy self-play.
    This performs no baseline selection, reward shaping or replay sampling.
    """
    if not isinstance(segments, tuple) or not segments:
        raise ValueError("Provide nonempty complete Connect Four seat sequences")
    batches = []
    for segment in segments:
        if (
            not isinstance(segment, ConnectFourTrajectory)
            or type(segment.player_id) is not int
            or segment.player_id not in (0, 1)
            or isinstance(segment.bootstrap_value, bool)
            or segment.bootstrap_value != 0.0
        ):
            raise ValueError("Expected a complete own-seat Connect Four trajectory")
        n = segment.actions.numel()
        steps = segment.step_indices
        if (
            n == 0
            or n > 21
            or steps.shape != (n,)
            or steps.dtype != torch.int64
            or steps.device.type != "cpu"
            or int(steps[0]) not in (0, 1)
            or int(steps[-1]) >= 42
            or not ((steps[1:] - steps[:-1]) == 2).all().item()
        ):
            raise ValueError("Invalid complete Connect Four decision step sequence")
        if segment.observations.device.type != "cpu":
            raise ValueError("Collected Connect Four trajectories must be on CPU")
        advantages, returns = generalized_advantage_estimation(
            segment.rewards,
            segment.values,
            bootstrap_value=0.0,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        batch = PPOBatch(
            observations=segment.observations.detach().clone(),
            legal_masks=segment.legal_masks.detach().clone(),
            actions=segment.actions.detach().clone(),
            old_log_probs=segment.log_probs.detach().clone(),
            advantages=advantages,
            returns=returns,
        )
        validate_connect_four_batch(batch)
        batches.append(batch)
    return PPOBatch(
        **{
            f.name: torch.cat([getattr(b, f.name) for b in batches])
            for f in fields(PPOBatch)
        }
    )


def connect_four_ppo_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    batch: PPOBatch,
    config: PPOConfig = DEFAULT_CONFIG,
) -> PPOLoss:
    """Validated Connect Four entry to the unchanged clipped PPO loss."""
    validate_connect_four_batch(batch)
    return _validated_ppo_loss(logits, values, batch, config)


class ConnectFourPPOTrainer(_PPOTrainer[ConnectFourActorCritic]):
    """Separate typed adapter with saveable successful update boundaries only.

    A failed update may have performed minibatches; reload the saved checkpoint
    before continuing. Validation/deadline rejection before work leaves the
    boundary valid. Run scheduling, metrics recovery and provenance are separate.
    """

    def __init__(
        self,
        model: ConnectFourActorCritic,
        config: PPOConfig = DEFAULT_CONFIG,
        *,
        shuffle_seed: int = 0,
    ) -> None:
        if not isinstance(model, ConnectFourActorCritic):
            raise ValueError("Expected a Connect Four actor-critic model")
        if type(shuffle_seed) is not int or not 0 <= shuffle_seed < 2**64:
            raise ValueError("Shuffle seed must be an unsigned 64-bit integer")
        super().__init__(model, config, shuffle_seed=shuffle_seed)
        self.generator.manual_seed(shuffle_seed)
        self.at_update_boundary = True
        self.samples_seen = 0

    def _validate_batch(self, batch: PPOBatch) -> None:
        validate_connect_four_batch(batch)

    def update(self, batch: PPOBatch, *, deadline: float | None = None) -> PPOUpdate:
        if not self.at_update_boundary:
            raise ValueError("Reload a saved boundary after a failed PPO update")
        validate_connect_four_batch(batch)
        _check_deadline(deadline)
        self.at_update_boundary = False
        result = super().update(batch, deadline=deadline)
        self.samples_seen += result.num_samples
        self.at_update_boundary = True
        return result
