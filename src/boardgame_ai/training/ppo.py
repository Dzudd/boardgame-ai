"""Bounded CPU/CUDA PPO updates over detached player-visible rollout samples."""

from dataclasses import dataclass, fields
from math import isfinite
from time import monotonic

import torch

from boardgame_ai.core import create_rng
from boardgame_ai.models import ConnectFourActorCritic, TicTacToeActorCritic
from boardgame_ai.models.masking import mask_action_logits
from boardgame_ai.training.advantages import generalized_advantage_estimation
from boardgame_ai.training.rollout import PlayerTrajectory


@dataclass(frozen=True, slots=True, kw_only=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        for name in (
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_ratio",
            "value_coefficient",
            "entropy_coefficient",
            "max_grad_norm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError("PPO coefficients must be finite numbers")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Learning rate and gradient bound must be positive")
        if (
            not 0 <= self.gamma <= 1
            or not 0 <= self.gae_lambda <= 1
            or not 0 < self.clip_ratio < 1
        ):
            raise ValueError("Invalid discount, GAE, or clipping coefficient")
        if self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("Loss coefficients must be nonnegative")
        if any(
            type(v) is not int or v <= 0 for v in (self.epochs, self.minibatch_size)
        ):
            raise ValueError("Epochs and minibatch size must be positive integers")
        if type(self.normalize_advantages) is not bool:
            raise ValueError("normalize_advantages must be boolean")


_DEFAULT_CONFIG = PPOConfig()


@dataclass(frozen=True, slots=True, kw_only=True)
class PPOBatch:
    """Owned float32 samples after per-segment GAE; targets never carry graphs."""

    observations: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def to(self, device: torch.device | str) -> "PPOBatch":
        """Explicitly transfer all samples to one device without changing dtype."""
        return PPOBatch(
            **{f.name: getattr(self, f.name).to(device) for f in fields(self)}
        )

    def subset(self, indices: torch.Tensor) -> "PPOBatch":
        return PPOBatch(
            **{f.name: getattr(self, f.name)[indices] for f in fields(self)}
        )


def validate_ppo_batch(batch: PPOBatch) -> None:
    _validate_ppo_batch(batch, (6, 3, 3), 9)


def _validate_ppo_batch(
    batch: PPOBatch, observation_shape: tuple[int, ...], action_count: int
) -> None:
    n = batch.actions.numel()
    shapes = {
        "observations": (n, *observation_shape),
        "legal_masks": (n, action_count),
        "actions": (n,),
        "old_log_probs": (n,),
        "advantages": (n,),
        "returns": (n,),
    }
    if n == 0:
        raise ValueError("PPO requires at least one decision sample")
    device = batch.observations.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("PPO batches require CPU or CUDA")
    finite_checks = []
    for name, shape in shapes.items():
        tensor = getattr(batch, name)
        dtype = (
            torch.bool
            if name == "legal_masks"
            else torch.int64
            if name == "actions"
            else torch.float32
        )
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
            raise ValueError("PPO batch shapes/dtypes/devices must match")
        if tensor.is_floating_point():
            finite_checks.append(torch.isfinite(tensor).all())
    if not torch.stack(finite_checks).all().item():
        raise ValueError("PPO samples must be finite")
    if not ((batch.actions >= 0) & (batch.actions < action_count)).all().item():
        raise ValueError("PPO action indices must be in range")
    if not batch.legal_masks.gather(1, batch.actions[:, None]).all().item():
        raise ValueError("PPO recorded actions must be legal")
    if (batch.old_log_probs > 0).any().item():
        raise ValueError("Recorded log probabilities must be nonpositive")


def prepare_ppo_batch(
    segments: tuple[PlayerTrajectory, ...], config: PPOConfig = _DEFAULT_CONFIG
) -> PPOBatch:
    """Compute GAE within each segment, then combine samples without mixing time.

    Callers supply only segments collected with the model being updated; opponent
    samples from other policies must be excluded. Rewards keep the player decision
    clock. Empty segments are skipped; an entirely empty batch rejects.
    """
    batches: list[PPOBatch] = []
    for segment in segments:
        advantages, returns = generalized_advantage_estimation(
            segment.rewards,
            segment.values,
            bootstrap_value=segment.bootstrap_value,
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
        if not len(segment.actions):
            if any(
                t.numel()
                for t in (
                    batch.observations,
                    batch.legal_masks,
                    batch.old_log_probs,
                    batch.advantages,
                    batch.returns,
                )
            ):
                raise ValueError("Empty segments must contain no samples")
            continue
        validate_ppo_batch(batch)
        batches.append(batch)
    if not batches:
        raise ValueError("PPO requires nonempty player segments")
    combined = PPOBatch(
        **{
            f.name: torch.cat([getattr(b, f.name) for b in batches])
            for f in fields(PPOBatch)
        }
    )
    validate_ppo_batch(combined)
    return combined


@dataclass(frozen=True, slots=True, kw_only=True)
class PPOLoss:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


def ppo_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    batch: PPOBatch,
    config: PPOConfig = _DEFAULT_CONFIG,
) -> PPOLoss:
    """Clipped policy objective + weighted half-MSE - legal-action entropy bonus.

    No value clipping is applied. Old probabilities/advantages/returns detach
    defensively. Illegal entropy terms use zero log values to avoid 0 * -inf.
    This helper does not normalize advantages; the trainer normalizes once over
    the full batch using population standard deviation with a 1e-8 floor.
    """
    validate_ppo_batch(batch)
    return _validated_ppo_loss(logits, values, batch, config)


def _validated_ppo_loss(
    logits: torch.Tensor, values: torch.Tensor, batch: PPOBatch, config: PPOConfig
) -> PPOLoss:
    # Internal slices of the trainer-owned validated batch need no repeated input
    # validation. Output checks remain on every minibatch; public ppo_loss checks all.
    if (
        logits.shape != batch.legal_masks.shape
        or values.shape != batch.actions.shape
        or logits.dtype != torch.float32
        or values.dtype != torch.float32
        or logits.device != batch.observations.device
        or values.device != batch.observations.device
    ):
        raise ValueError("Model outputs must match the float32 PPO batch device")
    if not torch.isfinite(values).all().item():
        raise ValueError("Predicted values must be finite")
    log_probs = mask_action_logits(logits, batch.legal_masks).log_softmax(dim=-1)
    selected = log_probs.gather(1, batch.actions[:, None]).squeeze(1)
    log_ratio = selected - batch.old_log_probs.detach()
    ratio = log_ratio.exp()
    advantages = batch.advantages.detach()
    surrogate = torch.minimum(
        ratio * advantages,
        ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantages,
    )
    policy = -surrogate.mean()
    value_loss = 0.5 * (values - batch.returns.detach()).square().mean()
    safe_logs = torch.where(batch.legal_masks, log_probs, torch.zeros_like(log_probs))
    entropy = -(log_probs.exp() * safe_logs).sum(dim=-1).mean()
    approximate_kl = ((ratio - 1) - log_ratio).mean().detach()
    clip_fraction = ((ratio - 1).abs() > config.clip_ratio).float().mean().detach()
    total = (
        policy
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * entropy
    )
    if (
        not torch.isfinite(
            torch.stack((total, policy, value_loss, entropy, approximate_kl))
        )
        .all()
        .item()
    ):
        raise ValueError("PPO loss must remain finite")
    return PPOLoss(
        total=total,
        policy=policy,
        value=value_loss,
        entropy=entropy,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PPOUpdate:
    num_samples: int
    optimizer_steps: int
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float


def _check_deadline(deadline: float | None) -> None:
    if deadline is None:
        return
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not isfinite(deadline)
    ):
        raise ValueError("Deadline must be a finite monotonic-clock time")
    if monotonic() >= deadline:
        raise TimeoutError("PPO optimization deadline reached")


class _PPOTrainer[Model: (TicTacToeActorCritic, ConnectFourActorCritic)]:
    """Own Adam and shuffle RNG; update one supplied on-policy batch.

    No rollout scheduling, game access, opponent selection, or training experiment.
    Update errors after successful minibatches do not roll those steps back.
    """

    def __init__(
        self,
        model: Model,
        config: PPOConfig = _DEFAULT_CONFIG,
        *,
        shuffle_seed: int | None = None,
    ) -> None:
        device = model.policy_head.weight.device
        if device.type not in ("cpu", "cuda") or any(
            p.dtype != torch.float32 or p.device != device for p in model.parameters()
        ):
            raise ValueError(
                "PPO trainer requires CPU float32 or CUDA float32 parameters"
            )
        self.device = device
        self.model: Model = model
        self.config = config
        self.generator = torch.Generator(device="cpu").manual_seed(
            create_rng(shuffle_seed).getrandbits(64)
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.update_count = 0
        self.optimizer_steps = 0

    def _validate_batch(self, batch: PPOBatch) -> None:
        validate_ppo_batch(batch)

    def _check_update_deadline(self, deadline: float | None) -> None:
        _check_deadline(deadline)

    def update(self, batch: PPOBatch, *, deadline: float | None = None) -> PPOUpdate:
        self._validate_batch(batch)
        self._check_update_deadline(deadline)
        # Snapshot input samples and freeze targets across all update epochs.
        batch = PPOBatch(
            **{f.name: getattr(batch, f.name).detach().clone() for f in fields(batch)}
        )
        if self.config.normalize_advantages:
            adv = batch.advantages
            normalized = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
            if not torch.isfinite(normalized).all().item():
                raise ValueError("Normalized advantages must remain finite")
            batch = PPOBatch(
                observations=batch.observations,
                legal_masks=batch.legal_masks,
                actions=batch.actions,
                old_log_probs=batch.old_log_probs,
                advantages=normalized,
                returns=batch.returns,
            )
        batch = batch.to(self.device)
        # Accumulate in float64 as the former Python float totals did; one host
        # transfer per update instead of six synchronizations per minibatch.
        totals = torch.zeros(6, dtype=torch.float64, device=self.device)
        n = len(batch.actions)
        steps = 0
        for _ in range(self.config.epochs):
            self._check_update_deadline(deadline)
            order = torch.randperm(n, generator=self.generator, device="cpu")
            for start in range(0, n, self.config.minibatch_size):
                self._check_update_deadline(deadline)
                indices = order[start : start + self.config.minibatch_size]
                part = batch.subset(indices.to(self.device))
                logits, values = self.model.forward(part.observations)
                loss = _validated_ppo_loss(logits, values, part, self.config)
                self.optimizer.zero_grad(set_to_none=True)
                loss.total.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.optimizer_steps += 1
                steps += 1
                totals += torch.stack(
                    (
                        loss.total,
                        loss.policy,
                        loss.value,
                        loss.entropy,
                        loss.approximate_kl,
                        loss.clip_fraction,
                    )
                ).detach().to(torch.float64) * len(indices)
        self.update_count += 1
        means = (totals / (n * self.config.epochs)).tolist()
        return PPOUpdate(
            num_samples=n,
            optimizer_steps=steps,
            total_loss=means[0],
            policy_loss=means[1],
            value_loss=means[2],
            entropy=means[3],
            approximate_kl=means[4],
            clip_fraction=means[5],
        )


class PPOTrainer(_PPOTrainer[TicTacToeActorCritic]):
    """Tic-Tac-Toe PPO adapter; original shapes, defaults and arithmetic."""
