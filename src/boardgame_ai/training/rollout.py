"""Bounded Tic-Tac-Toe collection with separate player decision sequences."""

from dataclasses import dataclass, field
from math import isfinite

import torch

from boardgame_ai.core import GameEngine, create_rng
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.models.encoders import (
    decode_tic_tac_toe_action,
    encode_tic_tac_toe_legal_actions,
    encode_tic_tac_toe_observation,
)
from boardgame_ai.models.masking import mask_action_logits


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayerTrajectory:
    """One player's detached CPU records; tensors are owned but remain mutable.

    Observations (T,6,3,3), masks (T,9), actions/step_indices (T,) int64,
    log_probs/values/rewards (T,) float32. Rewards accumulate until the next
    decision or rollout end. Step indices count global atomic steps from reset.
    bootstrap_value is zero at termination, otherwise a filtered cutoff value.
    This is trusted learning data, not a public log or another player's input.
    """

    player_id: int
    observations: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    step_indices: torch.Tensor
    bootstrap_value: float


@dataclass(frozen=True, slots=True, kw_only=True)
class TicTacToeRollout:
    """One reset-to-boundary segment; termination and truncation are distinct."""

    players: tuple[PlayerTrajectory, PlayerTrajectory]
    num_steps: int
    total_rewards: tuple[float, float]
    terminated: bool
    truncated: bool


@dataclass
class _Pending:
    observations: list[torch.Tensor] = field(default_factory=list)
    masks: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)

    def finish(self, seat: int, bootstrap: float) -> PlayerTrajectory:
        trajectory = PlayerTrajectory(
            player_id=seat,
            observations=torch.stack(self.observations)
            if self.observations
            else torch.empty(0, 6, 3, 3, dtype=torch.float32, device="cpu"),
            legal_masks=torch.stack(self.masks)
            if self.masks
            else torch.empty(0, 9, dtype=torch.bool, device="cpu"),
            actions=torch.tensor(self.actions, dtype=torch.int64, device="cpu"),
            log_probs=torch.tensor(self.log_probs, dtype=torch.float32, device="cpu"),
            values=torch.tensor(self.values, dtype=torch.float32, device="cpu"),
            rewards=torch.tensor(self.rewards, dtype=torch.float32, device="cpu"),
            step_indices=torch.tensor(self.steps, dtype=torch.int64, device="cpu"),
            bootstrap_value=bootstrap,
        )
        for values in (trajectory.log_probs, trajectory.values, trajectory.rewards):
            if not torch.isfinite(values).all().item():
                raise ValueError("Trajectory scalars must fit finite float32 storage")
        return trajectory


def _predict(
    model: TicTacToeActorCritic, encoded: torch.Tensor
) -> tuple[torch.Tensor, float]:
    reference = model.policy_head.weight
    with torch.no_grad():
        logits, value = model.forward(
            encoded.to(device=reference.device, dtype=reference.dtype)
        )
        if (
            logits.shape != (9,)
            or value.shape != ()
            or not torch.isfinite(value).item()
        ):
            raise ValueError("Model must produce nine logits and one finite value")
        return logits.detach().to(device="cpu"), float(value.item())


def collect_tic_tac_toe_rollout(
    game: GameEngine[TicTacToeObservation],
    models: tuple[TicTacToeActorCritic, TicTacToeActorCritic],
    *,
    max_steps: int,
    engine_seed: int | None = None,
    sampling_seeds: tuple[int | None, int | None] = (None, None),
) -> TicTacToeRollout:
    """Reset and collect one bounded game using frozen model predictions.

    Models may be shared between seats. Only detached decision_for inputs reach
    inference. Sampling uses separate CPU generators seeded through private
    create_rng(seed).getrandbits(64), never global Torch/Python/NumPy RNG.
    Models' parameters, gradients, and train/eval modes remain untouched.

    Rewards are incremental per-seat learning signals, accumulated after that
    seat's latest action, including opponent steps. Rewards before a seat's
    first decision appear only in total_rewards. True termination beats a step
    limit and bootstraps zero without terminal inference. Live cutoffs bootstrap
    each nonempty player segment from that player's filtered observe view.
    Empty segments bootstrap zero. Each call resets; no continuation/batching,
    training updates, or exception retries. Errors propagate after any steps.
    """
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if game.num_players != 2 or len(models) != 2 or len(sampling_seeds) != 2:
        raise ValueError("Provide two models and sampling seeds for a two-seat game")
    if any(not isinstance(model, TicTacToeActorCritic) for model in models):
        raise ValueError("Expected Tic-Tac-Toe actor-critic models")
    create_rng(engine_seed)
    generators = tuple(
        torch.Generator(device="cpu").manual_seed(create_rng(seed).getrandbits(64))
        for seed in sampling_seeds
    )
    game.reset(engine_seed)
    pending = (_Pending(), _Pending())
    totals = [0.0, 0.0]
    num_steps = 0
    while True:
        terminated = game.is_terminal()
        actor = game.current_player()
        if terminated != (actor is None):
            raise RuntimeError("Engine actor and termination disagree")
        if terminated or num_steps == max_steps:
            break
        assert actor is not None
        game.validate_player(actor)
        view, legal = game.decision_for(actor)
        if view.player_id != actor or view.current_player != actor:
            raise RuntimeError("Engine decision view does not identify the actor")
        encoded = encode_tic_tac_toe_observation(view)
        mask = encode_tic_tac_toe_legal_actions(legal)
        logits, value = _predict(models[actor], encoded)
        masked = mask_action_logits(logits, mask)
        log_probs = masked.log_softmax(dim=-1)
        index = int(
            torch.multinomial(log_probs.exp(), 1, generator=generators[actor]).item()
        )
        action = decode_tic_tac_toe_action(index, legal)
        record = pending[actor]
        record.observations.append(encoded)
        record.masks.append(mask)
        record.actions.append(index)
        record.log_probs.append(float(log_probs[index].item()))
        record.values.append(value)
        record.rewards.append(0.0)
        record.steps.append(num_steps)
        result = game.step(action)
        if (
            result.next_player != game.current_player()
            or result.terminated != game.is_terminal()
            or len(result.rewards) != 2
        ):
            raise RuntimeError("Engine step result disagrees with engine contract")
        for seat, reward in enumerate(result.rewards):
            totals[seat] += reward
            if pending[seat].rewards:
                pending[seat].rewards[-1] += reward
            if not isfinite(totals[seat]):
                raise ValueError("Accumulated rewards must remain finite")
        num_steps += 1

    trajectories: list[PlayerTrajectory] = []
    for seat in (0, 1):
        bootstrap = 0.0
        if not terminated and pending[seat].actions:
            view = game.observe(seat)
            if view.player_id != seat or view.current_player != actor:
                raise RuntimeError("Engine cutoff view identifies the wrong player")
            _, bootstrap = _predict(models[seat], encode_tic_tac_toe_observation(view))
        trajectories.append(pending[seat].finish(seat, bootstrap))
    return TicTacToeRollout(
        players=(trajectories[0], trajectories[1]),
        num_steps=num_steps,
        total_rewards=(totals[0], totals[1]),
        terminated=terminated,
        truncated=not terminated,
    )
