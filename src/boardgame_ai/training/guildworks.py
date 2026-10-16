"""Full-episode recurrent PPO for Guildworks, separate from Tic-Tac-Toe PPO.

Each minibatch contains whole seat episodes; no padding, chunking, burn-in,
replayed trajectories, saved hidden states or cross-seat recurrent context.
"""

from copy import deepcopy
from dataclasses import dataclass, fields

import torch

from boardgame_ai.core import GameEngine, create_rng
from boardgame_ai.games.guildworks.observation import GuildworksObservation
from boardgame_ai.models.guildworks import (
    ACTION_COUNT,
    ACTIONS,
    FEATURES,
    GuildworksRecurrentPolicy,
    RecurrentGuildworksAgent,
    encode_actions,
)
from boardgame_ai.models.masking import mask_action_logits
from boardgame_ai.training.advantages import generalized_advantage_estimation
from boardgame_ai.training.ppo import PPOConfig, PPOUpdate

DEFAULT_CONFIG = PPOConfig(gamma=1.0, epochs=2, minibatch_size=8)


@dataclass(frozen=True, slots=True, kw_only=True)
class SeatEpisode:
    """One seat's detached visible stream and own learning targets; never public."""

    player_id: int
    tokens: torch.Tensor
    positions: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor

    def owned_copy(self) -> "SeatEpisode":
        return SeatEpisode(
            player_id=self.player_id,
            **{
                field.name: getattr(self, field.name).detach().cpu().clone()
                for field in fields(self)
                if field.name != "player_id"
            },
        )


def validate_episode(episode: SeatEpisode) -> None:
    """Only complete Guildworks seat episodes are accepted in this milestone."""
    if type(episode.player_id) is not int or episode.player_id not in (0, 1):
        raise ValueError("Invalid episode seat")
    shapes = {
        "positions": ((16,), torch.int64),
        "masks": ((16, ACTION_COUNT), torch.bool),
        "actions": ((16,), torch.int64),
        "log_probs": ((16,), torch.float32),
        "values": ((16,), torch.float32),
        "rewards": ((16,), torch.float32),
    }
    tokens = episode.tokens
    if (
        tokens.ndim != 2
        or tokens.shape[1] != FEATURES
        or len(tokens) < 16
        or tokens.dtype != torch.float32
        or tokens.device.type != "cpu"
        or not torch.isfinite(tokens).all().item()
    ):
        raise ValueError("Invalid visible episode tokens")
    for name, (shape, dtype) in shapes.items():
        tensor = getattr(episode, name)
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device.type != "cpu"
        ):
            raise ValueError("Invalid episode shape, dtype or device")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError("Nonfinite episode data")
    positions = episode.positions
    if (
        positions[0] < 0
        or positions[-1] != len(tokens) - 1
        or not (positions[1:] > positions[:-1]).all().item()
        or not (tokens[positions, 0] == 1).all().item()
        or not ((episode.actions >= 0) & (episode.actions < ACTION_COUNT)).all().item()
        or not episode.masks.gather(1, episode.actions[:, None]).all().item()
        or (episode.log_probs > 0).any().item()
    ):
        raise ValueError("Invalid episode decisions or action legality")


def collect_guildworks(
    game: GameEngine[GuildworksObservation],
    model: GuildworksRecurrentPolicy,
    *,
    engine_seed: int,
    sampling_seeds: tuple[int, int],
) -> tuple[SeatEpisode, SeatEpisode]:
    """Trusted complete-game collection; agents get only decision_for inputs.

    Each seat's reward belongs to its latest own decision, including intervening
    increments until its next decision/termination. No sign flip or time advance
    is inferred from alternating actors. Terminal continuation is exactly zero.
    """
    agents = (RecurrentGuildworksAgent(model), RecurrentGuildworksAgent(model))
    for agent, seed in zip(agents, sampling_seeds, strict=True):
        agent.reset(seed)
    game.reset(engine_seed)
    positions: list[list[int]] = [[], []]
    masks: list[list[torch.Tensor]] = [[], []]
    actions: list[list[int]] = [[], []]
    logs: list[list[float]] = [[], []]
    values: list[list[float]] = [[], []]
    rewards: list[list[float]] = [[], []]
    for _ in range(32):
        seat = game.current_player()
        if seat is None:
            raise ValueError("Expected a complete 32-decision Guildworks episode")
        view, legal = game.decision_for(seat)
        agent = agents[seat]
        action = agent.select_action(view, legal)
        positions[seat].append(len(agent.tokens) - 1)
        masks[seat].append(encode_actions(legal))
        actions[seat].append(ACTIONS.index(action))
        logs[seat].append(agent.last_log_prob)
        values[seat].append(agent.last_value)
        rewards[seat].append(0.0)
        result = game.step(action)
        if (
            result.next_player != game.current_player()
            or result.terminated != game.is_terminal()
        ):
            raise RuntimeError("Engine result disagrees with lifecycle")
        for player, reward in enumerate(result.rewards):
            if rewards[player]:
                rewards[player][-1] += reward
            elif reward != 0:
                raise ValueError(
                    "Reward before a player's first decision is unsupported"
                )
    if not game.is_terminal():
        raise ValueError("Expected terminal Guildworks episode")
    episodes = []
    for seat in (0, 1):
        episode = SeatEpisode(
            player_id=seat,
            tokens=torch.stack(agents[seat].tokens),
            positions=torch.tensor(positions[seat], dtype=torch.int64),
            masks=torch.stack(masks[seat]),
            actions=torch.tensor(actions[seat], dtype=torch.int64),
            log_probs=torch.tensor(logs[seat], dtype=torch.float32),
            values=torch.tensor(values[seat], dtype=torch.float32),
            rewards=torch.tensor(rewards[seat], dtype=torch.float32),
        )
        validate_episode(episode)
        episodes.append(episode)
    return episodes[0], episodes[1]


class RecurrentPPOTrainer:
    """Minibatch size counts full seat episodes, not individual decisions."""

    def __init__(
        self,
        model: GuildworksRecurrentPolicy,
        config: PPOConfig = DEFAULT_CONFIG,
        *,
        shuffle_seed: int | None = None,
    ) -> None:
        self.model, self.config = model, config
        self.device = model.policy_head.weight.device
        if self.device.type not in ("cpu", "cuda") or any(
            p.device != self.device or p.dtype != torch.float32
            for p in model.parameters()
        ):
            raise ValueError("Expected float32 CPU/CUDA model")
        self.generator = torch.Generator().manual_seed(
            create_rng(shuffle_seed).getrandbits(64)
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.update_count = 0
        self.optimizer_steps = 0

    def update(self, episodes: tuple[SeatEpisode, ...]) -> PPOUpdate:
        if not episodes:
            raise ValueError("PPO requires complete seat episodes")
        for episode in episodes:
            validate_episode(episode)
        owned = tuple(e.owned_copy() for e in episodes)
        targets = []
        for episode in owned:
            targets.append(
                generalized_advantage_estimation(
                    episode.rewards,
                    episode.values,
                    bootstrap_value=0.0,
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                )
            )
        advantages = torch.cat([a for a, _ in targets])
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / advantages.std(
                unbiased=False
            ).clamp_min(1e-8)
        advantages = advantages.reshape(len(owned), 16).to(self.device)
        # Validate behavior against the current weights before any optimizer step.
        # This catches accidental stale-policy batches; callers still own provenance.
        behavior_model = deepcopy(self.model).cpu().eval()
        with torch.no_grad():
            for episode in owned:
                logits, values, _ = behavior_model(episode.tokens)
                positions = episode.positions
                logs = mask_action_logits(logits[positions], episode.masks).log_softmax(
                    -1
                )
                selected = logs.gather(1, episode.actions[:, None]).squeeze(1)
                if not (
                    torch.allclose(
                        selected.cpu(), episode.log_probs, atol=1e-5, rtol=1e-5
                    )
                    and torch.allclose(
                        values[positions].cpu(), episode.values, atol=1e-5, rtol=1e-5
                    )
                ):
                    raise ValueError("Episode behavior does not match current policy")
        totals = torch.zeros(6, dtype=torch.float64, device=self.device)
        steps = 0
        for _ in range(self.config.epochs):
            order = torch.randperm(len(owned), generator=self.generator).tolist()
            for start in range(0, len(order), self.config.minibatch_size):
                indices = order[start : start + self.config.minibatch_size]
                all_logits, all_values = [], []
                for index in indices:
                    episode = owned[index]
                    # Full BPTT from zero with current weights on EVERY epoch.
                    logits, values, _ = self.model(episode.tokens.to(self.device))
                    positions = episode.positions.to(self.device)
                    all_logits.append(logits[positions])
                    all_values.append(values[positions])
                mask = torch.cat([owned[i].masks for i in indices]).to(self.device)
                action = torch.cat([owned[i].actions for i in indices]).to(self.device)
                old_logs = torch.cat([owned[i].log_probs for i in indices]).to(
                    self.device
                )
                returns = torch.cat([targets[i][1] for i in indices]).to(self.device)
                adv = advantages[indices].flatten()
                logits, values = torch.cat(all_logits), torch.cat(all_values)
                logs = mask_action_logits(logits, mask).log_softmax(-1)
                selected = logs.gather(1, action[:, None]).squeeze(1)
                log_ratio = selected - old_logs
                ratio = log_ratio.exp()
                policy = -torch.minimum(
                    ratio * adv,
                    ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                    * adv,
                ).mean()
                value_loss = 0.5 * (values - returns).square().mean()
                entropy = -(logs.exp() * torch.where(mask, logs, 0.0)).sum(-1).mean()
                kl = ((ratio - 1) - log_ratio).mean()
                clipped = ((ratio - 1).abs() > self.config.clip_ratio).float().mean()
                loss = (
                    policy
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                metrics = torch.stack((loss, policy, value_loss, entropy, kl, clipped))
                if not torch.isfinite(metrics).all().item():
                    raise ValueError("Nonfinite recurrent PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.optimizer_steps += 1
                steps += 1
                totals += metrics.detach().to(torch.float64) * len(indices)
        self.update_count += 1
        means = (totals / (len(owned) * self.config.epochs)).tolist()
        return PPOUpdate(
            num_samples=len(owned) * 16,
            optimizer_steps=steps,
            total_loss=means[0],
            policy_loss=means[1],
            value_loss=means[2],
            entropy=means[3],
            approximate_kl=means[4],
            clip_fraction=means[5],
        )
