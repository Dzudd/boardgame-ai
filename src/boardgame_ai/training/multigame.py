"""Latest-policy, complete-game recurrent self-play and equal-game PPO.

Rollouts and targets are trusted learning records, never player inputs. Evaluation
has a separate collector and cannot supply this trainer with opponent samples.
"""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic
from typing import Any

import torch

from boardgame_ai.core import GameEngine
from boardgame_ai.games.breakthrough import BreakthroughEngine
from boardgame_ai.games.connect_four import ConnectFourEngine
from boardgame_ai.games.guildworks import GuildworksEngine
from boardgame_ai.games.guildworks.observation import GuildworksObservation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.models.multigame import (
    GAMES,
    REWARD_SCALES,
    DecisionInput,
    MultiGamePolicy,
    candidate_features,
    observation_tokens,
)
from boardgame_ai.training.advantages import generalized_advantage_estimation
from boardgame_ai.training.connect_four_run import synchronize_device

MAX_STEPS = dict(zip(GAMES, (9, 32, 42, 80), strict=True))


def derive_seed(root: int, index: int, stream: str) -> int:
    return int.from_bytes(
        sha256(f"boardgame-ai/multigame/v1:{root}:{index}:{stream}".encode()).digest()[
            :8
        ],
        "big",
    )


def engine_for(game: str, starter: int = 0) -> GameEngine[Any]:
    if game == "tic_tac_toe":
        return TicTacToeEngine(starting_player=starter)
    if game == "connect_four":
        return ConnectFourEngine(starting_player=starter)
    if game == "guildworks":
        return GuildworksEngine()
    if game == "breakthrough":
        return BreakthroughEngine(starting_player=starter)
    raise ValueError("Unsupported game")


def deadline_check(deadline: float, device: torch.device | None = None) -> None:
    if device is not None:
        synchronize_device(device)
    if monotonic() >= deadline:
        raise TimeoutError("Cumulative time cap reached")


@dataclass(frozen=True)
class SeatEpisode:
    game: str
    decisions: tuple[DecisionInput, ...]
    actions: tuple[int, ...]
    logs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor


@dataclass(frozen=True)
class GameRollout:
    episodes: tuple[SeatEpisode, SeatEpisode]
    raw_returns: tuple[float, float]
    decisions: int


def collect_game(
    model: MultiGamePolicy,
    game_id: str,
    *,
    root: int,
    index: int,
    deadline: float = float("inf"),
) -> GameRollout:
    """Both seats act from one latest frozen copy, with separate memory/RNG."""
    deadline_check(deadline)
    policy = deepcopy(model).cpu().eval().requires_grad_(False)
    game = engine_for(game_id, index % 2)
    game.reset(derive_seed(root, index, f"{game_id}/engine"))
    generators = [
        torch.Generator().manual_seed(
            derive_seed(root, index, f"{game_id}/seat:{seat}")
        )
        for seat in (0, 1)
    ]
    memories: list[torch.Tensor | None] = [None, None]
    cursors = [0, 0]
    decisions: list[list[DecisionInput]] = [[], []]
    actions: list[list[int]] = [[], []]
    logs: list[list[float]] = [[], []]
    values: list[list[float]] = [[], []]
    rewards: list[list[float]] = [[], []]
    totals = [0.0, 0.0]
    steps = 0
    while not game.is_terminal():
        deadline_check(deadline)
        if steps >= MAX_STEPS[game_id]:
            raise ValueError("Incomplete game: step bound exceeded")
        seat = game.current_player()
        assert seat is not None
        view, legal = game.decision_for(seat)
        if not legal:
            raise ValueError("Live actor has no choices")
        decision = DecisionInput(
            observation_tokens(game_id, view, history_start=cursors[seat]),
            candidate_features(game_id, legal, seat),
        )
        if isinstance(view.data, GuildworksObservation):
            cursors[seat] = len(view.data.history)
        with torch.no_grad():
            logits, value, memory = policy(game_id, decision, memories[seat])
            probabilities = logits.log_softmax(-1)
            action = int(
                torch.multinomial(
                    probabilities.exp(),
                    1,
                    generator=generators[seat],
                ).item()
            )
        memories[seat] = memory.detach()
        decisions[seat].append(decision)
        actions[seat].append(action)
        logs[seat].append(float(probabilities[action]))
        values[seat].append(float(value))
        rewards[seat].append(0.0)
        result = game.step(legal[action])
        if (
            result.next_player != game.current_player()
            or result.terminated != game.is_terminal()
        ):
            raise ValueError("Engine lifecycle mismatch")
        for player, reward in enumerate(result.rewards):
            totals[player] += reward
            if rewards[player]:
                rewards[player][-1] += reward / REWARD_SCALES[game_id]
            elif reward:
                raise ValueError("Reward before first own decision")
        steps += 1
    episodes = tuple(
        SeatEpisode(
            game_id,
            tuple(decisions[seat]),
            tuple(actions[seat]),
            torch.tensor(logs[seat]),
            torch.tensor(values[seat]),
            torch.tensor(rewards[seat]),
        )
        for seat in (0, 1)
    )
    deadline_check(deadline)
    return GameRollout((episodes[0], episodes[1]), (totals[0], totals[1]), steps)


def episode_predictions(
    model: MultiGamePolicy,
    episode: SeatEpisode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    memory = None
    logs, values, entropies = [], [], []
    if not episode.decisions or len(episode.actions) != len(episode.decisions):
        raise ValueError("Invalid complete seat episode")
    for decision, action in zip(episode.decisions, episode.actions, strict=True):
        logits, value, memory = model(
            episode.game,
            DecisionInput(
                decision.tokens.to(device),
                decision.candidates.to(device),
            ),
            memory,
        )
        if not 0 <= action < len(logits):
            raise ValueError("Invalid candidate index")
        probabilities = logits.log_softmax(-1)
        logs.append(probabilities[action])
        values.append(value)
        entropies.append(-(probabilities.exp() * probabilities).sum())
    return torch.stack(logs), torch.stack(values), torch.stack(entropies)


class MultiGameTrainer:
    """Full BPTT per seat; equal game loss, per-game advantage normalization.

    Fixed gamma=1, lambda=.95; no replay, opponents, teachers or reward bonuses.
    Each epoch is one accumulated equal-game optimizer step. A failed update
    invalidates the in-memory boundary; callers must reload its durable predecessor.
    """

    def __init__(self, model: MultiGamePolicy) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        self.updates = 0
        self.optimizer_steps = 0
        self.valid_boundary = True

    def update(
        self,
        rollouts: tuple[GameRollout, ...],
        *,
        deadline: float = float("inf"),
    ) -> dict[str, Any]:
        if not self.valid_boundary or not rollouts:
            raise ValueError("Reload a saved boundary or supply complete self-play")
        episodes = [episode for rollout in rollouts for episode in rollout.episodes]
        grouped = {
            game: [e for e in episodes if e.game == game]
            for game in sorted({e.game for e in episodes})
        }
        device = next(self.model.parameters()).device
        # Match behavior on CPU, the placement used by collection, before mutation.
        behavior = deepcopy(self.model).cpu().eval()
        with torch.no_grad():
            for episode in episodes:
                logs, values, _ = episode_predictions(behavior, episode)
                if not (
                    torch.allclose(logs, episode.logs, atol=1e-5, rtol=1e-5)
                    and torch.allclose(values, episode.values, atol=1e-5, rtol=1e-5)
                ):
                    raise ValueError("Stale or foreign self-play behavior")
        targets = {}
        for game, records in grouped.items():
            pairs = [
                generalized_advantage_estimation(
                    e.rewards,
                    e.values,
                    bootstrap_value=0.0,
                    gamma=1.0,
                    gae_lambda=0.95,
                )
                for e in records
            ]
            adv = torch.cat([p[0] for p in pairs])
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
            targets[game] = (
                adv.to(device),
                torch.cat([p[1] for p in pairs]).to(device),
            )
        metrics: dict[str, Any] = {}
        self.valid_boundary = False
        for _ in range(2):
            deadline_check(deadline, device)
            self.optimizer.zero_grad(set_to_none=True)
            for game, records in grouped.items():
                deadline_check(deadline, device)
                predictions = [episode_predictions(self.model, e) for e in records]
                logs, values, entropy = (
                    torch.cat([p[i] for p in predictions]) for i in range(3)
                )
                old = torch.cat([e.logs for e in records]).to(device)
                adv, returns = targets[game]
                ratio = (logs - old).exp()
                policy = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
                value_loss = 0.5 * (values - returns).square().mean()
                loss = policy + 0.5 * value_loss - 0.01 * entropy.mean()
                if not torch.isfinite(loss).item():
                    raise ValueError("Nonfinite PPO loss")
                (loss / len(grouped)).backward()  # type: ignore[no-untyped-call]
                metrics[game] = {
                    "loss": float(loss.detach()),
                    "policy_loss": float(policy.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.mean().detach()),
                    "approximate_kl": float(
                        ((ratio - 1) - (logs - old)).mean().detach()
                    ),
                    "clip_fraction": float(
                        ((ratio - 1).abs() > 0.2).float().mean().detach()
                    ),
                    "decisions": len(logs),
                    "presentations": 2 * len(logs),
                    "reward_scale": REWARD_SCALES[game],
                }
            deadline_check(deadline, device)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 0.5, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.optimizer_steps += 1
        deadline_check(deadline, device)
        self.updates += 1
        self.valid_boundary = True
        return {"per_game": metrics, "num_samples": sum(r.decisions for r in rollouts)}
