"""CPU-only public search-target learning, separate from PPO and its samples."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from typing import Any

import torch

from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.models.encoders import (
    decode_tic_tac_toe_action,
    encode_tic_tac_toe_action,
    encode_tic_tac_toe_legal_actions,
    encode_tic_tac_toe_observation,
)
from boardgame_ai.models.masking import mask_action_logits
from boardgame_ai.search.neural import NeuralEvaluator
from boardgame_ai.search.puct import SearchAgent, SearchConfig
from boardgame_ai.search.simulator import PublicTicTacToeSimulator

LEARNER_ID = "ttt-public-search-target-v1"
MODEL_ID = "ttt-search-outcome-obs6-actions9-v1"


def derive_seed(root: int, index: int, purpose: str) -> int:
    text = f"boardgame-ai/search-learning/v1:{root}:{index}:{purpose}"
    return int.from_bytes(sha256(text.encode()).digest()[:8], "big")


def model_digest(model: TicTacToeActorCritic) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initial_model(seed: int) -> TicTacToeActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        return TicTacToeActorCritic()


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchTrainingConfig:
    """Functional defaults from the bridge design, not an experiment budget."""

    games_per_update: int = 64
    optimizer_steps_per_update: int = 4
    batch_size: int = 256
    replay_capacity_games: int = 10000
    seed: int = 42
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    unroll_steps: int = 0
    search: SearchConfig = SearchConfig(exploration=True)

    def __post_init__(self) -> None:
        for value in (
            self.games_per_update,
            self.optimizer_steps_per_update,
            self.batch_size,
            self.replay_capacity_games,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("Search training counts must be positive integers")
        if type(self.seed) is not int:
            raise ValueError("Search training seed must be an integer")
        if type(self.unroll_steps) is not int or self.unroll_steps != 0:
            raise ValueError("The public feed-forward learner supports only K=0")
        for coefficient in (self.learning_rate, self.max_grad_norm):
            if (
                type(coefficient) not in (float, int)
                or not isfinite(coefficient)
                or coefficient <= 0
            ):
                raise ValueError(
                    "Learning rate and gradient bound must be finite positive"
                )
        if (
            not isinstance(self.search, SearchConfig)
            or self.search.exploration is not True
        ):
            raise ValueError("Search training requires root noise and visit sampling")


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchSample:
    observation: torch.Tensor
    legal_mask: torch.Tensor
    visits: torch.Tensor
    action: int
    seat: int
    step: int
    own_step: int
    outcome: float


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchGame:
    """Trusted replay, never an agent input or a generic public export."""

    samples: tuple[SearchSample, ...]
    rewards: tuple[tuple[float, float], ...]
    game_index: int
    policy_version: int
    policy_sha256: str
    search: SearchConfig
    evaluator_calls: int
    source: str = "latest-policy-search-self-play-v1"


def collect_search_game(
    evaluator: NeuralEvaluator,
    *,
    root_seed: int,
    game_index: int,
    policy_version: int,
    search: SearchConfig,
) -> SearchGame:
    """Both seats use one frozen evaluator; only engine-filtered views reach it.

    No caller-supplied opponents, evaluation records, solver targets or live
    engine clones enter this collector. All records are from a completed game.
    """
    if search.exploration is not True:
        raise ValueError("Self-play collection requires exploration")
    if type(root_seed) is not int or any(
        type(value) is not int or value < 0 for value in (game_index, policy_version)
    ):
        raise ValueError("Collection requires integer seed and nonnegative indices")
    game = TicTacToeEngine()
    game.reset(derive_seed(root_seed, game_index, "engine"))
    agents = (SearchAgent(evaluator, search), SearchAgent(evaluator, search))
    for seat_index, agent in enumerate(agents):
        agent.reset(derive_seed(root_seed, game_index, f"seat:{seat_index}"))
    pending: list[dict[str, Any]] = []
    rewards: list[tuple[float, float]] = []
    own_steps = [0, 0]
    evaluator_calls = 0
    while not game.is_terminal():
        seat = game.current_player()
        assert seat is not None
        view, legal = game.decision_for(seat)
        action = agents[seat].select_action(view, legal)
        result = agents[seat].last_result
        assert result is not None
        evaluator_calls += result.evaluator_calls
        visits = torch.zeros(9, dtype=torch.int64)
        for edge in result.root.edges:
            visits[encode_tic_tac_toe_action(edge.action)] = edge.visits
        pending.append(
            {
                "observation": encode_tic_tac_toe_observation(view),
                "legal_mask": encode_tic_tac_toe_legal_actions(legal),
                "visits": visits,
                "action": encode_tic_tac_toe_action(action),
                "seat": seat,
                "step": len(pending),
                "own_step": own_steps[seat],
            }
        )
        own_steps[seat] += 1
        transition = game.step(action)
        rewards.append((transition.rewards[0], transition.rewards[1]))
    totals = tuple(sum(reward[seat] for reward in rewards) for seat in (0, 1))
    collected = SearchGame(
        samples=tuple(
            SearchSample(**sample, outcome=totals[sample["seat"]]) for sample in pending
        ),
        rewards=tuple(rewards),
        game_index=game_index,
        policy_version=policy_version,
        policy_sha256=model_digest(evaluator.model),
        search=search,
        evaluator_calls=evaluator_calls,
    )
    validate_game(collected)
    return collected


def _tensor(tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.shape == shape
        and tensor.dtype == dtype
        and tensor.device.type == "cpu"
        and not tensor.requires_grad
        and bool(torch.isfinite(tensor).all())
    )


def validate_game(game: SearchGame) -> None:
    """Verify public trajectory, target perspective, visits and completion.

    Schema validation cannot prove how caller-supplied data were generated;
    production replay is populated exclusively by the self-play collector.
    """
    if (
        game.source != "latest-policy-search-self-play-v1"
        or type(game.game_index) is not int
        or game.game_index < 0
        or type(game.policy_version) is not int
        or game.policy_version < 0
        or not isinstance(game.policy_sha256, str)
        or len(game.policy_sha256) != 64
        or any(c not in "0123456789abcdef" for c in game.policy_sha256)
        or not 5 <= len(game.samples) <= 9
        or len(game.rewards) != len(game.samples)
        or type(game.evaluator_calls) is not int
        or not len(game.samples)
        <= game.evaluator_calls
        <= len(game.samples) * (game.search.simulations + 1)
        or game.search.exploration is not True
    ):
        raise ValueError("Invalid search game provenance or completion metadata")
    state = PublicTicTacToeSimulator((None,) * 9, 0)
    own_steps = [0, 0]
    totals = [0.0, 0.0]
    for index, sample in enumerate(game.samples):
        seat = state.actor
        if seat is None:
            raise ValueError("Replay contains decisions after termination")
        legal = state.legal_actions()
        if (
            any(
                type(value) is not int
                for value in (sample.seat, sample.step, sample.own_step, sample.action)
            )
            or sample.seat != seat
            or sample.step != index
            or sample.own_step != own_steps[seat]
            or not _tensor(sample.observation, (6, 3, 3), torch.float32)
            or not _tensor(sample.legal_mask, (9,), torch.bool)
            or not _tensor(sample.visits, (9,), torch.int64)
            or not torch.equal(
                sample.observation, encode_tic_tac_toe_observation(state.observe())
            )
            or not torch.equal(
                sample.legal_mask, encode_tic_tac_toe_legal_actions(legal)
            )
            or bool((sample.visits < 0).any())
            or bool((sample.visits[~sample.legal_mask] != 0).any())
            or int(sample.visits.sum()) != game.search.simulations
            or not 0 <= sample.action < 9
            or sample.visits[sample.action] <= 0
        ):
            raise ValueError("Invalid search replay sample or visit target")
        state, reward = state.step(decode_tic_tac_toe_action(sample.action, legal))
        if game.rewards[index] != reward:
            raise ValueError("Search replay rewards disagree with public rules")
        own_steps[seat] += 1
        for player in (0, 1):
            totals[player] += reward[player]
    if state.actor is not None or any(
        type(sample.outcome) not in (float, int)
        or sample.outcome != totals[sample.seat]
        for sample in game.samples
    ):
        raise ValueError("Search replay requires complete own-seat outcomes")


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchBatch:
    observations: torch.Tensor
    legal_masks: torch.Tensor
    targets: torch.Tensor
    outcomes: torch.Tensor


def search_target_loss(
    model: TicTacToeActorCritic,
    batch: SearchBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross entropy to search visits plus half-MSE to own-seat outcomes; no PPO."""
    size = batch.observations.shape[0] if batch.observations.ndim else 0
    if (
        size < 1
        or not _tensor(batch.observations, (size, 6, 3, 3), torch.float32)
        or not _tensor(batch.legal_masks, (size, 9), torch.bool)
        or not _tensor(batch.targets, (size, 9), torch.float32)
        or not _tensor(batch.outcomes, (size,), torch.float32)
        or bool((batch.targets < 0).any())
        or bool((batch.targets[~batch.legal_masks] != 0).any())
        or not torch.allclose(
            batch.targets.sum(-1), torch.ones(size), atol=1e-6, rtol=0
        )
        or bool((batch.outcomes.abs() > 1).any())
    ):
        raise ValueError("Invalid detached search-target batch")
    logits, values = model(batch.observations)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite search value prediction")
    logs = mask_action_logits(logits, batch.legal_masks).log_softmax(-1)
    policy_loss = (
        -(batch.targets * logs.masked_fill(~batch.legal_masks, 0)).sum(-1).mean()
    )
    value_loss = 0.5 * (values - batch.outcomes).square().mean()
    if not bool(torch.isfinite(policy_loss + value_loss)):
        raise ValueError("Nonfinite search-target loss")
    return policy_loss, value_loss


DEFAULT_CONFIG = SearchTrainingConfig()


class SearchTrainer:
    """FIFO games; uniform game then uniform position, sampled with replacement.

    Acting is latest-policy only. Older self-generated targets may be replayed.
    Collection/optimization failures require reload of the last saved boundary.
    """

    def __init__(self, config: SearchTrainingConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.model = initial_model(derive_seed(config.seed, 0, "initialization"))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.generator = torch.Generator().manual_seed(
            derive_seed(config.seed, 0, "replay")
        )
        self.replay: list[SearchGame] = []
        self.update_count = self.optimizer_steps = self.games_seen = (
            self.decisions_seen
        ) = 0

    def sample_batch(self) -> tuple[SearchBatch, list[tuple[int, int]]]:
        if not self.replay:
            raise ValueError("Cannot sample empty self-play replay")
        selections: list[tuple[int, int]] = []
        samples = []
        for _ in range(self.config.batch_size):
            game_slot = int(
                torch.randint(len(self.replay), (), generator=self.generator)
            )
            game = self.replay[game_slot]
            position = int(
                torch.randint(len(game.samples), (), generator=self.generator)
            )
            selections.append((game_slot, position))
            samples.append(game.samples[position])
        return SearchBatch(
            observations=torch.stack([sample.observation for sample in samples]),
            legal_masks=torch.stack([sample.legal_mask for sample in samples]),
            targets=torch.stack(
                [
                    sample.visits.float() / self.replay[slot].search.simulations
                    for sample, (slot, _) in zip(samples, selections, strict=True)
                ]
            ),
            outcomes=torch.tensor(
                [sample.outcome for sample in samples], dtype=torch.float32
            ),
        ), selections

    def update(self) -> dict[str, Any]:
        evaluator = NeuralEvaluator(self.model)
        decisions = 0
        evaluator_calls = 0
        for offset in range(self.config.games_per_update):
            game = collect_search_game(
                evaluator,
                root_seed=self.config.seed,
                game_index=self.games_seen + offset,
                policy_version=self.update_count,
                search=self.config.search,
            )
            self.replay.append(deepcopy(game))
            del self.replay[: -self.config.replay_capacity_games]
            decisions += len(game.samples)
            evaluator_calls += game.evaluator_calls
        losses = []
        ages: list[int] = []
        self.model.train()
        for _ in range(self.config.optimizer_steps_per_update):
            batch, selections = self.sample_batch()
            ages.extend(
                self.update_count - self.replay[slot].policy_version
                for slot, _ in selections
            )
            policy_loss, value_loss = search_target_loss(self.model, batch)
            self.optimizer.zero_grad(set_to_none=True)
            (policy_loss + value_loss).backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters()):
                raise ValueError("Nonfinite updated search parameters")
            self.optimizer_steps += 1
            losses.append((float(policy_loss.detach()), float(value_loss.detach())))
        self.update_count += 1
        self.games_seen += self.config.games_per_update
        self.decisions_seen += decisions
        return {
            "update": self.update_count,
            "games_seen": self.games_seen,
            "decisions_seen": self.decisions_seen,
            "optimizer_steps": self.optimizer_steps,
            "collected_games": self.config.games_per_update,
            "collected_decisions": decisions,
            "search_simulations": decisions * self.config.search.simulations,
            "evaluator_calls": evaluator_calls,
            "policy_loss": sum(p for p, _ in losses) / len(losses),
            "value_loss": sum(v for _, v in losses) / len(losses),
            "sampled_positions": len(ages),
            "replay_games": len(self.replay),
            "replay_age_min": min(ages),
            "replay_age_max": max(ages),
            "replay_age_mean": sum(ages) / len(ages),
        }


def game_payload(game: SearchGame) -> dict[str, Any]:
    return asdict(game)


def game_from_payload(payload: dict[str, Any]) -> SearchGame:
    if (
        not isinstance(payload, dict)
        or set(payload) != set(SearchGame.__dataclass_fields__)
        or not isinstance(payload["search"], dict)
        or set(payload["search"]) != set(SearchConfig.__dataclass_fields__)
        or not isinstance(payload["samples"], (list, tuple))
        or any(
            not isinstance(sample, dict)
            or set(sample) != set(SearchSample.__dataclass_fields__)
            for sample in payload["samples"]
        )
    ):
        raise ValueError("Invalid replay game fields")
    return SearchGame(
        **{
            **payload,
            "search": SearchConfig(**payload["search"]),
            "samples": tuple(SearchSample(**sample) for sample in payload["samples"]),
        }
    )
