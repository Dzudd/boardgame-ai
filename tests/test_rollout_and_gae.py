"""Independent reward/GAE oracles and bounded player-specific collection."""

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from boardgame_ai.core import Action, GameEvent, StepResult
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.models.masking import mask_action_logits
from boardgame_ai.training import (
    collect_tic_tac_toe_rollout,
)
from boardgame_ai.training import (
    generalized_advantage_estimation as gae,
)


@pytest.fixture
def model() -> TicTacToeActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        network = TicTacToeActorCritic()
    with torch.no_grad():
        network.value_head.weight.zero_()
        network.value_head.bias.fill_(2.0)
    return network


def test_gae_hand_calculation_and_no_graph() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
    advantages, returns = gae(
        rewards, values, bootstrap_value=4.0, gamma=0.5, gae_lambda=0.8
    )
    # deltas = [1, 1.75, 3.5], recursion multiplier = .4.
    assert torch.allclose(
        advantages, torch.tensor([2.26, 3.15, 3.5], dtype=torch.float64)
    )
    assert torch.allclose(returns, advantages + values.detach())
    assert not advantages.requires_grad and not returns.requires_grad
    assert torch.equal(
        rewards.detach(), torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    )


@pytest.mark.parametrize("gamma", [0.0, 0.7, 1.0])
def test_lambda_one_matches_independent_discounted_returns(gamma: float) -> None:
    rewards = [1.0, -2.0, 3.0]
    bootstrap = 4.0
    expected = []
    for index in range(3):
        expected.append(
            sum(gamma ** (j - index) * rewards[j] for j in range(index, 3))
            + gamma ** (3 - index) * bootstrap
        )
    advantages, returns = gae(
        torch.tensor(rewards),
        torch.tensor([0.2, -0.5, 0.7]),
        bootstrap_value=bootstrap,
        gamma=gamma,
        gae_lambda=1.0,
    )
    assert torch.allclose(returns, torch.tensor(expected), atol=1e-6)
    assert torch.allclose(advantages, returns - torch.tensor([0.2, -0.5, 0.7]))


def test_lambda_zero_is_one_step_td_and_empty_segments() -> None:
    _, returns = gae(
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
        bootstrap_value=5.0,
        gamma=0.5,
        gae_lambda=0.0,
    )
    assert torch.equal(returns, torch.tensor([3.0, 4.5]))
    advantages, returns = gae(torch.empty(0), torch.empty(0), bootstrap_value=0.0)
    assert advantages.shape == returns.shape == (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": -0.1},
        {"gamma": 1.1},
        {"gamma": True},
        {"gamma": float("nan")},
        {"gae_lambda": -1.0},
        {"gae_lambda": float("inf")},
        {"bootstrap_value": True},
        {"bootstrap_value": float("nan")},
        {"bootstrap_value": 1e100},
    ],
)
def test_invalid_gae_scalars_reject(kwargs: dict[str, Any]) -> None:
    options: dict[str, Any] = {"bootstrap_value": 0.0}
    options.update(kwargs)
    with pytest.raises(ValueError):
        gae(torch.zeros(2), torch.zeros(2), **options)


@pytest.mark.parametrize(
    "rewards,values",
    [
        (torch.zeros(2, 1), torch.zeros(2, 1)),
        (torch.zeros(2), torch.zeros(3)),
        (torch.zeros(2), torch.zeros(2, dtype=torch.float64)),
        (torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64)),
        (torch.tensor([float("nan")]), torch.zeros(1)),
        (torch.zeros(1), torch.tensor([float("inf")])),
        (torch.zeros(2), torch.empty(2, device="meta")),
    ],
)
def test_invalid_gae_tensors_reject(
    rewards: torch.Tensor, values: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        gae(rewards, values, bootstrap_value=0.0)


@pytest.mark.parametrize("starter", [0, 1])
def test_full_game_records_and_masked_old_log_probs(
    model: TicTacToeActorCritic, starter: int
) -> None:
    game = TicTacToeEngine(starting_player=starter)
    rng_before = torch.random.get_rng_state().clone()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    rollout = collect_tic_tac_toe_rollout(
        game, (model, model), max_steps=9, engine_seed=12, sampling_seeds=(100, 101)
    )
    assert rollout.terminated and not rollout.truncated
    assert rollout.num_steps <= 9 and game.is_terminal()
    assert sum(len(player.actions) for player in rollout.players) == rollout.num_steps
    for player in rollout.players:
        t = len(player.actions)
        assert player.observations.shape == (t, 6, 3, 3)
        assert player.legal_masks.shape == (t, 9)
        assert player.actions.dtype == player.step_indices.dtype == torch.int64
        assert player.legal_masks.dtype == torch.bool
        assert player.bootstrap_value == 0.0
        assert player.rewards.sum().item() == rollout.total_rewards[player.player_id]
        assert (
            player.observations.device.type == "cpu"
            and player.observations.dtype == torch.float32
        )
        assert (
            not player.observations.requires_grad and not player.log_probs.requires_grad
        )
        assert torch.equal(player.values, torch.full((t,), 2.0))
        logits, _ = model.forward(player.observations)
        expected = mask_action_logits(logits, player.legal_masks).log_softmax(dim=-1)
        expected = expected.gather(-1, player.actions.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(player.log_probs, expected)
        assert player.legal_masks.gather(-1, player.actions.unsqueeze(-1)).all().item()
        assert player.step_indices.tolist() == list(
            range(0 if player.player_id == starter else 1, rollout.num_steps, 2)
        )
        assert player.rewards[:-1].count_nonzero().item() == 0
        advantages, targets = gae(
            player.rewards,
            player.values,
            bootstrap_value=player.bootstrap_value,
            gamma=1.0,
            gae_lambda=1.0,
        )
        assert torch.allclose(
            targets, torch.full((t,), rollout.total_rewards[player.player_id])
        )
        assert torch.isfinite(advantages).all().item()
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert all(
        torch.equal(value, weights[key]) for key, value in model.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training


def test_reproducible_sampling_and_record_ownership(
    model: TicTacToeActorCritic,
) -> None:
    kwargs: dict[str, Any] = {
        "max_steps": 9,
        "engine_seed": 42,
        "sampling_seeds": (10, 20),
    }
    first = collect_tic_tac_toe_rollout(TicTacToeEngine(), (model, model), **kwargs)
    second = collect_tic_tac_toe_rollout(TicTacToeEngine(), (model, model), **kwargs)
    for a, b in zip(first.players, second.players, strict=True):
        for field in (
            "observations",
            "legal_masks",
            "actions",
            "log_probs",
            "values",
            "rewards",
            "step_indices",
        ):
            assert torch.equal(getattr(a, field), getattr(b, field))
    first.players[0].observations.zero_()
    assert second.players[0].observations.count_nonzero().item() > 0


@pytest.mark.parametrize("limit", [0, 1, 2, 3])
def test_live_cutoff_and_empty_seat_bootstraps(
    model: TicTacToeActorCritic, limit: int
) -> None:
    game = TicTacToeEngine()
    rollout = collect_tic_tac_toe_rollout(
        game, (model, model), max_steps=limit, sampling_seeds=(1, 2)
    )
    assert rollout.truncated and not rollout.terminated and not game.is_terminal()
    assert rollout.num_steps == limit
    for player in rollout.players:
        assert player.bootstrap_value == (2.0 if len(player.actions) else 0.0)
        _, returns = gae(
            player.rewards,
            player.values,
            bootstrap_value=player.bootstrap_value,
            gamma=1.0,
            gae_lambda=1.0,
        )
        assert torch.equal(returns, torch.full_like(returns, 2.0))
        if not len(player.actions):
            assert player.observations.shape == (0, 6, 3, 3)
            assert player.legal_masks.shape == (0, 9)


class ScriptedEngine(TicTacToeEngine):
    """Test-only public fixture with actors 0,0,1 and incremental rewards."""

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.cursor = 0

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.cursor = 0

    def step(self, action: Action) -> StepResult:
        if self.fail:
            raise RuntimeError("scripted failure")
        super().step(action)
        reward = ((1.0, 10.0), (2.0, 20.0), (3.0, 30.0))[self.cursor]
        self.cursor += 1
        actor = (0, 1, None)[self.cursor - 1]
        assert self._state is not None
        self._state = replace(
            self._state,
            current_player=actor,
            phase="finished" if actor is None else "play",
        )
        self._rewards = reward
        return StepResult(
            next_player=actor,
            terminated=actor is None,
            rewards=reward,
            events=(GameEvent(kind="private", recipients=()),),
        )


def test_repeated_actors_reward_intervals_and_predecision_rewards(
    model: TicTacToeActorCritic,
) -> None:
    rollout = collect_tic_tac_toe_rollout(
        ScriptedEngine(), (model, model), max_steps=3, sampling_seeds=(1, 2)
    )
    assert (
        rollout.terminated and not rollout.truncated
    )  # termination wins at exact limit
    assert rollout.total_rewards == (6.0, 60.0)
    assert rollout.players[0].step_indices.tolist() == [0, 1]
    assert rollout.players[1].step_indices.tolist() == [2]
    assert rollout.players[0].rewards.tolist() == [1.0, 5.0]
    assert rollout.players[1].rewards.tolist() == [
        30.0
    ]  # before first action: no policy credit
    _, targets = gae(
        rollout.players[0].rewards,
        rollout.players[0].values,
        bootstrap_value=0.0,
        gamma=0.5,
        gae_lambda=1.0,
    )
    assert torch.equal(targets, torch.tensor([3.5, 5.0]))


def test_failures_propagate_without_retries(model: TicTacToeActorCritic) -> None:
    game = ScriptedEngine(fail=True)
    with pytest.raises(RuntimeError, match="scripted failure"):
        collect_tic_tac_toe_rollout(game, (model, model), max_steps=3)
    assert game.cursor == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": -1},
        {"max_steps": True},
        {"max_steps": 1.5},
        {"sampling_seeds": (True, 1)},
        {"sampling_seeds": (1,)},
        {"engine_seed": True},
    ],
)
def test_configuration_rejection_precedes_engine_reset(
    model: TicTacToeActorCritic, kwargs: dict[str, Any]
) -> None:
    game = TicTacToeEngine()
    game.reset()
    game.step(Action(kind="place", parameters=(("cell", 4),)))
    before = game.observe(0)
    options: dict[str, Any] = {"max_steps": 9}
    options.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        collect_tic_tac_toe_rollout(game, (model, model), **options)
    assert game.observe(0) == before


def test_wrong_models_reject_before_reset(model: TicTacToeActorCritic) -> None:
    with pytest.raises(ValueError):
        collect_tic_tac_toe_rollout(
            TicTacToeEngine(), cast(Any, (model, object())), max_steps=9
        )


class ForcedGame(TicTacToeEngine):
    """Test-only constrained choices following real win/draw trajectories."""

    def __init__(self, moves: tuple[int, ...], starter: int) -> None:
        super().__init__(starting_player=starter)
        self.moves = moves
        self.cursor = 0

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.cursor = 0

    def legal_actions(self, player_id: int) -> tuple[Action, ...]:
        available = super().legal_actions(player_id)
        if not available:
            return ()
        return tuple(
            a for a in available if a.parameters == (("cell", self.moves[self.cursor]),)
        )

    def step(self, action: Action) -> StepResult:
        result = super().step(action)
        self.cursor += 1
        return result


class PerspectiveModel(TicTacToeActorCritic):
    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert observations.shape == (6, 3, 3)
        assert observations[5, 0, 0].item() == 0  # terminal inference must never run
        logits, _ = super().forward(observations)
        return logits, 10 * observations[3, 0, 0] + 20 * observations[4, 0, 0]


@pytest.mark.parametrize(
    "moves,outcome",
    [
        ((0, 3, 1, 4, 2), (1.0, -1.0)),
        ((0, 3, 1, 4, 8, 5), (-1.0, 1.0)),
        ((0, 1, 2, 4, 3, 5, 7, 6, 8), (0.0, 0.0)),
    ],
)
@pytest.mark.parametrize("starter", [0, 1])
def test_terminal_winner_and_loser_both_receive_final_rewards(
    moves: tuple[int, ...], outcome: tuple[float, float], starter: int
) -> None:
    with torch.random.fork_rng(devices=[]):
        model = PerspectiveModel()
    rollout = collect_tic_tac_toe_rollout(
        ForcedGame(moves, starter),
        (model, model),
        max_steps=len(moves),
        sampling_seeds=(1, 2),
    )
    expected = outcome if starter == 0 else (outcome[1], outcome[0])
    assert rollout.total_rewards == expected
    assert rollout.terminated and not rollout.truncated
    for seat, player in enumerate(rollout.players):
        assert player.rewards[-1].item() == expected[seat]
        assert player.bootstrap_value == 0.0
        _, targets = gae(
            player.rewards,
            player.values,
            bootstrap_value=0.0,
            gamma=1.0,
            gae_lambda=1.0,
        )
        assert torch.equal(targets, torch.full_like(targets, expected[seat]))


def test_cutoff_bootstrap_uses_each_players_perspective() -> None:
    with torch.random.fork_rng(devices=[]):
        model = PerspectiveModel()
    rollout = collect_tic_tac_toe_rollout(
        TicTacToeEngine(), (model, model), max_steps=2, sampling_seeds=(1, 2)
    )
    # At this boundary actor 0 sees own turn; inactive seat 1 sees opponent turn.
    assert rollout.players[0].bootstrap_value == 10.0
    assert rollout.players[1].bootstrap_value == 20.0


class InitiallyTerminalGame(TicTacToeEngine):
    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        assert self._state is not None
        self._state = replace(self._state, current_player=None, phase="finished")


def test_terminal_reset_produces_empty_segments(model: TicTacToeActorCritic) -> None:
    rollout = collect_tic_tac_toe_rollout(
        InitiallyTerminalGame(), (model, model), max_steps=9
    )
    assert rollout.terminated and rollout.num_steps == 0 and not rollout.truncated
    assert all(
        len(p.actions) == 0 and p.bootstrap_value == 0.0 for p in rollout.players
    )


def test_gae_arithmetic_overflow_rejects() -> None:
    with pytest.raises(ValueError, match="finite"):
        gae(torch.tensor([3e38]), torch.tensor([0.0]), bootstrap_value=3e38)


class InvalidOutputModel(TicTacToeActorCritic):
    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(9), torch.tensor(float("nan"))


def test_nonfinite_model_value_rejects_before_step() -> None:
    with torch.random.fork_rng(devices=[]):
        model = InvalidOutputModel()
    game = TicTacToeEngine()
    with pytest.raises(ValueError, match="finite value"):
        collect_tic_tac_toe_rollout(game, (model, model), max_steps=9)
    assert game.observe(0).data.board == (None,) * 9


def test_collection_preserves_existing_gradients_and_eval_mode(
    model: TicTacToeActorCritic,
) -> None:
    model.eval()
    parameter = model.policy_head.weight
    parameter.grad = torch.ones_like(parameter)
    collect_tic_tac_toe_rollout(
        TicTacToeEngine(), (model, model), max_steps=2, sampling_seeds=(1, 2)
    )
    assert not model.training
    assert torch.equal(parameter.grad, torch.ones_like(parameter))
