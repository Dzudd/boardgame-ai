"""Independent reward, sampling, model and ownership checks; no training run."""

import random
from copy import deepcopy
from dataclasses import fields
from typing import Any, cast

import numpy as np
import pytest
import torch

from boardgame_ai.core import Action, Observation, StepResult
from boardgame_ai.games.connect_four import ConnectFourEngine, ConnectFourObservation
from boardgame_ai.models import ConnectFourActorCritic
from boardgame_ai.models.connect_four import initial_connect_four_model
from boardgame_ai.models.encoders import (
    encode_connect_four_legal_actions,
    encode_connect_four_observation,
)
from boardgame_ai.models.masking import mask_action_logits
from boardgame_ai.training.advantages import generalized_advantage_estimation as gae
from boardgame_ai.training.connect_four import (
    ConnectFourRollout,
    derive_connect_four_seed,
)
from boardgame_ai.training.connect_four import (
    collect_connect_four_self_play as collect,
)

DRAW = (
    3,
    3,
    0,
    2,
    5,
    4,
    5,
    6,
    4,
    2,
    2,
    3,
    3,
    5,
    5,
    2,
    2,
    4,
    3,
    3,
    1,
    1,
    2,
    1,
    5,
    5,
    0,
    4,
    0,
    4,
    4,
    6,
    6,
    6,
    6,
    0,
    1,
    0,
    6,
    0,
    1,
    1,
)


class ScriptedPolicy(ConnectFourActorCritic):
    """Test-only public-board oracle to force terminal reward cases, not training."""

    def __init__(self, moves: tuple[int, ...]) -> None:
        super().__init__()
        self.moves = moves

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert observations.shape == (6, 6, 7)
        assert observations[3].all() and not observations[5].any()
        step = int(observations[:2].sum().item())
        logits = torch.full((7,), -1000.0)
        logits[self.moves[step]] = 1000.0
        return logits, torch.tensor(2.0)


def assert_rollouts_equal(a: ConnectFourRollout, b: ConnectFourRollout) -> None:
    assert a.num_steps == b.num_steps and a.total_rewards == b.total_rewards
    for x, y in zip(a.players, b.players, strict=True):
        for f in fields(x):
            first, second = getattr(x, f.name), getattr(y, f.name)
            if isinstance(first, torch.Tensor):
                assert torch.equal(first, second), f.name
            else:
                assert first == second


@pytest.mark.parametrize("batch_shape", [(), (1,), (4,), (2, 3)])
def test_model_shape_parameter_count_and_batch_equivalence(
    batch_shape: tuple[int, ...],
) -> None:
    model = initial_connect_four_model(17)
    assert sum(p.numel() for p in model.parameters()) == 20872
    game = ConnectFourEngine()
    game.reset()
    encoded = encode_connect_four_observation(game.observe(0))
    logits, values = model.forward(encoded.expand(*batch_shape, 6, 6, 7))
    single_logits, single_value = model.forward(encoded)
    assert logits.shape == (*batch_shape, 7) and values.shape == batch_shape
    assert logits.dtype == values.dtype == torch.float32
    assert torch.allclose(logits, single_logits.expand_as(logits), atol=1e-7)
    assert torch.allclose(values, single_value.expand_as(values), atol=1e-7)


def test_masked_policy_and_value_gradients_reach_shared_features() -> None:
    model = initial_connect_four_model(17)
    encoded = torch.ones(6, 6, 7, requires_grad=True)
    original = encoded.detach().clone()
    logits, value = model.forward(encoded)
    mask = torch.tensor([True, False, True, False, False, False, False])
    probabilities = mask_action_logits(logits, mask).softmax(-1)
    assert probabilities[~mask].count_nonzero() == 0
    (-probabilities[0].log()).backward(retain_graph=True)  # type: ignore[no-untyped-call]
    gradient = model.policy_head.weight.grad
    assert gradient is not None and not gradient[~mask].any()
    assert gradient[mask].abs().sum() > 0
    assert model.value_head.weight.grad is None
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.body.parameters()
    )
    model.zero_grad(set_to_none=True)
    value.backward()  # type: ignore[no-untyped-call]
    assert model.policy_head.weight.grad is None
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in (*model.body.parameters(), *model.value_head.parameters())
    )
    assert torch.equal(encoded.detach(), original)


def test_raw_heads_and_explicit_float64_conversion() -> None:
    model = initial_connect_four_model(17)
    with torch.no_grad():
        model.policy_head.weight.zero_()
        model.policy_head.bias.copy_(torch.arange(7) - 3)
        model.value_head.weight.zero_()
        model.value_head.bias.fill_(3.5)
    logits, value = model.forward(torch.zeros(6, 6, 7))
    assert torch.equal(logits, torch.arange(7) - 3) and value == 3.5
    model.double()
    assert (
        model.forward(torch.zeros(6, 6, 7, dtype=torch.float64))[0].dtype
        == torch.float64
    )
    with pytest.raises(ValueError, match="dtype"):
        model.forward(torch.zeros(6, 6, 7))


def test_initialization_is_cpu_float32_even_with_changed_default_dtype() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = initial_connect_four_model(17)
        assert all(
            p.dtype == torch.float32 and p.device.type == "cpu"
            for p in model.parameters()
        )
    finally:
        torch.set_default_dtype(previous)
    other = initial_connect_four_model(18)
    assert any(
        not torch.equal(p, q)
        for p, q in zip(model.parameters(), other.parameters(), strict=True)
    )


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(6, 3, 3),
        torch.zeros(0, 6, 6, 7),
        torch.zeros(6, 6, 7, dtype=torch.int64),
        torch.zeros(6, 6, 7, dtype=torch.float64),
        torch.full((6, 6, 7), float("nan")),
        torch.full((6, 6, 7), float("inf")),
        torch.zeros(6, 6, 7, device="meta"),
        cast(Any, object()),
    ],
)
def test_invalid_model_inputs_reject(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        initial_connect_four_model(17).forward(bad)


@pytest.mark.parametrize("bad", [True, -1, 2**64, 1.5])
def test_invalid_initialization_seeds_reject(bad: Any) -> None:
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        initial_connect_four_model(bad)
    assert torch.equal(before, torch.random.get_rng_state())


def test_seed_schedule_matches_independent_digest_and_streams() -> None:
    from hashlib import sha256

    expected = int.from_bytes(
        sha256(b"boardgame-ai/connect-four/ppo/v1:630601:12:seat:1").digest()[:8], "big"
    )
    assert derive_connect_four_seed(630601, 12, "seat:1") == expected
    seeds = {
        derive_connect_four_seed(630601, i, stream)
        for i in range(3)
        for stream in ("engine", "seat:0", "seat:1", "initialization", "shuffle")
    }
    assert len(seeds) == 15


@pytest.mark.parametrize(
    "args",
    [
        (True, 0, "engine"),
        (1, -1, "engine"),
        (1, False, "engine"),
        (1, 0, "evaluation"),
    ],
)
def test_invalid_seed_schedule_rejects(args: tuple[Any, Any, Any]) -> None:
    with pytest.raises(ValueError):
        derive_connect_four_seed(*args)


@pytest.mark.parametrize("starter", [0, 1])
@pytest.mark.parametrize(
    "moves", [(0, 6, 1, 6, 2, 5, 3), (0, 1, 0, 1, 2, 1, 2, 1), DRAW]
)
def test_complete_games_own_rewards_and_hand_calculated_gae(
    starter: int, moves: tuple[int, ...]
) -> None:
    game = ConnectFourEngine(starting_player=starter)
    rollout = collect(
        game, ScriptedPolicy(moves), sampling_seeds=(10, 11), max_steps=len(moves)
    )
    assert game.is_terminal() and rollout.num_steps == len(moves)
    draw = len(moves) == 42
    winner = starter if len(moves) % 2 else 1 - starter
    expected_totals = tuple(
        0.0 if draw else 1.0 if seat == winner else -1.0 for seat in (0, 1)
    )
    assert rollout.total_rewards == expected_totals
    for seat, sequence in enumerate(rollout.players):
        expected_steps = list(range(0 if seat == starter else 1, len(moves), 2))
        assert sequence.step_indices.tolist() == expected_steps
        assert sequence.actions.tolist() == [moves[i] for i in expected_steps]
        count = len(expected_steps)
        expected_reward = expected_totals[seat]
        assert sequence.rewards.tolist() == [0.0] * (count - 1) + [expected_reward]
        assert sequence.values.tolist() == [2.0] * count
        assert sequence.log_probs.tolist() == [0.0] * count
        assert sequence.bootstrap_value == 0.0
        advantages, returns = gae(
            sequence.rewards,
            sequence.values,
            bootstrap_value=sequence.bootstrap_value,
            gamma=1.0,
            gae_lambda=0.95,
        )
        expected = torch.tensor(
            [0.95 ** (count - 1 - t) * (expected_reward - 2.0) for t in range(count)]
        )
        assert torch.allclose(advantages, expected, atol=1e-6)
        assert torch.allclose(returns, expected + 2.0, atol=1e-6)


@pytest.mark.parametrize("starter", [0, 1])
def test_replay_reproduces_supplied_masks_probabilities_and_values(
    starter: int,
) -> None:
    model = initial_connect_four_model(17)
    game = ConnectFourEngine(starting_player=starter)
    result = collect(game, model, sampling_seeds=(10, 11), engine_seed=12)
    replay = ConnectFourEngine(starting_player=starter)
    replay.reset(12)
    generators = [torch.Generator().manual_seed(seed) for seed in (10, 11)]
    cursors = [0, 0]
    for step in range(result.num_steps):
        actor = replay.current_player()
        assert actor is not None
        sequence = result.players[actor]
        i = cursors[actor]
        view, legal = replay.decision_for(actor)
        encoded = encode_connect_four_observation(view)
        mask = encode_connect_four_legal_actions(legal)
        assert sequence.observations.shape[1:] == (6, 6, 7)
        assert sequence.legal_masks.shape[1:] == (7,)
        assert torch.equal(sequence.observations[i], encoded)
        assert torch.equal(sequence.legal_masks[i], mask)
        logits, value = model.forward(encoded)
        index = int(sequence.actions[i].item())
        assert mask[index]
        expected_log_prob = mask_action_logits(logits, mask).log_softmax(-1)[index]
        expected_index = torch.multinomial(
            mask_action_logits(logits, mask).softmax(-1),
            1,
            generator=generators[actor],
        ).item()
        assert index == expected_index
        assert sequence.log_probs[i] == expected_log_prob
        assert sequence.values[i] == value
        assert sequence.step_indices[i] == step
        action = next(a for a in legal if a.parameters[0][1] == index)
        replay.step(action)
        cursors[actor] += 1
    assert replay.is_terminal() and replay.get_rewards() == result.total_rewards
    for sequence in result.players:
        for f in fields(sequence):
            tensor = getattr(sequence, f.name)
            if isinstance(tensor, torch.Tensor):
                assert tensor.device.type == "cpu" and not tensor.requires_grad
                assert tensor.is_contiguous()


def test_repeatability_global_rng_and_source_model_preserved() -> None:
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    cuda_initialized = torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
    cuda_rng = torch.cuda.get_rng_state_all() if cuda_initialized else []
    model = initial_connect_four_model(17)
    same = initial_connect_four_model(17)
    assert all(
        torch.equal(v, same.state_dict()[k]) for k, v in model.state_dict().items()
    )
    model.body.eval()  # Preserve mixed child modes as well as the root mode.
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    modes = [m.training for m in model.modules()]
    original = deepcopy(model.state_dict())
    first = collect(ConnectFourEngine(), model, sampling_seeds=(10, 11))
    second = collect(ConnectFourEngine(), model, sampling_seeds=(10, 11))
    assert_rollouts_equal(first, second)
    assert modes == [m.training for m in model.modules()]
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    assert all(
        p.grad is not None and torch.all(p.grad == 1) for p in model.parameters()
    )
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0] and np.array_equal(
        after_numpy[1], numpy_rng[1]
    )
    assert after_numpy[2:] == numpy_rng[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else [], strict=True
        )
    )
    first.players[0].observations.zero_()
    first.players[0].legal_masks.zero_()
    assert second.players[0].observations.any() and second.players[0].legal_masks.any()


def test_policy_snapshot_is_frozen_even_if_source_weights_change() -> None:
    model = initial_connect_four_model(17)
    baseline = collect(ConnectFourEngine(), model, sampling_seeds=(10, 11))

    class MutatingEngine(ConnectFourEngine):
        def step(self, action: Action) -> StepResult:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(10)
            return super().step(action)

    actual = collect(MutatingEngine(), model, sampling_seeds=(10, 11))
    assert_rollouts_equal(baseline, actual)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"max_steps": True},
        {"max_steps": 43},
        {"sampling_seeds": (True, 1)},
        {"sampling_seeds": (-1, 1)},
        {"sampling_seeds": (2**64, 1)},
        {"sampling_seeds": (1,)},
        {"sampling_seeds": [1, 2]},
        {"engine_seed": True},
        {"deadline": float("nan")},
        {"deadline": True},
    ],
)
def test_configuration_rejection_preserves_existing_game(
    kwargs: dict[str, Any],
) -> None:
    game = ConnectFourEngine()
    game.reset()
    game.step(Action(kind="drop", parameters=(("column", 3),)))
    before = game.observe(0)
    options: dict[str, Any] = {"sampling_seeds": (10, 11), **kwargs}
    with pytest.raises(ValueError):
        collect(game, initial_connect_four_model(17), **options)
    assert game.observe(0) == before


def test_live_cutoff_fails_instead_of_returning_partial_training_game() -> None:
    game = ConnectFourEngine()
    with pytest.raises(RuntimeError, match="complete game"):
        collect(
            game, initial_connect_four_model(17), sampling_seeds=(10, 11), max_steps=1
        )
    assert len(game.observe(0).data.history) == 1


def test_deadline_before_reset_and_mid_game(monkeypatch: pytest.MonkeyPatch) -> None:
    game = ConnectFourEngine()
    game.reset()
    game.step(Action(kind="drop", parameters=(("column", 3),)))
    before = game.observe(0)
    with pytest.raises(TimeoutError):
        collect(
            game, initial_connect_four_model(17), sampling_seeds=(10, 11), deadline=-1
        )
    assert game.observe(0) == before
    calls = 0

    def fake_time() -> float:
        nonlocal calls
        calls += 1
        return 1.0 if calls >= 6 else 0.0

    monkeypatch.setattr("boardgame_ai.training.connect_four.monotonic", fake_time)
    with pytest.raises(TimeoutError):
        collect(
            game, initial_connect_four_model(17), sampling_seeds=(10, 11), deadline=0.5
        )
    assert len(game.observe(0).data.history) == 1


@pytest.mark.parametrize("corruption", ["dtype", "nan", "wrong-model"])
def test_incompatible_model_rejects_before_reset(corruption: str) -> None:
    model: Any = initial_connect_four_model(17)
    if corruption == "dtype":
        model.double()
    elif corruption == "nan":
        with torch.no_grad():
            model.value_head.bias.fill_(float("nan"))
    else:
        model = object()
    game = ConnectFourEngine()
    game.reset()
    before = game.observe(0)
    with pytest.raises(ValueError):
        collect(game, model, sampling_seeds=(10, 11))
    assert game.observe(0) == before


def test_wrong_actor_observation_rejects() -> None:
    class WrongViewEngine(ConnectFourEngine):
        def decision_for(
            self, player_id: int
        ) -> tuple[Observation[ConnectFourObservation], tuple[Action, ...]]:
            _, legal = super().decision_for(player_id)
            return self.observe(1 - player_id), legal

    with pytest.raises(RuntimeError, match="actor"):
        collect(
            WrongViewEngine(), initial_connect_four_model(17), sampling_seeds=(10, 11)
        )


@pytest.mark.parametrize("corruption", ["shape", "logits", "value", "dtype"])
def test_invalid_predictions_fail_before_executing_an_action(corruption: str) -> None:
    class InvalidPolicy(ConnectFourActorCritic):
        def forward(
            self, observations: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.zeros(6 if corruption == "shape" else 7)
            value = torch.tensor(float("inf") if corruption == "value" else 0.0)
            if corruption == "logits":
                logits[0] = float("nan")
            if corruption == "dtype":
                logits = logits.double()
            return logits, value

    game = ConnectFourEngine()
    with pytest.raises(ValueError):
        collect(game, InvalidPolicy(), sampling_seeds=(10, 11))
    assert game.observe(0).data.history == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward_gradients_and_cpu_collection_preserve_source() -> None:
    model = initial_connect_four_model(17).cuda()
    cpu = deepcopy(model).cpu()
    encoded = torch.ones(2, 6, 6, 7, device="cuda")
    logits, value = model.forward(encoded)
    assert logits.is_cuda and value.is_cuda
    (logits.sum() + value.sum()).backward()  # type: ignore[no-untyped-call]
    assert all(p.grad is not None and p.grad.is_cuda for p in model.parameters())
    rng = torch.cuda.get_rng_state_all()
    initial_connect_four_model(18)
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    actual = collect(ConnectFourEngine(), model, sampling_seeds=(10, 11))
    expected = collect(ConnectFourEngine(), cpu, sampling_seeds=(10, 11))
    assert_rollouts_equal(actual, expected)
    assert all(
        torch.equal(a, b)
        for a, b in zip(rng, torch.cuda.get_rng_state_all(), strict=True)
    )
    assert all(p.is_cuda for p in model.parameters())
    for original, parameter in zip(grads, model.parameters(), strict=True):
        assert parameter.grad is not None and torch.equal(original, parameter.grad)
