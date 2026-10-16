"""Visible sequence encoding, independent recurrent memory and on-policy PPO."""

import random
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest
import torch

from boardgame_ai.core import Action, GameEvent
from boardgame_ai.games.guildworks import GuildworksEngine
from boardgame_ai.models.guildworks import (
    ACTION_COUNT,
    ACTIONS,
    FEATURES,
    GuildworksRecurrentPolicy,
    RecurrentGuildworksAgent,
    decision_tokens,
    decode_action,
    encode_actions,
    encode_event,
    encode_snapshot,
)
from boardgame_ai.models.masking import mask_action_logits
from boardgame_ai.simulation import run_match
from boardgame_ai.training.guildworks import (
    RecurrentPPOTrainer,
    SeatEpisode,
    collect_guildworks,
    validate_episode,
)
from boardgame_ai.training.guildworks_run import initial_model
from boardgame_ai.training.ppo import PPOConfig


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def episodes(model: GuildworksRecurrentPolicy) -> tuple[SeatEpisode, SeatEpisode]:
    return collect_guildworks(
        GuildworksEngine(), model, engine_seed=81, sampling_seeds=(12, 13)
    )


def test_action_mapping_masks_and_strict_validation() -> None:
    assert encode_actions(()).tolist() == [False] * ACTION_COUNT
    assert encode_actions(ACTIONS).all()
    for index, action in enumerate(ACTIONS):
        assert decode_action(index, ACTIONS) is action
    for choices in (
        (ACTIONS[0], ACTIONS[0]),
        (Action(kind="draw"),),
        (Action(kind="craft", parameters=(("card", True),)),),
    ):
        with pytest.raises(ValueError):
            encode_actions(choices)
    for index in (-1, 8, True):
        with pytest.raises(ValueError):
            decode_action(index, ACTIONS)
    with pytest.raises(ValueError):
        decode_action(0, (ACTIONS[-1],))


def test_snapshot_schema_ownership_and_inactive_terminal_encoding() -> None:
    game = GuildworksEngine()
    game.reset(123)
    for seat in (0, 1):
        snapshot = encode_snapshot(game.observe(seat))
        tokens = decision_tokens(game.observe(seat))
        assert snapshot.shape == (FEATURES,) and snapshot.dtype == torch.float32
        assert tokens.shape == (
            5,
            FEATURES,
        )  # two public setup counts, two own cards, snapshot
        assert snapshot[0] == 1 and snapshot[1] == 0
        assert torch.equal(tokens[-1], snapshot)
        tokens.fill_(99)
        assert not torch.equal(tokens[-1], encode_snapshot(game.observe(seat)))
    assert not encode_actions(game.legal_actions(1)).any()
    while not game.is_terminal():
        actor = game.current_player()
        assert actor is not None
        game.step(game.legal_actions(actor)[-1])
    for seat in (0, 1):
        assert torch.isfinite(decision_tokens(game.observe(seat))).all()
        assert not encode_actions(game.legal_actions(seat)).any()


def test_events_are_relative_visible_and_history_is_ordered() -> None:
    private = GameEvent(kind="card", parameters=(("card", 3),), recipients=(0,))
    assert encode_event(private, 0).shape == (FEATURES,)
    with pytest.raises(ValueError):
        encode_event(private, 1)
    game = GuildworksEngine()
    game.reset(123)
    view = game.observe(0)
    altered = replace(
        view, data=replace(view.data, history=tuple(reversed(view.data.history)))
    )
    assert torch.equal(encode_snapshot(view), encode_snapshot(altered))
    assert not torch.equal(decision_tokens(view), decision_tokens(altered))
    with pytest.raises(ValueError):
        decision_tokens(view, history_start=999)
    assert decision_tokens(view, history_start=len(view.data.history)).shape == (
        1,
        FEATURES,
    )


@pytest.mark.parametrize("observer", [0, 1])
@pytest.mark.parametrize("phase", ["place", "craft"])
def test_hidden_worlds_have_identical_tokens_masks_predictions_and_actions(
    observer: int,
    phase: str,
) -> None:
    game = GuildworksEngine()
    game.reset(123)
    if observer == 1:
        game.step(ACTIONS[0])
        game.step(ACTIONS[-1])
    if phase == "craft":
        game.step(game.legal_actions(observer)[0])
    twin = game.clone()
    other = 1 - observer
    old = twin._hands[other][0]
    index = next(i for i, c in enumerate(twin._deck) if c != old)
    new = twin._deck[index]
    twin._deck[index] = old
    twin._hands[other][0] = new
    i = next(i for i, e in enumerate(twin._history) if e.recipients == (other,))
    twin._history[i] = replace(twin._history[i], parameters=(("card", new),))
    twin._deck.reverse()
    left, right = game.decision_for(observer), twin.decision_for(observer)
    assert left == right
    assert torch.equal(decision_tokens(left[0]), decision_tokens(right[0]))
    assert torch.equal(encode_actions(left[1]), encode_actions(right[1]))
    model = initial_model(42)
    a, b = RecurrentGuildworksAgent(model), RecurrentGuildworksAgent(model)
    a.reset(17)
    b.reset(17)
    assert a.select_action(*left) == b.select_action(*right)
    assert a.hidden is not None and b.hidden is not None
    assert torch.equal(a.hidden, b.hidden)
    assert a.last_log_prob == b.last_log_prob and a.last_value == b.last_value


def test_incremental_memory_matches_full_sequence_and_resets_per_seat() -> None:
    model = initial_model(4)
    a, b = RecurrentGuildworksAgent(model), RecurrentGuildworksAgent(model)
    game = GuildworksEngine()
    game.reset(10)
    for _ in range(4):
        actor = game.current_player()
        assert actor is not None
        agent = (a, b)[actor]
        action = agent.select_action(*game.decision_for(actor))
        with torch.no_grad():
            logits, values, hidden = model(torch.stack(agent.tokens))
        assert agent.hidden is not None
        torch.testing.assert_close(agent.hidden, hidden)
        assert agent.last_value == pytest.approx(float(values[-1]), abs=1e-6)
        legal = encode_actions(game.legal_actions(actor))
        logs = mask_action_logits(logits[-1], legal).log_softmax(-1)
        assert agent.last_log_prob == pytest.approx(
            float(logs[ACTIONS.index(action)]), abs=1e-6
        )
        game.step(action)
    saved = b.hidden.clone() if b.hidden is not None else None
    a.reset(123)
    assert a.hidden is None and not a.tokens and not a.history and a.seat is None
    assert saved is not None and b.hidden is not None and torch.equal(b.hidden, saved)


def test_memory_history_and_bad_input_reject_without_advancing() -> None:
    game = GuildworksEngine()
    game.reset(10)
    agent = RecurrentGuildworksAgent(initial_model(0))
    action = agent.select_action(*game.decision_for(0))
    game.step(action)
    state = agent.generator.get_state().clone()
    hidden = agent.hidden.clone() if agent.hidden is not None else None
    for view, choices in (
        (game.observe(1), ()),
        (
            replace(game.observe(0), data=replace(game.observe(0).data, history=())),
            game.legal_actions(0),
        ),
    ):
        with pytest.raises(ValueError):
            agent.select_action(view, choices)
    with pytest.raises(ValueError):
        agent.reset(True)
    assert torch.equal(state, agent.generator.get_state())
    assert hidden is not None and agent.hidden is not None
    assert torch.equal(hidden, agent.hidden)


def test_complete_runner_and_agent_weight_snapshot() -> None:
    model = initial_model(3)
    a, b = RecurrentGuildworksAgent(model), RecurrentGuildworksAgent(model)
    with torch.no_grad():
        next(model.parameters()).add_(10)
    assert not torch.equal(next(model.parameters()), next(a.model.parameters()))
    game = GuildworksEngine()
    result = run_match(game, (a, b), max_steps=32, engine_seed=8, agent_seeds=(2, 3))
    assert result.terminated and result.num_steps == 32
    assert result.total_rewards == game.observe(0).data.scores
    assert a.seat == 0 and b.seat == 1


def test_collection_behavior_recompute_and_rule_rewards() -> None:
    model = initial_model(1)
    game = GuildworksEngine()
    pair = collect_guildworks(game, model, engine_seed=81, sampling_seeds=(12, 13))
    for seat, episode in enumerate(pair):
        validate_episode(episode)
        assert episode.player_id == seat
        assert not episode.tokens.requires_grad
        with torch.no_grad():
            logits, values, _ = model(episode.tokens)
            logs = mask_action_logits(
                logits[episode.positions], episode.masks
            ).log_softmax(-1)
        torch.testing.assert_close(values[episode.positions], episode.values)
        torch.testing.assert_close(
            logs.gather(1, episode.actions[:, None]).squeeze(1), episode.log_probs
        )
        expected = torch.tensor(
            [
                1.0 if a == 2 else float(a - 2) if 4 <= a <= 6 else 0.0
                for a in episode.actions.tolist()
            ]
        )
        assert torch.equal(episode.rewards, expected)
        assert float(episode.rewards.sum()) == game.observe(seat).data.scores[seat]
    duplicate = episodes(model)
    assert torch.equal(pair[0].tokens, duplicate[0].tokens)
    assert torch.equal(pair[1].actions, duplicate[1].actions)


def test_full_bptt_retains_early_visible_history_gradient() -> None:
    model = initial_model(2)
    tokens = episodes(model)[0].tokens.clone().requires_grad_()
    logits, values, _ = model(tokens)
    (logits[-1].sum() + values[-1]).backward()
    assert tokens.grad is not None and tokens.grad[0].abs().sum() > 0
    assert model.recurrent.get_parameter("weight_hh_l0").grad is not None


def test_ppo_loss_matches_hand_calculated_first_epoch_and_updates_gru() -> None:
    model = initial_model(3)
    pair = episodes(model)
    config = PPOConfig(gamma=1, gae_lambda=1, epochs=1, minibatch_size=2)
    expected_returns = [e.rewards.flip(0).cumsum(0).flip(0) for e in pair]
    errors = torch.cat(
        [e.values - r for e, r in zip(pair, expected_returns, strict=True)]
    )
    expected_value = float(0.5 * errors.square().mean())
    entropy = []
    with torch.no_grad():
        for e in pair:
            logits, _, _ = model(e.tokens)
            logs = mask_action_logits(logits[e.positions], e.masks).log_softmax(-1)
            entropy.append(-(logs.exp() * torch.where(e.masks, logs, 0.0)).sum(-1))
    expected_entropy = float(torch.cat(entropy).mean())
    initial = model.recurrent.get_parameter("weight_hh_l0").detach().clone()
    trainer = RecurrentPPOTrainer(model, config, shuffle_seed=9)
    report = trainer.update(pair)
    assert report.num_samples == 32 and report.optimizer_steps == 1
    assert report.value_loss == pytest.approx(expected_value, abs=1e-5)
    assert report.policy_loss == pytest.approx(0.0, abs=1e-6)
    assert report.entropy == pytest.approx(expected_entropy, abs=1e-6)
    assert report.total_loss == pytest.approx(
        config.value_coefficient * expected_value
        - config.entropy_coefficient * expected_entropy,
        abs=1e-5,
    )
    assert not torch.equal(initial, model.recurrent.get_parameter("weight_hh_l0"))
    assert trainer.update_count == trainer.optimizer_steps == 1


@pytest.mark.parametrize("damage", ["positions", "actions", "tokens", "values", "seat"])
def test_invalid_sequence_rejected_before_optimizer_changes(damage: str) -> None:
    model = initial_model(3)
    first, second = episodes(model)
    if damage == "positions":
        first = replace(first, positions=first.positions.flip(0))
    elif damage == "actions":
        first = replace(first, actions=torch.full((16,), 99))
    elif damage == "tokens":
        first = replace(first, tokens=first.tokens * float("nan"))
    elif damage == "values":
        first = replace(first, values=first.values + 1)  # stale behavior
    else:
        first = replace(first, player_id=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    trainer = RecurrentPPOTrainer(model)
    with pytest.raises(ValueError):
        trainer.update((first, second))
    assert not trainer.optimizer.state
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_all_operations_preserve_global_rng_and_model_collection_state() -> None:
    python = random.getstate()
    numpy = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    model = initial_model(2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    pair = episodes(model)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    RecurrentPPOTrainer(model, shuffle_seed=3).update(pair)
    assert random.getstate() == python
    current = np.random.get_state()
    np.testing.assert_array_equal(current[1], numpy[1])
    assert current[0] == numpy[0] and current[2:] == numpy[2:]
    assert torch.equal(torch_rng, torch.get_rng_state())


def test_model_validation_and_no_grad_inference() -> None:
    model = initial_model(2)
    for tokens in (
        torch.zeros(0, FEATURES),
        torch.zeros(2, 51),
        torch.zeros(2, FEATURES, dtype=torch.float64),
        torch.full((2, FEATURES), float("nan")),
    ):
        with pytest.raises(ValueError):
            model(tokens)
    with pytest.raises(ValueError):
        model(torch.zeros(2, FEATURES), torch.zeros(32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cuda_recurrent_update_and_cpu_agent_collection() -> None:
    cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
    model = initial_model(5).cuda()
    pair = episodes(model)
    assert pair[0].tokens.device.type == "cpu"
    trainer = RecurrentPPOTrainer(model, shuffle_seed=2)
    report = trainer.update(pair)
    assert report.optimizer_steps == 2
    assert all(
        p.device.type == "cuda" and p.grad is not None for p in model.parameters()
    )
    assert all(
        s["exp_avg"].device.type == "cuda" for s in trainer.optimizer.state.values()
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
    )
