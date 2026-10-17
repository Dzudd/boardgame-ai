"""Disposable Phase 6 functional checks, never campaign/strength evidence."""

from copy import deepcopy
from dataclasses import replace
from random import Random

import pytest
import torch

from boardgame_ai.core import Action
from boardgame_ai.games.breakthrough import BreakthroughEngine
from boardgame_ai.games.guildworks import GuildworksEngine
from boardgame_ai.models.multigame import (
    GAMES,
    REWARD_SCALES,
    DecisionInput,
    candidate_features,
    initial_multigame_model,
    observation_tokens,
)
from boardgame_ai.training.multigame import (
    MultiGameTrainer,
    collect_game,
    engine_for,
)
from tests.contracts import assert_engine_contract, assert_rejection_preserves_game


def test_breakthrough_contract() -> None:
    assert_engine_contract(BreakthroughEngine, max_steps=80)
    game = BreakthroughEngine()
    game.reset(13)
    assert_rejection_preserves_game(game, Action(kind="pass"), max_steps=80)


def test_breakthrough_against_independent_public_transition_oracle() -> None:
    captures = 0
    for seed in range(20):
        game = BreakthroughEngine(starting_player=seed % 2)
        game.reset(seed)
        rng = Random(seed)
        for _ in range(80):
            if game.is_terminal():
                break
            seat = game.current_player()
            assert seat is not None
            board = game.observe(seat).data.board
            pairs = []
            for source in range(25):
                for target in range(25):
                    if (
                        board[source] == seat
                        and target // 5 - source // 5 == (-1 if seat == 0 else 1)
                        and abs(target % 5 - source % 5) <= 1
                        and board[target] != seat
                        and (target % 5 != source % 5 or board[target] is None)
                    ):
                        pairs.append((source, target))
            legal = game.legal_actions(seat)
            assert {tuple(v for _, v in a.parameters) for a in legal} == set(pairs)
            action = rng.choice(legal)
            selected_source, selected_target = (v for _, v in action.parameters)
            assert type(selected_source) is int and type(selected_target) is int
            captures += board[selected_target] is not None
            result = game.step(action)
            next_board = list(board)
            next_board[selected_source], next_board[selected_target] = None, seat
            assert game.observe(seat).data.board == tuple(next_board)
            if selected_target // 5 == (0 if seat == 0 else 4):
                assert result.terminated and result.rewards[seat] == 1
            if result.terminated:
                assert result.rewards == ((1.0, -1.0) if seat == 0 else (-1.0, 1.0))
            else:
                assert result.rewards == (0.0, 0.0)
        assert game.is_terminal()
    assert captures > 0


def test_shared_guildworks_hidden_world_equivalence() -> None:
    game = GuildworksEngine()
    game.reset(123)
    twin = game.clone()
    old = twin._hands[1][0]
    index = next(i for i, card in enumerate(twin._deck) if card != old)
    new = twin._deck[index]
    twin._deck[index], twin._hands[1][0] = old, new
    event_index = next(i for i, e in enumerate(twin._history) if e.recipients == (1,))
    twin._history[event_index] = replace(
        twin._history[event_index], parameters=(("card", new),)
    )
    twin._deck.reverse()
    left, right = game.decision_for(0), twin.decision_for(0)
    assert left == right
    model = initial_multigame_model(12)
    with torch.no_grad():
        a = model(
            "guildworks",
            DecisionInput(
                observation_tokens("guildworks", left[0]),
                candidate_features("guildworks", left[1], 0),
            ),
        )
        b = model(
            "guildworks",
            DecisionInput(
                observation_tokens("guildworks", right[0]),
                candidate_features("guildworks", right[1], 0),
            ),
        )
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("game_id", GAMES)
def test_candidate_permutation_and_visible_model(game_id: str) -> None:
    game = engine_for(game_id)
    game.reset(11)
    view, legal = game.decision_for(0)
    tokens = observation_tokens(game_id, view)
    candidates = candidate_features(game_id, legal, 0)
    policy = initial_multigame_model(12)
    logits, value, memory = policy(game_id, DecisionInput(tokens, candidates))
    permuted, other_value, _ = policy(
        game_id, DecisionInput(tokens, candidates.flip(0))
    )
    assert torch.allclose(permuted, logits.flip(0), atol=1e-7)
    assert torch.equal(value, other_value)
    assert memory.shape == (1, 1, 32)
    assert candidate_features(game_id, (), 0).shape[0] == 0
    with pytest.raises(ValueError):
        candidate_features(game_id, (legal[0], legal[0]), 0)
    with pytest.raises(ValueError):
        candidate_features(game_id, (Action(kind="foreign"),), 0)


@pytest.mark.parametrize("game_id", GAMES)
def test_self_play_rewards_reproducibility_and_update(game_id: str) -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        state = torch.random.get_rng_state().clone()
        model = initial_multigame_model(42)
        original = deepcopy(model.state_dict())
        rollout = collect_game(model, game_id, root=43, index=0)
        repeated = collect_game(model, game_id, root=43, index=0)
        assert rollout.decisions == repeated.decisions
        assert rollout.raw_returns == repeated.raw_returns
        for seat in (0, 1):
            episode, twin = rollout.episodes[seat], repeated.episodes[seat]
            assert episode.actions == twin.actions
            assert torch.equal(episode.logs, twin.logs)
            assert float(episode.rewards.sum()) == pytest.approx(
                rollout.raw_returns[seat] / REWARD_SCALES[game_id],
                abs=1e-6,
            )
        assert all(torch.equal(model.state_dict()[k], v) for k, v in original.items())
        trainer = MultiGameTrainer(model)
        metrics = trainer.update((rollout,))
        assert metrics["num_samples"] == rollout.decisions
        assert trainer.optimizer_steps == 2
        assert not torch.equal(
            original["recurrent.weight_ih_l0"],
            model.state_dict()["recurrent.weight_ih_l0"],
        )
        assert torch.equal(state, torch.random.get_rng_state())
        with pytest.raises(ValueError, match="Stale"):
            trainer.update((rollout,))
    finally:
        torch.set_num_threads(previous)


def test_expired_collector_and_update_leave_model_unchanged() -> None:
    model = initial_multigame_model(1)
    with pytest.raises(TimeoutError):
        collect_game(model, "tic_tac_toe", root=1, index=0, deadline=0)
    rollout = collect_game(model, "tic_tac_toe", root=1, index=0)
    trainer = MultiGameTrainer(model)
    original = deepcopy(model.state_dict())
    with pytest.raises(TimeoutError):
        trainer.update((rollout,), deadline=0)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in original.items())


def test_guildworks_history_consumed_once_and_seat_isolation() -> None:
    rollout = collect_game(initial_multigame_model(5), "guildworks", root=6, index=0)
    for episode in rollout.episodes:
        assert len(episode.decisions) == 16
        assert sum(int(d.tokens[:, 0].sum()) for d in episode.decisions) == 16
        # Every decision ends with a snapshot and starts with only new events.
        assert all(d.tokens[-1, 0] == 1 for d in episode.decisions)
        assert all((d.tokens[:-1, 1] == 1).all() for d in episode.decisions)
    bad = replace(rollout.episodes[0], logs=rollout.episodes[0].logs + 1)
    with pytest.raises(ValueError, match="Stale"):
        MultiGameTrainer(initial_multigame_model(5)).update(
            (
                replace(
                    rollout,
                    episodes=(bad, rollout.episodes[1]),
                ),
            )
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_multigame_cuda_update() -> None:
    model = initial_multigame_model(7).cuda()
    rollout = collect_game(model, "guildworks", root=8, index=0)
    metrics = MultiGameTrainer(model).update((rollout,))
    assert metrics["num_samples"] == 32
