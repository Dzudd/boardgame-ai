"""Batch replay, independent seed ranges, lifecycle, and fail-fast behavior."""

import random
from typing import Any

import pytest

from boardgame_ai.agents import RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.core import GameNotReadyError, IllegalActionError
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.simulation import run_batch, run_match
from tests.hidden_game import CardView, HiddenGame
from tests.test_interfaces import DummyGame
from tests.test_simulation import IllegalAgent, RecordingAgent


@pytest.mark.parametrize("seed", [0, 42, -17])
def test_batch_replay_and_individual_matches(seed: int) -> None:
    game = TicTacToeEngine()
    agents = (TicTacToeHeuristicAgent(), RandomAgent[TicTacToeObservation]())
    batch = run_batch(game, agents, num_matches=12, seed=seed, max_steps=9)
    board = game.observe(0)
    assert run_batch(game, agents, num_matches=12, seed=seed, max_steps=9) == batch
    assert game.observe(0) == board
    assert [entry.match_index for entry in batch] == list(range(12))
    for entry in batch:
        assert entry.result.terminated and not entry.result.truncated
        assert (
            run_match(
                game,
                agents,
                engine_seed=entry.engine_seed,
                agent_seeds=entry.agent_seeds,
                max_steps=9,
            )
            == entry.result
        )


def test_ranges_and_zero_length_batch() -> None:
    game = DummyGame()
    agents = (RecordingAgent[int](), RecordingAgent[int]())
    full = run_batch(game, agents, num_matches=10, seed=7, max_steps=3)
    assert (
        run_batch(game, agents, num_matches=4, seed=7, start_index=6, max_steps=3)
        == full[6:]
    )
    before = game.observe(0)
    seeds = [agent.seeds.copy() for agent in agents]
    assert run_batch(game, agents, num_matches=0, seed=7, max_steps=3) == ()
    assert game.observe(0) == before
    assert [agent.seeds for agent in agents] == seeds


def test_seed_streams_are_separate_and_independent_of_game_length() -> None:
    game = DummyGame()
    agents = (RecordingAgent[int](), RecordingAgent[int]())
    full = run_batch(game, agents, num_matches=5, seed=10, max_steps=3)
    short = run_batch(game, agents, num_matches=5, seed=10, max_steps=1)
    assert [(r.engine_seed, r.agent_seeds) for r in full] == [
        (r.engine_seed, r.agent_seeds) for r in short
    ]
    all_seeds = [s for entry in full for s in (entry.engine_seed, *entry.agent_seeds)]
    assert len(set(all_seeds)) == len(all_seeds)
    changed = run_batch(game, agents, num_matches=5, seed=11, max_steps=3)
    assert [r.engine_seed for r in changed] != [r.engine_seed for r in full]
    assert all(entry.result.truncated for entry in short)
    assert [agent.seeds[-5:] for agent in agents] == [
        [entry.agent_seeds[seat] for entry in changed] for seat in range(2)
    ]
    assert [view.data for view in agents[0].views] == [0, 1]


def test_hidden_chance_batch_replays_and_leaves_global_randomness_untouched() -> None:
    game = HiddenGame()
    agents = (RandomAgent[CardView](), RandomAgent[CardView]())
    global_state = random.getstate()
    first = run_batch(game, agents, num_matches=10, seed=42, max_steps=3)
    views = tuple(game.observe(seat) for seat in range(2))
    assert run_batch(game, agents, num_matches=10, seed=42, max_steps=3) == first
    assert tuple(game.observe(seat) for seat in range(2)) == views
    for entry in first:
        run_match(
            game,
            agents,
            engine_seed=entry.engine_seed,
            agent_seeds=entry.agent_seeds,
            max_steps=3,
        )
        reference = HiddenGame()
        reference.reset(entry.engine_seed)
        for _ in range(3):
            player = reference.current_player()
            assert player is not None
            reference.step(reference.legal_actions(player)[0])
        assert game.observe(0) == reference.observe(0)
        assert game.observe(1) == reference.observe(1)
    assert random.getstate() == global_state


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": None},
        {"seed": True},
        {"seed": "42"},
        {"num_matches": -1},
        {"num_matches": True},
        {"num_matches": 1.5},
        {"start_index": -1},
        {"start_index": False},
        {"start_index": 1.5},
        {"max_steps": -1},
    ],
)
def test_invalid_batch_config_does_not_reset(kwargs: dict[str, Any]) -> None:
    game = DummyGame()
    agents = (RecordingAgent[int](), RecordingAgent[int]())
    config: dict[str, Any] = {"num_matches": 2, "seed": 42, "max_steps": 3}
    config.update(kwargs)
    with pytest.raises(ValueError):
        run_batch(game, agents, **config)
    assert not any(agent.seeds for agent in agents)
    with pytest.raises(GameNotReadyError):
        game.is_terminal()


def test_invalid_agents_rejected_even_for_empty_batch() -> None:
    agent = RecordingAgent[int]()
    for agents in ((agent,), (agent, agent)):
        with pytest.raises(ValueError):
            run_batch(DummyGame(), agents, num_matches=0, seed=42, max_steps=3)
    assert agent.seeds == []


def test_batch_stops_on_exception_without_later_resets() -> None:
    game = DummyGame()
    agents = (IllegalAgent(), RecordingAgent[int]())
    with pytest.raises(IllegalActionError):
        run_batch(game, agents, num_matches=10, seed=42, max_steps=3)
    assert [len(agent.seeds) for agent in agents] == [1, 1]
    assert game.observe(0).data == 0
