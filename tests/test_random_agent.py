"""Random baseline reproducibility, boundaries, and engine integration."""

import random
from copy import deepcopy
from typing import Any

import pytest

from boardgame_ai.agents import Agent, RandomAgent
from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from tests.hidden_game import CardView, HiddenGame


def view() -> Observation[list[int]]:
    return Observation(player_id=0, current_player=0, phase="choose", data=[1, 2])


CHOICES = tuple(Action(kind="choose", parameters=(("index", i),)) for i in range(5))


@pytest.mark.parametrize("seed", [0, 42, -17])
def test_seeded_uniform_selection_and_reset(seed: int) -> None:
    agent: Agent[list[int]] = RandomAgent(seed)
    reference = random.Random(seed)
    expected = [reference.choice(CHOICES) for _ in range(100)]
    assert [agent.select_action(view(), CHOICES) for _ in expected] == expected
    agent.reset(seed)
    assert [agent.select_action(view(), CHOICES) for _ in expected] == expected


def test_returns_supplied_objects_without_mutating_inputs() -> None:
    agent = RandomAgent[list[int]](42)
    observation = view()
    before = deepcopy(observation)
    selected = [agent.select_action(observation, CHOICES) for _ in range(100)]
    assert all(any(action is choice for choice in CHOICES) for action in selected)
    assert set(selected) == set(CHOICES)
    assert observation == before
    assert agent.select_action(observation, (CHOICES[0],)) is CHOICES[0]


def test_empty_choices_preserve_random_sequence() -> None:
    agent = RandomAgent[list[int]](12)
    reference = RandomAgent[list[int]](12)
    with pytest.raises(ValueError, match="at least one legal action"):
        agent.select_action(view(), ())
    assert [agent.select_action(view(), CHOICES) for _ in range(30)] == [
        reference.select_action(view(), CHOICES) for _ in range(30)
    ]


@pytest.mark.parametrize("seed", [True, False, 1.5, "42", object()])
def test_invalid_seed_rejection_is_atomic(seed: Any) -> None:
    with pytest.raises(ValueError, match="Seed must be an integer or None"):
        RandomAgent[list[int]](seed)
    agent = RandomAgent[list[int]](7)
    reference = RandomAgent[list[int]](7)
    assert agent.select_action(view(), CHOICES) == reference.select_action(
        view(), CHOICES
    )
    with pytest.raises(ValueError, match="Seed must be an integer or None"):
        agent.reset(seed)
    assert [agent.select_action(view(), CHOICES) for _ in range(30)] == [
        reference.select_action(view(), CHOICES) for _ in range(30)
    ]


def test_instances_and_global_randomness_are_independent() -> None:
    global_state = random.getstate()
    agent = RandomAgent[list[int]](9)
    reference = RandomAgent[list[int]](9)
    other = RandomAgent[list[int]](9)
    for _ in range(30):
        other.select_action(view(), CHOICES)
    other.reset()
    unseeded = RandomAgent[list[int]]()
    unseeded.select_action(view(), CHOICES)
    assert [agent.select_action(view(), CHOICES) for _ in range(30)] == [
        reference.select_action(view(), CHOICES) for _ in range(30)
    ]
    assert random.getstate() == global_state


def test_hidden_world_changes_do_not_affect_selection_or_engine_rng() -> None:
    game = HiddenGame()
    game.reset(42)
    hidden_world = game.clone()
    hidden_world._hands[1] = [999]
    hidden_world._deck.reverse()
    inputs = game.decision_for(0)
    assert hidden_world.decision_for(0) == inputs
    before = deepcopy(inputs)
    engine_rng = game._rng.getstate()
    agents = (RandomAgent[CardView](3), RandomAgent[CardView](3))
    assert agents[0].select_action(*inputs) == agents[1].select_action(
        *hidden_world.decision_for(0)
    )
    assert inputs == before
    assert game._rng.getstate() == engine_rng


@pytest.mark.parametrize("seed", [0, 1, 42, -17])
def test_complete_games_replay_with_agent_seeds(seed: int) -> None:
    game = TicTacToeEngine()
    agents: tuple[Agent[TicTacToeObservation], ...] = (RandomAgent(), RandomAgent())
    trajectories: list[list[Action]] = []
    outcomes: list[tuple[float, ...]] = []
    for _ in range(2):
        game.reset(100)
        for seat, agent in enumerate(agents):
            agent.reset(seed + seat)
        assert game._rng is not None
        engine_rng = game._rng.getstate()
        moves: list[Action] = []
        for _ in range(9):
            if game.is_terminal():
                break
            player = game.current_player()
            assert player is not None
            observation, legal = game.decision_for(player)
            action = agents[player].select_action(observation, legal)
            assert action in legal
            game.step(action)
            moves.append(action)
        assert game.is_terminal()
        assert 5 <= len(moves) <= 9
        assert game._rng.getstate() == engine_rng
        trajectories.append(moves)
        outcomes.append(game.get_rewards())
    assert trajectories[0] == trajectories[1]
    assert outcomes[0] == outcomes[1]
