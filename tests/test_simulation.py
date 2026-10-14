"""Runner lifecycle, actor order, reward accounting, and information boundaries."""

import random
from copy import deepcopy
from dataclasses import fields
from typing import Any

import pytest

from boardgame_ai.agents import Agent, RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.core import (
    Action,
    GameNotReadyError,
    IllegalActionError,
    Observation,
    StepResult,
)
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.simulation import MatchResult, play_game, run_match
from tests.hidden_game import CardView, HiddenGame
from tests.test_interfaces import DummyGame


class RecordingAgent[Data](Agent[Data]):
    def __init__(self) -> None:
        self.seeds: list[int | None] = []
        self.views: list[Observation[Data]] = []

    def reset(self, seed: int | None = None) -> None:
        self.seeds.append(seed)
        self.views.clear()

    def select_action(
        self, observation: Observation[Data], legal_actions: tuple[Action, ...]
    ) -> Action:
        self.views.append(deepcopy(observation))
        return legal_actions[0]


class RewardGame(DummyGame):
    def step(self, action: Action) -> StepResult:
        result = super().step(action)
        self._rewards = (0.5, 2.0)
        return StepResult(
            next_player=result.next_player,
            terminated=result.terminated,
            rewards=self._rewards,
        )


def test_match_resets_agents_and_follows_repeated_actors() -> None:
    game = DummyGame()
    agents = (RecordingAgent[int](), RecordingAgent[int]())
    result = run_match(game, agents, max_steps=3, engine_seed=42, agent_seeds=(7, 8))
    assert result == MatchResult(
        num_steps=3, total_rewards=(1.0, -1.0), terminated=True, truncated=False
    )
    assert [view.data for view in agents[0].views] == [0, 1]
    assert [view.data for view in agents[1].views] == [2]
    assert [agent.seeds for agent in agents] == [[7], [8]]
    assert all(
        view.player_id == seat
        for seat, agent in enumerate(agents)
        for view in agent.views
    )
    assert run_match(game, agents, max_steps=3) == result
    assert [agent.seeds for agent in agents] == [[7, None], [8, None]]
    assert len(agents[0].views) == 2  # Previous-game memory cleared by reset.


def test_rewards_sum_each_executed_step_once() -> None:
    game = RewardGame()
    agents = (RecordingAgent[int](), RecordingAgent[int]())
    result = run_match(game, agents, max_steps=3)
    assert result.total_rewards == (1.5, 6.0)  # General-sum incremental rewards.
    assert game.get_rewards() == (0.5, 2.0)


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_truncation_and_resume_do_not_reset_or_count_earlier_rewards(
    limit: int,
) -> None:
    game = RewardGame()
    agents = (RecordingAgent[int](), RecordingAgent[int]())
    first = run_match(game, agents, max_steps=limit)
    assert first.num_steps == limit
    assert first.total_rewards == (limit * 0.5, limit * 2.0)
    assert first.truncated and not first.terminated
    assert not game.is_terminal()
    rest = play_game(game, agents, max_steps=3 - limit)
    assert rest.terminated and not rest.truncated
    assert rest.num_steps == 3 - limit
    assert rest.total_rewards == ((3 - limit) * 0.5, (3 - limit) * 2.0)
    assert [agent.seeds for agent in agents] == [[None], [None]]


class TerminalAtReset(DummyGame):
    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._moves = [Action(kind="advance")] * 3


def test_terminal_reset_and_finished_position_require_no_decisions() -> None:
    game = TerminalAtReset()
    agents = (RecordingAgent[int](), RecordingAgent[int]())
    result = run_match(game, agents, max_steps=0)
    assert result == MatchResult(
        num_steps=0, total_rewards=(0.0, 0.0), terminated=True, truncated=False
    )
    assert not any(agent.views for agent in agents)
    assert play_game(game, agents, max_steps=3) == result


def test_play_game_requires_initialized_engine() -> None:
    with pytest.raises(GameNotReadyError):
        play_game(
            DummyGame(), (RecordingAgent[int](), RecordingAgent[int]()), max_steps=3
        )


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_limit_rejects_before_resets(limit: Any) -> None:
    game = DummyGame()
    agents = (RecordingAgent[int](), RecordingAgent[int]())
    with pytest.raises(ValueError, match="max_steps"):
        run_match(game, agents, max_steps=limit)
    assert not any(agent.seeds for agent in agents)
    with pytest.raises(GameNotReadyError):
        game.is_terminal()


def test_agent_count_and_shared_instance_reject_before_reset() -> None:
    agent = RecordingAgent[int]()
    for agents in ((agent,), (agent, agent)):
        with pytest.raises(ValueError):
            run_match(DummyGame(), agents, max_steps=3)
    assert agent.seeds == []


@pytest.mark.parametrize(
    "engine_seed, agent_seeds",
    [
        (True, (1, 2)),
        (1, (2,)),
        (1, (2, "bad")),
        (1, (2, False)),
    ],
)
def test_invalid_seeds_preserve_existing_game_and_agent_state(
    engine_seed: Any, agent_seeds: Any
) -> None:
    game = DummyGame()
    game.reset(42)
    game.step(Action(kind="advance"))
    agents = (RecordingAgent[int](), RecordingAgent[int]())
    before = game.observe(0)
    with pytest.raises(ValueError):
        run_match(
            game, agents, max_steps=3, engine_seed=engine_seed, agent_seeds=agent_seeds
        )
    assert game.observe(0) == before
    assert not any(agent.seeds for agent in agents)


class MutatingAgent(RecordingAgent[CardView]):
    def select_action(
        self, observation: Observation[CardView], legal_actions: tuple[Action, ...]
    ) -> Action:
        action = super().select_action(observation, legal_actions)
        observation.data.own_hand.append(999)
        return action


def test_hidden_chance_inputs_are_detached_and_results_remain_trusted() -> None:
    game = HiddenGame()
    reference = HiddenGame()
    reference.reset(42)
    agents = (MutatingAgent(), MutatingAgent())
    result = run_match(game, agents, max_steps=3, engine_seed=42, agent_seeds=(5, 6))
    for _ in range(3):
        reference.step(Action(kind="draw"))
    for seat in range(2):
        assert game.observe(seat) == reference.observe(seat)
        for view in agents[seat].views:
            assert view.player_id == view.current_player == seat
            assert 999 not in view.data.own_hand
    assert [len(view.data.own_hand) for view in agents[0].views] == [1, 2]
    assert [field.name for field in fields(result)] == [
        "num_steps",
        "total_rewards",
        "terminated",
        "truncated",
    ]


class IllegalAgent(RecordingAgent[int]):
    def select_action(
        self, observation: Observation[int], legal_actions: tuple[Action, ...]
    ) -> Action:
        return Action(kind="unavailable")


def test_illegal_actions_propagate_without_substitution() -> None:
    game = DummyGame()
    agents = (IllegalAgent(), RecordingAgent[int]())
    with pytest.raises(IllegalActionError):
        run_match(game, agents, max_steps=3)
    assert game.observe(0).data == 0
    assert agents[1].views == []


class FailingAgent(RecordingAgent[int]):
    def select_action(
        self, observation: Observation[int], legal_actions: tuple[Action, ...]
    ) -> Action:
        raise RuntimeError("Selection failed")


def test_agent_exceptions_stop_before_step() -> None:
    game = DummyGame()
    with pytest.raises(RuntimeError, match="Selection failed"):
        run_match(game, (FailingAgent(), RecordingAgent[int]()), max_steps=3)
    assert game.observe(0).data == 0


class EmptyChoices(DummyGame):
    def legal_actions(self, player_id: int) -> tuple[Action, ...]:
        super().legal_actions(player_id)
        return ()


class MissingActor(DummyGame):
    def current_player(self) -> int | None:
        self._require_ready()
        return None


class BadResult(DummyGame):
    def step(self, action: Action) -> StepResult:
        super().step(action)
        return StepResult(next_player=1, terminated=False, rewards=(0.0, 0.0))


@pytest.mark.parametrize("factory", [EmptyChoices, MissingActor, BadResult])
def test_engine_contract_failures_stop_execution(factory: type[DummyGame]) -> None:
    with pytest.raises(RuntimeError, match="Engine"):
        run_match(
            factory(), (RecordingAgent[int](), RecordingAgent[int]()), max_steps=3
        )


@pytest.mark.parametrize("heuristic_seat", [0, 1])
def test_tic_tac_toe_match_is_reproducible_without_global_rng_use(
    heuristic_seat: int,
) -> None:
    game = TicTacToeEngine()
    agents: tuple[Agent[TicTacToeObservation], ...] = (
        TicTacToeHeuristicAgent() if heuristic_seat == 0 else RandomAgent(),
        TicTacToeHeuristicAgent() if heuristic_seat == 1 else RandomAgent(),
    )
    global_state = random.getstate()
    result = run_match(game, agents, max_steps=9, engine_seed=10, agent_seeds=(20, 30))
    board = game.observe(0)
    assert result.terminated and not result.truncated
    assert 5 <= result.num_steps <= 9
    assert result.total_rewards == game.get_rewards()
    assert (
        run_match(game, agents, max_steps=9, engine_seed=10, agent_seeds=(20, 30))
        == result
    )
    assert game.observe(0) == board
    assert random.getstate() == global_state
