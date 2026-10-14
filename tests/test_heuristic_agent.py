"""Public-board tactics, deterministic behavior, and complete baseline games."""

import random
from copy import deepcopy
from typing import cast

import pytest

from boardgame_ai.agents import Agent, RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.games.tic_tac_toe.state import Board


def place(cell: int) -> Action:
    return Action(kind="place", parameters=(("cell", cell),))


def inputs(
    board: tuple[int | None, ...], player: int = 0
) -> tuple[Observation[TicTacToeObservation], tuple[Action, ...]]:
    return (
        Observation(
            player_id=player,
            current_player=player,
            phase="play",
            data=TicTacToeObservation(board=cast(Board, board)),
        ),
        tuple(place(cell) for cell, mark in enumerate(board) if mark is None),
    )


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize(
    "line",
    [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ],
)
@pytest.mark.parametrize("blocking", [False, True])
def test_wins_and_blocks_every_line(
    player: int, line: tuple[int, int, int], blocking: bool
) -> None:
    board: list[int | None] = [None] * 9
    board[line[0]] = board[line[1]] = 1 - player if blocking else player
    observation, legal = inputs(tuple(board), player)
    assert TicTacToeHeuristicAgent().select_action(observation, legal) == place(line[2])


def test_win_takes_priority_over_block() -> None:
    observation, legal = inputs((0, 0, None, 1, 1, None, None, None, None))
    assert TicTacToeHeuristicAgent().select_action(observation, legal) == place(2)


@pytest.mark.parametrize(
    ("board", "expected"),
    [
        ((None,) * 9, 4),
        ((None, None, None, None, 1, None, None, None, None), 0),
        ((1, None, None, None, 0, None, None, None, None), 2),
    ],
)
def test_positional_preferences(board: tuple[int | None, ...], expected: int) -> None:
    observation, legal = inputs(board)
    assert TicTacToeHeuristicAgent().select_action(observation, legal) == place(
        expected
    )


def test_order_independent_ties_input_purity_and_no_randomness() -> None:
    observation, legal = inputs((0, 0, None, 0, 1, 1, None, 1, None))
    before = deepcopy((observation, legal))
    global_rng = random.getstate()
    agent: Agent[TicTacToeObservation] = TicTacToeHeuristicAgent()
    for seed in (None, 0, 42, -17):
        agent.reset(seed)
        selected = agent.select_action(observation, tuple(reversed(legal)))
        assert selected == place(2)  # Both cells 2 and 6 win; choose row-major.
        assert any(selected is action for action in legal)
    assert (observation, legal) == before
    assert random.getstate() == global_rng


def test_selects_only_supplied_candidates() -> None:
    observation, _ = inputs((None,) * 9)
    supplied = (place(7), place(5))
    assert TicTacToeHeuristicAgent().select_action(observation, supplied) is supplied[1]
    edges = (place(7), place(5), place(3), place(1))
    assert TicTacToeHeuristicAgent().select_action(observation, edges) is edges[3]
    assert (
        TicTacToeHeuristicAgent().select_action(observation, (supplied[0],))
        is supplied[0]
    )


@pytest.mark.parametrize(
    "choices",
    [
        (),
        (Action(kind="draw"),),
        (Action(kind="place"),),
        (place(9),),
        (Action(kind="place", parameters=(("cell", True),)),),
        (place(4),),
    ],
)
def test_rejects_empty_or_incompatible_choices(choices: tuple[Action, ...]) -> None:
    observation, _ = inputs((None, None, None, None, 1, None, None, None, None))
    with pytest.raises(ValueError):
        TicTacToeHeuristicAgent().select_action(observation, choices)


@pytest.mark.parametrize("player, actor", [(0, 1), (1, None), (2, 2), (True, True)])
def test_requires_valid_observing_actor(player: int, actor: int | None) -> None:
    observation, legal = inputs((None,) * 9)
    invalid = Observation(
        player_id=player, current_player=actor, phase="play", data=observation.data
    )
    with pytest.raises(ValueError):
        TicTacToeHeuristicAgent().select_action(invalid, legal)


@pytest.mark.parametrize("heuristic_seat", [0, 1])
@pytest.mark.parametrize("seed", [0, 42, -17])
def test_complete_games_against_random_replay(heuristic_seat: int, seed: int) -> None:
    game = TicTacToeEngine()
    agents: list[Agent[TicTacToeObservation]] = [RandomAgent(), RandomAgent()]
    agents[heuristic_seat] = TicTacToeHeuristicAgent()
    histories: list[list[Action]] = []
    rewards: list[tuple[float, ...]] = []
    for _ in range(2):
        game.reset(42)
        for seat, agent in enumerate(agents):
            agent.reset(seed + seat)
        assert game._rng is not None
        rng_state = game._rng.getstate()
        history: list[Action] = []
        for _ in range(9):
            if game.is_terminal():
                break
            player = game.current_player()
            assert player is not None
            observation, legal = game.decision_for(player)
            before = deepcopy((observation, legal))
            action = agents[player].select_action(observation, legal)
            assert action in legal
            assert (observation, legal) == before
            game.step(action)
            history.append(action)
        assert game.is_terminal()
        assert game._rng.getstate() == rng_state
        histories.append(history)
        rewards.append(game.get_rewards())
    assert histories[0] == histories[1]
    assert rewards[0] == rewards[1]


def test_heuristic_self_play_draws() -> None:
    game = TicTacToeEngine()
    game.reset()
    agents = (TicTacToeHeuristicAgent(), TicTacToeHeuristicAgent())
    for agent in agents:
        agent.reset(42)
    for _ in range(9):
        if game.is_terminal():
            break
        player = game.current_player()
        assert player is not None
        game.step(agents[player].select_action(*game.decision_for(player)))
    assert game.is_terminal()
    assert game.get_rewards() == (0.0, 0.0)
