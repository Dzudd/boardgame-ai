"""Independent exhaustive rules checks and bounded-search semantic tests."""

import random
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from boardgame_ai.core import Action, GameOverError, IllegalActionError
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.search.neural import NeuralEvaluator
from boardgame_ai.search.puct import SearchAgent, SearchConfig, backup
from boardgame_ai.search.simulator import PublicTicTacToeSimulator


def uniform(view: Any, legal: tuple[Action, ...]) -> tuple[tuple[float, ...], float]:
    return (1 / len(legal),) * len(legal), 0.4


@pytest.mark.parametrize("starter", [0, 1])
def test_exhaustive_engine_agreement(starter: int) -> None:
    game = TicTacToeEngine(starting_player=starter)
    game.reset(912)
    seen: set[tuple[int | None, ...]] = set()
    transitions = 0

    def visit(engine: TicTacToeEngine) -> None:
        nonlocal transitions
        seat = engine.current_player()
        view = engine.observe(0 if seat is None else seat)
        if view.data.board in seen:
            return
        seen.add(view.data.board)
        legal = engine.legal_actions(view.player_id)
        sim = PublicTicTacToeSimulator.from_observation(view, legal)
        assert sim.legal_actions() == legal
        assert (sim.actor is None) == engine.is_terminal()
        if engine.is_terminal():
            with pytest.raises(GameOverError):
                sim.step(Action(kind="place", parameters=(("cell", 0),)))
            with pytest.raises(ValueError):
                SearchAgent(uniform).select_action(view, legal)
            return
        for i in range(9):
            action = Action(kind="place", parameters=(("cell", i),))
            if action not in legal:
                with pytest.raises(IllegalActionError):
                    sim.step(action)
                continue
            branch = engine.clone()
            result = branch.step(action)
            child, reward = sim.step(action)
            transitions += 1
            assert reward == result.rewards
            assert child.actor == result.next_player
            assert (child.actor is None) == result.terminated
            assert child.board == branch.observe(0).data.board
            assert child.observe().phase == branch.observe(0).phase
            assert sim.board == view.data.board == engine.observe(0).data.board
            visit(branch)

    visit(game)
    assert len(seen) == 5478
    assert transitions == 16167


def test_public_constructor_rejections() -> None:
    game = TicTacToeEngine()
    game.reset()
    view, legal = game.decision_for(0)
    for invalid in (
        legal[:-1],
        legal + legal[:1],
        (Action(kind="place", parameters=(("cell", 0.0),)),) + legal[1:],
    ):
        with pytest.raises(ValueError):
            PublicTicTacToeSimulator.from_observation(view, invalid)
    with pytest.raises(ValueError):
        PublicTicTacToeSimulator.from_observation(replace(view, player_id=1), legal)
    with pytest.raises(ValueError):
        PublicTicTacToeSimulator.from_observation(
            replace(
                view,
                data=replace(view.data, board=(0, 0, 0, 1, 1, 1, None, None, None)),
            ),
            (),
        )
    with pytest.raises(ValueError):
        PublicTicTacToeSimulator.from_observation(
            replace(view, phase="finished", current_player=None), ()
        )


@pytest.mark.parametrize("simulations", [1, 16, 32])
@pytest.mark.parametrize("nodes", [1, 2, 289])
@pytest.mark.parametrize("depth", [1, 2, 9])
def test_limits_visits_and_mask(simulations: int, nodes: int, depth: int) -> None:
    game = TicTacToeEngine()
    game.reset()
    game.step(game.legal_actions(0)[0])
    view, legal = game.decision_for(1)
    agent = SearchAgent(
        uniform, SearchConfig(simulations=simulations, max_nodes=nodes, max_depth=depth)
    )
    assert agent.select_action(view, tuple(reversed(legal))) in legal
    r = agent.last_result
    assert r is not None
    assert sum(e.visits for e in r.root.edges) == simulations
    assert [e.action for e in r.root.edges] == list(legal)
    assert r.nodes <= nodes and r.depth <= depth
    assert r.evaluator_calls <= simulations + 1
    assert all(e.prior > 0 for e in r.root.edges)
    assert sum(e.prior for e in r.root.edges) == pytest.approx(1)
    if depth == 1:
        assert r.depth_cutoffs == simulations
    elif nodes == 1:
        assert r.node_cutoffs == simulations

    def check(node: Any) -> None:
        for e in node.edges:
            if e.child:
                assert sum(c.visits for c in e.child.edges) == e.visits - 1
                check(e.child)

    check(r.root)


@pytest.mark.parametrize("seat", [0, 1])
def test_terminal_parent_value_and_zero_bootstrap(seat: int) -> None:
    game = TicTacToeEngine(starting_player=seat)
    game.reset()
    for cell in (0, 3, 1, 4):
        game.step(Action(kind="place", parameters=(("cell", cell),)))
    view, legal = game.decision_for(seat)

    def forced(v: Any, actions: tuple[Action, ...]) -> tuple[tuple[float, ...], float]:
        assert v.current_player is not None
        return tuple(1.0 if a.parameters[0][1] == 2 else 0 for a in actions), -1.0

    agent = SearchAgent(forced, SearchConfig(simulations=16))
    assert agent.select_action(view, legal).parameters[0][1] == 2
    r = agent.last_result
    assert r is not None
    e = r.root.edges[0]
    assert e.action.parameters[0][1] == 2
    assert e.visits == 16 and r.evaluator_calls == 1
    assert e.total[seat] / e.visits == 1
    assert e.total[1 - seat] / e.visits == -1


def test_vector_backup_same_actor_and_rewards() -> None:
    # Vector backup is independent of actor changes, including consecutive own moves.
    leaf = (0.25, -0.25)
    assert backup((0, 0), backup((0.5, -0.5), leaf)) == (0.75, -0.75)
    assert backup((1, -1), (0, 0)) == (1, -1)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_evaluator(value: float) -> None:
    game = TicTacToeEngine()
    game.reset()
    view, legal = game.decision_for(0)

    def invalid(v: Any, a: tuple[Action, ...]) -> tuple[tuple[float, ...], float]:
        return (1.0,) * len(a), value

    with pytest.raises(ValueError):
        SearchAgent(invalid).select_action(view, legal)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"simulations": 0},
        {"max_nodes": 0},
        {"max_depth": 0},
        {"simulations": True},
        {"max_depth": 1.5},
    ],
)
def test_invalid_budgets(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)


def test_exploration_reproducibility_and_rng() -> None:
    game = TicTacToeEngine()
    game.reset()
    view, legal = game.decision_for(0)
    py = random.getstate()
    npstate = np.random.get_state()
    ts = torch.get_rng_state().clone()
    agent = SearchAgent(uniform, SearchConfig(exploration=True))

    def trace(seed: int) -> tuple[Any, ...]:
        agent.reset(seed)
        action = agent.select_action(view, legal)
        r = agent.last_result
        assert r is not None
        return action, [(e.prior, e.visits, e.total) for e in r.root.edges]

    assert trace(123) == trace(123)
    assert trace(124) != trace(123)
    assert random.getstate() == py
    after = np.random.get_state()
    assert (
        after[0] == npstate[0]
        and np.array_equal(after[1], npstate[1])
        and after[2:] == npstate[2:]
    )
    assert torch.equal(torch.get_rng_state(), ts)


def test_neural_mask_and_snapshot() -> None:
    model = TicTacToeActorCritic()
    evaluator = NeuralEvaluator(model)
    game = TicTacToeEngine()
    game.reset()
    game.step(game.legal_actions(0)[0])
    view, legal = game.decision_for(1)
    before = evaluator(view, legal)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(100)
    assert before == evaluator(view, legal)
    probabilities, _ = before
    assert len(probabilities) == 8 and sum(probabilities) == pytest.approx(1)


def test_multiple_wins_accepted_and_ties() -> None:
    game = TicTacToeEngine()
    game.reset()
    for cell in (0, 1, 4, 3, 2, 7):
        game.step(Action(kind="place", parameters=(("cell", cell),)))
    view, legal = game.decision_for(0)
    # Both 6 and 8 win; search need not choose a single designated answer.
    agent = SearchAgent(uniform, SearchConfig(simulations=32))
    assert agent.select_action(view, legal).parameters[0][1] in (6, 8)
    game.reset()
    view, legal = game.decision_for(0)
    agent = SearchAgent(uniform, SearchConfig(simulations=1))
    assert agent.select_action(view, tuple(reversed(legal))) == legal[0]


@pytest.mark.parametrize("seat", [0, 1])
def test_leaf_perspective_and_clipping(seat: int) -> None:
    game = TicTacToeEngine(starting_player=seat)
    game.reset()
    view, legal = game.decision_for(seat)

    def opponent_value(
        v: Any, a: tuple[Action, ...]
    ) -> tuple[tuple[float, ...], float]:
        # Child actor is the opponent: positive leaf value is negative for parent.
        return (1.0,) * len(a), 10.0

    agent = SearchAgent(opponent_value, SearchConfig(simulations=1, max_depth=1))
    agent.select_action(view, legal)
    r = agent.last_result
    assert r is not None
    edge = r.root.edges[0]
    assert edge.total[seat] == -1 and edge.total[1 - seat] == 1


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -1.0, 1e308])
def test_invalid_priors(probability: float) -> None:
    game = TicTacToeEngine()
    game.reset()
    view, legal = game.decision_for(0)

    def invalid(v: Any, a: tuple[Action, ...]) -> tuple[tuple[float, ...], float]:
        return (probability,) * len(a), 0.0

    with pytest.raises(ValueError):
        SearchAgent(invalid).select_action(view, legal)


def test_no_engine_or_clone_access(monkeypatch: pytest.MonkeyPatch) -> None:
    game = TicTacToeEngine()
    game.reset()
    view, legal = game.decision_for(0)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Authoritative engine accessed")

    for name in ("__init__", "clone", "observe", "legal_actions", "step"):
        monkeypatch.setattr(TicTacToeEngine, name, forbidden)
    agent = SearchAgent(uniform)
    assert agent.select_action(view, legal) in legal


@pytest.mark.parametrize("seat", [0, 1])
def test_deep_tree_parent_actor_and_visit_returns(seat: int) -> None:
    game = TicTacToeEngine(starting_player=seat)
    game.reset()
    for cell in (0, 3, 1):
        game.step(Action(kind="place", parameters=(("cell", cell),)))
    actor = 1 - seat
    view, legal = game.decision_for(actor)
    agent = SearchAgent(uniform, SearchConfig(simulations=256))
    agent.select_action(view, legal)
    r = agent.last_result
    assert r is not None
    visited_terminal = 0

    def check(node: Any) -> None:
        nonlocal visited_terminal
        for edge in node.edges:
            child, reward = node.state.step(edge.action)
            if child.actor is None and edge.visits:
                visited_terminal += 1
                assert edge.total == pytest.approx(
                    tuple(edge.visits * x for x in reward)
                )
            if edge.child:
                assert edge.child.state.actor == 1 - node.state.actor
                check(edge.child)

    check(r.root)
    assert visited_terminal > 0
    # The opponent must discover the required block rather than cooperate with X/O.
    assert r.action.parameters[0][1] == 2


def test_neural_rng_and_source_mode_preserved() -> None:
    model = TicTacToeActorCritic().train()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    snapshot = [p.detach().clone() for p in model.parameters()]
    py = random.getstate()
    ns = np.random.get_state()
    ts = torch.get_rng_state().clone()
    evaluator = NeuralEvaluator(model)
    game = TicTacToeEngine()
    game.reset()
    view, legal = game.decision_for(0)
    SearchAgent(evaluator, SearchConfig(exploration=True)).select_action(view, legal)
    assert model.training
    for before, p in zip(snapshot, model.parameters(), strict=True):
        assert (
            torch.equal(before, p) and p.grad is not None and bool((p.grad == 1).all())
        )
    assert random.getstate() == py and torch.equal(ts, torch.get_rng_state())
    after = np.random.get_state()
    assert ns[0] == after[0] and np.array_equal(ns[1], after[1]) and ns[2:] == after[2:]
