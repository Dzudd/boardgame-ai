"""Trace integrity and observational equivalence without local checkpoint fixtures."""

from dataclasses import replace
from typing import Any

import pytest

from boardgame_ai.core import Action
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.search.diagnosis import decision_trace, verify_trace
from boardgame_ai.search.puct import SearchAgent, SearchConfig, SimulationTrace


def predictor(view: Any, legal: tuple[Action, ...]) -> tuple[tuple[float, ...], float]:
    return (1 / len(legal),) * len(legal), 0.25


@pytest.mark.parametrize("exploration", [False, True])
@pytest.mark.parametrize("limits", [{}, {"max_nodes": 1}, {"max_depth": 1}])
def test_tracing_preserves_search(exploration: bool, limits: dict[str, int]) -> None:
    game = TicTacToeEngine()
    game.reset()
    for cell in (0, 3, 1):
        game.step(Action(kind="place", parameters=(("cell", cell),)))
    view, legal = game.decision_for(1)
    config = SearchConfig(simulations=32, exploration=exploration, **limits)
    traces: list[SimulationTrace] = []
    values: list[float] = []

    def recording(v: Any, a: tuple[Action, ...]) -> tuple[tuple[float, ...], float]:
        probabilities, value = predictor(v, a)
        values.append(value)
        return probabilities, value

    plain = SearchAgent(predictor, config)
    observed = SearchAgent(recording, config, on_simulation=traces.append)
    plain.reset(913)
    observed.reset(913)
    assert plain.select_action(view, legal) == observed.select_action(view, legal)
    assert plain.last_result == observed.last_result
    r = observed.last_result
    assert r is not None
    assert len(traces) == 32
    verify_trace(r, traces, values)
    if "max_nodes" in limits:
        assert any(t.stop == "nodes" for t in traces)
    if "max_depth" in limits:
        assert any(t.stop == "depth" for t in traces)


def test_trace_rejects_corrupt_return() -> None:
    game = TicTacToeEngine()
    game.reset()
    view, legal = game.decision_for(0)
    traces: list[SimulationTrace] = []
    agent = SearchAgent(
        predictor, SearchConfig(simulations=1), on_simulation=traces.append
    )
    agent.select_action(view, legal)
    result = agent.last_result
    assert result is not None
    traces[0] = replace(traces[0], returns=((0.25, -0.25),))
    with pytest.raises(AssertionError):
        verify_trace(result, traces, [0.25, 0.25])


@pytest.mark.parametrize("seat", [0, 1])
def test_terminal_trace_zero_and_no_inference(seat: int) -> None:
    game = TicTacToeEngine(starting_player=seat)
    game.reset()
    for cell in (0, 3, 1, 4):
        game.step(Action(kind="place", parameters=(("cell", cell),)))
    view, legal = game.decision_for(seat)
    values: list[float] = []
    traces: list[SimulationTrace] = []

    def winning(v: Any, a: tuple[Action, ...]) -> tuple[tuple[float, ...], float]:
        assert v.current_player is not None
        values.append(1.0)
        return tuple(1.0 if x.parameters[0][1] == 2 else 0.0 for x in a), 1.0

    agent = SearchAgent(
        winning, SearchConfig(simulations=16), on_simulation=traces.append
    )
    agent.select_action(view, legal)
    result = agent.last_result
    assert result is not None
    verify_trace(result, traces, values)
    assert len(values) == 1
    assert all(t.stop == "terminal" and t.continuation == (0.0, 0.0) for t in traces)
    assert all(t.returns[0][seat] == 1.0 for t in traces)


def test_public_probes_do_not_change_search() -> None:
    game = TicTacToeEngine()
    game.reset()
    for cell in (4, 3, 0):
        game.step(Action(kind="place", parameters=(("cell", cell),)))
    view, legal = game.decision_for(1)
    plain = SearchAgent(predictor, SearchConfig(simulations=16))
    plain.reset(81)
    action = plain.select_action(view, legal)
    traced = decision_trace(predictor, view, legal, 16, 81)
    assert traced["selected_cell"] == action.parameters[0][1]
    assert traced["backup_verification"]
    assert traced["valid_tactical_cells"] == (8,)
    probes = {p["cell"]: p for p in traced["candidate_probes"]}
    assert probes[2]["immediate_reply_wins"] == (8,)
    assert probes[8]["immediate_reply_wins"] == ()
    assert traced["probe_evaluator_calls"] == len(legal)
    # Terminal public metadata does not masquerade as a searched visit.
    for edge in traced["tree"]["edges"]:
        child = edge["child"]
        if child:
            for reply in child["edges"]:
                if reply["cell"] == 8 and edge["cell"] == 2:
                    assert reply["terminal"] and reply["reward"] == (1.0, -1.0)
