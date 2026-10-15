"""Read-only public search diagnosis; replay, not a new strength evaluation."""

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from math import isclose, sqrt
from pathlib import Path
from typing import Any

import torch

from boardgame_ai.agents import TicTacToeHeuristicAgent
from boardgame_ai.agents.policy_agent import TicTacToePolicyAgent
from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.models.encoders import encode_tic_tac_toe_action
from boardgame_ai.policy_audit import tactical_moves
from boardgame_ai.search.neural import NeuralEvaluator
from boardgame_ai.search.puct import (
    Evaluator,
    Node,
    SearchAgent,
    SearchConfig,
    SearchResult,
    SimulationTrace,
)
from boardgame_ai.search.simulator import PublicTicTacToeSimulator
from boardgame_ai.training.checkpoint import load_ppo_checkpoint
from boardgame_ai.training.monitoring import isolated_rng


def verify_trace(
    result: SearchResult, traces: list[SimulationTrace], values: list[float]
) -> None:
    """Independently reconstruct rewards, continuations, visits and vector W."""
    totals: dict[tuple[int, ...], list[float]] = {}
    live = 1  # Root prediction is not backed up.
    for index, trace in enumerate(traces):
        assert trace.index == index
        state = result.root.state
        prefixes = []
        cells: tuple[int, ...] = ()
        assert len(trace.actions) == len(trace.rewards) == len(trace.returns)
        for action, reward in zip(trace.actions, trace.rewards, strict=True):
            state, expected = state.step(action)
            assert reward == expected
            cells += (encode_tic_tac_toe_action(action),)
            prefixes.append(cells)
        assert state == trace.leaf
        if state.actor is None:
            assert trace.stop == "terminal"
            expected_g = (0.0, 0.0)
        else:
            assert trace.stop in ("new-leaf", "nodes", "depth")
            value = max(-1.0, min(1.0, values[live]))
            live += 1
            expected_g = (value, -value) if state.actor == 0 else (-value, value)
        assert trace.continuation == expected_g
        for i in reversed(range(len(trace.actions))):
            reward = trace.rewards[i]
            expected_g = (reward[0] + expected_g[0], reward[1] + expected_g[1])
            assert trace.returns[i] == expected_g
            entry = totals.setdefault(prefixes[i], [0.0, 0.0, 0.0])
            entry[0] += 1
            entry[1] += expected_g[0]
            entry[2] += expected_g[1]
    assert live == len(values) == result.evaluator_calls
    assert sum(e.visits for e in result.root.edges) == len(traces)

    def check(node: Node, prefix: tuple[int, ...]) -> None:
        for edge in node.edges:
            path = prefix + (encode_tic_tac_toe_action(edge.action),)
            n, x, o = totals.get(path, [0.0, 0.0, 0.0])
            assert edge.visits == n
            assert isclose(edge.total[0], x, abs_tol=1e-12)
            assert isclose(edge.total[1], o, abs_tol=1e-12)
            if edge.child is not None:
                check(edge.child, path)

    check(result.root, ())


def describe_tree(node: Node) -> dict[str, Any]:
    actor = node.state.actor
    assert actor is not None
    visits = sum(e.visits for e in node.edges)
    rows = []
    for edge in node.edges:
        child, reward = node.state.step(edge.action)
        q = edge.total[actor] / edge.visits if edge.visits else 0.0
        u = 1.5 * edge.prior * sqrt(1 + visits) / (1 + edge.visits)
        rows.append(
            {
                "cell": encode_tic_tac_toe_action(edge.action),
                "prior": edge.prior,
                "visits": edge.visits,
                "W": edge.total,
                "Q_parent": q,
                "U_after_search": u,
                "PUCT_after_search": q + u,
                "reward": reward,
                "terminal": child.actor is None,
                "child": describe_tree(edge.child) if edge.child else None,
            }
        )
    return {"board": node.state.board, "actor": actor, "edges": rows}


def decision_trace(
    evaluator: Evaluator,
    view: Observation[TicTacToeObservation],
    legal: tuple[Action, ...],
    budget: int,
    seed: int,
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    traces: list[SimulationTrace] = []

    def measured(
        v: Observation[TicTacToeObservation], actions: tuple[Action, ...]
    ) -> tuple[tuple[float, ...], float]:
        priors, value = evaluator(v, actions)
        evaluations.append(
            {
                "board": v.data.board,
                "actor": v.player_id,
                "legal_cells": [encode_tic_tac_toe_action(a) for a in actions],
                "raw_priors": priors,
                "raw_value": value,
                "clipped_value": max(-1.0, min(1.0, value)),
            }
        )
        return priors, value

    agent = SearchAgent(
        measured, SearchConfig(simulations=budget), on_simulation=traces.append
    )
    agent.reset(seed)
    action = agent.select_action(view, legal)
    result = agent.last_result
    assert result is not None
    verify_trace(result, traces, [row["raw_value"] for row in evaluations])
    # Public one-step probes are diagnostics only, never fed back into search.
    public = PublicTicTacToeSimulator.from_observation(view, legal)
    probes = []
    for candidate in legal:
        child, reward = public.step(candidate)
        replies = child.legal_actions()
        if child.actor is None:
            probes.append(
                {
                    "cell": encode_tic_tac_toe_action(candidate),
                    "reward": reward,
                    "terminal": True,
                }
            )
            continue
        category, wins = tactical_moves(child.observe(), replies)
        _, value = evaluator(child.observe(), replies)
        probes.append(
            {
                "cell": encode_tic_tac_toe_action(candidate),
                "child_actor": child.actor,
                "child_raw_value": value,
                "O_child_continuation": max(-1.0, min(1.0, value))
                * (1 if child.actor == 1 else -1),
                "immediate_reply_wins": wins if category == "win" else (),
            }
        )
    category, valid = tactical_moves(view, legal)
    return {
        "board": view.data.board,
        "actor": view.player_id,
        "selected_cell": encode_tic_tac_toe_action(action),
        "tactical_category": category,
        "valid_tactical_cells": valid,
        "evaluator_calls": result.evaluator_calls,
        "probe_evaluator_calls": sum(not p.get("terminal", False) for p in probes),
        "nodes": result.nodes,
        "max_depth": result.depth,
        "node_cutoffs": result.node_cutoffs,
        "depth_cutoffs": result.depth_cutoffs,
        "tree": describe_tree(result.root),
        "evaluations": evaluations,
        "simulations": [asdict(t) for t in traces],
        "candidate_probes": probes,
        "backup_verification": True,
    }


def run(checkpoint: Path, previous_report: Path, output: Path) -> None:
    """Replay match zero in the existing consumed schedule; never overwrite."""
    if output.exists():
        raise ValueError("Diagnosis output must not exist")
    source = json.loads(previous_report.read_text())
    paths = [p for p in checkpoint.parent.rglob("*") if p.is_file()] + [previous_report]
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in paths}
    digest = hashes[str(checkpoint)]
    if digest != source["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from measured report")
    trainer = load_ppo_checkpoint(checkpoint)
    evaluator = NeuralEvaluator(trainer.model)
    cases = []
    for condition in source["conditions"]:
        budget = condition["simulations"]
        case = next(
            c
            for c in condition["cases"]
            if c["opponent"] == "heuristic-v1" and c["seat"] == 1
        )
        record = case["report"]["records"][0]
        game = TicTacToeEngine()
        game.reset(record["engine_seed"])
        heuristic = TicTacToeHeuristicAgent()
        raw = TicTacToePolicyAgent(trainer.model, mode="greedy")
        raw.reset(record["agent_seeds"][1])
        moves = []
        decisions = []
        rewards = [0.0, 0.0]
        while not game.is_terminal():
            seat = game.current_player()
            assert seat is not None
            view, legal = game.decision_for(seat)
            if seat == 0:
                action = heuristic.select_action(view, legal)
            elif budget == 0:
                action = raw.select_action(view, legal)
                decisions.append(
                    {
                        "board": view.data.board,
                        "selected_cell": encode_tic_tac_toe_action(action),
                        "tactics": tactical_moves(view, legal),
                    }
                )
            else:
                row = decision_trace(
                    evaluator, view, legal, budget, record["agent_seeds"][1]
                )
                decisions.append(row)
                action = next(
                    a
                    for a in legal
                    if encode_tic_tac_toe_action(a) == row["selected_cell"]
                )
            moves.append({"seat": seat, "cell": encode_tic_tac_toe_action(action)})
            step = game.step(action)
            rewards = [x + y for x, y in zip(rewards, step.rewards, strict=True)]
        assert record["result"]["num_steps"] == len(moves)
        assert record["result"]["total_rewards"] == rewards
        cases.append(
            {
                "simulations": budget,
                "match_index": record["match_index"],
                "moves": moves,
                "decisions": decisions,
                "rewards": rewards,
            }
        )
    assert all(sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    report = {
        "schema_version": 1,
        "purpose": "public frozen heuristic-O diagnosis, no strength test or tuning",
        "checkpoint_sha256": digest,
        "updates": trainer.update_count,
        "replayed_seed": source["seed"],
        "fresh_seed_consumed": False,
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "source_hashes": hashes,
        "source_preserved": True,
        "conditions": cases,
    }
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.mkdir()
    (output / "trace.json").write_text(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("previous_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with isolated_rng():
        run(args.checkpoint, args.previous_report, args.output)


if __name__ == "__main__":
    main()
