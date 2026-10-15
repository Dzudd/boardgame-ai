"""Run the predeclared frozen-search evaluation, separately from all training."""

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from boardgame_ai.agents import RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.agents.base import Agent
from boardgame_ai.agents.policy_agent import TicTacToePolicyAgent
from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.games.tic_tac_toe.evaluation import tic_tac_toe_outcome
from boardgame_ai.policy_audit import reachable_decisions, tactical_moves
from boardgame_ai.policy_evaluation import _case_seed
from boardgame_ai.search.neural import NeuralEvaluator
from boardgame_ai.search.puct import SearchAgent, SearchConfig
from boardgame_ai.simulation.evaluation import evaluate_batch
from boardgame_ai.training.checkpoint import load_ppo_checkpoint
from boardgame_ai.training.monitoring import isolated_rng


class MeasuredAgent(Agent[TicTacToeObservation]):
    def __init__(self, agent: Agent[TicTacToeObservation]) -> None:
        self.agent = agent
        self.times: list[float] = []
        self.calls = 0

    def reset(self, seed: int | None = None) -> None:
        self.agent.reset(seed)

    def select_action(
        self,
        observation: Observation[TicTacToeObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        start = perf_counter()
        action = self.agent.select_action(observation, legal_actions)
        self.times.append(perf_counter() - start)
        if isinstance(self.agent, SearchAgent):
            assert self.agent.last_result is not None
            self.calls += self.agent.last_result.evaluator_calls
        else:
            self.calls += 1
        return action

    def metrics(self) -> dict[str, Any]:
        return {
            "decisions": len(self.times),
            "evaluator_calls": self.calls,
            "latency_ms_mean": float(np.mean(self.times)) * 1000,
            "latency_ms_p50": float(np.percentile(self.times, 50)) * 1000,
            "latency_ms_p95": float(np.percentile(self.times, 95)) * 1000,
        }


def run(checkpoint: Path, output: Path) -> None:
    output.mkdir(exist_ok=False)
    sources = [p for p in checkpoint.parent.rglob("*") if p.is_file()]
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in sources}
    trainer = load_ppo_checkpoint(checkpoint)
    result: dict[str, Any] = {
        "protocol": "PHASE7_PROTOCOL.md",
        "seed": 20261112,
        "checkpoint_sha256": hashes[str(checkpoint)],
        "updates": trainer.update_count,
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "conditions": [],
    }
    for budget in (0, 16, 32):

        def factory(simulations: int = budget) -> Agent[TicTacToeObservation]:
            if simulations == 0:
                return TicTacToePolicyAgent(trainer.model, mode="greedy")
            return SearchAgent(
                NeuralEvaluator(trainer.model), SearchConfig(simulations=simulations)
            )

        warm = factory()
        game = TicTacToeEngine()
        game.reset()
        view, legal = game.decision_for(0)
        for _ in range(16):
            warm.select_action(view, legal)
        cases = []
        for name in ("random-v1", "heuristic-v1"):
            for seat in (0, 1):
                learner = MeasuredAgent(factory())
                opponent: Agent[TicTacToeObservation] = (
                    RandomAgent() if name == "random-v1" else TicTacToeHeuristicAgent()
                )
                agents = (learner, opponent) if seat == 0 else (opponent, learner)
                report = evaluate_batch(
                    TicTacToeEngine(),
                    agents,
                    outcome_for=tic_tac_toe_outcome,
                    agent_labels=("X", "O"),
                    num_matches=256,
                    seed=_case_seed(20261112, name, seat),
                    max_steps=9,
                )
                cases.append(
                    {
                        "opponent": name,
                        "seat": seat,
                        "report": asdict(report),
                        "metrics": learner.metrics(),
                    }
                )
        auditor = MeasuredAgent(factory())
        tactics = {(s, c): [0, 0] for s in (0, 1) for c in ("win", "block")}
        for view, legal in reachable_decisions():
            category, valid = tactical_moves(view, legal)
            action = auditor.select_action(view, legal)
            if valid:
                entry = tactics[view.player_id, category]
                entry[0] += 1
                entry[1] += int(action.parameters[0][1] in valid)
        result["conditions"].append(
            {
                "simulations": budget,
                "cases": cases,
                "tactics": [
                    {
                        "seat": s,
                        "category": c,
                        "positions": n,
                        "successes": ok,
                        "rate": ok / n,
                    }
                    for (s, c), (n, ok) in tactics.items()
                ],
                "census_metrics": auditor.metrics(),
            }
        )
        (output / f"condition-{budget}.json").write_text(
            json.dumps(result["conditions"][-1], indent=2, allow_nan=False) + "\n"
        )
        print(f"Completed {budget} simulations", flush=True)
    if any(sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError("Source artifacts changed")
    result["source_preserved"] = True
    result["source_hashes"] = hashes
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with isolated_rng():
        run(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
