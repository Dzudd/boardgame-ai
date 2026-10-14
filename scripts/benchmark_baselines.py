"""Save trusted metric reports for fixed-seat Tic-Tac-Toe baseline matchups."""

from pathlib import Path

from boardgame_ai.agents import Agent, RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.games.tic_tac_toe.evaluation import tic_tac_toe_outcome
from boardgame_ai.simulation import evaluate_batch, save_evaluation


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "runs" / "baseline_metrics"
    directory.mkdir(parents=True, exist_ok=True)
    configurations: tuple[
        tuple[str, tuple[str, ...], tuple[Agent[TicTacToeObservation], ...]], ...
    ] = (
        ("random_random", ("random", "random"), (RandomAgent(), RandomAgent())),
        (
            "heuristic_random",
            ("heuristic", "random"),
            (TicTacToeHeuristicAgent(), RandomAgent()),
        ),
        (
            "random_heuristic",
            ("random", "heuristic"),
            (RandomAgent(), TicTacToeHeuristicAgent()),
        ),
    )
    for name, labels, agents in configurations:
        report = evaluate_batch(
            TicTacToeEngine(),
            agents,
            outcome_for=tic_tac_toe_outcome,
            agent_labels=labels,
            num_matches=1000,
            seed=20261014,
            max_steps=9,
        )
        repeated = evaluate_batch(
            TicTacToeEngine(),
            agents,
            outcome_for=tic_tac_toe_outcome,
            agent_labels=labels,
            num_matches=1000,
            seed=20261014,
            max_steps=9,
        )
        if repeated != report:
            raise RuntimeError("Evaluation reports did not reproduce")
        save_evaluation(report, directory / f"{name}.json")
        print(f"{name}: {report.summary}")


if __name__ == "__main__":
    main()
