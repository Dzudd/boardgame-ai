"""Verify three 1,000-match Tic-Tac-Toe batches and their seeded reruns."""

from boardgame_ai.agents import Agent, RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.simulation import run_batch


def main() -> None:
    configurations: tuple[tuple[str, tuple[Agent[TicTacToeObservation], ...]], ...] = (
        ("random/random", (RandomAgent(), RandomAgent())),
        ("heuristic/random", (TicTacToeHeuristicAgent(), RandomAgent())),
        ("random/heuristic", (RandomAgent(), TicTacToeHeuristicAgent())),
    )
    for name, agents in configurations:
        game = TicTacToeEngine()
        first = run_batch(game, agents, num_matches=1000, seed=20261014, max_steps=9)
        final_view = game.observe(0)
        second = run_batch(game, agents, num_matches=1000, seed=20261014, max_steps=9)
        if first != second or final_view != game.observe(0):
            raise RuntimeError("Seeded batch reproduction failed")
        if not all(
            entry.result.terminated and not entry.result.truncated for entry in first
        ):
            raise RuntimeError("A match failed to complete")
        print(f"{name}: 1000 matches replayed identically, all completed")


if __name__ == "__main__":
    main()
