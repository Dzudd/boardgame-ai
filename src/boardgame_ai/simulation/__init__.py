"""Trusted orchestration for bounded games and reproducible match batches."""

from boardgame_ai.simulation.batch import SeededMatchResult, run_batch
from boardgame_ai.simulation.evaluation import (
    EvaluationReport,
    EvaluationSummary,
    GameOutcome,
    MatchRecord,
    evaluate_batch,
    save_evaluation,
)
from boardgame_ai.simulation.runner import MatchResult, play_game, run_match

__all__ = [
    "EvaluationReport",
    "EvaluationSummary",
    "GameOutcome",
    "MatchRecord",
    "MatchResult",
    "SeededMatchResult",
    "evaluate_batch",
    "play_game",
    "run_batch",
    "run_match",
    "save_evaluation",
]
