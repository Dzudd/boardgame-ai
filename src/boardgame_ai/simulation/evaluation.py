"""Trusted outcome evaluation and explicit JSON summary export."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

from boardgame_ai.agents import Agent
from boardgame_ai.core import GameEngine, IllegalActionError
from boardgame_ai.simulation.batch import _derive_seed, _validate_batch
from boardgame_ai.simulation.runner import MatchResult, run_match


@dataclass(frozen=True, slots=True, kw_only=True)
class GameOutcome:
    """Game-defined terminal scores, winners, and explicit draw classification.

    Multiple winners are supported. Games without winners may use an empty tuple
    and draw=False (for example cooperative games). Scores need not be rewards.
    """

    scores: tuple[float, ...]
    winners: tuple[int, ...] = ()
    draw: bool = False

    def __post_init__(self) -> None:
        if not self.scores or not all(isfinite(score) for score in self.scores):
            raise ValueError("Outcome scores must be nonempty and finite")
        if len(set(self.winners)) != len(self.winners) or any(
            type(seat) is not int or not 0 <= seat < len(self.scores)
            for seat in self.winners
        ):
            raise ValueError("Outcome winners must be distinct valid seats")
        if self.draw and self.winners:
            raise ValueError("Draw outcomes cannot have winners")


@dataclass(frozen=True, slots=True, kw_only=True)
class MatchRecord:
    match_index: int
    engine_seed: int
    agent_seeds: tuple[int, ...]
    result: MatchResult | None
    outcome: GameOutcome | None
    error: Literal["illegal_action", "match_failed", "outcome_failed"] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationSummary:
    attempted: int
    completed: int
    truncated: int
    errors: int
    wins: tuple[int, ...]
    draws: int
    total_scores: tuple[float, ...]
    mean_scores: tuple[float, ...] | None
    mean_game_length: float | None
    min_game_length: int | None
    max_game_length: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationReport:
    game_id: str
    agent_labels: tuple[str, ...]
    batch_seed: int
    start_index: int
    max_steps: int
    records: tuple[MatchRecord, ...]
    summary: EvaluationSummary


def evaluate_batch[Data](
    game: GameEngine[Data],
    agents: tuple[Agent[Data], ...],
    *,
    outcome_for: Callable[[GameEngine[Data]], GameOutcome],
    agent_labels: tuple[str, ...],
    num_matches: int,
    seed: int,
    max_steps: int,
    start_index: int = 0,
) -> EvaluationReport:
    """Record a fixed-seat batch, continuing after ordinary match errors.

    The trusted outcome adapter runs only after termination. Never infer draws
    from truncation/errors or winners from rewards. Exception messages, classes,
    and tracebacks are not retained. Interrupts propagate. Subsequent matches
    reset the supplied objects; failed matches are not retried or rolled back.
    """
    _validate_batch(game, agents, num_matches, seed, max_steps, start_index)
    if len(agent_labels) != game.num_players or any(
        not label for label in agent_labels
    ):
        raise ValueError("Provide a nonempty agent label per seat")
    records: list[MatchRecord] = []
    wins = [0] * game.num_players
    scores = [0.0] * game.num_players
    lengths: list[int] = []
    truncated = errors = draws = 0
    for index in range(start_index, start_index + num_matches):
        engine_seed = _derive_seed(seed, index, "engine")
        agent_seeds = tuple(
            _derive_seed(seed, index, f"agent:{seat}")
            for seat in range(game.num_players)
        )
        result: MatchResult | None = None
        outcome: GameOutcome | None = None
        error: Literal["illegal_action", "match_failed", "outcome_failed"] | None = None
        try:
            result = run_match(
                game,
                agents,
                max_steps=max_steps,
                engine_seed=engine_seed,
                agent_seeds=agent_seeds,
            )
        except IllegalActionError:
            error = "illegal_action"
        except Exception:
            error = "match_failed"
        else:
            if result.truncated:
                truncated += 1
            else:
                try:
                    outcome = outcome_for(game)
                    if len(outcome.scores) != game.num_players:
                        raise ValueError("Outcome score count must match seats")
                except Exception:
                    outcome = None
                    error = "outcome_failed"
                else:
                    lengths.append(result.num_steps)
                    draws += int(outcome.draw)
                    for seat in outcome.winners:
                        wins[seat] += 1
                    for seat, score in enumerate(outcome.scores):
                        scores[seat] += score
        errors += int(error is not None)
        records.append(
            MatchRecord(
                match_index=index,
                engine_seed=engine_seed,
                agent_seeds=agent_seeds,
                result=result,
                outcome=outcome,
                error=error,
            )
        )
    completed = len(lengths)
    return EvaluationReport(
        game_id=game.spec.game_id,
        agent_labels=agent_labels,
        batch_seed=seed,
        start_index=start_index,
        max_steps=max_steps,
        records=tuple(records),
        summary=EvaluationSummary(
            attempted=num_matches,
            completed=completed,
            truncated=truncated,
            errors=errors,
            wins=tuple(wins),
            draws=draws,
            total_scores=tuple(scores),
            mean_scores=tuple(score / completed for score in scores)
            if completed
            else None,
            mean_game_length=sum(lengths) / completed if completed else None,
            min_game_length=min(lengths) if lengths else None,
            max_game_length=max(lengths) if lengths else None,
        ),
    )


def save_evaluation(report: EvaluationReport, path: str | Path) -> None:
    """Write trusted summary JSON to an explicitly chosen path (overwrites it).

    Parent directories must exist. No game observations or exception text are
    exported. Seeds can reconstruct hidden setup; this is not a player replay.
    """
    payload = {"schema_version": 1, **asdict(report)}
    Path(path).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
