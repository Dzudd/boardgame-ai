"""Read-only phase assessment: paired baseline controls and training diagnostics."""

import json
from dataclasses import asdict, dataclass
from math import fsum, isfinite, sqrt
from pathlib import Path
from typing import Any, Literal

from boardgame_ai.agents import RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.agents.base import Agent
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.games.tic_tac_toe.evaluation import tic_tac_toe_outcome
from boardgame_ai.policy_evaluation import PolicyEvaluation, evaluate_policy_checkpoint
from boardgame_ai.simulation.evaluation import EvaluationReport, evaluate_batch


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreEstimate:
    n: int
    mean: float
    standard_error: float | None
    ci_low: float | None
    ci_high: float | None


def estimate_scores(values: tuple[float, ...]) -> ScoreEstimate:
    """Mean ±1.96 sample standard error; uncertainty is over match RNG, not runs."""
    if not values or any(not isfinite(v) for v in values):
        raise ValueError("Score estimates require nonempty finite samples")
    n = len(values)
    mean = fsum(values) / n
    se = sqrt(fsum((v - mean) ** 2 for v in values) / (n - 1) / n) if n > 1 else None
    return ScoreEstimate(
        n=n,
        mean=mean,
        standard_error=se,
        ci_low=mean - 1.96 * se if se is not None else None,
        ci_high=mean + 1.96 * se if se is not None else None,
    )


def _scores(report: EvaluationReport, seat: int) -> tuple[float, ...]:
    if (
        type(seat) is not int
        or seat not in (0, 1)
        or report.game_id != "tic_tac_toe"
        or not report.records
    ):
        raise ValueError("Expected nonempty Tic-Tac-Toe records and a valid seat")
    if (
        report.summary.errors
        or report.summary.truncated
        or report.summary.completed != len(report.records)
    ):
        raise ValueError("Comparison requires every match completed without errors")
    scores = []
    for record in report.records:
        if (
            record.error is not None
            or record.outcome is None
            or record.result is None
            or not record.result.terminated
            or record.result.truncated
        ):
            raise ValueError("Comparison requires classified complete matches")
        scores.append(record.outcome.scores[seat])
    return tuple(scores)


def paired_score_effect(
    policy: EvaluationReport, control: EvaluationReport, seat: int
) -> ScoreEstimate:
    """Difference per matched seed/index; trajectories/actions need not coincide."""
    if (
        policy.batch_seed,
        policy.start_index,
        policy.max_steps,
        len(policy.records),
    ) != (
        control.batch_seed,
        control.start_index,
        control.max_steps,
        len(control.records),
    ):
        raise ValueError("Policy/control protocols must match")
    for first, second in zip(policy.records, control.records, strict=True):
        if (first.match_index, first.engine_seed, first.agent_seeds) != (
            second.match_index,
            second.engine_seed,
            second.agent_seeds,
        ):
            raise ValueError("Policy/control match seeds must align")
    return estimate_scores(
        tuple(
            a - b
            for a, b in zip(_scores(policy, seat), _scores(control, seat), strict=True)
        )
    )


def balanced_effect(first: ScoreEstimate, second: ScoreEstimate) -> ScoreEstimate:
    """Equal-seat mean and stratified SE: .5*sqrt(SE_X²+SE_O²)."""
    if first.n != second.n:
        raise ValueError("Seat-balanced comparison requires equal game counts")
    mean = (first.mean + second.mean) / 2
    se = (
        sqrt(first.standard_error**2 + second.standard_error**2) / 2
        if first.standard_error is not None and second.standard_error is not None
        else None
    )
    return ScoreEstimate(
        n=first.n + second.n,
        mean=mean,
        standard_error=se,
        ci_low=mean - 1.96 * se if se is not None else None,
        ci_high=mean + 1.96 * se if se is not None else None,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ControlComparison:
    opponent: str
    learner_seat: int
    control: Literal["random-v1", "heuristic-v1"]
    report: EvaluationReport
    policy_score: ScoreEstimate
    control_score: ScoreEstimate
    paired_effect: ScoreEstimate


@dataclass(frozen=True, slots=True, kw_only=True)
class Phase4Comparison:
    policy: PolicyEvaluation
    controls: tuple[ControlComparison, ...]
    random_opponent_effect: ScoreEstimate
    improvement_supported: bool
    interval_method: str = "normal-95-paired-stratified-seats"


def compare_checkpoint(
    checkpoint: str | Path,
    *,
    games_per_seat: int = 1000,
    seed: int = 20261015,
    start_index: int = 0,
) -> Phase4Comparison:
    """Evaluate sampled policy + two controls on identical case/opponent seeds.

    Frozen CPU policy, no updates. Only the learner is replaced for controls.
    A positive 95% lower bound against random controls versus random opponents,
    with >=30 matches per seat, supports the minimal checkpoint-level criterion.
    It does not establish robust tactical play or variation across training runs.
    """
    policy = evaluate_policy_checkpoint(
        checkpoint,
        games_per_seat=games_per_seat,
        seed=seed,
        start_index=start_index,
        mode="sampled",
    )
    results: list[ControlComparison] = []
    for case in policy.cases:
        for control in ("random-v1", "heuristic-v1"):
            learner: Agent[TicTacToeObservation] = (
                RandomAgent[TicTacToeObservation]()
                if control == "random-v1"
                else TicTacToeHeuristicAgent()
            )
            opponent: Agent[TicTacToeObservation] = (
                RandomAgent[TicTacToeObservation]()
                if case.opponent == "random-v1"
                else TicTacToeHeuristicAgent()
            )
            agents = (
                (learner, opponent) if case.learner_seat == 0 else (opponent, learner)
            )
            labels = (
                (control, case.opponent)
                if case.learner_seat == 0
                else (case.opponent, control)
            )
            report = evaluate_batch(
                TicTacToeEngine(),
                agents,
                outcome_for=tic_tac_toe_outcome,
                agent_labels=labels,
                num_matches=games_per_seat,
                seed=case.report.batch_seed,
                max_steps=case.report.max_steps,
                start_index=start_index,
            )
            results.append(
                ControlComparison(
                    opponent=case.opponent,
                    learner_seat=case.learner_seat,
                    control=control,
                    report=report,
                    policy_score=estimate_scores(
                        _scores(case.report, case.learner_seat)
                    ),
                    control_score=estimate_scores(_scores(report, case.learner_seat)),
                    paired_effect=paired_score_effect(
                        case.report, report, case.learner_seat
                    ),
                )
            )
    random = [
        r.paired_effect
        for r in results
        if r.control == "random-v1" and r.opponent == "random-v1"
    ]
    effect = balanced_effect(random[0], random[1])
    return Phase4Comparison(
        policy=policy,
        controls=tuple(results),
        random_opponent_effect=effect,
        improvement_supported=games_per_seat >= 30
        and effect.ci_low is not None
        and effect.ci_low > 0,
    )


def save_comparison(result: Phase4Comparison, path: str | Path) -> None:
    encoded = (
        json.dumps({"schema_version": 1, **asdict(result)}, indent=2, allow_nan=False)
        + "\n"
    )
    Path(path).write_text(encoded, encoding="utf-8")


def load_training_log(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Validate finite, ordered training diagnostics; gaps are retained, not filled."""
    rows: list[dict[str, Any]] = []
    previous = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("ppo"), dict):
            raise ValueError("Malformed training diagnostic record")
        for key in ("update", "games", "x_wins", "o_wins", "draws"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError("Training counts must be nonnegative integers")
        if (
            row["update"] <= previous
            or row["games"] <= 0
            or row["x_wins"] + row["o_wins"] + row["draws"] != row["games"]
        ):
            raise ValueError("Training updates/order/outcome counts are inconsistent")
        for key in ("num_samples", "optimizer_steps"):
            if type(row["ppo"].get(key)) is not int or row["ppo"][key] <= 0:
                raise ValueError("Training sample/step counts must be positive")
        for key in (
            "total_loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "approximate_kl",
            "clip_fraction",
        ):
            value = row["ppo"].get(key)
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError("Training diagnostics must be finite numbers")
        if not 0 <= row["ppo"]["clip_fraction"] <= 1:
            raise ValueError("Clip fraction must be in [0,1]")
        rows.append(row)
        previous = row["update"]
    if not rows:
        raise ValueError("Training log contains no completed update diagnostics")
    return tuple(rows)


def trailing_mean(
    updates: tuple[int, ...], values: tuple[float, ...], window: int = 25
) -> tuple[float, ...]:
    """Trailing observed-update average, restarting after gaps; no interpolation."""
    if len(updates) != len(values) or type(window) is not int or window <= 0:
        raise ValueError("Smoothing needs aligned series and a positive window")
    recent: list[float] = []
    result: list[float] = []
    for i, value in enumerate(values):
        if i and updates[i] != updates[i - 1] + 1:
            recent.clear()
        recent.append(value)
        recent = recent[-window:]
        result.append(fsum(recent) / len(recent))
    return tuple(result)
