"""Independent paired statistics, protocol validation, and reusable figures."""

import json
from dataclasses import replace
from math import log, sqrt
from pathlib import Path
from typing import Any

import pytest
import torch

from boardgame_ai.cli import main
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.phase4_analysis import (
    balanced_effect,
    compare_checkpoint,
    estimate_scores,
    load_training_log,
    paired_score_effect,
    save_comparison,
    trailing_mean,
)
from boardgame_ai.simulation.evaluation import (
    EvaluationReport,
    EvaluationSummary,
    GameOutcome,
    MatchRecord,
)
from boardgame_ai.simulation.runner import MatchResult
from boardgame_ai.training import PPOBatch, PPOConfig, PPOTrainer, save_ppo_checkpoint


def report(scores: tuple[float, ...]) -> EvaluationReport:
    return EvaluationReport(
        game_id="tic_tac_toe",
        agent_labels=("a", "b"),
        batch_seed=1,
        start_index=0,
        max_steps=9,
        records=tuple(
            MatchRecord(
                match_index=i,
                engine_seed=i,
                agent_seeds=(i + 10, i + 20),
                result=MatchResult(
                    num_steps=9, total_rewards=(v, -v), terminated=True, truncated=False
                ),
                outcome=GameOutcome(scores=(v, -v)),
                error=None,
            )
            for i, v in enumerate(scores)
        ),
        summary=EvaluationSummary(
            attempted=len(scores),
            completed=len(scores),
            truncated=0,
            errors=0,
            wins=(0, 0),
            draws=0,
            total_scores=(0.0, 0.0),
            mean_scores=(0.0, 0.0),
            mean_game_length=9.0,
            min_game_length=9,
            max_game_length=9,
        ),
    )


def test_mean_se_and_paired_difference_hand_calculation() -> None:
    result = estimate_scores((1.0, 0.0, -1.0))
    assert result.mean == 0.0 and result.standard_error == pytest.approx(1 / sqrt(3))
    assert result.ci_high == pytest.approx(1.96 / sqrt(3))
    effect = paired_score_effect(report((1.0, 1.0, 0.0)), report((0.0, 1.0, -1.0)), 0)
    # differences [1,0,1], mean 2/3 and SE 1/3.
    assert effect.mean == pytest.approx(2 / 3)
    assert effect.standard_error == pytest.approx(1 / 3)
    balanced = balanced_effect(effect, effect)
    assert balanced.n == 6 and balanced.mean == pytest.approx(2 / 3)
    assert balanced.standard_error == pytest.approx(1 / (3 * sqrt(2)))


def test_singleton_unknown_uncertainty_and_invalid_estimates() -> None:
    estimate = estimate_scores((1.0,))
    assert estimate.ci_low is estimate.ci_high is estimate.standard_error is None
    for values in ((), (float("nan"),), (float("inf"),)):
        with pytest.raises(ValueError):
            estimate_scores(values)
    with pytest.raises(ValueError):
        balanced_effect(estimate_scores((1.0,)), estimate_scores((1.0, 1.0)))


def test_paired_seed_mismatch_and_incomplete_records_reject() -> None:
    first = report((1.0, 0.0))
    with pytest.raises(ValueError):
        paired_score_effect(first, replace(first, batch_seed=2), 0)
    record = replace(first.records[0], agent_seeds=(100, 200))
    with pytest.raises(ValueError):
        paired_score_effect(
            first, replace(first, records=(record, first.records[1])), 0
        )
    with pytest.raises(ValueError):
        paired_score_effect(
            first, replace(first, summary=replace(first.summary, errors=1)), 0
        )
    with pytest.raises(ValueError):
        paired_score_effect(
            first,
            replace(
                first,
                records=(replace(first.records[0], outcome=None), first.records[1]),
            ),
            0,
        )


def row(update: int = 1) -> dict[str, Any]:
    return {
        "update": update,
        "games": 1,
        "x_wins": 1,
        "o_wins": 0,
        "draws": 0,
        "device": "cpu",
        "ppo": {
            "num_samples": 7,
            "optimizer_steps": 1,
            "total_loss": 0.1,
            "policy_loss": -0.1,
            "value_loss": 0.4,
            "entropy": 0.5,
            "approximate_kl": 0.01,
            "clip_fraction": 0.2,
        },
    }


def test_log_gaps_and_smoothing_do_not_interpolate(tmp_path: Path) -> None:
    path = tmp_path / "updates.jsonl"
    path.write_text(json.dumps(row(1)) + "\n" + json.dumps(row(3)) + "\n")
    records = load_training_log(path)
    assert [r["update"] for r in records] == [1, 3]
    assert trailing_mean((1, 2, 5, 6), (1.0, 3.0, 10.0, 14.0), 2) == (
        1.0,
        2.0,
        10.0,
        12.0,
    )
    with pytest.raises(ValueError):
        trailing_mean((1, 2), (1.0,), 2)


@pytest.mark.parametrize(
    "change", ["nan", "outcomes", "count", "clip", "order", "empty", "malformed"]
)
def test_bad_training_log_rejects(tmp_path: Path, change: str) -> None:
    record = row()
    if change == "nan":
        record["ppo"]["entropy"] = float("nan")
    if change == "outcomes":
        record["x_wins"] = 2
    if change == "count":
        record["ppo"]["num_samples"] = 0
    if change == "clip":
        record["ppo"]["clip_fraction"] = 2.0
    path = tmp_path / "log.jsonl"
    text = json.dumps(record) + "\n"
    if change == "order":
        text += json.dumps(record) + "\n"
    if change == "empty":
        text = ""
    if change == "malformed":
        text = "[]"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_training_log(path)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = TicTacToeActorCritic()
    trainer = PPOTrainer(model, PPOConfig(epochs=1), shuffle_seed=42)
    batch = PPOBatch(
        observations=torch.zeros(1, 6, 3, 3),
        legal_masks=torch.ones(1, 9, dtype=torch.bool),
        actions=torch.tensor([0]),
        old_log_probs=torch.tensor([-log(9)]),
        advantages=torch.tensor([1.0]),
        returns=torch.tensor([1.0]),
    )
    trainer.update(batch)
    path = tmp_path / "checkpoint.pt"
    save_ppo_checkpoint(trainer, path)
    (tmp_path / "updates.jsonl").write_text(json.dumps(row()) + "\n")
    return path


def test_controls_use_same_match_seeds_and_reproduce(
    checkpoint: Path, tmp_path: Path
) -> None:
    before = checkpoint.read_bytes()
    comparison = compare_checkpoint(checkpoint, games_per_seat=2, seed=123)
    assert comparison == compare_checkpoint(checkpoint, games_per_seat=2, seed=123)
    assert len(comparison.controls) == 8 and not comparison.improvement_supported
    for control in comparison.controls:
        policy = next(
            c
            for c in comparison.policy.cases
            if c.opponent == control.opponent and c.learner_seat == control.learner_seat
        )
        assert control.paired_effect == paired_score_effect(
            policy.report, control.report, control.learner_seat
        )
        assert control.report.agent_labels[control.learner_seat] == control.control
    assert checkpoint.read_bytes() == before
    save_comparison(comparison, tmp_path / "comparison.json")
    payload = json.loads((tmp_path / "comparison.json").read_text())
    assert payload["interval_method"] == "normal-95-paired-stratified-seats"


def test_figures_and_cli_output_protection(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")
    from boardgame_ai.phase4_plots import plot_baseline_comparison, plot_training_curves

    output = tmp_path / "figures"
    output.mkdir()
    comparison = compare_checkpoint(checkpoint, games_per_seat=2)
    plot_training_curves((row(),), output)
    plot_baseline_comparison(comparison, output)
    for filename in (
        "training-curves.png",
        "training-curves.svg",
        "baseline-comparison.png",
        "baseline-comparison.svg",
    ):
        assert (output / filename).stat().st_size > 1000
    with pytest.raises(SystemExit) as error:
        main(
            [
                "analyze",
                str(checkpoint),
                "--output",
                str(output),
                "--games-per-seat",
                "2",
            ]
        )
    assert error.value.code == 2
    main(
        [
            "analyze",
            str(checkpoint),
            "--output",
            str(tmp_path / "analysis"),
            "--games-per-seat",
            "2",
        ]
    )
    assert (tmp_path / "analysis" / "comparison.json").exists()
    assert "improvement supported" in capsys.readouterr().out


def test_plot_gap_breaks_line_and_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("matplotlib")
    from boardgame_ai.phase4_plots import _gaps

    x, y = _gaps((1, 3), (2.0, 4.0))
    assert x == [1.0, 2.0, 3.0] and y[0] == 2.0 and y[2] == 4.0
    assert y[1] != y[1]  # NaN breaks the line
    with pytest.raises(SystemExit) as error:
        main(["analyze", "--help"])
    assert error.value.code == 0 and "--training-log" in capsys.readouterr().out
