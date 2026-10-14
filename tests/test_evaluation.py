"""Outcome semantics, correct denominators, errors, and safe summary exports."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from boardgame_ai.core import Action, GameEngine, Observation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.games.tic_tac_toe.evaluation import tic_tac_toe_outcome
from boardgame_ai.simulation import (
    GameOutcome,
    evaluate_batch,
    run_batch,
    save_evaluation,
)
from tests.test_interfaces import DummyGame
from tests.test_simulation import IllegalAgent, RecordingAgent, RewardGame


def dummy_outcome(game: GameEngine[int]) -> GameOutcome:
    assert game.is_terminal()
    # Intentionally unrelated to the fixture's incremental rewards.
    return GameOutcome(scores=(10.0, 20.0), winners=(0, 1))


def test_general_scores_multiple_winners_and_reward_totals_are_separate() -> None:
    game = RewardGame()
    agents = (RecordingAgent[int](), RecordingAgent[int]())
    report = evaluate_batch(
        game,
        agents,
        outcome_for=dummy_outcome,
        agent_labels=("a", "b"),
        num_matches=3,
        seed=42,
        max_steps=3,
    )
    summary = report.summary
    assert summary.attempted == summary.completed == 3
    assert summary.errors == summary.truncated == summary.draws == 0
    assert summary.wins == (3, 3)
    assert summary.total_scores == (30.0, 60.0)
    assert summary.mean_scores == (10.0, 20.0)
    assert (
        summary.mean_game_length
        == summary.min_game_length
        == summary.max_game_length
        == 3
    )
    assert all(
        r.result is not None and r.result.total_rewards == (1.5, 6.0)
        for r in report.records
    )
    assert report == evaluate_batch(
        game,
        agents,
        outcome_for=dummy_outcome,
        agent_labels=("a", "b"),
        num_matches=3,
        seed=42,
        max_steps=3,
    )
    batch = run_batch(game, agents, num_matches=3, seed=42, max_steps=3)
    assert [(r.engine_seed, r.agent_seeds, r.result) for r in report.records] == [
        (r.engine_seed, r.agent_seeds, r.result) for r in batch
    ]


@pytest.mark.parametrize("count, limit", [(0, 3), (3, 0), (3, 2)])
def test_empty_and_truncated_reports_do_not_invent_outcomes(
    count: int, limit: int
) -> None:
    def forbidden(game: GameEngine[int]) -> GameOutcome:
        raise AssertionError("Must not classify an unfinished game")

    agents = (RecordingAgent[int](), RecordingAgent[int]())
    report = evaluate_batch(
        DummyGame(),
        agents,
        outcome_for=forbidden,
        agent_labels=("a", "b"),
        num_matches=count,
        seed=42,
        max_steps=limit,
    )
    assert (
        report.summary.completed == report.summary.errors == report.summary.draws == 0
    )
    assert report.summary.truncated == count
    assert report.summary.wins == (0, 0)
    assert report.summary.mean_scores is None
    assert report.summary.mean_game_length is None
    assert report.summary.min_game_length is report.summary.max_game_length is None
    assert all(r.outcome is None and r.error is None for r in report.records)
    if count == 0:
        assert not any(agent.seeds for agent in agents)


class SometimesFailingAgent(RecordingAgent[int]):
    def select_action(
        self, observation: Observation[int], legal_actions: tuple[Action, ...]
    ) -> Action:
        if len(self.seeds) == 2 and observation.data == 1:
            raise RuntimeError("PRIVATE_HAND=999")
        return super().select_action(observation, legal_actions)


def test_error_after_partial_play_is_unknown_length_and_next_match_resets(
    tmp_path: Path,
) -> None:
    agents = (SometimesFailingAgent(), RecordingAgent[int]())
    report = evaluate_batch(
        DummyGame(),
        agents,
        outcome_for=dummy_outcome,
        agent_labels=("a", "b"),
        num_matches=3,
        seed=42,
        max_steps=3,
    )
    assert report.summary.attempted == 3
    assert report.summary.completed == 2
    assert report.summary.errors == 1
    assert report.summary.wins == (2, 2)
    assert report.summary.mean_scores == (10.0, 20.0)
    assert report.records[1].error == "match_failed"
    assert report.records[1].result is None
    assert report.records[1].outcome is None
    assert report.records[2].outcome == dummy_outcome(DummyTerminal())
    path = tmp_path / "metrics.json"
    save_evaluation(report, path)
    text = path.read_text(encoding="utf-8")
    assert "PRIVATE_HAND" not in text and "RuntimeError" not in text
    data = json.loads(text)
    assert data["schema_version"] == 1
    assert data["summary"]["errors"] == 1
    assert "observations" not in text and "events" not in text


class DummyTerminal(DummyGame):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        for _ in range(3):
            self.step(Action(kind="advance"))


def test_illegal_actions_have_fixed_category_and_are_not_draws() -> None:
    report = evaluate_batch(
        DummyGame(),
        (IllegalAgent(), RecordingAgent[int]()),
        outcome_for=dummy_outcome,
        agent_labels=("bad", "a"),
        num_matches=3,
        seed=42,
        max_steps=3,
    )
    assert report.summary.errors == 3
    assert report.summary.completed == report.summary.draws == 0
    assert all(r.error == "illegal_action" for r in report.records)


@pytest.mark.parametrize("bad_scores", [True, False])
def test_adapter_failures_retain_length_but_exclude_outcome_metrics(
    bad_scores: bool,
) -> None:
    def broken(game: GameEngine[int]) -> GameOutcome:
        if bad_scores:
            return GameOutcome(scores=(1.0,))
        raise ValueError("secret")

    report = evaluate_batch(
        DummyGame(),
        (RecordingAgent[int](), RecordingAgent[int]()),
        outcome_for=broken,
        agent_labels=("a", "b"),
        num_matches=2,
        seed=42,
        max_steps=3,
    )
    assert report.summary.errors == 2 and report.summary.completed == 0
    assert all(
        r.error == "outcome_failed"
        and r.outcome is None
        and r.result is not None
        and r.result.num_steps == 3
        for r in report.records
    )


def test_interrupts_propagate() -> None:
    def interrupt(game: GameEngine[int]) -> GameOutcome:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluate_batch(
            DummyGame(),
            (RecordingAgent[int](), RecordingAgent[int]()),
            outcome_for=interrupt,
            agent_labels=("a", "b"),
            num_matches=3,
            seed=42,
            max_steps=3,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_matches": -1},
        {"seed": True},
        {"start_index": -1},
        {"agent_labels": ("a",)},
        {"agent_labels": ("", "b")},
    ],
)
def test_invalid_configuration_rejects_before_reset(kwargs: dict[str, Any]) -> None:
    agents = (RecordingAgent[int](), RecordingAgent[int]())
    config: dict[str, Any] = dict(
        num_matches=3, seed=42, max_steps=3, agent_labels=("a", "b")
    )
    config.update(kwargs)
    with pytest.raises(ValueError):
        evaluate_batch(DummyGame(), agents, outcome_for=dummy_outcome, **config)
    assert not any(agent.seeds for agent in agents)


@pytest.mark.parametrize(
    "scores, winners, draw",
    [
        ((), (), False),
        ((float("nan"),), (), False),
        ((1.0, 2.0), (2,), False),
        ((1.0, 2.0), (True,), False),
        ((1.0, 2.0), (0, 0), False),
        ((1.0, 2.0), (0,), True),
    ],
)
def test_invalid_outcomes(
    scores: tuple[float, ...], winners: tuple[int, ...], draw: bool
) -> None:
    with pytest.raises(ValueError):
        GameOutcome(scores=scores, winners=winners, draw=draw)


@pytest.mark.parametrize(
    "moves, winners, scores, draw",
    [
        ((0, 3, 1, 4, 2), (0,), (1.0, -1.0), False),
        ((0, 3, 1, 4, 8, 5), (1,), (-1.0, 1.0), False),
        ((0, 1, 2, 4, 3, 5, 7, 6, 8), (), (0.0, 0.0), True),
    ],
)
def test_tic_tac_toe_public_outcomes(
    moves: tuple[int, ...],
    winners: tuple[int, ...],
    scores: tuple[float, ...],
    draw: bool,
) -> None:
    game = TicTacToeEngine()
    game.reset(42)
    for cell in moves:
        game.step(Action(kind="place", parameters=(("cell", cell),)))
    before = game.observe(0)
    assert tic_tac_toe_outcome(game) == GameOutcome(
        scores=scores, winners=winners, draw=draw
    )
    assert game.observe(0) == before


def test_tic_tac_toe_adapter_rejects_live_position() -> None:
    game = TicTacToeEngine()
    game.reset()
    with pytest.raises(ValueError, match="terminal"):
        tic_tac_toe_outcome(game)


def test_terminal_draws_and_cooperative_outcomes() -> None:
    from boardgame_ai.agents import TicTacToeHeuristicAgent

    report = evaluate_batch(
        TicTacToeEngine(),
        (TicTacToeHeuristicAgent(), TicTacToeHeuristicAgent()),
        outcome_for=tic_tac_toe_outcome,
        agent_labels=("h", "h"),
        num_matches=3,
        seed=42,
        max_steps=9,
    )
    assert report.summary.draws == report.summary.completed == 3
    assert report.summary.wins == (0, 0)
    assert report.summary.mean_scores == (0.0, 0.0)
    assert report.summary.mean_game_length == 9
    outcome = GameOutcome(scores=(5.0, 5.0))
    assert outcome.winners == () and not outcome.draw
    assert asdict(report)["summary"]["errors"] == 0
