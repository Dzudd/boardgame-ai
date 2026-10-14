"""Frozen policy boundaries, fixed cases, held-out schedules, and CPU reports."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import torch

from boardgame_ai.agents.policy_agent import TicTacToePolicyAgent
from boardgame_ai.cli import main
from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.policy_evaluation import (
    evaluate_policy_checkpoint,
    save_policy_evaluation,
)
from boardgame_ai.training import PPOTrainer, save_ppo_checkpoint
from boardgame_ai.training.run import _seed as training_seed


@pytest.fixture
def model() -> TicTacToeActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        result = TicTacToeActorCritic()
    with torch.no_grad():
        for p in result.parameters():
            p.zero_()
    return result


@pytest.fixture
def checkpoint(model: TicTacToeActorCritic, tmp_path: Path) -> Path:
    path = tmp_path / "policy.pt"
    save_ppo_checkpoint(PPOTrainer(model, shuffle_seed=1), path)
    return path


def actor_view() -> Observation[TicTacToeObservation]:
    return Observation(
        player_id=0,
        current_player=0,
        phase="play",
        data=TicTacToeObservation(board=(None,) * 9),
    )


def choices() -> tuple[Action, ...]:
    return tuple(Action(kind="place", parameters=(("cell", cell),)) for cell in (6, 3))


def test_greedy_masks_unavailable_logits_ties_and_returns_supplied_action(
    model: TicTacToeActorCritic,
) -> None:
    with torch.no_grad():
        model.policy_head.bias[8] = 1000  # unavailable despite much larger score
    agent = TicTacToePolicyAgent(model, mode="greedy")
    legal = choices()
    assert (
        agent.select_action(actor_view(), legal) is legal[1]
    )  # lowest cell, not tuple order
    assert agent.select_action(actor_view(), (legal[0],)) is legal[0]


def test_policy_snapshot_source_weights_mode_gradients_and_global_rng(
    model: TicTacToeActorCritic,
) -> None:
    source_grad = torch.ones_like(model.policy_head.weight)
    model.policy_head.weight.grad = source_grad.clone()
    agent = TicTacToePolicyAgent(model, mode="greedy")
    with torch.no_grad():
        model.policy_head.bias[6] = (
            1000  # mutation after snapshot must not change agent
        )
    rng = torch.random.get_rng_state().clone()
    assert agent.select_action(actor_view(), choices()).parameters == (("cell", 3),)
    assert model.training and model.policy_head.weight.requires_grad
    assert torch.equal(model.policy_head.weight.grad, source_grad)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_sampled_policy_reset_and_invalid_inputs_preserve_rng(
    model: TicTacToeActorCritic,
) -> None:
    agent = TicTacToePolicyAgent(model)
    legal = choices()
    agent.reset(123)
    sequence = [agent.select_action(actor_view(), legal) for _ in range(20)]
    agent.reset(123)
    assert [agent.select_action(actor_view(), legal) for _ in range(20)] == sequence
    rng = agent._generator.get_state().clone()
    with pytest.raises(ValueError):
        agent.select_action(actor_view(), ())
    with pytest.raises(ValueError):
        agent.select_action(replace(actor_view(), current_player=1), legal)
    with pytest.raises(ValueError):
        agent.reset(True)
    assert torch.equal(agent._generator.get_state(), rng)


@pytest.mark.parametrize("mode", ["sampled", "greedy"])
def test_frozen_policy_completes_both_seat_games(
    model: TicTacToeActorCritic, mode: Any
) -> None:
    from boardgame_ai.agents import TicTacToeHeuristicAgent
    from boardgame_ai.simulation import run_match

    for seat in (0, 1):
        learner = TicTacToePolicyAgent(model, mode=mode)
        other = TicTacToeHeuristicAgent()
        agents = (learner, other) if seat == 0 else (other, learner)
        result = run_match(
            TicTacToeEngine(),
            agents,
            max_steps=9,
            engine_seed=123,
            agent_seeds=(10, 20),
        )
        assert result.terminated and not result.truncated


def test_invalid_policy_mode_or_model_dtype(model: TicTacToeActorCritic) -> None:
    with pytest.raises(ValueError):
        TicTacToePolicyAgent(model, mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TicTacToePolicyAgent(model.double())


def test_fixed_suite_balances_seats_and_uses_disjoint_training_seed_namespace(
    checkpoint: Path,
) -> None:
    rng = torch.random.get_rng_state().clone()
    original = checkpoint.read_bytes()
    result = evaluate_policy_checkpoint(checkpoint, games_per_seat=3, seed=42)
    assert result.checkpoint_sha256 == sha256(original).hexdigest()
    assert result.checkpoint_updates == result.checkpoint_optimizer_steps == 0
    assert result.held_out_seed == 42 and result.policy_mode == "sampled"
    assert len(result.cases) == 4
    assert {(c.opponent, c.learner_seat) for c in result.cases} == {
        ("random-v1", 0),
        ("random-v1", 1),
        ("heuristic-v1", 0),
        ("heuristic-v1", 1),
    }
    all_eval_seeds: set[int] = set()
    for case in result.cases:
        summary = case.report.summary
        assert summary.attempted == summary.completed == 3
        assert summary.errors == summary.truncated == 0
        assert sum(summary.wins) + summary.draws == 3
        assert case.report.agent_labels[case.learner_seat] == "checkpoint-policy"
        assert 5 <= summary.min_game_length <= summary.max_game_length <= 9  # type: ignore[operator]
        for record in case.report.records:
            all_eval_seeds.add(record.engine_seed)
            all_eval_seeds.update(record.agent_seeds)
    train_seeds = {
        training_seed(42, i, stream)
        for i in range(3)
        for stream in ("engine", "seat:0", "seat:1")
    }
    assert all_eval_seeds.isdisjoint(train_seeds)
    assert checkpoint.read_bytes() == original
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_suite_repeats_and_shards_keep_absolute_seeds(checkpoint: Path) -> None:
    first = evaluate_policy_checkpoint(checkpoint, games_per_seat=3, seed=123)
    second = evaluate_policy_checkpoint(checkpoint, games_per_seat=3, seed=123)
    assert first == second
    shard = evaluate_policy_checkpoint(
        checkpoint, games_per_seat=1, seed=123, start_index=1
    )
    for full, part in zip(first.cases, shard.cases, strict=True):
        assert full.report.records[1:2] == part.report.records


def test_one_checkpoint_snapshot_even_if_source_changes(checkpoint: Path) -> None:
    expected = evaluate_policy_checkpoint(checkpoint, games_per_seat=2, seed=123)
    original = checkpoint.read_bytes()

    def change_source(case: Any) -> None:
        checkpoint.write_bytes(b"replaced source")  # only this test-owned file

    result = evaluate_policy_checkpoint(
        checkpoint, games_per_seat=2, seed=123, on_case=change_source
    )
    assert (
        result == expected and result.checkpoint_sha256 == sha256(original).hexdigest()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"games_per_seat": 0},
        {"games_per_seat": True},
        {"seed": True},
        {"start_index": -1},
        {"mode": "unknown"},
    ],
)
def test_bad_configuration_rejects_before_loading(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        evaluate_policy_checkpoint(tmp_path / "missing.pt", **kwargs)


def test_error_reports_do_not_export_exception_text(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Action:
        raise RuntimeError("PRIVATE SECRET")

    monkeypatch.setattr(TicTacToePolicyAgent, "select_action", fail)
    result = evaluate_policy_checkpoint(checkpoint, games_per_seat=1)
    assert all(
        c.report.summary.errors == 1 and c.report.summary.completed == 0
        for c in result.cases
    )
    destination = tmp_path / "evaluation.json"
    save_policy_evaluation(result, destination)
    text = destination.read_text()
    assert "PRIVATE SECRET" not in text and "match_failed" in text
    payload = json.loads(text)
    assert payload["schema_version"] == 1
    assert all(
        r["error"] == "match_failed"
        for c in payload["cases"]
        for r in c["report"]["records"]
    )
    assert "observations" not in text and "model_state" not in text


def test_cli_cpu_evaluation_report_and_source_preservation(
    checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    original = checkpoint.read_bytes()
    output = tmp_path / "evaluation.json"
    main(
        [
            "evaluate",
            str(checkpoint),
            "--output",
            str(output),
            "--games-per-seat",
            "1",
            "--mode",
            "greedy",
        ]
    )
    payload = json.loads(output.read_text())
    assert payload["policy_mode"] == "greedy" and len(payload["cases"]) == 4
    assert checkpoint.read_bytes() == original
    assert "W/D/L=" in capsys.readouterr().out


@pytest.mark.parametrize(
    "name", ["latest.pt", "training.json", "updates.jsonl", "policy.pt"]
)
def test_cli_protects_training_artifacts(
    checkpoint: Path, tmp_path: Path, name: str
) -> None:
    output = tmp_path / name
    original = checkpoint.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                str(checkpoint),
                "--output",
                str(output),
                "--games-per-seat",
                "1",
            ]
        )
    assert error.value.code == 2 and checkpoint.read_bytes() == original
    if output != checkpoint:
        assert not output.exists()


def test_cli_help_and_error_status_saved(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--help"])
    assert error.value.code == 0 and "--games-per-seat" in capsys.readouterr().out

    def fail(*args: Any, **kwargs: Any) -> Action:
        raise RuntimeError("PRIVATE SECRET")

    monkeypatch.setattr(TicTacToePolicyAgent, "select_action", fail)
    output = tmp_path / "failed.json"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                str(checkpoint),
                "--output",
                str(output),
                "--games-per-seat",
                "1",
            ]
        )
    assert error.value.code == 2
    assert output.exists() and "PRIVATE SECRET" not in output.read_text()
