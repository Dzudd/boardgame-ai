"""Tactical edge cases, independent signal checks, and audit isolation."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.games.tic_tac_toe.state import Board
from boardgame_ai.models import TicTacToeActorCritic
from boardgame_ai.policy_audit import (
    audit_checkpoint,
    audit_decision,
    diagnose_update,
    reachable_decisions,
    summarize,
    tactical_moves,
    trace_rollout,
)
from boardgame_ai.training import PPOConfig, PPOTrainer, save_ppo_checkpoint
from boardgame_ai.training.ppo import prepare_ppo_batch
from boardgame_ai.training.rollout import collect_tic_tac_toe_rollout


def decision(
    board: tuple[int | None, ...], seat: int
) -> tuple[Observation[TicTacToeObservation], tuple[Action, ...]]:
    return (
        Observation(
            player_id=seat,
            current_player=seat,
            phase="play",
            data=TicTacToeObservation(board=cast(Board, board)),
        ),
        tuple(
            Action(kind="place", parameters=(("cell", i),))
            for i, v in enumerate(board)
            if v is None
        ),
    )


@pytest.mark.parametrize(
    ("board", "seat", "expected"),
    [
        ((0, 0, None, 1, 1, None, None, None, None), 0, ("win", (2,))),
        ((0, 0, None, 1, None, None, None, None, None), 1, ("block", (2,))),
        ((0, 0, None, None, 1, 1, None, None, 0), 1, ("win", (3,))),
        ((0, 0, None, 0, 1, None, None, 1, None), 1, ("unavoidable", ())),
        ((0, 0, None, 0, 1, 1, None, 1, None), 0, ("win", (2, 6))),
        ((None,) * 9, 0, ("quiet", ())),
    ],
)
def test_public_tactics(
    board: tuple[int | None, ...], seat: int, expected: tuple[str, tuple[int, ...]]
) -> None:
    assert tactical_moves(*decision(board, seat)) == expected


def test_supplied_choices_and_perspective() -> None:
    view, legal = decision((0, 0, None, 0, 1, 1, None, 1, None), 0)
    assert tactical_moves(view, legal[1:]) == ("win", (6,))
    swapped = replace(
        view,
        player_id=1,
        current_player=1,
        data=TicTacToeObservation(
            board=cast(
                Board, tuple(1 - v if v is not None else None for v in view.data.board)
            )
        ),
    )
    assert tactical_moves(swapped, legal) == tactical_moves(view, legal)
    with pytest.raises(ValueError):
        tactical_moves(replace(view, player_id=1), legal)
    with pytest.raises(ValueError):
        tactical_moves(view, ())
    with pytest.raises(ValueError):
        tactical_moves(view, (Action(kind="place", parameters=(("cell", 0),)),))


def test_exhaustive_census_and_engine_agreement() -> None:
    decisions = reachable_decisions()
    assert len(decisions) == 4520
    assert len({view.data.board for view, _ in decisions}) == 4520
    # Independent forward engine traversal supplies the actual reachable decisions.
    seen: dict[Board, tuple[int, tuple[Action, ...]]] = {}

    def visit(game: TicTacToeEngine) -> None:
        seat = game.current_player()
        if seat is None:
            return
        view, legal = game.decision_for(seat)
        if view.data.board in seen:
            return
        seen[view.data.board] = seat, legal
        for action in legal:
            child = game.clone()
            child.step(action)
            visit(child)

    game = TicTacToeEngine()
    game.reset()
    visit(game)
    assert {
        view.data.board: (view.player_id, legal) for view, legal in decisions
    } == seen


@pytest.fixture
def trainer() -> PPOTrainer:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = TicTacToeActorCritic()
    return PPOTrainer(model, PPOConfig(epochs=2, minibatch_size=7), shuffle_seed=8)


def test_multiple_moves_probability_mass_and_empty_denominators(
    trainer: PPOTrainer,
) -> None:
    for parameter in trainer.model.parameters():
        parameter.data.zero_()
    view, legal = decision((0, 0, None, 0, 1, 1, None, 1, None), 0)
    row = audit_decision(trainer.model, view, tuple(reversed(legal)))
    assert row["valid_probability"] == pytest.approx(2 / 3)
    assert row["greedy_valid"]  # Lowest cell tie, independent of tuple order.
    assert sum(row["probabilities"]) == pytest.approx(1)
    assert row["probabilities"][0] == 0
    summary = summarize([row, row])
    assert summary[0]["samples"] == 2 and summary[0]["distinct_boards"] == 1
    assert summary[4]["mean_valid_probability"] is None


def test_both_winners_terminal_credit_and_corruption_detection(
    trainer: PPOTrainer,
) -> None:
    winners = set()
    for seed in range(60):
        rollout = collect_tic_tac_toe_rollout(
            TicTacToeEngine(),
            (trainer.model, trainer.model),
            max_steps=9,
            sampling_seeds=(seed, seed + 100),
        )
        trace, _ = trace_rollout(rollout, trainer)
        assert trace["checks_passed"]
        if rollout.total_rewards != (0, 0):
            winners.add(rollout.total_rewards)
            for seat, segment in enumerate(rollout.players):
                assert segment.rewards[-1] == rollout.total_rewards[seat]
                assert segment.bootstrap_value == 0
            bad = deepcopy(rollout)
            bad.players[1].rewards[-1] = 0
            with pytest.raises(AssertionError):
                trace_rollout(bad, trainer)
        if len(winners) == 2:
            break
    assert winners == {(1, -1), (-1, 1)}


def test_update_matches_oracle_and_preserves_source(trainer: PPOTrainer) -> None:
    rollout = collect_tic_tac_toe_rollout(
        TicTacToeEngine(),
        (trainer.model, trainer.model),
        max_steps=9,
        sampling_seeds=(2, 3),
    )
    batch = prepare_ppo_batch(rollout.players, trainer.config)
    # Populate real Adam moments before diagnosis.
    trainer.update(batch)
    before = deepcopy(trainer)
    rng = torch.random.get_rng_state().clone()
    result = diagnose_update(trainer, batch)
    assert result["checks_passed"] and result["max_parameter_change"] > 0
    assert result["update"]["optimizer_steps"] == 2 * ((len(batch.actions) + 6) // 7)
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(tensor, before.model.state_dict()[name])
    assert torch.equal(trainer.generator.get_state(), before.generator.get_state())
    assert trainer.update_count == before.update_count
    for key, state in trainer.optimizer.state_dict()["state"].items():
        for name, tensor in state.items():
            assert torch.equal(
                tensor, before.optimizer.state_dict()["state"][key][name]
            )


def test_report_reproducibility_hashes_and_output_guards(
    tmp_path: Path, trainer: PPOTrainer
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "latest.pt"
    save_ppo_checkpoint(trainer, checkpoint)
    (source / "training.json").write_text("config", encoding="utf-8")
    (source / "updates.jsonl").write_text("log", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    rng = torch.random.get_rng_state().clone()
    result = audit_checkpoint(checkpoint, tmp_path / "a.json", games=2, update_games=1)
    again = audit_checkpoint(checkpoint, tmp_path / "b.json", games=2, update_games=1)
    assert result == again
    assert result["source_preserved"]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    for destination in (
        checkpoint,
        source / "training.json",
        source / "new.json",
        tmp_path / "a.json",
    ):
        with pytest.raises(ValueError):
            audit_checkpoint(checkpoint, destination, games=1, update_games=1)
    for kwargs in ({"games": 0}, {"update_games": 0}, {"games": True}, {"seed": True}):
        with pytest.raises(ValueError):
            audit_checkpoint(
                checkpoint, tmp_path / "invalid.json", **cast(dict[str, Any], kwargs)
            )
