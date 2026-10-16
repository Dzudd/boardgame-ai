"""Tiny public search-target runs: correctness and resume, never strength claims."""

import json
import random
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
from math import log
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from boardgame_ai.agents.policy_agent import TicTacToePolicyAgent
from boardgame_ai.models.encoders import (
    decode_tic_tac_toe_action,
    encode_tic_tac_toe_legal_actions,
    encode_tic_tac_toe_observation,
)
from boardgame_ai.search.neural import NeuralEvaluator
from boardgame_ai.search.puct import SearchConfig
from boardgame_ai.search.simulator import PublicTicTacToeSimulator
from boardgame_ai.training.checkpoint import load_ppo_checkpoint, save_ppo_checkpoint
from boardgame_ai.training.ppo import PPOTrainer
from boardgame_ai.training.search_checkpoint import (
    load_search_checkpoint,
    save_search_checkpoint,
)
from boardgame_ai.training.search_learning import (
    SearchBatch,
    SearchGame,
    SearchSample,
    SearchTrainer,
    SearchTrainingConfig,
    collect_search_game,
    initial_model,
    model_digest,
    search_target_loss,
    validate_game,
)
from boardgame_ai.training.search_run import main, run_search_training


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config() -> SearchTrainingConfig:
    return SearchTrainingConfig(
        seed=12,
        games_per_update=2,
        optimizer_steps_per_update=1,
        batch_size=8,
        replay_capacity_games=3,
        search=SearchConfig(simulations=4, exploration=True),
    )


def scripted_game(actions: tuple[int, ...], *, game_index: int = 0) -> SearchGame:
    state = PublicTicTacToeSimulator((None,) * 9, 0)
    samples = []
    rewards = []
    own_steps = [0, 0]
    for step, index in enumerate(actions):
        seat = state.actor
        assert seat is not None
        legal = state.legal_actions()
        visits = torch.zeros(9, dtype=torch.int64)
        visits[index] = 4
        samples.append(
            SearchSample(
                observation=encode_tic_tac_toe_observation(state.observe()),
                legal_mask=encode_tic_tac_toe_legal_actions(legal),
                visits=visits,
                action=index,
                seat=seat,
                step=step,
                own_step=own_steps[seat],
                outcome=0.0,
            )
        )
        state, reward = state.step(decode_tic_tac_toe_action(index, legal))
        rewards.append(reward)
        own_steps[seat] += 1
    totals = [sum(r[seat] for r in rewards) for seat in (0, 1)]
    return SearchGame(
        samples=tuple(replace(s, outcome=totals[s.seat]) for s in samples),
        rewards=tuple(rewards),
        game_index=game_index,
        policy_version=0,
        policy_sha256="a" * 64,
        search=small_config().search,
        evaluator_calls=len(samples),
    )


def assert_nested_equal(first: Any, second: Any) -> None:
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            assert_nested_equal(left, right)
    else:
        assert first == second


@pytest.mark.parametrize(
    "actions",
    [
        (0, 3, 1, 4, 2),  # X wins
        (0, 3, 1, 4, 8, 5),  # O wins
        (0, 1, 2, 4, 3, 5, 7, 6, 8),  # draw
    ],
)
def test_own_seat_outcomes_are_undiscounted_and_terminal_reward_counted_once(
    actions: tuple[int, ...],
) -> None:
    game = scripted_game(actions)
    validate_game(game)
    for sample in game.samples:
        assert sample.outcome == sum(
            r[sample.seat] for r in game.rewards[sample.step :]
        )
    assert all(reward == (0, 0) for reward in game.rewards[:-1])


def test_collection_is_reproducible_and_preserves_model_and_all_global_rng() -> None:
    trainer = SearchTrainer(small_config())
    before_model = deepcopy(trainer.model.state_dict())
    torch_state = torch.random.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    evaluator = NeuralEvaluator(trainer.model)
    games = [
        collect_search_game(
            evaluator,
            root_seed=73,
            game_index=0,
            policy_version=0,
            search=trainer.config.search,
        )
        for _ in range(2)
    ]
    from dataclasses import asdict

    assert_nested_equal(asdict(games[0]), asdict(games[1]))
    assert_nested_equal(before_model, trainer.model.state_dict())
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    assert np.array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert games[0].policy_sha256 == model_digest(trainer.model)
    for sample in games[0].samples:
        assert int(sample.visits.sum()) == 4
        assert sample.visits[sample.action] > 0
        assert (sample.visits[~sample.legal_mask] == 0).all()
        assert not sample.observation.requires_grad


@pytest.mark.parametrize(
    "damage",
    [
        "outcome",
        "mask",
        "visits",
        "action",
        "observation",
        "reward",
        "source",
        "truncated",
    ],
)
def test_replay_validation_rejects_wrong_targets_and_trajectories(damage: str) -> None:
    game = scripted_game((0, 3, 1, 4, 2))
    sample = game.samples[0]
    if damage == "outcome":
        game = replace(game, samples=(replace(sample, outcome=-1.0), *game.samples[1:]))
    elif damage == "mask":
        sample.legal_mask[0] = False
    elif damage == "visits":
        sample.visits[0] += 1
    elif damage == "action":
        game = replace(game, samples=(replace(sample, action=1), *game.samples[1:]))
    elif damage == "observation":
        sample.observation.fill_(1)
    elif damage == "reward":
        game = replace(game, rewards=((1.0, -1.0), *game.rewards[1:]))
    elif damage == "source":
        game = replace(game, source="fixed-opponent-evaluation")
    else:
        game = replace(game, samples=game.samples[:-1], rewards=game.rewards[:-1])
    with pytest.raises(ValueError):
        validate_game(game)


def test_loss_matches_hand_computation_and_masks_gradients() -> None:
    model = initial_model(10)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    game = scripted_game((0, 3, 1, 4, 2))
    samples = (game.samples[0], game.samples[-1])
    batch = SearchBatch(
        observations=torch.stack([s.observation for s in samples]),
        legal_masks=torch.stack([s.legal_mask for s in samples]),
        targets=torch.stack([s.visits.float() / 4 for s in samples]),
        outcomes=torch.tensor([s.outcome for s in samples]),
    )
    policy, value = search_target_loss(model, batch)
    assert float(policy.detach()) == pytest.approx((log(9) + log(5)) / 2)
    assert float(value.detach()) == pytest.approx(0.5)
    # Isolate a late board so unavailable policy logits must have zero gradient.
    late = SearchBatch(
        observations=batch.observations[1:],
        legal_masks=batch.legal_masks[1:],
        targets=batch.targets[1:],
        outcomes=batch.outcomes[1:],
    )
    p, v = search_target_loss(model, late)
    (p + v).backward()  # type: ignore[no-untyped-call]
    gradient = model.policy_head.bias.grad
    assert gradient is not None
    assert (gradient[~late.legal_masks[0]] == 0).all()
    assert gradient[2] < 0
    assert model.value_head.bias.grad is not None and model.value_head.bias.grad[0] < 0


@pytest.mark.parametrize(
    "damage", ["illegal_mass", "normalization", "nonfinite", "attached", "outcome"]
)
def test_loss_rejects_invalid_targets_before_optimization(damage: str) -> None:
    model = initial_model(10)
    sample = scripted_game((0, 3, 1, 4, 2)).samples[-1]
    batch = SearchBatch(
        observations=sample.observation.unsqueeze(0),
        legal_masks=sample.legal_mask.unsqueeze(0),
        targets=sample.visits.float().unsqueeze(0) / 4,
        outcomes=torch.tensor([1.0]),
    )
    if damage == "illegal_mass":
        batch.targets[0, 0] = 1
    elif damage == "normalization":
        batch.targets.zero_()
    elif damage == "nonfinite":
        batch.targets[0, 2] = float("nan")
    elif damage == "attached":
        batch.targets.requires_grad_(True)
    else:
        batch.outcomes[0] = 2
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        search_target_loss(model, batch)
    assert_nested_equal(before, model.state_dict())


def test_replay_is_uniform_over_games_then_positions_not_flat_decisions() -> None:
    trainer = SearchTrainer(replace(small_config(), batch_size=6000))
    trainer.replay = [
        scripted_game((0, 3, 1, 4, 2)),
        scripted_game((0, 1, 2, 4, 3, 5, 7, 6, 8), game_index=1),
    ]
    before = trainer.generator.get_state().clone()
    batch, choices = trainer.sample_batch()
    counts = [sum(slot == i for slot, _ in choices) for i in (0, 1)]
    # A fixed private stream gives a deterministic frequency check; flat-position
    # sampling would assign the short game about 5/14, rather than half the mass.
    assert all(2700 < count < 3300 for count in counts)
    for slot, game in enumerate(trainer.replay):
        frequencies = [
            sum(pair == (slot, position) for pair in choices)
            for position in range(len(game.samples))
        ]
        assert all(
            0.7 * counts[slot] / len(game.samples)
            < count
            < 1.3 * counts[slot] / len(game.samples)
            for count in frequencies
        )
    trainer.generator.set_state(before)
    repeated, repeated_choices = trainer.sample_batch()
    assert repeated_choices == choices and torch.equal(batch.targets, repeated.targets)
    assert (
        batch.observations.data_ptr()
        != trainer.replay[0].samples[0].observation.data_ptr()
    )


def test_fifo_retention_and_latest_frozen_policy_versions() -> None:
    trainer = SearchTrainer(small_config())
    first_digest = model_digest(trainer.model)
    first = trainer.update()
    assert all(game.policy_sha256 == first_digest for game in trainer.replay)
    next_digest = model_digest(trainer.model)
    assert next_digest != first_digest
    second = trainer.update()
    assert [game.game_index for game in trainer.replay] == [1, 2, 3]
    assert [game.policy_version for game in trainer.replay] == [0, 1, 1]
    assert all(game.policy_sha256 == next_digest for game in trainer.replay[1:])
    assert first["replay_age_max"] == 0 and second["replay_age_max"] == 1
    assert second["sampled_positions"] == trainer.config.batch_size


def test_uninterrupted_and_resumed_runs_match_every_saved_state_and_report(
    tmp_path: Path,
) -> None:
    config = small_config()
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    run_search_training(full, config, updates=3)
    run_search_training(resumed, config, updates=1)
    run_search_training(resumed, config, updates=3, resume=True)
    left = torch.load(full / "latest.pt", weights_only=True)
    right = torch.load(resumed / "latest.pt", weights_only=True)
    assert_nested_equal(left, right)
    assert {p.name: p.read_bytes() for p in full.glob("*.json")} == {
        p.name: p.read_bytes() for p in resumed.glob("*.json")
    }
    trainer, _ = load_search_checkpoint(resumed / "latest.pt")
    rng = torch.random.get_rng_state().clone()
    sample = trainer.replay[-1].samples[0]
    # The trained weights support the observation-only policy adapter directly;
    # generic PPO checkpoint evaluation deliberately cannot load this format.
    state = PublicTicTacToeSimulator((None,) * 9, 0)
    assert (
        TicTacToePolicyAgent(trainer.model).select_action(
            state.observe(), state.legal_actions()
        )
        in state.legal_actions()
    )
    assert not sample.observation.requires_grad and torch.equal(
        rng, torch.random.get_rng_state()
    )


def test_resume_after_report_publication_failure_does_not_repeat_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from boardgame_ai.training import search_run

    config = small_config()
    with monkeypatch.context() as patch:
        patch.setattr(
            search_run,
            "_publish_report",
            lambda *args: (_ for _ in ()).throw(OSError("injected")),
        )
        with pytest.raises(OSError, match="injected"):
            run_search_training(tmp_path, config, updates=1)
    before = (tmp_path / "latest.pt").read_bytes()
    assert not list(tmp_path.glob("update-*.json"))
    trainer = run_search_training(tmp_path, config, updates=1, resume=True)
    assert trainer.update_count == 1 and (tmp_path / "latest.pt").read_bytes() == before
    assert json.loads((tmp_path / "update-00000001.json").read_text())["update"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 13),
        ("batch_size", 7),
        ("replay_capacity_games", 4),
        ("optimizer_steps_per_update", 2),
        ("games_per_update", 1),
    ],
)
def test_resume_rejects_schedule_changes_without_writes(
    tmp_path: Path, field: str, value: int
) -> None:
    config = small_config()
    run_search_training(tmp_path, config, updates=1)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    changes: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match="must match"):
        run_search_training(
            tmp_path, replace(config, **changes), updates=2, resume=True
        )
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    "damage",
    [
        "identity",
        "counter",
        "weights",
        "adam",
        "rng",
        "replay",
        "report",
        "report_nan",
        "config",
        "replay_fields",
    ],
)
def test_checkpoint_corruption_rejects_without_changing_rng(
    tmp_path: Path, damage: str
) -> None:
    run_search_training(tmp_path, small_config(), updates=1)
    path = tmp_path / "latest.pt"
    payload = torch.load(path, weights_only=True)
    if damage == "identity":
        payload["learner_id"] = "ppo"
    elif damage == "counter":
        payload["games_seen"] += 1
    elif damage == "weights":
        next(iter(payload["model_state"].values())).fill_(float("nan"))
    elif damage == "adam":
        payload["optimizer_state"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif damage == "rng":
        payload["replay_rng_state"] = torch.zeros(1, dtype=torch.uint8)
    elif damage == "replay":
        payload["replay"][0]["samples"][0]["outcome"] = 2.0
    elif damage == "report":
        payload["last_report"]["update"] += 1
    elif damage == "report_nan":
        payload["last_report"]["policy_loss"] = float("nan")
    elif damage == "replay_fields":
        payload["replay"][0]["samples"][0]["teacher_label"] = 1
    else:
        payload["config"]["seed"] = False
    torch.save(payload, path)
    before = path.read_bytes()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        load_search_checkpoint(path)
    assert path.read_bytes() == before and torch.equal(
        rng, torch.random.get_rng_state()
    )


def test_ppo_and_search_checkpoint_formats_are_incompatible(tmp_path: Path) -> None:
    trainer = SearchTrainer(small_config())
    search = tmp_path / "search.pt"
    ppo = tmp_path / "ppo.pt"
    save_search_checkpoint(trainer, search)
    save_ppo_checkpoint(PPOTrainer(initial_model(4)), ppo)
    with pytest.raises(ValueError):
        load_ppo_checkpoint(search)
    with pytest.raises(ValueError):
        load_search_checkpoint(ppo)


def test_failed_optimizer_update_keeps_saved_boundary_and_resume_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(small_config(), optimizer_steps_per_update=2)
    interrupted = tmp_path / "interrupted"
    complete = tmp_path / "complete"
    run_search_training(complete, config, updates=2)
    run_search_training(interrupted, config, updates=1)
    before = (interrupted / "latest.pt").read_bytes()
    original_step = torch.optim.Adam.step
    calls = 0

    def interrupted_step(optimizer: torch.optim.Adam, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("injected partial update")
        return original_step(optimizer, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", interrupted_step)
        with pytest.raises(InterruptedError):
            run_search_training(interrupted, config, updates=2, resume=True)
    assert (interrupted / "latest.pt").read_bytes() == before
    run_search_training(interrupted, config, updates=2, resume=True)
    assert_nested_equal(
        torch.load(interrupted / "latest.pt", weights_only=True),
        torch.load(complete / "latest.pt", weights_only=True),
    )


def test_failed_checkpoint_replace_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    trainer = SearchTrainer(small_config())
    path = tmp_path / "latest.pt"
    save_search_checkpoint(trainer, path)
    before = path.read_bytes()
    report = trainer.update()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected replace failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail)
        with pytest.raises(OSError, match="injected"):
            save_search_checkpoint(trainer, path, last_report=report)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_existing_boundary_report_cannot_be_replaced(tmp_path: Path) -> None:
    config = small_config()
    run_search_training(tmp_path, config, updates=1)
    report = tmp_path / "update-00000001.json"
    report.write_bytes(b"unrelated existing evidence")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="mismatch"):
        run_search_training(tmp_path, config, updates=2, resume=True)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_invalid_budget_and_smaller_resume_are_readonly(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_search_training(tmp_path / "absent", small_config(), updates=-1)
    assert not (tmp_path / "absent").exists()
    run_search_training(tmp_path, small_config(), updates=1)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="precede"):
        run_search_training(tmp_path, small_config(), updates=0, resume=True)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_cli_tiny_explicit_budget_and_zero_budget_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "search_run",
            str(tmp_path),
            "--updates",
            "1",
            "--games-per-update",
            "1",
            "--simulations",
            "2",
            "--batch-size",
            "2",
            "--optimizer-steps",
            "1",
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out)["games_seen"] == 1
    trainer, _ = load_search_checkpoint(tmp_path / "latest.pt")
    assert trainer.update_count == 1
    empty = tmp_path / "empty"
    run_search_training(empty, small_config(), updates=0)
    before = {p.name: p.read_bytes() for p in empty.iterdir()}
    run_search_training(empty, small_config(), updates=0, resume=True)
    assert {p.name: p.read_bytes() for p in empty.iterdir()} == before
    with pytest.raises(ValueError):
        run_search_training(tmp_path, small_config(), updates=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": True},
        {"batch_size": 0},
        {"learning_rate": float("nan")},
        {"max_grad_norm": -1},
        {"search": SearchConfig(exploration=False)},
        {"games_per_update": True},
        {"unroll_steps": 1},
    ],
)
def test_invalid_training_configuration(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(small_config(), **changes)
