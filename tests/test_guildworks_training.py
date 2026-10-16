"""Recurrent checkpoint, complete self-play schedule and evaluation isolation."""

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from boardgame_ai.games.guildworks import GuildworksEngine
from boardgame_ai.training.guildworks import RecurrentPPOTrainer, collect_guildworks
from boardgame_ai.training.guildworks_run import (
    RecurrentRunConfig,
    derive_seed,
    evaluate_model,
    initial_model,
    load_checkpoint,
    main,
    run_training,
    save_checkpoint,
)
from boardgame_ai.training.ppo import PPOConfig


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def trained() -> RecurrentPPOTrainer:
    trainer = RecurrentPPOTrainer(initial_model(1), shuffle_seed=44)
    pair = collect_guildworks(
        GuildworksEngine(), trainer.model, engine_seed=3, sampling_seeds=(4, 5)
    )
    trainer.update(pair)
    return trainer


def test_checkpoint_exact_next_update_and_no_episode_data(tmp_path: Path) -> None:
    trainer = trained()
    path = tmp_path / "latest.pt"
    rng = torch.get_rng_state().clone()
    save_checkpoint(trainer, path)
    loaded = load_checkpoint(path)
    assert torch.equal(rng, torch.get_rng_state())
    payload = torch.load(path, weights_only=True)
    assert payload["game_id"] == "guildworks"
    assert set(payload) == {
        "schema_version",
        "game_id",
        "model_id",
        "encoding_id",
        "config",
        "model_state",
        "optimizer_state",
        "shuffle_rng_state",
        "update_count",
        "optimizer_steps",
        "model_training",
    }
    assert all(v.device.type == "cpu" for v in payload["model_state"].values())
    pair = collect_guildworks(
        GuildworksEngine(), trainer.model, engine_seed=13, sampling_seeds=(14, 15)
    )
    assert trainer.update(pair) == loaded.update(pair)
    assert_nested_equal(trainer.model.state_dict(), loaded.model.state_dict())
    assert_nested_equal(trainer.optimizer.state_dict(), loaded.optimizer.state_dict())
    assert torch.equal(trainer.generator.get_state(), loaded.generator.get_state())


@pytest.mark.parametrize(
    "damage",
    [
        "game",
        "encoding",
        "model",
        "nan",
        "shape",
        "counter",
        "rng",
        "adam",
        "config",
        "extra",
    ],
)
def test_corrupt_checkpoint_rejects(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "broken.pt"
    save_checkpoint(trained(), path)
    payload = torch.load(path, weights_only=True)
    if damage == "game":
        payload["game_id"] = "tic_tac_toe"
    elif damage == "encoding":
        payload["encoding_id"] = "unfiltered"
    elif damage == "model":
        payload["model_id"] = "other"
    elif damage == "nan":
        payload["model_state"]["policy_head.weight"][0, 0] = float("nan")
    elif damage == "shape":
        payload["model_state"]["policy_head.weight"] = torch.zeros(1)
    elif damage == "counter":
        payload["update_count"] = True
    elif damage == "rng":
        payload["shuffle_rng_state"] = torch.zeros(2)
    elif damage == "adam":
        payload["optimizer_state"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif damage == "config":
        payload["config"]["extra"] = 1
    else:
        payload["history"] = "secret"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_invalid_save_preserves_file_and_zero_update_checkpoint(tmp_path: Path) -> None:
    trainer = RecurrentPPOTrainer(initial_model(2))
    path = tmp_path / "latest.pt"
    save_checkpoint(trainer, path)
    assert load_checkpoint(path).update_count == 0
    before = path.read_bytes()
    with torch.no_grad():
        next(trainer.model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError):
        save_checkpoint(trainer, path)
    assert path.read_bytes() == before


def test_resume_matches_uninterrupted_and_private_rng(tmp_path: Path) -> None:
    config = RecurrentRunConfig(
        updates=2,
        games_per_update=2,
        seed=18,
        ppo=PPOConfig(gamma=1, epochs=2, minibatch_size=3),
    )
    full = run_training(tmp_path / "full", config)
    directory = tmp_path / "split"
    run_training(directory, replace(config, updates=1))
    split = run_training(directory, config, resume=True)
    assert full.update_count == split.update_count == 2
    assert full.optimizer_steps == split.optimizer_steps == 8
    assert_nested_equal(full.model.state_dict(), split.model.state_dict())
    assert_nested_equal(full.optimizer.state_dict(), split.optimizer.state_dict())
    assert torch.equal(full.generator.get_state(), split.generator.get_state())
    assert (directory / "updates.jsonl").read_bytes() == (
        tmp_path / "full" / "updates.jsonl"
    ).read_bytes()
    rows = [
        json.loads(line)
        for line in (directory / "updates.jsonl").read_text().splitlines()
    ]
    assert all(r["ppo"]["num_samples"] == 64 for r in rows)
    before = (directory / "latest.pt").read_bytes()
    run_training(directory, config, resume=True)
    assert (directory / "latest.pt").read_bytes() == before


@pytest.mark.parametrize(
    "change", [{"seed": 9}, {"games_per_update": 3}, {"updates": 0}]
)
def test_resume_schedule_mismatch_preserves_checkpoint(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    config = RecurrentRunConfig(updates=1, games_per_update=1)
    run_training(tmp_path, config)
    before = (tmp_path / "latest.pt").read_bytes()
    with pytest.raises(ValueError):
        run_training(tmp_path, replace(config, **change), resume=True)
    assert (tmp_path / "latest.pt").read_bytes() == before
    with pytest.raises(ValueError):
        run_training(tmp_path, config)


def test_evaluation_is_frozen_and_does_not_supply_training_samples() -> None:
    trainer = trained()
    model = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    generator = trainer.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    expected = collect_guildworks(
        GuildworksEngine(), trainer.model, engine_seed=4, sampling_seeds=(5, 6)
    )
    report = evaluate_model(trainer.model, games_per_seat=4, seed=31)
    assert report == evaluate_model(trainer.model, games_per_seat=4, seed=31)
    assert len(report) == 2
    assert all(
        r["completed"] == 4 and r["errors"] == r["truncated"] == 0 for r in report
    )
    assert_nested_equal(model, trainer.model.state_dict())
    assert torch.equal(generator, trainer.generator.get_state())
    assert torch.equal(global_rng, torch.get_rng_state())
    actual = collect_guildworks(
        GuildworksEngine(), trainer.model, engine_seed=4, sampling_seeds=(5, 6)
    )
    for a, b in zip(expected, actual, strict=True):
        assert torch.equal(a.actions, b.actions) and torch.equal(a.tokens, b.tokens)


def test_tiny_cli_run_and_output_protection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = tmp_path / "run"
    monkeypatch.setattr(
        "sys.argv",
        [
            "guildworks_run",
            str(directory),
            "--updates",
            "1",
            "--games-per-update",
            "1",
            "--eval-games-per-seat",
            "2",
        ],
    )
    main()
    assert load_checkpoint(directory / "latest.pt").update_count == 1
    report = json.loads((directory / "evaluation.json").read_text())
    assert report["trained"][0]["completed"] == 2
    assert '"guildworks-functional-v1"' in capsys.readouterr().out
    before = (directory / "latest.pt").read_bytes()
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert (directory / "latest.pt").read_bytes() == before


@pytest.mark.parametrize(
    "kwargs", [{"updates": -1}, {"games_per_update": 0}, {"seed": True}]
)
def test_invalid_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RecurrentRunConfig(**kwargs)


def test_training_seed_namespaces_are_separate() -> None:
    streams = {
        derive_seed(1, i, s)
        for i in (0, 1)
        for s in (
            "initialization",
            "shuffle",
            "engine",
            "seat:0",
            "seat:1",
            "evaluation",
        )
    }
    assert len(streams) == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cuda_checkpoint_portability_and_continuation(tmp_path: Path) -> None:
    trainer = RecurrentPPOTrainer(initial_model(2).cuda(), shuffle_seed=3)
    pair = collect_guildworks(
        GuildworksEngine(), trainer.model, engine_seed=4, sampling_seeds=(5, 6)
    )
    trainer.update(pair)
    path = tmp_path / "cuda.pt"
    save_checkpoint(trainer, path)
    cpu = load_checkpoint(path)
    gpu = load_checkpoint(path, device="cuda")
    assert all(p.device.type == "cpu" for p in cpu.model.parameters())
    assert all(p.device.type == "cuda" for p in gpu.model.parameters())
    assert all(s["exp_avg"].device.type == "cuda" for s in gpu.optimizer.state.values())
    assert_nested_equal(trainer.model.state_dict(), gpu.model.state_dict())
    pair = collect_guildworks(
        GuildworksEngine(), gpu.model, engine_seed=10, sampling_seeds=(11, 12)
    )
    gpu.update(pair)
    assert gpu.update_count == 2
