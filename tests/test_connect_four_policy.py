"""Frozen Connect Four checkpoint opponents and terminal play regressions."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from boardgame_ai.agents import RandomAgent
from boardgame_ai.agents.connect_four_policy import ConnectFourPolicyAgent
from boardgame_ai.cli import main
from boardgame_ai.core import Action
from boardgame_ai.games.connect_four import ConnectFourEngine, ConnectFourObservation
from boardgame_ai.models.connect_four import (
    ConnectFourActorCritic,
    initial_connect_four_model,
)
from boardgame_ai.simulation import run_match
from boardgame_ai.training.connect_four_checkpoint import save_connect_four_checkpoint
from boardgame_ai.training.connect_four_ppo import ConnectFourPPOTrainer


@pytest.fixture
def model() -> ConnectFourActorCritic:
    result = initial_connect_four_model(81)
    with torch.no_grad():
        for p in result.parameters():
            p.zero_()
    return result


@pytest.fixture
def checkpoint(model: ConnectFourActorCritic, tmp_path: Path) -> Path:
    path = tmp_path / "portable.pt"
    save_connect_four_checkpoint(ConnectFourPPOTrainer(model), path)
    return path


def test_masked_greedy_tie_order_and_supplied_identity(
    model: ConnectFourActorCritic,
) -> None:
    game = ConnectFourEngine()
    game.reset(1)
    with torch.no_grad():
        model.policy_head.bias[0] = 1000
    agent = ConnectFourPolicyAgent(model, mode="greedy")
    view, legal = game.decision_for(0)
    supplied = (legal[5], legal[2])
    assert agent.select_action(view, supplied) is supplied[1]
    assert agent.select_action(view, (legal[6],)) is legal[6]


@pytest.mark.parametrize("mode", ["sampled", "greedy"])
def test_full_column_and_complete_games_both_seats(
    model: ConnectFourActorCritic, mode: Any
) -> None:
    with torch.no_grad():
        model.policy_head.bias[0] = 1000
    agent = ConnectFourPolicyAgent(model, mode=mode)
    game = ConnectFourEngine()
    game.reset(0)
    for _ in range(6):
        game.step(Action(kind="drop", parameters=(("column", 0),)))
    view, legal = game.decision_for(0)
    for _ in range(10):
        assert agent.select_action(view, legal) in legal
    for seat in (0, 1):
        opponent: RandomAgent[ConnectFourObservation] = RandomAgent()
        agents = (agent, opponent) if seat == 0 else (opponent, agent)
        result = run_match(
            game, agents, engine_seed=1, agent_seeds=(2, 3), max_steps=42
        )
        assert result.terminated and not result.truncated


def test_direct_seed_reset_matches_hand_sample_and_rejection_preserves_rng(
    model: ConnectFourActorCritic,
) -> None:
    agent = ConnectFourPolicyAgent(model)
    game = ConnectFourEngine()
    game.reset(0)
    view, legal = game.decision_for(0)
    seed = 2**64 - 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    expected = [
        int(torch.multinomial(torch.ones(7) / 7, 1, generator=generator).item())
        for _ in range(20)
    ]
    agent.reset(seed)
    actual = [agent.select_action(view, legal).parameters[0][1] for _ in range(20)]
    assert actual == expected
    agent.reset(seed)
    assert [
        agent.select_action(view, legal).parameters[0][1] for _ in range(20)
    ] == actual
    state = agent._generator.get_state().clone()
    for bad in (True, -1, 2**64):
        with pytest.raises(ValueError):
            agent.reset(bad)
    with pytest.raises(ValueError):
        agent.select_action(view, ())
    with pytest.raises(ValueError):
        agent.select_action(game.observe(1), legal)
    assert torch.equal(state, agent._generator.get_state())


def test_frozen_snapshot_source_gradients_modes_and_rng(
    model: ConnectFourActorCritic,
) -> None:
    model.body[0].eval()
    model.policy_head.weight.grad = torch.ones_like(model.policy_head.weight)
    modes = [m.training for m in model.modules()]
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None  # type: ignore[no-untyped-call]
    agent = ConnectFourPolicyAgent(model, mode="greedy")
    with torch.no_grad():
        model.policy_head.bias[6] = 1000
    game = ConnectFourEngine()
    game.reset(0)
    view, legal = game.decision_for(0)
    agent.reset(19)
    assert agent.select_action(view, legal) is legal[0]
    assert [m.training for m in model.modules()] == modes
    assert torch.equal(
        model.policy_head.weight.grad, torch.ones_like(model.policy_head.weight)
    )
    assert all(
        p.grad is None and not p.requires_grad for p in agent._model.parameters()
    )
    assert random.getstate() == python
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), cpu)
    if cuda is not None:
        for before, after in zip(cuda, torch.cuda.get_rng_state_all(), strict=True):
            assert torch.equal(before, after)


@pytest.mark.parametrize("bad", ["dtype", "nonfinite", "mode"])
def test_invalid_policy_rejected(model: ConnectFourActorCritic, bad: str) -> None:
    mode: Any = "sampled"
    if bad == "dtype":
        model.double()
    elif bad == "nonfinite":
        with torch.no_grad():
            model.policy_head.bias[0] = float("nan")
    else:
        mode = "unknown"
    with pytest.raises(ValueError):
        ConnectFourPolicyAgent(model, mode=mode)


@pytest.mark.parametrize("starts", ["human", "ai"])
@pytest.mark.parametrize("mode", ["sampled", "greedy"])
def test_complete_checkpoint_cli_games_read_only(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    starts: str,
    mode: str,
) -> None:
    original = checkpoint.read_bytes()
    values = iter([str(c) for _ in range(42) for c in range(1, 8)])
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))
    main(
        [
            "play",
            "--game",
            "connect-four",
            "--checkpoint",
            str(checkpoint),
            "--starts",
            starts,
            "--policy-mode",
            mode,
            "--agent-seed",
            "42",
        ]
    )
    output = capsys.readouterr().out
    assert f"checkpoint policy ({mode})" in output
    assert "Final position:" in output and "1 2 3 4 5 6 7" in output
    assert any(text in output for text in ("You win!", "AI wins.", "Draw."))
    assert checkpoint.read_bytes() == original


@pytest.mark.parametrize("stop", ["quit", "eof", "interrupt"])
def test_checkpoint_cli_cancellation(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stop: str,
) -> None:
    def input_value(prompt: str) -> str:
        if stop == "eof":
            raise EOFError
        if stop == "interrupt":
            raise KeyboardInterrupt
        return "quit"

    monkeypatch.setattr("builtins.input", input_value)
    main(["play", "--game", "connect-four", "--checkpoint", str(checkpoint)])
    output = capsys.readouterr().out
    assert "Game stopped." in output and "Final position:" not in output


@pytest.mark.parametrize(
    "kind", ["missing", "corrupt", "foreign", "run-envelope", "nonfinite"]
)
def test_bad_checkpoints_reject_before_input_and_preserve_files(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    if kind == "missing":
        checkpoint = checkpoint.with_name("missing.pt")
    elif kind == "corrupt":
        checkpoint.write_bytes(b"garbage")
    else:
        payload = torch.load(checkpoint, weights_only=True)
        if kind == "foreign":
            payload["game_id"] = "tic_tac_toe"
        elif kind == "run-envelope":
            payload = {"schema_version": 1, "trainer": payload}
        else:
            payload["model_state"]["policy_head.bias"][0] = float("nan")
        torch.save(payload, checkpoint)
    original = checkpoint.read_bytes() if checkpoint.exists() else None

    def forbidden(prompt: str) -> str:
        raise AssertionError("Must validate before human input")

    monkeypatch.setattr("builtins.input", forbidden)
    with pytest.raises(SystemExit) as error:
        main(["play", "--game", "connect-four", "--checkpoint", str(checkpoint)])
    assert error.value.code == 2
    assert "Cannot load or use Connect Four opponent" in capsys.readouterr().err
    assert (checkpoint.read_bytes() if checkpoint.exists() else None) == original


@pytest.mark.parametrize(
    "extra",
    [
        ["--game", "tic-tac-toe", "--checkpoint", "x.pt"],
        ["--game", "guildworks", "--checkpoint", "x.pt"],
        ["--game", "connect-four", "--checkpoint", "x.pt", "--opponent", "random"],
        ["--game", "connect-four", "--checkpoint", "x.pt", "--save-replay", "x.json"],
        ["--game", "connect-four", "--policy-mode", "greedy"],
    ],
)
def test_invalid_cli_combinations(
    extra: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(prompt: str) -> str:
        raise AssertionError("Must reject before human input")

    monkeypatch.setattr("builtins.input", forbidden)
    with pytest.raises(SystemExit) as error:
        main(["play", *extra])
    assert error.value.code == 2


def test_checkpoint_cli_seed_validation_and_reproducibility(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def play() -> str:
        values = iter([str(c) for _ in range(42) for c in range(1, 8)])
        monkeypatch.setattr("builtins.input", lambda prompt: next(values))
        main(
            [
                "play",
                "--game",
                "connect-four",
                "--checkpoint",
                str(checkpoint),
                "--agent-seed",
                "17",
            ]
        )
        return capsys.readouterr().out

    assert play() == play()

    def forbidden(prompt: str) -> str:
        raise AssertionError("Must reject seed before input")

    monkeypatch.setattr("builtins.input", forbidden)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "play",
                "--game",
                "connect-four",
                "--checkpoint",
                str(checkpoint),
                "--agent-seed",
                "-1",
            ]
        )
    assert error.value.code == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_source_snapshot_and_cpu_play(model: ConnectFourActorCritic) -> None:
    model.cuda()
    model.policy_head.weight.grad = torch.ones_like(model.policy_head.weight)
    modes = [m.training for m in model.modules()]
    states = torch.cuda.get_rng_state_all()
    agent = ConnectFourPolicyAgent(model, mode="greedy")
    assert all(p.device.type == "cpu" for p in agent._model.parameters())
    game = ConnectFourEngine()
    game.reset(1)
    view, legal = game.decision_for(0)
    assert agent.select_action(view, legal) is legal[0]
    assert [m.training for m in model.modules()] == modes
    assert model.policy_head.weight.device.type == "cuda"
    assert torch.equal(
        model.policy_head.weight.grad, torch.ones_like(model.policy_head.weight)
    )
    for before, after in zip(states, torch.cuda.get_rng_state_all(), strict=True):
        assert torch.equal(before, after)


def test_play_optimized_checkpoint_preserves_saved_optimizer_and_global_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from boardgame_ai.training.connect_four import collect_connect_four_self_play
    from boardgame_ai.training.connect_four_ppo import prepare_connect_four_batch

    trainer = ConnectFourPPOTrainer(initial_connect_four_model(38))
    rollout = collect_connect_four_self_play(
        ConnectFourEngine(), trainer.model, sampling_seeds=(51, 52), engine_seed=53
    )
    trainer.update(prepare_connect_four_batch(rollout.players, trainer.config))
    path = tmp_path / "trained.pt"
    save_connect_four_checkpoint(trainer, path)
    original = path.read_bytes()
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    values = iter([str(c) for _ in range(42) for c in range(1, 8)])
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))
    main(
        [
            "play",
            "--game",
            "connect-four",
            "--checkpoint",
            str(path),
            "--agent-seed",
            "51",
        ]
    )
    assert "Final position:" in capsys.readouterr().out
    assert path.read_bytes() == original
    assert random.getstate() == python
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), cpu)
    assert torch.get_num_threads() == threads
