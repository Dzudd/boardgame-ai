"""Human CLI flow, reproducibility and filtered presentation boundaries."""

from collections.abc import Iterator

import pytest

from boardgame_ai.agents import Agent
from boardgame_ai.cli import main
from boardgame_ai.cli_guildworks import (
    HumanGuildworksAgent,
    _ReportingGuildworksAgent,
)
from boardgame_ai.cli_play import PlayCancelled
from boardgame_ai.core import Action, Observation
from boardgame_ai.games.guildworks import GuildworksEngine, GuildworksObservation
from boardgame_ai.games.guildworks.evaluation import guildworks_outcome


def inputs(monkeypatch: pytest.MonkeyPatch, values: Iterator[str]) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))


@pytest.mark.parametrize("phase", ["place", "craft"])
def test_retries_return_supplied_action_without_side_effects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], phase: str
) -> None:
    game = GuildworksEngine()
    game.reset(42)
    if phase == "craft":
        game.step(Action(kind="place", parameters=(("space", "archive"),)))
    before = game.clone()
    observation, legal = game.decision_for(0)
    inputs(monkeypatch, iter(["", "bad", "1.5", "0", "-1", "99", " 1 "]))
    action = HumanGuildworksAgent().select_action(observation, legal)
    assert action is legal[0]
    for seat in (0, 1):
        assert game.observe(seat) == before.observe(seat)
    assert game._rng.getstate() == before._rng.getstate()
    assert game.get_rewards() == before.get_rewards()
    output = capsys.readouterr().out
    assert output.count("Enter a listed choice number") == 3
    assert output.count("That choice is unavailable") == 3
    assert "Seat 0 hand" in output and "Seat 1 hand" not in output
    assert ("Place a worker:" if phase == "place" else "Craft or pass:") in output
    assert game.step(action) == before.step(action)


def test_forced_pass_and_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    game = GuildworksEngine()
    game.reset(42)
    game.step(Action(kind="place", parameters=(("space", "festival"),)))
    game._ink[0] = 0
    observation, legal = game.decision_for(0)
    inputs(monkeypatch, iter(["1"]))
    assert HumanGuildworksAgent().select_action(observation, legal) is legal[0]
    assert legal == (Action(kind="pass"),)
    inputs(monkeypatch, iter([]))
    with pytest.raises(ValueError, match="legal choices"):
        HumanGuildworksAgent().select_action(observation, ())


@pytest.mark.parametrize("text", ["q", "quit", "EXIT", " Quit "])
def test_quit_submits_no_action(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    game = GuildworksEngine()
    game.reset(42)
    before = game.observe(0)
    rng = game._rng.getstate()
    inputs(monkeypatch, iter([text]))
    with pytest.raises(PlayCancelled):
        HumanGuildworksAgent().select_action(*game.decision_for(0))
    assert game.observe(0) == before
    assert game._rng.getstate() == rng


@pytest.mark.parametrize("starts", ["human", "ai"])
def test_real_random_complete_games_and_reproducible_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], starts: str
) -> None:
    outputs: list[str] = []
    for _ in range(2):
        game = GuildworksEngine()

        def game_factory(instance: GuildworksEngine = game) -> GuildworksEngine:
            return instance

        monkeypatch.setattr(
            "boardgame_ai.cli_guildworks.GuildworksEngine", game_factory
        )
        # First offered placement/craft, or forced pass: 16 human decisions.
        inputs(monkeypatch, iter(["1"] * 16))
        main(
            [
                "play",
                "--game",
                "guildworks",
                "--starts",
                starts,
                "--engine-seed",
                "42",
                "--agent-seed",
                "17",
            ]
        )
        output = capsys.readouterr().out
        outputs.append(output)
        seat = 0 if starts == "human" else 1
        assert game.is_terminal()
        assert output.count("AI (seat") == 16
        assert output.count("Place a worker:") == 8
        assert output.count("Craft or pass:") == 8
        assert f"Seat {seat} hand" in output
        assert f"Seat {1 - seat} hand" not in output
        assert "Game stopped" not in output
        assert "Final position:" in output
        scores = game.observe(seat).data.scores
        assert f"Final scores: you {scores[seat]}, AI {scores[1 - seat]}." in output
        outcome = guildworks_outcome(game)
        message = (
            "Draw."
            if outcome.draw
            else ("You win!" if seat in outcome.winners else "AI wins.")
        )
        assert output.rstrip().endswith(message)
        if starts == "ai":
            assert output.index("AI (seat") < output.index("Place a worker:")
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("reason", ["quit", "eof", "interrupt"])
@pytest.mark.parametrize("phase", ["place", "craft"])
def test_cancellation_is_clean(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reason: str,
    phase: str,
) -> None:
    calls = 0

    def stop(prompt: str) -> str:
        nonlocal calls
        calls += 1
        if phase == "craft" and calls == 1:
            return "1"
        if reason == "eof":
            raise EOFError
        if reason == "interrupt":
            raise KeyboardInterrupt
        return "quit"

    monkeypatch.setattr("builtins.input", stop)
    main(["play", "--game", "guildworks"])
    output = capsys.readouterr().out
    assert "Game stopped." in output
    assert "Final position:" not in output and "Final scores:" not in output
    assert "You win!" not in output and "AI wins." not in output
    assert "Draw." not in output
    assert "AI (seat" not in output


@pytest.mark.parametrize(
    "arguments", [["--opponent", "heuristic"], ["--save-replay", "unused.json"]]
)
def test_unsupported_options_fail_before_play(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    inputs(monkeypatch, iter([]))
    with pytest.raises(SystemExit) as error:
        main(["play", "--game", "guildworks", *arguments])
    assert error.value.code == 2
    assert capsys.readouterr().out == ""


class FixedAgent(Agent[GuildworksObservation]):
    def select_action(
        self,
        observation: Observation[GuildworksObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        return legal_actions[0]


def test_ai_reporting_does_not_print_private_view(
    capsys: pytest.CaptureFixture[str],
) -> None:
    game = GuildworksEngine()
    game.reset(42)
    twin = game.clone()
    twin._hands[0] = [3, 3]
    twin._deck.reverse()
    reporter = _ReportingGuildworksAgent(FixedAgent())
    outputs = []
    for world in (game, twin):
        action = reporter.select_action(*world.decision_for(0))
        assert action.kind == "place"
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1] == "AI (seat 0): Place at inkwell (gain 2 ink).\n"


def test_help_mentions_guildworks(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["play", "--help"])
    assert error.value.code == 0
    assert "guildworks" in capsys.readouterr().out
