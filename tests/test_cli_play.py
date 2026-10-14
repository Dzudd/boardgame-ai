"""Scripted terminal interaction through public inputs and validated steps."""

from collections.abc import Iterator

import pytest

from boardgame_ai.agents import Agent
from boardgame_ai.cli import main
from boardgame_ai.cli_play import HumanTicTacToeAgent, PlayCancelled, render_choices
from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation


def input_sequence(monkeypatch: pytest.MonkeyPatch, values: Iterator[str]) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))


def test_human_reprompts_without_mutating_game(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    game = TicTacToeEngine()
    game.reset(42)
    game.step(Action(kind="place", parameters=(("cell", 4),)))
    before = game.observe(1)
    assert game._rng is not None
    rng = game._rng.getstate()
    observation, legal = game.decision_for(1)
    input_sequence(monkeypatch, iter(["", "bad", "1.5", "-1", "9", "4", " 2 "]))
    selected = HumanTicTacToeAgent().select_action(observation, legal)
    assert selected == Action(kind="place", parameters=(("cell", 2),))
    assert any(selected is action for action in legal)
    assert game.observe(1) == before
    assert game._rng.getstate() == rng
    output = capsys.readouterr().out
    assert "Available cells: 0, 1, 2, 3, 5, 6, 7, 8" in output
    assert output.count("Enter a cell number") == 3
    assert output.count("That cell is unavailable") == 3
    assert output.count("Game board    Available cells") == 7
    game.step(selected)
    assert game.observe(1).data.board[2] == 1


@pytest.mark.parametrize("text", ["q", "quit", "EXIT", " Quit "])
def test_human_quit_preserves_position(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    game = TicTacToeEngine()
    game.reset()
    before = game.observe(0)
    input_sequence(monkeypatch, iter([text]))
    with pytest.raises(PlayCancelled):
        HumanTicTacToeAgent().select_action(*game.decision_for(0))
    assert game.observe(0) == before


def test_empty_choices_fail_without_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    game = TicTacToeEngine()
    game.reset()
    input_sequence(monkeypatch, iter([]))
    with pytest.raises(ValueError, match="legal choices"):
        HumanTicTacToeAgent().select_action(game.observe(0), ())


class ScriptedOpponent(Agent[TicTacToeObservation]):
    def __init__(self, moves: tuple[int, ...]) -> None:
        self.moves = moves
        self.index = 0
        self.seed: int | None = None

    def reset(self, seed: int | None = None) -> None:
        self.index = 0
        self.seed = seed

    def select_action(
        self,
        observation: Observation[TicTacToeObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        cell = self.moves[self.index]
        self.index += 1
        return next(
            action for action in legal_actions if action.parameters == (("cell", cell),)
        )


@pytest.mark.parametrize(
    "human, human_moves, ai_moves, message",
    [
        ("X", (0, 1, 2), (3, 4), "You win!"),
        ("O", (3, 4, 5), (0, 1, 8), "You win!"),
        ("X", (0, 1, 8), (3, 4, 5), "AI wins."),
        ("O", (3, 4), (0, 1, 2), "AI wins."),
        ("X", (0, 2, 3, 7, 8), (1, 4, 5, 6), "Draw."),
        ("O", (1, 4, 5, 6), (0, 2, 3, 7, 8), "Draw."),
    ],
)
def test_full_cli_games_report_final_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    human: str,
    human_moves: tuple[int, ...],
    ai_moves: tuple[int, ...],
    message: str,
) -> None:
    policy = ScriptedOpponent(ai_moves)
    monkeypatch.setattr("boardgame_ai.cli_play._make_opponent", lambda name: policy)
    input_sequence(monkeypatch, iter(str(cell) for cell in human_moves))
    main(
        [
            "play",
            "--starts",
            "human" if human == "X" else "ai",
            "--agent-seed",
            "17",
            "--engine-seed",
            "42",
        ]
    )
    output = capsys.readouterr().out
    assert f"you are {human}" in output
    assert "Game board    Available cells" in output and "Final board:" in output
    assert output.rstrip().endswith(message)
    assert policy.seed == 17
    assert policy.index == len(ai_moves)


@pytest.mark.parametrize("human", ["X", "O"])
@pytest.mark.parametrize("opponent", ["random", "heuristic"])
def test_real_opponents_complete_and_reproduce_scripted_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    human: str,
    opponent: str,
) -> None:
    outputs: list[str] = []
    for _ in range(2):
        # Rejected occupied cells do not consume a step or opponent randomness.
        input_sequence(monkeypatch, iter(str(cell) for cell in list(range(9)) * 6))
        main(
            [
                "play",
                "--starts",
                "human" if human == "X" else "ai",
                "--opponent",
                opponent,
                "--agent-seed",
                "42",
            ]
        )
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1]
    assert "Final board:" in outputs[0] and "Game stopped" not in outputs[0]
    if human == "O":
        assert outputs[0].index("AI (X) chooses") < outputs[0].index("Available cells:")


@pytest.mark.parametrize("reason", ["quit", "eof", "interrupt"])
def test_cli_cancellation_is_clean(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reason: str,
) -> None:
    def stop(prompt: str) -> str:
        if reason == "eof":
            raise EOFError
        if reason == "interrupt":
            raise KeyboardInterrupt
        return "quit"

    monkeypatch.setattr("builtins.input", stop)
    main(["play"])
    output = capsys.readouterr().out
    assert "Game stopped." in output
    assert "Final board:" not in output
    assert "Draw." not in output and "wins" not in output


@pytest.mark.parametrize(
    "arguments",
    [
        ["--human", "Z"],
        ["--opponent", "unknown"],
        ["--agent-seed", "abc"],
        ["--engine-seed", "1.5"],
        ["--starts", "unknown"],
    ],
)
def test_invalid_play_options_fail_before_prompt(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    input_sequence(monkeypatch, iter([]))
    with pytest.raises(SystemExit) as error:
        main(["play", *arguments])
    assert error.value.code == 2


def test_play_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["play", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--game" in output and "--opponent" in output
    assert "--human" not in output
    assert "--engine-seed" in output and "--agent-seed" in output
    assert "--starts" in output


def test_available_numbers_align_and_hide_occupied_cells() -> None:
    game = TicTacToeEngine()
    game.reset()
    game.step(Action(kind="place", parameters=(("cell", 4),)))
    observation, legal = game.decision_for(1)
    assert render_choices(observation, legal) == (
        "Game board    Available cells\n"
        ". | . | .    0 | 1 | 2\n"
        "---------    ---------\n"
        ". | X | .    3 | - | 5\n"
        "---------    ---------\n"
        ". | . | .    6 | 7 | 8"
    )


@pytest.mark.parametrize("starts", ["human", "ai"])
def test_starter_options_control_first_actor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    starts: str,
) -> None:
    input_sequence(monkeypatch, iter(["quit"]))
    main(["play", "--game", "tic-tac-toe", "--starts", starts])
    output = capsys.readouterr().out
    assert "X starts." in output
    assert f"you are {'X' if starts == 'human' else 'O'}" in output
    if starts == "human":
        assert "AI (" not in output
        assert ". | . | .    0 | 1 | 2" in output
    else:
        assert "AI (X) chooses cell 4." in output
        assert output.index("AI (") < output.index("Game board")
        assert "3 | - | 5" in output


@pytest.mark.parametrize(
    "arguments",
    [
        ["--starts", "X"],
        ["--starts", "O"],
        ["--human", "X"],
        ["--game", "chess"],
    ],
)
def test_removed_mark_options_and_unknown_game(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["play", *arguments])
    assert error.value.code == 2
