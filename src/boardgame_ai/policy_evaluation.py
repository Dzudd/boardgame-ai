"""Held-out fixed-opponent CPU evaluation of one immutable checkpoint snapshot."""

import json
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

from boardgame_ai.agents import RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.agents.base import Agent
from boardgame_ai.agents.policy_agent import PolicyMode, TicTacToePolicyAgent
from boardgame_ai.artifact_safety import validate_evaluation_destination
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.games.tic_tac_toe.evaluation import tic_tac_toe_outcome
from boardgame_ai.simulation.evaluation import EvaluationReport, evaluate_batch
from boardgame_ai.training.checkpoint import load_ppo_checkpoint


@dataclass(frozen=True, slots=True, kw_only=True)
class FixedOpponentCase:
    opponent: Literal["random-v1", "heuristic-v1"]
    learner_seat: int
    report: EvaluationReport


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyEvaluation:
    checkpoint_sha256: str
    checkpoint_updates: int
    checkpoint_optimizer_steps: int
    policy_mode: PolicyMode
    held_out_seed: int
    seed_schedule: str
    games_per_seat: int
    start_index: int
    cases: tuple[FixedOpponentCase, ...]


def _case_seed(root: int, opponent: str, learner_seat: int) -> int:
    value = f"boardgame-ai/evaluation/v1:{root}:{opponent}:{learner_seat}"
    return int.from_bytes(sha256(value.encode("utf-8")).digest(), "big")


def evaluate_policy_checkpoint(
    checkpoint: str | Path,
    *,
    games_per_seat: int = 1000,
    seed: int = 20261015,
    start_index: int = 0,
    mode: PolicyMode = "sampled",
    on_case: Callable[[FixedOpponentCase], None] | None = None,
) -> PolicyEvaluation:
    """Evaluate four fixed cases (random/heuristic × learner X/O), on CPU.

    Read the checkpoint once, hash those bytes, and validate/load a temporary
    snapshot. Source/checkpoint/training files are never written. The model
    remains fixed across all matches; no learning or CUDA computation occurs.
    Use after training finishes to avoid touching a checkpoint being replaced.

    Evaluation seeds use an independent versioned namespace before the existing
    batch schedule. Identical boards may occur in training and evaluation;
    held-out seeds mean independent RNG schedules, not disjoint state coverage.
    Match errors are retained as fixed categories; no exception text is exported.
    """
    if type(games_per_seat) is not int or games_per_seat <= 0:
        raise ValueError("games_per_seat must be a positive integer")
    if type(seed) is not int or type(start_index) is not int or start_index < 0:
        raise ValueError("Evaluation seed/index must be integers; index nonnegative")
    if mode not in ("sampled", "greedy"):
        raise ValueError("Policy mode must be sampled or greedy")
    snapshot = Path(checkpoint).read_bytes()
    digest = sha256(snapshot).hexdigest()
    with tempfile.TemporaryDirectory(prefix="boardgame-ai-eval-") as folder:
        path = Path(folder) / "snapshot.pt"
        path.write_bytes(snapshot)
        trainer = load_ppo_checkpoint(path)  # CPU is explicit loader default.
    cases: list[FixedOpponentCase] = []
    for opponent_name in ("random-v1", "heuristic-v1"):
        for seat in (0, 1):
            learner = TicTacToePolicyAgent(trainer.model, mode=mode)
            opponent: Agent[TicTacToeObservation] = (
                RandomAgent[TicTacToeObservation]()
                if opponent_name == "random-v1"
                else TicTacToeHeuristicAgent()
            )
            agents = (learner, opponent) if seat == 0 else (opponent, learner)
            labels = (
                ("checkpoint-policy", opponent_name)
                if seat == 0
                else (opponent_name, "checkpoint-policy")
            )
            report = evaluate_batch(
                TicTacToeEngine(),
                agents,
                outcome_for=tic_tac_toe_outcome,
                agent_labels=labels,
                num_matches=games_per_seat,
                seed=_case_seed(seed, opponent_name, seat),
                max_steps=9,
                start_index=start_index,
            )
            case = FixedOpponentCase(
                opponent=opponent_name, learner_seat=seat, report=report
            )
            cases.append(case)
            if on_case is not None:
                on_case(case)
    return PolicyEvaluation(
        checkpoint_sha256=digest,
        checkpoint_updates=trainer.update_count,
        checkpoint_optimizer_steps=trainer.optimizer_steps,
        policy_mode=mode,
        held_out_seed=seed,
        seed_schedule="fixed-opponents-v1+batch-v1",
        games_per_seat=games_per_seat,
        start_index=start_index,
        cases=tuple(cases),
    )


def save_policy_evaluation(result: PolicyEvaluation, path: str | Path) -> None:
    """Export a new trusted JSON report; never replace an existing artifact.

    No boards, observations, private events, exception text, or full state.
    Validate JSON serialization before writing; seeds are trusted metadata.
    """
    encoded = (
        json.dumps(
            {"schema_version": 1, "game_id": "tic_tac_toe", **asdict(result)},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    destination = validate_evaluation_destination(path)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
