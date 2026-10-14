"""Sequential match batches with independently reproducible seed schedules."""

from dataclasses import dataclass
from hashlib import sha256

from boardgame_ai.agents import Agent
from boardgame_ai.core import GameEngine
from boardgame_ai.simulation.runner import MatchResult, _validate_setup, run_match


@dataclass(frozen=True, slots=True, kw_only=True)
class SeededMatchResult:
    """Trusted single-match summary and lifecycle seeds, without private inputs."""

    match_index: int
    engine_seed: int
    agent_seeds: tuple[int, ...]
    result: MatchResult


def _derive_seed(batch_seed: int, match_index: int, stream: str) -> int:
    # Versioned UTF-8 decimal encoding; independent of Python's randomized hash.
    value = f"boardgame-ai/batch/v1:{batch_seed}:{match_index}:{stream}"
    return int.from_bytes(sha256(value.encode("utf-8")).digest(), "big")


def _validate_batch[Data](
    game: GameEngine[Data],
    agents: tuple[Agent[Data], ...],
    num_matches: int,
    seed: int,
    max_steps: int,
    start_index: int,
) -> None:
    _validate_setup(game, agents, max_steps)
    if type(seed) is not int:
        raise ValueError("Batch seed must be an integer")
    if type(num_matches) is not int or num_matches < 0:
        raise ValueError("num_matches must be a nonnegative integer")
    if type(start_index) is not int or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")


def run_batch[Data](
    game: GameEngine[Data],
    agents: tuple[Agent[Data], ...],
    *,
    num_matches: int,
    seed: int,
    max_steps: int,
    start_index: int = 0,
) -> tuple[SeededMatchResult, ...]:
    """Run a fixed-seat batch, resetting each match with derived private seeds.

    Require an integer batch seed and nonnegative count/start index. A range's
    seeds depend only on seed, absolute match index, and engine/seat identifier.
    Reproduction also requires identical configuration, code, and runtime, plus
    engines/agents that fully reset and honor their seeds. Exceptions propagate
    immediately; no partial batch is returned, logged, or rolled back.
    """
    _validate_batch(game, agents, num_matches, seed, max_steps, start_index)

    results: list[SeededMatchResult] = []
    for index in range(start_index, start_index + num_matches):
        engine_seed = _derive_seed(seed, index, "engine")
        agent_seeds = tuple(
            _derive_seed(seed, index, f"agent:{seat}")
            for seat in range(game.num_players)
        )
        result = run_match(
            game,
            agents,
            max_steps=max_steps,
            engine_seed=engine_seed,
            agent_seeds=agent_seeds,
        )
        results.append(
            SeededMatchResult(
                match_index=index,
                engine_seed=engine_seed,
                agent_seeds=agent_seeds,
                result=result,
            )
        )
    return tuple(results)
