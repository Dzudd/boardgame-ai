"""Reusable game loop, keeping authoritative results outside agent inputs."""

from collections.abc import Callable
from dataclasses import dataclass

from boardgame_ai.agents import Agent
from boardgame_ai.core import Action, GameEngine, create_rng


@dataclass(frozen=True, slots=True, kw_only=True)
class MatchResult:
    """Trusted summary; total rewards are not game scores or winner labels.

    Rewards sum only steps executed by this call, excluding any earlier play.
    Truncation leaves the engine at its live position and is not a draw.
    """

    num_steps: int
    total_rewards: tuple[float, ...]
    terminated: bool
    truncated: bool


def _validate_setup[Data](
    game: GameEngine[Data], agents: tuple[Agent[Data], ...], max_steps: int
) -> None:
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if len(agents) != game.num_players:
        raise ValueError("Provide one agent per player seat")
    if len({id(agent) for agent in agents}) != len(agents):
        raise ValueError("Each player seat requires a separate agent instance")


def play_game[Data](
    game: GameEngine[Data],
    agents: tuple[Agent[Data], ...],
    *,
    max_steps: int,
    on_step: Callable[[int, Action], None] | None = None,
) -> MatchResult:
    """Play an initialized position without resetting engine or agents.

    Follow engine actors, including repeated actors. Stop at termination or the
    decision limit. Exceptions propagate without retry or action substitution.
    Optional on_step runs after a validated successful step with actor/action
    only. It is trusted orchestration, not agent input; callback errors propagate
    after the executed step and are not rolled back.
    """
    _validate_setup(game, agents, max_steps)
    total_rewards = [0.0] * game.num_players
    num_steps = 0
    terminated = game.is_terminal()
    player = game.current_player()
    while True:
        if terminated != (player is None):
            raise RuntimeError("Engine actor and termination disagree")
        if terminated or num_steps == max_steps:
            return MatchResult(
                num_steps=num_steps,
                total_rewards=tuple(total_rewards),
                terminated=terminated,
                truncated=not terminated,
            )
        # The termination invariant above ensures a live actor exists.
        assert player is not None
        game.validate_player(player)
        observation, legal = game.decision_for(player)
        if observation.player_id != player or observation.current_player != player:
            raise RuntimeError("Engine decision view does not identify the actor")
        if not legal:
            raise RuntimeError("Engine live actor has no legal actions")
        action = agents[player].select_action(observation, legal)
        result = game.step(action)
        if (
            result.next_player != game.current_player()
            or result.terminated != game.is_terminal()
            or len(result.rewards) != game.num_players
        ):
            raise RuntimeError("Engine step result disagrees with engine contract")
        for seat, reward in enumerate(result.rewards):
            total_rewards[seat] += reward
        num_steps += 1
        if on_step is not None:
            on_step(player, action)
        player = result.next_player
        terminated = result.terminated


def run_match[Data](
    game: GameEngine[Data],
    agents: tuple[Agent[Data], ...],
    *,
    max_steps: int,
    engine_seed: int | None = None,
    agent_seeds: tuple[int | None, ...] | None = None,
    on_step: Callable[[int, Action], None] | None = None,
) -> MatchResult:
    """Reset and run one match with separate explicit engine and agent seeds.

    Omitted agent seeds mean reset(None) for every seat. Validate configuration
    before resets. Custom reset failures propagate; resets across objects are not
    transactional. Optional on_step is forwarded to play_game; recording is opt-in.
    No seed derivation or batch execution is performed.
    """
    _validate_setup(game, agents, max_steps)
    seeds = (None,) * game.num_players if agent_seeds is None else agent_seeds
    if len(seeds) != game.num_players:
        raise ValueError("Provide one agent seed per player seat")
    create_rng(engine_seed)  # Shared seed validation; no global RNG use.
    for seed in seeds:
        create_rng(seed)
    game.reset(engine_seed)
    for agent, seed in zip(agents, seeds, strict=True):
        agent.reset(seed)
    return play_game(game, agents, max_steps=max_steps, on_step=on_step)
