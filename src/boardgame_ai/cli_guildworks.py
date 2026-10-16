"""Filtered terminal adapters for Guildworks human-versus-random play."""

from boardgame_ai.agents import Agent, RandomAgent
from boardgame_ai.cli_play import PlayCancelled
from boardgame_ai.core import Action, Observation
from boardgame_ai.games.guildworks import GuildworksEngine, GuildworksObservation
from boardgame_ai.games.guildworks.engine import MAX_STEPS, render_guildworks
from boardgame_ai.games.guildworks.evaluation import guildworks_outcome
from boardgame_ai.simulation import run_match

_SPACE_EFFECTS = {
    "inkwell": "gain 2 ink",
    "archive": "draw up to 2 cards",
    "festival": "score 1 point",
    "lottery": "gain 1-3 ink",
}


def describe_action(action: Action) -> str:
    """Describe a supplied choice; placement choices contain no chance outcome."""
    if action.kind == "place":
        space = str(action.parameters[0][1])
        return f"Place at {space} ({_SPACE_EFFECTS[space]})"
    if action.kind == "craft":
        card = action.parameters[0][1]
        assert type(card) is int
        return f"Craft card {card} (pay {card} ink, score {card + 1} points)"
    return "Pass crafting"


class HumanGuildworksAgent(Agent[GuildworksObservation]):
    """Prompt from detached own/public inputs only, returning a supplied action."""

    def select_action(
        self,
        observation: Observation[GuildworksObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        if not legal_actions:
            raise ValueError("Human player requires legal choices")
        while True:
            print(render_guildworks(observation))
            print(
                "Place a worker:" if observation.phase == "place" else "Craft or pass:"
            )
            for index, action in enumerate(legal_actions, 1):
                print(f"  {index}. {describe_action(action)}")
            text = input(f"Choose 1-{len(legal_actions)} (or quit): ").strip()
            if text.lower() in ("q", "quit", "exit"):
                raise PlayCancelled
            try:
                index = int(text)
            except ValueError:
                print("Enter a listed choice number, or quit.")
                continue
            if not 1 <= index <= len(legal_actions):
                print("That choice is unavailable. Choose a listed number.")
                continue
            return legal_actions[index - 1]


class _ReportingGuildworksAgent(Agent[GuildworksObservation]):
    """Print only the chosen public action, never the AI's observation/hand."""

    def __init__(self, policy: Agent[GuildworksObservation]) -> None:
        self._policy = policy

    def reset(self, seed: int | None = None) -> None:
        self._policy.reset(seed)

    def select_action(
        self,
        observation: Observation[GuildworksObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        action = self._policy.select_action(observation, legal_actions)
        print(f"AI (seat {observation.player_id}): {describe_action(action)}.")
        return action


def play_guildworks(
    *, engine_seed: int | None, agent_seed: int | None, starts: str = "human"
) -> None:
    """Run one bounded match; final presentation also uses the human's view."""
    if starts not in ("human", "ai"):
        raise ValueError("Starter must be human or ai")
    human_seat = 0 if starts == "human" else 1
    human = HumanGuildworksAgent()
    ai = _ReportingGuildworksAgent(RandomAgent[GuildworksObservation]())
    agents: tuple[Agent[GuildworksObservation], ...] = (
        (human, ai) if human_seat == 0 else (ai, human)
    )
    seeds = (None, agent_seed) if human_seat == 0 else (agent_seed, None)
    game = GuildworksEngine()
    print(f"Guildworks: you are seat {human_seat}; opponent is random. Seat 0 starts.")
    print("Four rounds, two workers each per round; the round starter alternates.")
    print("Place on a free space, then craft one affordable card or pass.")
    print("Highest score wins; equal scores draw. Enter quit to stop.")
    try:
        result = run_match(
            game,
            agents,
            max_steps=MAX_STEPS,
            engine_seed=engine_seed,
            agent_seeds=seeds,
        )
    except (PlayCancelled, EOFError, KeyboardInterrupt):
        print("\nGame stopped.")
        return
    print("Final position:")
    print(render_guildworks(game.observe(human_seat)))
    if result.truncated:
        print("Game stopped at the decision limit.")
        return
    outcome = guildworks_outcome(game)
    scores = game.observe(human_seat).data.scores
    print(f"Final scores: you {scores[human_seat]}, AI {scores[1 - human_seat]}.")
    if outcome.draw:
        print("Draw.")
    elif human_seat in outcome.winners:
        print("You win!")
    else:
        print("AI wins.")
