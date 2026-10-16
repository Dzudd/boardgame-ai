"""Human Connect Four play against a random or frozen checkpoint opponent."""

from pathlib import Path
from typing import Literal

from boardgame_ai.agents import Agent, RandomAgent
from boardgame_ai.cli_play import PlayCancelled
from boardgame_ai.core import Action, Observation
from boardgame_ai.games.connect_four import ConnectFourEngine, ConnectFourObservation
from boardgame_ai.games.connect_four.evaluation import connect_four_outcome
from boardgame_ai.games.connect_four.rendering import render_board
from boardgame_ai.simulation import run_match


class HumanConnectFourAgent(Agent[ConnectFourObservation]):
    """Return supplied actions, translating human columns 1–7 to indices 0–6."""

    def select_action(
        self,
        observation: Observation[ConnectFourObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        if not legal_actions:
            raise ValueError("Human player requires legal choices")
        print(render_board(observation))
        while True:
            text = input("Drop in column 1-7 (or quit): ").strip()
            if text.lower() in ("q", "quit", "exit"):
                raise PlayCancelled
            try:
                column = int(text) - 1
            except ValueError:
                print("Enter a column number from 1 to 7, or quit.")
                continue
            for action in legal_actions:
                if action.parameters == (("column", column),):
                    return action
            print("That column is full or unavailable. Choose a column from 1 to 7.")


class _ReportingAgent(Agent[ConnectFourObservation]):
    def __init__(self, policy: Agent[ConnectFourObservation]) -> None:
        self._policy = policy

    def reset(self, seed: int | None = None) -> None:
        self._policy.reset(seed)

    def select_action(
        self,
        observation: Observation[ConnectFourObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        action = self._policy.select_action(observation, legal_actions)
        column = action.parameters[0][1]
        assert type(column) is int
        print(f"AI drops in column {column + 1}.")
        return action


def play_connect_four(
    *,
    starts: str = "human",
    engine_seed: int | None,
    agent_seed: int | None,
    checkpoint: str | Path | None = None,
    policy_mode: Literal["sampled", "greedy"] | None = None,
) -> None:
    """Validate/load before human input; never modify checkpoint or run artifacts."""
    if starts not in ("human", "ai"):
        raise ValueError("Starter must be human or ai")
    policy: Agent[ConnectFourObservation]
    if checkpoint is None:
        if policy_mode is not None:
            raise ValueError("Policy mode requires a checkpoint")
        policy = RandomAgent()
    else:
        # Preserve lightweight imports for random-only play and CLI help.
        from boardgame_ai.agents.connect_four_policy import ConnectFourPolicyAgent
        from boardgame_ai.training.connect_four_checkpoint import (
            load_connect_four_checkpoint,
        )

        trainer = load_connect_four_checkpoint(checkpoint, device="cpu")
        policy = ConnectFourPolicyAgent(trainer.model, mode=policy_mode or "sampled")
        policy.reset(agent_seed)  # Reject invalid seeds before printing/prompting.
        del trainer
    human_seat = 0 if starts == "human" else 1
    human = HumanConnectFourAgent()
    ai = _ReportingAgent(policy)
    agents: tuple[Agent[ConnectFourObservation], ...] = (
        (human, ai) if human_seat == 0 else (ai, human)
    )
    game = ConnectFourEngine()
    print(f"Connect Four: you are {'X' if human_seat == 0 else 'O'}; X starts.")
    if checkpoint is not None:
        print(f"Opponent: checkpoint policy ({policy_mode or 'sampled'}).")
    print("Drop a disc in columns 1-7. Four in a row wins. Enter quit to stop.")
    try:
        result = run_match(
            game,
            agents,
            max_steps=42,
            engine_seed=engine_seed,
            agent_seeds=(None, agent_seed) if human_seat == 0 else (agent_seed, None),
        )
    except (PlayCancelled, EOFError, KeyboardInterrupt):
        print("\nGame stopped.")
        return
    print("Final position:")
    print(game.render(human_seat))
    if result.truncated:
        print("Game stopped at the decision limit.")
        return
    outcome = connect_four_outcome(game)
    print(
        "Draw."
        if outcome.draw
        else "You win!"
        if human_seat in outcome.winners
        else "AI wins."
    )
