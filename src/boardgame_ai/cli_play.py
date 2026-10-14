"""Terminal presentation adapters for human-versus-agent Tic-Tac-Toe."""

from boardgame_ai.agents import Agent, RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.core import Action, Observation
from boardgame_ai.games.tic_tac_toe import (
    TicTacToeEngine,
    TicTacToeObservation,
    render_board,
)
from boardgame_ai.games.tic_tac_toe.evaluation import tic_tac_toe_outcome
from boardgame_ai.games.tic_tac_toe.replay import (
    ReplayStatus,
    TicTacToeRecorder,
    load_replay,
    replay_positions,
    save_replay,
)
from boardgame_ai.simulation import run_match


class PlayCancelled(Exception):
    """The human requested cancellation before submitting another action."""


def render_choices(
    observation: Observation[TicTacToeObservation],
    legal_actions: tuple[Action, ...],
) -> str:
    """Render public board beside supplied available cell numbers."""
    available = {action.parameters[0][1] for action in legal_actions}
    rows = [
        " | ".join(
            str(cell) if cell in available else "-" for cell in range(start, start + 3)
        )
        for start in (0, 3, 6)
    ]
    numbered = "\n---------\n".join(rows).splitlines()
    board = render_board(observation).splitlines()
    return "Game board    Available cells\n" + "\n".join(
        f"{left}    {right}" for left, right in zip(board, numbered, strict=True)
    )


class HumanTicTacToeAgent(Agent[TicTacToeObservation]):
    """Read a terminal choice using only detached player decision inputs.

    Presentation adapter, not a training baseline. No engine, RNG, or memory is
    retained. The inherited reset is a no-op.
    """

    def select_action(
        self,
        observation: Observation[TicTacToeObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        if not legal_actions:
            raise ValueError("Human player requires legal choices")
        cells = {action.parameters[0][1]: action for action in legal_actions}
        while True:
            print(render_choices(observation, legal_actions))
            print("Available cells: " + ", ".join(str(cell) for cell in cells))
            text = input("Choose cell 0-8 (or quit): ").strip()
            if text.lower() in ("q", "quit", "exit"):
                raise PlayCancelled
            try:
                cell = int(text)
            except ValueError:
                print("Enter a cell number from 0 to 8, or quit.")
                continue
            if cell not in cells:
                print("That cell is unavailable. Choose an available cell.")
                continue
            return cells[cell]


class _ReportingAgent(Agent[TicTacToeObservation]):
    """Print the opponent's chosen public placement without engine access."""

    def __init__(self, policy: Agent[TicTacToeObservation]) -> None:
        self._policy = policy

    def reset(self, seed: int | None = None) -> None:
        self._policy.reset(seed)

    def select_action(
        self,
        observation: Observation[TicTacToeObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        action = self._policy.select_action(observation, legal_actions)
        mark = "X" if observation.player_id == 0 else "O"
        print(f"AI ({mark}) chooses cell {action.parameters[0][1]}.")
        return action


def _make_opponent(name: str) -> Agent[TicTacToeObservation]:
    if name == "random":
        return RandomAgent()
    if name == "heuristic":
        return TicTacToeHeuristicAgent()
    raise ValueError("Unknown opponent")


def play_tic_tac_toe(
    *,
    opponent: str,
    engine_seed: int | None,
    agent_seed: int | None,
    starts: str = "human",
    replay_path: str | None = None,
) -> None:
    """Run one CLI session through the standard bounded match runner."""
    if starts not in ("human", "ai"):
        raise ValueError("Starter must be human or ai")
    human_seat = 0 if starts == "human" else 1
    human = "X" if human_seat == 0 else "O"
    starting_player = 0
    game = TicTacToeEngine(starting_player=starting_player)
    human_agent = HumanTicTacToeAgent()
    ai_agent = _ReportingAgent(_make_opponent(opponent))
    agents: tuple[Agent[TicTacToeObservation], ...] = (
        (human_agent, ai_agent) if human_seat == 0 else (ai_agent, human_agent)
    )
    seeds = (None, agent_seed) if human_seat == 0 else (agent_seed, None)
    first_mark = "X" if starting_player == 0 else "O"
    print(f"Tic-Tac-Toe: you are {human}; opponent is {opponent}. {first_mark} starts.")
    recorder = TicTacToeRecorder() if replay_path is not None else None
    try:
        result = run_match(
            game,
            agents,
            max_steps=9,
            engine_seed=engine_seed,
            agent_seeds=seeds,
            on_step=recorder.record if recorder is not None else None,
        )
    except (PlayCancelled, EOFError, KeyboardInterrupt):
        print("\nGame stopped.")
        if recorder is not None and replay_path is not None:
            save_replay(recorder.finish(starting_player, "cancelled"), replay_path)
            print(f"Replay saved: {replay_path}")
        return
    if recorder is not None and replay_path is not None:
        status: ReplayStatus = "completed" if result.terminated else "truncated"
        save_replay(recorder.finish(starting_player, status), replay_path)
        print(f"Replay saved: {replay_path}")
    print("Final board:")
    print(render_choices(game.observe(human_seat), game.legal_actions(human_seat)))
    if result.truncated:
        print("Game stopped at the decision limit.")
        return
    outcome = tic_tac_toe_outcome(game)
    if outcome.draw:
        print("Draw.")
    elif human_seat in outcome.winners:
        print("You win!")
    else:
        print("AI wins.")


def inspect_replay(path: str) -> None:
    """Validate before printing, then show the initial board and every placement."""
    replay = load_replay(path)
    positions = replay_positions(replay)
    print("Replay: Tic-Tac-Toe")
    print(f"Starting mark: {'X' if replay.starting_player == 0 else 'O'}")
    print("Initial board:")
    print(render_board(positions[0]))
    for index, (move, position) in enumerate(
        zip(replay.moves, positions[1:], strict=True), 1
    ):
        mark = "X" if move.player == 0 else "O"
        print(f"Move {index}: {mark} places at cell {move.cell}")
        print(render_board(position))
    print(f"Status: {replay.status}; placements: {len(replay.moves)}")
    if replay.status == "completed":
        game = TicTacToeEngine(starting_player=replay.starting_player)
        game.reset(0)
        for move in replay.moves:
            game.step(Action(kind="place", parameters=(("cell", move.cell),)))
        outcome = tic_tac_toe_outcome(game)
        print(
            "Draw."
            if outcome.draw
            else f"{'X' if outcome.winners[0] == 0 else 'O'} wins."
        )
