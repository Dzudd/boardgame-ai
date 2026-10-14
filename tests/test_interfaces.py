"""Exercise the abstract contracts with a tiny test-only game and agent."""

from copy import deepcopy
from inspect import isabstract
from typing import Self

from boardgame_ai.agents import Agent
from boardgame_ai.core import (
    Action,
    GameEngine,
    GameNotReadyError,
    GameOverError,
    GameSpec,
    IllegalActionError,
    Observation,
    StepResult,
    create_rng,
)


class DummyGame(GameEngine[int]):
    """Three decisions: player 0 acts twice, then player 1 finishes the game."""

    def __init__(self) -> None:
        self._moves: list[Action] = []
        self._rewards = (0.0, 0.0)
        self._ready = False

    def _require_ready(self) -> None:
        if not self._ready:
            raise GameNotReadyError("Call reset before gameplay")

    @property
    def spec(self) -> GameSpec:
        return GameSpec(game_id="dummy", name="Dummy", min_players=2, max_players=2)

    @property
    def num_players(self) -> int:
        return 2

    def reset(self, seed: int | None = None) -> None:
        create_rng(seed)  # Validate the seed even though this game is deterministic.
        self._moves = []
        self._rewards = (0.0, 0.0)
        self._ready = True

    def _validate_player(self, player_id: int) -> None:
        self._require_ready()
        self.validate_player(player_id)

    def observe(self, player_id: int) -> Observation[int]:
        self._validate_player(player_id)
        return Observation(
            player_id=player_id,
            current_player=self.current_player(),
            phase="finished" if self.is_terminal() else "choose",
            data=len(self._moves),
        )

    def legal_actions(self, player_id: int) -> tuple[Action, ...]:
        self._validate_player(player_id)
        if player_id != self.current_player():
            return ()
        return (Action(kind="advance"),)

    def current_player(self) -> int | None:
        if self.is_terminal():
            return None
        return 0 if len(self._moves) < 2 else 1

    def step(self, action: Action) -> StepResult:
        player = self.current_player()
        if player is None:
            raise GameOverError("Game is over")
        if action not in self.legal_actions(player):
            raise IllegalActionError("Action is not legal")
        self._moves.append(action)
        self._rewards = (1.0, -1.0) if self.is_terminal() else (0.0, 0.0)
        return StepResult(
            next_player=self.current_player(),
            terminated=self.is_terminal(),
            rewards=self._rewards,
        )

    def is_terminal(self) -> bool:
        self._require_ready()
        return len(self._moves) == 3

    def get_rewards(self) -> tuple[float, ...]:
        self._require_ready()
        return self._rewards

    def clone(self) -> Self:
        self._require_ready()
        return deepcopy(self)


class FirstActionAgent(Agent[int]):
    """Test policy, not a production baseline agent."""

    def select_action(
        self, observation: Observation[int], legal_actions: tuple[Action, ...]
    ) -> Action:
        return legal_actions[0]


def test_interfaces_require_concrete_implementations() -> None:
    class IncompleteGame(GameEngine[int]):
        pass

    class IncompleteAgent(Agent[int]):
        pass

    assert isabstract(GameEngine)
    assert isabstract(Agent)
    assert isabstract(IncompleteGame)
    assert isabstract(IncompleteAgent)
    assert not isabstract(DummyGame)
    assert not isabstract(FirstActionAgent)


def test_gameplay_through_typed_interfaces() -> None:
    game: GameEngine[int] = DummyGame()
    agents: tuple[Agent[int], ...] = (FirstActionAgent(), FirstActionAgent())
    game.reset(seed=42)
    for agent in agents:
        agent.reset(seed=42)
    assert game.spec.min_players <= game.num_players <= game.spec.max_players
    assert game.get_rewards() == (0.0, 0.0)
    actors: list[int] = []

    # A bound catches accidentally nonterminating test implementations.
    for _ in range(4):
        if game.is_terminal():
            break
        player = game.current_player()
        assert player is not None
        actors.append(player)
        observation = game.observe(player)
        assert observation.player_id == observation.current_player == player
        assert game.legal_actions(1 - player) == ()
        action = agents[player].select_action(observation, game.legal_actions(player))
        result = game.step(action)
        assert result.next_player == game.current_player()
        assert result.terminated == game.is_terminal()
        assert result.rewards == game.get_rewards()
        assert len(result.rewards) == game.num_players

    assert actors == [0, 0, 1]
    assert game.is_terminal()
    assert game.current_player() is None
    assert game.get_rewards() == (1.0, -1.0)
    for player in range(game.num_players):
        assert game.legal_actions(player) == ()
        assert game.observe(player).current_player is None


def test_clone_is_independent_and_reset_clears_progress() -> None:
    game = DummyGame()
    game.reset()
    action = Action(kind="advance")
    game.step(action)
    cloned = game.clone()
    assert cloned is not game
    assert cloned.observe(0) == game.observe(0)

    cloned_result = cloned.step(action)
    assert game.observe(0).data == 1
    assert cloned.observe(0).data == 2
    assert game.step(action) == cloned_result
    assert game.observe(0) == cloned.observe(0)

    cloned.step(action)
    assert cloned.is_terminal()
    cloned.reset()
    assert not cloned.is_terminal()
    assert cloned.observe(0).data == 0
    assert cloned.get_rewards() == (0.0, 0.0)
    assert game.observe(0).data == 2
