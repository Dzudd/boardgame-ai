"""Bounded vector-return PUCT with private namespaced exploration."""

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite, sqrt
from random import Random

from boardgame_ai.agents.base import Agent
from boardgame_ai.core import Action, Observation, create_rng
from boardgame_ai.games.tic_tac_toe.observation import TicTacToeObservation
from boardgame_ai.search.simulator import PublicTicTacToeSimulator

type Evaluator = Callable[
    [Observation[TicTacToeObservation], tuple[Action, ...]],
    tuple[tuple[float, ...], float],
]
type Vector = tuple[float, float]


def backup(reward: Vector, continuation: Vector) -> Vector:
    return reward[0] + continuation[0], reward[1] + continuation[1]


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 32
    max_nodes: int = 289
    max_depth: int = 9
    exploration: bool = False

    def __post_init__(self) -> None:
        for value in (self.simulations, self.max_nodes, self.max_depth):
            if type(value) is not int or value < 1:
                raise ValueError("Search budgets must be positive integers")


@dataclass
class Edge:
    action: Action
    prior: float
    visits: int = 0
    total: Vector = (0.0, 0.0)
    child: "Node | None" = None


@dataclass
class Node:
    state: PublicTicTacToeSimulator
    edges: list[Edge] = field(default_factory=list)


@dataclass
class SearchResult:
    action: Action
    root: Node
    evaluator_calls: int
    nodes: int
    depth: int
    depth_cutoffs: int
    node_cutoffs: int


@dataclass(frozen=True)
class SimulationTrace:
    """Optional public diagnostic record; never a training sample."""

    index: int
    actions: tuple[Action, ...]
    rewards: tuple[Vector, ...]
    leaf: PublicTicTacToeSimulator
    continuation: Vector
    returns: tuple[Vector, ...]
    stop: str


class SearchAgent(Agent[TicTacToeObservation]):
    def __init__(
        self,
        evaluator: Evaluator,
        config: SearchConfig | None = None,
        *,
        on_simulation: Callable[[SimulationTrace], None] | None = None,
    ) -> None:
        self.evaluator, self.config = evaluator, config or SearchConfig()
        self.on_simulation = on_simulation
        self.last_result: SearchResult | None = None
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.seed = create_rng(seed).getrandbits(256)
        self.last_result = None

    def _rng(self, seat: int, decision: int, purpose: str) -> Random:
        text = f"boardgame-ai/search/v1:{self.seed}:{seat}:{decision}:{purpose}"
        return Random(int.from_bytes(sha256(text.encode()).digest(), "big"))

    def select_action(
        self,
        observation: Observation[TicTacToeObservation],
        legal_actions: tuple[Action, ...],
    ) -> Action:
        state = PublicTicTacToeSimulator.from_observation(observation, legal_actions)
        if state.actor is None:
            raise ValueError("Search requires a live root")
        decision = 9 - state.board.count(None)
        root = Node(state)
        calls, nodes, depth, depth_cuts, node_cuts = 0, 1, 0, 0, 0

        def evaluate(state: PublicTicTacToeSimulator, node: Node | None) -> Vector:
            nonlocal calls
            legal = state.legal_actions()
            probabilities, value = self.evaluator(state.observe(), legal)
            calls += 1
            if (
                len(probabilities) != len(legal)
                or not isfinite(value)
                or any(not isfinite(p) or p < 0 for p in probabilities)
            ):
                raise ValueError("Invalid evaluator output")
            priors = [max(p, 1e-8) for p in probabilities]
            scale = sum(priors)
            if not isfinite(scale):
                raise ValueError("Invalid evaluator prior sum")
            if node is not None:
                node.edges = [
                    Edge(a, p / scale) for a, p in zip(legal, priors, strict=True)
                ]
            v = max(-1.0, min(1.0, value))
            return (v, -v) if state.actor == 0 else (-v, v)

        evaluate(state, root)
        if self.config.exploration:
            rng = self._rng(observation.player_id, decision, "root-noise")
            noise = [rng.gammavariate(1, 1) for _ in root.edges]
            total = sum(noise)
            for edge, n in zip(root.edges, noise, strict=True):
                edge.prior = 0.75 * edge.prior + 0.25 * n / total
        for index in range(self.config.simulations):
            node = root
            path: list[tuple[Edge, Vector]] = []
            d = 0
            while True:
                actor = node.state.actor
                assert actor is not None
                visits = sum(e.visits for e in node.edges)
                edge = max(
                    node.edges,
                    key=lambda e: (
                        (e.total[actor] / e.visits if e.visits else 0)
                        + 1.5 * e.prior * sqrt(1 + visits) / (1 + e.visits)
                    ),
                )
                child_state, reward = node.state.step(edge.action)
                path.append((edge, reward))
                d += 1
                depth = max(depth, d)
                if child_state.actor is None:
                    continuation = (0.0, 0.0)
                    stop = "terminal"
                    break
                if d >= self.config.max_depth:
                    stop = "depth"
                    depth_cuts += 1
                    continuation = evaluate(child_state, None)
                    break
                if edge.child is None:
                    if nodes >= self.config.max_nodes:
                        stop = "nodes"
                        node_cuts += 1
                        continuation = evaluate(child_state, None)
                    else:
                        stop = "new-leaf"
                        edge.child = Node(child_state)
                        nodes += 1
                        continuation = evaluate(child_state, edge.child)
                    break
                node = edge.child
            leaf_continuation = continuation
            backed: list[Vector] = []
            for edge, reward in reversed(path):
                continuation = backup(reward, continuation)
                if self.on_simulation is not None:
                    backed.append(continuation)
                edge.visits += 1
                edge.total = backup(edge.total, continuation)
            if self.on_simulation is not None:
                self.on_simulation(
                    SimulationTrace(
                        index=index,
                        actions=tuple(edge.action for edge, _ in path),
                        rewards=tuple(reward for _, reward in path),
                        leaf=child_state,
                        continuation=leaf_continuation,
                        returns=tuple(reversed(backed)),
                        stop=stop,
                    )
                )
        if self.config.exploration:
            chosen = self._rng(
                observation.player_id, decision, "action-sampling"
            ).choices(root.edges, weights=[e.visits for e in root.edges])[0]
        else:
            chosen = max(root.edges, key=lambda e: e.visits)
        action = next(a for a in legal_actions if a == chosen.action)
        self.last_result = SearchResult(
            action, root, calls, nodes, depth, depth_cuts, node_cuts
        )
        return action
