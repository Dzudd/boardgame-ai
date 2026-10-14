# General-Purpose Board Game AI — Project Plan

**Version:** 0.2<br>
**Status:** Phase 3 complete — next: Phase 4, Task 1<br>
**Primary language:** Python<br>
**Initial ML framework:** PyTorch

## 1. Vision and goals

Build a reusable reinforcement-learning platform that can learn to play **multiple board games**, including games with:

- Hidden information (private hands, unknown decks, unrevealed objectives)
- Random events (drawing cards, dice, setup randomness)
- Multiple decisions within a single player's turn
- Different action spaces, boards, scoring systems, and player counts
- Human-versus-AI play after training

Long-term research goal: test whether training across many games helps a shared model learn **new games faster** than a model trained from scratch. This is distinct from merely becoming strong at a fixed set of 100 games.

## 2. Core principles and decisions

| Area | Initial decision | Notes |
|---|---|---|
| Game rules | Separate authoritative game engine | The agent never directly changes game state. |
| Game state | Engine holds full state | Includes hidden hands, deck order, positions, resources, phases, etc. |
| Observations | Player-specific filtered views | Never expose inaccessible private information. |
| Actions | One atomic decision per step | A turn may contain multiple steps by the same player. |
| Legal moves | Query on demand | Do not enumerate entire future action sequences. |
| Randomness | Resolved by engine | Agent learns a draw outcome only after drawing. |
| Initial learning | PPO with legal-action masking | Start feed-forward; add recurrence for partial observability. |
| Opponents | Random, heuristic, self-play and historical snapshots | Use fixed baselines for evaluation. |
| Multiple games | Start with separate models | Compare against a shared-backbone approach later. |
| Long-term model | Shared backbone + game-specific components | Candidate-action scoring can handle variable legal-action counts. |
| Planning | Optional later extension | Explore information-set search and MuZero-inspired models. |

These are working hypotheses, not irrevocable design commitments.

## 3. Important conceptual distinctions

### Game engine versus agent

The **engine** knows and enforces all rules. It stores every card and resource, validates actions, applies effects, resolves randomness, detects termination, and computes rewards.

The **agent** sees only its observation and its available legal actions, then selects one. It can learn strategies without receiving a rule simulator inside its neural network.

The engine is required **both during training and after training**. A trained agent still relies on the engine to execute real moves, maintain authoritative state, and determine legal actions.

### Decision versus turn versus round

- **Decision/step:** one atomic choice, such as discarding a card.
- **Turn:** a sequence of decisions belonging to a player, potentially with interruptions or responses.
- **Round:** a game-defined sequence of turns or phases.

A turn does not have to end after each `step`. The engine exposes the **next decision-maker** and **current phase**.

### Hidden information and public deductions

The engine may know an opponent holds Card E, but the agent should only see Card E if that knowledge is legitimately available. It may receive public card counts, played cards, own hand, opponent hand sizes, and observation history. If four of five unique cards have been publicly played, the agent can deduce the fifth card without cheating.

A **belief state** is a probability distribution over hidden states compatible with observations. We can initially provide public remaining-card counts, later add learned belief estimation. Do not give the model privileged probabilities based on hidden engine data.

### Chance and planning

An agent can choose `DRAW_CARD` without knowing the result. The engine then samples the card and exposes the resulting observation before the next decision. Do not represent an action as `draw A, then play A` if A was unknown at decision time.

Ordinary MuZero's deterministic learned dynamics do not automatically solve chance events or imperfect-information planning. Searching separate fully specified sampled worlds can also cause **strategy fusion** (acting as if the player knows hidden information). Advanced search must respect information sets and uncertainty.

## 4. System architecture

```text
Game-specific authoritative engines
    ├── complete private/public state
    ├── rules and chance outcomes
    └── legal action generation
                 │
                 ▼
Standardized environment interface
    ├── observe(player_id)
    ├── legal_actions(player_id)
    ├── step(action)
    └── rewards / termination / next actor
                 │
                 ▼
Agent
    ├── observation encoder
    ├── optional recurrent memory
    ├── policy / value heads
    └── candidate-action scorer
                 │
                 ▼
Training and evaluation
    ├── rollout collection / self-play
    ├── PPO updates
    ├── opponent pool
    └── metrics and checkpoints
```

For a future multi-game model, experiment with **game-specific encoders → shared backbone → game-specific or candidate-action policy/value heads**. Shared layers may transfer useful patterns, but negative transfer is possible and must be measured.

## 5. Proposed environment API

Illustrative Python interface (not yet a finalized contract):

```python
class GameEngine:
    def reset(self, seed=None):
        """Initialize authoritative state."""
        ...

    def observe(self, player_id):
        """Return only information visible to this player."""
        ...

    def legal_actions(self, player_id):
        """Return legal atomic actions at this decision point."""
        ...

    def current_player(self):
        """Return player responsible for the next decision."""
        ...

    def step(self, action):
        """Validate and execute one action; resolve required chance effects."""
        ...

    def is_terminal(self):
        ...

    def get_rewards(self):
        ...

    def clone(self):
        """Deep-copy state for tests and optional future simulation."""
        ...
```

Additional planned concepts:

- `GameState`: complete authoritative state, not normally passed to agents.
- `Observation`: filtered, player-specific view, including phase and relevant public history.
- `Action`: typed, serializable atomic command; no hidden outcomes embedded.
- `StepResult`: next actor, termination, rewards, and public/private events.
- `Agent`: `select_action(observation, legal_actions)` plus optional recurrent state.
- `GameSpec`: game metadata and action/observation encoder configuration.

**Security/information boundary:** `get_full_state()` is for engine internals, testing, and authorized debugging—not the agent-facing API. Training critics must also avoid privileged hidden-state leakage unless explicitly researching centralized training with carefully separated execution inputs.

## 6. Runtime gameplay

```python
while not game.is_terminal():
    player = game.current_player()
    observation = game.observe(player)
    legal = game.legal_actions(player)

    action = agents[player].select_action(observation, legal)
    game.step(action)
```

The same player may act repeatedly. Human actions go through exactly the same validation and execution path. For games with automatic chance events or reaction windows, the engine must resolve them and expose the correct next actor before returning control.

## 7. Algorithm choices

| Approach | Advantages | Main challenges | Role |
|---|---|---|---|
| PPO | Simple model-free training; handles atomic decisions | Sample efficiency; long-horizon credit assignment | Initial learner |
| Recurrent PPO (LSTM/GRU) | Uses history in partially observed games | Sequence training, memory management | First hidden-information learner |
| AlphaZero-style MCTS | Powerful planning with a simulator | Must adapt to chance, hidden information, same-player actions | Future baseline |
| MuZero | Learns latent dynamics for planning | High complexity; imperfect information/chance not solved automatically | Research extension |
| ISMCTS / information-set search | Explicit imperfect-information planning | Sampling quality, strategy fusion and opponent modeling | Advanced search |
| CFR / ReBeL-style methods | Strong conceptual match for some imperfect-information games | Complex, often game-structure dependent | Research option |

**Initial recommendation:** build a simple PPO agent and compare it with random/heuristic baselines; add recurrent PPO for hidden information; only then invest in search or learned dynamics.

## 8. Development phases and actionable tasks

### Phase 1 — Project foundation

**Deliverable:** a tested Python package and minimal engine contract.

- [x] Create repository, virtual environment, dependency management, linting and formatting.
- [x] Add PyTorch, NumPy, pytest and minimal CLI tooling.
- [x] Define `GameState`, `Observation`, `Action`, `StepResult` and `GameSpec`.
- [x] Define the abstract `GameEngine` and `Agent` interfaces.
- [x] Establish player visibility and information-boundary rules.
- [x] Define illegal-action handling and deterministic random seeds.
- [x] Add unit tests and CI.

**Completion:** all Phase 1 tasks are implemented. Test-only deterministic and
hidden-card engines exercise the contract; see `INFORMATION_BOUNDARIES.md` for
visibility, failure, and randomness policies. CI is configured for GitHub Actions;
hosted execution runs in the private
[GitHub repository](https://github.com/Dzudd/boardgame-ai/actions) on pushes to `master`.

**Acceptance criteria:** a dummy game can reset, expose observations/actions, execute steps, terminate, and pass interface tests.

### Phase 2 — First deterministic game

**Deliverable:** complete Tic-Tac-Toe engine.

- [x] Implement initial board state and reset.
- [x] Implement legal actions and validation.
- [x] Implement state transitions and player switching.
- [x] Implement terminal detection and rewards.
- [x] Implement observations and board rendering.
- [x] Test wins, draws, illegal moves and deterministic resets.
- [x] Implement `clone()` and test independence of cloned states.

**Completion:** the concrete Tic-Tac-Toe engine implements the full engine
contract, public observations, text rendering, and independent cloning. Tests
audit all 5,478 distinct reachable positions and exercise the shared engine and
rejection conformance helpers. All 152 tests and local development checks pass.

**Acceptance criteria:** full games complete correctly with no invalid transitions.

### Phase 3 — Simulation and baselines

**Deliverable:** automated games and benchmark results.

- [x] Implement `RandomAgent`.
- [x] Implement a simple heuristic agent.
- [x] Build reusable game loop and match runner.
- [x] Run batches of matches with reproducible seeds.
- [x] Record wins, draws, scores, errors and game lengths.
- [x] Add a command-line human-versus-agent mode.
- [x] Save and inspect game replays/logs.

**Acceptance criteria:** agents can play thousands of valid games automatically and benchmark results are reproducible.

**Task 1 complete:** the generic random baseline selects uniformly from supplied
legal choices using agent-owned seeded randomness. Reset, failure, visibility,
and complete-game replay tests pass. Match runners and benchmarks remain pending.

**Task 2 complete:** `TicTacToeHeuristicAgent` uses the public board to win,
block, then prefer center/corners/edges with deterministic ties. Tactical and
complete-game tests pass; this simple baseline does not guarantee optimal play.

**Task 3 complete:** generic `play_game` and `run_match` provide bounded single-game
orchestration, detached actor inputs, separate lifecycle seeds, and a trusted
reward/termination summary. Batch scheduling, metrics, and replays remain pending.

**Task 4 complete:** `run_batch` derives separate engine and per-seat seeds from
an integer batch seed and absolute match index, retaining seeds with each result.
Three 1,000-match baseline batches and their reruns completed identically (6,000
executions). Metric aggregation, human play, and replay logging remain pending.

**Task 5 complete:** trusted evaluation records and JSON export report explicit
game-defined outcomes/scores, fixed error categories, lengths, and aggregates.
Tic-Tac-Toe benchmark results are in `BASELINE_BENCHMARKS.md`. Human CLI play and
replay logging remain before Phase 3 completion.

**Task 6 complete:** `boardgame-ai play` runs human-versus-random/heuristic
Tic-Tac-Toe through the standard match runner, with X/O selection, separate seeds,
public board presentation, input retries, final outcomes, and clean cancellation.
Saving and inspecting replays/logs remains Task 7.

**Task 6 follow-up:** starter selection is independent of human mark, and available
cell numbers are rendered beside the public board on every prompt/final display.
X-first remains the default; optional O-first is engine-owned configuration.

**Task 7 complete:** opt-in versioned public Tic-Tac-Toe action logs can be
saved from CLI or automated matches and validated/reconstructed for inspection.
CLI player choices are now human/ai, with starter assigned X; --game explicitly
selects Tic-Tac-Toe. Hidden-game replay adapters remain game-specific future work.

**Phase 3 complete:** all seven tasks are implemented. Baseline benchmarks run
thousands of valid games and reproduce, metrics are recorded, humans can play,
and public game logs round-trip through engine validation. All 360 tests and
local development checks pass. The private GitHub remote is configured and hosted
CI runs on pushes; see [Actions](https://github.com/Dzudd/boardgame-ai/actions).

### Phase 4 — First learning agent

**Deliverable:** a trained agent that outperforms a random baseline on the first game.

- [ ] Encode observations as PyTorch tensors.
- [ ] Encode legal actions and implement action masking.
- [ ] Implement policy and value networks.
- [ ] Implement trajectory collection and advantage estimation (GAE).
- [ ] Implement PPO losses, optimization and checkpointing.
- [ ] Train with self-play and/or baseline opponents.
- [ ] Add fixed evaluation opponents and held-out random seeds.
- [ ] Plot training curves and compare against baselines.

**Acceptance criteria:** a reproducible training run produces measurable improvement over a random agent, evaluated independently of training opponents.

### Phase 5 — Complex mechanics and partial observability

**Deliverable:** a small custom card/worker-placement test game.

- [ ] Implement multiple atomic decisions within one turn.
- [ ] Add explicit decision phases and turn/round transitions.
- [ ] Implement hidden hands, deck, discard pile and public card counts.
- [ ] Implement drawing cards and other chance events.
- [ ] Implement filtered observations and player-specific history.
- [ ] Add tests ensuring hidden state never leaks.
- [ ] Add recurrent agent memory and recurrent PPO training.
- [ ] Test edge cases: empty decks, skipped phases, interrupts, extra turns.

**Acceptance criteria:** the agent can play complete partially observed games; no action reveals an unresolved random outcome; same-player consecutive actions work.

### Phase 6 — Multi-game learning

**Deliverable:** independent and shared agents trained on multiple games.

- [ ] Add at least two more games using the same engine interface.
- [ ] Implement game-specific observation encoders.
- [ ] Implement candidate-action encoding and scoring.
- [ ] Build a shared Transformer or recurrent backbone.
- [ ] Implement game-specific policy/value adapters where necessary.
- [ ] Create a multi-game training sampler and normalize/track rewards carefully.
- [ ] Compare separate models versus a shared model at matched compute budgets.
- [ ] Test transfer: train on games A/B, then measure learning speed on game C.
- [ ] Check for negative transfer and catastrophic forgetting.

**Acceptance criteria:** quantitative comparison of per-game strength, training cost, and adaptation speed on a held-out game.

### Phase 7 — Advanced search and research

**Deliverable:** experimentally validated planning enhancements.

- [ ] Implement explicit or learned hidden-state belief estimates.
- [ ] Validate belief estimates against known simulated outcomes without leaking them at inference.
- [ ] Implement information-set-aware search baseline.
- [ ] Define evaluator perspective and return semantics separately from action selection.
- [ ] Add bounded rollout evaluation and engine-based MCTS on a perfect-information game.
- [ ] Compare heuristic, rollout, neural, and hybrid evaluation against fixed baselines.
- [ ] Integrate policy/value guidance into search.
- [ ] Explore MuZero-style learned latent dynamics as a separate experiment.
- [ ] Test chance-event modeling and same-player search backups.
- [ ] Benchmark planning strength versus computation and latency.

**Acceptance criteria:** advanced methods improve defined metrics relative to a simpler baseline under comparable budgets.

### Phase 8 — Productization and human play

**Deliverable:** reliable human-versus-AI experience and tournament tools.

- [ ] Provide CLI or web UI for human play.
- [ ] Add selectable games, agents and difficulty levels.
- [ ] Save/load games and replay moves.
- [ ] Add tournament runner and ratings or matchup statistics.
- [ ] Profile inference latency and search budgets.
- [ ] Document how to add a new game.

**Acceptance criteria:** a user can select a supported game, play against a trained agent, and finish a game without manual state edits.

## 9. Initial repository layout

```text
boardgame-ai/
├── README.md
├── PROJECT_PLAN.md
├── ARCHITECTURE.md
├── PROGRESS.md
├── pyproject.toml
├── core/
│   ├── game.py
│   ├── state.py
│   ├── action.py
│   └── observation.py
├── games/
│   ├── tic_tac_toe/
│   └── card_game/
├── agents/
│   ├── base.py
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   └── neural_agent.py
├── models/
│   ├── encoders/
│   ├── policy.py
│   ├── value.py
│   └── backbone.py
├── training/
│   ├── ppo.py
│   ├── self_play.py
│   └── evaluation.py
├── configs/
├── tests/
├── scripts/
└── main.py
```

Keep generated checkpoints and large datasets out of Git by default.

## 10. Evaluation strategy

Measure more than raw self-play win rate:

- **Game correctness:** legal actions, state invariants, reproducibility, no hidden-state leakage.
- **Agent quality:** win/draw rate, normalized score, performance against fixed and historical opponents.
- **Training efficiency:** environment steps, wall-clock time, hardware use, learning stability.
- **Multi-game tradeoffs:** per-game performance versus separate specialist agents, forgetting, interference.
- **Transfer:** learning curve and sample efficiency on held-out game C after pretraining on A/B.
- **Inference cost:** action selection latency, memory, optional search simulations.

Evaluate multiple seeds and report uncertainty where practical.

## 11. Risks and open questions

1. **Observation/action standardization:** wildly different game structures may need game-specific encoders and action schemas.
2. **Legal actions in imagined states:** real engine legality applies to real or fully simulated states, not automatically to MuZero's latent states.
3. **Imperfect information:** belief-state reasoning and information-set planning are nontrivial.
4. **Same-player actions:** value backups in tree search must track the player perspective, not flip signs after every step.
5. **Multiplayer objectives:** zero-sum, general-sum and cooperative games require different value/reward treatments.
6. **Negative transfer:** one shared model may underperform specialists; test rather than assume benefits.
7. **Training cost:** scaling to 100 games is a long-term ambition, not an initial milestone.
8. **New-game generalization:** decide whether the eventual goal is strong play on known games, rapid adaptation to unseen games, or both.
9. **Game-rule access:** some games may be proprietary; implement rules and assets only with appropriate rights.

## 12. Position evaluation and forward planning (new in v0.2)

### Three separate responsibilities

1. **Authoritative game engine:** knows the rules and true state, validates and applies one atomic action, resolves chance, exposes player-specific observations and legal actions.
2. **Position evaluator:** estimates the desirability of a state for a particular player. It can be heuristic, rollout-based, neural, or hybrid. It does **not** necessarily choose an action.
3. **Search/planner:** chooses which candidate actions and future states to explore, decides how deep or how many simulations to run, and uses evaluations to select an action.

**Key rule:** the engine can calculate one step forward, but should not be responsible for enumerating the entire future game tree. Search belongs to an agent-side module. The engine is the source of truth for actual play even after training.

### Engine-based MCTS versus MuZero

| Dimension | Engine-based search (MCTS / ISMCTS) | MuZero-style search |
|---|---|---|
| Imagined transition | Clone/simulate using actual game rules | Learned latent dynamics network |
| Position estimate | Rollout, heuristic or neural value | Neural prediction/value head |
| Action suggestions | Search statistics, optionally neural policy | Neural policy prior plus MCTS |
| Rule knowledge during search | Simulator has rules | Neural search does not call rule simulator for imagined transitions |
| Hidden information | Needs beliefs and information-set-aware planning | Needs extra machinery beyond original MuZero |
| Chance outcomes | Simulator samples actual game randomness | Needs chance-aware/stochastic model or another approximation |

For future search work, **start with engine-based forward simulation** because we already need a complete rules engine. Keep MuZero as an optional research experiment, not a prerequisite. A learned evaluator can be added before any learned dynamics model. This preference applies to search implementation; it does not replace the current PPO-first Phase 4 roadmap.

### Forward-simulation example

Suppose a player can collect wood, then build, while an opponent may respond. A planner can:

1. Obtain the acting player's observation and legal actions.
2. Construct a simulation state consistent with that player's knowledge (trivial in perfect-information games; harder with hidden information).
3. Clone the state and apply `COLLECT_WOOD`.
4. Ask the simulated engine for the **next** legal actions, possibly for the same player.
5. Explore `BUILD` or other legal decisions, then opponent responses.
6. Stop after a simulation budget or depth limit and evaluate the resulting position.
7. Repeat selectively; choose a real action without committing any simulated changes to the authoritative game.

A turn can contain many decision steps. Search must track the current decision-maker and value perspective rather than assuming players alternate after every action. Multiplayer games may require vector-valued outcomes.

### Suggested engine capabilities

```python
class GameEngine:
    def observe(self, player_id): ...
    def legal_actions(self, player_id): ...
    def step(self, action): ...
    def clone(self): ...
    def current_player(self): ...
    def is_terminal(self): ...
    def get_rewards(self): ...

    # Advanced extension; not required in the first implementation:
    def sample_hidden_state(self, player_id, history, rng): ...
```

- `clone()` creates an **independent** copy; a simulation must never mutate the real game.
- Deterministic, seed-controlled randomness supports reproducible tests. Cloning should define whether RNG state is copied or independently seeded.
- The implemented clone contract copies current RNG state into an independently owned generator; it does not reseed on clone. Alternative simulation streams must be an explicit future design, not a change to existing cloning behavior.
- `sample_hidden_state` must sample only worlds compatible with the player's information and public/private observation history. It must **not** simply expose or copy the actual opponent hand.
- Legal-action queries are on demand, including at simulated intermediate decision points; no need to precompute every possible full-turn sequence.
- Game-specific rules remain separate from generic planner code.

These are proposed future capabilities, not additions to the current Agent or
GameEngine interfaces. Agents currently receive only detached observations and
legal actions. A future simulation context must be constructed from permitted
player information through a separately reviewed interface; an authoritative
engine clone is never passed directly to an agent, even as a convenience for search.

### Evaluator interfaces and experiments

A first implementation can offer:

- **Heuristic evaluator:** hand-designed material/resources/score estimates; useful for baselines and debugging.
- **Monte Carlo rollout evaluator:** play simulated continuations with a specified rollout policy; estimate expected final outcomes. Accuracy depends on the rollout policy and hidden-state assumptions.
- **Neural value evaluator:** predict expected final reward from a player observation/history; fast at inference after training.
- **Hybrid evaluator:** combine learned value with selective search or rollouts, benchmarking added strength against computation cost.

A common interface should specify **player perspective**, return semantics (e.g., expected final rewards, preferably a vector for multiplayer games), and whether the evaluator consumes an observation/history or a simulated state. Do not force rollout evaluators to operate on observations alone.

```python
class ObservationValueEvaluator:
    def evaluate(self, observation, history, player_id):
        """Estimate expected final outcome from this player's information."""
        ...

class SimulationEvaluator:
    def evaluate(self, simulation_context, player_id, budget):
        """Estimate outcome using safe simulated continuations."""
        ...
```

Keep **planning** separate from **evaluation**: MCTS selects branches; a value function scores leaves. Do not automatically replace the game's true terminal reward with heuristic scores in PPO training, since that may reward the wrong objective.

### Hidden-information safeguards

In a game with unknown cards, an engine clone of the **actual** full state would allow a search algorithm to cheat. Instead, simulations need possible hidden states sampled from the player's belief, with correct public counts, prior reveals, and event history. Even then, naively searching each sampled complete world can lead to **strategy fusion**: selecting later actions as if hidden cards were known. Information-set-aware methods, opponent modeling, and correct chance handling are advanced research tasks, not features to assume a basic MCTS implementation already solves.

### Recommended implementation order

The following is an optional evaluator/search experiment track imported with
v0.2, not a renumbering of the development phases. Foundation and the simple-game
engine are complete; position evaluators, rollout search, and MCTS are not yet
implemented. The existing Tic-Tac-Toe heuristic is an action-selection policy,
and GameOutcome evaluation classifies terminal results; neither is a general
position-value evaluator. Phase 4, Task 1 remains the next implementation task.

1. **Foundation:** implement the engine interface, independent cloning, deterministic tests, and legal-action generation at each decision point.
2. **Simple game:** test forward simulations and a heuristic evaluator on Tic-Tac-Toe or another fully observable deterministic game.
3. **Baseline search:** add bounded rollout evaluation, then MCTS; compare to random and heuristic agents.
4. **Neural baseline:** implement PPO with a policy/value network, compare playing with and without optional search.
5. **Complex game:** add a tiny hidden-card game with multi-step turns and chance events; verify no hidden information leaks.
6. **Advanced search:** introduce belief-state sampling and information-set-aware planning; benchmark strength and latency.
7. **MuZero research:** optionally implement learned latent dynamics and compare them against real-engine simulations on controlled games.
8. **Multi-game scale:** evaluate independent agents versus shared-backbone models and whether search helps transfer to new games.

**Success criteria:** simulations never alter live state; action legality is respected; no player receives inaccessible hidden information; repeated seeded tests reproduce results; evaluator values use consistent player perspectives; search strength and latency are measured against fixed baselines.

## 13. Immediate next steps

Phases 1–3 are complete: the package, shared contracts, tested Tic-Tac-Toe
engine, simulation/baselines, benchmarks, human play, and replays are implemented.

1. Encode observations as PyTorch tensors (Phase 4, Task 1).
2. Implement legal-action encoding/masking and policy/value networks in later Phase 4 tasks.
3. Build and test the PPO training pipeline one task at a time.
4. Move to the minimal hidden-card game after the deterministic pipeline works.

**First milestone:** one fully working deterministic game, one reproducible training pipeline, and an agent that reliably improves over a random baseline.

---

### Project maintenance

Keep `BOARDGAME_AI_PROJECT_PLAN.md` for the roadmap, `ARCHITECTURE.md` for interface/algorithm decisions, and `PROGRESS.md` for current status and next tasks. Update these documents when the design changes, so future sessions can resume from repository state rather than chat history.
