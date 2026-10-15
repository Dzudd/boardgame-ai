# Architecture decisions

## 2026-10-15: Self-play methodology and Phase 4.5 (design only)

This decision supersedes earlier suggestions of baseline training opponents,
optional-only MuZero development, and proceeding directly to Phase 5.
Phase 4 remains complete under its original criterion; Phase 4.5 is planned,
starting with read-only tactical and learning-signal diagnosis. Its task list
and acceptance criteria live in BOARDGAME_AI_PROJECT_PLAN.md. No training code,
configuration, checkpoint, or experiment is changed by this documentation update.

- Training experience comes exclusively from self-play. The current and planned
  default is one latest policy frozen for both seats during collection. External
  engines, perfect-play teachers, baseline training opponents, expert labels,
  and historical opponent pools are excluded from the current plan. Historical
  self-generated replay is a separate future mechanism, not an opponent pool.
- Existing random/heuristic evaluations remain isolated measurements, never
  optimizer input or target generators. A perfect-play solver is not required.
  Tactical tests may inspect public immediate wins/blocks without turning those
  checks into supervised labels. Development evaluations may guide checkpoint
  selection; reserve fresh final tests and do not claim evaluation is uninvolved
  in tuning once it guides a design decision.
- Current protocol: collect 64 complete empty-board X-first games per update,
  with the same frozen model for both seats and private per-seat sampled actions.
  Compute GAE per seat/episode, combine all learner decisions, normalize advantages,
  and shuffle all samples into minibatches of 64 for four epochs by default.
  One Adam step occurs per minibatch, including the final partial batch. This
  already separates acting from optimization; it does not subsample games down
  to one or two decisions. Sample weighting is per decision, so longer trajectories
  can contribute more samples. Terminal rewards are +1/-1/0, intermediate zero;
  both seats receive their own targets. Do not directly punish random moves solely
  because a game was lost or add heuristic reward shaping to hide a failure.
- Current exploration is categorical sampling of masked policy probabilities
  plus a fixed .01 entropy coefficient. No guaranteed uniform exploration,
  opponent curriculum, replay, MCTS, or learned dynamics exists. Low entropy,
  low loss, and X-dominated self-play do not establish strength or a root cause.
- Pending Phase 4.5 work: diagnose credit/targets and tactics, retain periodic
  checkpoints and isolated evaluations without disturbing training RNG/resume,
  run bounded multi-seed comparisons of justified changes, and specify the
  search-learning bridge. Report both seats, coverage, sample/optimizer budgets,
  and failures. Checkpoint monitoring is not a historical training opponent pool.
- MuZero-style learning is the long-term direction. Validate bounded search
  with a simulation reconstructed from permitted public views first; agents
  must never receive the authoritative engine. Then design a separate learner
  for self-play search visit policies, outcomes, and sampled replay sequences,
  followed by learned representation/dynamics/prediction. Search-selected actions
  cannot be inserted into existing PPO as though sampled from its raw policy.
- For long games, define collection limits, replay sampling/reuse, sequence
  context, game-length weighting, and optimization budgets explicitly. Preserve
  GAE/recurrent context and player perspective; stale replay is not ordinary
  on-policy PPO. No fixed one-or-two-position quota is mandated. Hidden state,
  chance, and same-player actions retain their separate modeling requirements.

## 2026-10-15: Phase 4 observation encoding audit

The existing pure observation encoder remains unchanged: fresh contiguous CPU
float32 `(6, 3, 3)` tensors, observer-relative mark/turn planes, row-major
coordinates, and board-preserving terminal views with zero turn planes.
An independent coordinate-based rules oracle enumerates 5,478 reachable boards
without authoritative engine state or engine move generation. Tests recover
each visible board from its encoding for both observers and check invariance
when all seat labels (observer, actor, marks) are exchanged. This also covers
O-first views. All 958 terminal boards retain the terminal contract.
This verifies representation semantics, not learned tactical competence or
hidden-information filtering for future games. Existing interfaces remain intact.

Current status: Phases 1–3 are complete, including baselines, seeded simulation,
outcome metrics, human play, and public Tic-Tac-Toe replays. Phase 4, Tasks 1–7
are complete: the user finished CUDA self-play and fixed-opponent evaluation is
implemented. Task 8 curves/baseline controls are complete; the minimal random-baseline
criterion is met, with heuristic robustness explicitly unresolved. The active
next task is Phase 4.5, Task 1; Phase 5 remains pending.
The private repository is [Dzudd/boardgame-ai](https://github.com/Dzudd/boardgame-ai),
with `master` tracking `origin/master` and Windows/Linux checks in GitHub Actions.
Earlier references to a local-only repository describe the state at that task.
TicTacToeEngine supports the full engine contract. The entries below record decisions as each task
was implemented; intermediate limitations are superseded by later entries.

## 2026-10-14: Python project foundation

- Python 3.12 is the development baseline; package metadata permits newer versions.
- Use a `src/boardgame_ai` package rather than top-level importable `core`, `games`,
  and `agents` directories. Future modules will live inside this namespace.
  This prevents accidental imports from the checkout and avoids generic module names.
- Use setuptools and standard `venv`/pip, configured in `pyproject.toml`.
  No additional environment manager is required.
- Use pytest for tests, Ruff for linting/formatting, and strict mypy for typing.
  Ship `py.typed` so downstream tools recognize package annotations.
- Keep runtime dependencies empty for Task 1. Add PyTorch/NumPy in Task 2;
  engine code must remain independent of the learning framework.
- Defer game contracts and module scaffolding until their respective tasks.
  The plan's filtered observations and atomic decisions remain design constraints.

The canonical roadmap remains `BOARDGAME_AI_PROJECT_PLAN.md`; do not maintain a
second copy under `PROJECT_PLAN.md`.

## 2026-10-14: Numerical dependencies and CLI

- Add NumPy 2.x and PyTorch 2.x (minimum 2.6) as runtime dependencies, with
  compatible major-version ranges consistent with the initial dependency policy.
- Use a CPU PyTorch wheel for initial local verification. Hardware-specific
  accelerator installation remains an environment choice, not package metadata.
- Use standard-library `argparse`, with a console script and module entry point.
  The initial `info` command reports the installed numerical environment.
- Import numerical libraries only when executing `info`; package imports and CLI
  help remain lightweight. Future engine modules must not depend on PyTorch.

## 2026-10-14: Core game records

- Shared contracts live in `boardgame_ai.core`, implemented with standard-library
  dataclasses and no NumPy/PyTorch imports. All records use keyword-only fields.
- `GameState[StateData]` is authoritative engine data; `Observation[ObservationData]`
  is a separate player view. Generic payloads allow each game to define precise
  types without a shared untyped dictionary or fixed board representation.
- Records are frozen, but payloads are not recursively frozen or copied. Engines
  must independently construct safe observations and own state copying. These
  records do not themselves enforce hidden-information isolation.
- Player IDs are zero-based integer seats. `current_player`/`next_player` is None
  at termination. Automatic chance resolution happens inside the engine before
  control returns to a player. Nonterminal steps require a next actor, which may
  be the same actor as before; no alternating-player assumption is built in.
- `Action` uses a game-defined kind and named scalar parameters represented as
  tuples. `dataclasses.asdict` produces JSON-serializable data; deserialization
  and game-specific parameter validation remain future engine/replay work.
  Parameters exclude unresolved chance outcomes and reject duplicate names and
  nonfinite floats. This initial format is intentionally limited to scalars.
- `StepResult.rewards` contains finite, incremental rewards for every seat in
  order. Rewards can be general-sum, cooperative, or zero-sum. Final cumulative
  game scores and the future engine's reward query are separate API decisions.
- `GameEvent` carries a kind, scalar parameters, and explicit recipients: None
  means public; a tuple restricts visibility (an empty tuple is internal-only).
  Full `StepResult` objects are engine-facing. `events_for(player_id)` filters
  events, but engines must also avoid private content in public event payloads.
- `GameSpec` defines a stable ID, display name, and supported player count range,
  including solo games. Encoder configuration is deferred until encoding work.
- Engine/agent interfaces, illegal-action policy, seed handling, and integrated
  information-boundary tests remain their own tasks.

## 2026-10-14: Engine and agent interfaces

- Use abstract base classes (`ABC`) to require explicit engine and agent
  implementations. Both are generic over observation payloads; engine state
  representation remains private to each engine.
- Export `GameEngine` from `boardgame_ai.core` and `Agent` from
  `boardgame_ai.agents`. Neither interface imports numerical frameworks.
- Engines expose read-only `spec` and `num_players` properties, and methods
  `reset`, `observe`, `legal_actions`, `current_player`, `step`, `is_terminal`,
  `get_rewards`, and `clone`. Call reset before gameplay. No full-state accessor
  is included in the shared interface.
- `reset(seed)` returns None. It initializes the game and clears rewards; callers
  then query the next actor and that actor's observation. Chance is resolved
  internally before the engine returns control to a player.
- `get_rewards()` returns the most recent step's incremental rewards (zeros after
  reset), matching `StepResult.rewards`. Reading rewards is not consumption;
  runners collect rewards once per executed step. Cumulative scores are separate.
- `legal_actions` returns an immutable tuple, empty for inactive players or after
  termination. A nonterminal actor must have an available choice. Observations
  remain available to any valid seat after termination. Queries do not advance RNG
  or game state. Invalid-seat and illegal-action exception policies are deferred.
- `clone() -> Self` requires an independent copy of mutable and RNG state,
  preserving the concrete engine type. Each engine supplies its own copy logic.
- Agents implement `select_action(observation, legal_actions) -> Action` and may
  override the default no-op `reset(seed)`. Use an agent instance per seat and
  reset it per game so future recurrent memory does not cross player boundaries.
  An agent gets no engine reference; engines must still validate its selection.
- Test-only implementations exercise a full three-decision game with actors
  `[0, 0, 1]`, terminal queries, reward consistency, reset, and independent cloning.
  These establish interface usability, not hidden-information or RNG guarantees
  for future engines. Detailed policies and reusable conformance tests follow.

## 2026-10-14: Phase 1 visibility, failures, randomness, and CI

- `INFORMATION_BOUNDARIES.md` is the normative player visibility and lifecycle
  policy. `decision_for` deep-copies already-filtered observations and actions;
  it does not attempt to infer which game-specific fields are secret.
- Shared errors distinguish an uninitialized game, invalid seat, illegal action,
  and terminal step. Engines validate before mutation or chance consumption;
  rejection is atomic and diagnostics do not disclose private state.
- `create_rng` provides engine-owned standard-library randomness. Explicit integer
  seeds restart setup and chance, None uses system entropy, and clones preserve
  independent RNG state. No global numerical or Python RNG is touched.
- Reusable bounded test helpers check interface consistency, deterministic
  trajectories, cloning, and rejection invariants. A separate hidden-card fixture
  tests private observations, event visibility, mutable payload isolation, and
  unseen-state equivalence. It is test infrastructure, not Phase 5 implementation.
- GitHub Actions checks Python 3.12 on Linux and Windows with CPU PyTorch.
  At this task's completion, hosted CI awaited publication; the repository is now
  published with checks at https://github.com/Dzudd/boardgame-ai/actions.
  Dependency ranges remain in place; cross-version bitwise reproducibility and
  exhaustive game correctness are not promised by these checks.

## 2026-10-14: Tic-Tac-Toe initial state and reset

- Game modules live in `boardgame_ai.games.tic_tac_toe`, without numerical
  framework dependencies. `TicTacToeState` is a frozen, slotted payload with a
  nine-cell immutable tuple in row-major order. None means empty; seats 0 and 1
  represent X and O. Actor and phase live only in the enclosing `GameState`.
- `TicTacToeEngine` owns private state, rewards, and RNG. Metadata is available
  before reset; implemented state/reward queries require reset. No public
  authoritative-state accessor is added.
- Reset always produces an empty board, actor 0, phase `play`, and zero rewards
  for both seats. Setup is deterministic regardless of seed. The shared
  `create_rng` validates the seed before live state changes and keeps randomness
  separate from global generators; reset replaces the private RNG.
- The engine remains an abstract generic subclass of `GameEngine` until later
  tasks implement legal actions, steps, terminal detection, observations, and
  cloning. Its observation payload type is deliberately unresolved. A test-only
  subclass enables reset tests without production gameplay stubs.
- Full engine conformance tests require complete gameplay and are deferred to
  the remaining Phase 2 tasks. Current tests inspect trusted private state only;
  that data is never an agent input.

## 2026-10-14: Tic-Tac-Toe legal actions and validation

- A placement is `Action(kind="place", parameters=(("cell", index),))`, with
  integer indices 0–8 in row-major order. The actor comes from engine state;
  actions do not carry a player ID or a mark.
- `legal_actions` checks initialization and seat validity, then returns an
  immutable tuple of empty-cell placements in ascending index order. Inactive
  seats and terminal states receive no actions. Terminal state is currently
  identified by `current_player=None`; detecting wins/draws remains Task 4.
- `_validate_action` returns the chosen cell for the future `step` implementation.
  It requires the exact kind and one named `cell` parameter, with a real integer
  index in range and an empty destination. Booleans and floats are rejected
  explicitly despite Python equality with integers. Rejections use shared
  `IllegalActionError` or, after termination, `GameOverError`.
- Queries and validation change neither state, rewards, nor RNG. Task 3 must
  call this validator before applying a move. `step` remains abstract; no move
  execution or player switching is included in Task 2.
- Tests use trusted state setup for occupied boards, the second actor, and
  finished games until transitions and terminal detection are implemented.

## 2026-10-14: Tic-Tac-Toe placements and player switching

- `step` calls `_validate_action` before constructing or committing the next
  state. The selected empty cell receives the current seat's mark, all other
  cells retain their values, and the next actor switches between seats 0 and 1.
  The engine replaces immutable state rather than modifying prior snapshots.
- A local board tuple is cast to the fixed nine-cell `Board` type: enumeration
  replaces one value without changing the existing board's length. No mutable
  board alias escapes the engine.
- The new state and `StepResult` are built before committing state and rewards.
  Successful nonterminal placements return zero incremental rewards, no events,
  and the next actor; the reward query matches that result. No RNG is consumed.
- This is an intermediate nonterminal transition implementation. Task 4 must
  detect wins/draws on the resulting board and set terminal actor, phase, and
  rewards before the engine supports complete games. It remains abstract for
  terminal detection, observations, and cloning. Tests stop before game-ending
  positions and verify stored terminal states reject steps atomically.

## 2026-10-14: Tic-Tac-Toe terminal detection and rewards

- After each validated placement, `step` checks the resulting board's eight
  winning lines for the actor's mark, then treats a full board without a win as
  a draw. A win on the ninth placement takes precedence over a draw. Checking
  the actor is sufficient because reachable games stop at their first outcome.
- Terminal states use phase `finished` and `current_player=None`.
  `is_terminal` queries this stored outcome without recomputing or mutating it.
  The state and StepResult are constructed before committing, with consistent
  actor, termination, and reward fields. Both seats have no legal actions after
  termination, and further steps raise `GameOverError` without changing anything.
- Rewards are sparse and zero-sum: +1 for the winner, -1 for the loser, and
  zero for draws and nonterminal steps. `get_rewards` retains the latest step's
  reward tuple on repeated reads; reset clears it. No reward events or cumulative
  score API are introduced.
- Rule tests play all eight winning lines for both seats, a full-board draw,
  and a ninth-move win. They also check intermediate results, terminal query
  purity, rejected terminal steps, RNG preservation, and reset after a real win.
  Observations and cloning remain abstract, so full interface conformance remains
  deferred until those tasks are complete.

## 2026-10-14: Tic-Tac-Toe observations and text rendering

- `TicTacToeEngine` now fixes its observation type to `TicTacToeObservation`,
  a separate frozen, slotted public payload containing only the board. The
  shared Observation wrapper supplies observer ID, current actor, and phase.
  Both seats see the same absolute seat-based marks; no perspective encoding
  is introduced. Rewards, seed, RNG, and authoritative records stay out of views.
- `observe` requires initialization and a valid seat, including when finished,
  and builds fresh wrapper/payload objects. Sharing the board tuple is safe:
  it contains immutable scalar cells and steps replace rather than mutate it.
  Old observations remain stable across later moves and resets.
- `render_board` is a pure formatter accepting a public Observation, exported
  from the game package. It uses X/O/dots and horizontal separators with no
  trailing newline. `engine.render(player_id)` delegates through `observe`,
  preserving seat/lifecycle validation. Rendering returns text without printing.
- Tests cover view schema, invalid seats, immutable and stable snapshots,
  terminal views, decision_for integration, exact rendered board orientation,
  seed-independent public inputs, and unchanged state/rewards/RNG on queries.
  The engine remains abstract only for cloning, which is deferred to Task 7.

## 2026-10-14: Tic-Tac-Toe rule regression audit

- Production behavior is unchanged. Added a bounded traversal that replays legal
  prefixes from reset and checks every distinct reachable board against a
  coordinate-based outcome oracle independent of the engine's winning-line table.
  It verifies mark counts, outcomes, actor/phase, legal choices, observations,
  rewards, StepResult consistency, and terminal step rejection.
- The traversal deduplicates boards, not symmetries or move sequences. It visits
  5,478 distinct positions, including 626 X wins, 316 O wins, and 16 draws.
  Stopping expansion on independently checked terminal outcomes bounds depth
  to nine decisions. This audits reachable states, not arbitrary injected states
  or every ordering of moves leading to an identical board.
- Full-game replay regressions exercise X wins, O wins, and draws across None,
  zero, positive, and negative seeds, before and after reset. Illegal-action
  attempts are inserted before moves; public inputs and latest rewards must
  remain unchanged, and subsequent outcomes match an untouched reference.
- Tests continue to use the existing harness only because production cloning is
  abstract. Shared clone-dependent conformance helpers remain deferred to Task 7.

## 2026-10-14: Tic-Tac-Toe cloning and Phase 2 completion

- `clone() -> Self` requires initialized state and returns `deepcopy(self)`.
  This preserves the concrete engine type, state, latest rewards, and current
  private RNG sequence without drawing randomness. Mutable subclass attributes
  and RNG instances are independently copied; immutable scalar tuples may be
  safely shared. No full-state API or serialization format is introduced.
- `TicTacToeEngine` is now concrete and implements every shared engine method.
  The trusted state-setup harness inherits all production gameplay methods.
- Production-engine tests run the shared engine and rejection helpers with a
  nine-step bound. Additional tests cover divergent moves, resets on either
  branch, identical continuations ending in both players' wins and a draw,
  terminal clones, concrete subclass preservation, and progressed RNG isolation.
- Phase 2 is complete. The distinct reachable-state audit, targeted rule tests,
  public-view tests, and clone conformance establish local correctness coverage.
  Simulation runners and baseline agents remained Phase 3 work at this stage;
  hosted CI then awaited publication. See the current status above.

## 2026-10-14: RandomAgent baseline

- Export `RandomAgent[ObservationData]` from `boardgame_ai.agents`, implementing
  the existing Agent contract without game-specific logic or numerical imports.
- Select uniformly over supplied tuple entries with private `random.Random.choice`.
  Return the selected supplied Action; do not inspect observations or mutate inputs.
  The engine still validates selections, and callers supply the actor's choices.
- Reuse `create_rng` for separate agent-owned randomness. Construction accepts an
  optional seed; `reset(seed)` replaces the generator, with None using system
  entropy. Invalid seeds reject before replacing the current sequence.
- Empty legal choices raise ValueError before consuming randomness. This is a
  caller contract failure, distinct from engine illegal-action errors.
- Replay requires identical agent seeds, ordered legal tuples, selection calls,
  and runtime versions. Use one agent per seat and reset each before a game.
- Tests exercise seeded choice sequences, reset, input preservation, private RNG
  isolation, hidden-world equivalence, and bounded complete Tic-Tac-Toe replays.
  Production runners, heuristic policies, metrics, and CLI play remain later tasks.

## 2026-10-14: Tic-Tac-Toe heuristic baseline

- Export `TicTacToeHeuristicAgent` from `boardgame_ai.agents`, implementing
  `Agent[TicTacToeObservation]`. Its explicit name prevents implying that a
  game-specific heuristic works for every observation schema.
- Read only the supplied public board, observing actor, and legal placements.
  Check public winning lines locally, without importing engine internals or
  accepting an engine, state snapshot, rewards, or simulation API.
- Take an immediate win, otherwise block an immediate opponent win, otherwise
  prefer center, corners, then edges. Tactical ties use ascending cell indices;
  positional ties use row-major order within each category, independent of the
  legal tuple ordering. Always return a supplied Action object.
- The policy is deterministic and stateless, so it inherits Agent's no-op reset
  and uses no RNG. Seeds do not affect choices. RandomAgent retains its private
  seeded RNG; engine and global RNGs remain separate from both policies.
- Reject empty choices, incompatible placement parameters, occupied cells, and
  observations not belonging to a valid acting seat with ValueError. These are
  caller contract errors; engines still enforce authoritative move legality.
- This is a tactical baseline, without fork handling, minimax, or an optimal-play
  guarantee. Tests cover both seats and every winning line, precedence, positional
  preferences, deterministic ties, input purity, and complete replay/self-play.
  A production game loop and match runner remain Phase 3, Task 3.

## 2026-10-14: Reusable game loop and single-match runner

- Export `play_game`, `run_match`, and frozen `MatchResult` from
  `boardgame_ai.simulation`. Orchestration depends on generic engine/agent
  contracts and contains no game rules or numerical framework dependencies.
- `play_game` advances an initialized position without resets. `run_match` resets
  the supplied engine and one distinct agent per seat, then calls that loop.
  Supplying instances allows configured games and policies without a registry.
- Require an explicit nonnegative integer `max_steps`. Each successful atomic
  step counts once. Termination on the last allowed step is completion; otherwise
  hitting the limit is truncation, leaving the live position available to resume.
  Terminal initial positions execute no steps; zero is a valid decision budget.
- Follow engine-reported actors and StepResult next actors, including consecutive
  decisions by the same seat. Use `decision_for` for detached filtered inputs;
  never pass engine objects, rewards, or full results to agents.
- Check actor/termination consistency, actor views, nonempty live legal choices,
  and StepResult actor, termination, and reward-count agreement. Configuration
  errors raise ValueError; detected engine contract failures raise RuntimeError.
  Agent/engine exceptions propagate immediately, without retries or substitution.
  Executed steps are not rolled back on a later error.
- Keep `engine_seed` and optional per-seat `agent_seeds` explicit and separate.
  Omitted seeds use None; no derived seed schedule or runner RNG exists. Validate
  counts, distinct agent instances, budgets, and all seed values before resets.
  Custom reset failures are not transactional across engine and agents.
- MatchResult contains num_steps, summed incremental total_rewards, terminated,
  and truncated. For resumed games it summarizes only this call's executed steps.
  It retains no private events or observations and is trusted orchestration data,
  not a player-facing result. Reward totals are not scores or inferred winners;
  truncation is not a draw. Batch runs, aggregate metrics, and logs remain deferred.

## 2026-10-14: Reproducible match batches

- Export generic `run_batch` and frozen `SeededMatchResult` from simulation.
  Reuse supplied configured engine/agent instances with fixed seat assignments;
  call run_match for each match so lifecycle resets and input boundaries apply.
- Require an integer batch seed. Derive each seed as the big-endian integer of
  SHA-256 over UTF-8 text `boardgame-ai/batch/v1:{seed}:{index}:{stream}`,
  using decimal integers, stream `engine`, and streams `agent:{seat}`.
  This schedule uses no RNG, Python hash(), chance outcomes, or game inputs.
  Domain separation avoids intentionally sharing engine and agent seed streams.
- `start_index` and `num_matches` select a nonnegative absolute index range.
  Seeds depend only on batch seed, index, and stream; running a subrange reproduces
  the matching full-batch portion regardless of earlier match lengths. Explicit
  recorded seeds also allow rerunning a match directly with run_match.
- Validate configuration before any reset, including empty batches. Return an
  immutable tuple of per-match indices, engine/agent seeds, and MatchResult.
  A zero count leaves engine and agents untouched. Positive batches leave them
  at the final executed match; truncated matches are followed by the next reset.
- Execute sequentially and propagate the first exception. No partial tuple is
  returned and executed games are not rolled back. Do not log exception content,
  private events, or observations. Error aggregation belongs to Task 5.
- Reproducibility requires identical configured engine/agents, code, runtime,
  action ordering, and full resets honoring seeds. Schedule version v1 is explicit;
  this is not a cross-version guarantee for game or agent RNG sequences.
- Add a repeatable verification script for three 1,000-match Tic-Tac-Toe batches
  and reruns. Metrics, CLI play, seat rotation, and replay persistence are deferred.

## 2026-10-14: Match records, outcome metrics, and explicit summary export

- Add evaluate_batch alongside run_batch; preserve run_batch's fail-fast contract.
  Share batch configuration validation and the versioned seed schedule. Evaluation
  continues after ordinary reset/play exceptions by resetting the next match.
  KeyboardInterrupt/SystemExit propagate; no action is substituted or retried.
- A trusted callable maps the completed GameEngine to GameOutcome with per-seat
  finite scores, distinct winner seats, and an explicit draw flag. This supports
  multiplayer ties and winnerless cooperative outcomes. No generic winner/draw
  classification is inferred from rewards. The adapter runs only on termination.
- Implement the Tic-Tac-Toe adapter in its game evaluation module, reading the
  terminal public board. Its result-score convention is +1 winner/-1 loser/zero
  draw. GameEngine and Agent interfaces remain unchanged; adapters are trusted
  orchestration helpers, never agent inputs or privileged critics.
- Frozen MatchRecord stores seeds, optional MatchResult, optional GameOutcome,
  and a fixed error category: illegal_action, match_failed, or outcome_failed.
  No exception message, class name, traceback, private input, or event is retained.
  Reset/play failure has unknown length and rewards (no MatchResult), even if
  partial play occurred. Outcome failure retains the successful runner result.
- EvaluationSummary partitions attempts into completed/classified, truncated,
  and errors. Wins, draws, total/mean scores, and min/mean/max lengths use only
  completed/classified matches. Undefined means and length extrema are None,
  not zero. Per-match successful/truncated lengths remain in records. Errors and
  truncations never become draws; multiple winners may produce several wins.
- EvaluationReport records game ID, caller-supplied seat labels, batch seed,
  start index, limit, records, and summary. Caller labels are trusted metadata.
  save_evaluation explicitly writes schema-version-1 JSON to a chosen path,
  overwriting it; parent directories must exist. Seeds remain trusted metadata.
  This summary export has no action histories or replay logging (Task 7).
- Benchmark script reruns three 1,000-match baseline matchups, compares complete
  reports, and writes ignored runs/baseline_metrics JSON. Document measurements
  and scoring in BASELINE_BENCHMARKS.md. CLI play remains Task 6.

## 2026-10-14: CLI human-versus-agent play

- Add `play` to the existing argparse CLI, defaulting to human X against the
  heuristic. Select X/O and random/heuristic opponent with explicit options;
  engine setup and opponent RNG seeds remain separate optional integers.
- Keep terminal I/O in cli_play rather than reusable baseline or engine code.
  HumanTicTacToeAgent implements Agent[TicTacToeObservation] and renders only
  its detached public board/legal inputs. It has no engine reference or RNG.
- A reporting agent wraps the opponent, forwards reset seeds, and prints its
  public placement choice. Both players use the unchanged run_match and step
  validation path, with distinct agent instances and a nine-decision bound.
- Input parsing retries nonintegers and unavailable cells without returning an
  action or executing a step. A valid choice returns the supplied Action object.
  The UI uses the engine's existing row-major 0–8 numbering and shows a cell map.
- Handle q/quit/exit, EOF, and KeyboardInterrupt at the presentation boundary.
  Cancellation submits no new action and prints no terminal outcome. Genuine
  engine/agent errors continue to propagate. No persistence or logging is added.
- Render the final human public view and classify wins/losses/draws using the
  existing terminal board adapter. Results are public for this game; no generic
  private score or trusted seed report is printed to the player.
- Import gameplay only when executing play; info/help retain lightweight parsing
  without loading numerical frameworks until info execution. Test scripted play
  from both seats, outcomes, retries, cancellation, seeds, and CLI validation.

## 2026-10-14: Human-play starter selection and adjacent choice board

- Add CLI `--starts human|ai|X|O`, independent of human mark; retain X-first as
  the default. The CLI maps this choice to TicTacToeEngine(starting_player=...).
- The engine validates a constructor starting seat (0/1, excluding booleans),
  stores it as configuration, and uses it during reset. Cloning preserves the
  setting. Mark/seat meanings, rewards, Agent, and GameEngine interfaces stay fixed.
  O-first is an explicit configuration; existing audits/benchmarks use default X.
- Render a game board beside a choice board using only observation and supplied
  legal placements. Available cells show their indices; all others show dashes.
  Render on every human prompt/retry and the final board; terminal legal choices
  are empty. No private engine data or extra RNG is used for presentation.
- Keep the deterministic tactical policy unchanged. Repeated draws do not imply
  optimal play; identical human choices reproduce games, and fork weaknesses remain.

## 2026-10-14: CLI simplification and public replay logs (Phase 3 completion)

- Replace CLI mark choices with --starts human/ai (default human); map the starter
  to X, other player to O. Remove --human and --starts X/O. Keep optional engine
  O-first configuration for programmatic games and replay reconstruction.
- Add --game tic-tac-toe, currently the default/only supported game. Additional
  games will require explicit CLI dispatch and game-specific presentation/replay
  adapters; no new engine or generic registry is introduced prematurely.
- Add an optional trusted on_step(actor, action) callback to play_game/run_match,
  after a successful step and contract checks. No observations or full StepResults
  are supplied. Exceptions propagate after execution, without rollback. Other
  simulation APIs retain their existing behavior; no default logging is added.
- TicTacToeRecorder records successful public placements. Frozen TicTacToeReplay
  stores starting_player, actor/cell moves, and completed/cancelled/truncated status.
  JSON includes schema_version=1 and game_id=tic_tac_toe. No seeds, private events,
  rewards, engine snapshots, or observations are exported.
- Save validates before overwriting the explicit path; parent directory must
  exist. Load accepts only the supported fields/schema/game, at most nine moves,
  and replays through a fresh configured engine. Actor, legal placement, no moves
  after termination, and ending status must agree with the engine. Public
  reconstructed observations are returned only after the entire log validates.
- CLI --save-replay saves completed play or cancellation's executed prefix.
  `replay PATH` validates before showing initial/step boards and final status.
  This is inspection, not resumption or policy re-execution. Unexpected play
  failures propagate without an exported completion record.
- Replay privacy is game-specific. These logs are safe public Tic-Tac-Toe data;
  hidden-game actions cannot be assumed public merely because on_step sees them.
  Future games need explicit player-filtered formats and chance reconstruction.
- Phase 3 acceptance is met by tested baselines/runners, reproduced benchmark
  batches/metrics, human play, and validated replay inspection. Begin Phase 4
  only with observation encoding; learning has not been implemented.

## 2026-10-14: Plan v0.2 — position evaluation and forward planning

- Integrated the new planning material while preserving completed Phases 1–3
  and Phase 4 observation encoding as the next task. The suggested search-first
  sequence is an optional experiment track, not a reordered implementation roadmap.
- Distinguish authoritative engine transitions, position-value evaluation, and
  branch selection/search. The current heuristic agent selects actions; terminal
  GameOutcome adapters classify results. Neither implements a general position
  evaluator. Proposed evaluator interfaces remain future design sketches.
- Prefer actual engine rules for future search before experimenting with learned
  latent dynamics. Evaluate heuristic, rollout, neural, and hybrid estimators with
  explicit perspective/return semantics and bounded computation. Keep true terminal
  rewards separate from heuristic estimates used for search.
- Keep clone RNG semantics unchanged: independently owned copied RNG state.
  Future planning contexts must respect the existing observation-only Agent
  boundary. Do not pass an authoritative hidden-state clone to a planner; hidden
  simulations need information-compatible worlds and information-set-aware action
  selection to avoid leakage and strategy fusion. No code/API changes were made.

## 2026-10-14: Observation tensor encoding (Phase 4, Task 1)

- Export `encode_tic_tac_toe_observation` from `boardgame_ai.models.encoders`.
  This pure game-specific function consumes only
  `Observation[TicTacToeObservation]`; no engine/state, legal choices, rewards,
  seed, or RNG is accepted or queried. Existing core/agent interfaces and
  GameSpec remain unchanged. Numerical imports stay in the learning layer.
- Return a fresh contiguous CPU `torch.float32` tensor shaped `(6, 3, 3)`,
  without a batch dimension or autograd history. Channels in order are own
  marks, opponent marks, empty cells, own turn, opponent turn, terminal.
  All values are 0/1; the last three channels are constant planes.
- Perspective follows `player_id`, including inactive observers. Swap seats
  by swapping mark and turn channels, keeping empty/terminal channels and board
  coordinates fixed. Row-major cell `i` maps to `(i // 3, i % 3)`.
- Live phase `play` requires actor 0 or 1. Finished phase `finished` requires
  actor None: keep all board channels, zero both turn planes, fill terminal
  with ones. No result/reward channel is added. Terminal encoding does not
  authorize action selection or make private information public.
- Reject incompatible observation payloads, seats (including bool/float),
  nine-cell tuple schemas, and phase/actor pairs with ValueError. This checks
  schema compatibility, not reachability or game rules. Diagnostics omit values.
- Allocation is explicit about dtype/device and consumes no randomness.
  Callers may use `torch.stack` to batch or `.to(device)` to transfer later.
  Avoid generic encoder registries and model configuration until needed.
- Tests cover standalone views, exact orientation/perspective, detached actor
  inputs and inactive views, both starters, terminal wins/draws, fresh storage,
  snapshot stability, RNG purity, schema rejection, and GameState rejection.
  Only observation encoding is implemented; Task 2 remains next.

## 2026-10-14: Legal-action encoding and masking (Phase 4, Task 2)

- Add game-specific action helpers in `boardgame_ai.models.encoders`: encode a
  place/cell Action as a Python integer 0–8, encode supplied legal choices as a
  fresh contiguous CPU bool mask `(9,)`, and decode an index back to the matching
  supplied Action object. Coordinates remain row-major for both observers.
- Actions require exactly one integer cell parameter; booleans/floats, wrong
  schemas, duplicate indices, and unavailable decoded indices raise ValueError.
  Legal inputs remain tuples. No engine/board query or rules inference is added;
  callers must supply the correct player's already-filtered legal choices.
- Empty tuples produce all-false masks for terminal/inactive records. Even a
  terminal board with empty cells has no selectable actions. Action decoding
  from an empty tuple rejects; masking an all-false row also rejects.
- `boardgame_ai.models.masking.mask_action_logits` requires finite floating-point
  logits and bool masks of identical nonempty shape `(..., action_count)` and
  device. No broadcasting or automatic device conversion occurs. Every row must
  have at least one legal choice; filter terminal rows before constructing a
  distribution. Empty batches reject explicitly.
- Return a new tensor with illegal logits -inf, retaining legal logits, dtype,
  device, and autograd connectivity. Softmax gives illegal choices zero probability;
  illegal logit gradients are zero. Inputs are not mutated; no randomness,
  distribution, sampling policy, network, or training implementation is added.
- Masks may be stacked and explicitly moved to the logits' device by callers.
  This fixed nine-action adapter is specific to Tic-Tac-Toe, while the masking
  helper accepts any action count. No action-space registry is introduced.
- Tests cover all cell round-trips, supplied-object identity, order independence,
  storage/RNG purity, both-seat gameplay, terminal/inactive masks, schema failures,
  batched probabilities/gradients across float dtypes, and shape/device failures.
  Next is Phase 4, Task 3: policy and value networks.

## 2026-10-14: Policy and value networks (Phase 4, Task 3)

- Export `TicTacToeActorCritic` from `boardgame_ai.models`. This game-specific
  nn.Module flattens the six observation planes to 54 features, then uses
  two 64-unit tanh layers shared by linear policy (9) and value (1) heads.
  A small shared body keeps the initial PPO model simple; no recurrence,
  convolution, search, or general game registry is added.
- Forward accepts nonempty tensors shaped `(..., 6, 3, 3)`, returning a tuple
  of raw logits `(..., 9)` and scalar values `(...)`. Single observations produce
  logits `(9,)` and a rank-zero value. Arbitrary leading batch dimensions and
  noncontiguous tensors work; only trailing observation dimensions are flattened.
- Policy indices retain the fixed row-major action contract. Logits are neither
  probabilities nor masked. Apply the existing mask helper separately to live
  decisions using supplied legal choices; no sampling or Agent adapter is added.
- The value is an unbounded estimate of future return for observation.player_id,
  regardless of current actor or absolute mark. Its eventual discount/target
  construction belongs to trajectory work. Both heads consume exactly the same
  filtered tensor; no privileged critic, reward input, or engine reference exists.
- Terminal/inactive views are accepted as observations. Terminal output values
  remain predictions, not exact outcomes or hard-coded zeros. Future trajectory
  code must bootstrap zero at true termination independently of model output;
  all-false terminal/inactive masks still reject action selection.
- Parameters start CPU float32 explicitly, independent of default floating dtype.
  Use standard PyTorch Linear initialization and caller-controlled PyTorch seeds;
  construction consumes PyTorch randomness, while forward does not. No custom
  model seed API, global reseeding, dropout, or mutable running statistics is added.
- Model/input dtype and device must match; callers use `.to(...)` explicitly.
  Wrong/empty shapes, non-floating or nonfinite inputs, and mismatches raise
  ValueError without input values in diagnostics. Validation checks tensor
  compatibility, not binary-plane semantics, visibility, or game reachability.
- Differentiation remains enabled by default. For inference use eval() with
  no_grad()/inference_mode(); train/eval produce the same predictions here.
  Both heads can backpropagate into the shared body without mutating inputs.
- Tests verify batching against individual predictions, raw output semantics,
  separate policy/value gradient paths, masked illegal gradients, deterministic
  seeded construction, inference purity, dtype conversion, seat relabeling,
  detached gameplay and terminal views, and invalid tensor/device rejection.
  Trajectory collection/GAE is next; no optimizer or checkpoint workflow is added.

## 2026-10-14: Trajectory collection and GAE (Phase 4, Task 4)

- Export `collect_tic_tac_toe_rollout`, `PlayerTrajectory`, `TicTacToeRollout`,
  and `generalized_advantage_estimation` from `boardgame_ai.training`.
  Collection is trusted orchestration, specialized to two-seat Tic-Tac-Toe
  observation/action schemas. Core/Agent/runner interfaces remain unchanged.
- Reset the supplied engine and collect one bounded game with two supplied
  actor-critic models (a shared instance is allowed). Follow engine actors,
  including consecutive decisions by one seat. Pass only detached decision_for
  views through encoders to inference, and decode back to a supplied legal action.
  Standard actor/result/termination checks apply; errors propagate without retries.
- Frozen records own mutable detached CPU tensors. Per player with T decisions:
  observations `(T,6,3,3)` float32, legal_masks `(T,9)` bool, actions and global
  step_indices `(T,)` int64, old log_probs/values/rewards `(T,)` float32, plus
  scalar bootstrap_value. Empty seats retain these shapes and zero bootstrap.
  Finite float32 storage is checked. These are trusted per-seat learning records,
  not public replay logs; no seeds, full observations, events, or state are retained.
- Inference uses no_grad and explicit conversion to each model's dtype/device.
  Modes, weights, and existing gradients are unchanged. Mask raw logits, compute
  log_softmax, and sample with torch.multinomial using one private CPU generator
  per seat. Generator seeds are create_rng(sampling_seed).getrandbits(64);
  None uses system entropy. No global RNG or engine RNG is consumed by sampling.
  Reproduction requires fixed weights, seeds, ordering, runtime, and configuration.
- Each record's reward sums that seat's incremental engine rewards after its
  action until just before its next action or the rollout end, including opponent
  actions. Thus both winner and loser receive terminal rewards. Increments before
  a seat's first action belong only to total_rewards, not policy credit.
- Explicit discount clock: gamma advances once per player's recorded decision;
  intervening rewards are summed without internal discounting. GAE lambda also
  advances on this clock. This is not atomic-step discounting, and no alternating
  sign trick is used. Future games requiring time-based discount need a separate
  duration-aware contract. Never interleave seats or concatenate episodes for GAE.
- True termination takes precedence on the final allowed step and bootstraps zero
  without terminal model calls. Reaching max_steps while live is truncation, not
  a draw: bootstrap each nonempty segment from its own filtered cutoff observe,
  including inactive views. Empty or initially terminal games have no samples.
  max_steps=0 is valid. The engine remains at the cutoff, but every collection
  call resets; resumable collector/RNG state and batched schedules are deferred.
- GAE accepts matching finite floating 1-D reward/value tensors, finite bootstrap,
  and gamma/lambda in [0,1]. Backward recursion uses delta=r+gamma*next_value-value,
  A=delta+gamma*lambda*A_next; returns=A+value. Outputs are detached and retain
  dtype/device; empty segments are supported. Invalid inputs/overflow reject.
  Bootstrap zero at true termination is the collector/caller's responsibility.
- Tests use hand-calculated GAE, independent discounted-return sums, repeated
  actors/general-sum interval rewards, real wins/losses/draws for both starters,
  terminal inference rejection, seat-dependent cutoff bootstraps, empty segments,
  masked log-prob reconstruction, deterministic sampling, storage/RNG/gradient
  isolation, and input/failure checks. PPO losses, optimization, checkpointing,
  baseline opponent collection, and training schedules remain future tasks.

## 2026-10-14: PPO losses, optimization, and checkpoints (Phase 4, Task 5)

- Add `PPOConfig`, `PPOBatch`, `PPOLoss`, `PPOUpdate`, `PPOTrainer`,
  `prepare_ppo_batch`, `ppo_loss`, and save/load checkpoint helpers, exported
  from training. No engine/Agent interface changes or training schedule are added.
- Prepare GAE separately for each player's episode segment before concatenating
  samples. Skip empty segments, reject an entirely empty batch, and clone/detach
  visible observations, masks, actions, old log probabilities, advantages, and
  returns. Callers must supply only samples from the policy being updated;
  exclude baseline/other-policy opponents. Policy provenance is not inferred.
- Batches use CPU float32 observations `(N,6,3,3)` and targets `(N,)`, bool masks
  `(N,9)`, and int64 actions `(N,)`. Require finite data, nonpositive old log
  probabilities, and sampled actions legal under saved masks. No rewards/state
  become model features; old targets detach defensively in the loss.
- Policy loss is negative mean min(r*A, clamp(r,1-epsilon,1+epsilon)*A), with
  r=exp(new_log_prob-old_log_prob). Value loss is half mean squared error to
  detached GAE returns, without value clipping. Total = policy + value_coefficient
  * value_loss - entropy_coefficient * mean legal-action entropy. Replace illegal
  log terms with zero before entropy multiplication to avoid 0 times -infinity.
- Configuration defaults: learning rate 3e-4, gamma .99, lambda .95, clip .2,
  value coefficient .5, entropy coefficient .01, four epochs, minibatches 64,
  gradient norm limit .5, advantage normalization enabled. Normalize once across
  the full update batch with population standard deviation and 1e-8 floor;
  singleton/constant advantages become zero, preserving finite value/entropy loss.
- Initial updater explicitly supports CPU float32 actor-critic parameters, using
  Adam. Snapshot all batch tensors and freeze targets across bounded epochs.
  Private CPU randperm shuffling uses create_rng(shuffle_seed).getrandbits(64)
  without global RNG changes; include final partial minibatches. Clip gradients
  with nonfinite rejection before stepping. Model train/eval mode is unchanged.
- Diagnostics are sample-weighted means over pre-step minibatches/epochs:
  total/policy/value loss, entropy, approximate KL mean((r-1)-log(r)), clip fraction,
  sample count, and executed optimizer steps. These are update diagnostics, not
  evaluation scores. Track completed update calls and successful optimizer steps.
  Failed later minibatches do not roll back earlier steps; no KL early stopping,
  optimizer schedule, rollout scheduling, or claims of playing strength are added.
- Checkpoint version 1 fixes Tic-Tac-Toe observation/action/model identity and
  saves CPU model state, Adam moments/groups, complete config, update/step counters,
  model mode, and private shuffle RNG. It excludes games, rollouts, engine/sampling
  seeds, gradients, and global RNG. This is an update-boundary snapshot, not
  mid-minibatch/episode resumption or a complete future experiment checkpoint.
- Save validates before writing a temporary sibling and atomically replacing
  the chosen path (existing parent required). Load uses torch.load weights_only=True
  on CPU, checks exact schema/config/weights/Adam/RNG compatibility, then restores
  a fresh trainer. Model construction preserves global Torch RNG with fork_rng.
  Exact next-update continuation is tested for the same batch/runtime; cross-version
  optimizer compatibility is not promised. Unsupported/corrupt content rejects.
- Tests independently calculate clipping for both advantage signs, entropy and
  value losses, clipped/illegal gradients, detached targets, per-segment returns,
  actual parameter changes, gradient bounds, singleton normalization, configuration
  failures, deterministic checkpoint continuation, corruption rejection, and
  preservation of existing files on invalid saves. Task 6 remains next.

## 2026-10-14: User-run CUDA self-play (Phase 4, Task 6 implementation)

- Add `TrainingConfig`, `run_self_play_training`, explicit device resolution, and
  lazy `boardgame-ai train`. Default device is CUDA; cpu/cuda:N are explicit.
  Missing CUDA, unsupported/indexed devices fail before output/collection. The
  check-device flag prints device/GPU and exits without training or files.
- Extend PPOBatch/loss/trainer from CPU-only to colocated CPU/CUDA float32.
  Snapshot/normalize targets, then transfer the full update batch to model device.
  Shuffle with the existing private CPU generator, transferring indices explicitly.
  Model forward/backward, clipping, and Adam operate on the chosen device.
- Training is shared-policy self-play: freeze a CPU policy snapshot for every
  complete-game collection batch, sharing it across seats. All seat segments
  belong to the learner policy. CPU inference avoids a GPU transfer/kernel per
  move; engines, sampling, and GAE stay on CPU. No baseline scheduler or fixed
  evaluation opponents are introduced, and no speedup/strength claim is made.
- Default CLI budget is 1000 updates with 64 complete games per update; tune
  PPO flags explicitly. Initialization, shuffle, engine, and per-seat sampling
  seeds use separate streams. Versioned training seed derivation hashes UTF-8
  boardgame-ai/training/v1:{seed}:{absolute_game_index}:{stream}, retaining the
  first eight SHA-256 bytes as an unsigned big-endian integer. Initialization
  and shuffle streams use index zero; games use update_index*games_per_update+i.
  CPU initialization preserves global CPU/CUDA RNG, as do collection and shuffling.
- Save version-1 PPO checkpoints as portable CPU tensors, including Adam moments
  from CUDA. Loading optionally selects a device; validate on CPU, then restore
  the model and optimizer moments on requested hardware. CPU is load default;
  same-device/runtime next-update identity is tested, cross-device identity is not.
- A new run requires an absent/empty output directory. Save immutable training.json
  (format/game/mode, seed, games_per_update, complete PPO config) and initial
  latest.pt. Every completed update saves latest.pt before appending diagnostics
  to updates.jsonl and invoking the progress callback. Counters in checkpoint are
  authoritative; a crash between save/log can leave missing diagnostics.
- Resume requires the same manifest/PPO configuration and a total update budget
  at least the saved count. Future game seeds derive from absolute indices;
  saved Adam/shuffle state and frozen-policy collection reproduce uninterrupted
  runs on the same runtime/device. Interrupted unsaved updates restart from the
  preceding checkpoint. No mid-minibatch/episode resume or background launch.
- Diagnostics include update index, device, sample/optimizer counts, PPO metrics,
  and X/O wins/draws from training self-play. These are not held-out evaluation
  or public hidden-game logs. The user runs the actual experiment; this session
  performs only tiny unit verification and no full training.
- CPU tests exercise CLI preflight/config/output protection, sampling schedule,
  tiny self-play, exact resume, and previous PPO/checkpoint compatibility. Real
  CUDA update/portable-checkpoint and runner tests are conditional. Current
  environment has PyTorch 2.14.1+cpu and CUDA unavailable despite an RTX 4070 Ti;
  CUDA dependency installation and GPU execution remain user actions.
- Task 6 implementation is ready, but the training checkbox and Phase 4 acceptance
  remain pending user execution. Task 7 held-out evaluation is next afterward.


## 2026-10-14: Windows CUDA environment repair and hardware validation

- Root cause was the initial CPU wheel installation, not PPO placement or driver
  incompatibility: the project venv imported torch 2.14.1+cpu, with
  torch.version.cuda=None and zero CUDA devices. Python is 3.12.14, 64-bit Windows;
  NVIDIA detects RTX 4070 Ti with driver 610.88. User/system site packages are
  excluded, no PYTHONPATH/Conda/CUDA override was present, and only one torch
  distribution was installed. Bare python resolves to the WindowsApps alias;
  always use the explicit project interpreter and its python -m pip.
- Replaced only torch with the official cp312 win_amd64 2.14.1+cu130 wheel,
  within torch>=2.6,<3. Used --no-deps to preserve existing dependencies;
  pip check passes. No driver, dependency range, or application code change.
- Consulted official PyTorch Windows installation and wheel indexes plus NVIDIA
  CUDA compatibility documentation (links in README). Driver CUDA support 13.3
  is distinct from the wheel runtime 13.0 and from a local Toolkit. No separate
  Toolkit is required for these prebuilt operations; no custom CUDA build exists.
- Hardware checks now see one device, compute capability 8.9. CUDA matmul and
  actual PPO optimization pass. Strengthened conditional tests cover all batch
  fields, parameters, finite gradients, both Adam moments, CPU-portable serialized
  tensors, restored CUDA moments, identical next updates, and tiny GPU runner
  resume matching uninterrupted parameters/Adam/shuffle state.
- Preserve intentional CPU game simulation, frozen rollout inference, and GAE.
  CUDA optimization and fail-fast device selection already behaved correctly.
  Player information boundaries and checkpoint contents are unchanged.
- Full training and held-out strength/speed evaluation remain user-run work.

## 2026-10-15: Fixed evaluation opponents and held-out seeds (Task 7)

- Add TicTacToePolicyAgent in its own module, preserving lightweight baseline
  imports and the existing Agent contract. It owns a frozen CPU float32 model
  snapshot in eval mode with no gradients and an independent resettable generator.
  Use only player-visible observations/supplied legal choices. Default sampled
  selection matches collection; greedy mode uses lowest-index ties. No engine,
  rewards, privileged value inputs, or search enters the agent.
- Add evaluate_policy_checkpoint/save_policy_evaluation and CLI evaluate. Read
  source bytes once, hash them, validate/load a temporary CPU snapshot with the
  existing restricted checkpoint loader, and never reopen source during games.
  This ties all cases to exactly the reported bytes/update/optimizer counters.
  Source/training files are never written; no optimization or CUDA work occurs.
- Fixed cases: existing random-v1 and heuristic-v1, each with learner X/O, equal
  games_per_seat (default 1000), X-first engine rules, nine-step bound. Reuse
  evaluate_batch, run_match, and the terminal public-board outcome adapter.
  Do not infer wins from reward totals, truncation, or model value estimates.
- Independent seed root defaults 20261015. Case seeds are unsigned big-endian
  SHA-256 of UTF-8 boardgame-ai/evaluation/v1:{root}:{opponent}:{learner_seat},
  then the unchanged batch-v1 engine/seat seed schedule uses absolute indices.
  start_index supports deterministic shards. This separates RNG schedules from
  training; Tic-Tac-Toe board positions are not guaranteed disjoint.
- Reports contain checkpoint SHA-256/counters, mode, schedule/root/index, fixed
  case labels, original per-match outcomes/lengths/seeds, and fixed error categories.
  No weights, observations, boards, private events, or exception text are exported.
  Sampled/greedy results are separate protocols. Deterministic greedy versus the
  heuristic repeats one path per seat, not independent policy trials.
- CLI prints W/D/L, completion/error/truncation counts and saves trusted JSON.
  Guard source-checkpoint and known training-artifact overwrite paths. Parent
  must exist; explicit report destinations overwrite after JSON serialization.
  Errors/truncations remain in reports and produce a nonzero CLI exit status.
- User completed Task 6 with 1000 CUDA updates/64 games per update/seed 42.
  CPU evaluation of the unchanged checkpoint executed 4000 games with no errors.
  Random: X 770/66/164 and O 484/73/443 W/D/L. Heuristic: X 0/3/997 and O 0/0/1000.
  Record identity/protocol/limitations in POLICY_EVALUATION.md. Positive random
  results do not establish robustness against tactics or matched-control improvement.
- Tests cover snapshots/source isolation, legal selection/ties/RNG/reset, both
  seats, fixed cases, separate seed namespaces, exact repeats/shards, checkpoint
  replacement during evaluation, configuration failures, sanitized error reports,
  and CLI artifact protection. No training implementation/configuration changed.
  Task 8 plots and controlled baseline comparisons remain before phase acceptance.

## 2026-10-15: Curves, baseline controls, and Phase 4 acceptance (Task 8)

- Add CPU analysis/CLI analyze, optional Matplotlib extra, and headless PNG/SVG
  figure generation. Read saved diagnostics/checkpoint only; all outputs go to a
  separate absent/empty directory. No optimization, game-rule change, or retraining.
- Freeze the sampled policy with Task 7's protocol; replace only the learner with
  random/heuristic controls. Reuse identical engine/seat seeds, indices, bounds,
  opponent versions, and both-seat game counts. Reject incomplete/error records
  or misaligned protocols before comparisons. Actions/trajectories can differ.
- Primary assessment: paired policy-minus-random score versus random opponents,
  equal seat weighting. Normal approximate 95% intervals use sample SE of paired
  differences; stratified SE = .5*sqrt(SE_X²+SE_O²). Positive lower bound and >=30
  games per seat support the minimal checkpoint-level criterion. Match uncertainty
  does not measure variation across training runs or establish heuristic robustness.
- Parse strictly increasing finite log rows/counts. Plot faint raw series and
  labeled 25-update trailing means; gaps break curves and restart smoothing.
  Self-play curves are not held-out strength trajectories. Export grouped mean-score
  controls with match-mean intervals. Figures were rendered and visually inspected.
- Actual 12,000-game comparison reproduces Task 7 exactly; no errors/truncations.
  Random control scores X +.318/O -.303 versus random; policy +.606/+.041.
  Balanced effect +.3160, approximate 95% interval [.2644,.3676]. Heuristic control
  +.963/+.843 versus random, while policy loses nearly all heuristic games and is
  worse there than random controls. Last-25 self-play X win rate reaches 99.81%
  and entropy .1455: low loss/self-play dominance do not establish robust tactics.
- Preserve chart copies under docs/phase4 and protocol/acceptance in PHASE4_RESULTS.md.
  Raw JSON/logs/weights remain ignored in runs. Optional analysis dependencies are
  installed in CI for render tests. All eight tasks and the minimal phase criterion
  are complete; no Phase 5 work was started and no additional training was run.
