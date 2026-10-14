# Progress

Current status: Phases 1–3 are complete. All 360 tests and local development checks
pass. Next is Phase 4, Task 1: encode observations as PyTorch tensors.
Entries below are historical
task records; earlier references to incomplete behavior describe that task's stage.

## Phase 1, Task 1 — Python project and development environment

Implemented on 2026-10-14:

- Existing Git repository retained.
- Added setuptools package configuration and `src/boardgame_ai` namespace.
- Configured pytest, Ruff formatting/linting, and strict mypy.
- Added a package installation smoke test and developer setup instructions.
- Added ignore rules for environments, caches, build output, and training artifacts.

Validation: package installation smoke test passed (1 test), Ruff lint and format
checks passed, and strict mypy passed (2 source files).

Local environment: `.venv` uses bundled Python 3.12.14 because the shell's
`python` command is an inaccessible Windows app alias. The bundled runtime's
`ensurepip` failed, so the environment was created with `--without-pip` and
bootstrapped using the bundled interpreter's `pip --python .venv` command.
On a fresh machine, install Python 3.12+ and follow README setup instructions.
Git is already initialized; its executable is also absent from shell PATH.

## Phase 1, Task 2 — Numerical dependencies and minimal CLI

Completed on 2026-10-14:

- Added NumPy and PyTorch to runtime dependency metadata.
- Installed NumPy 2.5.4 and PyTorch 2.14.1+cpu in `.venv`.
- Added `boardgame-ai` and `python -m boardgame_ai` entry points with help,
  version output, and an `info` command. CUDA availability is false for this CPU build.
- Added CLI behavior tests and a NumPy/PyTorch interoperability and autograd test.
- Documented CPU installation and the future accelerator installation choice.

Validation: all 5 tests passed; Ruff lint and format checks, strict mypy (6 source
files), and `pip check` passed. Both module `info` and console `--version` entry
points were executed successfully. A targeted mypy ignore covers PyTorch's
untyped `backward` method; strict checking remains enabled elsewhere.

## Phase 1, Task 3 — Shared game types

Implemented on 2026-10-14:

- Added `boardgame_ai.core` with generic `GameState` and `Observation`, scalar
  parameter `Action`, `StepResult`, and `GameSpec` records.
- Added `GameEvent` and event filtering with explicit player recipients.
- Defined zero-based player seats, per-step reward semantics, terminal actor
  handling, and support for consecutive decisions by the same player.
- Added validation of action parameters, step-result invariants, and player counts.
- Documented shallow immutability and the engine's responsibility to construct
  observations without private state references.
- Added 15 contract tests, including serialization, event visibility, multiplayer
  rewards, solo games, terminal results, and invalid data.

Validation: all 20 tests passed; strict mypy passed for 9 source files; Ruff lint
and formatting checks passed.

## Phase 1, Task 4 — Abstract engine and agent interfaces

Implemented on 2026-10-14:

- Added generic `GameEngine` and `Agent` abstract base classes, with public exports.
- Defined engine metadata, reset, observations, legal actions, atomic steps,
  next-actor queries, termination, latest-step rewards, and independent cloning.
- Added a default agent reset hook for future memory and agent-owned randomness.
- Documented lifecycle, reward semantics, and separation of agents from engines.
- Added test-only concrete implementations covering a complete game with
  consecutive same-player decisions, cloning, terminal queries, and reset.

Validation: all 23 tests passed; strict mypy passed for 13 source files; Ruff
lint and formatting checks passed.

## Phase 1, Task 5 — Player visibility and information boundaries

Completed on 2026-10-14:

- Added the normative `INFORMATION_BOUNDARIES.md` policy covering observations,
  legal actions, private events, rewards, history, diagnostics, and training inputs.
- Added `decision_for` to detach agent inputs using deep copies.
- Tested observation mutation, hidden-world equivalence, post-draw visibility,
  private event filtering, and messages that exclude private state.

## Phase 1, Task 6 — Illegal actions and deterministic randomness

Completed on 2026-10-14:

- Added shared errors for uninitialized games, invalid players, illegal actions,
  and terminal steps. Defined validation-before-mutation requirements.
- Added private RNG construction with integer seeds or system-seeded None.
- Updated the deterministic fixture to follow lifecycle and error policies.
- Added a test-only hidden-card/chance fixture; verified rejected actions preserve
  future chance outcomes, cloned RNGs are independent, queries do not consume
  randomness, and global random state is untouched.

## Phase 1, Task 7 — Unit tests and CI

Completed on 2026-10-14:

- Added reusable bounded engine and rejection conformance helpers.
- Exercised full games against both deterministic and hidden/chance fixtures.
- Added GitHub Actions checks for Python 3.12 on Windows and Linux with CPU
  PyTorch, including dependency, lint, formatting, typing, tests, and CLI checks.

Final local validation: all 35 tests passed; Ruff lint/format checks, strict mypy
(19 source files), and `pip check` passed. Hosted CI is configured but has not
executed: this repository has no GitHub remote. Linux checks await hosted CI.

**Phase 1 complete.** The dummy engines reset, expose filtered observations and
legal actions, execute atomic decisions (including repeated actors), terminate,
and pass the interface checks. Abstract contracts and shared helpers cannot
guarantee correctness of future engines; each game must pass its own rule and
visibility tests.

## Phase 2, Task 1 — Tic-Tac-Toe initial board state and reset

Completed on 2026-10-14:

- Added the game package and immutable nine-cell, row-major board payload.
- Added an abstract Tic-Tac-Toe engine foundation with two-player metadata,
  private authoritative state, reset, current-player and latest-reward queries.
- Reset starts X (seat 0) on an empty board in phase `play`, clears rewards,
  and replaces the private RNG only after validating the seed.
- Added 12 tests covering lifecycle, metadata, reset across seeds, clearing
  prior state/rewards, instance independence, global RNG isolation, and atomic
  rejection of invalid seeds, including preservation of private RNG state.
- Kept gameplay and observations deferred; the production engine is explicitly
  abstract, and tests use a minimal reset-only harness. No full-state API exists.

Validation: all 47 tests passed; Ruff lint/format checks and strict mypy passed
(24 source files). Full game conformance awaits the remaining Phase 2 tasks.

## Phase 2, Task 2 — Tic-Tac-Toe legal actions and validation

Completed on 2026-10-14:

- Defined `place` actions with one integer `cell` parameter, indexed 0–8.
- Implemented legal-action queries with stable row-major ordering, initialization
  and seat validation, and empty tuples for inactive seats or finished games.
- Added an internal placement validator for Task 3's step implementation. It
  rejects incorrect kinds/parameters, boolean and noninteger indices, out-of-range
  cells, occupied cells, and terminal actions without changing state or RNG.
- Added 26 tests for initial and occupied boards, both actors, invalid seats,
  malformed game-specific actions, terminal rejection, and query/validation
  purity. Updated the reset harness to use production legal-action queries.

Validation: all 73 tests passed; Ruff lint/format checks and strict mypy passed
(25 source files). Move execution remains abstract; full engine conformance
awaits the remaining Phase 2 tasks.

## Phase 2, Task 3 — Tic-Tac-Toe state transitions and player switching

Completed on 2026-10-14:

- Implemented validated placements using the current seat's mark and a fresh
  immutable board/state, preserving all unselected cells and prior snapshots.
- Successful placements switch actors and return a nonterminal `StepResult`
  with zero incremental rewards, matching the latest-reward query, and no events.
- Validation precedes mutation; successful and rejected steps consume no RNG.
- Added 25 tests covering every first-move cell, alternating placements by X/O,
  updated legal choices, reset after actual moves, latest rewards, and atomic
  rejection of occupied cells, invalid actions, and stored terminal states.
- Removed the reset harness's step stub so tests exercise the production method.

Validation: all 98 tests passed; Ruff lint/format checks and strict mypy passed
(26 source files). The engine remains abstract. Transitions currently handle
nonterminal play only; complete games require Task 4's win/draw detection and
terminal rewards, followed by observations and cloning.

## Phase 2, Task 4 — Tic-Tac-Toe terminal detection and rewards

Completed on 2026-10-14:

- Added detection of all row, column, and diagonal wins after placement, plus
  full-board draws, with wins taking precedence on the ninth move.
- Terminal results set phase `finished` and no next actor. Implemented
  `is_terminal`; terminal legal choices and step rejection use the stored outcome.
- Wins award +1/-1 by seat; draws and nonterminal moves award zero. Latest-step
  rewards remain available on repeated queries and are cleared by reset.
- Added 19 tests using actual move trajectories for both players on all eight
  winning lines, a draw, a ninth-move win, terminal lifecycle, and reset after a win.
- Removed the test harness's terminal-query stub to exercise production code.

Validation: all 117 tests passed; Ruff lint/format checks and strict mypy passed
(27 source files). Observations and cloning remain abstract; full interface
conformance awaits those implementations.

## Phase 2, Task 5 — Tic-Tac-Toe observations and board rendering

Completed on 2026-10-14:

- Added a separate immutable public board observation type and fixed the engine's
  generic observation contract to that type.
- Implemented fresh player views for both valid seats during and after play,
  with lifecycle/seat validation and no authoritative state, rewards, or RNG data.
- Added a pure exported `render_board(observation)` formatter and
  `engine.render(player_id)`, using X/O/dots in row-major board orientation.
- Added 12 tests for visibility schema, invalid seats, immutable snapshots,
  terminal views, decision inputs, rendering, and state/reward/RNG query purity.
- Removed the harness's observation stub to exercise production views.

Validation: all 129 tests passed; Ruff lint/format checks and strict mypy passed
(30 source files). The engine remains abstract for cloning; complete reusable
conformance checks await Task 7.

## Phase 2, Task 6 — Tic-Tac-Toe rule and deterministic-reset tests

Completed on 2026-10-14:

- Audited focused coverage of all winning lines for both seats, draws, invalid
  moves, terminal rejection, and reset lifecycle.
- Added an independent coordinate-based rule oracle and a bounded traversal
  of all 5,478 distinct reachable board positions. Verified 626 X-win boards,
  316 O-win boards, and 16 draws, with consistent actors, phase, rewards,
  public inputs, legal choices, and terminal rejection.
- Added 12 full-game replay cases across four seed settings for X wins, O wins,
  and draws, including inserted illegal moves and replay after terminal reset.
- No production changes or cloning implementation were needed.

Validation: all 142 tests passed, including 13 new regression cases; Ruff
lint/format checks and strict mypy passed (31 source files). The reachable-board
audit covers distinct positions rather than every possible move ordering.
Shared clone-dependent conformance tests remain for the final Phase 2 task.

## Phase 2, Task 7 — Tic-Tac-Toe cloning and phase completion

Completed on 2026-10-14:

- Implemented clone using deep copying after initialization validation, preserving
  concrete type, state, latest rewards, and independently owned RNG state.
- Made the production engine concrete and removed the harness's final stub.
- Added 10 tests exercising shared engine/rejection conformance, divergent moves,
  independent resets, matching continuations, terminal clones, subclass copying,
  and independent progressed RNG sequences.
- Updated README with a runnable complete-game example and clone usage.

Final local validation: all 152 tests passed; Ruff lint/format checks, strict mypy
(32 source files), and pip check passed. Hosted CI remains unexecuted because the
repository has no GitHub remote; Linux verification awaits hosted execution.

**Phase 2 complete.** Tic-Tac-Toe supports reset, legal actions, validated moves,
player switching, wins/draws, rewards, safe public observations, text rendering,
and independent clones. The rule audit covers all 5,478 distinct reachable boards,
and the concrete engine passes shared interface/rejection conformance checks.

## Phase 3, Task 1 — RandomAgent

Completed on 2026-10-14:

- Added and exported generic RandomAgent using the existing Agent interface.
- Uniformly selects supplied legal actions without inspecting observations or
  accessing engines; uses a private generator and shared seed validation.
- Construction and reset accept integer seeds or system-seeded None. Invalid
  resets preserve the sequence; empty choices reject before an RNG draw.
- Added 16 cases covering seeded sequences and reset, supplied-action identity,
  mutable observation preservation, invalid seeds, independent agent/global/engine
  RNGs, hidden-world equivalence, and complete seeded Tic-Tac-Toe replays.
- Updated roadmap, architecture, visibility policy, and README usage example.

Validation: all 168 tests pass; Ruff lint/format checks, strict mypy (34 source
files), and pip check pass. Hosted CI remains unexecuted without a GitHub remote.
Only Task 1 is complete; no production runner or heuristic agent was added.

Next: Phase 3, Task 2 — implement a simple heuristic agent.

## Phase 3, Task 2 — Simple heuristic agent

Completed on 2026-10-14:

- Added and exported TicTacToeHeuristicAgent using Agent[TicTacToeObservation].
- Uses only public decision inputs to win, block, then prefer center/corners/edges;
  ties are deterministic and selections return supplied actions without mutation.
- Stateless policy inherits no-op reset, consumes no RNG, and rejects incompatible
  choices or nonactor observations. Fork handling and optimal search are excluded.
- Added 55 cases covering win/block tactics for both seats on all eight lines,
  win precedence, preferences, order-independent ties, input and RNG preservation,
  failure cases, complete seeded games against RandomAgent in both seats, and a
  heuristic self-play draw.
- Updated roadmap, architecture, information boundaries, and README usage.

Validation: all 223 tests pass; Ruff lint/format, strict mypy (36 source files),
and pip check pass. Hosted CI remains unexecuted without a GitHub remote.
Only Task 2 was completed in this iteration.

Next: Phase 3, Task 3 — build the reusable game loop and match runner.

## Phase 3, Task 3 — Reusable game loop and match runner

Completed on 2026-10-14:

- Added generic play_game for initialized positions and run_match for reset and
  single-game execution, exported from boardgame_ai.simulation with MatchResult.
- Follow engine actors and detached decision inputs, support repeated actors,
  sum each executed step's rewards, and bound decisions with explicit max_steps.
- Separate engine and per-seat agent seeds; validate setup before resets and
  require distinct agents per seat. Truncation leaves a resumable live position.
- Propagate exceptions and detect engine contract failures without substituting
  actions. Results contain no observations, states, or private events.
- Added 23 cases for lifecycle, repeated actors, general-sum incremental rewards,
  terminal initial states, truncation/resumption, invalid setup/seeds, hidden-card
  chance and input mutation, failure paths, and seeded baseline match replay.
- Updated roadmap, architecture, information policy, and README runner example.

Validation: all 246 tests pass; Ruff lint/format checks, strict mypy (39 source
files), and pip check pass. Hosted CI remains unexecuted without a GitHub remote.
Only Task 3 is complete in this iteration; batch scheduling, metric aggregation,
human CLI play, and replay logging remain pending.

Next: Phase 3, Task 4 — run batches of matches with reproducible seeds.

## Phase 3, Task 4 — Reproducibly seeded match batches

Completed on 2026-10-14:

- Added run_batch and SeededMatchResult, reusing the single-match runner with
  fixed seats and separate engine and per-seat lifecycle seeds for every match.
- Versioned SHA-256 derivation uses integer batch seed, absolute match index,
  and engine/seat stream; no global RNG or game-dependent seed consumption.
- Supports explicit index ranges and individual-match reruns from recorded seeds.
  Validate before resets, preserve no-op empty batches, and stop on exceptions.
- Added 18 cases for batch replay, individual matches, subranges, independent
  streams, length independence, resets, hidden chance, global RNG isolation,
  invalid configuration, empty batches, and fail-fast errors.
- Added scripts/verify_batches.py. With batch seed 20261014, random/random,
  heuristic/random, and random/heuristic each completed 1,000 matches and an
  identical rerun: 6,000 successful executions, comparing summaries and final
  boards. This is reproducibility verification, not a strength benchmark.
- Updated roadmap, architecture, boundaries, and README usage.

Validation: all 264 tests pass; Ruff lint/format, strict mypy (41 source files,
plus the verification script), and pip check pass. Hosted CI remains unexecuted
without a GitHub remote. Only Task 4 is complete in this iteration.

Next: Phase 3, Task 5 — record wins, draws, scores, errors and game lengths.
Tasks 6–7 (human CLI play and replays/logs) also remain before Phase 3 completion.

## Phase 3, Task 5 — Record outcomes, scores, errors, and game lengths

Completed on 2026-10-14:

- Added evaluate_batch with frozen outcome, per-match record, summary, and report
  types. Reuses lifecycle seeds and detached agent inputs; run_batch stays fail-fast.
- Added explicit game outcome adapter for Tic-Tac-Toe terminal public boards,
  with +1/-1/0 result scores. Generic scores/winners are separate from rewards.
- Count wins/draws, completed/truncated/error attempts, totals/mean scores, and
  min/mean/max decision lengths. Undefined averages use None. Failures and
  truncations never become draws; partial failed-run lengths remain unknown.
- Record only fixed error categories, continue via next-match reset, and omit
  exception content/private inputs. Explicit schema-versioned JSON saves metrics.
- Added 25 tests for outcomes, score/reward separation, multiplayer winners,
  denominators, errors/recovery, adapter failures, interrupts, JSON, and validation.
- Added reproducible benchmark script and BASELINE_BENCHMARKS.md. Three 1,000-game
  reports and reruns agree (6,000 executions), with no errors or truncations.
  Random/random wins: 599/273 with 128 draws; heuristic/random: 961/0 with 39 draws;
  random/heuristic: 10/842 with 148 draws. Full metrics and seed context are documented.
- Updated roadmap, architecture, boundaries, and README usage.

Validation: all 289 tests pass; Ruff lint/format, strict mypy (44 source files
and both scripts), and pip check pass. Pytest's JSON export temporary-directory
test required execution outside the sandbox after a temporary-path access failure.
Hosted CI remains unexecuted without a GitHub remote. Only Task 5 was completed.

Next: Phase 3, Task 6 — command-line human-versus-agent mode.
Task 7 (save/inspect replays and logs) remains afterward.

## Phase 3, Task 6 — Command-line human-versus-agent mode

Completed on 2026-10-14:

- Added play command with human X/O, random/heuristic opponent, independent
  engine/agent seed options, and public board/cell-map presentation.
- Added CLI human Agent adapter that returns supplied legal placements, and an
  opponent presentation wrapper. Both use the unchanged run_match/step path.
- Retry invalid text and unavailable cells without steps or RNG consumption;
  q/quit/exit, EOF, and Ctrl+C cancel cleanly without declaring a game result.
- Show final public board and outcome using the existing Tic-Tac-Toe adapter.
- Added 24 cases for input purity, retries, quitting, both-seat outcomes, real
  seeded opponents, cancellation, argument validation, and command help.
- Updated roadmap, architecture, information boundaries, and README usage.

Validation: all 313 tests pass; Ruff lint/format, strict mypy (46 source files
and both scripts), and pip check pass. Module and console-script play entry points
were smoke-tested with help and cancellation. Full pytest used approved execution
outside the sandbox for its JSON export temporary-directory test. Hosted CI still
awaits a GitHub remote. Only Task 6 was completed; no replays/logs were added.

Next: Phase 3, Task 7 — save and inspect game replays/logs, then assess Phase 3
completion before starting Phase 4.

## Phase 3, Task 6 follow-up — Starter and available-number display

Completed on 2026-10-14 at the user's request after playing:

- Added --starts human/ai/X/O independently of --human; default stays X-first.
- Added validated engine starting_player configuration, used on reset and
  preserved by cloning. O-first placements, wins, draws, and rewards keep seats fixed.
- Show available cell numbers beside the game board, with dashes for unavailable
  cells, on each prompt/retry and the final display.
- Added 23 regression cases for starter combinations, complete O-first CLI/engine
  games, reset/clone configuration, invalid starters, and exact display alignment.
- Documented why deterministic tactical play can draw repeatedly without being
  unbeatable. Baseline benchmark configuration remains default X-first.

Validation: all 336 tests pass; Ruff lint/format, strict mypy (47 source files),
script typing, and pip check pass. CLI starter/display smoke check passed. Full
pytest used approved execution for temporary-directory access. Task 7 remains next.

## Phase 3, Task 7 — Public replay logs and phase completion

Completed on 2026-10-14, including the user's CLI refinements:

- Removed CLI X/O player choices: --starts human/ai assigns X to the starter.
  Added --game tic-tac-toe as explicit supported-game selection (also the default).
- Added optional runner on_step callback, called after successful validated steps.
  Added game-specific recorder, frozen replay/move records, strict version-1 JSON
  save/load, and public board reconstruction through engine validation.
- Added --save-replay PATH and replay PATH. Save completed games or cancelled
  prefixes; inspect only after validation. No RNG seeds, private data, reward
  tuples, or snapshot histories are exported. Hidden-game replay work is deferred.
- Added 28 replay tests covering X/O-first wins/draws, round trips, callbacks,
  truncation/resumption, cancellation, strict schema/actor/cell/status validation,
  preservation of files on invalid save, and CLI saving/inspection.
- Reworked CLI tests for human/ai options and explicit game selection, removing
  obsolete CLI O-first cases; programmatic O-first engine tests remain.
- Updated roadmap, architecture, information policy, and README workflows.

Validation: all 360 tests pass; Ruff lint/format checks, strict mypy (49 source
files and both scripts), pip check, complete CLI save/inspect smoke checks, and
the 6,000-execution baseline benchmark reproduction pass. Full pytest used
approved execution for temporary files. Hosted CI remains unexecuted without a remote.

**Phase 3 complete.** All seven tasks and local acceptance criteria are met.
No Phase 4 implementation was started.

Next: Phase 4, Task 1 — encode observations as PyTorch tensors.

## Project plan v0.2 — Forward planning and position evaluation

Integrated on 2026-10-14 from the user-provided downloaded plan:

- Added engine/evaluator/planner responsibilities, engine-based search versus
  MuZero, an atomic forward-simulation example, proposed simulation/evaluator
  capabilities, hidden-information safeguards, and search success criteria.
- Added future evaluator/rollout/MCTS experiment tasks in Phase 7; preserved all
  completed tasks and current implementation order. Search-first suggestions are
  documented as an optional experiment track, not a prerequisite for Phase 4.
- Reconciled suggested clone behavior with existing copied/independent RNG state
  and future simulation contexts with the observation-only Agent boundary.
- Kept BOARDGAME_AI_PROJECT_PLAN.md as the canonical roadmap and corrected the
  immediate-next-steps summary to reflect completion of Phases 1–3.

Documentation-only update; no implementation or completion status changed.
Verified the imported section, phase checkboxes, numbering, and next task.
Next remains Phase 4, Task 1 — encode observations as PyTorch tensors.
