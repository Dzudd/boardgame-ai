# Progress

Current status: Phases 1–4 are complete under the minimal random-baseline
criterion. All 617 tests and development checks pass. The unchanged checkpoint
improves balanced score over a random control by +0.3160 (approximate 95% interval
[+0.2644,+0.3676]), but heuristic robustness remains poor. Next roadmap task is
Phase 4.5, Task 1: read-only tactical and learning-signal diagnosis. Phase 4.5
is planned, not implemented; Phase 5 has not started. Training remains latest
shared-policy self-play only, with MuZero-style search learning as the long-term
direction. The latest change adds a planned Phase 4.5 throughput task and records
the preliminary CPU/CUDA timing probe; no optimization has been implemented.
Published privately at [Dzudd/boardgame-ai](https://github.com/Dzudd/boardgame-ai),
with `master` tracking `origin/master`. Hosted Windows/Linux CI is enabled; current
results are available in [GitHub Actions](https://github.com/Dzudd/boardgame-ai/actions).
Entries below are historical
task records; earlier references to incomplete behavior describe that task's stage.
Earlier statements about an absent remote or unexecuted hosted CI also describe
the state at that task's completion; see the publication record below.

## 2026-10-15: CPU/CUDA timing probe and planned throughput work

- Verified train --device cuda --check-device selects RTX 4070 Ti with PyTorch
  2.14.1+cu130. Existing 1,000-update logs identify cuda:0; collection/inference
  and GAE deliberately remain CPU-side. Correct placement is not utilization.
- Ran two temporary fresh-policy updates per device (CUDA first, then CPU),
  64 games/update, default PPO, seed 42, eight CPU threads. No existing training
  checkpoint was used or changed. Temporary outputs were removed automatically.
  Timing wrappers synchronized CUDA after measured calls when initialized.
  The second update had 483 decisions and 32 optimizer steps on each device:

  | Stage | CUDA run | CPU run |
  |---|---:|---:|
  | Collection including inference | 0.176 s | 0.208 s |
  | GAE / batch preparation | 0.042 s | 0.040 s |
  | PPO optimization | 0.183 s | 0.068 s |
  | Checkpoint save | 0.007 s | 0.004 s |
  | End-to-end update | 0.413 s | 0.325 s |

  First-update totals were 2.180 s CUDA / 0.313 s CPU and include startup costs.
  Only one post-warm-up sample per device was measured; order, runtime warm-up,
  instrumentation, and changing policies limit inference. This does not establish
  sustained performance, original-run duration, or an expected speedup.
- Added Phase 4.5 Task 3 for repeated profiling, synchronization reduction, and
  batched inference assessment. Existing learning-experiment/search/review tasks
  move to Tasks 4–6; Task 1 remains the immediate next task. Preserve failure
  checks, information boundaries, RNG ownership, targets, and resume guarantees.
  Larger minibatches/changed sampling require learning-quality comparisons.
- Updated roadmap, architecture, README, and this record. This follow-up changes
  documentation only; it implements no optimization and launches no new training.

Validation: local Markdown links/heading targets and diff whitespace checked.
No test rerun for this documentation-only change; prior suite result is 617
passing tests. The bounded timing workload above ran before this doc update.

## 2026-10-15: Self-play learning direction and Phase 4.5 planning

Documentation-only update after discussion of the weak trained policy:

- Kept Phase 4's original minimal acceptance and recorded results intact. Added
  planned Phase 4.5 to diagnose tactical/credit-assignment failures, add periodic
  checkpoint monitoring and isolated evaluations, compare justified self-play
  improvements across bounded multi-seed runs, and specify the search-learning
  bridge before choosing the next milestone. No claim that the weakness is fixed.
- Established self-play as the sole training source: latest frozen shared policy
  in both seats, no external/perfect/heuristic training opponents, expert labels,
  or historical opponent pool. Existing baseline evaluations stay measurement only.
  No perfect solver is required. The earlier opponent-mixture recommendation is
  superseded; historical records below retain their original task context.
- Clarified actual PPO behavior against run.py, rollout.py, ppo.py and the saved
  training manifest: 64 complete games per collection, both-seat GAE, all collected
  decisions shuffled into minibatches of 64 for four epochs, no updates during
  gameplay. This is batching, not subsampling one or two moves per game. Categorical
  sampling and fixed .01 entropy are present; epsilon exploration/replay are absent.
- Recorded future position/sequence replay as a separately designed search learner,
  with self-generated search policies/outcomes and explicit sample/compute budgets.
  It must preserve credit/sequence context; stale trajectories are not ordinary PPO.
- Promoted MuZero-style learning to the long-term direction, via permitted
  rule-based search validation and search-guided self-play. Search, replay learning,
  learned dynamics, checkpoint archives, and automatic evaluation remain unimplemented.
- Updated roadmap, architecture, README, information boundaries, and result-report
  follow-ups. Historical benchmark values and checkpoint identity are unchanged.

Validation: reviewed documentation against current source and training manifest;
checked Markdown links, current-task references, and git diff whitespace. No code
or tests changed in this documentation task, and the suite was not rerun. The
previous encoding-audit validation remains 617 passing tests plus development
checks. No training, checkpoint edits, commit/push, or hosted CI execution occurred.

Next: Phase 4.5, Task 1 only — diagnose the frozen policy's public-board tactics
and trace representative per-seat learning signals before changing training.

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

## 2026-10-15: Phase 4 status review and observation encoding audit

- Reviewed the roadmap, architecture, information boundaries, progress, README,
  existing encoder/tests, and Phase 4 results. All eight Phase 4 tasks already
  exist; the random-baseline acceptance criterion is met for the recorded run,
  while near-universal heuristic losses remain unresolved.
- Kept the documented encoding design and production implementation unchanged.
  Added an exhaustive public-observation test over 5,478 reachable boards,
  both observer seats, and simultaneous mark/actor/observer relabeling (O-first).
  It verifies CPU float32 `(6, 3, 3)` layout, binary/partition properties,
  lossless visible-board recovery, turn planes, and all 958 terminal boards.
- Enumeration uses an independent coordinate-based outcome oracle without
  engine state access. Existing isolation/schema tests remain in place; this
  public-game audit makes no hidden-information or policy-strength claim.
- Updated roadmap, architecture, visibility documentation, README, and progress.

Validation: 27 targeted encoding tests pass; full suite 617 passed, no skips.
Ruff lint/format, strict mypy (78 files), and pip check pass. Full pytest used
approved execution outside the sandbox after the initial targeted invocation
was slow; that initial invocation also finished successfully. No training,
checkpoint changes, or hosted CI run occurred.

Next roadmap task: Phase 5, Task 1 — multiple atomic decisions within a turn.
If continuing strength work first, the next small task is a public-observation
tactical failure audit of the frozen policy; diagnose before changing training.

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

## Initial GitHub publication — Phases 1–3 and roadmap v0.2

Published on 2026-10-14:

- Created the private personal repository `Dzudd/boardgame-ai` after confirming
  that it did not already exist. Preserved the existing `master` branch; the local
  repository had no earlier commits or remotes.
- Committed the complete foundation as `3e758b3`, covering Phases 1–3, public
  replay support, baseline measurements, and the v0.2 forward-planning roadmap.
  No Phase 4 implementation is included. Package version remains `0.1.0`;
  `0.2` is the roadmap version.
- Reviewed all 61 committed paths and checked for credential patterns. Expanded
  ignore rules for credentials, machine configuration, environments, caches,
  generated runs/data, and model artifacts. Local artifacts remain untracked.
- Re-ran all 360 tests, Ruff lint/format, strict mypy for the package/tests and
  both scripts, dependency checks, CLI diagnostics/help, batch verification,
  and baseline metric reproduction. All passed. Pytest required execution
  outside the sandbox to access its temporary directories.
- Configured `origin`, pushed `master`, and verified the remote commit matched
  local HEAD. The initial push triggered Windows and Linux jobs in
  [CI run 38636204906](https://github.com/Dzudd/boardgame-ai/actions/runs/38636204906).
  Both were in progress when first inspected; use Actions for final results.
- Updated current documentation to record publication and hosted CI, while
  retaining earlier task records as historical context.

Next remains Phase 4, Task 1 — encode observations as PyTorch tensors.

## Phase 4, Task 1 — Observation tensor encoding

Completed on 2026-10-14:

- Added pure `encode_tic_tac_toe_observation`, exported from
  `boardgame_ai.models.encoders`, using the existing filtered Observation.
- Defined fresh contiguous CPU float32 tensors shaped `(6, 3, 3)` with own,
  opponent, empty, own-turn, opponent-turn, and terminal planes. Perspective is
  observer-relative with fixed row-major coordinates, supporting inactive seats.
- Finished views retain their board with zero turn planes and a one terminal
  plane. Schema errors raise ValueError without exposing input contents;
  validation does not attempt engine rules or reachability checks.
- Kept authoritative state, rewards, seeds, RNG, and legal actions outside encoder
  inputs; core/agent/engine interfaces and GameSpec are unchanged.
- Added 26 tests for exact channel values/orientation, seat perspective, both
  starters, wins/draws, actor and inactive observations, independent storage,
  deterministic snapshot behavior, PyTorch RNG purity, and schema/state rejection.
- Updated roadmap, architecture, information boundaries, and README example.

Validation: all 386 tests pass; Ruff lint/format, strict mypy (53 source files
plus both scripts), pip check, and CLI numerical diagnostics pass. Full pytest
required approved execution outside the sandbox because existing temporary-file
fixtures could not create directories inside it. Hosted CI was not run this task.
Only Task 1 is complete; no action masking, networks, training, or search was added.

Next: Phase 4, Task 2 — encode legal actions and implement action masking.

## Phase 4, Task 2 — Legal-action encoding and masking

Completed on 2026-10-14:

- Added fixed Tic-Tac-Toe action indices 0–8, fresh CPU bool legal masks `(9,)`,
  and decoding that returns the matching supplied Action object.
- Kept row-major coordinates identical for both players. Supplied legal tuples
  determine availability; no board/engine query or legality inference is used.
- Empty terminal/inactive choices encode as all-false masks. Schema errors,
  duplicate cells, invalid/unavailable indices reject with ValueError.
- Added differentiable `mask_action_logits` with exact shape/device validation,
  finite floating-point logits, bool masks, and at least one legal entry per row.
  Illegal logits become -inf and have zero gradients; legal values/gradients
  preserve dtype/device. Inputs and RNG remain unchanged. Batched inputs work.
- Added 44 tests for all cell round-trips, object identity, ordering and storage,
  both-seat complete games, inactive/terminal choices, probabilities/gradients,
  malformed action schemas, empty rows/batches, and shape/dtype/device failures.
- Updated roadmap, architecture, visibility policy, and README usage.

Validation: all 430 tests pass; Ruff lint/format, strict mypy (56 source files
plus both scripts), pip check, and CLI diagnostics pass. Full pytest used approved
execution outside the sandbox for existing temporary-directory fixtures. Hosted
CI was not run this task. Only Task 2 was completed; no networks, trajectory
collection, training, or search was added.

Next: Phase 4, Task 3 — implement policy and value networks.

## Phase 4, Task 3 — Policy and value networks

Completed on 2026-10-14:

- Added exported `TicTacToeActorCritic` nn.Module: shared 54→64→64 tanh body,
  linear nine-action policy head, and unbounded scalar value head.
- Defined nonempty input shape `(..., 6, 3, 3)` and output tuple shapes
  `(..., 9)`/`(...)`, supporting single views and arbitrary batch dimensions.
- Both heads consume only player-visible encoded observations. Value perspective
  is the observing player; masking remains separate and uses supplied choices.
- Construction uses CPU float32 and standard caller-seeded PyTorch initialization.
  Explicit model/input dtype/device conversion is supported; malformed, empty,
  nonfinite, or mismatched inputs reject without printing their values.
- Terminal tensors produce raw predictions; future trajectory collection must
  bootstrap zero at true termination. Terminal/inactive masks still forbid actions.
- Added 25 tests for batching, raw logits/unbounded values, both-head/shared-body
  gradients, masking, seeded initialization, inference purity, dtype conversion,
  equivalent player views, both starters, terminal inputs, and tensor rejection.
- Updated roadmap, architecture, information boundaries, and README usage.

Validation: all 455 tests pass; Ruff lint/format, strict mypy (58 source files
plus both scripts), pip check, and CLI diagnostics pass. Full pytest used approved
execution outside the sandbox for existing temporary-file fixtures. Hosted CI
was not run this task. Only Task 3 was completed; no neural Agent adapter,
trajectory collection, GAE, optimizer, checkpoint workflow, training, or search
was added. The model is untrained; playing strength has not been evaluated.

Next: Phase 4, Task 4 — implement trajectory collection and advantage estimation
(GAE).

## Phase 4, Task 4 — Trajectory collection and advantage estimation

Completed on 2026-10-14:

- Added bounded reset-to-boundary Tic-Tac-Toe collection and separate per-seat
  trajectories: visible observation tensors, supplied-choice masks, sampled action
  indices, old log probabilities, values, interval rewards, and global step indices.
- Used independent per-seat CPU sampling generators; shared or separate models
  receive only filtered tensors. Collection changes no global RNG, weights,
  gradients, or model train/eval modes. Records own detached CPU tensor storage.
- Defined per-player decision-clock discounting. A decision receives that seat's
  incremental rewards until its next decision or rollout end, including rewards
  from opponent steps. Pre-first-decision rewards appear only in total_rewards.
- Implemented detached GAE and lambda-return targets over one player's segment,
  with explicit gamma/lambda/bootstrap, empty-vector handling, and validation.
- Forced zero bootstrap at true termination without terminal inference; live
  step-limit cutoffs use each nonempty seat's filtered view, including inactive
  seats. Termination beats a coinciding limit. Every collection call resets.
- Added 48 tests for independent GAE calculations, coefficient/shape/overflow
  failures, both-seat wins/losses/draws, repeated actors and general-sum rewards,
  terminal/cutoff/empty segments, old log probabilities, reproducible sampling,
  storage/RNG/gradient isolation, seed/config validation, and failure propagation.
- Updated roadmap, architecture, visibility policy, and README usage/contracts.

Validation: all 503 tests pass; Ruff lint/format, strict mypy (62 source files
plus both scripts), pip check, and CLI diagnostics pass. Full pytest used approved
execution outside the sandbox for existing temporary-file fixtures. Hosted CI
was not run this task. Only Task 4 was completed; no PPO losses, optimization,
checkpointing, baseline opponent schedule, training updates, or search was added.
Models remain untrained; no playing-strength claim is made.

Next: Phase 4, Task 5 — implement PPO losses, optimization, and checkpointing.

## Phase 4, Task 5 — PPO losses, optimization, and checkpointing

Completed on 2026-10-14:

- Added immutable PPO configuration, owned detached on-policy batches, and
  per-player-segment GAE preparation before sample concatenation.
- Implemented clipped policy loss, weighted half-MSE value loss, legal-only
  entropy, approximate KL, and clip fraction. Old targets detach defensively;
  illegal entropy terms avoid 0 times negative infinity.
- Added bounded CPU float32 Adam updates with private seeded minibatch shuffling,
  optional full-batch advantage normalization, partial minibatches, nonfinite
  checks, gradient clipping, and update/optimizer-step counters/diagnostics.
- Added version-1 checkpoints with complete model/config/Adam state, mode,
  counters, and shuffle RNG. Restricted loading validates compatibility before
  returning a fresh CPU trainer; construction preserves global Torch RNG.
- Saving validates then atomically replaces the explicit path. Checkpoints omit
  engine/rollout data, sampling seeds, gradients, and global RNG. Exact continuation
  is at an update boundary with the same next batch/runtime, not mid-episode.
- Added 54 tests covering hand-calculated clipping/value/entropy, detached and
  illegal gradients, actual parameter updates and gradient bounds, per-segment
  targets, normalization/validation, exact restored continuation, corrupted
  checkpoints, and preservation of files on invalid saves.
- Updated roadmap, architecture, information boundaries, and README examples.

Validation: all 557 tests pass; Ruff lint/format, strict mypy (66 source files
plus both scripts), pip check, and CLI diagnostics pass. Full pytest used approved
execution outside the sandbox for temporary-file fixtures. Hosted CI was not run
this task. Only Task 5 was completed; updates were exercised in tests, but no
self-play/baseline training schedule, strength experiment, held-out evaluation,
or search was added. PPO requires caller-selected samples from the learner policy;
other-policy opponent trajectories must be excluded.

Next: Phase 4, Task 6 — train with self-play and/or baseline opponents.

## Phase 4, Task 6 — User-run self-play and CUDA support

Implemented on 2026-10-14; full training deliberately left to the user:

- Added train CLI with CUDA default, explicit cpu/cuda:N selection, device-only
  preflight, budgets, PPO flags, output protection, progress, and resume.
- Extended PPO tensors/losses/Adam from CPU-only to CPU/CUDA float32 with explicit
  batch/index transfers. Parameters, forward/backward, clipping, and Adam use
  the selected device; engines, frozen-policy inference, sampling, and GAE stay CPU.
- Added seeded shared-policy self-play collection/update scheduling. Both seats'
  segments are eligible; no baseline-opponent samples are mixed into PPO.
- Added immutable run manifest, per-update portable latest.pt, and trusted JSONL
  diagnostics/self-play outcomes. Absolute update/game seed derivation and saved
  optimizer/shuffle state reproduce uninterrupted execution in CPU unit tests.
- Extended checkpoint loading with explicit device selection and portable CPU
  serialization for GPU weights/Adam moments. Existing version-1 CPU files remain
  supported; same-runtime/device identity is tested, cross-device identity is not.
- Added 24 tests: 22 passed for tiny CPU orchestration, resume, CLI/preflight,
  output/config validation, RNG preservation; 2 real CUDA tests skipped because
  a CUDA-enabled runtime is unavailable. No full experiment was launched.
- Updated roadmap, architecture, information boundaries, and README installation,
  launch, tuning, interrupt/resume, and artifact instructions.

Validation: 579 tests pass and 2 conditional CUDA tests skip (581 collected).
Ruff lint/format, strict mypy (70 source files plus both scripts), pip check,
CLI diagnostics/train help, and CPU device-only preflight pass. Full pytest used
approved execution outside the sandbox for temporary files; only tiny test runs
executed updates. Hosted CI and CUDA execution were not performed this task.

Hardware diagnostic: NVIDIA GeForce RTX 4070 Ti, driver 610.88. The installed
PyTorch is 2.14.1+cpu with CUDA unavailable. The user must install an appropriate
CUDA-enabled PyTorch build and verify train --device cuda --check-device before
launching. No dependencies or drivers were changed in this session.

**Task 6 implementation ready; training experiment pending user execution.**
The roadmap training checkbox remains unchecked and no playing-strength claim
is made. Next: the user runs training, then Task 7 adds fixed evaluation opponents
and held-out seeds. Baseline training scheduling remains optional future work.


## Windows CUDA investigation and repair

Completed on 2026-10-14; no full training experiment launched:

- Reverified RTX 4070 Ti (12 GB), NVIDIA driver 610.88 (CUDA driver support 13.3),
  Python 3.12.14, and project-local pip/interpreter/import paths. The venv was
  isolated from system/user packages; no conflicting torch distribution or
  environment override was found. Bare python is a WindowsApps alias.
- Root cause: README setup installed torch 2.14.1+cpu; its CUDA runtime was None,
  availability False, device count zero. Existing PPO transfers and fail-fast
  CUDA selection were correct; no demonstrated application defect required fixing.
- Checked current official PyTorch Windows instructions, official wheel availability,
  and NVIDIA driver compatibility. Replaced only torch with 2.14.1+cu130 using
  the official CUDA 13.0 index, --force-reinstall --no-deps. Python, driver,
  dependency ranges, and other installed packages were preserved. No Toolkit needed.
- Actual GPU diagnostics: torch.version.cuda=13.0, is_available=True, one device,
  NVIDIA GeForce RTX 4070 Ti, capability 8.9. A CUDA 4x4 tensor matmul returned the
  expected value 14.0. Both requested info and train --device cuda --check-device
  commands passed; preflight selected cuda:0 without training/output creation.
- Strengthened the two conditional CUDA tests: every PPO batch field, model
  parameter, finite gradient, and Adam exp_avg/exp_avg_sq resides on CUDA during
  optimization. Portable checkpoint model/optimizer tensors serialize on CPU;
  loading restores CUDA moments and reproduces the next update exactly. Tiny
  GPU runner save/load/resume matches uninterrupted parameters, optimizer, counters,
  and shuffle state. Adam's scalar step bookkeeping may remain on CPU by design.
- Targeted CUDA selection: 6 passed, 575 deselected, zero skipped. Both real
  hardware tests executed. Full suite: 581 passed, zero skipped (14.10 seconds).
  Ruff check and format check passed (72 files); strict mypy passed (70 source
  files); pip check found no broken requirements. Hosted CI was not triggered.
- Updated README setup/repair commands and compatibility explanation, architecture,
  roadmap, and this progress record. No player-visible information boundary changed;
  CPU simulation/inference/GAE and CUDA PPO optimization remain intentional.

Limits: only small diagnostics/tests ran; no full experiment, speed benchmark,
held-out evaluation, or strength claim. Task 6 training remains pending user execution.

## Phase 4, Task 6 run completion and Task 7 — Independent evaluation

Completed on 2026-10-15:

- User confirmed 1,000 completed CUDA updates in runs/tic-tac-toe-cuda/latest.pt.
  The saved checkpoint reports 29,020 optimizer steps; training configuration
  has seed 42 and 64 games per update (64,000 self-play games). No additional
  training experiment was launched by the agent.
- Added frozen CPU TicTacToePolicyAgent using the existing Agent interface,
  player-visible encoders/legal masks, private resettable sampling RNG, and
  explicit sampled/greedy modes. Source model state/mode/gradients remain untouched.
- Added fixed random/heuristic evaluation cases balanced across X/O, with separate
  versioned held-out RNG seed namespaces, deterministic absolute-index shards,
  immutable checkpoint byte identity/counters, and trusted JSON match summaries.
- Added evaluate CLI using existing match/outcome adapters. No source checkpoint,
  training configuration, or training log is written. CLI protects known training
  artifacts from report overwrite and surfaces errors/truncations without private
  exception text. Evaluation runs entirely on CPU without optimizer updates.
- Added 21 tests for model/source snapshots, masked legal selection, RNG/reset,
  both seats, fixed cases, seed separation, reproducibility/sharding, checkpoint
  source replacement, sanitized failures, and CLI output protections.
- After training finished, evaluated the unchanged checkpoint on 4,000 games
  with held-out root seed 20261015, sampled mode, 1,000 per opponent/learner seat:
  random X 770/66/164 and O 484/73/443 W/D/L; heuristic X 0/3/997 and O 0/0/1000.
  All games completed without errors/truncations. Identity/protocol/limitations
  are recorded in POLICY_EVALUATION.md and the generated trusted JSON report.
- The results expose tactical weakness hidden by the final self-play outcomes.
  No model/configuration change or claim of robust/optimal play was made.
- Updated roadmap, architecture, information boundaries, README, and progress.

Validation: initially 600 CPU tests passed with 2 CUDA workloads deliberately
excluded while the user's training was active. After completion, the full suite
passed all 602 tests with no skips. Ruff lint/format, strict mypy (74 source files
plus both scripts), and pip check pass. Pytest and actual CPU evaluation required
approved execution outside the sandbox for temporary checkpoint snapshots/files.
The checkpoint's SHA-256 remained unchanged after evaluation. Hosted CI was not run.

Task 6 and Task 7 are complete. Next: Phase 4, Task 8 — plot training curves and
compare against baseline controls; assess independent improvement before declaring
Phase 4 complete. A matched-seed random-policy control has not yet been measured.

## Phase 4, Task 8 — Curves, controls, and acceptance

Completed on 2026-10-15:

- Added reusable paired baseline assessment, validated log parsing, and analyze CLI.
  Controls replace only the learner under matching engine/opponent/index schedules.
  Estimate matched mean-score effects with equal-seat stratified normal intervals.
- Added optional Matplotlib analysis extra and CI installation. Installed Matplotlib
  3.11.2 and its plotting dependencies; PyTorch remains 2.14.1+cu130. No training
  configuration, log, checkpoint, or engine behavior was changed.
- Generated and visually verified PNG/SVG training and control-comparison figures.
  Raw curves and labeled 25-update trailing means retain gaps; real log has all
  1,000 rows. Preserved aggregate chart copies in docs/phase4 for branch publication.
- Compared 12,000 games with no errors: policy Task 7 results reproduced exactly.
  Matched random controls score +.318 X / -.303 O versus random, versus policy
  +.606 / +.041. Equal-seat improvement +.3160 with approximate 95% interval
  [.2644,.3676] supports minimal Phase 4 acceptance for this checkpoint/protocol.
- Heuristic control remains much stronger; policy is worse than random controls
  against the heuristic. Last-25 self-play X wins reach 99.81% as entropy falls
  to .1455. Root cause is not diagnosed and no model was silently repaired.
- Added 14 tests covering independent statistics, pairing/protocol failures,
  log validation, smoothing/gaps, reproducible controls, figure generation, and
  CLI artifact protection. Updated all roadmap/architecture/boundary/usage records
  and added PHASE4_RESULTS.md with protocol, figures, acceptance, and limitations.

Validation: all 616 tests pass including CUDA checks; Ruff lint/format, strict
mypy (78 source files plus both scripts), pip check, and diagnostics pass.
Chart generation/comparison and tests used approved execution outside the sandbox
for temporary snapshots/caches. Checkpoint SHA-256 remains unchanged. Hosted CI
was not run locally; the branch push will exercise configured hosted checks.

Phase 4 complete under its minimal criterion; single-run/heuristic weaknesses
remain documented. No additional training or Phase 5 implementation. Stop for
today; next roadmap task is Phase 5, Task 1. User requested committing and pushing
these changes to a new codex branch, without changing master.
