# Current progress

Updated: 2026-10-16. This document summarizes current status, verified evidence,
limitations and the next milestone. The [roadmap](BOARDGAME_AI_PROJECT_PLAN.md)
owns phase tasks and acceptance criteria; linked result documents own experiment
protocols, measurements and detailed limitations.

## Status

| Area | State |
|---|---|
| Phases 1–3 | Complete: contracts, engine, simulation, baselines, CLI and replays |
| Phase 4 | Complete under its original improvement-over-random criterion |
| Phase 4.5 | Tasks 1–6 complete at the declared bounded scope |
| Phase 5 | Complete at declared Guildworks scope: engine, CLI, recurrent memory/PPO and bounded functional run |
| Phase 6 | Two additional engines and independent Connect Four PPO/run/resume/evaluation available; bounded CPU functional run complete; learning-quality, shared training and transfer pending |
| Phase 7 | Partial: public simulator, bounded search evaluation/diagnosis and separate CPU search-target learner implemented; learning-quality comparison and broader research remain |
| Phase 8 | Partial: baseline and Connect Four checkpoint human CLI play, plus Tic-Tac-Toe public replays; broader trained play and product tasks remain |

**Next milestone:** predeclare the next independent Connect Four learning-quality
study with fresh evaluation roots and multiple training seeds, selecting compute
budgets after bounded throughput measurement. GPU run-layer support remains to be
implemented before GPU experiments. The user runs campaigns in their own console.

The [functional run](docs/phase6/CONNECT_FOUR_FUNCTIONAL_RESULTS.md) completed through
update-two pause/resume to four: 32 self-play games, 664 decisions, 46 optimizer
steps, followed by 192 error-free evaluation/control games. Balanced random-opponent
score was .50000 initially and .65625 finally, versus .50000 for the random control.
This single tiny run demonstrates operation, not robust strength or shared learning.
An earlier Git startup failure produced zero learning work and remains preserved.
The separate search-target/PPO comparison still needs throughput-based budgets;
shared-game/transfer protocols, hidden-information search and MuZero remain unfinished.

## Verified evidence

- **Phase 6 independent prerequisites:** Connect Four observer-relative tensors,
  supplied-column masks/decoding, a 20,872-parameter policy/value model and a
  complete-game self-play collector are implemented. The 43 encoder tests and
  56 model/collector tests cover perspectives, legal masked sampling, own-seat
  terminal rewards and hand-calculated GAE, full columns, wins/draws, deadline/
  cutoff rejection, frozen snapshots, RNG/storage isolation and CUDA-source
  collection. A further 47 PPO/checkpoint tests cover hand-calculated clipped
  losses and gradients, optimizer changes, corrupt/incompatible file rejection,
  atomic-save failures, partial-update/deadline protection, exact CPU next-update
  and two-plus-two self-play continuation, and CUDA portable restore. These are
  disposable functional tests, not a strength experiment.
  The independent-baseline [functional protocol](docs/PHASE6_CONNECT_FOUR_PROTOCOL.md)
  is now specified with versioned numeric settings: 32 self-play games, both-seat
  initial/final random evaluation, fixed caps and CPU resume requirements. The user
  completed it in their console; [results and raw evidence](docs/phase6/CONNECT_FOUR_FUNCTIONAL_RESULTS.md)
  record both the failed initialization and the successful resumed run. The bounded
  CPU run layer now saves immutable provenance and
  initial/final snapshots, schedules absolute game seeds, resumes validated run
  checkpoints, recovers metrics tails and charges cumulative failed/unsaved work.
  Separate frozen evaluation reports both seats, paired random controls and
  approximate paired effects without supplying PPO samples. Thirty-four new
  disposable run-layer tests cover exact two-plus-two continuation (one game per
  update), byte-preserving no-op resume, identity/counter/corruption rejection,
  metrics backup/gaps, save failure, optimizer-timeout rollback, remaining caps,
  RNG/thread isolation and evaluation errors/truncations. Hard termination spends
  outstanding time reservations conservatively; partial evaluation reports are
  preserved without mid-evaluation continuation. Read-only new/resume/evaluation
  preflight and console exit codes now expose readiness and incomplete stops
  without starting work or repairing logs. The declared new-run preflight reported
  ready on CPU/one thread with full caps before the user executed the schedule.
  Guildworks and Connect Four count as the two additional engines beyond
  Tic-Tac-Toe; a future unseen transfer game remains to be selected before shared
  training.

- **Search-target learner:** separate CPU latest-policy search self-play, complete
  own-seat outcome/visit targets, uniform-game/position FIFO replay, finite
  cross-entropy/half-MSE optimization, dedicated checkpoints and exact tiny-run
  resume are implemented. [Implementation and tests](docs/reviews/2026-10-16-search-target-implementation.md).
  No strength experiment or throughput comparison was executed.

- **Reliability:** evaluation reports require exclusive creation and protect
  managed artifact destinations; Tic-Tac-Toe resume backs up and recovers an
  unfinished final metrics record while rejecting earlier corruption. Versioned
  experiment settings are separate from prose, with a read-only verified
  compatibility path for the recorded Phase 4.5 v1 protocol. Original artifacts
  and hashes remain intact. [Resolution and validation](docs/reviews/2026-10-16-reliability-resolution.md).
- **Guildworks:** complete scripted/random games, filtered human CLI, hidden-world
  equivalence, recurrent memory/PPO and exact CPU resume are implemented and tested.
  The functional run completed 16 updates / 256 self-play games / 8,192 decisions.
  Initial/final balanced random-opponent scores were .515625 and .56640625 across
  256 evaluation games each, with no errors or truncations. This single small run
  demonstrates operation, not reliable improvement or a benefit from memory.
  [Phase 5 results](PHASE5_RESULTS.md) preserve the protocol, artifacts and limits.
- **Tic-Tac-Toe PPO:** the original 64,000-game run met its improvement-over-random
  criterion but remained weak against the heuristic. Signal/update checks found
  no demonstrated learning-signal defect; the six-run entropy comparison did not
  establish improvement. Repeated profiling supports limited same-device
  optimization claims; batched collection was not adopted.
  See [Phase 4 results](PHASE4_RESULTS.md) and [Phase 4.5 results](PHASE45_RESULTS.md).
- **Public search:** frozen greedy raw/16/32 evaluation reduced random-X losses
  from 19 to 9/9 and random-O losses from 96 to 50/41 out of 256 each, with greater
  cost. Heuristic-X remained all draws and heuristic-O all losses. Public traces
  reproduce branch starvation associated with concentrated priors and clipped
  leaf guidance; blocking is not proven to save the game.
  [Phase 7 results](PHASE7_RESULTS.md) contain the census, diagnosis and cost data.
- **Connect Four:** the shared-interface engine, public history, gravity, all four
  win directions, draw/reward handling, rendering, random matches and human CLI
  are implemented, with 35 recorded rule/CLI tests. This adds a playable game;
  it does not establish multi-game learning. Portable-checkpoint human CLI play now
  adds a frozen CPU policy, supplied-action masking and private seeded sampled or
  greedy decisions. Tests cover both seats, cancellation, corrupt/foreign/run-file
  rejection before input, source/RNG isolation, CUDA snapshotting and a disposable
  checkpoint after one PPO update. It writes no training/evaluation artifacts.
  Usage is in [README](README.md#connect-four).

## Current limitations

- Guildworks mechanics and visibility tests do not establish balance or robust
  strength. Its human CLI opponent is random only; heuristic play and public
  replay export remain unavailable.
- Search supports public deterministic Tic-Tac-Toe only. Hidden/chance and
  same-player game simulation are unsupported; generic same-actor backup
  arithmetic tests do not establish game support. Search-target learning is
  CPU Tic-Tac-Toe/K=0 only; its strength is unmeasured, and generic checkpoint
  evaluation still supports Tic-Tac-Toe PPO only. MuZero and multi-game learning
  are absent.
- Connect Four has an independent bounded CPU run/resume/evaluation layer, but its
  declared functional protocol is complete; robust strength is not established. Portable
  checkpoint human play is available; search and replay remain future work. Run
  recovery assumes one writer; initialization crashes can leave incomplete directories that reject
  resume, and hard-kill budget reservations remain spent. Fixed column encoding
  does not establish general candidate-action scoring or shared-model learning.
- Evaluation baselines and tactical labels never enter latest-policy self-play
  training. Results identify consumed final seeds; subsequent tuning requires
  fresh final-test seeds. Limited measured gains do not establish robust play.

## Validation

Latest full local validation: **1,159 tests passed, 1 skipped** (82.78 seconds),
including 27 console/preflight tests, 28 policy/checkpoint-play tests, all 34
run-layer tests and available CPU/CUDA regressions. The skip requires Windows
symlink privilege. Ruff lint/format, strict mypy, dependency, CLI help and staged
whitespace checks pass. The suite used the approved unrestricted setup for the
known sandbox temporary-directory restriction. Git attributes preserve identical
canonical/packaged settings bytes across checkouts, matching the saved run digest.
Disposable tests are distinct from the operator-run functional campaign; artifact
review did not rerun games or optimization.

The original 168 `runs/` files and four failed-initialization files remain
preserved. The user created ten successful-run files; all 182 run files retain
their hashes across results review/documentation. The agent did not launch
training/evaluation. Offline checks independently
verified raw case seeds/outcomes, summaries, paired intervals, checkpoint identities,
finite/nonzero parameter changes and counters; no runtime code changed in this review.

See the [resolution record](docs/reviews/2026-10-16-reliability-resolution.md) for
historical recovery evidence and compatibility limits, and the
[learner implementation record](docs/reviews/2026-10-16-search-target-implementation.md)
for the separate still-unexecuted learning comparison. Hosted CI is not asserted.

Protocol definition checks also verified JSON/PPO configuration compatibility,
20,872 model parameters, 32 self-play games, at most 1,344 training
decisions/88 optimizer steps, 192 evaluation/control games and 9,408 total
decisions. Updated document links and whitespace checks pass. The functional
training/evaluation budget has now been executed by the user: 224 games and 4,804
decisions, with 3.921 s training and 2.203 s evaluation ledger time. Tests still use
disposable fixtures and did not modify recorded run artifacts.

The public simulator comparison covers 5,478 boards and 16,167 legal transitions
per starting seat; search checks cover budgets, perspective, terminal handling,
private RNG and trace equivalence/corruption detection. Evaluation replay
reproduced match/tactical results. Search latency measurements are single-run
observations, not an isolated benchmark. Hosted CI is not asserted.
