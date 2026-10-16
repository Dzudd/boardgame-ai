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
| Phase 6 | Two additional engines and Connect Four encoders/model/collector/PPO/trainer restore available; bounded run layer, shared training and transfer pending |
| Phase 7 | Partial: public simulator, bounded search evaluation/diagnosis and separate CPU search-target learner implemented; learning-quality comparison and broader research remain |
| Phase 8 | Partial: baseline human CLI play and Tic-Tac-Toe public replays available; trained-agent human play and other product tasks remain |

**Next milestone:** add the bounded Connect Four run layer under the
[functional protocol](docs/PHASE6_CONNECT_FOUR_PROTOCOL.md): saved protocol
provenance, absolute self-play schedule, cumulative caps, metrics recovery and
isolated evaluation. Model/collector/PPO and trainer checkpoint restore are now
implemented. Shared learning requires later independent learning-quality
evidence. The separate search-target/PPO comparison
still needs throughput-based budgets; shared-game/transfer protocols,
hidden-information search and MuZero remain unfinished.

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
  initial/final random evaluation, fixed caps and CPU resume requirements. It has
  not been executed; run-level provenance/scheduling/resume/metrics and evaluation
  integration remain. Guildworks and Connect Four count as the two additional
  engines beyond Tic-Tac-Toe; a future unseen transfer game remains to be selected
  before shared training.

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
  it does not establish multi-game learning. Usage is in [README](README.md#connect-four).

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
- Connect Four has programmatic PPO and trainer checkpoint restore; the bounded
  run layer, run-level resume, evaluation, search and replay remain future work. Its
  fixed column encoding does not establish general candidate-action scoring
  or shared-model learning.
- Evaluation baselines and tactical labels never enter latest-policy self-play
  training. Results identify consumed final seeds; subsequent tuning requires
  fresh final-test seeds. Limited measured gains do not establish robust play.

## Validation

Latest full local validation: **1,070 tests passed, 1 skipped** (72.86 seconds),
including 47 new PPO/checkpoint tests and existing CPU/CUDA regressions. The skip
requires Windows symlink privilege; hard-link and resolved-path tests pass.
Ruff lint/format, strict mypy, dependency, CLI smoke and whitespace checks passed.
The suite used the approved unrestricted setup because previous sandbox runs
encountered temporary-directory permission errors in existing tests.
All 168 existing run artifacts remain unchanged, and the three principal
checkpoint hashes retain their previously verified historical identities. See the [resolution record](docs/reviews/2026-10-16-reliability-resolution.md)
for recovery evidence and compatibility limits; the [learner implementation record](docs/reviews/2026-10-16-search-target-implementation.md)
records its functional scope and the still-unexecuted learning comparison.

Protocol definition checks also verified JSON/PPO configuration compatibility,
20,872 model parameters, 32 self-play games, at most 1,344 training
decisions/88 optimizer steps, 192 evaluation/control games and 9,408 total
decisions. Updated document links and whitespace checks pass. The functional
training/evaluation budget remains unexecuted; collection, optimizer and checkpoint
tests use disposable fixtures without modifying recorded runs or source artifacts.

The public simulator comparison covers 5,478 boards and 16,167 legal transitions
per starting seat; search checks cover budgets, perspective, terminal handling,
private RNG and trace equivalence/corruption detection. Evaluation replay
reproduced match/tactical results. Search latency measurements are single-run
observations, not an isolated benchmark. Hosted CI is not asserted.
