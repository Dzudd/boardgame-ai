# Current progress

Updated: 2026-10-18. This document summarizes current status, verified evidence,
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
| Phase 6 | Complete at frozen v1 comparison scope: all 36 operator jobs/evaluations reviewed; mixed/negative transfer findings and one uncertain wall-cost measurement retained |
| Phase 7 | Partial: public simulator, bounded search evaluation/diagnosis and separate CPU search-target learner implemented; learning-quality comparison and broader research remain |
| Phase 8 | Partial: baseline and Connect Four checkpoint human CLI play, plus Tic-Tac-Toe public replays; broader trained play and product tasks remain |

**Phase 6 outcome:** the operator completed all 36 jobs and evaluations in the
frozen study. The [results](docs/phase6/MULTIGAME_RESULTS_V1.md) and
[machine-readable evidence](docs/phase6/multigame-v1-operator-results.json) now
provide per-game strength, realized costs, unseen adaptation curves, negative
transfer and retention measurements. The quantitative comparison criterion is
fulfilled at this declared scope; consistent positive transfer and universal
noninferiority were not demonstrated. One recovered job has uncertain wall timing,
so only two wall-budget seed pairs have verified realized matching. No missing
operator results remain for this study. Codex reviewed evidence without executing
training/evaluation; later phases were not expanded or started.

The user may also resume the separately versioned
[CUDA 100,000-game training/play schedule](docs/phase6/CONNECT_FOUR_CUDA_100K_V1.md)
from its saved boundary, under the original six-hour training
and ten-minute evaluation caps. A single in-place progress bar is enabled by default. The
operator has not completed the campaign or final evaluation; no GPU speedup or
strength claim follows from the partial run. Later learning-quality studies need
fresh evaluation roots, multiple training seeds and throughput-based budgets.

The [functional run](docs/phase6/CONNECT_FOUR_FUNCTIONAL_RESULTS.md) completed through
update-two pause/resume to four: 32 self-play games, 664 decisions, 46 optimizer
steps, followed by 192 error-free evaluation/control games. Balanced random-opponent
score was .50000 initially and .65625 finally, versus .50000 for the random control.
This single tiny run demonstrates operation, not robust strength or shared learning.
An earlier Git startup failure produced zero learning work and remains preserved.
The separate search-target/PPO comparison still needs throughput-based budgets;
hidden-information search and MuZero remain unfinished. The shared/transfer study
is complete with the limitations in its results, not a general strength claim.

## Verified evidence

- **Phase 6 operator results reviewed (2026-10-18):** 80,698 accepted self-play
  games / 1,140,839 decisions; 64,512 evaluation/control games / 1,197,185 decisions;
  zero evaluation errors/partial reports. All 36 complete metrics logs, case
  schedules, paired initialization/reset Adam and original-adapter retention were
  checked offline. All 778 saved run files retain their hashes. Training ledgers
  total 8,877.907 s (including one uncertain 900 s reservation); separate evaluation
  totals 491.097 s. Shared-minus-separate decision-ceiling Guildworks score is
  -4.82 percentage points [approximate seed interval -7.26, -2.38]. Consistent
  >=1.2x adaptation speedup fails: only one of six regime/seed pairs meets both
  ratios. Retention noninferiority is unresolved for wall Connect Four and
  decision-ceiling Tic-Tac-Toe. These negative/uncertain outcomes are preserved;
  completion means quantified comparison, not a successful transfer hypothesis.

- **Shared Phase 6 implementation (2026-10-17):** game-specific visible streams,
  supplied variable candidate scoring, a common GRU, per-game value adapters,
  complete latest-policy self-play and full-episode equal-game PPO are available.
  Fixed reward divisors keep Guildworks own points/36 separate from zero-sum
  returns. Separate models have independent backbones/optimizers with matched
  adapter capacity; original learners remain intact. Breakthrough 5x5 v1 is frozen
  as unseen, excluded from pretraining. Pretrained/fresh adaptation resets Adam
  and pairs untouched C adapters. Atomic portable run checkpoints, source archives,
  metrics-tail recovery, conservative crash reservations and per-game attempted
  versus accepted cost logs support the operator study. Reports retain raw seat
  outcomes, training-seed intervals, cost differences, censored target crossings,
  negative transfer and old-game retention. Quantitative hypotheses have now been
  reviewed in the results above; the implementation tests alone were not evidence.
  The [task-by-task audit](docs/phase6/MULTIGAME_STATUS_V1.md) separates implemented
  functionality from the experimental evidence that was then pending. Its
  pre-execution snapshot is superseded by the current 36/36 results review.

- **Current saved independent runs (read-only audit 2026-10-17):** the 100k CUDA
  run is paused at update 3,743 / 29,944 games / 598,598 decisions; the overnight
  1m run is paused at update 7,287 / 58,296 games / 1,211,631 decisions. Their
  statuses report Ctrl+C with no cap overrun. These supersede the historical
  update-100/update-373 observations below. Artifacts remain preserved; partial
  counts are not new strength or completion evidence.

- **Live training feedback:** one bounded-width in-place progress bar shows saved
  update percentage/count, phase and remaining cumulative cap, throttled to four
  refreshes/second. Redirected output logs at most every 30 seconds plus the final
  state. Detailed diagnostics remain in metrics; final stdout stays JSON and
  `--quiet` suppresses progress. An exact reviewed source-digest mapping permits the existing
  original and verbose-console CUDA runs to resume without rewriting saved provenance. Unknown
  source/runtime/settings mismatches still reject; subsequent training status
  records the execution source. Read-only review found the operator paused at
  update 100 / 800 games / 16,143 decisions / 1,102 optimizer steps, 21.234 seconds
  charged and no discarded games or reported cap overrun. No final evaluation or
  strength evidence is available yet. Codex did not execute the campaign.

- **CUDA run-layer milestone (operator campaign paused):** device-aware preflight,
  initialization, restore, manifests and console summaries; synchronized minibatch
  deadline checks and timing/publication/settlement boundaries. Unavailable CUDA
  rejects before outputs or repair. Portable snapshots, absolute schedules, atomic
  updates, metrics recovery, cumulative caps and isolated CPU evaluation remain.
  The separate `connect-four-cuda-100k-v1` settings schedule exactly 100,000 games
  (12,500 updates of eight), fresh roots 631001/631002/631003 and 1,536 separately
  capped evaluation/control games. Early cap stops report incomplete counts;
  completion, GPU speedup and reliable strength are not guaranteed. Exact Windows
  CMD setup/preflight/train/resume/evaluate/play commands are in the linked runbook.

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
  Tic-Tac-Toe; unseen Breakthrough is now frozen in the separate shared protocol.
  Its operator transfer results are now reviewed in the Phase 6 results document.

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
  evaluation still supports Tic-Tac-Toe PPO only. MuZero remains absent; the separate
  shared multi-game implementation now has the bounded random-opponent comparison;
  robust stronger-opponent strength and consistent transfer benefits remain unproven.
- Connect Four has an independent bounded CPU/CUDA run/resume layer with isolated
  CPU evaluation; its declared functional protocol is complete; robust strength is not established. Portable
  checkpoint human play is available; search and replay remain future work. Run
  recovery assumes one writer; initialization crashes can leave incomplete directories that reject
  resume, and hard-kill budget reservations remain spent. Fixed column encoding
  does not establish general candidate-action scoring or shared-model learning.
- Evaluation baselines and tactical labels never enter latest-policy self-play
  training. Results identify consumed final seeds; subsequent tuning requires
  fresh final-test seeds. Limited measured gains do not establish robust play.

## Validation

**2026-10-18 results review:** all 36 saved runs/evaluations validate; 64,512 raw
cases independently checked for schedules, seat/index uniqueness, outcome/return
consistency where available, lengths and summary arithmetic. Checkpoint comparisons
verify paired C adapters, reset Adam, unchanged old adapters and changed recurrent
weights. All 65,365 update rows contain finite checked PPO diagnostics. All 778
existing run files are hash-identical before/after review; source code, frozen
settings and original reports were unchanged. Only new derived results and shared
status documentation were written. This is offline validation, not more games or
a repeat campaign. Functional test counts below are the earlier implementation
and recovery checks, not tests newly executed during this documentation review.

**Windows finalization recovery (2026-10-17):** operator jobs 1–6 completed;
job 7 (`wall-661102-shared`) failed replacing its budget file with WinError 5
after saving update 2,066 / 6,198 games at 899.86 of 900 seconds. The conservative
reservation remained spent. Bounded Windows publication retries and an explicit
no-training, no-ledger-reset finalization command are implemented. Exact source
compatibility preserves existing manifests/archives/settings. Recovery backups and
receipts flag uncertain timing, which disqualifies an exact matched wall-cost claim.
Read-only eligibility preflight passes; **49 focused tests**, including available
CUDA continuation and fault cases, pass. Ruff lint/format and strict mypy (148
files) pass. All **303 existing run artifact hashes** are unchanged by development.
At that development checkpoint the operator had not yet applied recovery; Codex did not train, evaluate or edit
the saved run. See the runbook's recovery section for the single-line CMD command.

**2026-10-17 final validation: 1,216 tests passed, 1 skipped (83.69 seconds).**
This includes 25 new Phase 6 disposable tests and available CUDA update/portable
continuation coverage. The skip is the existing Windows symlink-privilege case.
Ruff lint/format, strict mypy (146 source/test files), CUDA device preflight,
dependency and whitespace checks pass. Pytest ran outside the sandbox from its
first invocation. New-study read-only preflight accepts all 36 frozen jobs on
the RTX 4070 Ti; read-only reporting correctly shows 0/36 evaluated and pending
acceptance. No experiment directory was created. Both saved CUDA campaigns pass
read-only resume checks through exact additive-source compatibility. All 200
existing run files retain their original hashes. Existing uncommitted work remains
uncommitted, and no branch, PR or campaign was created. Quantitative Phase 6
acceptance still awaits operator results.

The paragraphs below preserve earlier validation evidence and historical run
boundaries; they do not replace the current validation or artifact audit above.

The subsequent progress-bar refinement passed **93 focused CPU/CUDA/run/console
tests** (32.48 seconds), including throttling, bounded terminal width, carriage-return
redraws, one final newline, sparse redirected output and learning/resume isolation.
Ruff lint/format, strict mypy and whitespace checks pass. Exact source mappings
accept the original and preceding verbose-console implementations and reject
unknown sources. Read-only preflight validates the operator's Ctrl+C boundary at
373 updates / 2,984 games / 59,905 decisions, with 21,521.235 training seconds
remaining. All 190 run files retain their pre-refinement hashes. No campaign was
launched by Codex; the full-suite result below predates this renderer refinement.

Earlier full local validation: **1,190 tests passed, 1 skipped** (99.20 seconds),
including 11 new console-progress/source-compatibility cases and the existing
CPU/CUDA regressions. A final refinement checks the deadline after console output
and before checkpoint publication; its extra slow-output regression and the focused
progress/run/console/device suite passed **93 tests** (32.77 seconds). The skip
requires Windows symlink privilege. Ruff lint/format, strict mypy, dependency,
runner CLI help and whitespace checks pass. Pytest ran outside the sandbox from
the first invocation per the local workflow; no repeated folder-creation failure.

Read-only resume preflight accepts the operator's update-100 CUDA boundary on the
RTX 4070 Ti: 800 games saved, 99,200 scheduled games remaining, 21,578.766 training
seconds and all 600 evaluation seconds remaining. The exact console-only source
compatibility transition preserves the original saved manifest/protocol/checkpoint
identities and rejects unknown source/runtime changes. All **190** current run-file
SHA-256 hashes (182 prior artifacts plus eight files from the operator's pause)
remain unchanged across this fix. Original CPU protocol/settings/results and
versioned CUDA settings are unchanged. Codex executed only disposable tests and
read-only inspection/preflight, never campaign work. Changes remain uncommitted.

The original 168 `runs/` files and four failed-initialization files remain
preserved. The user created ten successful-run files; all 182 run files retain
their hashes across results review/documentation. During that functional-results review, the agent did not launch
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
