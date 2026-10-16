# Connect Four independent baseline: functional protocol v1

Declared 2026-10-16, before learner implementation or measurement. Status:
**model/collector/PPO/trainer restore, bounded run/evaluation and checkpoint CLI
implemented; functional run completed by the user and reviewed 2026-10-16**. This defines one
independent baseline, not a strength experiment or multi-game protocol.
The [numeric settings](phase6/connect-four-functional-v1.json) are the canonical
machine-readable declaration; changing them requires a new protocol version.
The separate `boardgame_ai.training.connect_four_run` module consumes this file.
The [recorded results](phase6/CONNECT_FOUR_FUNCTIONAL_RESULTS.md) describe the
completed CPU functional run and its limitations. See
[run usage](DEVELOPMENT.md#connect-four-bounded-runs).
The [console runbook](phase6/CONNECT_FOUR_CONSOLE_RUNBOOK.md) gives operator commands
and a read-only preflight. These readiness checks do not execute the protocol.

## Game roles and scope

Tic-Tac-Toe is the original game. Guildworks and Connect Four are the two
additional engines using the shared interface, satisfying that engine task.
Their engine conformance does not establish shared learning. Tic-Tac-Toe and
Guildworks learners remain independent references; this milestone specifies an
independent Connect Four PPO baseline. Their historical results are not matched
compute comparisons with this run.

For later shared-model work, the current three games are development games.
Transfer game C must be an additional, as-yet-unselected game, chosen and frozen
before shared training and adaptation evaluation. Connect Four is not an unseen
transfer game after this development work. The shared architecture, candidate
scoring, sampler, transfer protocol and final strength seeds remain unspecified.
No shared-learning or transfer acceptance criterion is completed here.

## Policy, actions and value

Use the implemented observer-relative float32 `(6, 6, 7)` encoding: own discs,
opponent discs, empty cells, own turn, opponent turn, terminal. Flatten 252
features into Linear(252,64), tanh, Linear(64,64), tanh; independent linear heads
produce seven raw logits and one unbounded own-seat return estimate. Total:
20,872 parameters. No dropout, recurrence, search, privileged critic or extra
history features. Initial parameters use standard PyTorch CPU initialization
inside an isolated seeded context. No existing checkpoint format is reused.

Mask logits using supplied legal actions; sample the categorical distribution
at temperature 1 with private per-seat CPU generators, then decode to the supplied
`drop(column)` object. Indices are 0 through 6; human labels remain 1 through 7.
Never infer availability from board emptiness alone. Inactive/terminal tensors
are representable but have empty masks; do not sample them. True termination
always bootstraps zero, irrespective of network output.

## Self-play and optimization

Every update collects eight complete games with a single frozen latest policy
in both seats, starting seat 0, at most 42 decisions/game. A live game at the
bound is a failure, not a training truncation. Use no historical opponents,
evaluation games, teacher labels or replay. Optimize all collected decisions
exactly twice, in shuffled minibatches of 32 with the final smaller batch kept.

The game supplies zero intermediate rewards and terminal winner/loser +1/-1,
or 0/0 for a draw. Accumulate each seat's own incremental rewards from its last
decision through the next own decision or termination. In particular, the seat
that does not act on the terminal move still receives its terminal increment.
GAE is separate per seat and episode, with gamma=1 and lambda=.95 on the own-seat
decision clock. Values estimate own return, never the opponent's prediction.
Combine only after GAE; normalize advantages across the entire update with the
existing PPO normalization convention. No reward shaping or reward normalization.

Use Adam and the existing clipped PPO objective, half-MSE value loss, masked
entropy, gradient clipping and nonfinite checks. All coefficients are explicit
in the numeric settings. Existing feed-forward PPO batch validation assumes
Tic-Tac-Toe shapes `(6,3,3)` and nine actions. Implementation must add explicit
Connect Four support or a separate adapter, retaining old defaults, checkpoint
identity, numeric behavior and regression tests. Passing larger tensors to the
current validator is unsupported. The implemented Connect Four adapter shares
the loss/optimizer loop through an explicit seven-action validation path;
the existing Tic-Tac-Toe adapter retains its original shapes and defaults.

## Seeds and isolation

Derive an unsigned integer from the first eight SHA-256 digest bytes, big endian,
of UTF-8 `namespace:root:index:stream`. Training uses its declared namespace and
root 630601. Initialization/shuffle use index 0 and streams `initialization` and
`shuffle`. Game index is zero-based absolute update index times eight plus local
game index; streams are `engine`, `seat:0`, `seat:1`. Private CPU sampling
generators are seeded directly with the derived unsigned 64-bit integers and
reset between games.
Private shuffle state is persisted, not recreated on resume. Initialization,
acting, optimization scheduling and evaluation preserve global Python, NumPy,
CPU Torch and already-initialized CUDA RNG state.

Evaluation uses its separate namespace and root 630602. For seat s and case game
i, use index `s*32+i` with streams `engine`, `policy`, `opponent`. Initial/final
evaluation reuses this schedule for matched comparisons. The random control uses
root 630603 only for the designated seat's `policy` stream; its `engine` and
`opponent` streams retain root 630602 and the same index rule, matching the
learned-policy evaluation. This preserves the paired comparison's opponent seeds.
These are functional evaluation seeds, not held-out strength-test seeds. Do not
reuse them for an eventual final strength claim. Future strength studies must
predeclare at least three training seeds and fresh final roots before execution.

## Fixed functional budget and stop rules

The CPU, one-thread functional run has four updates: 32 self-play games and
224 to 1,344 decisions. Each update has at most 336 samples; two epochs and
minibatches of 32 imply at most 88 optimizer steps overall. Record actual counts.
Compare initial and fixed update-4 checkpoints against RandomAgent for 32 games
per learned seat per checkpoint: 128 evaluation games total. Also run a matched
random-versus-random control for 32 games per designated seat: 64 games. The
entire scheduled run is 224 games with at most 9,408 engine decisions. No
checkpoint selection, intermediate development evaluation or greedy sweep.

Training has a 120-second wall cap including collection, optimization and saves;
evaluation has a separate 120-second cap including checkpoint loads and control
games. Check time between decisions/minibatches and before new work. Finish at
the cap without publishing an unfinished update; retain the last saved boundary
and report incomplete counts. A single indivisible operation may overrun the
check; report it. Stop on nonfinite values, illegal choices, truncation, artifact
identity mismatch or information-boundary failure. Do not extend caps or budgets
in response to weak outcomes. This document schedules no execution now.

## Checkpoints, outputs and resume

Implementation status: trainer restore and the run-level requirements below are
available through a separate CPU/one-thread runner. Disposable tests exercise
two-update stop/resume to four with one game/update under a separately named test
protocol; they do not provide the declared 32-game run's measurements. The user
subsequently completed that run through update-two pause/resume, then the fixed
evaluation; the linked result report preserves its artifacts and failed startup.

New output must be absent/empty. Preserve a copy and SHA-256 of the numeric
settings and this protocol, plus code revision/dirty-tree identity, runtime,
device and threads. The numeric settings digest is the resume identity; prose
edits outside a run cannot silently change its saved provenance. Save immutable
initial/final snapshots, `training.json`, `latest.pt`, `updates.jsonl` and new
evaluation reports. Existing artifacts must never be overwritten by evaluation.

Use a separate versioned checkpoint identifying game, model and encoding. It
contains CPU-portable weights, Adam, mode, configuration, counters, cumulative
games/decisions/optimizer work, cumulative elapsed budget and private shuffle state.
Exclude engines,
observations, rollout/history, global RNG and per-game live agent state. Validate
all identities, shapes/dtypes/finiteness, optimizer structure and counters with
restricted weights-only loading before returning a fresh trainer or writing.
Reject Tic-Tac-Toe, Guildworks and search checkpoints without modifying outputs.

Publish `latest.pt` atomically after each complete update, then append metrics.
Resume only at saved boundaries with unchanged settings; repeat unsaved work
using absolute game indices. A no-op resume must leave bytes unchanged. Support
the same validated tail-recovery/backup contract as the Tic-Tac-Toe metrics log;
never invent missing rows. Exact uninterrupted versus resumed CPU weights, Adam,
shuffle state and future metrics must match except recorded timing. Test a
two-update stop/resume to four in disposable output; it is a continuation of the
32-game budget, not an additional experiment. A demonstration must also respect
remaining cumulative wall time; budgets cannot reset on resume. CUDA portability
is tested separately on disposable tiny fixtures, without cross-device identity
or speed claims. Portable-checkpoint human CLI play is implemented; see
[usage](../README.md#connect-four). It is separate from this evaluation schedule,
and its sampled/greedy options do not change the protocol settings.

The run checkpoint is a versioned envelope around the existing validated trainer
payload. `budget.json` is a separate cumulative training/evaluation ledger and
`training-status.json` reports saved counts, known discarded complete-game work,
stop reason and overruns. Partial collector decisions are explicitly unknown.
The ledger reserves all remaining time before work, then settles actual elapsed
time on normal completion or exceptions. A hard process kill leaves the reservation
spent conservatively; it cannot reset a cap. Tiny final ledger-write overhead is
not recursively charged. Checkpoint boundary elapsed time is a lower bound on the
ledger, which includes failed/unsaved attempts and initialization/saves. A crash
during initial setup can leave an incomplete directory that rejects resume;
preserve it and investigate rather than treating it as a completed run.

Resume validates saved settings/prose/initial snapshot hashes, model/learner/run
identities, source digest, Python/Torch versions, counters, budget and metrics
before repairing a tail. Current external prose is not reread. Source/runtime
changes reject; Git revision/dirty status and diff/source digests remain recorded
provenance. Complete metrics logs must agree with checkpoint sample counts; gaps
remain gaps. A missing final snapshot can be published from the final boundary on
resume without further collection. Healthy no-op resumes leave bytes unchanged.

Evaluation is explicitly requested separately and requires the fixed final
boundary. It uses immutable initial/final trainer snapshots and private per-game
streams, including the paired random control. An exclusive `evaluation.json`
records raw completed cases, checkpoint hashes, per-seat summaries, paired effects,
failures and partial counts. A partial report is preserved and cannot be overwritten
or rerun in place. Evaluation budget reservations also survive hard termination;
there is no mid-evaluation continuation. These are single-writer APIs, not a
concurrent run manager. CUDA trainer portability remains separately tested; this
runner deliberately implements the protocol's CPU execution only.

## Reporting and acceptance

Report initial/final/control W/D/L, score (win=1, draw=.5, loss=0), errors,
truncations and lengths separately by learned/designated seat. Balanced score
weights seats equally. Report final-minus-initial and final-minus-random effects
descriptively with paired game-level uncertainty; do not imply training-seed
uncertainty from this single seed. Use paired score differences d at matching
seat/index: estimate mean(d) with approximate interval mean(d) +/- 1.96 times
sample-SD(d)/sqrt(32); balanced effect is the half-sum of seat means with standard
error .5*sqrt(SE_0^2+SE_1^2). Label these small-sample normal intervals approximate;
include raw outcomes and do not use the interval as a pass/fail strength gate.
Policy and control use separate generators with matching engine/opponent seed
schedules; resets make opponent streams comparable for matching cases. Actual
opponent moves can diverge when the policies produce different legal choices.

Record self-play seat outcomes, entropy/loss/KL/clipping, unique public acting
positions and action frequencies, decisions, parameter count, optimizer steps,
sample presentations and replay reuse (zero). Separate collection, optimization,
checkpoint and evaluation wall time. Report hardware and threads. Raw self-play
outcomes and low losses do not establish external strength.

Functional acceptance requires complete legal games, finite nonzero parameter
updates, correct both-seat reward/GAE, isolated evaluation, reproducible seeds,
validated corruption rejection and exact CPU resume. Add hand-calculated reward
and PPO tests plus encoder/agent/CLI regressions; run lint, formatting, strict
typing, full regression and available CUDA tests. Strength improvement is not
required. Publish failures and partial counts honestly. No phase-wide task beyond
the existing two additional engines is claimed complete by this specification.

## Later matched-compute and transfer studies

Separate models versus a shared model require a new predeclared protocol after
bounded throughput measurements. Match total training wall time on the same
hardware/threads, counting collection, inference, optimization and checkpointing;
report per-game sample allocations, optimizer presentations, parameter counts
and inference latency separately. Also report an equal-environment-decision
comparison, which is not inherently compute-matched. Evaluation has a separately
fixed equal budget. The 32-game functional run is not either comparison.

Transfer must compare shared-pretrained versus fresh initialization on the same
held-out C seeds, architecture/adapters and adaptation budgets, showing the whole
strength-versus-cost curve. Evaluate A/B before and after C adaptation for
forgetting, and compare shared versus separate per-game scores for negative
transfer. Freeze C, thresholds, time caps, training seeds and fresh final roots
before shared fitting. Guildworks raw points and zero-sum game returns must keep
separate reports and declared fixed value/loss scales; do not pool raw rewards or
derive normalization from evaluation data. These study details remain future work.
