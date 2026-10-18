# Phase 7 public learner allocation v2: three-hour maximum

Declared 2026-10-18 before execution. **Prepared, not executed. Phase 7 remains
incomplete; no Phase 8 work is included.** This is a new equal-game follow-up to
the completed [v1 comparison](PUBLIC_STUDY_V1_RESULTS.md), not an extension,
rerun or relabelling of its results. The hypothesis is that the unchanged recipes
at a larger self-play allocation can establish search-target improvement over
independent PPO. Training alone cannot establish that hypothesis.

V1 retains its failed primary criterion and heuristic-X guard: .32764 versus
.34440, paired effect -1.68 points [-3.81, +0.46], 7.31x measured search training
wall time, and improved in-sample replay fit without established external benefit.
Its 6,144 evaluation games had zero errors/truncations. Its final roots
20261208/20261209/20261210 are consumed. Phase 6 remains complete only at its
declared comparison scope, with negative findings and uncertain recovered wall
timing retained. Neither study is edited by this allocation.

## Frozen allocation and stopping rule

[Machine settings](public-study-three-hour-v2.json) are byte-pinned by the
[operator adapter](../../scripts/phase7_three_hour_v2.py).

| Setting | Declaration |
|---|---|
| Output | `runs/phase7-public-study-three-hour-v2/`, initially absent |
| Training roots | 772201 / 772202 / 772203 |
| Learners | Independent PPO baseline and existing public search-target learner |
| Order | PPO then search for each root, sequentially, one writer |
| Primary budget axis | Equal complete self-play games, not matched compute |
| Per job work limit | 512 updates x 4 games = 2,048 games |
| Total accepted game ceiling | 12,288 across six jobs |
| Per job cumulative supervised time cap | 1,740 seconds (29 minutes) |
| Six training reservations | 10,440 seconds (2 hours 54 minutes) total |
| Overall training ceiling | 10,800 seconds (3 hours), including 360 seconds headroom |
| Selection | Final scheduled checkpoint only; no best-score selection |
| Final evaluation roots | 20261301 / 20261302 / 20261303 (integer seeds, not dates) |
| Separately invoked final evaluation | Raw sampled policy, 256 games/opponent/seat/arm |
| Evaluation allowance | 120 seconds/seed pair, 360 seconds total, outside training |

The launcher stops at **both work and time limits**: each successful job ends at
512 saved updates, possibly much earlier than 29 minutes. A timeout, Ctrl+C,
failure or exhausted reservation stops the launcher before later jobs. A cap
stop leaves an incomplete allocation, not a selected final model. Do not extend
caps, drop seeds or select an earlier stronger checkpoint after seeing outcomes.
The larger replay/checkpoint costs have not been benchmarked at this scale;
completion within the caps is not promised and no duration estimate is claimed.

Training time includes child startup/load/initialization, collection, optimization,
native checkpoint/metric publication and supervision. Parent read-only checks,
source archival, post-child validation and final reporting are outside the timer;
the terminal command can therefore take longer. All attempts share each job's
original cumulative allowance, including across resumes. The adapter sums charged
time and outstanding allowances before each new job and rejects allocations
exceeding 10,800 seconds. The 360-second headroom is unallocated: it cannot buy
more games or a new attempt allowance. The existing supervisor kills and reaps
expired children and reports measured OS termination overshoot. This is a process
deadline, not a hard real-time guarantee if Windows stalls during termination.

All learner parameters are unchanged from v1: CPU/one thread, latest shared
policy in both seats, PPO four epochs/minibatch 64, search 32 simulations,
four Adam steps/update, batch 256, FIFO capacity 10,000 games, K=0. Models start
fresh, with native initialization/seed streams. Root pairing does not force
equal initial weights across learners. No old checkpoint is loaded for training.

Per-job accepted ceilings are 18,432 environment decisions; PPO at most 2,048
Adam steps and 73,728 presentations; search at most 589,824 simulations,
608,256 evaluator calls, 2,048 Adam steps and 524,288 presentations. Actual
decisions, reuse, replay ages and wall costs must be reported from all six jobs.
Equal games and equal caps do not imply equal realized computation. There is no
matched-wall, isolated search-causality or universal-strength claim.

## Recovery, provenance and information boundaries

The adapter lives outside `src/boardgame_ai` because the completed-study reporter
requires the exact package-wide digest recorded by its manifests. No package
Python file or compatibility map changes. The adapter pins that existing digest
and verifies the original frozen settings/calibration evidence. Existing v1
reporting and saved CUDA/Phase 6 compatibility checks remain usable.

Before any training mutation, every intended job must pass read-only preflight.
Fresh execution rejects any existing study root. The adapter publishes an immutable
allocation identity with its own source/launcher hashes and byte copies. Each
native job retains settings/source/runtime provenance, a package source archive,
atomic authoritative checkpoint, immutable initial/final snapshots, native metrics,
immutable attempt receipts and cumulative budget ledger. The existing throttled
single-line progress display remains in use.

Explicit resume verifies allocation identity and all native identities before
repair or work. Complete jobs are byte-preserving no-ops; absent later jobs may
start under their original allocations. PPO retains torn-tail backup/recovery;
search can recover its current metric report from the saved checkpoint. No metrics
are invented. An initialization crash without a valid checkpoint rejects resume.
An outstanding hard-parent-kill reservation remains spent; it never resets.

Interrupted unsaved games, decisions, simulations, calls and optimizer work have
**unknown sample cost**, even when elapsed attempt time was measured. Retain them
as unknown, never zero. Unknown discarded work or incomplete metrics blocks the
comparable-budget improvement gate. Final checkpoint selection and evaluation
require every scheduled training job complete. No fallback to a last pre-timeout
checkpoint is permitted under this version.

Mandatory boundaries remain latest-policy self-play only; evaluation isolated from
all learning/replay/RNG schedules; no teachers, solver targets, baseline training
opponents, historical acting-policy pools, tactical rewards, privileged critics or
inaccessible hidden-state clones. Self-generated public target replay stays within
the existing learner's declared contract. No new research features are introduced.

## Predeclared success criteria

Final measurement uses only each job's final scheduled immutable checkpoint and
fresh paired roots above, random-v1/heuristic-v1, both learner seats, X starts.
There are 6,144 games in total. Evaluation is a separate explicit command;
`train` and `resume` never launch evaluation. No strength checks occur in training.

Use the unchanged strict study aggregator: equal-weight four-case search-minus-PPO
score per training seed (win=1/draw=.5/loss=0), descriptive 95% seed interval
`mean +/- 4.303*SD/sqrt(3)`. Require primary lower endpoint > 0 and each heuristic
seat's lower endpoint >= -.05, with complete provenance/cost/schedule evidence
and fully accounted attempts. Report all seed effects, per-seat W/D/L, latency,
unequal work and negative/inconclusive outcomes; distinguish game uncertainty from
training-seed uncertainty. These are bounded recipe comparisons, not proof that
search alone caused a difference. A positive result would not complete beliefs,
hidden/chance/same-player search, learned dynamics or all Phase 7 acceptance tasks.

## Single-line Windows CMD commands

Run from any CMD directory. Read-only new-allocation preflight:

```bat
"C:\Users\micha\Documents\ChatGPT\Board Game AI\run_phase7_three_hour_v2.cmd" check
```

Start training only:

```bat
"C:\Users\micha\Documents\ChatGPT\Board Game AI\run_phase7_three_hour_v2.cmd" train
```

Read-only resume preflight, then explicit resume (separate commands):

```bat
"C:\Users\micha\Documents\ChatGPT\Board Game AI\run_phase7_three_hour_v2.cmd" check-resume
```

```bat
"C:\Users\micha\Documents\ChatGPT\Board Game AI\run_phase7_three_hour_v2.cmd" resume
```

After all six jobs complete, `check-evaluation` is read-only; `evaluate` explicitly
consumes the separately reserved final schedules; `report` is read-only stdout.
Use those action names with the same launcher. Preserve any failed/partial output;
do not delete evidence, restart consumed evaluation or reset budgets.

## Validation

Full suite: **1,255 passed, 1 skipped in 192.92 seconds**, including available
CUDA checks; the skip is Windows symlink privilege. Seven new tests cover frozen
allocation/total budget, read-only preflight, all-job checks before writes,
fail-fast behavior, spent crash reservations, tiny six-job execution/no-op resume
and actual CMD invocation from another directory with exit-code propagation.
Ruff lint/format, strict mypy, dependency/CUDA device checks and whitespace checks
pass. Pytest ran outside the sandbox from the first invocation. The package
digest remains `95f5629e5384de75487861c058b835ce2fbfe1bda60eb794ed881588b87c04e3`.
Read-only v1 reporting and both saved CUDA/recovered Phase 6 resume checks pass.
All 7,252 saved artifacts match the retained audit; the new output is absent.
Development tests use disposable tiny allocations; they are not execution of this
campaign. [Progress](../../PROGRESS.md) records the validation and phase limits.
