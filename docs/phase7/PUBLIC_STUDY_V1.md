# Phase 7 public learner study v1

Frozen 2026-10-18 after offline review of the operator's six calibration jobs,
before study training/evaluation. This is a small public Tic-Tac-Toe comparison,
not Phase 7 completion or a hidden-information/MuZero implementation.

**Completed:** all frozen training and final evaluations are now reviewed in
[public study v1 results](PUBLIC_STUDY_V1_RESULTS.md). The improvement hypothesis
and heuristic-X guard did not pass. Final roots are consumed; the original
declaration/commands below remain for provenance, not a rerun or cap extension.

## Calibration evidence

All six jobs completed two updates/eight games within their separate 60-second
reservations. Saved requests, source hashes, metrics, report costs and validated
checkpoints agree. Every repetition has identical checkpoint bytes within its
learner. [Machine-readable review](calibration-results-v1.json) preserves all six
measurements; its hash is pinned by the [study settings](public-study-v1.json).

| Per repetition | Independent PPO | Search-target |
|---|---:|---:|
| End-to-end seconds, all three | 3.109 / 3.141 / 3.218 | 3.891 / 3.891 / 3.984 |
| Games / environment decisions | 8 / 65 | 8 / 56 |
| Search simulations / collection evaluator calls | 0 / 65 | 1,792 / 1,347 |
| Adam steps / sample presentations | 8 / 260 | 8 / 2,048 |
| Presentations per accepted decision | 4.00 | 36.57 |

These are startup-inclusive short measurements, not steady-state training speed,
hardware-general throughput or strength. The slower arm did fewer environment
decisions and substantially more search/sample work. Neither wall ratio nor sample
reuse establishes a learning advantage. Replay had at most eight games, so
full-capacity validation/save overhead was not measured.

## Frozen experiment

The unchanged learners use latest-policy self-play in both seats. No teachers,
solver targets, baseline training opponents, historical opponent pools, reward
bonuses or privileged critics. Search receives only permitted public observations
and supplied choices, never authoritative hidden-state clones. Evaluation remains
separate from training, replay, initialization, optimizer and RNG schedules.

- Training roots: **771101, 771102, 771103**. Preserve native independent
  initialization/sampling schedules; pairing roots does not force equal initial
  weights across algorithms. Both models retain the same architecture capacity.
- Primary allocation: **64 updates × 4 complete games = 256 games per arm/seed**.
  Six jobs total **1,536 accepted games** if all complete. Both learners preserve
  calibration parameters, including PPO four epochs and search 32 simulations,
  four Adam steps/update, batch 256, replay capacity 10,000, K=0.
- Each job has **180 cumulative child-work seconds**, including initialization,
  resume, collection, optimization and checkpoint publication. Caps are fixed,
  not targets. Their allowance exceeds the 99.5–127.5-second naive scaling of all
  startup-inclusive calibration timings to 256 games; that extrapolation is only
  a conservative scheduling guide, not evidence about growing replay costs. No
  cap extension if this estimate fails. Report an incomplete comparison honestly.
- Per completed job: at most **2,304 accepted environment decisions**; search at
  most **73,728 simulations / 76,032 evaluator calls / 256 Adam steps / 65,536
  presentations**; PPO at most **2,304 collection calls / 256 Adam steps / 9,216
  presentations**. Interrupted unsaved work is unknown and costs additional wall
  time; it cannot be zero-imputed or treated as exact attempted sample matching.
- Frozen checkpoint selection: final scheduled checkpoint only, no development-
  best selection and no strength checks during training. Preserve initial/final
  snapshots, original learner checkpoints/logs, source archives and attempts.
- Final evaluation roots: **20261208/20261209/20261210**, respectively paired by
  training root. Raw sampled policy only; **256 games/opponent/seat/arm**,
  random-v1 and heuristic-v1, both learner seats, X starts. This is **2,048 games
  per seed pair / 6,144 total**, at most 18,432 environment decisions/calls per
  seed pair. Each paired evaluation has a separate **120-second** reservation.
  Root/index/engine/agent streams are paired across arms; changed actions may
  change later opponent RNG consumption. No optional search-assisted condition
  enters this primary study; existing evaluator support stays separately labelled.
- Maximum combined child-work reservations: **1,440 seconds (24 minutes)**:
  6×180 training + 3×120 evaluation. Parent preflight, source setup and final
  aggregation/publication are extra. Do not run jobs concurrently.

This study compares equal complete-game allocations, **not matched wall cost**.
Report actual wall time, decisions, simulations, evaluator calls, optimizer work,
presentation/reuse counts, per-update replay ages and costs from every seed.
Unequal compute is part of the result. A later compute-matched study needs a new
predeclared schedule and fresh roots; do not relabel these data after outcomes.

## Success criteria and analysis

For each training seed, compute the search-minus-PPO difference in the equally
weighted four raw-policy random/heuristic × X/O case scores (win=1/draw=.5/loss=0).
Report all three effects and the descriptive 95% seed interval
`mean ± 4.303*SD/sqrt(3)`. The primary lower endpoint must exceed zero; both
heuristic seat effects must separately have lower endpoints >= -.05. Report
negative/failed/inconclusive hypotheses and all per-seat W/D/L; no seed dropping,
best-seed selection, after-the-fact budget extensions or multiple-comparison-
adjusted/universal-strength claim. Keep within-case game uncertainty separate from
training-seed uncertainty. Baseline comparisons are measurements only.

The read-only aggregator validates complete training/cost histories, immutable
attempt receipts/cumulative budgets, settings/source/runtime identities, selected
checkpoint hashes, exact final schedules, unique raw cases, terminal outcome/W/D/L
consistency, and raw versus search condition. Partial or mismatched evidence cannot
enter acceptance. Exact realized cost claims remain unavailable for uncertain or
unsaved attempted work. A failed/interrupted worker with unknown discarded sample
costs cannot satisfy the comparable-budget improvement gate even if its strength
criterion alone passes. A normal pause at a saved target has complete accepted
costs and can resume; a failed partial update does not. Even a positive bounded public comparison does not imply
that beliefs, information-set search, chance/same-player search, learned dynamics
or all Phase 7 criteria have been completed.

## Publication, recovery and limits

Each job saves its settings identity, source/runtime digest and source archive.
The original learner publishes its authoritative `training/latest.pt` before
metrics. Resume loads that boundary and retains native recovery: PPO backs up
recoverable torn JSONL tails; search recovers its current report from the saved
checkpoint without repeating optimization. No initial/final checkpoint is
overwritten. A no-op complete resume preserves bytes, except it may publish a
missing final snapshot from the final authoritative boundary.

Before each attempt the supervisor reserves the entire remaining cumulative wall
allowance. Normal exit or Ctrl+C settles measured elapsed time and publishes an
immutable attempt receipt; the child's interrupted unsaved learning work remains
unknown. A hard parent termination leaves the full reservation spent and timing
uncertain, not a new allowance on resume. Partial initialization without a valid
checkpoint rejects continuation. Existing records and checkpoints remain for
inspection; no automatic rollback, deletion, migration or silent fresh run occurs.
The process timer supervises worker startup/load/train/publication, with OS
termination overrun reported. Parent setup and final reporting are outside it.
Run one writer per study root and keep code/runtime unchanged during the study.

Evaluation is exclusive and cannot resume after consuming its final reservation.
Completed per-game records survive timeout or interrupt. Inspect partial evidence
with the existing `phase7_comparison --report-only` command. The strict study
aggregator intentionally rejects missing/incomplete final evaluations.

PPO records outcomes per training update; search replay retains its own complete
public games and replay ages. These provide self-play outcome/age coverage, not a
guarantee that every public board is visited. There is no shared board-diversity
census in this study. State that limitation when interpreting strength.

## Single-line Windows CMD commands

The launcher works from any CMD directory and stops on the first failure. **Check
first** (read-only, no training/evaluation):

```bat
"C:\Users\micha\Documents\ChatGPT\Board Game AI\run_phase7_study_v1.cmd" check
```

Start the six frozen training jobs sequentially:

```bat
"C:\Users\micha\Documents\ChatGPT\Board Game AI\run_phase7_study_v1.cmd" train
```

After a normal pause/Ctrl+C with a valid saved boundary, explicitly resume using
only the remaining cumulative budgets; completed jobs are skipped and missing
later jobs begin fresh:

```bat
"C:\Users\micha\Documents\ChatGPT\Board Game AI\run_phase7_study_v1.cmd" resume
```

After training completes, check final-evaluation readiness separately:

```bat
"C:\Users\micha\Documents\ChatGPT\Board Game AI\run_phase7_study_v1.cmd" check-evaluation
```

Then start the three paired evaluations (consumes the reserved final roots):

```bat
"C:\Users\micha\Documents\ChatGPT\Board Game AI\run_phase7_study_v1.cmd" evaluate
```

Read-only derived results (stdout; no new games, repairs or report overwrites):

```bat
"C:\Users\micha\Documents\ChatGPT\Board Game AI\run_phase7_study_v1.cmd" report
```

Outputs stay under fresh `runs/phase7-public-study-v1/`. If any command fails,
preserve the directory and share the error. Do not delete evidence or restart a
spent final evaluation. No study training or final evaluation has been run by
Codex. Operator training and final evaluation are now complete; the outcome is
the failed improvement hypothesis linked above.

## Validation

Final full suite: **1,248 passed, 1 skipped in 166.11 seconds**, including available
CUDA regressions. Seven new tiny study tests cover settings pinning, read-only
preflight, exact model continuation through both native recovery paths, cumulative
reservation/source rejection, complete three-seed reporting and corrupted/unknown-
cost evidence rejection. The Windows symlink-privilege test remains skipped.
Ruff lint/format, strict mypy (152 files), dependency/CUDA preflight and whitespace
checks pass. Pytest ran outside the sandbox from its first invocation. All six
frozen study preflights report ready without creating outputs. Exact additive-
source mappings preserve saved CUDA/Phase 6 resume behavior; every pre-existing
Python source byte remains unchanged by this study module. All 841 existing run
files retained their hashes, including operator calibration. At that development
boundary the study directory was absent; operator training subsequently completed.

## Operator training reviewed, 2026-10-18

All six jobs reached the frozen 64-update/256-game boundary with one complete
attempt and certain measured timing. Metrics, native checkpoint configurations,
source/archive/runtime identities, cost counters, initial snapshots and matching
latest/final hashes validate. Every seed/arm retains 256 Adam steps. No study
source, settings, checkpoint, report or budget was edited during review.

| Training root | PPO seconds / decisions / presentations | Search seconds / decisions / simulations / calls / presentations |
|---|---|---|
| 771101 | 5.547 / 1,865 / 7,460 | 37.266 / 1,604 / 51,328 / 36,283 / 65,536 |
| 771102 | 4.687 / 1,910 / 7,640 | 35.484 / 1,680 / 53,760 / 38,712 / 65,536 |
| 771103 | 4.719 / 1,878 / 7,512 | 36.609 / 1,634 / 52,288 / 37,347 / 65,536 |

Total measured child work: 124.312 seconds, excluding parent setup/preflight/final
reporting. Unequal computation is retained; this is not a matched-wall study or a
playing-strength result. All three final-evaluation preflights report ready.
At that training review no evaluation directory existed and the reserved final
roots were unused. The later completed evaluation is recorded in the results above.
All 841 pre-study artifacts remain byte-identical; 1,090 current run files were
hashed during review. Windows blocked the agent's temporary checkpoint copies
inside the sandbox; the read-only preflight/audit passed with approved unrestricted
execution. No operator campaign, repair or run-file mutation occurred in review.
