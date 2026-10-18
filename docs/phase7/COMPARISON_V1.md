# Phase 7 public learner comparison: preparation v1

Declared 2026-10-18, after Phase 6 commit `d401087`. This is an evaluator and
calibration milestone, not a completed learning experiment. Phase 6's negative
transfer, failed consistent-speedup hypothesis and uncertain recovered wall timing
remain as recorded in its results. No Phase 8 work is included.

**Subsequent status:** the operator completed all six calibration jobs; their
[offline review](calibration-results-v1.json) validates requests, metrics, costs and
checkpoints. The [frozen public study v1](PUBLIC_STUDY_V1.md) now provides numerical
budgets, cumulative-capped training/resume and strict evidence reporting. The
preparation audit/declarations below retain their original scope and chronology;
statements about pending calibration/study code are superseded by that study
declaration. No learning-quality comparison outcome is available yet.

## Audit and dependency order

| Item | Implementation | Improvement evidence / next dependency |
|---|---|---|
| Independent Tic-Tac-Toe PPO | Existing learner, checkpoints, recovery | Retained unchanged as the simpler baseline |
| Public PUCT / search-target learner | Existing public simulator, own visit/outcome targets, FIFO replay, exact boundary resume | Historical frozen-PPO search gains against random only; no search-target versus PPO learning comparison |
| Paired checkpoint evaluator | New isolated CPU evaluation for both formats | Disposable correctness tests only |
| Throughput calibration | New separately supervised jobs and cost summaries | Operator measurements pending; eight games/job do not establish steady-state or full-replay throughput |
| Numerical study declaration / capped campaign runner | Not yet frozen / not implemented for this comparison | Calibration first; existing update-only runners must not be presented as wall-capped study commands |
| Multi-seed improvement | Not demonstrated | Fresh, complete, budget-comparable training and final evaluations |
| Beliefs and information-set search | Not implemented | Define observation/history belief state, indistinguishable-world tests, then held-out belief calibration and information-set baseline |
| Chance and same-player search | Not supported by public simulator | Explicit chance probability/reward/discount semantics and game-level tests; vector arithmetic alone is insufficient |
| MuZero representation/dynamics/prediction | Not implemented | Visible-history latent contract, self-generated sequence/replay targets, unroll masks, reward/value perspectives and independent dynamics tests |
| Planning comparisons | Historical single-checkpoint 0/16/32 probe only | Heuristic, rollout if warranted, neural and hybrid comparisons at declared inference/training budgets and latency; no phase-wide claim |

The next code milestone after calibration is a study-specific capped training
envelope and strict evidence aggregator, preserving the existing learners,
checkpoint publication order and metrics recovery. No implementation is counted
as measured improvement. Hidden-information work follows the public comparison
gate; chance/same-player semantics must precede a search implementation claiming
those mechanics. Learned dynamics follows explicit sequence/visibility contracts.
The roadmap's remaining tasks and acceptance criterion are unchanged.

## Evaluator contract

`boardgame_ai.training.phase7_comparison` accepts explicit `--ppo` and `--search`
paths. It validates the two distinct formats on CPU and records the exact loaded
byte hashes, format IDs, paths, update and optimizer counters. It loads a temporary
copy of those hashed bytes; no checkpoint, replay, optimizer, log or training
manifest is written. Run only against stopped checkpoints. Preflight validates
without starting games, modifying RNG/threads, repairing logs or creating output.
Preflight is neither a path reservation nor a bounded performance measurement.

The evaluator shares existing fixed-opponents-v1 + batch-v1 absolute-index
schedules across arms: random-v1/heuristic-v1, learner X/O, X starts. Schedules
pair engine/policy/opponent seeds, not necessarily later RNG consumption after
different actions. Primary evaluation is the sampled raw policy. Optional positive
`search_simulations` conditions use deterministic PUCT and are separately labelled
`search-assisted`; neither their scores nor their latency enter the primary
raw-policy result. Greedy versus deterministic heuristic repeats trajectories
and must never be treated as independent trials.

Each immutable `record-NNNNNNNN.json` contains one game's raw outcome, seeds,
length, errors/truncation and all learner decision latencies/evaluator-call counts.
`identities.json` identifies loaded checkpoint bytes. `results.json` reports
per-arm/opponent/seat/condition W/D/L and mean/p50/p95 inference latency. Cold CPU
latency times decision calls only; checkpoint loading, model-copy setup, runner
copies and disk I/O are excluded. There is no warmup subtraction or hardware-general
benchmark claim. Baseline games/diagnostics never enter either learner.

The parent claims a fresh absent directory and reserves the entire wall allowance
before launching one hidden child. A timer supervises child startup, validation,
inference/training and checkpoint/per-record publication. At the cap or Ctrl+C it
kills and reaps the child, then publishes a partial aggregate and returns exit 2.
OS scheduling/termination can overrun; the measured overrun is reported. Parent
request/provenance setup and final aggregation/publication are outside that timer.
There are also cumulative deterministic game/decision/simulation/call/optimizer/
presentation upper bounds in preflight and `request.json`, derived from the fixed
schedule and nine-decision game bound. No automatic cap extension or retries.

Every invocation has a **single-use cumulative reservation**, including failed
attempts. Existing outputs reject, including an empty directory. Evaluation cannot
resume. Hard-killing the parent leaves the reservation fully spent and already
published records intact; `--report-only` reads them without games or repair.
Unpublished attempted work is unknown, never counted as zero or fabricated as
accepted samples. A hard kill between checkpoint and callback publication may
leave an additional saved training boundary; original learner recovery remains
available but calibration is not restarted. A new measurement requires a separately
declared fresh allocation, not reuse of the spent job. Disk failures may prevent
the final aggregate; previously published records remain authoritative evidence.
The child verifies source hashes against the saved request before work and claims
an exclusive worker-start marker. One writer and unchanged source during each job
are required. Current code does not migrate any historical source identities.

## Frozen calibration allocation

Three sequential repetitions per arm, six jobs maximum: each has **60 seconds,
2 updates × 4 games**, CPU float32/one thread, seed 771001. Alternate arm order
as in the commands below. No held-out evaluation occurs during calibration.
Settings explicitly store every PPO/search parameter. Calibration retains the
unchanged native initialization/sampling schedules of each independent learner;
a common root seed does not promise identical initial weights across algorithms.
The manifest `preparation-v1.json` hashes these versioned settings.

The search calibration uses 32 simulations/decision, 4 optimizer steps/update,
batch 256, capacity 10,000 completed games. PPO retains its original independent
defaults, including four epochs. Each arm has at most 72 accepted environment
decisions/job. Search has at most 2,304 simulations, 2,376 evaluator calls, eight
Adam steps and 2,048 sample presentations/job. PPO has zero simulations, at most
72 collection evaluator calls, eight Adam steps and 288 presentations/job.
Six reservations total at most 360 child-work seconds; setup/finalization is extra.

Record accepted games/decisions, actual search simulations/evaluator calls, Adam
steps, sample presentations, presentation-to-decision ratio and search replay age
per update. `results.json.costs` sums durable records and divides accepted work by
end-to-end child wall time, including startup, initialization, optimizer work and
checkpoint/report I/O. Interrupted work remains an unknown lower-bound gap.
PPO epoch reuse is distinguished from historical replay, which PPO does not use.
Report all repetitions and ranges, not the fastest run. A full-capacity replay
validation cost cannot be extrapolated from eight-game calibration; if material,
declare a new separately capped calibration allocation before measuring it.

## Prospective study rules (not a runnable training declaration)

Reserve training roots **771101, 771102, 771103**; development evaluation root
**20261207**; final roots **20261208, 20261209, 20261210**, respectively paired by
training root. These roots were absent from the inspected project declarations;
they are reserved, not consumed by development tests. All later tuning after final
inspection consumes those roots and requires a new version/fresh final roots.

Keep independent PPO, latest-policy collection in both seats, existing visible
features and native initialization/collection schedules. No teachers, solver
targets, fixed baseline training opponents, reward bonuses, opponent pools or
privileged critics. Replay remains only the search learner's own completed games.
Preserve all models and failed hypotheses. No authoritative hidden-state clones.

Primary budget axis: equal complete self-play games, with decisions and every
search/optimizer/reuse cost reported separately. After calibration, freeze the
numeric total games, decision/simulation/call/presentation/optimizer ceilings,
wall allowances, checkpoint schedule and evaluation allocations in a **new study
version before training outcomes**. A compute-matched arm requires its own frozen
wall schedule, not post-hoc relabelling of the equal-game experiment. Compare
actual wall time and accepted/attempted sample costs separately; uncertain crash
reservations disqualify exact realized wall matching.

Selection rule: final scheduled saved checkpoint, no development-best selection.
Evaluate all three paired training seeds; primary score is the equal mean of the
four raw sampled-policy random/heuristic × X/O case scores. Compute search-minus-
PPO paired seed effects, mean and descriptive 95% t(2) interval
`mean ± 4.303*SD/sqrt(3)`. Advance only when its lower endpoint is >0 and both
heuristic seat effects separately have lower endpoints >= -0.05. Report per-seat
W/D/L, errors, within-case uncertainty separately from training-seed uncertainty,
latency, self-play coverage, costs and all negative/regression outcomes. No
multiple-comparison-adjusted or universal-strength claim. Failed or incomplete
seeds cannot be silently dropped. Boundary/provenance/isolation/nonfinite failures
stop immediately; caps cannot be extended because results are weak.

**Still required:** operator calibration reports; numerical study freeze; capped
study runner/evidence aggregator validation; then operator training and fresh final
evaluation. Neither the calibration allocation nor the 64-game development
evaluator template authorizes a learning-quality acceptance claim.

## Windows CMD runbook

For a single-line launcher from any CMD directory, run
`"C:\Users\micha\Documents\ChatGPT\Board Game AI\run_phase7_calibration_v1.cmd" check`
for the six read-only preflights. Then use the same command with `run` to recheck
and execute the six jobs sequentially. The launcher stops at the first failure,
never overwrites output and does not resume or retry spent jobs.

Run from the repository root. Every command below is one CMD line. Outputs must
be absent; do not pre-create them. Read-only preflight is separate from execution.

Preflight (no campaigns, no output creation):

```bat
.venv\Scripts\python.exe -m boardgame_ai.training.phase7_comparison runs\phase7-calibration-ppo-v1-r1 --settings docs\phase7\calibrate-ppo-v1.json --check-only
.venv\Scripts\python.exe -m boardgame_ai.training.phase7_comparison runs\phase7-calibration-search-v1-r1 --settings docs\phase7\calibrate-search-v1.json --check-only
```

Operator calibration only, the six declared reservations (do not run concurrently):

```bat
.venv\Scripts\python.exe -m boardgame_ai.training.phase7_comparison runs\phase7-calibration-ppo-v1-r1 --settings docs\phase7\calibrate-ppo-v1.json
.venv\Scripts\python.exe -m boardgame_ai.training.phase7_comparison runs\phase7-calibration-search-v1-r1 --settings docs\phase7\calibrate-search-v1.json
.venv\Scripts\python.exe -m boardgame_ai.training.phase7_comparison runs\phase7-calibration-search-v1-r2 --settings docs\phase7\calibrate-search-v1.json
.venv\Scripts\python.exe -m boardgame_ai.training.phase7_comparison runs\phase7-calibration-ppo-v1-r2 --settings docs\phase7\calibrate-ppo-v1.json
.venv\Scripts\python.exe -m boardgame_ai.training.phase7_comparison runs\phase7-calibration-ppo-v1-r3 --settings docs\phase7\calibrate-ppo-v1.json
.venv\Scripts\python.exe -m boardgame_ai.training.phase7_comparison runs\phase7-calibration-search-v1-r3 --settings docs\phase7\calibrate-search-v1.json
```

Inspect any job's durable evidence, even if the parent was terminated (replace
only the repetition/arm path for the other five jobs):

```bat
.venv\Scripts\python.exe -m boardgame_ai.training.phase7_comparison runs\phase7-calibration-ppo-v1-r1 --report-only
```

Optional paired **development** evaluation of the calibration checkpoints: 64
games/case, 512 total games, 4,608 environment decisions/calls maximum, 120 seconds.
This uses the development root, never final roots, and cannot test learning quality.
No current saved search campaign exists, so these paths become available only
after the operator's calibration. First preflight:

```bat
.venv\Scripts\python.exe -m boardgame_ai.training.phase7_comparison runs\phase7-paired-development-v1 --settings docs\phase7\paired-evaluation-v1.json --ppo runs\phase7-calibration-ppo-v1-r1\training\latest.pt --search runs\phase7-calibration-search-v1-r1\training\latest.pt --check-only
```

Then, only if requested by the operator, evaluation execution:

```bat
.venv\Scripts\python.exe -m boardgame_ai.training.phase7_comparison runs\phase7-paired-development-v1 --settings docs\phase7\paired-evaluation-v1.json --ppo runs\phase7-calibration-ppo-v1-r1\training\latest.pt --search runs\phase7-calibration-search-v1-r1\training\latest.pt
```

Reports and raw records stay in these fresh directories. Do not overwrite older
Phase 4/4.5/6/7 evidence or relabel calibration as training-study evidence.

## Development validation, 2026-10-18

Final full suite: **1,241 passed, 1 skipped in 103.06 seconds**, including available
CUDA regressions. Thirteen new tests cover both formats, read-only/RNG isolation,
paired raw/search schedules, W/D/L and latency, worker timeout reservations,
completed-record preservation, both learners' cost counts, invalid settings,
byte-frozen preparation configuration and read-only unfinalized-journal inspection.
Only disposable tests execute tiny games/updates; these are not calibration data.
Ruff lint/format, strict mypy (150 files), dependency/CUDA preflight and whitespace
checks pass. Pytest ran outside the sandbox from its first invocation. The skip is
the existing Windows symlink privilege case.

The first full run found five exact-source compatibility failures (1,234 passes,
one skip). Excluding the new Phase 7 module reproduced predecessor `9fba8b1f...`
exactly; all older Python source is unchanged. Updated exact allowlists retain
that predecessor and the prior reviewed identities without rewriting saved
provenance. The repair passed 35 focused tests and the final full suite above.
Read-only checks accept both saved CUDA campaigns and the recovered Phase 6 job.
All 778 original run artifacts retain their hashes; no new run files exist.
The new calibration preflights report ready without creating outputs. No training
or evaluation campaign was launched, and no changes were committed.
