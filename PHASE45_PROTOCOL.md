# Phase 4.5 bounded investigation protocol

This document preserves the predeclared experiment budgets and decision criteria.
Outcomes and limitations are recorded in [Phase 4.5 results](PHASE45_RESULTS.md).

Declared before the new benchmarks and training runs. Original update-1000
checkpoint, training manifest and logs are read-only. All new runs use separate
directories under `runs/phase45`. Both seats always use the latest shared frozen
policy during collection. Evaluation opponents and tactical categories never
provide training samples, labels, rewards or features.

## Monitoring

Opt-in immutable monitoring configuration: interval 50 updates, retain three
recent checkpoint snapshots plus the best if older, development root 20261022,
128 games per fixed-opponent/seat case and 64 diagnostic self-play games.
Select best by mean score over random/heuristic × X/O, all four cases equally
weighted; ties keep earliest. Resume checkpoint remains `latest.pt`. Development
reports record checkpoint identity, counters, both-seat outcomes/uncertainty,
public tactical probabilities, entropy, distinct positions and sampled actions.
Evaluation preserves training model/optimizer, CPU/CUDA/global Python/NumPy RNG
and per-game/seat training streams. Same-runtime/device monitored/unmonitored
and interrupted/resumed comparisons must agree. Reports are trusted diagnostics.

## Throughput

CPU and available CUDA, one CPU thread, seed 73, 64 games/update, default PPO.
Two warm-up updates, five measured updates per repetition, three repetitions;
alternate device order between repetitions. Synchronize CUDA at timing boundaries.
Measure snapshot/collection (including inference), inference subset, GAE/batch,
optimization, save and end-to-end time, with game/decision/optimizer counts.
Compare the pre-change PPO source with consolidated validation/diagnostic scalar
reads on the same runtime/schedule. Preserve all input/nonfinite checks and
learning arithmetic; validate parameter/optimizer/target/RNG equivalence. Assess
batched inference separately; numerical changes or sampling changes preclude
claiming it is a transparent optimization. Report all repetitions and limitations;
no speedup claim from GPU placement or a single timing sample.

## Learning comparison

Six fresh runs: seeds **42, 43, 44**, unchanged default PPO versus only entropy
coefficient **0.05** (default 0.01). Each run gets **200 updates × 64 games =
12,800 games**, four PPO epochs, minibatch 64. Maximum 115,200 collected decisions
and 7,200 Adam steps per run; actual counts depend on episode lengths and are
reported. Equal games/epochs are the primary budget, not equal wall time or an
assertion of equal decision counts. Total six-run game budget: **76,800**.
No extending budgets or adding variants after viewing results.

Use monitoring above at updates 50/100/150/200. Record all runs, all checkpoints'
development metrics, self-play coverage/action diversity and optimizer counts.
Choose each run's checkpoint using only the declared development score. Once all
six choices are frozen, evaluate each on **512 games per opponent/seat case**,
fresh final root **20261029**, sampled play. Preserve all raw reports.

Report mean outcome score and normal approximate game-level 95% intervals per
seat/opponent (not independent training-seed evidence). Report paired variant
minus baseline effects for all three training seeds and mean ± Student-t(2)
4.303 × paired standard error. Three seeds give weak precision; show individual
results. An improvement is supported only if the paired final balanced-score
interval excludes zero positively; examine both seats and heuristic results for
regressions. Tactical improvements alone do not establish overall strength.

## Decision and scope

If the bounded comparison fails to establish robust PPO improvement, retain PPO
as a baseline and prioritize the bounded public-observation search milestone.
Otherwise retain the supported PPO setting as a candidate requiring further
replication, without claiming solved play. Either decision must cite results.
Task 5 produces a search/learner design only; no MCTS, MuZero, teacher targets,
opponent pool, hidden-state simulation or later-phase implementation in this work.
