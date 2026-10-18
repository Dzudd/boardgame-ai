# Phase 4.5 investigation results

This report records monitoring validation, throughput measurements, the controlled
self-play comparison and the resulting bounded-scope decision.

Protocol: [predeclared budgets and criteria](PHASE45_PROTOCOL.md).
Diagnosis: [Task 1](docs/archive/POLICY_DIAGNOSIS.md). Search milestone:
[Task 5 design](docs/reference/SEARCH_LEARNING_DESIGN.md).

## Monitoring and isolation

Optional monitoring keeps `latest.pt` as the resume marker, stores immutable
update/hash-named snapshots, and points `best.json` to the checkpoint with the
highest equally weighted development score over random/heuristic × X/O. Ties
keep the earliest. Retention keeps three recent snapshots plus an older best
if needed; development reports remain available after snapshot pruning.

Monitoring has an immutable separate manifest and must match on resume; old
unmonitored runs still resume unchanged with monitoring disabled. Reports include
hashes, game/sample/optimizer counters, per-seat outcomes and approximate
game-level intervals, census/visited tactics, entropy, positions and action
diversity. Missing historical log rows make the cumulative sample count unknown,
not invented. A failed current-boundary evaluation can be retried on resume.
It does not roll back a completed saved training update.

CPU and CUDA tiny-run comparisons found monitored, unmonitored and resumed
parameters, Adam tensors and private shuffle state exactly equal. Tests verify
bounded retention, best pinning/earliest ties, immutable preservation, retry,
monitoring-config mismatch rejection and global RNG restoration on failure.
Policy evaluation loads an independent CPU copy; evaluation games never enter
training. This establishes tested isolation/resume behavior, not policy strength.

## Repeated throughput evidence

PyTorch 2.14.1+cu130, RTX 4070 Ti, one CPU thread. Seed 73, default PPO, 64
games/update; two warm-up updates followed by five measured updates in each of
three repetitions. Devices and reference/optimized order alternate. CUDA is
synchronized at stage boundaries. Snapshot transfer is in collection; inference
is a **subset** of collection, not an additional stage. Timings include diagnostic
instrumentation and checkpoint I/O, but not development monitoring.

Reference PPO source was copied before edits from commit
`fe66184e5f43269c41e1c79842ec54ee4c4ec273`. The adopted change consolidates finite
checks and six diagnostic scalar reads, accumulates losses in float64 to retain
Python-float summation semantics, and validates the trainer-owned batch once
rather than repeating input validation on each internal minibatch. Public loss
validation, per-minibatch output/gradient nonfinite checks, sample order, targets,
masking and Adam arithmetic remain. No collection/sampling protocol changed.

Every measured five-update group has 320 games, 2,392 decisions and 160 Adam
steps. Times below are group means per update, in seconds:

| Device / version | Rep | Collection | Inference subset | GAE/batch | PPO | Save | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU reference | 0 | .217033 | .071182 | .053311 | .069371 | .005835 | .345550 |
| CPU optimized | 0 | .272624 | .082028 | .067187 | .068463 | .006524 | .414798 |
| CPU reference | 1 | .171718 | .057315 | .040633 | .053930 | .004578 | .270859 |
| CPU optimized | 1 | .170709 | .057193 | .039474 | .048198 | .004701 | .263082 |
| CPU reference | 2 | .174294 | .058215 | .040722 | .054057 | .005089 | .274162 |
| CPU optimized | 2 | .173008 | .057903 | .040475 | .048735 | .004723 | .266941 |
| CUDA reference | 0 | .244703 | .079915 | .063134 | .260584 | .007720 | .576141 |
| CUDA optimized | 0 | .219188 | .071933 | .050213 | .195115 | .006863 | .471379 |
| CUDA reference | 1 | .177703 | .059210 | .041184 | .212888 | .006041 | .437816 |
| CUDA optimized | 1 | .172672 | .057449 | .040474 | .171230 | .006772 | .391149 |
| CUDA reference | 2 | .180144 | .060005 | .044471 | .223500 | .006223 | .454338 |
| CUDA optimized | 2 | .174825 | .058426 | .041443 | .169917 | .006257 | .392441 |

CUDA PPO and end-to-end time are lower in each repetition. CPU end-to-end results
are mixed, including a slower optimized repetition 0: **no general CPU speedup
claim**. CPU remains faster overall than CUDA for this small workload in the
recorded groups. Device order, startup/background load and only three repetitions
limit generalization. Game/decision rates and individual-update data are retained
in `runs/phase45/benchmark-final/benchmark.json`.

Reference versus optimized final parameters, Adam tensors, private shuffle state
and optimizer counts matched **exactly on each device** after all seven updates
in each repetition. CPU-versus-CUDA equality is not promised. Independent Task 1
PPO arithmetic tests and existing nonfinite/resume tests pass as well.

A fixed public-observation corpus from 16 self-play games (123 decisions) compares
single and batched inference over five measurements: .00959–.00992 s separately
versus .000173–.000246 s batched. Max logit/value/probability differences are
8.94e-8 / 7.45e-8 / 5.96e-8. This is a forward-pass probe, **not** an end-to-end
batched collector or guarantee of identical categorical draws/trajectories.
Batching could reduce inference overhead but changes floating arithmetic and
needs a separate per-game/seat RNG and collection-equivalence assessment. It was
not adopted; rollout targets, reward timing and training sampling stay unchanged.

A preliminary scalar-transfer-only candidate was also measured before the final
redundant-validation change. It is superseded for implementation selection, not
discarded from evidence: raw data remain in `runs/phase45/benchmark/benchmark.json`.
Same budget/runtime; the three repetition means (seconds/update) were:

| Preliminary device/version | End-to-end reps 0/1/2 | PPO reps 0/1/2 |
|---|---|---|
| cpu/reference | 0.274306, 0.263566, 0.266772 | 0.053989, 0.051860, 0.052960 |
| cpu/optimized | 0.293319, 0.271484, 0.266033 | 0.053822, 0.055074, 0.052582 |
| cuda/reference | 0.408676, 0.441363, 0.423108 | 0.183395, 0.220139, 0.208813 |
| cuda/optimized | 0.412956, 0.423600, 0.428692 | 0.191838, 0.201955, 0.213899 |

That candidate also passed exact final model/Adam/shuffle equivalence. The final
table above is the adopted version; neither comparison changes learning budgets.

## Controlled self-play comparison

All six predeclared runs completed: **76,800 training games**, 200 updates per
run, 64 games/update, identical defaults except entropy .01 versus .05. Both seats
used the latest shared frozen policy; no evaluation sample entered training.
All six development choices were frozen in `selections.json` before any final
evaluation. Each selected checkpoint played 512 sampled games per opponent/seat
on fresh final root 20261029 (12,288 final games total), with no errors/truncations.
These evaluate development-selected checkpoints, not necessarily update 200.
Actual decision/Adam counts differ with episode length under the declared equal-
game/epoch budget. No budget extension or extra variant was run.

| Setting / seed | Selected update | Full-run decisions | Full-run Adam steps | Final balanced score |
|---|---:|---:|---:|---:|
| baseline / 42 | 100 | 100,671 | 6,712 | -0.0356 |
| entropy / 42 | 150 | 94,641 | 6,344 | -0.1982 |
| baseline / 43 | 100 | 92,085 | 6,144 | -0.2217 |
| entropy / 43 | 200 | 84,926 | 5,712 | -0.2666 |
| baseline / 44 | 100 | 94,875 | 6,332 | -0.3765 |
| entropy / 44 | 200 | 92,495 | 6,212 | -0.1558 |

Every measured development score (equal-case mean, 128 games/case):

| Setting / seed | Update 50 | 100 | 150 | 200 |
|---|---:|---:|---:|---:|
| baseline / 42 | -0.0664 | +0.0156 | -0.2344 | -0.4531 |
| entropy / 42 | -0.3477 | -0.3086 | -0.2422 | -0.4043 |
| baseline / 43 | -0.3047 | -0.2207 | -0.2461 | -0.3125 |
| entropy / 43 | -0.3047 | -0.3184 | -0.3340 | -0.2949 |
| baseline / 44 | -0.4023 | -0.3906 | -0.4180 | -0.4434 |
| entropy / 44 | -0.4062 | -0.4629 | -0.3789 | -0.1621 |

Final per-seat scores and approximate 95% game-level intervals:

| Setting / seed | Random X | Random O | Heuristic X | Heuristic O |
|---|---|---|---|---|
| baseline / 42 | +0.438 [+0.364, +0.511] | -0.176 [-0.256, -0.096] | -0.191 [-0.226, -0.157] | -0.213 [-0.248, -0.177] |
| entropy / 42 | +0.525 [+0.460, +0.591] | -0.088 [-0.165, -0.011] | -0.256 [-0.295, -0.217] | -0.975 [-0.988, -0.961] |
| baseline / 43 | +0.564 [+0.500, +0.629] | -0.127 [-0.208, -0.046] | -0.324 [-0.365, -0.284] | -1.000 [-1.000, -1.000] |
| entropy / 43 | +0.621 [+0.554, +0.688] | +0.258 [+0.175, +0.341] | -0.967 [-0.989, -0.945] | -0.979 [-0.991, -0.966] |
| baseline / 44 | +0.506 [+0.440, +0.572] | -0.012 [-0.095, +0.072] | -1.000 [-1.000, -1.000] | -1.000 [-1.000, -1.000] |
| entropy / 44 | +0.402 [+0.331, +0.474] | -0.240 [-0.315, -0.166] | -0.199 [-0.236, -0.162] | -0.586 [-0.629, -0.543] |

Intervals above follow the predeclared normal approximation and do not measure
training-seed uncertainty. At all-loss boundaries the normal interval degenerates;
it must not imply certainty of losing every future game. A distribution-free
95% Hoeffding interval for a bounded [-1,1] score with n=512 has radius .1200
(clipped to [-1,1]); an all-loss case therefore also permits means up to -.880.
Identical held-out schedules across conditions support paired comparisons; games
within different seeds are not evidence of many independent training runs.

Variant-minus-baseline effects are **-.1626, -.0449, +.2207** for training seeds
42/43/44. Mean **+.0044**, paired Student-t(2) 95% interval **[-.4834, +.4922]**.
The declared positive-lower-bound improvement criterion fails. The three-seed
mean itself is close to zero and highly uncertain; this does not establish that
all entropy changes fail, nor that more training cannot help.

| Final case | Mean paired effect | Training-seed 95% interval |
|---|---:|---|
| random-v1 / X | +0.0137 | [-0.2414, +0.2688] |
| random-v1 / O | +0.0814 | [-0.6805, +0.8433] |
| heuristic-v1 / X | +0.0312 | [-1.7734, +1.8359] |
| heuristic-v1 / O | -0.1087 | [-1.5959, +1.3784] |

Coverage below is measured from **64 fresh diagnostic self-play games per
selected checkpoint**, not reconstructed historical training visitation. Repeated
decisions are correlated; distinct position counts depend on this finite budget.
P/A are distinct acting-seat boards / distinct board-action pairs. Entropy is
mean legal-policy entropy in nats. No monotonic diversity or strength claim.

| Setting / seed | X P/A | O P/A | X entropy | O entropy |
|---|---:|---:|---:|---:|
| baseline / 42 | 24/38 | 20/26 | 0.477 | 0.395 |
| entropy / 42 | 50/68 | 43/66 | 0.529 | 0.699 |
| baseline / 43 | 24/33 | 21/28 | 0.377 | 0.297 |
| entropy / 43 | 19/28 | 15/24 | 0.260 | 0.834 |
| baseline / 44 | 26/36 | 18/32 | 0.277 | 0.627 |
| entropy / 44 | 40/54 | 32/45 | 0.468 | 0.543 |

Census immediate-win mass remains roughly 50–54%; block mass 31–44% across
the selected checkpoints. Higher entropy sometimes expands sampled coverage,
but seed 43 still narrows substantially and seat/opponent regressions occur.
In particular, seed 42 heuristic-O score falls from -.213 to -.975; seed 44
improves strongly against the heuristic while losing score against random.
These differences reinforce the need to report seats and opponents separately.

Raw evidence: `runs/phase45/entropy-comparison/experiment.json`, all training
manifests/logs, 24 development reports, `selections.json`, six final reports and
`results.json`. The original update-1000 checkpoint, configuration and log retain
exactly the three SHA-256 identities in [POLICY_DIAGNOSIS.md](docs/archive/POLICY_DIAGNOSIS.md).
The original run was never resumed or overwritten. Final seed 20261029 is now consumed; reserve
new final seeds for future tuning or milestone assessment.

## Recorded decision — Task 6

**Retain default PPO as a reference and prioritize bounded public-observation
search evaluation before adding harder games.** The controlled entropy change
did not establish robust improvement at its declared budget. This is a reason
to stop extending this particular tuning loop, not evidence that PPO is incapable
or that search already improves play. Preserve all runs, including regressions.

The follow-on Phase 7 public simulator and bounded 16/32 search evaluation are
now implemented and measured in [Phase 7 results](PHASE7_RESULTS.md), using the
Task 5 design, reporting both seats, tactics, evaluator calls and latency
against raw-policy play. Search-target replay learning and learned MuZero dynamics
remain later steps; Phase 4.5 itself added no search code, external training
opponents, teacher labels, reward bonuses or opponent pool.

Phase 4.5 Tasks 1–6 are complete at the stated bounded scope. The numerical
learning checks establish tested signal/update consistency; monitoring makes
future changes inspectable; repeated timing supports limited same-device
optimization evidence. None of this fixes the frozen policy or establishes
optimal/robust play. Hidden-information generalization and stronger learning
remain unresolved research work.

The current milestone and phase statuses are maintained in [PROGRESS.md](PROGRESS.md)
and the [project plan](BOARDGAME_AI_PROJECT_PLAN.md).
