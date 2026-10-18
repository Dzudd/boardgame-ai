# Phase 7 public search evaluation

This report records the brought-forward public-search evaluation and subsequent
trace diagnosis. It does not claim completion of all planned Phase 7 tasks.

**2026-10-18 preparation update:** the search-target learner described in the
[implementation review](docs/reviews/2026-10-16-search-target-implementation.md)
already exists. A new isolated paired evaluator accepts its checkpoints alongside
independent PPO, with both-seat random/heuristic W/D/L, explicit byte identities,
raw-policy latency and separately identified optional search. A separate calibration
supervisor records environment/search/evaluator/optimizer/replay costs and enforces
single-use cumulative caps. [Current audit and operator runbook](docs/phase7/COMPARISON_V1.md).
At that preparation milestone only disposable development tests had run. The
subsequent operator calibration and completed learner comparison are reviewed below.
Phase 7 improvement acceptance remains unmet.
The dated results and consumed roots below retain their original scope; statements
about absent learners describe that historical evaluation, not current code.

**Operator calibration reviewed, 2026-10-18:** all six preparation jobs complete;
PPO 3.109–3.218 s and search-target 3.891–3.984 s/eight-game job. PPO used 65
environment decisions, zero simulations, 65 collection calls and 260 presentations;
search used 56 decisions, 1,792 simulations, 1,347 calls and 2,048 presentations.
Each used eight Adam steps. Saved requests/source identities, raw metrics, costs
and checkpoint counters agree, and checkpoint bytes repeat within each learner.
[All measurements](docs/phase7/calibration-results-v1.json). Short startup-inclusive
timing and unequal work do not demonstrate speedup or learning improvement.
The [public study v1](docs/phase7/PUBLIC_STUDY_V1.md) now freezes numeric game/time/
evaluation allocations and implements cumulative-capped supervision, resume and
strict reporting. Operator training and the subsequent final comparison are now
reviewed below. The completed comparison does not establish improvement.

**Operator public-study training reviewed, 2026-10-18:** six jobs each reached
64 updates/256 games in one complete measured attempt, with full metrics/cost
histories and valid matching latest/final checkpoint hashes. Across three seeds,
PPO used 5,653 decisions / 22,612 presentations / 14.953 seconds; search-target
used 4,918 decisions / 157,376 simulations / 112,342 calls / 196,608 presentations /
109.359 seconds. Both used 768 games and 768 Adam steps. No wall-matched or
strength benefit follows from equal games. At training review, all three reserved
final-evaluation preflights passed before the operator started evaluation. The pre-existing 841 artifacts
remain byte-identical, and 1,090 current run files were hashed during review.

**Completed public comparison v1, reviewed 2026-10-18:** all 6,144 final games
completed without errors/truncations. Search-target raw score is **.32764** versus
independent PPO **.34440**. The paired seed effect is **−1.68 percentage points
[−3.81, +0.46]**: improvement is not established, and the interval does not establish
PPO superiority either. All three point differences favor PPO. The predeclared
heuristic-X guard also fails; its lower endpoint is −7.13 points versus the required
−5. Heuristic-O passes the guard without establishing a positive improvement.
Search used **7.31×** measured training wall time and **8.69×** presentations under
the equal-game allocation, which is not a matched-compute study.

[Full results, W/D/L, latency and limits](docs/phase7/PUBLIC_STUDY_V1_RESULTS.md)
and [machine-readable evidence/artifact hashes](docs/phase7/public-study-v1-results.json)
preserve the negative/inconclusive outcome. A read-only replay-fit diagnostic shows
improved in-sample visit/outcome fit in all three search models; that fit does not
explain or reverse the failed external comparison. All 7,252 saved run files retain
their pre-review hashes. No additional game, learning update, repair or campaign
ran during review. Final roots 20261208/20261209/20261210 are consumed; retain the
independent PPO baseline and require a new declaration/fresh roots for any follow-up.
This completed bounded comparison does **not** satisfy Phase 7's improvement
acceptance or complete beliefs/search, chance/same-player, MuZero or broader
planning-cost tasks. Phase 6 negative findings and its uncertain recovered wall
measurement remain unchanged.

Protocol: [predeclared budgets](PHASE7_PROTOCOL.md). Date: 2026-10-15.
Frozen seed-45 update-1000 development-selected checkpoint, SHA256
`b2f8e1c4c7b53f24646e13cba92e4d8ebd6ae32f60dcffa2b4f436cc1f274a80`.
`best.json` selects identical bytes to `latest.pt`; no new checkpoint selection.

The saved sampled-policy final report confirms heuristic X 0/997/3
wins/draws/losses and O 0/0/1000. Its audit recorded 1,024 X wins in 1,024
self-play games, but census tactics remained weak. Self-play dominance did not
imply strength. Below, raw policy is **greedy**, unlike that earlier sampled mode.

## Fixed matches

Learner wins / draws / losses out of 256. All 3,072 matches completed without
errors or truncation. Both seats use X-first rules.

| Opponent / learner seat | Raw greedy | 16 simulations | 32 simulations |
|---|---:|---:|---:|
| random-v1 / X | 224 / 13 / 19 | 235 / 12 / 9 | 229 / 18 / 9 |
| random-v1 / O | 145 / 15 / 96 | 160 / 46 / 50 | 167 / 48 / 41 |
| heuristic-v1 / X | 0 / 256 / 0 | 0 / 256 / 0 | 0 / 256 / 0 |
| heuristic-v1 / O | 0 / 0 / 256 | 0 / 0 / 256 | 0 / 0 / 256 |

The deterministic heuristic repeats one trajectory per seat/condition; 256 copies
do not provide 256 independent strength trials. Search does not fix heuristic-O.
Paired score differences against random use matched absolute game indices and
normal 95% intervals across games (exploratory, unadjusted, not training-seed
uncertainty). Changed game lengths can change opponent stream consumption.

| Search minus raw score vs random | X | O |
|---|---:|---:|
| 16 | +0.0820 [+0.0340, +0.1301] | +0.2383 [+0.1637, +0.3128] |
| 32 | +0.0586 [+0.0065, +0.1107] | +0.3008 [+0.2182, +0.3833] |

Measurements support tactical gains and limited random-opponent gains for this
checkpoint. They do not establish robust strength, optimal play, heuristic gains,
or consistent benefit from doubling 16 to 32 simulations.

## Tactical census

Every live X-first board once (4,520), no symmetry reduction or visit weighting.
Immediate wins take precedence; all winning moves and complete blocks accepted.
Unavoidable threats/quiet positions excluded. Labels remain measurement only.

| Seat / category (positions) | Raw greedy | 16 simulations | 32 simulations |
|---|---:|---:|---:|
| X / win (1498) | 785 (52.40%) | 1063 (70.96%) | 1128 (75.30%) |
| X / block (396) | 167 (42.17%) | 195 (49.24%) | 221 (55.81%) |
| O / win (860) | 445 (51.74%) | 666 (77.44%) | 712 (82.79%) |
| O / block (580) | 225 (38.79%) | 284 (48.97%) | 319 (55.00%) |

## Cost

CPU float32, one thread, PyTorch 2.14.1+cu130. Sixteen empty-board warmup
decisions per condition excluded. Latency includes validation/search, excludes
runner copies. No terminal inference. At most simulations+1 calls per decision.

| Budget | Case | Decisions | Evaluator calls | Latency ms mean / p50 / p95 |
|---|---|---:|---:|---:|
| 0 | random-v1 / X | 922 | 922 | 0.169 / 0.166 / 0.182 |
| 0 | random-v1 / O | 864 | 864 | 0.167 / 0.165 / 0.178 |
| 0 | heuristic-v1 / X | 1280 | 1280 | 0.172 / 0.167 / 0.208 |
| 0 | heuristic-v1 / O | 512 | 512 | 0.192 / 0.176 / 0.269 |
| 16 | random-v1 / X | 894 | 8157 | 2.698 / 2.456 / 4.744 |
| 16 | random-v1 / O | 879 | 7224 | 2.547 / 2.486 / 4.549 |
| 16 | heuristic-v1 / X | 1280 | 9984 | 2.404 / 2.444 / 4.644 |
| 16 | heuristic-v1 / O | 512 | 8448 | 4.239 / 4.192 / 4.770 |
| 32 | random-v1 / X | 894 | 11097 | 4.460 / 3.298 / 9.007 |
| 32 | random-v1 / O | 887 | 10207 | 4.484 / 4.377 / 8.341 |
| 32 | heuristic-v1 / X | 1280 | 13824 | 4.160 / 3.931 / 8.851 |
| 32 | heuristic-v1 / O | 512 | 12032 | 7.213 / 7.572 / 8.063 |

Census cost (4,520 decisions each):

| Budget | Evaluator calls | Latency ms mean / p50 / p95 |
|---|---:|---:|
| 0 | 4520 | 0.155 / 0.152 / 0.170 |
| 16 | 19107 | 1.454 / 1.330 / 3.266 |
| 32 | 21720 | 2.020 / 1.815 / 4.704 |

Single sequential timing run, not repeated isolated benchmarking; concurrent
correctness checks can affect scheduling. No hardware-general latency claim.
A prior validation run is retained in `runs/phase7-seed45-search-20261112`.
After removing transient frontier Node allocations, the final run reproduced
identical match records and tactical results. Timings above supersede the first
run. This was semantic replay verification, not checkpoint selection or tuning.

## Correctness and limits

Independent exhaustive engine comparison covers 5,478 boards and 16,167 legal
transitions **per starting seat**, including terminal roots, legality, rewards,
phases and branch isolation. Tests cover parent-seat values, clipped leaf
perspective, terminal zero bootstrap/no double counting, vector arithmetic without
automatic negation, deep opponent blocks, multiple wins/ties, invalid evaluations,
masking, simulation/node/depth limits, visits, private exploration, global RNG,
and frozen weights/mode/gradients. A guard disables engine construction, cloning,
queries and steps during search. Generic same-actor backup arithmetic is tested;
same-actor game simulation is unsupported.

Public deterministic Tic-Tac-Toe only. Reachability is validation, not solved-value
lookup. PPO value is discounted heuristic guidance. Peaked priors and low budgets
can starve useful branches despite the 1e-8 floor. No tree reuse, transpositions,
batching, wall-clock cutoff, search-target learner, replay, teachers, opponent
pool, learned dynamics or reward bonuses. No training code changed.

Raw ignored reports: `runs/phase7-seed45-search-20261112-final/results.json` and
per-condition JSON contain seeds, records, source hashes, runtime and metrics.
Checkpoint/config/log hashes remained identical. Seed 20261112 is now consumed;
reserve fresh final seeds if these results guide tuning.

## Heuristic-O public trace diagnosis

Diagnosis completed 2026-10-16.

Replayed match index zero from each existing raw/16/32 heuristic-O case, using
its original engine/agent reset seeds and unchanged checkpoint/settings. No new
strength test, final seed, budget sweep or evaluator intervention. All three
reproduce cells **X4, O3, X0, O2, X8**, ending X +1 / O -1. Cells are zero-based.
Diagnostic output is `runs/phase7-seed45-heuristic-o-trace/trace.json`.

At the first O decision, O3 has prior .998911 and 9/25 visits at 16/32;
every other root move gets one visit. Its Q(O) is -.979780/-.986085.
Search anticipates X2 (prior .999115, 8/24 visits) after O3. The actual
heuristic reply X0 has prior .00007449 and **zero visits** in that subtree.
This is an observed mismatch between neural-guided search and this opponent,
not an oracle claim about either move's minimax value.

At the second O decision, board `X.. / OX. / ...`, cell 8 is the sole immediate
block. Root statistics are:

| O action | Prior | Visits 16 / 32 | Q(O) 16 / 32 | First child value for O |
|---|---:|---:|---:|---:|
| 2, selected | .99431650 | 11 / 27 | -.600094 / -.408561 | -.858250 |
| 8, required block | .00067678 | 1 / 1 | -1 / -1 | -1 (clipped) |

After O2, X can win immediately at 8. That edge has prior .00704158 and **zero
visits** in both trees. X5/X6 instead get 7/3 visits at 16 simulations and 18/8
at 32. The larger search includes draws after X misses its immediate win, raising
O2's mean from -.600 to -.409 despite the real opponent winning immediately.
This is an incomplete adversarial search at these budgets, not a strength gain.
Unvisited terminal metadata in the report comes from public rules inspection;
it was not a searched terminal reward or a backed-up value.

After the required O8 block, the next X-observer value is **+1.212914**, clipped
to +1 for X / -1 for O. The branch receives no second visit. This demonstrates
pessimistic guidance plus low prior exposure, but **does not prove the value is
wrong or that blocking saves the game**. No solver or intervention was used to
separate causal contributions. Likewise the negative root value is recorded for
diagnosis; search does not back up the root prediction itself.

All four search decisions hit neither node nor depth caps (maximum depth 4/6
for the first O decision, 6/6 for the second). Finite simulation budgets, strongly
concentrated priors at both players' nodes, and leaf-value guidance interact;
this is more specific than a resource-cap or proven backup-defect explanation.
Every recorded path was independently reconstructed from the public root:
transition rewards, leaf perspective/clipping, terminal zero bootstrap, all edge
visit counts and two-seat W sums agree. Terminal rewards enter once.

The optional `on_simulation` observer records immutable public paths, stop reasons,
leaf continuations and per-edge backed returns. It feeds nothing into selection.
Tests compare traced/untraced trees and actions, including seeded exploration
and resource cutoffs, reject corrupted returns, and verify no terminal inference.
Separate candidate probes report child values and immediate public winning replies;
they run after the decision and never alter search, training, or evaluator counts
reported for the original search. All source artifact hashes were preserved.

Reproduce with an absent output directory:

```powershell
python -m boardgame_ai.search.diagnosis runs/monitored-seed45/latest.pt runs/phase7-seed45-search-20261112-final/results.json runs/new-heuristic-o-trace
```

**Recorded decision after milestone review:** return to Phase 5's minimal playable
game, mechanics and visibility tests. That follow-up has since completed at the
bounded scope recorded in [Phase 5 results](docs/archive/PHASE5_RESULTS.md). The brought-forward
search milestone is complete; solving Tic-Tac-Toe or further prior tuning is not a
gate to broader platform work. See [PROGRESS.md](PROGRESS.md) for the current milestone.
A fixed-root prior-concentration ablation remains optional deferred research, with
unchanged value head and 16/32 budgets. Any later strength claim needs a fresh
reserved evaluation protocol. Search-target learning remains separate future work.
