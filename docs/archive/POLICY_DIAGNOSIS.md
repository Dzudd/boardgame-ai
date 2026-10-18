# Frozen policy diagnosis — Phase 4.5 Task 1

This report records the tactical census, sampled learning signals and independent
PPO arithmetic checks for one preserved checkpoint.

Measured 2026-10-15 on the update-1000 checkpoint (29,020 optimizer steps),
PyTorch 2.14.1+cu130, CPU, one thread. Task 1 is complete; policy strength is
unchanged. No demonstrated learning-signal implementation defect was found.

## Protocol and boundaries

The audit enumerates all **4,520 live reachable X-first boards**, once each,
in row-major depth-first order, stopping at wins/draws. No symmetry reduction,
history weighting, minimax, fork solver, or perfect-play labels. Each board has
equal weight within its seat/category; this census is not a policy visitation
distribution. The engine agrees with the complete generated decision set in tests.

An immediate win takes precedence over a block. Success includes **any** supplied
legal winning move. With no own win, a required block must prevent **every**
immediate opponent win. Multiple distinct threats that no placement can prevent
are counted separately as unavoidable, not failed block opportunities. This is
one-ply diagnosis, not a claim that quiet positions are safe or strategically good.

Policy and tactical functions consume only public acting-seat observations and
supplied legal actions. The network uses the existing observer-relative encoding;
coordinates stay row-major for both X and O. No rewards, seeds, engine references,
state snapshots, or tactical labels become policy/value features. Tactical labels
never enter an optimizer batch. Trusted orchestration replays actions and checks
seat-indexed incremental rewards; these reports are specific to public Tic-Tac-Toe.

Separately, **1,024 complete frozen latest-policy self-play games** use root seed
20261016 and `SHA256(boardgame-ai/audit/v1:{root}:{index}:{stream})`, first eight
bytes big-endian, streams `engine`, `seat:0`, `seat:1`. Existing collector sampling
derives private generators from these seeds. Visits are counted with repetition;
the first game of each observed outcome is traced. Seeds reserve independent RNG
streams, not unseen positions. No outcome-based rejection or teacher moves.

Action probabilities are the exact masked distribution, including zeros for
unavailable actions. Reported mass sums all valid tactical moves. Greedy uses
lowest-cell maximum ties. Uniform mass is a descriptive supplied-action reference,
not a training opponent. Undefined denominators are null. The census needs no
sampling interval; correlated self-play visits are not independent Bernoulli trials.

## Tactical evidence

| Seat / task | Census positions | Mean valid mass | Greedy success | Uniform mass |
|---|---:|---:|---:|---:|
| X immediate win | 1,498 | 53.57% | 53.81% | 43.93% |
| X required block | 396 | 35.85% | 35.35% | 33.06% |
| O immediate win | 860 | 53.09% | 53.37% | 42.21% |
| O required block | 580 | 31.84% | 31.90% | 30.17% |

Other census counts: X unavoidable 116 / quiet 413; O unavoidable 392 / quiet 265.
For example, on `·XX/O·O/···`, X can win at cell 0 but gives it only
1.61e-12 probability, preferring cell 8 with 0.999976. On
`·O·/X·X/···`, O must block cell 4 but gives it 1.11e-12 probability,
preferring cell 8 with 0.999991. These are reachable public boards, not necessarily
common self-play positions. The raw report retains every nine-action distribution.

| Seat / task | Self-play visits | Distinct boards | Mean valid mass | Actual successes |
|---|---:|---:|---:|---:|
| X immediate win | 1,026 | 10 | 99.60% | 1,023 |
| X required block | 1,017 | 2 | 99.98% | 1,017 |
| O immediate win | 0 | 0 | undefined | undefined |
| O required block | 1,021 | 7 | 99.05% | 1,011 |

There were **1,023 X wins, zero O wins, one draw** (game 309), with 7,162 decisions
on just **29 distinct acting-seat boards**. O encountered 1,016 unavoidable
positions, all the same board; X encountered none. Quiet visits: X 2,050 on six
boards, O 1,032 on three boards. High tactical success on this narrow visitation
distribution does not contradict weak census performance or the existing poor
fixed-heuristic evaluation. In particular, a late unavoidable loss is not itself
evidence of a missed block at that decision.

## Representative winning and losing learning signals

Game 0 follows cells **8, 6, 5, 2, 4, 3, 0**. X wins; the terminal global step
6 emits **(+1, -1)**. Every preceding increment is zero. X steps [0,2,4,6]
receive interval rewards [0,0,0,+1]; O steps [1,3,5] receive [0,0,-1], including
the reward on X's later move. Both terminal bootstraps are zero. No sampled O-win
trajectory exists in this budget; tests separately exercise both winner seats.
The first observed draw is retained too. Representatives illustrate the sample,
not all failure mechanisms.

Using saved gamma=.99 and lambda=.95, independently recomputed
`delta = r + gamma*next_value - value`,
`A = delta + gamma*lambda*A_next`, `target = A + value`:

| Seat | Stored values | GAE advantages | Value targets |
|---|---|---|---|
| X | .974797, .974080, .985425, 1.004503 | -.004812, .006003, .004798, -.004503 | .969985, .980083, .990223, 1.000000 |
| O | -.988194, -.993342, -.991987 | .008302, .003739, -.008013 | -.979892, -.989603, -1.000000 |

All 1,024 trajectories pass independent public replay, observation/mask,
behavior-log-probability/value, interval-reward, per-seat GAE and target checks.
Discount advances per **own decision**, not global ply. Positive advantages on
some losing-seat actions and negative advantages on some winning-seat actions
are consistent with their value baselines; outcome signs need not equal advantage
signs. O's saturated negative values yield small reward prediction errors on
the repeated losing path; this is observed, not proof of an optimizer defect.

## Disposable PPO update calculation

Only the **first 64 unselected self-play games**, all 448 decisions, form the
diagnostic batch. The saved PPO configuration is unchanged: Adam lr=.0003,
four epochs, minibatch 64, clip=.2, value coefficient=.5, entropy=.01,
gradient limit=.5, full-batch population-std advantage normalization.
Saved Adam moments and shuffle state are retained in **two disposable copies**.
No updated weights are saved or evaluated as a stronger policy.

One copy runs independent clipped-surrogate, half-MSE, legal entropy, KL and
clipping calculations, backward, gradient clipping and Adam. The other runs
the production `PPOTrainer.update`. Every minibatch loss agrees; final parameters,
Adam tensors and shuffle state agree exactly on this runtime. Sample-weighted
loss summaries and update/sample counters agree. The raw report maps every batch
index to game/seat/step and records ratios, normalized advantages, old/new log
probabilities, value predictions, policy/value terms, gradient norms and final
selected probabilities/values.

For game 0 in its first epoch, X step 0 has raw A=-.004812, normalized A=-.845545,
ratio=1, policy term=+.845545. Its target is .969985; minibatch value prediction
.977186 gives half-squared error .000025925. O step 3 has raw A=.003739,
normalized A=.498202, ratio=1.00000262 and policy term=-.498203; prediction
-.990101 versus target -.989603 gives value term 1.23733e-7. These predictions
follow earlier minibatch updates, while old probabilities and targets stay frozen.
After the whole disposable update, selected probabilities are .999961 (X step 0)
and .999502 (O step 3); neither a single probability change nor low loss is a
strength measurement.

The first minibatch has policy loss -.130522, value loss .000018154,
entropy .170126: total `policy + .5*value - .01*entropy` = -.132214.
Its pre-clip gradient norm is .593475. The full update makes 28 Adam steps:
mean total loss -.004964, policy -.003542, value .000015341, entropy .142988,
KL .001456, clip fraction .025670; max absolute parameter change .0065513.
This establishes that the update path operates consistently on the tested
on-policy data, not that it improves tactics or would escape specialization.

## Preservation and interpretation

The source bytes were identical before/after. SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| latest.pt | `5a86eced197ef40dbe1029641ed685f0d1f6acbc5ce4ce8b75416effc9fc3709` |
| training.json | `7b9142ede9d3738930307b7e006b3d57b1f5dd45da21f83fbd34c4a66242702e` |
| updates.jsonl | `245efcedd211074160eea4536507d719d9ff689db3cb6b8b8b55a866f5d49687` |

Raw evidence: `runs/phase45-task1.json` (ignored local artifact). Reproduce with
the [audit command](../../README.md#frozen-policy-tactical-and-learning-signal-audit) and
a new output filename. The report cannot overwrite an existing file or write into
the checkpoint's directory. The audit requires the
training process to be stopped; it does not lock files against concurrent writers.

**Established:** severe public tactical failures outside narrow self-play paths;
terminal rewards reach both seats; checked GAE/targets and PPO arithmetic, Adam
updates and reporting agree. No demonstrated implementation defect was identified
in these paths. Tests cover absent categories, multiple valid moves, unavoidable
threats, both winner seats, damaged terminal credit, preservation and determinism.

**Uncertain:** whether low exploration/entropy, optimization settings, positional
generalization or self-play specialization caused the weakness, and when during
training it emerged. Narrow coverage and near-deterministic openings support
these hypotheses without distinguishing them. One checkpoint, one training seed
and a bounded diagnostic cannot establish general correctness or robust strength.

Task 1 is complete. Subsequent monitoring, throughput evidence and controlled
comparisons are in [Phase 4.5 results](../../PHASE45_RESULTS.md); the current milestone is
maintained in [PROGRESS.md](../../PROGRESS.md).
The measurements above describe the preserved source policy, not later runs.
