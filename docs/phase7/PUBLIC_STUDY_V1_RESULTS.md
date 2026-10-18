# Phase 7 public learner comparison v1: results

Reviewed 2026-10-18 against the [frozen protocol](PUBLIC_STUDY_V1.md) and byte-pinned
[settings](public-study-v1.json). All six training jobs and three paired final
evaluations completed. This declared comparison is complete with a failed
improvement hypothesis; **Phase 7 acceptance remains unmet**.

The raw search-target policy did not establish an advantage over independent PPO.
Its mean score was **.32764 versus .34440**, a paired difference of **−1.68
percentage points [−3.81, +0.46]**. The interval includes zero, so this does not
establish that either learner is superior. All three point differences favor PPO.
The separately required heuristic-X noninferiority guard also fails. Greater
search training cost did not buy demonstrated strength in this frozen recipe.

[Machine-readable results](public-study-v1-results.json) preserve every seed's
W/D/L, latency, cost and replay-age summaries, within-case standard errors,
predeclared paired effects, checkpoint identities and all 6,411 study artifact
hashes. Original artifacts stay under `runs/phase7-public-study-v1/`.

## Frozen primary outcome and regression guards

Scores are win=1/draw=.5/loss=0, equally weighting random/heuristic × learner X/O.
Each case/seed/arm has 256 games. Intervals use the predeclared descriptive
`mean ± 4.303*SD/sqrt(3)` across three paired training seeds, not a game-level
interval or a multiple-comparison-adjusted claim.

| Training root | PPO score | Search-target score | Search minus PPO, points |
|---|---:|---:|---:|
| 771101 | .34473 | .31836 | −2.64 |
| 771102 | .34619 | .33203 | −1.42 |
| 771103 | .34229 | .33252 | −0.98 |
| Mean | **.34440** | **.32764** | **−1.68 [−3.81, +0.46]** |

The primary criterion required a strictly positive lower endpoint; it fails.
Heuristic-X effect is +1.50 points [−7.13, +10.13], failing the lower-endpoint
>= −5-point guard. This failure reflects unresolved uncertainty, not demonstrated
harm greater than five points. Heuristic-O effect is +1.17 points [−0.11, +2.46],
which passes that guard but does not demonstrate a positive effect. Both guards
and the primary criterion were required; the combined criterion is false.

## Both-seat strength and raw-policy latency

W/D/L totals below pool the three seeds, 768 games per cell. Scores retain equal
case/seed weighting. Both learners are particularly weak against the heuristic;
neither wins a heuristic-O game. Small positive heuristic point estimates do not
establish robust play or counteract the failed primary criterion.

| Opponent / learner seat | PPO W/D/L | PPO score | Search-target W/D/L | Search-target score |
|---|---|---:|---|---:|
| random-v1 / X | 546 / 71 / 151 | .75716 | 514 / 78 / 176 | .72005 |
| random-v1 / O | 343 / 91 / 334 | .50586 | 292 / 106 / 370 | .44922 |
| heuristic-v1 / X | 16 / 110 / 642 | .09245 | 22 / 121 / 625 | .10742 |
| heuristic-v1 / O | 0 / 34 / 734 | .02214 | 0 / 52 / 716 | .03385 |

All conditions use the raw sampled policy. Latency is cold CPU decision-call time,
excluding checkpoint/model setup, runner copies and disk publication, with no
warmup subtraction. Pooled quantiles below are recomputed from every original
decision sample, rather than averaging quantiles. They are descriptive sequential
observations on this runtime, not a repeated isolated hardware benchmark.

| Opponent / learner seat | PPO ms mean / p50 / p95 | Search-target ms mean / p50 / p95 |
|---|---|---|
| random-v1 / X | .212 / .206 / .232 | .211 / .206 / .233 |
| random-v1 / O | .212 / .207 / .233 | .213 / .207 / .242 |
| heuristic-v1 / X | .213 / .208 / .237 | .213 / .208 / .237 |
| heuristic-v1 / O | .216 / .213 / .236 | .216 / .212 / .236 |

The dedicated learner's training uses search, but these final matches do not add
inference-time search. Historical greedy frozen-PPO 16/32-simulation results in
[Phase 7 results](../../PHASE7_RESULTS.md) retain their different scope.

## Training and evaluation cost

These are **equal-game**, not matched-wall or equal-presentation, allocations.
Every training attempt completed normally at its saved final boundary; timing is
certain and accepted costs have complete histories. There was no unsaved failed
learning attempt to omit. Training wall scope includes child startup, initialization,
collection, optimization, checkpoint/report publication and supervision, excluding
parent preflight/archive setup/final reporting.

| Across three seeds | PPO | Search-target |
|---|---:|---:|
| Self-play games | 768 | 768 |
| Environment decisions | 5,653 | 4,918 |
| Search simulations | 0 | 157,376 |
| Collection evaluator calls | 5,653 | 112,342 |
| Adam steps | 768 | 768 |
| Sample presentations | 22,612 | 196,608 |
| Presentations / accepted decision | 4.00 | 39.98 |
| Measured child-work seconds | 14.953 | 109.359 |

Search-target used **7.31×** measured wall time and **8.69×** sample presentations
in this bounded study. Its 157,376 simulations and replay work are additional
costs, not environment samples. Equal Adam-step counts do not imply equal work.
PPO's four-epoch reuse stays separate from historical search replay. Per-update
replay-age statistics and self-play outcome coverage remain in machine-readable
results; they are not a complete board-diversity census.

All 6,144 final games completed without errors or truncation, executing 42,307
environment decisions and 20,674 learner evaluator calls. Separate evaluation
wall time was 15.235 / 14.844 / 15.125 seconds, **45.204 seconds total**. These
costs never enter training. Combined measured training/evaluation child work was
169.516 seconds; calibration was separately allocated and measured.

## Validation, limits and disposition

The offline review validated frozen settings and source/archive/runtime identities,
complete native metric histories and optimizer/cost counters, initial/final
snapshot identities, cumulative receipts and budgets, checkpoint hashes used by
evaluation, and complete unique paired schedules. An additional raw-record audit
checked every per-game outcome/summary, seat/index/seed, length, one raw-policy call
per learner decision, finite latency sample and cached aggregate. It recomputed
pooled latency quantiles and case effects from original records. All **7,252**
saved run files retain their pre-review hashes, including the **1,090** files that
predated final evaluation. No extra game, optimizer call, repair or campaign was
executed during review. Source and frozen settings remain unchanged; only new
derived results and documentation were published. Most recent code validation is
the prior 1,248-pass/one-skip suite, not a newly executed test suite for this review.

This is a small, three-seed comparison of complete frozen recipes. Native
initialization streams, objective, sample weighting and replay/presentation costs
differ. It does not isolate the causal contribution of search, replay reuse or
value targets. The 256-game allocation may be insufficient for some recipes, but
that is a hypothesis, not permission to extend the consumed budget. Positive
longer-run effects, broad planning benefits and superiority over PPO remain
unproven; failure here does not show that all search methods fail.

The primary improvement hypothesis and heuristic-X guard did not pass. Retain
independent PPO as the baseline, preserve the negative/inconclusive evidence and
do not promote the search learner on these results. Final roots **20261208,
20261209 and 20261210 are consumed**. Any outcome-informed change or follow-up
needs a new declaration, fresh roots and fresh outputs. There is no automatic
rerun, seed dropping, cap extension or post-hoc compute-matched relabelling.

## Offline replay-fit diagnosis

A subsequent [read-only diagnostic](public-study-v1-replay-fit.json) forwarded only
the stored permitted public observations/legal masks through loaded initial/final
search-model copies. It started no game and made no optimizer call, using no
baseline/solver/tactical labels. Metrics use uniform-game then uniform-position
weighting, matching the replay sampler's marginal weighting. Visit cross entropy
is computed against the original normalized saved visits; value fit is half-MSE
against each sample's saved own-seat terminal outcome. This is **in-sample fit**,
not held-out strength or an independent training-seed experiment.

| Seed | Saved public board/actor pairs | Visit CE, initial → final | Half outcome MSE, initial → final | Target-argmax agreement, initial → final |
|---|---:|---|---|---|
| 771101 | 548 | 1.764 → 1.569 | .529 → .332 | .274 → .622 |
| 771102 | 696 | 1.771 → 1.648 | .433 → .377 | .167 → .490 |
| 771103 | 570 | 1.743 → 1.555 | .430 → .345 | .385 → .614 |

All 256 self-play games and 64 collection policy versions are retained per seed.
The counts describe visited replay states, not all reachable boards. Final models
fit their own replay targets better than initialization on these metrics; the
learning pipeline is not simply unchanged. That fit did **not** establish external
improvement. The data cannot distinguish whether replay exposure, target quality,
sampling/optimization choices or limited self-play coverage caused the result.
The roughly 40 presentations/accepted decision, versus PPO's four epoch exposures,
is a concrete hypothesis for a separately controlled reuse ablation, not a proven
cause or permission to alter this completed study. A follow-up needs its own
predeclared caps and fresh seeds/roots before any outcomes; it may not turn
diagnostic labels into teachers or rewards.

Hidden-information beliefs/search, chance/same-player game semantics, learned
representation/dynamics/prediction and broader planning-cost comparisons remain
unfinished roadmap tasks. Generic interfaces, vector backups and this completed
negative public comparison do not establish those capabilities or Phase 7 acceptance.
