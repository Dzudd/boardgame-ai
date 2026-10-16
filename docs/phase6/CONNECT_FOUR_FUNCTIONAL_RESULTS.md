# Connect Four functional protocol v1: recorded results

Reviewed 2026-10-16. The user executed the CPU run in their own CMD console;
Codex inspected the saved artifacts without rerunning training or evaluation.
The four-update schedule completed through a two-update pause/resume, followed by
the fixed initial/final/random-control evaluation. This demonstrates bounded
operation, not reliable strength improvement or shared-model learning.

The [protocol](../PHASE6_CONNECT_FOUR_PROTOCOL.md),
[numeric settings](connect-four-functional-v1.json) and
[machine-readable evidence](connect-four-functional-v1-results.json) define the
schedule and record raw outcomes, metrics, saved provenance and artifact hashes.
The evidence includes exact saved protocol/settings text so later prose edits do
not replace the run's declaration. Original files remain under
`runs/connect-four-functional-retry1/`; checkpoints were not copied or overwritten.

## Completed work and costs

| Measure | Recorded value |
|---|---:|
| Device / Torch threads | CPU / 1 |
| Model parameters | 20,872 |
| Updates / self-play games | 4 / 32 |
| Training decisions | 664 |
| Optimizer steps / sample presentations | 46 / 1,328 |
| Replay reuse | 0 |
| Self-play seat-0 wins / seat-1 wins / draws | 18 / 14 / 0 |
| Evaluation/control games | 192 |
| Evaluation decisions | 4,140 |
| Total completed games / decisions | 224 / 4,804 |
| Cumulative training ledger time | 3.921 seconds |
| Cumulative evaluation ledger time | 2.203 seconds |
| Training / evaluation cap | 120 / 120 seconds |
| Reported overrun / discarded complete games | 0 / 0 |

The training ledger includes initialization, collection, optimization, validation
and saves across both processes. Per-update recorded collection totals 0.328 s,
optimization 0.532 s and checkpoint publication 0.030 s. These do not include all
startup/finalization work and must not be substituted for total cost. Evaluation
reports 2.188 s before report publication; its settled ledger records 2.203 s.
Small/zero operation timings reflect clock granularity, not free operations.
These are one-run observations, not an isolated benchmark or CPU/GPU comparison.
The recorded runtime is Windows 11, Python 3.12.14, Torch 2.14.1+cu130 on an AMD64
Family 25 Model 33 processor; the CUDA-enabled wheel does not imply GPU execution.

| Update | Decisions | Optimizer steps | Unique acting positions within update |
|---|---:|---:|---:|
| 1 | 169 | 12 | 160 |
| 2 | 159 | 10 | 148 |
| 3 | 169 | 12 | 159 |
| 4 | 167 | 12 | 157 |

The update-two boundary contained 328 decisions and 22 optimizer steps; resume
continued to 664 and 46. Column frequencies for indices 0–6 across all updates
were `[81, 72, 102, 126, 94, 101, 88]`. Unique-position counts cannot be summed
into whole-run coverage because positions can recur between updates. Complete
loss/entropy/KL/clipping metrics and self-play outcomes are in the JSON evidence.
All logged PPO scalars are finite. Initial-to-final parameter change has L2 norm
0.434206054, maximum absolute change 0.0107021835, and 15,240 changed scalar
parameters. Nonzero optimization is demonstrated; this does not certify strength.

## Fixed sampled evaluation against random

Seat 0 is X and starts. Each row contains 32 games, with score win=1/draw=.5/loss=0.

| Policy/control | Seat | Wins | Draws | Losses | Score | Mean decisions/game |
|---|---:|---:|---:|---:|---:|---:|
| Initial policy | 0 | 13 | 0 | 19 | .40625 | 23.03125 |
| Initial policy | 1 | 19 | 0 | 13 | .59375 | 20.03125 |
| Final policy | 0 | 23 | 0 | 9 | .71875 | 20.90625 |
| Final policy | 1 | 19 | 0 | 13 | .59375 | 21.34375 |
| Random control | 0 | 20 | 0 | 12 | .62500 | 22.56250 |
| Random control | 1 | 12 | 0 | 20 | .37500 | 21.50000 |

There were zero errors and truncations in every case. Balanced scores weight seats
equally: initial .50000, final .65625, control .50000. Final-minus-initial improvement
appears in seat 0 in this schedule; seat 1's aggregate score is unchanged. Matching
aggregate scores does not mean its individual case outcomes stayed unchanged.

## Paired effects and limits

Effects below are score differences. Intervals use the predeclared approximate
paired normal method: mean +/- 1.96 standard errors, with cases paired by seat and
index. They describe game-level uncertainty only, not training-seed uncertainty.

| Contrast | Seat 0 mean [interval] | Seat 1 mean [interval] | Balanced mean [interval] |
|---|---|---|---|
| Final minus initial | .31250 [.14933, .47567] | .00000 [-.15243, .15243] | .15625 [.04460, .26790] |
| Final minus random control | .09375 [-.14500, .33250] | .21875 [.00800, .42950] | .15625 [-.00298, .31548] |

The balanced final-minus-control interval includes zero. A single training seed,
32 self-play games and 32 evaluation cases/seat do not establish robust learning
or a pass/fail strength gate. Initial/final use the same policy/engine/opponent
seed schedule. The control changes only its designated policy stream; actual
opponent actions may differ as legal choices diverge. These functional roots are
consumed; a later strength study needs at least three predeclared training seeds
and fresh final evaluation roots. No stronger opponent or search-assisted/greedy
evaluation was substituted, and no checkpoint was selected by outcome.

## Provenance, failed setup and verification

The recorded code revision is `0af7052e3d8eb47c2291196c5ae378827a742118` with a dirty
working tree. The source digest identifies the implementation that was uncommitted
during execution; the recorded revision alone does not reproduce it. Source/runtime remained compatible through
resume and evaluation. Saved identities are:

| Artifact | SHA-256 |
|---|---|
| Numeric settings | `90b3dbfe607307c32fe04f5a1183277fd2783374abff3336ff9bc750dbd05bb6` |
| Saved protocol prose | `8445239f730575851bb0a7fdb4cfe6b5fcd08195d22201c28a529595a1cfbc80` |
| Source tree | `d3996e07a18bd6870fa75ecfe825861361a8f8f0d7f5524ed9f64c5fef03047f` |
| Initial checkpoint | `dc74f12491f6641515dfa1cb86fd0e26c0686b99f3e0eb29fe8af05e48eb9745` |
| Final checkpoint | `a6d8fde0036c3934978f5886dc63588395e96eb9d683effc6bcfdb7ea3d5a4c8` |

Before this completed attempt, `runs/connect-four-functional/` failed at Git
provenance collection. It contains an initial model and zero games, decisions,
optimizer steps or saved run boundaries. Its 120-second reservation is retained
conservatively; actual elapsed startup time was not recorded and is not included
in the successful attempt's measured ledger. The user corrected CMD's PATH and
started the clearly named retry directory. This was a new attempt after failed
initialization, not a successful resume of the failed directory. The subsequent
two-plus-two continuation occurred inside the successful retry directory.

Offline review validated restricted trainer/run loading, saved hashes, final
snapshot identity, update/decision/optimizer/presentation counts, finite/nonzero
parameter changes, every evaluation case and seed, W/D/L/length summaries, and
paired effects recomputed from raw outcomes. No games or optimization were rerun.
No missing metrics rows were found or invented. Existing exact CPU resume, GAE,
reward ownership and isolation tests remain supporting functional evidence; no
uninterrupted full-protocol comparison was executed. Human checkpoint CLI is
covered by disposable tests, not a recorded human match in this experiment.

The completed operator command sequence was:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.connect_four_run runs/connect-four-functional-retry1 --settings docs/phase6/connect-four-functional-v1.json --protocol docs/PHASE6_CONNECT_FOUR_PROTOCOL.md --stop-after 2
.\.venv\Scripts\python.exe -m boardgame_ai.training.connect_four_run runs/connect-four-functional-retry1 --settings docs/phase6/connect-four-functional-v1.json --protocol docs/PHASE6_CONNECT_FOUR_PROTOCOL.md --resume
.\.venv\Scripts\python.exe -m boardgame_ai.training.connect_four_run runs/connect-four-functional-retry1 --settings docs/phase6/connect-four-functional-v1.json --protocol docs/PHASE6_CONNECT_FOUR_PROTOCOL.md --evaluate
```

These commands record history, not instructions to rerun/overwrite the completed
comparison. The [runbook](CONNECT_FOUR_CONSOLE_RUNBOOK.md) describes general use.

## Remaining work

The bounded independent functional run is complete. Learning-quality experiments,
GPU run-layer support and measured throughput, larger fixed budgets, shared action
scoring/backbone/adapters/sampling, matched-compute comparisons, unseen-game transfer
and forgetting remain. The Phase 6 multi-game acceptance criterion is not met.
