# Phase 6 multi-game study v1: operator results

Reviewed 2026-10-18. The operator completed all 36 training jobs and all 36 isolated
evaluations under the [frozen protocol](MULTIGAME_PROTOCOL_V1.md). Codex reviewed
saved artifacts without running training, evaluation or additional games.
[Machine-readable results and artifact hashes](multigame-v1-operator-results.json)
retain per-job/per-seat W/D/L, raw returns, curve points, costs and paired effects.
Original checkpoints, settings, manifests and reports remain under
`runs/phase6-multigame-v1/`; all 778 existing `runs/` files stayed byte-identical
through this review.

**Phase 6's declared quantitative comparison is complete, with mixed and negative
findings.** This is completion of the predeclared comparison/evidence tasks, not
proof that sharing improves play, that transfer accelerates learning, or that
models play robustly against stronger opponents. Positive-transfer and universal
noninferiority hypotheses did not pass. One recovered wall-budget job has uncertain
timing and cannot establish exact wall-cost matching; that failed cost measurement
is retained rather than replaced or treated as measured time.

## Scope, counts and verification

Three training seeds (661101/661102/661103), three development games, held-out
Breakthrough 5x5, both fixed wall allowances and decision-ceiling regimes, and
paired shared-pretrained/fresh adaptation. Every frozen curve/control evaluation
has 64 games per seat, balanced across starters; no checkpoint selection or new
final roots were introduced after seeing outcomes.

| Measure | Recorded value |
|---|---:|
| Training/evaluation jobs completed | 36 / 36 |
| Accepted self-play games / decisions | 80,698 / 1,140,839 |
| Known complete-game attempted decisions | 1,141,576 |
| Known discarded or unsaved complete-game decisions | 737 |
| Sample presentations (two PPO epochs) | 2,281,678 |
| Saved update rows with finite PPO diagnostics | 65,365 |
| Evaluation/control games / decisions | 64,512 / 1,197,185 |
| Evaluation errors / partial reports | 0 / 0 |
| Training ledger total | 8,877.907 s (includes recovered reservation) |
| Separate evaluation ledger total | 491.097 s |
| Allocated parameters per model / three independent models | 29,572 / 88,716 |

Interrupted partial-game decision costs are unknown; the attempted-decision count
is a known complete-game lower bound. The training ledger includes initialization,
restore, collection, optimizer work and publication; one 900-second entry is a
conservative reservation, not an exact measurement. These totals are neither a
CPU/GPU benchmark nor a claim that all spent work became learning samples.

Offline checks validated all run/curve/settings/prose/source-archive identities,
portable tensor/Adam/counter consistency and all 36 complete metrics histories.
Every evaluation case was checked for unique seat/index, expected seed streams,
starter, permitted score/length and balanced-score reconstruction. Zero-sum
outcomes agree with recorded own returns; Guildworks outcomes retain the trusted
engine classification (its own points alone cannot determine a winner). All
484,188 checked PPO diagnostic scalars were finite. Finite loss is not strength.

For all six adaptation pairs, checkpoint comparisons verify copied pretrained
weights, identical untouched C adapters, fresh Adam in both arms, unchanged old
game adapters through C adaptation, and nonzero recurrent parameter changes.
No C training touched the development-game pretrained models. Seven job manifests
retain source `ba082fa1...`; 29 retain `9fba8b1f...` after the reviewed publication/
recovery fix. Original source archives/manifests were not migrated. Learning
arithmetic and numeric schedules are unchanged across that recorded transition.

## Per-game playing strength and negative transfer

Scores are balanced win=1/draw=.5/loss=0 against random, averaged equally over
three training seeds. Intervals are the predeclared descriptive approximate 95%
seed intervals (`4.303*SD/sqrt(3)`), without multiple-comparison correction.
Effects use paired training seeds. Wall rows include the recovered seed's valid
playing-strength outcomes, while its wall cost remains uncertain.

| Regime | Game | Shared score | Separate score | Shared minus separate, percentage points [interval] |
|---|---|---:|---:|---|
| Wall allowance | Tic-Tac-Toe | .6471 | .6133 | +3.39 [-17.91, +24.68] |
| Wall allowance | Guildworks | .9336 | .9336 | +0.00 [-0.97, +0.97] |
| Wall allowance | Connect Four | .7813 | .7214 | +5.99 [-13.06, +25.04] |
| Decision ceiling | Tic-Tac-Toe | .7096 | .6901 | +1.95 [-6.94, +10.85] |
| Decision ceiling | Guildworks | .8125 | .8607 | **-4.82 [-7.26, -2.38]** |
| Decision ceiling | Connect Four | .6797 | .6602 | +1.95 [-0.61, +4.52] |

Guildworks has a shared-model penalty in the decision-ceiling study: its interval
is below zero. The point estimate is just inside the predeclared five-percentage-
point harm margin, so this does not establish a penalty greater than that margin;
it also fails the noninferiority criterion. Do not summarize it as no negative
transfer. Other positive point estimates do not establish a general sharing gain.

For the fixed >.55 lower-interval strength threshold, shared Guildworks and Connect
Four pass in both regimes, and shared Tic-Tac-Toe passes in the decision-ceiling
regime. Wall Tic-Tac-Toe fails that stringent threshold in both arms; separate
decision-ceiling Tic-Tac-Toe narrowly fails (lower bound .54936). Full intervals and
flags are in JSON. Random-control mean scores were .5143/.4961/.4818 for
Tic-Tac-Toe/Guildworks/Connect Four, identical across regimes' paired schedules.

Guildworks own-point returns remain separate from win scores: wall shared/separate
mean points were 14.9219/14.9089; decision-ceiling shared/separate 14.4635/14.8802.
No raw points were pooled with zero-sum returns or reinterpreted as win probability.

## Realized cost and environment allocations

| Regime / seed | Shared total seconds | Sum of separate seconds | Shared minus separate | Matching verdict |
|---|---:|---:|---:|---|
| Wall / 661101 | 900.797 | 901.422 | -.625 s | Within 5% |
| Wall / 661102 | 900.000 reserved | 901.532 | Not a measured difference | **Timing uncertain** |
| Wall / 661103 | 900.843 | 901.500 | -.657 s | Within 5% |
| Decisions / 661101 | 222.016 | 223.078 | -1.062 s | Different actual costs, within 5% |
| Decisions / 661102 | 232.266 | 239.609 | -7.343 s | Different actual costs, within 5% |
| Decisions / 661103 | 228.891 | 240.328 | -11.437 s | Different actual costs, within 5% |

The wall study has two verified realized cost matches and one documented failed
timing measurement. The seed-661102 shared checkpoint stopped at 899.86 seconds;
Windows denied the budget replacement, leaving the full reservation. Explicit
operator recovery backed up the envelope, published the same trainer at p100,
and added a receipt without training or resetting the ledger. Its exact unsaved
time is unavailable. The other completed jobs are preserved. No exact three-seed
wall-match or speedup claim is supported by that reservation.

Equal wall time does not mean equal samples. In the wall study, shared Tic-Tac-Toe
accepted 14,971–16,170 decisions per seed versus separate 29,641–31,530; shared
Guildworks accepted 66,112–71,296 versus separate 40,480–42,368; shared Connect
Four accepted 42,544–44,833 versus separate 40,275–42,081. The sampler gives equal
complete-game allocations, and game lengths differ. These allocations may help
explain outcomes but do not establish a causal explanation for them.

In the decision-ceiling study, shared Tic-Tac-Toe accepted 9,992–10,000 decisions
versus separate 9,994–9,999; both Guildworks arms accepted 9,984; shared Connect
Four accepted 9,976–9,993 versus separate 9,973–10,000. Thus the 10,000 ceilings
give near-matched complete-game samples, not identical realized counts. Every
presentation count is twice accepted decisions; raw attempted costs and discarded
work remain in JSON. The modest observed cost differences are one sequential
study's measurements, not an isolated throughput benchmark.

## Unseen Breakthrough adaptation

All twelve adaptation runs reached the .60 score target at an observed checkpoint;
there are no target-censored or zero-origin cases. Ratios below are fresh cost
divided by pretrained cost; larger than one favors pretraining. The hypothesis
required at least 1.20 in both axes for **every** seed.

| Regime / seed | First observed target: pretrained | First observed target: fresh | Seconds ratio | Decisions ratio |
|---|---|---|---:|---:|
| Wall / 661101 | p25: 75.015 s / 7,662 decisions | p25: 75.016 s / 7,511 | 1.000 | .980 |
| Wall / 661102 | p25: 75.094 s / 7,988 | p25: 75.047 s / 7,828 | .999 | .980 |
| Wall / 661103 | p25: 75.094 s / 7,620 | p25: 75.062 s / 7,898 | 1.000 | 1.036 |
| Decisions / 661101 | p50: 20.438 s / 2,503 | p50: 21.000 s / 2,504 | 1.027 | 1.000 |
| Decisions / 661102 | p50: 20.921 s / 2,513 | p50: 20.406 s / 2,506 | .975 | .997 |
| Decisions / 661103 | p25: 10.203 s / 1,258 | p50: 20.500 s / 2,501 | 2.009 | 1.988 |

**Consistent adaptation speedup was not demonstrated.** Only one of six paired
regime/seed cases meets both ratios. Its early pretrained score was .609375 versus
fresh .59375 at the corresponding point, close to the threshold; coarse snapshots
and sampling uncertainty make the apparent doubling fragile. Wall targets were
already reached at the first nonzero checkpoint; their true crossing times are
somewhere before that observation. Near-equal observed ratios do not prove equal
underlying learning speed or exclude an unresolved earlier advantage.

Final Breakthrough mean scores were .9792 pretrained versus .9661 fresh in wall
jobs, and .9609 versus .9583 under decision ceilings. Both learn to beat random;
these scores do not prove robust play, superiority over fresh training, or
end-to-end compute savings. Pretraining adds approximately 900 seconds/shared
wall model or 222–232 seconds/shared decision-ceiling model and is excluded from
the adaptation-only axes. Full unselected curves are retained in JSON and raw cases.

## Forgetting after C-only adaptation

Original development-game adapters stay fixed; the recurrent backbone changes.
Effects are balanced-score percentage points, p100 minus p0 on the original games.

| Regime | Game | Mean change [approximate 95% seed interval] | Five-point retention noninferiority |
|---|---|---|---|
| Wall | Tic-Tac-Toe | -0.26 [-1.38, +0.86] | Supported |
| Wall | Guildworks | -0.65 [-2.13, +0.83] | Supported |
| Wall | Connect Four | -2.99 [-11.80, +5.81] | **Not established** |
| Decisions | Tic-Tac-Toe | -1.17 [-5.05, +2.71] | **Not established** |
| Decisions | Guildworks | -0.65 [-4.06, +2.76] | Supported |
| Decisions | Connect Four | +0.39 [-2.18, +2.96] | Supported |

No old game has an interval wholly below the -5-point harm margin; catastrophic
forgetting was not demonstrated by that predeclared test. This does not establish
its absence: wall Connect Four permits a substantial decline, and decision-ceiling
Tic-Tac-Toe narrowly fails retention noninferiority. Three seeds and random-only
opponents limit sensitivity.

## Acceptance disposition and preserved limitations

The deliverable now includes trained independent/shared agents, quantitative
per-game strength and costs in both budget regimes, full held-out adaptation
curves/first-observed target costs, negative-transfer and retention comparisons,
and explicit failure/uncertainty reports. This satisfies the roadmap's **comparison**
criterion at the frozen v1 scope. It does not turn the study's failed speedup,
strength or noninferiority hypotheses into successes. The protocol predeclared
that honest negative comparisons and failed cost evidence must be reported.

There are no missing scheduled jobs, partial evaluations or unreviewed acceptance
measurements. Exact cost recovery for one wall job remains a known limitation;
positive transfer, universal noninferiority and stronger-opponent performance remain
unproven research questions, not implied achievements. A follow-up intended to
establish them needs a new declaration, fresh final roots and fresh output; it
must not overwrite this study or reuse its evaluated seeds for new final claims.
No follow-up campaign or later phase was started by Codex.
