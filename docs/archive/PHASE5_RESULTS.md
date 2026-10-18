# Phase 5 — Guildworks recurrent learning

This report preserves the bounded functional-validation protocol, measured
outcomes and evidence limits for Guildworks recurrent learning.

## Bounded completion protocol

This is a functional validation of hidden-information recurrent self-play, not
a strength experiment or a requirement to beat random. Rules remain the original
four-round Guildworks game; there are no interrupts, extra turns or skipped
craft decisions. Empty hands/decks still lead to a craft decision with pass.
Round boundaries can give the same seat successive turns under the published
alternating-start rule. The existing engine schedule and rejection tests cover
these cases; unsupported action kinds reject rather than introduce new rules.

Predeclared before the run: seed 20261016, 16 updates of 16 complete self-play
games (256 games, 8,192 atomic decisions). Both seats use one frozen latest
policy per update. GRU hidden size 32, eight masked actions, raw rule rewards,
gamma 1, lambda .95, Adam learning rate .0003, two epochs per update, minibatches
of eight complete seat episodes, clipping .2, value coefficient .5, entropy
coefficient .01, gradient norm bound .5. This is 128 optimizer steps and 16,384
decision presentations across epochs. No tuning, teachers, opponent pool,
search or stored replay. CPU, one Torch thread; CUDA has separate update and
portable-checkpoint tests, without speed or cross-device identity claims.

After training, freeze the initial and final policies and evaluate each against
RandomAgent for 128 games per seat using evaluation root seed 20261030 in the
Guildworks evaluation namespace. Initial/final use matched engine/opponent seeds;
changed actions can change game paths. Evaluation trajectories never enter PPO.
This one training seed and small evaluation are insufficient for learning-quality,
balance, robustness or generalization claims. No acceptance threshold on win rate
was set. Compare outcomes and public point scores separately: rule rewards train
own point return, not zero-sum win probability.

Command (output must be absent or empty):

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai.training.guildworks_run runs/guildworks-phase5 --device cpu --threads 1 --updates 16 --games-per-update 16 --seed 20261016 --eval-games-per-seat 128 --eval-seed 20261030
```

Raw manifest, update-boundary checkpoint, metrics and evaluation summaries stay
under ignored `runs/guildworks-phase5`. They contain reconstructive training
metadata and are trusted artifacts, not player exports.


## Recorded results

The declared run completed 16 updates, 256 self-play games and 8,192 atomic
decisions, with 128 optimizer steps and 16,384 decision presentations over
epochs. All eight model tensors changed; all logged losses remained finite.
Runtime: Python 3.12.14, PyTorch 2.14.1+cu130, CPU optimization, one Torch thread.

| Frozen policy | Seat | Wins | Draws | Losses | Own mean points |
|---|---|---|---|---|---|
| Initial | 0 | 57 | 18 | 53 | 12.671875 |
| Initial | 1 | 58 | 16 | 54 | 12.796875 |
| Trained | 0 | 64 | 19 | 45 | 13.28125 |
| Trained | 1 | 65 | 13 | 50 | 13.4609375 |

All 512 evaluation games completed in exactly 32 decisions, with zero errors
or truncations. Balanced outcome score (win=1, draw=.5) was .515625 initially
and .56640625 after training. This is one small functional run with no strength
threshold; it does not establish reliable improvement or a benefit from memory.
No memory ablation was performed. Rewards optimize own points, not zero-sum wins.
Evaluation root 20261030 is consumed; reserve fresh final seeds if these results
inform later tuning. No additional training was performed during report verification.

Checkpoint SHA-256 (`runs/guildworks-phase5/latest.pt`):
`ec27d0033a2d3d0f5360d43d269c8aaf5a185bd948a7237ca1d7df4b6411e237`.
The stored evaluation report and checkpoint hash were rechecked after the run.

## Verification and limits

Final regression results are recorded in [PROGRESS.md](2026-10-18-PROGRESS.md#validation).
Tests cover shared engine conformance, same-player phase/round transitions,
termination, rule rewards, empty/partial decks, forced pass, blocked placements,
atomic rejection, clone independence, private/global RNG and hidden-world
observation/action equivalence. Recurrent checks extend equivalence to tokens,
masks, memory and predictions; training checks cover full-history gradients,
reward ownership, PPO arithmetic, CPU exact resume, corruption rejection and
CUDA update/checkpoint portability.

No game balance, robust strength, cross-device numerical identity or speed claim
is made. Human CLI opponents remain random only. Hidden-information search,
public replay export, MuZero and multi-game learning remain future work.
Phase 6 requires a bounded evaluation protocol and game choices before implementing
a multi-game learner; its status and next milestone are maintained in
[PROGRESS.md](../../PROGRESS.md) and the [project plan](../../BOARDGAME_AI_PROJECT_PLAN.md).
