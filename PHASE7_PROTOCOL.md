# Phase 7 frozen-search protocol

Declared before evaluation, 2026-10-15. Checkpoint: seed-45 development-selected
update 1000, SHA256 b2f8e1c4c7b53f24646e13cba92e4d8ebd6ae32f60dcffa2b4f436cc1f274a80
(same bytes as latest.pt). No training or selection in this experiment.

Fresh root seed 20261112; existing fixed-opponent-v1 + batch-v1 schedule.
Raw greedy policy, deterministic 16 and 32 simulations; 256 X-first games for
each random/heuristic × learner X/O case per condition (3,072 total).
Node cap 289, depth cap 9, c=1.5, gamma=1, prior floor 1e-8. No root noise,
no tree reuse. At most simulations+1 evaluator calls per decision. All 4,520
live X-first boards audited once per condition, accepting all immediate wins
or complete blocks. Tactical labels are measurement only.

One CPU thread; 16 empty-board warmup decisions per condition excluded from
timing. Report total evaluator calls and decision latency mean/p50/p95, both-seat
wins/draws/losses, errors/truncations, and win/block census success rates.
Time includes agent input validation and search, excludes runner copies.
No hard wall-clock cutoff: measured latency is an outcome; deterministic work
budgets are hard limits. Deterministic heuristic paths are repeated observations,
not independent trials. Match schedules are shared across conditions, but actions
can change later opponent RNG consumption. One trained checkpoint cannot establish
training-seed generalization. PPO value is clipped heuristic guidance, not minimax.
No universal strength claim; report per-seat/opponent differences and regressions.
New output directory only; checkpoint/config/log hashes must remain identical.
