# Current progress

Updated: 2026-10-15. The [roadmap](BOARDGAME_AI_PROJECT_PLAN.md) owns task lists;
[Phase 4.5 results](PHASE45_RESULTS.md) owns the measured protocols and decision.

## Status

| Area | State |
|---|---|
| Phases 1–3 | Complete: contracts, engine, simulation, baselines, CLI and replays |
| Phase 4 | Complete under its original random-baseline criterion |
| Phase 4.5 | Tasks 1–6 complete at the declared bounded scope |
| Next milestone | Public Tic-Tac-Toe simulator adapter and bounded search evaluation, before Phase 5 |

**Next small task:** implement and exhaustively test public-observation simulation,
then policy/value-guided search evaluation at 16/32 simulations. Follow the
[search-learning design](SEARCH_LEARNING_DESIGN.md). Search-target replay learning,
MuZero dynamics and hidden-game mechanics remain later steps. PPO stays a reference.

## Current evidence and concerns

- Original seed-42 update-1000 policy meets the Phase 4 random-control criterion,
  but remains tactically weak. The [frozen audit](POLICY_DIAGNOSIS.md) found about
  53% immediate-win mass and 32–36% block mass across 4,520 live boards. In 1,024
  self-play games it visited only 29 distinct acting-seat boards. Its source
  checkpoint, manifest and logs remain byte-identical.
- Terminal rewards reach both seats; checked per-seat GAE/targets and disposable
  PPO/Adam calculations agree. No signal implementation defect was demonstrated.
  Exploration, optimization and specialization remain causal hypotheses.
- Optional monitoring now keeps immutable bounded checkpoints, separate development
  selection and isolated evaluation. CPU/CUDA monitored/unmonitored/resumed tiny
  runs agree exactly on model/Adam/shuffle state. Missing log history is reported
  as unknown sample count; same-runtime/device guarantees do not imply cross-device
  or cross-version identity.
- Repeated synchronized timing supports reduced CUDA PPO cost after consolidating
  scalar reads and redundant input validation. CPU end-to-end changes are mixed;
  CPU remains faster for this workload. A faster batched forward-pass probe has
  small numerical differences and is not an adopted batched collector or speedup
  guarantee. Full timings and counts are in the results report.
- Six fresh runs completed 76,800 self-play games (559,693 decisions, 37,456 Adam
  steps), seeds 42/43/44, 200 updates/run. Only entropy .01 versus .05 changed.
  Development-selected checkpoints were frozen before 12,288 fresh final games.
  Mean paired balanced-score effect +.0044, training-seed 95% interval
  [-.4834,+.4922]: no established improvement, with seat/opponent regressions.
- More entropy sometimes increased diagnostic coverage without consistent strength.
  The bounded comparison does not establish PPO incapacity, search superiority,
  robust/optimal play or generalization to harder/hidden games. Final seed 20261029
  is consumed; subsequent tuning requires new reserved final seeds.
- Training remains exclusively latest-policy self-play with game rewards. No
  external training opponents, teachers, tactical bonuses or opponent pool.
  Search design is complete; search, replay learner and learned dynamics are not
  implemented. Preserve the failed and regressing results alongside improvements.

## Validation

Final local development checks: **643 tests passed**, including CUDA checks;
Ruff lint/format, strict mypy, dependency checks and diff whitespace checks passed.
Tests cover monitoring isolation/resume, retention/best ties, interrupted snapshot
publication/retry, both-seat signals and experiment uncertainty calculations.
The measured PPO reference comparisons matched final same-device model/Adam/
shuffle state exactly. Local checks do not assert hosted CI status.
