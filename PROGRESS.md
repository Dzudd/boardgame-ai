# Current progress

Updated: 2026-10-16. The [roadmap](BOARDGAME_AI_PROJECT_PLAN.md) owns task lists;
[Phase 4.5 results](PHASE45_RESULTS.md) and [Phase 7 results](PHASE7_RESULTS.md)
contain measured protocols, evidence and limitations.

## Status

| Area | State |
|---|---|
| Phases 1–3 | Complete: contracts, engine, simulation, baselines, CLI and replays |
| Phase 4 | Complete under its original improvement-over-random criterion |
| Phase 4.5 | Tasks 1–6 complete at the declared bounded scope |
| Phase 7 brought-forward milestone | Public simulator, bounded 16/32 search evaluation and heuristic-O diagnosis complete |
| Phase 5 | Next: minimal playable card/worker-placement game and visibility tests |

**Next small task:** define and implement the minimal Phase 5 game through the
existing engine/agent interfaces, with explicit decision phases, consecutive
same-player actions, hidden hands, public worker placements and engine-resolved
card draws. Establish rules, information boundaries and scripted/random complete
games before observation encoders or recurrent PPO. Tic-Tac-Toe need not be solved
before progressing. Prior-concentration ablation is optional deferred research;
search-target learning, replay and MuZero dynamics remain later work.

## Current evidence and concerns

- PPO self-play, per-seat rewards/GAE, checkpoints, CPU/CUDA resume and isolated
  monitoring are implemented and tested. Signal/update checks found no demonstrated
  learning-signal defect. The original policy beats random under its declared
  criterion but is not robust; self-play dominance is not evidence of strength.
- The six-run entropy comparison did not establish improvement: paired mean
  balanced-score effect +.0044, training-seed 95% interval [-.4834,+.4922].
  Profiling supports limited same-device optimization claims; batched collection
  was not adopted. Detailed counts and timings remain in Phase 4.5 results.
- Saved seed-45 sampled-policy reports confirm heuristic X 0/997/3 and O 0/0/1000
  wins/draws/losses. The selected update-1000 checkpoint matches latest.pt.
  Frozen greedy raw/16/32 evaluation gives random-X losses 19/9/9 and random-O
  losses 96/50/41 out of 256 each; heuristic-X draws every game and heuristic-O
  loses every game throughout. Census tactics improve, with more calls/latency.
  This supports limited measured gains, not robust strength or heuristic gains.
- Public heuristic-O traces reproduce X4/O3/X0/O2/X8. The required O8 block gets
  one visit; X's immediate winning reply after O2 gets none. Concentrated priors
  and clipped leaf guidance interact. Blocking is not proven to save the game.
  Node/depth caps are not hit; independent visit/reward/vector backups agree.
- Search receives observation-reconstructed public rules only. Hidden/chance
  simulation and same-player game simulation remain unsupported. New Phase 5
  mechanics must not be routed through this Tic-Tac-Toe-only adapter.
- Training remains latest-policy self-play with rule rewards. Baseline evaluation
  and tactical labels never enter learning. Existing checkpoints/configurations/
  logs remain preserved. Final seeds 20261029 and 20261112 are consumed; reserve
  fresh final seeds if evaluation informs later tuning.

## Validation

Latest full local suite: **706 tests passed**, including CUDA checks. Ruff
lint/format, strict mypy, dependency and whitespace checks passed. Public simulator
comparison covers 5,478 boards and 16,167 legal transitions per starting seat.
Tests cover search budgets/visits, backup perspective, terminal zero bootstrap,
private exploration, RNG preservation and trace equivalence/corruption detection.
Evaluation replay reproduced match/tactical results. Latency measurements are
single-run observations, not an isolated benchmark. Hosted CI is not asserted.
