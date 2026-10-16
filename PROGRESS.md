# Current progress

Updated: 2026-10-16. The [roadmap](BOARDGAME_AI_PROJECT_PLAN.md) owns task lists;
[Phase 4.5 results](PHASE45_RESULTS.md), [Phase 5 results](PHASE5_RESULTS.md)
and [Phase 7 results](PHASE7_RESULTS.md)
contain measured protocols, evidence and limitations.

## Status

| Area | State |
|---|---|
| Phases 1–3 | Complete: contracts, engine, simulation, baselines, CLI and replays |
| Phase 4 | Complete under its original improvement-over-random criterion |
| Phase 4.5 | Tasks 1–6 complete at the declared bounded scope |
| Phase 7 brought-forward milestone | Public simulator, bounded 16/32 search evaluation and heuristic-O diagnosis complete |
| Phase 5 | Complete at declared Guildworks scope: engine, CLI, recurrent memory/PPO and bounded functional run |

**Next small task:** define Phase 6's bounded multi-game evaluation protocol and
game choices before implementing a multi-game learner. Keep hidden-information
search, MuZero and replay learning for separately specified milestones.

## Connect Four addition

Implemented the shared engine, immutable public board/history, gravity and all
four win directions, draw/reward handling, rendering, outcome adapter and random
runner matches. Human CLI uses the agreed board with column labels 1–7 and full
column retries. Training, search and replay integration remain separate work.

## Current evidence and concerns

- Guildworks human CLI play against RandomAgent uses only filtered views and
  numbered supplied actions. Both starting seats complete reproducibly; retries,
  forced pass and cancellation in either phase are tested. AI output contains
  public chosen actions only; final presentation retains the human's view.
  There is no Guildworks heuristic opponent or replay export.

- Guildworks completes exactly 32 atomic decisions across four rounds. Placement
  and craft/pass are consecutive decisions by the same actor. Private seeded
  setup/draws and lottery rolls resolve before returning; public scores determine
  wins/ties and incremental rewards equal points earned. Rendering and outcome
  adaptation work through existing interfaces without changing Tic-Tac-Toe.
- Scripted and reproducible random matches complete through the shared runner;
  tests cover conformance, transitions, public placements, empty/partial decks,
  crafting costs, rewards, rejection/RNG invariants, clone independence and
  active/inactive/terminal hidden-world equivalence. This establishes mechanics
  and boundaries, not game balance, agent strength or learnability. Public deck
  composition permits deductions; terminal hands are not explicitly revealed.

- Guildworks now has visible-history GRU memory, eight supplied-action masks,
  complete-episode recurrent PPO, portable checkpoints and exact CPU resume.
  The predeclared functional run completed 16 updates / 256 self-play games /
  8,192 decisions. Initial/final balanced random-opponent scores were .515625
  and .56640625 across 256 evaluation games each, with no errors/truncations.
  This single small run establishes operation, not reliable improvement or a
  measured benefit from memory. Evaluation seed 20261030 is consumed. Raw run
  artifacts remain preserved under runs/guildworks-phase5; see Phase 5 results.

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

Latest full local suite: **837 tests passed**, including Guildworks engine, CLI,
recurrent and training checks, 35 Connect Four rule/CLI tests and existing
CPU/CUDA regressions. Ruff lint/format,
strict mypy, dependency and whitespace checks passed. The sandbox attempt hit
13 existing monitoring/evaluation permission failures during Phase 5 cleanup;
the approved unrestricted full reruns passed, including the Connect Four addition. Public simulator
comparison covers 5,478 boards and 16,167 legal transitions per starting seat.
Tests cover search budgets/visits, backup perspective, terminal zero bootstrap,
private exploration, RNG preservation and trace equivalence/corruption detection.
Evaluation replay reproduced match/tactical results. Latency measurements are
single-run observations, not an isolated benchmark. Hosted CI is not asserted.
