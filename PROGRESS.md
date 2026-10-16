# Current progress

Updated: 2026-10-16. This document summarizes current status, verified evidence,
limitations and the next milestone. The [roadmap](BOARDGAME_AI_PROJECT_PLAN.md)
owns phase tasks and acceptance criteria; linked result documents own experiment
protocols, measurements and detailed limitations.

## Status

| Area | State |
|---|---|
| Phases 1–3 | Complete: contracts, engine, simulation, baselines, CLI and replays |
| Phase 4 | Complete under its original improvement-over-random criterion |
| Phase 4.5 | Tasks 1–6 complete at the declared bounded scope |
| Phase 5 | Complete at declared Guildworks scope: engine, CLI, recurrent memory/PPO and bounded functional run |
| Phase 6 | Pending; multi-game protocol and game choices are the next milestone |
| Phase 7 | Partial: public simulator, bounded 16/32 search evaluation and heuristic-O diagnosis complete; broader research tasks remain |
| Phase 8 | Partial: baseline human CLI play and Tic-Tac-Toe public replays available; trained-agent human play and other product tasks remain |

**Next milestone:** define Phase 6's bounded multi-game evaluation protocol and
game choices before implementing a multi-game learner. Hidden-information search,
MuZero and search-target replay learning need separately specified milestones.

## Verified evidence

- **Guildworks:** complete scripted/random games, filtered human CLI, hidden-world
  equivalence, recurrent memory/PPO and exact CPU resume are implemented and tested.
  The functional run completed 16 updates / 256 self-play games / 8,192 decisions.
  Initial/final balanced random-opponent scores were .515625 and .56640625 across
  256 evaluation games each, with no errors or truncations. This single small run
  demonstrates operation, not reliable improvement or a benefit from memory.
  [Phase 5 results](PHASE5_RESULTS.md) preserve the protocol, artifacts and limits.
- **Tic-Tac-Toe PPO:** the original 64,000-game run met its improvement-over-random
  criterion but remained weak against the heuristic. Signal/update checks found
  no demonstrated learning-signal defect; the six-run entropy comparison did not
  establish improvement. Repeated profiling supports limited same-device
  optimization claims; batched collection was not adopted.
  See [Phase 4 results](PHASE4_RESULTS.md) and [Phase 4.5 results](PHASE45_RESULTS.md).
- **Public search:** frozen greedy raw/16/32 evaluation reduced random-X losses
  from 19 to 9/9 and random-O losses from 96 to 50/41 out of 256 each, with greater
  cost. Heuristic-X remained all draws and heuristic-O all losses. Public traces
  reproduce branch starvation associated with concentrated priors and clipped
  leaf guidance; blocking is not proven to save the game.
  [Phase 7 results](PHASE7_RESULTS.md) contain the census, diagnosis and cost data.
- **Connect Four:** the shared-interface engine, public history, gravity, all four
  win directions, draw/reward handling, rendering, random matches and human CLI
  are implemented, with 35 recorded rule/CLI tests. This adds a playable game;
  it does not establish multi-game learning. Usage is in [README](README.md#connect-four).

## Current limitations

- Guildworks mechanics and visibility tests do not establish balance or robust
  strength. Its human CLI opponent is random only; heuristic play and public
  replay export remain unavailable.
- Search supports public deterministic Tic-Tac-Toe only. Hidden/chance and
  same-player game simulation are unsupported; generic same-actor backup
  arithmetic tests do not establish game support. Search-target learning, MuZero
  and multi-game learning are not implemented.
- Connect Four training, search and replay integration remain future work.
- Evaluation baselines and tactical labels never enter latest-policy self-play
  training. Results identify consumed final seeds; subsequent tuning requires
  fresh final-test seeds. Limited measured gains do not establish robust play.

## Validation

Latest recorded implementation validation, before this documentation-only
reorganization: **837 tests passed**, including Guildworks engine/CLI/recurrent
training, 35 Connect Four rule/CLI tests and existing CPU/CUDA regressions.
Ruff lint/format, strict mypy, dependency and whitespace checks passed.

The public simulator comparison covers 5,478 boards and 16,167 legal transitions
per starting seat; search checks cover budgets, perspective, terminal handling,
private RNG and trace equivalence/corruption detection. Evaluation replay
reproduced match/tactical results. Search latency measurements are single-run
observations, not an isolated benchmark. Hosted CI is not asserted.
