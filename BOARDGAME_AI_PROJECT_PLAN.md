# Board Game AI — Roadmap

This document owns phase goals, tasks, completion status and acceptance criteria.
Implemented contracts belong in [architecture](ARCHITECTURE.md); current evidence,
limitations and the next milestone belong in [progress](PROGRESS.md).

**Status:** Phases 1–3 complete; Phase 4 complete under its original minimal
criterion; Phase 4.5 complete at its declared bounded investigation scope;
Phase 5 complete for the declared Guildworks rules, including recurrent memory/PPO.
Phase 7 has a public simulator, bounded search evaluation/diagnosis and functional
CPU search-target learner; its phase-wide criterion is not met. Phases 6–8 retain
unfinished tasks below.
**Next milestone:** predeclare the next independent Connect Four learning-quality
study with fresh roots and multiple training seeds, after bounded throughput
measurements. GPU experiments require run-layer integration first. The user runs
campaigns in their own console. The bounded CPU
[functional run](docs/phase6/CONNECT_FOUR_FUNCTIONAL_RESULTS.md) is complete;
portable-checkpoint human play is implemented and tested. A single tiny run does
not establish robust strength or the Phase 6 multi-game criteria.
Shared-model/transfer protocols and independent
learning-quality evidence remain pending, including the separate search/PPO
[comparison proposal](docs/reviews/2026-10-16-reliability-resolution.md#proposed-next-bounded-learning-experiment-not-executed).
PPO and Tic-Tac-Toe search remain references; robust Tic-Tac-Toe play is not a gate.

## Goal and principles

Build a reusable board-game learning platform, progressing from the implemented
PPO baseline toward MuZero-style search learning, then hidden-information and
multi-game learning. Eventually test transfer to new games, not just strength
on known games.

- Train exclusively through latest-policy self-play. Both seats share frozen
  weights during collection. No external teachers, expert/solver labels,
  baseline training opponents, or historical opponent pool are planned.
- Keep fixed baselines for isolated evaluation. No perfect solver is required.
- Use only player-visible observations. Engines own rules, chance, and secrets.
- Separate acting from batched optimization. Current PPO uses all collected
  decisions; future sampled self-play replay needs its own learner design.
- Keep game-defined rewards and per-player credit assignment. A win/loss does
  not prove every move was good/bad; heuristic bonuses are not a diagnosis.
- Measure learning quality and throughput separately, with explicit budgets,
  reproducible seeds, and honest limitations.

Technical contracts: [architecture](ARCHITECTURE.md) and
[information boundaries](INFORMATION_BOUNDARIES.md). Usage: [README](README.md).
Current evidence and concerns: [progress](PROGRESS.md).

## Completed phases

### Phase 1 — Project foundation

**Complete:** typed package, shared engine/agent contracts, numerical dependencies,
visibility rules, deterministic RNG, illegal-action policy, tests, and CI.

- [x] Package/dependencies, CLI foundation, lint/format/type checks, and CI.
- [x] GameState, Observation, Action, StepResult, GameSpec, and Agent/GameEngine.
- [x] Visibility, seed isolation, rejection invariants, and contract tests.

**Acceptance met:** a test game completes through the shared interface.

### Phase 2 — First deterministic game

**Complete:** Tic-Tac-Toe engine, public observations, rendering, and cloning.

- [x] Reset, legal moves, transitions, actor selection, wins/draws, and rewards.
- [x] Player observations, text board, independent cloning, and rule tests.

**Acceptance met:** all 5,478 reachable X-first boards pass the rule audit.

### Phase 3 — Simulation and baselines

**Complete:** automated matches, human CLI play, and public replays.

- [x] RandomAgent and tactical heuristic agent.
- [x] Bounded game loop, match runner, and reproducible seeded batches.
- [x] Wins/draws/scores/errors/length metrics and JSON reports.
- [x] Human-versus-baseline CLI and save/inspect public move logs.

**Acceptance met:** thousands of valid games reproduce; replay logs validate.
Measurements: [baseline benchmarks](BASELINE_BENCHMARKS.md).

### Phase 4 — First learning agent

**Complete under the original minimal criterion:** a reproducible trained policy
improves over random; tactical robustness remains unresolved.

- [x] Observation/action tensors, legal masking, and actor-critic network.
- [x] Per-player rollouts/GAE, PPO, optimization, and portable checkpoints.
- [x] CPU/CUDA latest-policy self-play and update-boundary resume.
- [x] Fixed evaluation, held-out seeds, matched controls, and training curves.
- [x] Observation audit across all reachable boards and both seat perspectives.

**Acceptance met:** the recorded 64,000-game run improves balanced score against
random by +0.3160 (approximate 95% interval [+0.2644, +0.3676]). It loses nearly
every heuristic matchup. This is one training seed, not robust or optimal play.
Evidence: [Phase 4 results](PHASE4_RESULTS.md).

## Milestone review and remaining work

### Phase 4.5 — Self-play diagnosis and learning reliability (complete)

**Deliverable:** a diagnosed, measurable self-play learning process before
expanding to harder games. Phase 4 retains its original completion status;
this phase investigated its unresolved policy weakness. All six tasks are complete
at the stated bounded scope. [Results and decision](PHASE45_RESULTS.md): no robust
entropy-setting improvement was established; prioritize public-observation search
evaluation. [Design](SEARCH_LEARNING_DESIGN.md); [protocol](PHASE45_PROTOCOL.md).

1. [x] **Diagnose the frozen policy.** Add a read-only public-observation audit of
   immediate wins and required blocks, with action probabilities, sample counts,
   and separate X/O results. Trace representative winning/losing trajectories
   through per-seat rewards, GAE, value targets, and PPO updates. Distinguish
   incorrect learning signals from exploration/optimization hypotheses. No solver
   teacher, perfect-play opponent, training labels, or source weight changes.
   Disposable-copy update checks found no demonstrated signal defect; census
   tactics remain weak outside narrow self-play paths.
2. [x] **Add checkpoint monitoring.** Retain periodic immutable checkpoints with
   hashes/counters and bounded retention. Run reproducible, isolated development
   evaluations at configured intervals, preserving model/optimizer/global and
   training-sampling RNG state. Report both seats, tactics, entropy, coverage,
   outcomes, and optimizer/sample counts. Keep the resume checkpoint separate
   from the checkpoint chosen by a declared development metric. Preserve exact
   update-boundary resume; reserve fresh final-test seeds for final assessment.
3. [x] **Profile and optimize training throughput.** Add a reproducible bounded
   CPU/CUDA benchmark with warm-up, repeated measurements, explicit CPU thread
   count, and CUDA synchronization at timing boundaries. Separate collection,
   inference, GAE/batch preparation, optimization, and checkpoint costs; report
   games/decisions per second and end-to-end time at equal learning budgets.
   First reduce repeated CUDA scalar reads, redundant validation, and diagnostic
   synchronization while retaining input/nonfinite failure checks. Then assess
   inference batched across games, preserving player-visible inputs, frozen
   collection weights, per-game/per-seat RNG ownership, targets, and resume.
   Benchmark CPU as a valid device for small models; GPU utilization alone is
   not success. Treat larger minibatches, collection sizes, or changed sampling
   order as learning-protocol experiments, not transparent speed optimizations.
   Test semantic equivalence and deterministic resume where promised; document
   numerical differences. No speedup claim without repeated end-to-end evidence.
4. [x] **Test self-play improvements.** After diagnosis, compare one change at a
   time to the unchanged PPO baseline: exploration/entropy settings, collection
   size, or optimizer settings as evidence warrants. Record distinct positions
   and action diversity; keep latest-policy self-play, game rewards, and both-seat
   training. Predeclare game and optimizer budgets, development criteria, and at
   least three training seeds; report every run and both-seat uncertainty. Fix
   proven defects before tuning. More randomness or more games alone is not a fix.
5. [x] **Define the search-learning bridge.** Specify bounded policy/value-guided
   MCTS using simulation constructed only from the permitted public observation,
   player-relative backups, terminal handling, and seeded exploration. Then
   specify a separate search-target learner: self-play search visit distributions,
   game outcomes, and configurable replay position/sequence sampling. Keep PPO
   as a comparison; do not insert search-chosen actions into PPO with incorrect
   behavior probabilities. This task is a design, not implementation of MCTS or
   MuZero. Learned representation/dynamics/prediction follow in the search track.
6. [x] **Review the evidence and next milestone.** Publish the diagnosis, all
   bounded experiment results, and an explicit decision to retain PPO or prioritize
   search-guided self-play. Do not require PPO to solve Tic-Tac-Toe indefinitely
   before progressing, and do not treat X dominance or low loss as agent strength.
   Decision: retain default PPO as reference; implement public simulation and
   bounded search evaluation next. The six-run entropy comparison did not establish
   improvement; no search-strength claim is made.

**Acceptance criteria:** traced learning signals are correct (or defects fixed
and tested); monitoring is reproducible and isolated; repeated CPU/CUDA timing
and correctness/resume checks substantiate any throughput change; controlled multi-seed
self-play comparisons and seat-specific diagnostics are reported; and a documented
decision identifies the next implementation task. No external training opponents
or teacher targets enter any run. Report failure to improve honestly; this phase
does not promise perfect play. Search implementation belongs to the prioritized
Phase 7 track, which can be brought forward after this review before Phase 5.

### Phase 5 — Complex mechanics and partial observability (complete)

**Deliverable:** a small custom card/worker-placement test game.

Start with one minimal playable engine through the shared interfaces: explicit
phases, consecutive same-player decisions, public worker placement, hidden hands,
public deck/discard counts and engine-resolved draws. Validate complete scripted/
random games and indistinguishable-world observations/actions before adding
encoders, recurrent memory or PPO. Keep the search milestone frozen as a reference;
additional Tic-Tac-Toe tuning is optional, not a prerequisite.

- [x] Implement multiple atomic decisions within one turn.
- [x] Add explicit decision phases and turn/round transitions.
- [x] Implement hidden hands, deck, discard pile and public card counts.
- [x] Implement drawing cards and other chance events.
- [x] Implement filtered observations and player-specific history.
- [x] Add game-specific hidden-world equivalence and visibility tests.
- [x] Add recurrent agent memory and recurrent PPO training.
- [x] Test empty/partial decks, forced passes and repeated actors at round boundaries;
  reject unsupported skip/interrupt/extra-turn actions atomically. These mechanics
  are outside the declared game rules rather than unimplemented live phases.

The declared Guildworks engine, filtered human CLI, full visible-history GRU memory
and complete-episode recurrent PPO are implemented. The 256-game functional run
and isolated initial/final random evaluation completed; see
[Phase 5 results](PHASE5_RESULTS.md) for evidence and
[architecture](ARCHITECTURE.md#guildworks-rules-and-transitions) for the rules.
This establishes functioning training, not robust strength, balance or a measured
benefit from memory. Interrupts, extra turns and skipped decision phases remain
outside the declared rules.

**Acceptance criteria:** the agent can play complete partially observed games; no action reveals an unresolved random outcome; same-player consecutive actions work.

Connect Four is additionally available as a standalone shared-interface engine
and human-versus-random CLI, with public observations/history and rule tests.
This adds another playable game; it does not establish multi-game learning or
select the Phase 6 evaluation protocol.

### Phase 6 — Multi-game learning

**Deliverable:** independent and shared agents trained on multiple games.

The [proposed first bounded milestone](docs/PHASE6_MILESTONE.md) describes an
independent Connect Four learning baseline and trained-checkpoint human play.
Its observation/fixed column-action encoders, independent policy/value model and
complete-game self-play collector, PPO optimization and trainer-level checkpoint
restore, bounded run-level resume/provenance/metrics recovery, cumulative caps
and isolated evaluation are implemented and tested. The declared CPU functional
run completed through pause/resume and fixed random/control evaluation;
portable-checkpoint human CLI play is implemented and tested. The
[functional protocol](docs/PHASE6_CONNECT_FOUR_PROTOCOL.md) now specifies the
independent model, self-play/evaluation budgets, seeds and resume contract;
it is not a strength comparison or a frozen multi-game experiment. Fixed action
indices do not implement general candidate-action scoring.

- [x] Validate the bounded independent Connect Four functional learner: 32 self-play
  games plus 192 fixed evaluation/control games. This does not complete shared learning.
- [x] Add at least two more games using the same engine interface: Guildworks and
  Connect Four beyond the original Tic-Tac-Toe. A new held-out transfer game is
  still required before testing adaptation.
- [ ] Implement game-specific observation encoders (Connect Four now added alongside
  Tic-Tac-Toe and Guildworks; future game selection/shared integration remain).
- [ ] Implement candidate-action encoding and scoring.
- [ ] Build a shared Transformer or recurrent backbone.
- [ ] Implement game-specific policy/value adapters where necessary (independent
  Connect Four heads available; shared-backbone integration remains).
- [ ] Create a multi-game training sampler and normalize/track rewards carefully.
- [ ] Compare separate models versus a shared model at matched compute budgets.
- [ ] Test transfer: train on games A/B, then measure learning speed on game C.
- [ ] Check for negative transfer and catastrophic forgetting.

**Acceptance criteria:** quantitative comparison of per-game strength, training cost, and adaptation speed on a held-out game.

### Phase 7 — Advanced search and research

**Deliverable:** experimentally validated planning enhancements.

**Partial:** the public-rule Tic-Tac-Toe simulator, bounded search evaluation and
diagnosis milestone is complete; [Phase 7 results](PHASE7_RESULTS.md) document its
limited gains. A separate [bounded CPU search-target learner](docs/reviews/2026-10-16-search-target-implementation.md)
is implemented and functionally tested; its learning-quality comparison is pending.
The phase-wide acceptance criterion is not yet met.

- [ ] Implement explicit or learned hidden-state belief estimates.
- [ ] Validate belief estimates against known simulated outcomes without leaking them at inference.
- [ ] Implement information-set-aware search baseline.
- [x] Define evaluator perspective and return semantics separately from action selection.
- [x] Add bounded policy/value PUCT on public-rule Tic-Tac-Toe simulation (no authoritative clone).
- [ ] Add separate rollout comparison if warranted.
- [ ] Compare heuristic, rollout, neural, and hybrid evaluation against fixed baselines.
- [x] Integrate policy/value guidance into bounded public search.
- [x] Implement search-guided self-play with policy/value targets and sampled self-generated replay as a separate learner (bounded CPU Tic-Tac-Toe, K=0; strength comparison pending).
- [ ] Implement MuZero-style learned representation, dynamics, and prediction as the long-term search-learning extension.
- [ ] Test chance-event modeling and same-player search backups.
- [ ] Benchmark planning strength versus computation and latency.

**Acceptance criteria:** advanced methods improve defined metrics relative to a simpler baseline under comparable budgets.

### Phase 8 — Productization and human play

**Deliverable:** reliable human-versus-AI experience and tournament tools.

- [x] Provide CLI human play for Tic-Tac-Toe against random/heuristic agents.
- [x] Provide filtered Guildworks human CLI play against RandomAgent.
- [x] Add Connect Four human CLI play against validated portable checkpoints,
  functionally tested using disposable initialized and optimized snapshots.
- [ ] Extend trained-checkpoint human play to other games and add a web UI if needed.
- [ ] Add selectable games, agents and difficulty levels.
- [x] Save and inspect public Tic-Tac-Toe move replays.
- [ ] Save/load resumable games and extend replay support to other games.
- [ ] Add tournament runner and ratings or matchup statistics.
- [ ] Profile inference latency and search budgets.
- [ ] Document how to add a new game.

**Acceptance criteria:** a user can select a supported game, play against a trained agent, and finish a game without manual state edits.

## Search direction

The Phase 4.5 review brought the first Phase 7 milestone forward. Public simulation,
16/32 search evaluation and heuristic-O trace diagnosis are complete. Phase 5 is
complete at its declared scope. Phase 6's independent Connect Four functional
protocol has an implemented model/collector/PPO/trainer restore and bounded
run/resume/evaluation layer; its bounded CPU functional run is complete. Independent
learning-quality evidence is still needed before shared-model comparisons; additional
search tuning or solving Tic-Tac-Toe is not a prerequisite. Prior-concentration
ablation is deferred optional research. The longer-term search sequence remains:

1. Validate bounded policy/value-guided search with permitted rule-based simulation.
2. Train from self-play search policies and outcomes using budgeted replay sampling.
3. Add learned representation, dynamics, and prediction; compare against rule-based search.
4. Extend explicitly for chance, hidden information, same-player actions, and multiple players.

PPO remains a comparison baseline. Public-rule bounded search and the separate
CPU Tic-Tac-Toe search-target learner are implemented; empirical learning-quality
comparison remains pending. A rules simulator
provides transitions, not expert opponent decisions. Hidden-information planning
must respect information sets; cloning true hidden state or searching sampled
worlds as though their secrets were known is invalid.

## Concerns for later phases

- Variable action spaces and game structures need explicit per-game adapters;
  imagined-state legality is not automatically supplied by real-engine masks.
- Hidden information/chance require beliefs, history, and information-set-aware
  planning; ordinary deterministic MuZero does not solve them automatically.
- Same-player decisions and multiplayer/general-sum outcomes need explicit
  perspective, duration/discount, and backup semantics.
- Shared models may suffer negative transfer/forgetting. Compare separate models
  at matched compute; define whether the goal is fixed-game strength or faster
  adaptation to unseen games. Scaling to many games is a long-term research goal.
- Sample efficiency, wall-clock cost, hardware utilization, strength uncertainty,
  and inference/search latency must be measured independently.
- ISMCTS, CFR/ReBeL-style approaches, learned beliefs, and stochastic learned
  dynamics remain research options, not implemented guarantees.
- Implement game rules/assets only where permitted by their rights/licenses.

## Maintaining these documents

Update existing status/checklists when work completes; remove superseded pending
statements. Keep only current concerns and one next task in PROGRESS.md. Put
technical contracts in ARCHITECTURE.md, visibility rules in INFORMATION_BOUNDARIES.md,
commands in README.md, and measured protocols/results in dedicated reports.
Git history preserves task chronology; do not append per-task completion essays
or duplicate the roadmap in another file.
