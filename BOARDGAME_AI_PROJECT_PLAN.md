# Board Game AI — Roadmap

**Status:** Phases 1–4 complete under the original minimal criterion.
**Active phase:** 4.5, planned. **Next:** Task 1, read-only learning diagnosis.

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
- Work one small task at a time. Measure learning quality and throughput
  separately, with explicit budgets, reproducible seeds, and honest limitations.

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

## Remaining work

### Phase 4.5 — Self-play diagnosis and learning reliability (planned)

**Deliverable:** a diagnosed, measurable self-play learning process before
expanding to harder games. Phase 4 retains its original completion status;
this phase addresses its unresolved policy weakness. All tasks below are pending.

1. [ ] **Diagnose the frozen policy.** Add a read-only public-observation audit of
   immediate wins and required blocks, with action probabilities, sample counts,
   and separate X/O results. Trace representative winning/losing trajectories
   through per-seat rewards, GAE, value targets, and PPO updates. Distinguish
   incorrect learning signals from exploration/optimization hypotheses. No solver
   teacher, perfect-play opponent, training labels, or weight changes.
2. [ ] **Add checkpoint monitoring.** Retain periodic immutable checkpoints with
   hashes/counters and bounded retention. Run reproducible, isolated development
   evaluations at configured intervals, preserving model/optimizer/global and
   training-sampling RNG state. Report both seats, tactics, entropy, coverage,
   outcomes, and optimizer/sample counts. Keep the resume checkpoint separate
   from the checkpoint chosen by a declared development metric. Preserve exact
   update-boundary resume; reserve fresh final-test seeds for final assessment.
3. [ ] **Profile and optimize training throughput.** Add a reproducible bounded
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
4. [ ] **Test self-play improvements.** After diagnosis, compare one change at a
   time to the unchanged PPO baseline: exploration/entropy settings, collection
   size, or optimizer settings as evidence warrants. Record distinct positions
   and action diversity; keep latest-policy self-play, game rewards, and both-seat
   training. Predeclare game and optimizer budgets, development criteria, and at
   least three training seeds; report every run and both-seat uncertainty. Fix
   proven defects before tuning. More randomness or more games alone is not a fix.
5. [ ] **Define the search-learning bridge.** Specify bounded policy/value-guided
   MCTS using simulation constructed only from the permitted public observation,
   player-relative backups, terminal handling, and seeded exploration. Then
   specify a separate search-target learner: self-play search visit distributions,
   game outcomes, and configurable replay position/sequence sampling. Keep PPO
   as a comparison; do not insert search-chosen actions into PPO with incorrect
   behavior probabilities. This task is a design, not implementation of MCTS or
   MuZero. Learned representation/dynamics/prediction follow in the search track.
6. [ ] **Review the evidence and next milestone.** Publish the diagnosis, all
   bounded experiment results, and an explicit decision to retain PPO or prioritize
   search-guided self-play. Do not require PPO to solve Tic-Tac-Toe indefinitely
   before progressing, and do not treat X dominance or low loss as agent strength.

**Acceptance criteria:** traced learning signals are correct (or defects fixed
and tested); monitoring is reproducible and isolated; repeated CPU/CUDA timing
and correctness/resume checks substantiate any throughput change; controlled multi-seed
self-play comparisons and seat-specific diagnostics are reported; and a documented
decision identifies the next implementation task. No external training opponents
or teacher targets enter any run. Report failure to improve honestly; this phase
does not promise perfect play. Search implementation belongs to the prioritized
Phase 7 track, which can be brought forward after this review before Phase 5.

### Phase 5 — Complex mechanics and partial observability

**Deliverable:** a small custom card/worker-placement test game.

- [ ] Implement multiple atomic decisions within one turn.
- [ ] Add explicit decision phases and turn/round transitions.
- [ ] Implement hidden hands, deck, discard pile and public card counts.
- [ ] Implement drawing cards and other chance events.
- [ ] Implement filtered observations and player-specific history.
- [ ] Add tests ensuring hidden state never leaks.
- [ ] Add recurrent agent memory and recurrent PPO training.
- [ ] Test edge cases: empty decks, skipped phases, interrupts, extra turns.

**Acceptance criteria:** the agent can play complete partially observed games; no action reveals an unresolved random outcome; same-player consecutive actions work.

### Phase 6 — Multi-game learning

**Deliverable:** independent and shared agents trained on multiple games.

- [ ] Add at least two more games using the same engine interface.
- [ ] Implement game-specific observation encoders.
- [ ] Implement candidate-action encoding and scoring.
- [ ] Build a shared Transformer or recurrent backbone.
- [ ] Implement game-specific policy/value adapters where necessary.
- [ ] Create a multi-game training sampler and normalize/track rewards carefully.
- [ ] Compare separate models versus a shared model at matched compute budgets.
- [ ] Test transfer: train on games A/B, then measure learning speed on game C.
- [ ] Check for negative transfer and catastrophic forgetting.

**Acceptance criteria:** quantitative comparison of per-game strength, training cost, and adaptation speed on a held-out game.

### Phase 7 — Advanced search and research

**Deliverable:** experimentally validated planning enhancements.

- [ ] Implement explicit or learned hidden-state belief estimates.
- [ ] Validate belief estimates against known simulated outcomes without leaking them at inference.
- [ ] Implement information-set-aware search baseline.
- [ ] Define evaluator perspective and return semantics separately from action selection.
- [ ] Add bounded rollout evaluation and engine-based MCTS on a perfect-information game.
- [ ] Compare heuristic, rollout, neural, and hybrid evaluation against fixed baselines.
- [ ] Integrate policy/value guidance into search.
- [ ] Implement search-guided self-play with policy/value targets and sampled self-generated replay as a separate learner.
- [ ] Implement MuZero-style learned representation, dynamics, and prediction as the long-term search-learning extension.
- [ ] Test chance-event modeling and same-player search backups.
- [ ] Benchmark planning strength versus computation and latency.

**Acceptance criteria:** advanced methods improve defined metrics relative to a simpler baseline under comparable budgets.

### Phase 8 — Productization and human play

**Deliverable:** reliable human-versus-AI experience and tournament tools.

- [x] Provide CLI human play for Tic-Tac-Toe against random/heuristic agents.
- [ ] Add human play against trained checkpoints and a web UI if needed.
- [ ] Add selectable games, agents and difficulty levels.
- [x] Save and inspect public Tic-Tac-Toe move replays.
- [ ] Save/load resumable games and extend replay support to other games.
- [ ] Add tournament runner and ratings or matchup statistics.
- [ ] Profile inference latency and search budgets.
- [ ] Document how to add a new game.

**Acceptance criteria:** a user can select a supported game, play against a trained agent, and finish a game without manual state edits.

## Search direction

After Phase 4.5 review, choose whether to bring Phase 7 search work forward before
Phase 5 game mechanics. The sequence is:

1. Validate bounded policy/value-guided search with permitted rule-based simulation.
2. Train from self-play search policies and outcomes using budgeted replay sampling.
3. Add learned representation, dynamics, and prediction; compare against rule-based search.
4. Extend explicitly for chance, hidden information, same-player actions, and multiple players.

PPO remains a comparison baseline. Search is not implemented. A rules simulator
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
