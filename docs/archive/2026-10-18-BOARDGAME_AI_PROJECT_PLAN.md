> Historical snapshot before documentation simplification. Current direction is in
> [the active roadmap](../../BOARDGAME_AI_PROJECT_PLAN.md) and [progress](../../PROGRESS.md).
> Old next-task statements are historical, not current instructions.

# Board Game AI — Roadmap

This document owns phase goals, tasks, completion status and acceptance criteria.
Implemented contracts belong in [architecture](../../ARCHITECTURE.md); current evidence,
limitations and the next milestone belong in [progress](../../PROGRESS.md).

**Primary goal (revised 2026-10-18):** deliver a reusable board-game engine and
learning platform. After platform development, the user can supply a new game's
rules and representations through a documented game plugin, then train a substantial
model from self-play on their GPU without rewriting the learner, collector or
checkpoint system. Optimize this pipeline for the user's hardware and make the
resulting checkpoint playable.
Tiny CPU models are development fixtures and comparison baselines, not the product.

**Overall status: unfinished.** The reusable engine foundation and several bounded
learning experiments exist. New-game onboarding, large-model GPU training,
sustained training and demonstrated learning on newly added games remain unfinished.
The mandatory delivery track below takes priority over the historical phase order.

**Historical scope status:** Phases 1–3 complete; Phase 4 complete under its original minimal
criterion; Phase 4.5 complete at its declared bounded investigation scope;
Phase 5 complete for the declared Guildworks rules, including recurrent memory/PPO.
Phase 7 has a public simulator, bounded search evaluation/diagnosis and functional
CPU search-target learner; its phase-wide criterion is not met. Phase 6 is complete
at the frozen v1 comparison scope, with mixed/negative findings and an uncertain
recovered wall-cost measurement. Phases 7–8 retain unfinished tasks below.
**Completed reference evidence:** all 36 operator jobs/evaluations in the three-seed
[shared/separate and unseen-transfer study](../../docs/phase6/MULTIGAME_PROTOCOL_V1.md)
are complete and [reviewed](../../docs/phase6/MULTIGAME_RESULTS_V1.md). Quantitative
strength/cost/adaptation comparisons are available; positive transfer is unproven.
The separate [CUDA 100,000-game schedule](../../docs/phase6/CONNECT_FOUR_CUDA_100K_V1.md)
and overnight attempts remain preserved and resumable under their original caps.
The user runs campaigns in their own console. The bounded CPU
[functional run](../../docs/phase6/CONNECT_FOUR_FUNCTIONAL_RESULTS.md) is complete;
portable-checkpoint human play is implemented and tested. A single tiny run does
not establish robust strength or the Phase 6 multi-game criteria.
Shared-model/transfer evidence and the failed public search/PPO v1 comparison
are now published. The prepared three-hour v2 study is optional reference research,
not the next mandatory milestone and not evidence of GPU scaling.
PPO and Tic-Tac-Toe search remain references; robust Tic-Tac-Toe play is not a gate.

## Goal and principles

Build a reusable engine and GPU-oriented self-play learning platform for new board
games supplied after development. Adding a game requires a rules plugin and declared
observation/action/reward representations, not a new training stack.
The model learns its policy/value estimates from its own games and game-defined
outcomes, starting from random initialization; the rules engine supplies legal
transitions, not expert decisions. General transfer and MuZero are later research
options, not prerequisites for delivering this capability.

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
- Treat large-model GPU training as a required engineering deliverable. CUDA
  optimizer support alone, a longer tiny-model run, or a completed experiment does
  not satisfy it. CPU rule execution is allowed; policy inference and optimization
  must have a working batched GPU path.
- Preserve completed protocols/results and their negative findings. New model
  architectures and collection methods use new configurations and output roots.
- Size the model against actual GPU memory and end-to-end throughput. More
  parameters do not guarantee stronger play; both capacity and learning must be
  measured. Do not silently substitute a tiny CPU configuration for the GPU goal.

Technical contracts: [architecture](../../ARCHITECTURE.md) and
[information boundaries](../../INFORMATION_BOUNDARIES.md). Usage: [README](../../README.md).
Current evidence and concerns: [progress](../../PROGRESS.md).

## Mandatory delivery track — reusable new-game training on GPU

This track defines project completion. Historical phases below retain their original
scope and evidence; their checkmarks do not complete these milestones. Implementation
can overlap where dependencies permit, but sustained target-game training depends on
validated rules, representations and the GPU pipeline.

**Meaning of any new game:** support new games through extension contracts rather
than a hard-coded game list. Explicitly define the supported mechanics, player
counts and reward semantics. Unsupported mechanics must fail clearly or require a
general contract extension. Supplying prose rules alone is not assumed to generate
a correct engine automatically, and strength cannot be guaranteed for every game.
Below, "target game" means a game integrated through this reusable workflow, not
one particular game around which the platform is built.

### A — Finish the reusable game integration contract

- [ ] Define versioned game-plugin contracts for rules, player count, scoring,
  phases, legal actions, chance, hidden observations and per-player rewards.
- [ ] Provide registration and a plugin scaffold so new games can be added without
  editing the learner or a hard-coded list of training games.
- [ ] Validate complete scripted/random games, edge cases, reproducible chance,
  illegal-action rejection, rewards and player-visible information boundaries.
- [ ] Define pluggable observation/action encodings, history and recurrent memory;
  support variable legal actions and declare dimensions through plugin metadata.
- [ ] Provide reusable conformance tests and a generic human/debug interface.
- [ ] Route mechanically different existing games through the same training entry
  point using their plugins and configurations, including hidden information,
  chance and repeated actors. Document the supported player/reward envelope.

**Acceptance:** multiple mechanically different games run through one learning
entry point without game-specific learner branches. The documented contract covers
rules, validated learning inputs, per-player outcomes and onboarding. Milestone D
must additionally prove extensibility using a newly added game.

### B — Define the large model and GPU capacity envelope

- [ ] Inventory the installed GPU, available VRAM, driver/runtime and host resources.
  Earlier measurements used an RTX 4070 Ti; verify the current hardware rather than
  assuming its usable memory or performance.
- [ ] Replace fixed tiny-network assumptions with configurable architecture,
  width/depth, memory capacity and versioned checkpoint reconstruction metadata.
  Select a suitable board/history encoder and policy/value heads for the target game.
- [ ] Benchmark a capacity ladder with approximately 1M, 10M and 30M trainable
  parameters where memory permits, then select and freeze the production size.
  These are engineering probe sizes, not claims of an optimal architecture.
- [ ] Target at least 10M trainable parameters for the first substantial production
  model. Record exact count, precision, inference/learner batch sizes, sequence
  lengths, peak VRAM and memory headroom. If that target is infeasible at useful
  throughput, document the measured constraint and agree a revised target; do not
  declare this milestone complete using the existing 64×64 networks.
- [ ] Include weights, gradients, optimizer state, activations, inference snapshots
  and buffers in memory accounting; verify full training steps, not just model load.

**Acceptance:** a reproducible target-game forward/backward/optimizer benchmark
demonstrates the declared large model fits the user's GPU with memory headroom.
This establishes capacity, not learning or sustained pipeline performance.

### C — Build and optimize the end-to-end GPU self-play pipeline

- [ ] Run many active game environments and batch their policy/value inference on
  the GPU. Tune environment concurrency and CPU worker count to supply that batch;
  preserve per-game/per-seat RNG, legal masks and independent recurrent memory.
- [ ] Keep collection weights frozen for each declared collection generation and
  publish new weights at explicit boundaries. Preserve latest-policy self-play;
  do not accidentally mix stale trajectories into on-policy PPO updates.
- [ ] Separate CPU rule simulation, encoding, batched inference, trajectory storage,
  target construction and GPU optimization with bounded queues and backpressure.
- [ ] Batch recurrent sequences/candidate actions correctly, including padding,
  masking, episode boundaries and same-player consecutive decisions.
- [ ] Minimize per-decision host/device transfers and scalar synchronization;
  profile data preparation, transfer, inference, simulation and optimization.
- [ ] Evaluate mixed precision, pinned staging buffers, compilation and transfer
  overlap where supported. Adopt only measured improvements with finite losses,
  valid distributions and acceptable numerical/learning behavior.
- [ ] Measure warmed-up end-to-end decisions/second, games/hour, optimizer/sample
  throughput, stage times and peak VRAM across repeated runs. Compare batch and
  concurrency settings at the same model and learning workload; GPU utilization
  alone is not acceptance. Freeze numerical throughput targets from these results.
- [ ] Implement reliable atomic checkpoints, portable inference export, optimizer/
  RNG/counter recovery and interruption/resume for the large GPU configuration.
  State the actual reproducibility guarantees; do not promise cross-device bitwise
  identity or silently discard pending trajectories on resume.

**Acceptance:** the declared large model completes collection and optimization
through the real game pipeline on the GPU, meets the frozen throughput and memory
targets, and survives a tested interruption/resume. No hidden CPU inference fallback.
Tiny CPU tests remain useful for correctness but cannot satisfy this acceptance.

### D — Demonstrate sustained learning on a newly added game

- [ ] After the contracts are stable, add an acceptance game not used to develop
  the training system. Integrate it using only plugin files, registration and
  configuration. Required learner changes reopen the extensibility milestone.
- [ ] Freeze a runnable production configuration: game/model versions, device,
  precision, concurrency, batches, learning objective, schedules, seeds, evaluation
  roots, checkpoint cadence and wall/sample budgets.
- [ ] Run a minimum two-hour integrated GPU stability trial or the full declared
  campaign if shorter; measure memory stability, throughput drift and checkpoint
  overhead. This trial alone does not establish playing strength.
- [ ] Train from random initialization using only self-play and game rewards.
  Choose PPO or a separately correct search-target learner based on the game's
  requirements. Search is optional; if selected, its inference must also be batched
  on GPU and its simulation must obey information boundaries.
- [ ] Track policy/value learning, action diversity, game outcomes, illegal actions,
  per-seat behavior and held-out strength over checkpoints. Use fixed opponents
  only for evaluation; prevent evaluation information from entering training.
- [ ] Predeclare target-game strength criteria and compare initial, trained and
  smaller reference models using held-out games and at least three training seeds
  for the learning claim. Report uncertainty, regressions and actual compute costs.
- [ ] Diagnose failed learning or throughput before increasing the campaign length;
  revise and version the experiment without rewriting completed results.

**Acceptance:** a sustained, resumable large-model GPU campaign is demonstrated,
and multi-seed evaluation meets the declared target-game improvement criterion.
Successful execution, larger parameter count and lower training loss are insufficient
on their own. If learning fails, this milestone stays open with the evidence retained.

### E — Play the trained model and reuse the platform

- [ ] Load the trained target-game checkpoint in human play, including correct
  legal-action handling and per-seat memory; finish games without manual state edits.
- [ ] Measure and expose practical inference latency and optional search budgets.
- [ ] Document the complete workflow: add rules/encoders, select model capacity,
  calibrate GPU settings, train/resume, evaluate and play.

**Acceptance:** after development, the user can add a new game plugin without
rewriting the learning system, train the declared large model
through GPU self-play, resume it, evaluate learning and play the trained checkpoint
using documented commands. A graphical interface is optional.

**Next task:** audit and finish the general game-plugin/training contract and
establish the hardware/model-capacity envelope (A and B). Then implement batched
GPU collection (C).
Do not default to another tiny Tic-Tac-Toe study as the next delivery milestone.

## Historical phases and reference research

The phase labels below describe completed bounded work and remaining research.
They are retained for traceability, not as a percentage of overall project completion.

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
at the stated bounded scope. [Results and decision](../../PHASE45_RESULTS.md): no robust
entropy-setting improvement was established; prioritize public-observation search
evaluation. [Design](../reference/SEARCH_LEARNING_DESIGN.md); [protocol](../../PHASE45_PROTOCOL.md).

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
[architecture](../../ARCHITECTURE.md#guildworks-rules-and-transitions) for the rules.
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

**Complete at frozen v1 comparison scope (2026-10-18).** All 36 operator training
jobs and evaluations were reviewed. The [results](../../docs/phase6/MULTIGAME_RESULTS_V1.md)
quantify per-game strength, cost differences, unseen adaptation and forgetting,
including negative findings. Exact wall matching is verified for two seed pairs;
the recovered third pair's timing remains uncertain. Successful positive transfer
and robust stronger-opponent play are not completion claims.

The [proposed first bounded milestone](../../docs/PHASE6_MILESTONE.md) describes an
independent Connect Four learning baseline and trained-checkpoint human play.
Its observation/fixed column-action encoders, independent policy/value model and
complete-game self-play collector, PPO optimization and trainer-level checkpoint
restore, bounded run-level resume/provenance/metrics recovery, cumulative caps
and isolated evaluation are implemented and tested. The declared CPU functional
run completed through pause/resume and fixed random/control evaluation;
portable-checkpoint human CLI play is implemented and tested. The
[functional protocol](../../docs/PHASE6_CONNECT_FOUR_PROTOCOL.md) now specifies the
independent model, self-play/evaluation budgets, seeds and resume contract;
it is not a strength comparison or a frozen multi-game experiment. Fixed action
indices do not implement general candidate-action scoring.

- [x] Validate the bounded independent Connect Four functional learner: 32 self-play
  games plus 192 fixed evaluation/control games. This does not complete shared learning.
- [x] Add explicit CUDA run-layer support and a separate fixed 100,000-game
  training/play configuration; tested support, separate original campaign incomplete.
- [x] Add at least two more games using the same engine interface: Guildworks and
  Connect Four beyond the original Tic-Tac-Toe. Additional unseen Breakthrough
  is frozen and has reviewed adaptation measurements.
- [x] Implement game-specific shared observation encoders: public boards and
  Guildworks visible streams, including the held-out Breakthrough adapter.
- [x] Implement supplied candidate-action encoding and permutation-equivariant scoring.
- [x] Build a shared recurrent GRU backbone with independent seat/episode memory.
- [x] Implement per-game observation, candidate/query and own-return value adapters.
- [x] Create a complete-game multi-game sampler, fixed reward/value divisors,
  equal-game PPO losses and separate raw/per-game metrics.
- [x] Prepare frozen separate/shared wall-budget and decision-ceiling comparisons,
  three seeds, cumulative caps, portable atomic checkpoints and isolated evaluation.
- [x] Freeze unseen Breakthrough 5x5 v1 before fitting and implement paired
  pretrained/fresh adaptation with identical fresh C adapters/reset Adam.
- [x] Implement negative-transfer, forgetting, cost and censored-adaptation reports.
- [x] Compare separate models versus a shared model at matched compute budgets:
  wall and near-matched decision ceilings; report all realized differences and
  the recovered wall seed's failed exact timing measurement.
- [x] Test transfer: train on development games, then measure learning speed on
  unseen Breakthrough; full curves/target costs reviewed, consistent speedup failed.
- [x] Check for negative transfer and catastrophic forgetting: Guildworks sample
  penalty measured; retention uncertainty preserved for two game/regime cases.

**Acceptance criteria:** quantitative comparison of per-game strength, training cost, and adaptation speed on a held-out game.

The [frozen protocol](../../docs/phase6/MULTIGAME_PROTOCOL_V1.md) is now executed;
[machine-readable evidence](../../docs/phase6/multigame-v1-operator-results.json) and the
results document preserve the full quantitative comparison and its failures.
Acceptance is evidence of comparison, not proof that all performance hypotheses
passed. No scheduled evidence remains missing. The timing uncertainty, limited
opponents/seeds, coarse adaptation checkpoints and unresolved positive-transfer/
noninferiority questions remain explicit limitations. Further studies need a new
declaration, fresh output and fresh final roots; none was launched here.

### Phase 7 — Advanced search and research

**Deliverable:** experimentally validated planning enhancements.

**Partial:** the public-rule Tic-Tac-Toe simulator, bounded search evaluation and
diagnosis milestone is complete; [Phase 7 results](../../PHASE7_RESULTS.md) document its
limited gains. A separate [bounded CPU search-target learner](../../docs/reviews/2026-10-16-search-target-implementation.md)
is implemented and functionally tested. Its first frozen learning-quality
comparison is complete with a failed improvement hypothesis: search-target raw
score .32764 versus independent PPO .34440, paired effect −1.68 points
[−3.81, +0.46], with higher training cost. [Full results](../../docs/phase7/PUBLIC_STUDY_V1_RESULTS.md).
The phase-wide acceptance criterion is not yet met.

The [2026-10-18 comparison audit and runbook](../../docs/phase7/COMPARISON_V1.md)
orders the next work: paired public learner evaluation, separately capped
throughput calibration, numerical study freeze/capped training envelope, then
measured improvement. The isolated dual-format evaluator and calibration harness
are implemented; the operator comparison is now reviewed and improvement remains
unproven. Independent PPO remains the baseline; historical search gains and
Phase 6 quantitative comparisons do not complete Phase 7.

- [x] Prepare isolated paired checkpoint evaluation for independent PPO/search-target formats, both seats and random/heuristic opponents, with identities, raw-policy strength, separately labelled search, latency, partial reports and cumulative caps.
- [x] Prepare separately capped throughput calibration and accepted-cost accounting without changing either learner.
- [x] Measure bounded calibration throughput/costs, then freeze numerical training/evaluation budgets and validate a capped comparison training envelope (public study v1; tiny calibration is not steady-state/full-replay speed evidence).
- [x] Execute the reserved multi-seed raw-policy comparison and report paired effects, per-seat regressions, wall/sample/search/optimizer/replay costs and failed hypotheses under the frozen success criteria (v1 complete; primary improvement and heuristic-X guard fail; no Phase 7 acceptance).

The operator completed the six preparation calibration jobs. The
[frozen public study v1](../../docs/phase7/PUBLIC_STUDY_V1.md) now declares the numeric
256-game/arm allocation, cumulative caps and fresh final evaluation schedule.
All six training jobs and all 6,144 final games completed and validate; the
comparison itself is complete, but its hypothesis did not pass. Preserve those
negative/inconclusive findings, the consumed final roots and unequal compute costs.
The read-only replay-fit diagnosis shows improved in-sample fit without established
external improvement; causal attribution or a new reuse ablation remains research.

A new [three-hour maximum v2 allocation](../../docs/phase7/PUBLIC_STUDY_THREE_HOUR_V2.md)
is prepared with fresh roots/output, equal-game work limits and cumulative caps
across all six jobs. It reuses the existing learners/supervisor; it has not been
executed and establishes no new strength or phase acceptance evidence.

- [ ] Implement explicit or learned hidden-state belief estimates.
- [ ] Validate belief estimates against known simulated outcomes without leaking them at inference.
- [ ] Implement information-set-aware search baseline.
- [x] Define evaluator perspective and return semantics separately from action selection.
- [x] Add bounded policy/value PUCT on public-rule Tic-Tac-Toe simulation (no authoritative clone).
- [ ] Add separate rollout comparison if warranted.
- [ ] Compare heuristic, rollout, neural, and hybrid evaluation against fixed baselines.
- [x] Integrate policy/value guidance into bounded public search.
- [x] Implement search-guided self-play with policy/value targets and sampled self-generated replay as a separate learner (bounded CPU Tic-Tac-Toe, K=0; v1 strength comparison failed to establish improvement).
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

The mandatory reusable-platform/GPU delivery track takes precedence over the
research sequence below. MuZero, transfer studies and additional tiny-model
comparisons are not prerequisites for delivering new-game GPU training.

The Phase 4.5 review brought the first Phase 7 milestone forward. Public simulation,
16/32 search evaluation and heuristic-O trace diagnosis are complete. Phase 5 is
complete at its declared scope. Phase 6's independent Connect Four functional
protocol has an implemented model/collector/PPO/trainer restore and bounded
run/resume/evaluation layer; its bounded CPU functional run is complete. Independent
learning-quality evidence is now published in the shared-model comparison results;
additional search tuning or solving Tic-Tac-Toe was not a prerequisite. Prior-concentration
ablation is deferred optional research. The longer-term search sequence remains:

1. Validate bounded policy/value-guided search with permitted rule-based simulation.
2. Train from self-play search policies and outcomes using budgeted replay sampling.
3. Add learned representation, dynamics, and prediction; compare against rule-based search.
4. Extend explicitly for chance, hidden information, same-player actions, and multiple players.

PPO remains a comparison baseline. Public-rule bounded search and the separate
CPU Tic-Tac-Toe search-target learner are implemented; empirical v1 comparison is
complete but failed to establish learning-quality improvement. A rules simulator
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
