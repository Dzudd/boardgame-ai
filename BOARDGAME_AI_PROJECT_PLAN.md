# Board Game AI — Roadmap

Build a reusable platform: add a new game through a documented plugin, then train
and play a substantial self-taught model on the user's GPU without rewriting the
training system. Tiny CPU models are development fixtures, not the final product.

**Status: unfinished.** Existing engines and bounded experiments are a foundation;
large-model GPU training and new-game extensibility remain required deliverables.
Read [PROGRESS.md](PROGRESS.md) for the one next task. Read only the milestone relevant
to your work below. [Historical phases and evidence](docs/archive/README.md) are optional references.

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

Technical contracts: [architecture](ARCHITECTURE.md) and
[information boundaries](INFORMATION_BOUNDARIES.md). Usage: [README](README.md).
Current evidence and concerns: [progress](PROGRESS.md).

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

## Historical work and later research

Existing engines, PPO learners, recurrent multi-game learning and CPU public search
remain available. Earlier completion labels apply only to their bounded studies.
Their full task lists, results, limitations and unfinished research are preserved
in the [previous roadmap](docs/archive/2026-10-18-BOARDGAME_AI_PROJECT_PLAN.md).

MuZero, learned beliefs, information-set search, transfer, broader tournaments and
web UI are optional later work. They do not displace milestones A–E. Failed learning
and transfer findings remain evidence; they are not successful capability claims.

## Maintaining this plan

Keep only active requirements here. Update checkboxes rather than appending session
narratives. Current state and one next task belong in PROGRESS.md; implemented
contracts in ARCHITECTURE.md; commands in README.md and its reference. Preserve
completed experiment protocols and results without applying their frozen settings
to unrelated new development.
