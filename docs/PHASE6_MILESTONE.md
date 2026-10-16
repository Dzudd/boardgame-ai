# Proposed first Phase 6 milestone

The learner/checkpoint-play scope below remains a development proposal. The
[independent functional protocol](PHASE6_CONNECT_FOUR_PROTOCOL.md) now fixes the
first learner's technical contract and bounded validation budget; it is not a
strength or multi-game experiment. Implemented prerequisites now include public
Connect Four observation/action encoders, an independent policy/value model and
complete-game latest-policy self-play collection, PPO optimization and portable
trainer checkpoint restoration.
See [architecture](../ARCHITECTURE.md#connect-four) for the implemented contract.
The [roadmap](../BOARDGAME_AI_PROJECT_PLAN.md#phase-6--multi-game-learning)
owns all Phase 6 tasks and acceptance criteria; [progress](../PROGRESS.md) owns
current status. Connect Four has an engine, human-versus-random CLI and a
programmatic self-play/PPO learner with trainer-level checkpoints. The bounded
training/evaluation command and human checkpoint play remain unimplemented.

## Completed prerequisite and remaining scope

The original encoder-only increment added no optimizer or experiment. Its tests cover both
seat perspectives/starters, lossless board recovery, wins/draws, full-column and
inactive/terminal masks, supplied-object decoding, malformed inputs, storage and
global RNG isolation. The independent functional model, budgets, seed schedules
and resume requirements are now specified in the linked protocol and versioned
numeric settings. The model and complete-game self-play collector are implemented
and tested, including both-seat terminal rewards/GAE, snapshot/RNG isolation,
deadlines, deterministic sampling and CUDA-source collection. PPO and separate
trainer checkpoint restore are now implemented and tested, with exact CPU
continuation and CUDA portability. The next bounded Phase 6 task is the run layer:
protocol provenance, absolute scheduling, metrics recovery, cumulative caps and
isolated evaluation. These run-level guarantees are not supplied by trainer restore.
The declared functional training/evaluation run has not been executed.

## Scope and protocol prerequisites

The proposed first bounded implementation is an independent Connect Four PPO
baseline plus human play against its saved checkpoint. Checkpoint human play is
a brought-forward subset of Phase 8, not completion of that phase. Independent
training alone does not establish multi-game learning or complete Phase 6.

The linked functional protocol defines game roles, the independent baseline,
per-seat outcomes/rewards, model, encoding, collection/update boundaries,
checkpoint/resume requirements and small training/evaluation caps. Guildworks
and Connect Four satisfy the two additional engine games beyond Tic-Tac-Toe.
Shared-model matched-compute comparisons and transfer/forgetting measurements
still require a separate protocol, a new unseen game and fresh final seeds.
The numeric functional budget does not authorize a lengthy strength campaign.

The initial milestone measures function; larger strength, shared-model, transfer,
and forgetting experiments remain later work. Report every count and outcome;
a successful functional run does not demonstrate robust playing strength.

## Technical requirements

- Reuse GameEngine, Agent, Observation, Action, and StepResult. Keep rules
  independent of PyTorch and preserve existing training behavior and checkpoint
  formats when reusing PPO infrastructure.
- Encode only public structured observations using player-relative board features,
  seven column outputs, and supplied legal-action masking. Decode to a supplied
  legal Action. Specify inactive/terminal encoding and zero terminal bootstrap.
- Begin with a small feed-forward policy/value model; recurrence is unnecessary
  for this fully observed baseline. Agent/model inputs exclude engines,
  authoritative clones, RNG, seeds, and debug state.
- Collect latest-policy self-play only, with correct per-seat rewards/GAE and
  private reproducible randomness that preserves global RNG. Evaluation games,
  baseline opponents, and tactical labels never enter optimizer batches.
- Provide a bounded training command, portable validated checkpoints, configuration
  and metrics, update-boundary resume, and frozen random-opponent evaluation in
  both seats. Reject incompatible checkpoints without damaging existing outputs.
- Add human play against a loaded model through an explicit checkpoint option.
  Preserve the board, separator, aligned column labels 1–7, input retries,
  cancellation, outcomes, and current random opponent. Human columns remain 1–7;
  engine/model actions remain zero-based 0–6.

## Validation and limits

Validate encoding perspective, masks/decoding, full-column exclusion, reward
ownership, terminal handling, legal complete games, reproducibility/global RNG
preservation, meaningful PPO updates, checkpoint rejection/resume, and human
checkpoint play. Validate CPU and CUDA when available, without promising
cross-device identity. Run relevant development checks and the full regression
suite; preserve existing Tic-Tac-Toe, Guildworks, search, and training artifacts.

Publish the predeclared functional run's counts/outcomes and exact train, resume,
and checkpoint-play commands alongside the implemented subset and remaining
Phase 6 tasks. Full phase completion requires measured evidence for every
acceptance criterion in the roadmap.

A shared backbone, multi-game sampler, additional new game, search, MuZero,
replay learner, and additional Tic-Tac-Toe tuning are outside this first proposal.
The [search-learning design](../SEARCH_LEARNING_DESIGN.md) retains the separate
future search-target learner rationale.
