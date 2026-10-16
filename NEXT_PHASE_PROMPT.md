# Next session prompt — Phase 6, first bounded milestone

Read BOARDGAME_AI_PROJECT_PLAN.md, ARCHITECTURE.md, INFORMATION_BOUNDARIES.md,
PROGRESS.md, README.md and PHASE5_RESULTS.md, then inspect the shared engine/agent
contracts, Connect Four engine/tests and existing PPO, checkpoint and evaluation
components. Preserve all existing Tic-Tac-Toe, Guildworks, search and training
artifacts. Phase 5 is complete at its declared scope; Connect Four currently has
an engine, random runner support and human-versus-random CLI, but no learner.

Before implementation:

1. Print EVERY task listed under Phase 6 in the current roadmap, with its current
   checkbox status, plus its deliverable and acceptance criteria. Do not present
   the first milestone as the whole phase or silently mark other tasks complete.
2. Explain how the existing games count toward the additional-game requirement;
   do not assume that requirement or multi-game learning is already complete.
3. Define a bounded Phase 6 evaluation protocol: intended game roles, separate
   versus shared baselines, per-seat outcomes/rewards, matched compute accounting,
   held-out evaluation seeds and later transfer/forgetting measurements. Distinguish
   what this milestone measures from experiments reserved for later milestones.
4. Explain the proposed Connect Four observation/action encoding, model, rewards,
   collection/update boundaries, checkpoint/resume contract and evaluation budget.

Then implement the first bounded milestone: an independent Connect Four PPO
baseline and human play against its saved checkpoint. The checkpoint opponent is
a small brought-forward Phase 8 task, not completion of Phase 8.

Requirements:

- Use the existing GameEngine, Agent, Observation, Action and StepResult interfaces.
  Keep rules independent of PyTorch. Reuse suitable PPO infrastructure without
  changing existing games' training behavior or checkpoint formats.
- Encode only public structured observations. Use player-relative board features,
  seven column outputs and supplied legal-action masking. Decode to a supplied
  legal Action. Specify inactive/terminal encoding and zero terminal bootstrap.
- Start with a small feed-forward policy/value model; memory is unnecessary for
  this fully observed game. Agents/models never receive an engine, authoritative
  clone, RNG, seed or debug state as input.
- Train through latest-policy self-play only. Keep per-seat rewards/GAE correct,
  with private reproducible randomness and no global RNG contamination. Evaluation
  games, baseline opponents and tactical labels must not enter optimizer batches.
- Provide a bounded training command, portable validated checkpoints, update-boundary
  resume, configuration/metrics and frozen evaluation versus random in both seats.
  Reject incompatible checkpoints without damaging existing outputs.
- Add human play against a loaded Connect Four model using an explicit checkpoint
  option. Preserve the current board, separator and aligned column labels 1–7,
  retries, cancellation, final outcomes and existing random-opponent behavior.
  Human input is 1–7; engine/model actions remain zero-based columns 0–6.
- Test encoding perspective, masks/decoding, full-column exclusion, reward ownership,
  terminal handling, legal complete games, reproducibility/global RNG preservation,
  meaningful PPO updates, checkpoint rejection/resume and human checkpoint play.
  Validate CPU; test CUDA when available without promising cross-device identity.
- Predeclare a small functional training/evaluation run and execute it. Report
  counts and outcomes honestly; functionality does not establish robust strength.
  Keep larger strength, shared-model and transfer experiments for later milestones.
- Do not add a shared backbone, multi-game sampler, new game, search, MuZero,
  replay learner or additional Tic-Tac-Toe tuning in this milestone.
- Make routine design choices autonomously. Run relevant development checks and
  the full regression suite. Existing Windows temporary-file/hard-link tests may
  require an approved unrestricted pytest run; do not weaken tests to bypass it.
- Update existing documentation concisely with established behavior, measurements,
  limitations and the next small task. Leave the new changes uncommitted.

At completion, report the implemented subset and remaining Phase 6 tasks, verified
checks, functional-run results and exact commands to train, resume and play against
the checkpoint. Do not claim Phase 6 complete until its full acceptance criteria
have measured evidence.
