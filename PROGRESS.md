# Current progress

Updated: 2026-10-18. This is the default task-selection entry point.
Goal and acceptance criteria: [roadmap](BOARDGAME_AI_PROJECT_PLAN.md).

## Current state

- Reusable engine interfaces and Tic-Tac-Toe, Guildworks, Connect Four and
  Breakthrough implementations exist; generic new-game training is not yet proven.
- Independent PPO, recurrent/shared learning, checkpoint/resume and some CUDA
  optimization exist. Models and collection remain small or specialized; shared
  collection uses CPU policy copies and public search learning is CPU-only.
- Large configurable models, batched GPU collection and an efficient sustained
  end-to-end training pipeline are not delivered.
- Earlier experiments established functionality and bounded comparisons, not
  robust general strength. Shared transfer and public search comparisons include
  negative/inconclusive findings. See [evidence index](docs/archive/README.md) only
  when those results affect the task.

## Next task — audit the reusable game-to-learner contract (milestone A)

Trace one complete path from game registration through observation/action encoding,
collection, model construction, training, checkpoint restore, evaluation and play.
Identify hard-coded game IDs, shapes, player counts and reward assumptions that
prevent adding a game through a plugin. Inspect relevant source/tests and only the
matching architecture sections; do not read all historical studies first.

**Deliverable:** a concise gap map tied to source locations and the smallest next
implementation slice, recorded here in place of this task when finished. Define
which mechanics the plugin contract supports and which require an extension.
Do not redesign every learner or launch a training campaign during this audit.

**Done when:** the required plugin responsibilities and shared-training changes are
concrete enough to implement, with acceptance tests identified. Then replace this
section with exactly one implementation task. Hardware/capacity calibration follows
under milestone B; it must not substitute for the integration contract.

## Constraints and open limits

- Train only through self-play; fixed opponents/labels are for isolated evaluation.
  Respect [information boundaries](INFORMATION_BOUNDARIES.md).
- Initial production capacity target: at least 10M parameters, subject to measured
  GPU memory/throughput and an explicit agreed revision if infeasible.
- The user runs campaigns in their console. Preserve existing runs, checkpoints,
  frozen protocols, source compatibility and evidence. New studies use fresh outputs.
- Prepared Phase 7 v2 and paused CUDA campaigns remain available; they are not the
  default next task. No new training evidence is claimed by this documentation edit.

Detailed previous status/validation history is [archived](docs/archive/2026-10-18-PROGRESS.md).
