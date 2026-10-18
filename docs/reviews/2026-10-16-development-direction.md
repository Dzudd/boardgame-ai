# Development direction after the repository audit

Date: 2026-10-16.

This records the reasoning discussed after the
[audit and feasibility assessment](2026-10-16-audit-and-feasibility.md).
It is a recommendation, not a change to phase completion status or authorization
to execute experiments.

## Judgment

The project makes sense and is worth continuing. Its software platform is further
along than its demonstrated learning strategy. Keep the engine ownership,
filtered observations, per-player rewards, and separate evaluation boundaries.
The audit identified repairable reliability defects, not a reason to rewrite
the system.

The main risk is building increasingly sophisticated infrastructure while agents
remain weak. Tic-Tac-Toe already demonstrates that correct training execution can
produce narrow self-play behavior that performs poorly against a different
opponent. More games, more parameters, or more training do not automatically
resolve that problem.

Three ambitions need separate evidence:

1. A common platform supporting multiple implemented games is realistic and
   partly achieved.
2. One model learning several games and transferring useful knowledge is a
   plausible research goal. It requires comparison against independent models.
3. Playing arbitrary unseen games from written rules is outside the current
   architecture. Game adapters and rules still have to be implemented.

Do not make MuZero the next implementation step. Exact simulators already exist
for the supported games; learned dynamics would add another source of error
before robust search-guided learning has been demonstrated. MuZero remains a
valid later research comparison, not a necessary prerequisite for progress.
See the [MuZero paper](https://arxiv.org/abs/1911.08265) for its learned-model
formulation; this sequencing recommendation is specific to this repository.

## Recommended order

1. Fix artifact-overwrite protection, interrupted-log recovery, and historical
   protocol compatibility. Preserve all existing experiment evidence.
2. Define a bounded learning-quality experiment with multiple training seeds,
   both seats, external evaluation, declared budgets, and fresh final seeds.
   Require measurable competence outside self-play, not perfect play.
3. Establish independent-model references before comparing a small shared model.
   Report per-game regressions, adaptation speed, and compute costs.
4. Investigate hidden-information search and learned dynamics separately, with
   their own contracts and acceptance criteria.

The next execution prompt therefore addresses reliability first. This is a
prerequisite milestone, not completion of roadmap Phase 6. The following learning
milestone still needs a protocol decision; neither a large shared backbone nor
an extended training campaign is implicitly approved.

## Next-session prompt

Copy the following into a new session in this repository:

```text
Implement a bounded reliability milestone before further learning development.

First read AI/INDEX.md and AI/AGENTS.md if present, then README.md,
BOARDGAME_AI_PROJECT_PLAN.md, ARCHITECTURE.md, INFORMATION_BOUNDARIES.md,
PROGRESS.md, docs/reviews/2026-10-16-audit-and-feasibility.md, and
docs/reviews/2026-10-16-development-direction.md. Read relevant source, tests,
and experiment protocols before editing. Inspect the current branch, commit,
and working tree; preserve existing user changes. Revalidate the findings
against the current checkout rather than assuming historical line numbers
or behavior remain unchanged.

Fix these three confirmed audit findings:

1. Evaluation artifact safety.
   Centralize protection against evaluation output overwriting checkpoints,
   training/monitoring manifests, best-checkpoint pointers, development reports,
   and retained snapshots. Prefer exclusive creation for new reports. If an
   intentional report-overwrite option is retained, scope it explicitly and
   do not let it bypass managed training-artifact protection. Cover resolved
   path aliases and existing-file identity where applicable. Preserve source
   artifacts byte-for-byte on rejection.

2. Resume after an interrupted metrics append.
   Allow recovery from a valid authoritative checkpoint when updates.jsonl
   has an incomplete final record. Distinguish that case from corruption in
   earlier records. Preserve the original damaged bytes in a recovery artifact,
   retain valid history, mark unknowable cumulative counts as unknown, and
   establish a safe append boundary. Recovery must not silently fabricate rows,
   duplicate updates, change sampling schedules, or modify model/Adam/shuffle
   state. Specify and test recovery behavior before wiring it into resume.

3. Stable experiment protocol identity.
   Separate immutable machine-readable experiment settings from editable
   explanatory Markdown, and retain the original protocol snapshot/hash as
   provenance. Design an explicit, verified compatibility path for historical
   manifests. Do not simply ignore hash mismatches or replace old hashes with
   current ones. If historical equivalence cannot be established, reject with
   actionable instructions. Verify compatibility on disposable copies of the
   recorded Phase 4.5 artifacts, never by rewriting the originals. Provide a
   read-only validation path that does not launch training or evaluation.

Keep this milestone small. Preserve current game rules, rewards, PPO arithmetic,
model architectures, information boundaries, and latest-policy-only self-play.
Do not implement Connect Four learning, a shared backbone, search-target learning,
MuZero, opponent pools, or new game mechanics in this task. Do not launch lengthy
training or resume any existing experiment. Tiny synthetic runs and disposable
artifact fixtures are appropriate for verification.

Add meaningful regression tests for each failure scenario and for uninterrupted
versus recovered training equivalence. Run focused tests, then the repository's
relevant full validation checks where feasible. Distinguish environment failures
from code failures and report any unverified claims.

Update the affected usage/contracts and PROGRESS.md to describe the actual fixes
and remaining work. Keep the historical audit intact; link a concise resolution
record containing test evidence. Do not mark Phase 6 or other unfinished phases
complete. Preserve original run artifacts, checkpoints, logs, and protocol
manifests. Leave changes uncommitted; do not push.

Finish with findings resolved or remaining, files changed, validation results,
compatibility limitations, and one proposed next learning milestone. That proposal
should identify a falsifiable question, baselines, both-seat metrics, seed and
budget requirements, and stopping criteria. Do not execute that experiment or
invent numerical budgets without a throughput basis.
```
