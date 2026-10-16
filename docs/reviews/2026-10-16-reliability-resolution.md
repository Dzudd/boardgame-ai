# Bounded reliability milestone

## Recovery contract (specified before implementation)

The validated `latest.pt` is authoritative. Read metrics as bytes and validate
all complete records before changing anything. Update indices must be positive,
strictly increasing and no newer than the checkpoint; gaps are permitted but
make cumulative sample counts unknown. Sample counts must be nonnegative integers.
Reject malformed newline-terminated records, malformed earlier records, invalid
record schemas, duplicate indices and history ahead of the checkpoint.

Only a final record without a newline that is a syntactically valid, unfinished
JSON prefix may be removed. Preserve the entire original log byte-for-byte in
an exclusively published recovery file identified by its SHA-256, then atomically
replace the log with its valid prefix. A complete final record without a newline
is retained and gets a newline using the same backup-before-replacement process.
No rows are invented. Missing checkpoint rows leave cumulative samples unknown.
The checkpoint, Adam, shuffle state and absolute game seed schedule are untouched.
Recovery is single-writer, like training; retries reuse only identical backups.

## Resolutions

Revalidated against branch `codex/documentation-reorganization`, commit
`e2b896d2fb94892ed78aa1bb260d87fb26be3485`. All three historical findings still
applied. The starting tree contained user README edits and untracked review
documents; their content was retained. The historical audit is unchanged.

1. **Evaluation protection resolved.** A shared destination validator serves
   CLI preflight and `save_policy_evaluation`. Checkpoints (`*.pt`), training and
   monitoring manifests, best pointers, development reports and snapshot paths
   reject even when absent. All existing destinations reject, covering resolved
   aliases and hard-linked identities. The write boundary uses exclusive creation;
   there is no overwrite flag. Rejection preserves existing bytes.
2. **Interrupted metrics recovery resolved for Tic-Tac-Toe.** The contract above
   is implemented before monitoring or further updates. Tests cover malformed
   earlier/terminated rows, invalid schemas/order, every truncation of an emitted
   representative record, valid records without final LF, backup/publication
   failure and retry, unknown monitoring counts, and exact uninterrupted versus
   recovered model/Adam/shuffle/future-row equivalence. No lost row is regenerated.
3. **Protocol identity resolved.** Frozen machine settings include explicit PPO
   parameters, budgets, seeds, monitoring, mode, schedule and selection identity.
   Their source digest is pinned; changing settings requires a new version.
   New experiments save settings and original Markdown separately with hashes.
   Resume verifies saved provenance, independently of current explanatory prose.
   The read-only `--validate-only` path never invokes an experiment.

The original Phase 4.5 protocol bytes were recovered with
`git show e56dd90:PHASE45_PROTOCOL.md`. Their SHA-256 is
`9d55a0f8ae4c4a34fb69d7ce44508139390d5b9290fd846a79f8ef0b706f834f`,
exactly the recorded manifest identity. Inspection of that protocol and the
historical runner establishes the frozen settings: three seeds 42/43/44, 200 ×
64 games, default PPO versus entropy .05, default monitoring, CPU/one thread,
sampled final evaluation at root 20261029 with 512 games per case, equal-case
development selection and paired t(2) comparison. The documentation reorganization
added explanatory text/links without changing these declarations.

Only this exact historical manifest is mapped to the frozen settings; every field
and the archived snapshot hash are checked. Original manifest/hash provenance is
retained in results and never relabelled. Tests validate a disposable full copy of
the recorded `runs/phase45/entropy-comparison` tree and verify its bytes unchanged.
Unknown historical hashes, changed settings, damaged saved snapshots, unsupported
schemas and type substitutions reject. The validation path certifies protocol
compatibility only, not checkpoint/report integrity or cross-runtime equivalence.

## Validation

The first focused run passed **90 tests**. Final full suite: **874 passed,
1 skipped in 51.44 seconds**, including available CUDA regressions. Ruff
lint/format, strict mypy (116 files), `pip check`, CLI info/help and
`git diff --check` pass. The skip is the Windows symlink-creation privilege test;
hard-link, resolved-path and exclusive-creation race tests pass.

All **168 existing files under `runs/`** retain their hashes across final
validation. The three principal checkpoints also match the original audit hashes.

The initial sandbox attempt failed at temporary-directory setup (26 passed,
64 setup errors); an approved unrestricted rerun passed. No tests were weakened.
The symlink-specific test requires Windows developer mode or link privilege.
No hosted CI is claimed.

## Changed files

- `artifact_safety.py`, `cli_evaluate.py`, `policy_evaluation.py`: shared protection
  and exclusive report creation.
- `training/metrics_log.py`, `training/run.py`: validated recovery and resume wiring.
- `training/experiment_protocol.py`, `training/experiments.py`,
  `training/protocols/phase45-v1.json`, `training/protocols/phase45-original.md`:
  frozen settings, retained provenance, compatibility and read-only CLI.
- `.gitattributes`, `pyproject.toml`: byte preservation and packaged protocol data.
- `tests/test_reliability.py`: failure-path, identity, recovery and compatibility
  regressions, including a disposable copy of the recorded experiment.
- `README.md`, `ARCHITECTURE.md`, `PHASE45_PROTOCOL.md`, `PROGRESS.md`,
  `BOARDGAME_AI_PROJECT_PLAN.md`, and this resolution record: usage, contracts,
  evidence and next proposal. Source paths above are relative to `src/boardgame_ai/`.

## Limits and remaining work

Recovery retains the existing single-writer assumption and requires same-directory
atomic replacement and hard-link publication for complete immutable backups.
It handles process-interrupted appends, not arbitrary disk corruption or recovery
of missing metrics. Guildworks' separate runner is unchanged. A new v2 experiment
interrupted during initial manifest creation requires manual inspection/new output;
this milestone does not add directory-level transactional initialization.

No original experiment was resumed, no lengthy training or evaluation campaign
was launched, and no learning algorithm, game, reward, PPO arithmetic, model,
visibility boundary or latest-policy-only sampling rule was changed. Tiny tests
use disposable artifacts. Phase 6 and all other unfinished roadmap tasks remain
unfinished. Changes remain uncommitted and unpushed.

## Proposed next bounded learning experiment (not executed)

**Question:** does a small public-rule Tic-Tac-Toe search-target learner improve
external both-seat performance over independent latest-policy PPO, under a frozen
budget? This requires a separately authorized bounded implementation of the
already designed learner; it is not a proposal to add shared weights or MuZero.

Use the unchanged independent PPO learner as the trained baseline, and fixed
random-v1/heuristic-v1 as evaluation controls. Both experimental training seats
use only the current frozen shared policy/search procedure; no baseline games,
tactical labels or historical opponent pool enter learning. Evaluate the learned
raw policy as primary; report any search-assisted play separately with latency.

Use at least three paired training seeds and declare every seed before execution.
Use fixed development roots and unused fresh final roots distinct from the
consumed Phase 4/4.5/7 roots; freeze checkpoint selection before final evaluation.
Report score, W/D/L, errors and uncertainty separately for both seats and both
opponents, plus self-play coverage and every training seed. Primary outcome is
the paired training-seed difference in equally weighted external case score;
report seed-level uncertainty separately from within-case game uncertainty.

Before choosing numerical budgets, measure short, separately capped throughput
for the new learner/search budget and PPO on the target runtime. Historical PPO
timings do not establish search-target throughput. Freeze games, environment
decisions, search simulations, replay reuse, optimizer work, evaluation games and
wall-time caps from those measurements and available time before opening outcomes.
Use equal self-play game budgets as the primary comparison and report unequal
optimizer/search costs explicitly; add a compute-matched comparison only if its
budget can be declared in advance. No numerical training budget is approved here.

Stop at the frozen cap; do not extend a weak result. Stop immediately on boundary,
provenance, nonfinite-value or evaluation-isolation failures. Advance only if the
paired primary interval excludes zero positively and predeclared per-seat
heuristic regression limits are met; freeze those limits before execution.
Otherwise report failure/inconclusive evidence and defer shared-model expansion.
