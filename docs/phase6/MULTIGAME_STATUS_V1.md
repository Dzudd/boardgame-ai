# Phase 6 pre-execution implementation audit

**Historical snapshot below (2026-10-17).** The operator subsequently completed
all 36 jobs and evaluations. [Results reviewed 2026-10-18](MULTIGAME_RESULTS_V1.md)
supersede the pending counts and task states in this original audit. Phase 6's
declared comparison is complete with mixed/negative findings and one uncertain
wall-cost measurement; positive transfer was not established.

2026-10-17. The frozen [protocol](MULTIGAME_PROTOCOL_V1.md),
[settings](multigame-v1.json) and [CMD runbook](MULTIGAME_RUNBOOK_V1.md) are ready.
The read-only report finds **0 of 36 jobs evaluated** and correctly says
`awaiting operator results`. The fresh output root does not exist. No campaign
was launched. Existing 200 `runs/` files retain their pre-session SHA-256 hashes.

| Requested task | Implemented/validated | Demonstrated learning quality |
|---|---|---|
| Shared observation encoding | All three development games plus unseen C; visible histories/relative boards | Pending |
| Candidate encoding/scoring | Variable supplied tuples, permutation-equivalent logits, legal tuple selection | Pending |
| Shared backbone | Common GRU, complete own-seat streams, full BPTT | Benefit of sharing/memory pending |
| Game adapters | Observation/candidate/query/value per game, own-return perspective | Pending |
| Sampler/scales/metrics | Latest-policy complete games, equal-game losses, Guildworks /36, other returns /1 | Pending |
| Separate versus shared | Independent matched-capacity models, wall and decision-ceiling regimes | Costs and per-game strength pending |
| Unseen transfer | Frozen Breakthrough 5x5, paired untouched adapters and reset Adam | Adaptation curves/speed pending |
| Negative transfer/forgetting | Shared-minus-separate and old-game post-minus-pre C scores, seed intervals | Effects pending |
| Phase acceptance evidence | Isolated evaluation and read-only reporting ready | **Not met** |

The model allocates **29,572 parameters**, including all four adapters. Separate
models each allocate that count (88,716 across the three-model set); inactive
adapters have no gradients/Adam history. Parameter count is not strength evidence.
Training caps total 21,600 seconds; evaluation caps separately total 32,400 seconds.
Actual time, incomplete jobs, accepted/attempted samples and cost differences
must replace allowances in results. No completion or stronger-play prediction.

## Functional evidence

The 25 disposable Phase 6 tests exercise an independent Breakthrough transition
oracle and engine conformance, candidate reordering/validation, own-seat reward
ownership, latest-weight/RNG isolation, hidden-world-equivalent Guildworks inputs,
meaningful shared-weight PPO updates, stale-data rejection, exact CPU pause/resume,
portable CUDA continuation, corrupt tensors, exclusive evaluation, matched C
initialization/reset Adam, torn metrics recovery, checkpoint-publication failure,
interrupt progress cleanup, evaluation-publication failure, and wall-cap rollback.
They establish operation and recovery, not strength or adaptation speed.

CUDA is available on the RTX 4070 Ti with Python 3.12.14/Torch 2.14.1+cu130.
The CUDA device check, Ruff lint/format, strict mypy and dependency check pass.
Final full suite: **1,216 passed, 1 Windows symlink-privilege skip, 83.69 seconds**.
Pytest was
requested outside the sandbox from its first invocation, as local guidance requires.

Existing Connect Four source files are byte-unchanged by this increment. The
exact source-compatibility map retains prior reviewed sources and identifies the
seven added Python files, allowing existing saved runs to resume without replacing
their provenance. Read-only preflight accepts both operator CUDA boundaries:
3,743 updates/29,944 games and 7,287 updates/58,296 games. The copied human-play
snapshot is preserved. None of these artifacts is evidence for shared learning.

## Operator evidence still required

Execute the frozen schedule and isolated evaluation, then use `--report` for the
results review. Required evidence is per-game strength and errors by seat/seed;
aggregate wall cost versus summed independent cost; realized environment/sample
cost differences; complete C learning curves and target crossings/censoring;
negative-transfer effects; and retained A/B/D scores after C adaptation. Report
failed hypotheses, weak play, overrun, gaps and incomplete/censored runs honestly.
Missing evidence must not be replaced with losses, sample volume, parameter count
or GPU utilization. Phase 6 remains open; later phases were not expanded.
