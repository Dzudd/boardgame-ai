# Connect Four CUDA 100,000-game protocol v1

Declared 2026-10-16. Implementation/configuration milestone; campaign never
executed by Codex. The operator has paused at update 373 (2,984 games);
training and fixed evaluation are not complete. The user runs campaigns in their own console. This is one
training seed for training and human play, not a multi-seed strength study or
Phase 6 completion. The original CPU functional protocol, numeric settings,
results and artifacts remain unchanged.

The canonical [settings](connect-four-cuda-100k-v1.json) have protocol identity
`connect-four-cuda-100k-v1`; an identical packaged copy is validated by the runner.
Changing settings under this identity rejects. A new device or budget requires
a new protocol version and fresh output, never an extension of a consumed cap.

| Setting | Declaration |
|---|---|
| Training output | `runs/connect-four-cuda-100k-v1` (must be absent/empty) |
| Optimization / threads | CUDA device 0 / one Torch CPU thread |
| Collection / evaluation / human inference | Frozen CPU float32 models |
| Initialization, shuffle, self-play root | 631001 |
| Evaluation engine/policy/opponent root | 631002 |
| Control designated-policy root | 631003 |
| Self-play | Exactly 100,000 scheduled games: 12,500 updates × 8 |
| Training cumulative wall cap | 21,600 seconds (6 hours) |
| Evaluation | Initial and fixed update-12,500, sampled against random |
| Evaluation/control games | 256 per seat per stage; 1,536 total |
| Evaluation cumulative wall cap | 600 seconds (10 minutes), separate ledger |
| Evaluation output | Exclusive `evaluation.json` inside the fresh training output |

These training/evaluation seed roots differ from consumed functional roots
630601/630602/630603. No intermediate checkpoint selection, development evaluation,
greedy sweep, teacher labels, reward shaping or replay. The model, encoding,
PPO coefficients, masked sampling and complete-game self-play are unchanged from
the [CPU protocol](../PHASE6_CONNECT_FOUR_PROTOCOL.md). Both seats use the latest
shared policy, frozen throughout collection. Absolute indices schedule the next
game as `completed_updates * 8`; private CPU shuffle state resumes from the file.
Evaluation pairs cases by seat/index with `seat * 256 + index`, keeping matching
engine/opponent seeds for initial/final/control. Paired approximate normal effects
use the same method as the CPU protocol with 256 cases/seat. They describe game
uncertainty for this one training seed, not training-seed robustness.

The six-hour training limit is a fixed operator ceiling for this larger schedule,
not an extrapolation of the tiny CPU run or a completion-time prediction. The
separate ten-minute evaluation limit bounds CPU snapshot loading and all random
matches. Neither cap will be extended automatically. No GPU speedup is promised:
engines, collection and sampling remain sequential on CPU; a small MLP and frequent
CPU snapshots may have substantial transfer/synchronization overhead.

There are 700,000–4,200,000 scheduled training decisions, at most 275,000 optimizer
steps and 8,400,000 sample presentations (two epochs). Evaluation has at most
64,512 decisions. Record actual counts, timings and hardware; these upper bounds
are not execution evidence.

## Device, timing and recovery contracts

New/preflight/resume training resolves the declared device without silent fallback.
Unavailable CUDA or device indices reject before output creation, metrics repair
or ledger changes. New preflight constructs no model and writes no files. It
reports requested/resolved device, device name, Torch/CUDA runtime, CPU inference
placement, scheduled counts and remaining cumulative caps. It checks Git access
in the operator process. Readiness is not a throughput or permission guarantee.

Initialization occurs on CPU using the private schedule, then parameters move to
the declared device before Adam construction. Manifests record requested/resolved
hardware, CUDA runtime, CPU inference placement, source/runtime provenance and
threads. Restore validates portable model/Adam/shuffle state and resumes optimization
on the configured device. Portable trainer snapshots and explicit `load_run(...,
device=...)` can be inspected/restored on CPU or CUDA; the versioned run's training
device stays fixed. Cross-device equality and cross-hardware exactness are not
promised. Tiny same-device continuation is tested, not a guarantee across runtimes.
Source/runtime/settings checks still reject incompatible old run continuation;
old artifacts are never migrated or rewritten. Historical snapshots remain playable.

CUDA synchronizes between bounded PPO minibatches before deadline checks and at
run timing boundaries, including after optimization, before checkpoint publication,
and before normal/error/interrupt budget settlement. The last queued step cannot
be published after its synchronization reveals an expired deadline. If device
synchronization itself fails, the outstanding conservative reservation remains.
CPU arithmetic/scheduling is unchanged. Collection, optimization and publication
metrics remain separate; total ledger time also charges setup, restore, validation,
metrics and unsuccessful work. A single indivisible operation can overrun the cap;
the status records overruns. This is not hard real-time cancellation.

Only a successful complete update is atomically published to `latest.pt`, before
metrics append. Timeout reloads the previous saved boundary on the configured
device. Interrupts/failures preserve that file and settle spent time; normal resume
repeats unsaved games from absolute indices. Metrics tail backups/recovery, gaps,
immutable snapshots and cumulative training/evaluation reservations retain their
existing contracts. One writer only. Hard termination spends the reservation
conservatively; never reset a ledger. Initialization crashes may leave directories
that cannot resume: retain them and investigate. Do not reuse the failed functional
output or either completed functional evaluation root.

A cap stop before update 12,500 is **incomplete**, even if useful learning occurred.
CLI exit 1 reports incomplete target work; Ctrl+C exits 130. Inspect
`training-status.json`, `budget.json`, `updates.jsonl` and the saved counters.
Known discarded complete-game work is reported; partial collector decisions remain
unknown. `final.pt` exists only at the declared final boundary, and protocol
evaluation requires it. `--stop-after` pauses at an absolute update boundary;
resuming consumes the original schedule and remaining time, not a new allowance.
No checkpoint selection or evaluation substitution is allowed after early stopping.

Evaluation loads fresh CPU copies, so it can run on a CPU-only host with the same
compatible source/runtime. Its preflight does not require CUDA. It writes only its
exclusive report and separate evaluation budget, preserving training bytes and RNG.
Partial evaluation is retained and cannot resume or be overwritten in place. A
single run's random-opponent score cannot establish reliable strength; a later
strength study needs at least three predeclared training seeds and fresh final roots.

## Live training progress and existing-run compatibility

Training commands now render one bounded-width progress-bar line in place on
stderr, refreshed at most four times per second. It shows saved-update percentage
and counts, current phase and remaining cumulative time cap. The stop state ends
the line with one newline; final stdout remains JSON. Redirected stderr logs at
most once every 30 seconds plus the final state. Add `--quiet` to suppress progress.
Detailed learning metrics and stage timings remain in `updates.jsonl`. Existing
processes must exit and be restarted with `--resume` to use the new renderer;
never start a second writer. Resume compatibility explicitly accepts the original
CUDA source and its immediately preceding verbose-console source, with exact
digest matching; arbitrary source changes still reject.

The operator's planned pause saved 100 updates / 800 games / 16,143 decisions /
1,102 optimizer steps, with 21.234 seconds charged, no discarded games and no
reported overrun. These are partial saved counters, not a speed benchmark or
strength result. The fixed final checkpoint/evaluation does not yet exist.
Read-only preflight validates that boundary with 99,200 games remaining.

A [byte-pinned compatibility mapping](../../src/boardgame_ai/training/protocols/connect-four-console-compat-v1.json)
permits only the explicitly listed preceding full-source digests to continue on this exact
console-progress implementation. No learning arithmetic, seed schedule, settings,
checkpoint identity or recovery behavior changed. The original saved manifest and
protocol stay intact; the next actual training attempt records its execution
source in status. Other source/runtime/settings mismatches continue to reject.
The current external runbook may evolve; resume validates the original saved prose.

## Exact Windows CMD commands

Run these in CMD from the checkout. The fallback Git path below was verified on
this machine on 2026-10-16; it is not a portable installation path. `set` changes
only the current console. If that file is absent, install/expose Git and rerun
`where git` and `git --version` before starting. No commands below install packages.

```bat
cd /d "C:\Users\micha\Documents\ChatGPT\Board Game AI"
where git
if exist "C:\Users\micha\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe" set "PATH=C:\Users\micha\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd;%PATH%"
where git
git --version
git rev-parse HEAD
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m boardgame_ai info
.\.venv\Scripts\python.exe -m boardgame_ai train --device cuda:0 --check-device
set "CF_RUN=runs/connect-four-cuda-100k-v1"
set "CF_SETTINGS=docs/phase6/connect-four-cuda-100k-v1.json"
set "CF_PROTOCOL=docs/phase6/CONNECT_FOUR_CUDA_100K_V1.md"
```

Read-only new-run preflight:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.connect_four_run "%CF_RUN%" --settings "%CF_SETTINGS%" --protocol "%CF_PROTOCOL%" --check-only
```

Start the fixed schedule (operator only; no extra evaluation):

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.connect_four_run "%CF_RUN%" --settings "%CF_SETTINGS%" --protocol "%CF_PROTOCOL%"
```

For a planned pause, append `--stop-after 100` to the start command (800 games).
Choose either the full start or the pause start, never two writers. Resume:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.connect_four_run "%CF_RUN%" --settings "%CF_SETTINGS%" --protocol "%CF_PROTOCOL%" --resume --check-only
.\.venv\Scripts\python.exe -m boardgame_ai.training.connect_four_run "%CF_RUN%" --settings "%CF_SETTINGS%" --protocol "%CF_PROTOCOL%" --resume
```

After complete training, run the separately capped evaluation once:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.connect_four_run "%CF_RUN%" --settings "%CF_SETTINGS%" --protocol "%CF_PROTOCOL%" --evaluate --check-only
.\.venv\Scripts\python.exe -m boardgame_ai.training.connect_four_run "%CF_RUN%" --settings "%CF_SETTINGS%" --protocol "%CF_PROTOCOL%" --evaluate
```

Play the final portable checkpoint (CPU; writes no artifacts):

```bat
.\.venv\Scripts\python.exe -m boardgame_ai play --game connect-four --checkpoint "%CF_RUN%/final.pt" --policy-mode sampled --agent-seed 42 --starts human
.\.venv\Scripts\python.exe -m boardgame_ai play --game connect-four --checkpoint "%CF_RUN%/final.pt" --policy-mode greedy --agent-seed 42 --starts ai
```

Human games are outside protocol evaluation. If training stops early there is no
final checkpoint; do not pass the run envelope `latest.pt` to human play. Inspect
and resolve the stop using saved status before deciding on a separately versioned
future schedule. These commands do not authorize Codex to run a campaign.

## Development validation (not campaign results)

Latest full local suite: 1,190 passed, one Windows symlink-privilege skip (99.20 s),
including 11 console-progress/compatibility cases. A final deadline check after
console output and before publication, with an additional slow-console test, passed
the focused progress/run/console/device suite: 93 tests (32.77 s). CPU/CUDA tests
cover logging-neutral learning and resume, portable restore, interruption/timeout
protection, synchronized caps, metrics recovery, unavailable hardware rejection
and isolated CPU evaluation. Ruff lint/format, strict mypy, dependency, runner help
and whitespace checks pass. Pytest ran unrestricted from the outset according to
the standing local workflow. Read-only resume preflight validates the operator's
800-game pause without modifying any of the 190 current run files. No campaign,
throughput benchmark or strength study was executed by Codex.

Progress-bar refinement validation: 93 focused CPU/CUDA/run/console tests passed
(32.48 s); lint/format, strict typing and whitespace checks pass. The original and
verbose-console source digests remain explicitly compatible. Read-only resume
preflight validates update 373 / 2,984 saved games after the operator interrupted;
all 190 run-file hashes are unchanged across the refinement. No campaign was run
by Codex. The full-suite result above precedes this renderer refinement.
