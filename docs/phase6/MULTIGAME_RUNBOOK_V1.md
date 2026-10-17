# Phase 6 operator commands v1

**Study completed and reviewed 2026-10-18:** all 36 jobs/evaluations exist.
[Results](MULTIGAME_RESULTS_V1.md) include mixed/negative findings and recovered
timing uncertainty. Commands below document the original schedule and recovery;
do not rerun/overwrite this evaluated study. Use `--report` to inspect its evidence.

Run from the repository root in Windows CMD with the existing virtual environment.
These commands launch the frozen [protocol](MULTIGAME_PROTOCOL_V1.md); Codex has
executed only disposable tests and read-only checks. Every command is one line.
The root below is fresh; preserve it after the first attempt. Do not run jobs
concurrently. Training caps total at most six hours, plus separately capped CPU
evaluation. Completion/stronger play are not promised.

Check Git from this CMD console, CUDA and protocol readiness:

```bat
git --version
.\.venv\Scripts\python.exe -m boardgame_ai info
.\.venv\Scripts\python.exe -m boardgame_ai train --device cuda --check-device
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --all --check-only
```

If CMD cannot find Git, fix its PATH before starting. The Codex environment's
bundled Git path is not a portable promise; `git --version` must work in this
console. Preflight validates the new root without creating models or files.
List all 36 fixed jobs and their caps/dependencies:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --list-jobs
```

Start all training jobs in dependency order; this is the operator campaign:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --all
```

Ctrl+C retains the previous update and settles attempted time. Resume existing
jobs and start subsequent absent jobs with the original cumulative caps:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --all --resume
```

For a deliberate bounded pause, select one job and an absolute update count.
This example pauses the first shared job after two saved updates; it is part of
that job's budget, not an extra study. Run it instead of the initial `--all`:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --job wall-661101-shared --stop-after 2
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --job wall-661101-shared --resume --check-only
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --all --resume
```

After training completes, validate and launch the isolated fixed evaluation:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --all --evaluate --check-only
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --all --evaluate
```

Repeating `--all --evaluate` preserves validated complete reports and starts only
missing ones. Partial reports reject; preserve and investigate them. Hard-kill
reservations remain spent. Do not remove reports, reset caps or overwrite results
to obtain a better outcome. There is no evaluation resume within a job.

The reporting command is read-only and works even while results are missing:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --report
```

It emits JSON with availability/rejections, per-seat W/D/L, per-game shared versus
separate effects and costs, attempted versus accepted decisions, transfer curves,
censoring, speed ratios and retention effects. It never marks the roadmap complete.
Keep all job directories for the results review. Final acceptance still needs
operator evidence on per-game strength, training cost and unseen adaptation speed.

`--quiet` suppresses the existing throttled single-line progress renderer.
Displayed progress is a cost fraction (0–100), not an estimate of games or strength;
only durable saved boundaries advance it. Detailed per-game metrics remain in
JSONL. Final stdout is JSON; live progress is stderr. A cap/failure stops `--all`
at that job with a nonzero exit. Complete training means the declared budget
boundary was reached, not that a learning-quality threshold was met.

## Windows access-denied recovery at a wall-cap boundary

2026-10-17: operator job `wall-661102-shared` preserved 2,066 updates at
899.86 seconds, then Windows denied atomic replacement of `budget.json`.
The 900-second reservation remained spent and finalization did not publish.
The first six jobs completed. Existing checkpoints and budgets were not edited
during diagnosis or development; the failure's precise locking cause is unknown.

Atomic JSON/checkpoint publication now retries Windows errors 5/32/33 up to six
attempts (1.55 seconds total backoff). Persistent failures still raise and preserve
the old destination. Learning arithmetic, schedules, settings and caps are unchanged.
An exact source map permits the recorded preceding implementation to resume,
without rewriting its manifest, source archive or saved protocol.

An explicit `--finalize-capped` command can preserve and finalize a nonzero saved
wall checkpoint within min(5 seconds, 1%) of its spent cap. It collects no games,
does no optimization and never changes the ledger. It backs up original `latest.pt`,
writes an immutable receipt and publishes p100 with exactly the saved trainer.
Early checkpoints, zero-update states, unspent budgets and sample jobs reject.
The receipt flags timing uncertainty: a reservation is not exact measured wall
time, so the report cannot assert a matched wall-cost comparison for that job.
This is explicit recovery, not automatic reclassification of hard-killed jobs.

Read-only eligibility check:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --job wall-661102-shared --finalize-capped --check-only
```

After the failed process has exited, this single CMD line fixes that console's
Git PATH, finalizes job 7 and resumes the remaining campaign. Completed jobs are
validated and skipped. This command starts further training after recovery:

```bat
set "PATH=C:\Users\micha\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd;%PATH%" && .\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --job wall-661102-shared --finalize-capped && .\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --all --resume
```

Codex ran only disposable tests and read-only preflight. The operator applies the
recovery and launches continuation. Keep all backup/receipt files with the study.
