# Connect Four functional run: console runbook

For the separate, unexecuted CUDA 100,000-game schedule, use the
[Windows CMD runbook](CONNECT_FOUR_CUDA_100K_V1.md). This document retains the
original CPU functional commands and historical output locations.

Status: the user completed the declared CPU training/evaluation run through a
two-update pause/resume. [Recorded results](CONNECT_FOUR_FUNCTIONAL_RESULTS.md)
include a preceding Git startup failure and all successful-run counts/outcomes.
The operator runs campaigns in their own console. The commands below describe
general usage; do not rerun or overwrite the completed comparison. A read-only
preflight alone is not evidence of completed training or playing strength.

This follows [functional protocol v1](../PHASE6_CONNECT_FOUR_PROTOCOL.md) and its
unchanged [numeric settings](connect-four-functional-v1.json): four CPU/one-thread
updates, eight self-play games per update, 32 training games, 192 separately
scheduled evaluation/control games, and cumulative 120-second caps for training
and evaluation respectively. Do not enlarge these settings in response to scores
or cap exhaustion. Larger learning-quality studies need a separate protocol.

## 1. Prepare and check without training

Open PowerShell at the repository root, using the existing `.venv`. These variables
only prepare arguments; defining them does not start any work. Keep this console
open for subsequent commands. Use an absent/empty output directory.
Git must be available in this same console (`git --version` must work), and the
working directory must be the Git repository root. The runner records revision,
dirty status and source identity. Both preflight and actual new-run startup check
Git before creating output files or initializing a model.

```powershell
$connectFourRunArgs = @(
    '-m', 'boardgame_ai.training.connect_four_run'
    'runs/connect-four-functional'
    '--settings', 'docs/phase6/connect-four-functional-v1.json'
    '--protocol', 'docs/PHASE6_CONNECT_FOUR_PROTOCOL.md'
)
& .\.venv\Scripts\python.exe @connectFourRunArgs --check-only
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; inspect the message before proceeding.' }
```

Expected: `check_only: true`, `ready: true`, CPU, one thread, 32 scheduled training
games, 192 scheduled evaluation/control games, and both remaining caps at 120
seconds. This command creates no run directory, collects no games, initializes no
new trainer and performs no optimization. It checks settings/path compatibility;
it cannot guarantee future file permissions, completion within a cap or strength.

For an existing run, use `--check-only --resume` instead. Resume checks validate
saved snapshots, provenance, counters and metrics without repairing logs, changing
files, or charging time. Training settings bytes, Python/Torch versions and source
digest must match. Finish training and evaluation before changing source/runtime.
Current external prose edits do not replace the run's saved prose.

## 2. Run training yourself, with a saved pause boundary

This command **starts training**. It deliberately pauses after update two so the
same run can demonstrate boundary continuation, with the original total budget.

```powershell
& .\.venv\Scripts\python.exe @connectFourRunArgs --stop-after 2
if ($LASTEXITCODE -ne 0) { throw 'Training stopped early; inspect status and budget before continuing.' }
Get-Content -Encoding utf8 runs/connect-four-functional/training-status.json
```

Expected: `target_reached: true`, `updates: 2`, `complete: false`. Sixteen complete
self-play games are saved. This is an intentional pause, not a failed final run.
`latest.pt` is authoritative; metrics follow checkpoint publication.

## 3. Resume the same run to update four

```powershell
& .\.venv\Scripts\python.exe @connectFourRunArgs --check-only --resume
if ($LASTEXITCODE -ne 0) { throw 'Resume preflight failed; preserve the existing run.' }
& .\.venv\Scripts\python.exe @connectFourRunArgs --resume
if ($LASTEXITCODE -ne 0) { throw 'Training is incomplete; inspect status and budget.' }
Get-Content -Encoding utf8 runs/connect-four-functional/training-status.json
Get-Content -Encoding utf8 runs/connect-four-functional/budget.json
```

Expected: `target_reached: true`, `complete: true`, `updates: 4`, `games: 32`, and a
portable `final.pt`. Decisions and optimizer steps vary with actual game lengths;
the protocol bounds them to at most 1,344 decisions and 88 optimizer steps. Resume
continues absolute game seeds and cumulative time; it does not add another 32
games. Healthy completed resume preserves bytes. A missing final snapshot can be
recreated by resume from the saved final boundary without additional collection.

Alternatively, to run without the deliberate pause, use the argument array with
no `--stop-after` on a new absent/empty output. Choose one route, not two campaigns.
The tests demonstrate exact continuation on disposable tiny settings; this console
sequence alone does not establish uninterrupted/resumed numeric identity.

## 4. Evaluate fixed snapshots separately

Only evaluate after the final training boundary is complete. These commands do
not optimize weights. Evaluation uses the declared initial/final snapshots and
paired random control, without checkpoint selection or a greedy sweep.

```powershell
& .\.venv\Scripts\python.exe @connectFourRunArgs --check-only --evaluate
if ($LASTEXITCODE -ne 0) { throw 'Evaluation preflight failed; inspect the run before proceeding.' }
& .\.venv\Scripts\python.exe @connectFourRunArgs --evaluate
if ($LASTEXITCODE -ne 0) { throw 'Evaluation is incomplete; preserve its partial report.' }
$connectFourEvaluation = Get-Content -Raw -Encoding utf8 runs/connect-four-functional/evaluation.json | ConvertFrom-Json
$connectFourEvaluation.complete
$connectFourEvaluation.summaries | ConvertTo-Json -Depth 8
$connectFourEvaluation.effects | ConvertTo-Json -Depth 8
```

Expected: `complete: true`, 64 cases each for initial/final/control, no errors or
truncations, per-seat W/D/L, scores and lengths, and approximate paired effects.
There is one training seed; the intervals describe game-level uncertainty only.
Even a complete run does not demonstrate robust improvement or multi-game learning.

Evaluation is exclusive: a completed or partial `evaluation.json` is preserved
and cannot be overwritten. There is no mid-evaluation continuation. Do not delete
reports or reset budgets to rerun the same declared comparison.

## 5. Play against the resulting checkpoint

After completing the fixed evaluation, optional interactive human play reads the
portable snapshot without changing training/evaluation artifacts.

```powershell
& .\.venv\Scripts\python.exe -m boardgame_ai play --game connect-four --checkpoint runs/connect-four-functional/final.pt --agent-seed 42
```

Human labels are 1–7; full columns and invalid input retry. `--starts ai` swaps the
human seat. Sampled play is the default; `--policy-mode greedy` is available for
interactive play but does not replace the declared sampled evaluation. Use
`final.pt` or `initial.pt`, not the run-level `latest.pt` envelope.

## Stops, failures and files to inspect

- Exit 0 means the requested operation/absolute pause boundary succeeded. An
  intentional update-two pause is successful even though the full run is incomplete.
- Exit 1 means an incomplete requested boundary/evaluation or an exhausted cap
  reported by preflight. Inspect outputs before taking another step.
- Exit 2 means invalid arguments, incompatible/corrupt artifacts, or another
  validation/I/O failure. Preserve outputs; do not work around identity checks.
- Exit 130 means Ctrl+C. When interruption reaches the run's update loop, elapsed
  time is settled and `training-status.json` records the interruption. Resume uses
  the last saved boundary. A hard kill retains a conservatively spent reservation;
  initialization interruption can leave an incomplete directory that rejects resume.
- Never reset `budget.json`, overwrite snapshots or manually repair metrics. The
  runner validates recovery and backs up recoverable torn tails; gaps remain gaps.

### Git unavailable in the operator console

The original startup path could report only `[WinError 2]` if Codex could run Git
but the operator's console could not find it. It could leave `settings.json`,
`protocol.md`, `initial.pt` and a reserved budget before failing, with no manifest,
latest checkpoint, self-play or optimizer steps. This was reproduced and fixed:
Git/repository failures now identify the dependency and reject before output.

Make an installed Git executable available on PATH in the console that will run
Python, then verify `git --version`. Codex's own tool PATH does not guarantee the
same tools exist on the operator's PATH. Preserve any old incomplete setup without
deleting it or resetting its reserved budget. A directory without `training.json`
and `latest.pt` is not resumable. After fixing the dependency, use a clearly named
new output for a fresh attempt and keep the failed startup as evidence; do not
present it as a successful resume or silently overwrite it.

Keep `training.json`, `settings.json`, `protocol.md`, `budget.json`,
`training-status.json`, `updates.jsonl`, all recovery backups, and `evaluation.json`.
The checkpoints stay in the same run directory. To review results with Codex,
provide the run path and console errors/output; no upload of model weights is
needed when the workspace is shared. Report partial counts and failures as well as
successful results. Published strength conclusions require a later study.
