> Command reference. Run commands from the repository root. Read only the needed
> section; historical study recipes retain their original scope and caps.

## Command line

```powershell
.\.venv\Scripts\boardgame-ai.exe --help
.\.venv\Scripts\boardgame-ai.exe info
.\.venv\Scripts\python.exe -m boardgame_ai info
```

The CLI supports `info`, `play`, `replay`, `train`, `evaluate`, and `analyze`.
`info` reports Python/NumPy/PyTorch versions and CUDA availability; it does not
select the training device. Individual commands expose `--help`.

## Continuous integration

[The CI workflow](../../.github/workflows/ci.yml) runs lint, formatting, strict typing,
tests, dependency checks, and CLI smoke checks on Python 3.12 for Windows and
Linux with CPU PyTorch. It runs on pushes, pull requests, and manual dispatch.
[GitHub Actions](https://github.com/Dzudd/boardgame-ai/actions) holds hosted results;
local validation does not imply a hosted run passed.

## Human-versus-agent play

```powershell
.\.venv\Scripts\boardgame-ai.exe play
.\.venv\Scripts\boardgame-ai.exe play --game tic-tac-toe --starts ai
.\.venv\Scripts\python.exe -m boardgame_ai play --game tic-tac-toe --opponent random --agent-seed 42
.\.venv\Scripts\python.exe -m boardgame_ai play --help
```

Tic-Tac-Toe is the default game. `--starts human` is the default; `--starts ai`
assigns the first seat to the opponent. Tic-Tac-Toe's starter is X. Opponents are
`random` or `heuristic`; the heuristic is deterministic, takes immediate wins,
blocks immediate threats, then prefers center/corners/edges. It is not unbeatable
and does not handle every fork. `--agent-seed` controls random play independently
of `--engine-seed`; Tic-Tac-Toe setup itself does not depend on the seed.

Enter row-major cells 0–8. The board appears beside available cell numbers;
occupied cells show dashes. Invalid text/occupied cells retry without advancing
play. The final board shows win/loss/draw. Enter `q`, `quit`, or `exit`, send EOF,
or press Ctrl+C to stop without declaring an outcome. Replay saving is opt-in.

### Tic-Tac-Toe

The programmatic engine also supports O-first play, independent cloning, public
observations, and terminal rewards +1/−1/0. See the
[development examples](../../docs/DEVELOPMENT.md#engine-lifecycle) and
[implemented contract](../../ARCHITECTURE.md#tic-tac-toe-and-baseline-agents).

## Connect Four

The independent bounded PPO run/resume/evaluation layer is available through a
[separate module command](../../docs/DEVELOPMENT.md#connect-four-bounded-runs). Its
declared functional protocol is complete; [results](../../docs/phase6/CONNECT_FOUR_FUNCTIONAL_RESULTS.md)
record a final balanced score of .65625 against random in one tiny run. Shared
learning and robust strength evidence remain pending.
CUDA optimization is available with explicit device-aware preflight/restore and
synchronized cumulative timing. Training CLI commands show one in-place progress bar
on stderr with saved-update percentage/count, current phase and remaining cap.
Refresh is limited to four times/second; redirected output logs at most once every
30 seconds plus a final summary. The final JSON stays on stdout; `--quiet`
suppresses progress.
The separate
[100,000-game settings and Windows CMD runbook](../../docs/phase6/CONNECT_FOUR_CUDA_100K_V1.md)
provide fresh seeds/output, six-hour training and ten-minute evaluation caps, and
exact preflight/start/resume/evaluate/play commands. The operator has paused that campaign after 2,984 saved games;
training and final evaluation are not complete. CPU inference remains sequential; GPU speedup and reliable
strength from one run are not promised.
Human checkpoint play is available with `--checkpoint`.
Use the [console runbook](../../docs/phase6/CONNECT_FOUR_CONSOLE_RUNBOOK.md) for a
read-only `--check-only` preflight and the train/resume/evaluate commands to run
in your own PowerShell console.

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai play --game connect-four
.\.venv\Scripts\python.exe -m boardgame_ai play --game connect-four --starts ai --agent-seed 42
.\.venv\Scripts\python.exe -m boardgame_ai play --game connect-four --checkpoint runs/connect-four-functional-retry1/final.pt --agent-seed 42
.\.venv\Scripts\python.exe -m boardgame_ai play --game connect-four --checkpoint runs/connect-four-functional-retry1/final.pt --policy-mode greedy --starts ai
```

Checkpoint examples use the completed local run; in a fresh clone, create or
restore a compatible snapshot first. `--checkpoint` accepts portable
trainer snapshots (`initial.pt`, `final.pt`, or programmatically saved files),
and rejects run-level `latest.pt`. It cannot be combined with `--opponent`.
`--policy-mode sampled` is the default; `greedy` picks the highest legal logit
with lowest-column ties. For checkpoint play, `--agent-seed` must be an unsigned
64-bit integer; repeating it reproduces sampled choices for the same human moves.
Loading validates the whole snapshot on CPU before human input and writes no files.

The CLI shows a 7-column, 6-row board with aligned column labels 1–7. Enter a
column to drop a disc; full columns and invalid input retry. X starts; four in a
horizontal, vertical, or diagonal line wins; a full board without a winner draws.
Quit, EOF, and Ctrl+C cancel. The default opponent is random.

`ConnectFourEngine` and `ConnectFourObservation` support the shared `run_match`
interface with `max_steps=42`; `connect_four_outcome` adapts results. Engine/model
columns are zero-based 0–6; human input is 1–7. Search and replay export
remain unimplemented. Public observation and supplied-column
encoders are available from `boardgame_ai.models.encoders`; their shapes and
semantics are in [architecture](../../ARCHITECTURE.md#connect-four). An independent
policy/value model and complete-game self-play collector are available
programmatically; see the [development example](../../docs/DEVELOPMENT.md#connect-four-self-play).
Programmatic PPO updates and portable trainer checkpoints are also available.
The bounded CPU run command provides run-level resume, protocol provenance,
metrics recovery, cumulative caps and isolated evaluation. The user completed the
declared functional training/evaluation run; larger learning-quality studies remain.

## Guildworks

Four rounds, two workers per player per round. Place on a free public space for
ink, cards, points, or a resolved ink roll, then craft one affordable hidden card
or pass. Scores determine the winner after 32 decisions. Hands/deck remain private;
resources, scores, discard, and counts are public. Full
[rules](../../ARCHITECTURE.md#guildworks-rules-and-transitions) and
[visibility requirements](../../INFORMATION_BOUNDARIES.md#guildworks-player-views-and-history)
define chance, rewards, and history.

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai play --game guildworks
.\.venv\Scripts\python.exe -m boardgame_ai play --game guildworks --starts ai --engine-seed 42 --agent-seed 100
```

Select a numbered action at each placement and craft/pass prompt. The display
shows your hand and public information; opponent messages show public actions
only. `--starts` assigns the first seat, then the round starter alternates.
Random is the only human CLI opponent. Invalid input retries; quit/EOF/Ctrl+C
cancels. Final scores and win/loss/tie appear after completion. Replay saving is
rejected; hidden-information search is future work. Keep reconstructive engine
seeds in trusted outputs. [Programmatic examples](../../docs/DEVELOPMENT.md#guildworks-matches)
cover random/scripted and recurrent-policy agent use.

Recurrent training has a separate command and checkpoint format:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai.training.guildworks_run runs/new-guildworks --device cpu --threads 1 --updates 16 --games-per-update 16
# Resume to a larger total budget with the same seed and schedule:
.\.venv\Scripts\python.exe -m boardgame_ai.training.guildworks_run runs/new-guildworks --device cpu --threads 1 --updates 32 --games-per-update 16 --resume
```

Default seed is 20261016. Each update collects complete latest-policy self-play
games and optimizes full recurrent seat episodes. Default PPO uses eight episodes
per minibatch, two epochs, and gamma=1. `--device cuda` selects CUDA optimization;
acting stays on CPU. New output must be absent/empty. The run saves `training.json`,
`latest.pt`, and `updates.jsonl`, then initial/final random-opponent evaluation to
`evaluation.json` (default 128 games/seat, seed 20261030). CLI flags configure
budgets/seeds/device; programmatic `RecurrentRunConfig` also accepts `PPOConfig`.
Resume restores model, Adam, and shuffle state at an update boundary. See
[recurrent contracts](../../ARCHITECTURE.md#guildworks-recurrent-learning) and
[functional results and limitations](../archive/PHASE5_RESULTS.md).

## Training and resume

The Tic-Tac-Toe `train` command defaults to CUDA and fails if unavailable; request
`--device cpu` explicitly for CPU. Preflight selects the device and exits without
training or output files:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai train --device cuda --check-device
.\.venv\Scripts\python.exe -m boardgame_ai train --device cpu --output runs/tic-tac-toe-ppo --updates 1000 --games-per-update 64 --seed 42
```

For CUDA, replace `--device cpu` with `--device cuda` or `cuda:1`. If unavailable,
check that this interpreter has a compatible CUDA PyTorch build and driver.
No driver/dependency installation happens automatically. Tests use CPU in CI;
CUDA tests run conditionally. Local machine troubleshooting is separate from
portable setup; historical hardware measurements remain in results documents.

Each PPO update collects 64 complete self-play games using frozen current weights
in both seats, then optimizes all on-policy decisions. Simulation and inference
stay on CPU; optimization uses the selected device. CUDA support does not imply
speedup for this small model. `train --help` lists PPO settings.

Each update prints progress and saves:

- `training.json`: immutable seed, rollout schedule, and PPO configuration.
- `latest.pt`: portable model/Adam/shuffle checkpoint, authoritative for resume.
- `updates.jsonl`: losses/entropy/KL and training self-play outcomes, not independent strength scores.

New output must be absent/empty. Ctrl+C stops a run. Resume with the same seed,
games-per-update, and PPO settings:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai train --device cpu --output runs/tic-tac-toe-ppo --updates 2000 --games-per-update 64 --seed 42 --resume
```

`--updates` is the total budget; this continues toward 2000. Resume derives seeds
from absolute indices and repeats an interrupted unsaved update. A crash between
checkpoint saving and logging can leave a missing or unfinished final row.
Tic-Tac-Toe resume preserves a damaged log in `updates.jsonl.recovery-SHA256.bak`
before atomically repairing its append boundary. Complete history is retained;
missing rows are not invented and cumulative sample counts remain unknown.
Earlier corruption, newline-terminated malformed records, duplicate indices,
and history ahead of the checkpoint reject without rewriting the log.
Changing seed, collection
size, or PPO configuration rejects. Device changes are allowed without numerical
identity across devices/versions. There is no mid-episode/minibatch resume.
[Architecture](../../ARCHITECTURE.md#seeds-checkpoints-and-resume) defines the contract;
[Phase 4 results](../archive/PHASE4_RESULTS.md) records the original CUDA experiment.

## Optional checkpoint monitoring

For a new run, add monitoring flags and repeat them on resume:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai train --device cpu --output runs/monitored --updates 200 --seed 42 --monitor-interval 50 --monitor-retain 3 --monitor-games-per-seat 128 --monitor-self-play-games 64 --monitor-seed 20261022
```

Monitoring is disabled by default. `monitoring.json` fixes its configuration;
resume rejects changes. `development-NNNNNNNN.json` reports hashes/counters,
fixed opponents, both-seat tactics, entropy, coverage, and action diversity.
`snapshots/` retains the newest three checkpoints plus an older best if needed.
`best.json` selects equal-case mean random/heuristic × X/O score, earliest tie
wins; `latest.pt` remains the resume marker.

Missing current-boundary evaluation retries on resume; failure does not undo a
saved update. Corrupt files reject. Snapshot storage requires hard-link support.
Old unmonitored runs resume with monitoring disabled. Missing log history leaves
cumulative samples unknown. Evaluation copies/RNG are isolated from training.
See [monitoring contracts](../../ARCHITECTURE.md#development-monitoring-and-performance).

## Fixed-opponent evaluation

After training, evaluate a frozen checkpoint on CPU:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai evaluate runs/tic-tac-toe-ppo/latest.pt --output runs/tic-tac-toe-evaluation.json --games-per-seat 1000 --seed 20261015 --mode sampled
```

This plays 1,000 games per random/heuristic × X/O case (4,000 total), hashes/loads
one checkpoint snapshot, and reports outcomes, lengths, seeds, counters, and error
categories. No training or search occurs; source files remain untouched. The
output parent must exist. Reports use exclusive creation and must be new; select
a different filename to repeat evaluation. The CLI and report writer share guards
for checkpoints (`*.pt`), training/monitoring manifests, best pointers,
development reports, and snapshot directories. Existing destinations, including
resolved aliases and hard links, are never replaced; no overwrite option exists.

`sampled` matches the masked training distribution; `greedy` selects maximum
logits with lowest-cell ties. A greedy policy versus a deterministic heuristic
repeats the same game per seat; more seeds do not make independent games.
`--start-index` supports deterministic shards. Match protocols consistently and
reserve fresh final seeds if results guide tuning; seeds need not imply unseen
boards. [Policy evaluation](../archive/POLICY_EVALUATION.md) owns recorded results and limits.

## Frozen policy tactical and learning-signal audit

With training stopped, run the bounded CPU diagnostic:

```powershell
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m boardgame_ai.policy_audit runs/tic-tac-toe-cuda/latest.pt --output runs/phase45-task1.json --games 1024 --update-games 64 --seed 20261016
```

The report must be new, its parent must exist, and it must be outside the source
checkpoint directory. This audits public boards, per-seat self-play signals, and
a bounded PPO update on disposable copies. Updated weights are never saved;
source hashes are checked unchanged. No tactical labels enter learning.
[Policy diagnosis](../archive/POLICY_DIAGNOSIS.md) defines the census, sampling/update budget,
measured signals, and limits. The historical command uses its recorded checkpoint.

## Training curves and baseline controls

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[analysis]"
.\.venv\Scripts\python.exe -m boardgame_ai analyze runs/tic-tac-toe-cuda/latest.pt --output runs/phase4-analysis-rerun --games-per-seat 1000 --seed 20261015
```

The optional Matplotlib extra renders headless PNG/SVG charts. Analysis reads
`updates.jsonl` beside the checkpoint (`--training-log` overrides), then compares
the frozen sampled policy with matched random/heuristic controls: 12 cases,
12,000 games by default. Output must be separate and absent/empty. Reports contain
hashes, raw data, paired intervals, and a checkpoint-level assessment. No training
or checkpoint modification occurs. Curves describe training, not held-out strength
over time; smoothing does not fill missing rows. See [Phase 4 results](../archive/PHASE4_RESULTS.md)
for measurements, plotted conventions, tracked charts, and limitations.

## Phase 4.5 bounded benchmark and comparison

The [declared protocol](../../PHASE45_PROTOCOL.md) fixes budgets and selection before
execution; [results](../../PHASE45_RESULTS.md) owns measurements and consumed seeds.

```powershell
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m boardgame_ai.training.experiments --output runs/phase45/entropy-comparison
```

Output must be absent; use `--resume` with identical machine settings if interrupted.
New experiments retain `settings.json` and the original `protocol.md` snapshot
with hashes in `experiment.json`. Editing explanatory Markdown does not change
experiment identity. The packaged immutable [settings](../../src/boardgame_ai/training/protocols/phase45-v1.json)
fix PPO values as well as budgets, seeds, monitoring and schedule identities.
Raw reports and immutable selection manifests remain there. Reserve fresh final
seeds for subsequent tuning claims.

Check protocol compatibility without launching training or evaluation:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai.training.experiments --output runs/phase45/entropy-comparison --validate-only
```

This read-only check supports the exact historical v1 manifest whose original
protocol hash begins `9d55a0f8`, verified against the archived original snapshot.
It preserves the original manifest/hash; it does not migrate artifacts or certify
checkpoint/report integrity. Unknown historical identities or changed settings
reject with restoration/review instructions. See the [resolution record](../../docs/reviews/2026-10-16-reliability-resolution.md).

The benchmark executes an explicitly supplied saved reference PPO source. To
reproduce the historical comparison, ensure `runs/` exists and use new outputs:

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
git show fe66184e5f43269c41e1c79842ec54ee4c4ec273:src/boardgame_ai/training/ppo.py | Set-Content runs/phase45-reference-ppo.py
.\.venv\Scripts\python.exe -m boardgame_ai.training.benchmark --reference-ppo runs/phase45-reference-ppo.py --output runs/phase45-benchmark-rerun
```

It compares repeated CPU/CUDA timings, checks same-device parameter/Adam/shuffle
equivalence, and runs a separate fixed-corpus batched-inference probe. Production
collection stays sequential; faster forward passes alone do not show end-to-end
speedup. The protocol and results retain warm-up, budget, and timing details.

## Baseline verification and benchmarks

```powershell
.\.venv\Scripts\python.exe scripts/verify_batches.py
.\.venv\Scripts\python.exe scripts/benchmark_baselines.py
```

The first verifies reproducible batches (6,000 executions); the second writes
three fixed-seat metric reports. [Baseline benchmarks](../archive/BASELINE_BENCHMARKS.md)
defines scoring, counts, lengths, and limitations. [Development examples](../../docs/DEVELOPMENT.md#baseline-agents-and-simulation)
show single games, deterministic batch subsets, and outcome adapters.

## Public search-target training

The separate learner supports CPU training and self-generated replay. A tiny
functional smoke run (not a learning-quality experiment):

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai.training.search_run runs/new-search-smoke --updates 1 --games-per-update 1 --simulations 2 --batch-size 2 --optimizer-steps 1
```

Output must be absent/empty. Total `--updates` is required. Resume with `--resume`,
a larger total budget and identical seed/collection/search/replay/optimizer
settings. `latest.pt` restores weights, Adam, replay, RNG and counters. Immutable
`update-NNNNNNNN.json` files report collected decisions, simulations, evaluator
calls, losses and replay age. There is no automatic evaluation. Search-outcome
checkpoints cannot be passed to PPO-only `evaluate`; their loaded CPU model
supports `TicTacToePolicyAgent` programmatically. See
[implementation and limits](../../docs/reviews/2026-10-16-search-target-implementation.md).

## Public search evaluation

Frozen CPU public-search evaluation is separate from training:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai.search.evaluation runs/monitored-seed45/latest.pt runs/new-search-evaluation
```

Output must not exist. The fixed [Phase 7 protocol](../../PHASE7_PROTOCOL.md) evaluates
3,072 games total and 4,520 public boards per condition. Checkpoint/configuration/log
files are preserved; evaluation never trains. [Results](../../PHASE7_RESULTS.md) record
measured gains and persistent heuristic-O failure. Only public Tic-Tac-Toe search
is supported; hidden/chance/same-player simulation remains future work. The
separate CPU search-target learner above is functional, with strength unmeasured.

## Save and inspect replays

```powershell
.\.venv\Scripts\boardgame-ai.exe play --game tic-tac-toe --starts ai --save-replay game.json
.\.venv\Scripts\boardgame-ai.exe replay game.json
```

Saving overwrites the selected file; its parent must exist. Version-1 public
Tic-Tac-Toe logs record starting seat, executed actor/cell placements, and
completed/cancelled/truncated status. Cancelled prefixes, including empty games,
are valid; rejected input is not saved. Engine/agent failures propagate.

Inspection validates the complete log, rebuilds through a fresh engine, and
prints public boards and outcome. It does not resume the original game or trust
saved snapshots. Unknown schemas/games, invalid actors/cells, post-terminal moves,
and inconsistent status reject. No seeds, private events, or authoritative state
are exported. [Programmatic recording](../../docs/DEVELOPMENT.md#recording-automated-tic-tac-toe-matches)
uses the same public adapter; other games require separate replay support.

## Phase 6 shared learning study

Shared recurrent candidate-scoring PPO now supports Tic-Tac-Toe, Guildworks and
Connect Four, with frozen unseen Breakthrough 5x5 adaptation. Independent learners
remain available. All 36 operator jobs/evaluations are now
[reviewed](../../docs/phase6/MULTIGAME_RESULTS_V1.md). Phase 6's declared comparison is
complete with mixed/negative results: consistent transfer speedup is unproven,
Guildworks has a sample-matched sharing penalty, and one recovered wall cost is
uncertain. This does not certify stronger-opponent play or universal noninferiority.

The [frozen protocol](../../docs/phase6/MULTIGAME_PROTOCOL_V1.md) defines three seeds,
wall-time and decision-ceiling arms, reward/value scales and success criteria.
The [Windows CMD runbook](../../docs/phase6/MULTIGAME_RUNBOOK_V1.md) provides one-line
preflight/train/resume/evaluate/report commands and the historical fixed caps.
Preserve the completed outputs; do not restart/overwrite this evaluated study.
Inspect its recorded evidence read-only:

```bat
.\.venv\Scripts\python.exe -m boardgame_ai.training.multigame_run runs/phase6-multigame-v1 --settings docs/phase6/multigame-v1.json --report
```
