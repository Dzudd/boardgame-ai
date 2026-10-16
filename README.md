# General-Purpose Board Game AI

A Python platform for board-game engines, isolated player observations, simulation,
and self-play learning. It implements Tic-Tac-Toe PPO, recurrent Guildworks PPO,
and a playable Connect Four engine. MuZero-style search learning and shared
multi-game learning remain future work.

## Supported capabilities

| Game | Play and simulation | Learning and search | Replay |
|---|---|---|---|
| Tic-Tac-Toe | Engine, random/heuristic agents, human CLI | PPO, separate CPU search-target self-play/replay learner, frozen PPO and bounded PUCT evaluation | Public move logs |
| Guildworks | Hidden-card worker placement, random/scripted matches, filtered human CLI | Visible-history GRU and recurrent PPO; no hidden-information search | Not implemented |
| Connect Four | Public engine, random/scripted matches, human CLI | No learner or search integration | Not implemented |

Human CLI opponents are baselines; play against trained checkpoints is still
planned. Training uses latest-policy self-play only; baseline opponents and
public tactical labels are evaluation instruments, never optimizer samples.
The implemented learners and bounded search have measured limitations; none is
claimed to provide robust or optimal play. See [current evidence](PROGRESS.md).

## Project documents

| Document | Responsibility |
|---|---|
| [README](README.md) | Purpose, setup, supported capabilities, and command usage |
| [Development guide](docs/DEVELOPMENT.md) | Programmatic examples for engines, tensors, PPO, simulation, and metrics |
| [Architecture](ARCHITECTURE.md) | Implemented architecture, interfaces, tensor contracts, seeds, and resume |
| [Information boundaries](INFORMATION_BOUNDARIES.md) | Normative visibility, ownership, and learning/evaluation separation |
| [Roadmap](BOARDGAME_AI_PROJECT_PLAN.md) | Phase goals, complete task lists, statuses, and acceptance criteria |
| [Progress](PROGRESS.md) | Concise current status, verified evidence, limitations, and next milestone |
| [Baseline benchmarks](BASELINE_BENCHMARKS.md) | Reproducible random/heuristic control measurements |
| [Phase 4 results](PHASE4_RESULTS.md) | Training curves, matched controls, measurements, and limitations |
| [Policy evaluation](POLICY_EVALUATION.md) | Frozen checkpoint identity and fixed-opponent results |
| [Policy diagnosis](POLICY_DIAGNOSIS.md) | Public tactics, per-seat signals, and isolated PPO update evidence |
| [Phase 4.5 protocol](PHASE45_PROTOCOL.md) / [results](PHASE45_RESULTS.md) | Declared monitoring/performance/comparison methods and measured outcomes |
| [Phase 5 results](PHASE5_RESULTS.md) | Guildworks recurrent learning protocol and functional validation |
| [Phase 7 protocol](PHASE7_PROTOCOL.md) / [results](PHASE7_RESULTS.md) | Bounded public-search evaluation and heuristic-O diagnosis |
| [Search-learning design](SEARCH_LEARNING_DESIGN.md) | Public search and separate search-target learner contracts |
| [Search-target implementation](docs/reviews/2026-10-16-search-target-implementation.md) | Bounded CPU learner, replay/resume validation and remaining experiment work |
| [Phase 6 milestone proposal](docs/PHASE6_MILESTONE.md) | Proposed Connect Four PPO baseline; protocol still to be defined |
| [2026-10-16 audit and feasibility review](docs/reviews/2026-10-16-audit-and-feasibility.md) | Historical audit findings, architecture risks, and proposed learning-validation gates |
| [Development direction and next-session prompt](docs/reviews/2026-10-16-development-direction.md) | Reasoning for prioritizing reliability and bounded learning validation |
| [Reliability resolution](docs/reviews/2026-10-16-reliability-resolution.md) | Artifact safety, metrics recovery, protocol compatibility, validation and proposed next experiment |

Shared documentation is self-contained in a clone. Optional machine-specific AI
instructions, session handoffs, continuation prompts, and temporary notes belong
in the ignored root `AI/` directory. To use an existing local copy, explicitly ask
a new AI session to read `AI/INDEX.md` and `AI/AGENTS.md` before relevant shared
docs. Arbitrary Markdown files and nested instruction files are not assumed to
load automatically; there is no root `AGENTS.md` entry point. A fresh clone can
use the shared document index without local context.

The `/AI/` ignore rule prevents future normal additions of untracked local files;
it does not untrack already indexed files or remove earlier Git history. The
previously committed root handoffs remain in history even after their removal
from the shared tree. Do not put source, tests, checkpoints, or canonical project
evidence in `AI/`.

## Development setup

Use Python 3.12 or newer; 3.12 is the development and CI baseline. From the
repository root in PowerShell, this installs a CPU environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev,analysis]"
```

For CUDA, select a build compatible with your hardware using the
[official PyTorch installer](https://pytorch.org/get-started/locally/) before the
editable install. Activation is optional; explicit interpreter paths avoid
PowerShell execution-policy changes. On macOS/Linux use `.venv/bin/python` and
`.venv/bin/boardgame-ai` instead. Dependencies and supported version ranges live
in `pyproject.toml`, not a lockfile. NumPy and PyTorch are runtime dependencies;
`analysis` adds Matplotlib, and `dev` adds pytest, Ruff, and strict mypy checks.

The CUDA wheel supplies runtime libraries; this project uses no custom CUDA
extensions or source builds requiring a separate CUDA Toolkit. `nvidia-smi`
reports driver CUDA support, not an installed Toolkit. See
[NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
```

Format code with `.\.venv\Scripts\python.exe -m ruff format .`.
Source lives in `src/boardgame_ai/` and tests in `tests/`. Generated environments,
caches, checkpoints, datasets, and runs are ignored by Git. Tracked evidence,
including the charts in `docs/phase4/`, remains shared.

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

[The CI workflow](.github/workflows/ci.yml) runs lint, formatting, strict typing,
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
[development examples](docs/DEVELOPMENT.md#engine-lifecycle) and
[implemented contract](ARCHITECTURE.md#tic-tac-toe-and-baseline-agents).

## Connect Four

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai play --game connect-four
.\.venv\Scripts\python.exe -m boardgame_ai play --game connect-four --starts ai --agent-seed 42
```

The CLI shows a 7-column, 6-row board with aligned column labels 1–7. Enter a
column to drop a disc; full columns and invalid input retry. X starts; four in a
horizontal, vertical, or diagonal line wins; a full board without a winner draws.
Quit, EOF, and Ctrl+C cancel. The opponent is random only.

`ConnectFourEngine` and `ConnectFourObservation` support the shared `run_match`
interface with `max_steps=42`; `connect_four_outcome` adapts results. Engine/model
columns are zero-based 0–6; human input is 1–7. Training, learned opponents, search,
and replay export are not implemented. See [architecture](ARCHITECTURE.md#connect-four).

## Guildworks

Four rounds, two workers per player per round. Place on a free public space for
ink, cards, points, or a resolved ink roll, then craft one affordable hidden card
or pass. Scores determine the winner after 32 decisions. Hands/deck remain private;
resources, scores, discard, and counts are public. Full
[rules](ARCHITECTURE.md#guildworks-rules-and-transitions) and
[visibility requirements](INFORMATION_BOUNDARIES.md#guildworks-player-views-and-history)
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
seeds in trusted outputs. [Programmatic examples](docs/DEVELOPMENT.md#guildworks-matches)
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
[recurrent contracts](ARCHITECTURE.md#guildworks-recurrent-learning) and
[functional results and limitations](PHASE5_RESULTS.md).

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
[Architecture](ARCHITECTURE.md#seeds-checkpoints-and-resume) defines the contract;
[Phase 4 results](PHASE4_RESULTS.md) records the original CUDA experiment.

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
See [monitoring contracts](ARCHITECTURE.md#development-monitoring-and-performance).

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
boards. [Policy evaluation](POLICY_EVALUATION.md) owns recorded results and limits.

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
[Policy diagnosis](POLICY_DIAGNOSIS.md) defines the census, sampling/update budget,
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
over time; smoothing does not fill missing rows. See [Phase 4 results](PHASE4_RESULTS.md)
for measurements, plotted conventions, tracked charts, and limitations.

## Phase 4.5 bounded benchmark and comparison

The [declared protocol](PHASE45_PROTOCOL.md) fixes budgets and selection before
execution; [results](PHASE45_RESULTS.md) owns measurements and consumed seeds.

```powershell
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m boardgame_ai.training.experiments --output runs/phase45/entropy-comparison
```

Output must be absent; use `--resume` with identical machine settings if interrupted.
New experiments retain `settings.json` and the original `protocol.md` snapshot
with hashes in `experiment.json`. Editing explanatory Markdown does not change
experiment identity. The packaged immutable [settings](src/boardgame_ai/training/protocols/phase45-v1.json)
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
reject with restoration/review instructions. See the [resolution record](docs/reviews/2026-10-16-reliability-resolution.md).

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
three fixed-seat metric reports. [Baseline benchmarks](BASELINE_BENCHMARKS.md)
defines scoring, counts, lengths, and limitations. [Development examples](docs/DEVELOPMENT.md#baseline-agents-and-simulation)
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
[implementation and limits](docs/reviews/2026-10-16-search-target-implementation.md).

## Public search evaluation

Frozen CPU public-search evaluation is separate from training:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai.search.evaluation runs/monitored-seed45/latest.pt runs/new-search-evaluation
```

Output must not exist. The fixed [Phase 7 protocol](PHASE7_PROTOCOL.md) evaluates
3,072 games total and 4,520 public boards per condition. Checkpoint/configuration/log
files are preserved; evaluation never trains. [Results](PHASE7_RESULTS.md) record
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
are exported. [Programmatic recording](docs/DEVELOPMENT.md#recording-automated-tic-tac-toe-matches)
uses the same public adapter; other games require separate replay support.
