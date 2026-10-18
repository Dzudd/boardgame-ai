# General-Purpose Board Game AI

A reusable board-game engine and self-play learning platform under development.
The goal is to add new games through plugins and train substantial models efficiently
on the user's GPU. Existing small learners are working foundations, not completion
of that goal. Shared recurrent learning and a separate CPU search-target learner
exist; scalable new-game GPU training remains unfinished.

## Start here

- Development task: [PROGRESS.md](PROGRESS.md), then the relevant
  [roadmap milestone](BOARDGAME_AI_PROJECT_PLAN.md).
- AI read order and work rules: [AGENTS.md](AGENTS.md).
- Technical contracts: [architecture map](ARCHITECTURE.md) and
  [information boundaries](INFORMATION_BOUNDARIES.md).
- API examples: [development guide](docs/DEVELOPMENT.md).
- Historical results/protocols: [evidence index](docs/archive/README.md), only as needed.

## Supported capabilities

Engines: Tic-Tac-Toe, Guildworks, Connect Four and Breakthrough. Learning includes
independent PPO, recurrent/shared PPO and CPU Tic-Tac-Toe search-target learning.
Human play supports baseline agents and portable Connect Four checkpoints. Robust
strength, positive transfer and general large-model GPU training are not established.

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
.\.venv\Scripts\python.exe -m boardgame_ai --help
.\.venv\Scripts\python.exe -m boardgame_ai info
```

Detailed recipes are in the [command reference](docs/reference/COMMANDS.md).
Production campaigns are run by the user; select the relevant versioned runbook
before starting or resuming one.

## Continuous integration

[Commands and details](docs/reference/COMMANDS.md#continuous-integration).

## Human-versus-agent play

[Commands and details](docs/reference/COMMANDS.md#human-versus-agent-play).

## Connect Four

[Commands and details](docs/reference/COMMANDS.md#connect-four).

## Guildworks

[Commands and details](docs/reference/COMMANDS.md#guildworks).

## Training and resume

[Commands and details](docs/reference/COMMANDS.md#training-and-resume).

## Optional checkpoint monitoring

[Commands and details](docs/reference/COMMANDS.md#optional-checkpoint-monitoring).

## Fixed-opponent evaluation

[Commands and details](docs/reference/COMMANDS.md#fixed-opponent-evaluation).

## Frozen policy tactical and learning-signal audit

[Commands and details](docs/reference/COMMANDS.md#frozen-policy-tactical-and-learning-signal-audit).

## Training curves and baseline controls

[Commands and details](docs/reference/COMMANDS.md#training-curves-and-baseline-controls).

## Phase 4.5 bounded benchmark and comparison

[Commands and details](docs/reference/COMMANDS.md#phase-45-bounded-benchmark-and-comparison).

## Baseline verification and benchmarks

[Commands and details](docs/reference/COMMANDS.md#baseline-verification-and-benchmarks).

## Public search-target training

[Commands and details](docs/reference/COMMANDS.md#public-search-target-training).

## Public search evaluation

[Commands and details](docs/reference/COMMANDS.md#public-search-evaluation).

## Save and inspect replays

[Commands and details](docs/reference/COMMANDS.md#save-and-inspect-replays).

## Phase 6 shared learning study

[Commands and details](docs/reference/COMMANDS.md#phase-6-shared-learning-study).
