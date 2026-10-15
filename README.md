# General-Purpose Board Game AI

A Python platform with a tested Tic-Tac-Toe engine and PyTorch PPO self-play
learner. The long-term direction is MuZero-style search learning, with future
hidden-information games and multiple atomic decisions per turn.

## Current direction and learning protocol

Phases 1–4 meet the original minimal criterion, but the trained policy loses
nearly every heuristic matchup. [Phase 4.5](BOARDGAME_AI_PROJECT_PLAN.md#phase-45--self-play-diagnosis-and-learning-reliability-planned)
is planned before harder games: diagnose learning signals and tactical failures,
add checkpoint monitoring, profile/optimize throughput, test controlled self-play improvements, and design
the search-learning bridge. None of these new tasks is implemented yet.

Training uses **self-play only**: one latest-policy snapshot plays both seats.
No random/heuristic/perfect-play training opponents, expert move targets, or
historical opponent pool are planned. Existing random/heuristic matches measure
strength separately; their decisions and rewards do not enter training batches.

By default, collect 64 complete games with frozen weights, calculate GAE within
each player's trajectory, then train on **all** collected decisions in shuffled
minibatches of 64 for four epochs. There is no update after each played move.
Shuffling is not subsampling one or two moves per game, and game results do not
automatically identify which moves were good or bad. Both seats contribute their
own reward/value-based targets, including a loser's terminal reward.

Exploration currently uses legal-action probability sampling and a fixed .01
entropy coefficient. No epsilon-random moves, replay buffer, MCTS, learned
dynamics, periodic checkpoint archive, or automatic evaluation during training
exists. X starts every training game from an empty board. The recorded final
25-update X win rate of 99.81% describes an imbalance between the shared model's
seats, not proof of strength; its cause is still undiagnosed.

The planned path is current-policy self-play → bounded search-guided self-play
→ learning from search policies/outcomes → learned dynamics. Future replay may
sample positions and short sequences from self-generated games, with explicit
budgets and credit assignment. It requires its own learner design; old PPO
trajectories cannot simply become ordinary on-policy training batches. A perfect
solver is not required for this methodology. See the roadmap and architecture
for the current constraints and pending tasks.

CUDA placement is verified on the RTX 4070 Ti, but small-model training need not
be faster on GPU. A short temporary timing probe measured one post-warm-up
64-game update at 0.413 s on CUDA versus 0.325 s on CPU. These are preliminary
samples, not a general speed claim. Phase 4.5 will benchmark repeated stage and
end-to-end timings, reduce unnecessary CPU/GPU synchronization, and assess
batched game inference while preserving learning semantics. Minibatch-size
changes remain separate learning experiments. See PROGRESS.md for probe details.

## Development setup

Use Python 3.12 or newer. Python 3.12 is the initial development baseline.
From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -e ".[dev,analysis]"
```

Activation is optional; the explicit interpreter path avoids PowerShell execution
policy changes. On macOS/Linux use `.venv/bin/python` instead.

Run the development checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
```

Format code with `.\.venv\Scripts\python.exe -m ruff format .`.
Dependencies live in `pyproject.toml`; `dev` installs testing, formatting, linting,
and strict type checking. Package metadata uses version ranges, not a lockfile.
NumPy and PyTorch are runtime dependencies. The command above selects the verified
Windows CUDA build for this RTX 4070 Ti; the editable install adds the remaining
dependencies. For CPU-only development/CI, use the `/whl/cpu` index instead.
For other hardware, select a compatible build using the
[official PyTorch installer](https://pytorch.org/get-started/locally/).

## Command line

```powershell
.\.venv\Scripts\boardgame-ai.exe --help
.\.venv\Scripts\boardgame-ai.exe info
.\.venv\Scripts\python.exe -m boardgame_ai info
```

`info` reports Python, NumPy, and PyTorch versions and whether CUDA is available.
CUDA availability is diagnostic; it does not select a training device.
The CLI uses standard-library `argparse` and supports play, replay, train,
evaluate, and analyze commands in addition to info.

Source code lives in `src/boardgame_ai/`, with tests in `tests/`. The installed
package must be used to run tests; no test-only import path overrides are needed.
Generated environments, caches, checkpoints, datasets, and runs are ignored by Git.

## Core game types

`boardgame_ai.core` exports `GameState`, `Observation`, `Action`, `StepResult`,
`GameSpec`, and `GameEvent`. State and observation payloads are generic so each
game can define its own typed data. Actions represent atomic choices, and step
results allow consecutive decisions by the same player.

State and complete step results are engine-facing. Construct player observations
independently and filter events with `result.events_for(player_id)` before
delivering them to a player. Frozen records do not recursively freeze payloads.
See the architecture notes for field semantics and information-boundary limits.

`GameEngine[ObservationData]` (from `boardgame_ai.core`) defines the engine
interface. Implement its metadata properties and gameplay methods in each game.
`Agent[ObservationData]` (from `boardgame_ai.agents`) defines action selection
from a filtered observation and a tuple of legal actions. Agents may override
`reset(seed)` to clear memory or seed their own randomness.

Reset engines and agents before each game. At each decision, query
`current_player()`, obtain that player's observation and legal actions, ask its
agent to select an action, and call `step(action)`. Use the returned next player;
one player can make several consecutive decisions. `get_rewards()` reports the
latest step's incremental rewards, or zeros after reset, rather than cumulative
scores. The interfaces are abstract; Tic-Tac-Toe is the first concrete game.
Random and Tic-Tac-Toe heuristic baselines are available. A test-only implementation exercises
the general interface in `tests/test_interfaces.py`.

Use `engine.decision_for(player_id)` to obtain detached agent inputs. See
[information boundaries](INFORMATION_BOUNDARIES.md) for visibility, event filtering,
illegal-action errors, reset requirements, and deterministic seed rules.

## Continuous integration

`.github/workflows/ci.yml` runs lint, formatting, strict typing, tests, dependency
checks, and CLI smoke checks on Python 3.12 for Windows and Linux, using CPU
PyTorch. It triggers on pushes, pull requests, and manual dispatch. The private repository is
[Dzudd/boardgame-ai](https://github.com/Dzudd/boardgame-ai), with `master` tracking
`origin/master`. See [GitHub Actions](https://github.com/Dzudd/boardgame-ai/actions)
for hosted check results. These checks can also be run locally using the commands
above.

## Tic-Tac-Toe

Phases 1 and 2 are complete. `TicTacToeEngine` provides the full engine contract,
public observations, and text rendering. Seat 0 is X and starts by default; seat 1
is O. `TicTacToeEngine(starting_player=1)` configures O to start, preserved across
reset and cloning. Place actions use row-major cell indices 0–8. Wins award +1 to the winner
and -1 to the loser; draws and nonterminal moves award zero.

```python
from boardgame_ai.core import Action
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine

game = TicTacToeEngine()
game.reset(seed=42)
for cell in (0, 3, 1, 4, 2):
    game.step(Action(kind="place", parameters=(("cell", cell),)))

assert game.is_terminal()
assert game.get_rewards() == (1.0, -1.0)
print(game.render(player_id=0))

copy = game.clone()  # Preserves the terminal position and rewards.
copy.reset(seed=42)  # The original remains finished.
assert game.is_terminal() and not copy.is_terminal()
```

Call reset before gameplay or cloning. Both players can observe throughout play
and after termination. `clone()` preserves the concrete type and creates an
independent RNG; subsequent moves or resets do not affect the other instance.
Tests cover all 5,478 distinct reachable board positions and shared conformance.
Phases 1–4 are complete under the minimal random-baseline criterion. CUDA
training, held-out evaluation, curves, and controls are documented in
[Phase 4 results](PHASE4_RESULTS.md); heuristic robustness remains weak.

## Observation tensor encoding

```python
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.models.encoders import encode_tic_tac_toe_observation

game = TicTacToeEngine()
game.reset()
observation, _ = game.decision_for(0)
tensor = encode_tic_tac_toe_observation(observation)
assert tensor.shape == (6, 3, 3)
```

The pure encoder returns a fresh contiguous CPU `torch.float32` tensor with
channels: own marks, opponent marks, empty cells, observer turn, opponent turn,
terminal. Values are binary; the last three channels are constant planes.
Perspective uses `observation.player_id`, even for inactive observers; coordinates
stay row-major for both seats. Terminal views retain the board, zero both turn
planes, and fill the terminal plane with ones. Only the player-visible observation
is consumed, with no engine reference, rewards, or RNG access.

There is no batch dimension or autograd history. Use `torch.stack` for batches and
`.to(device)` for later device transfers. Incompatible payloads, seat/cell values,
and phase/actor combinations raise ValueError; validation checks the schema,
not game reachability. An exhaustive test audits all 5,478 reachable boards for
both observers, including 958 terminal boards, and checks seat-label symmetry
and lossless recovery of the visible board. This is representation validation;
the trained policy's heuristic weakness remains unresolved. Search remains pending.

## Action encoding and masking

```python
import torch
from boardgame_ai.models.encoders import (
    decode_tic_tac_toe_action,
    encode_tic_tac_toe_action,
    encode_tic_tac_toe_legal_actions,
)
from boardgame_ai.models.masking import mask_action_logits

# Reuse the initialized game from the observation example.
observation, legal = game.decision_for(0)
mask = encode_tic_tac_toe_legal_actions(legal)
logits = torch.zeros(9)  # Demonstration scores for the masking helper.
masked = mask_action_logits(logits, mask)
index = int(masked.argmax().item())
action = decode_tic_tac_toe_action(index, legal)
assert encode_tic_tac_toe_action(action) == index
```

Placement indices are always row-major cells 0–8, for either player's perspective.
The legal encoder returns a fresh CPU bool mask `(9,)`, True exactly at supplied
cells, independent of tuple ordering. Decoding returns the matching supplied
Action; malformed actions, duplicate cells, and unavailable indices reject.
These helpers consume supplied choices without querying a board or engine.

Masking sets illegal logits to negative infinity, so softmax assigns them zero
probability. Legal logits and gradients retain their dtype/device; inputs are
not mutated. Logits must be finite floating-point tensors and masks bool with
identical nonempty shapes/devices. Batches shaped `(..., action_count)` work
without broadcasting; stack masks and move them to the logits' device explicitly.

Empty legal tuples encode as all-false masks, including after termination or for
inactive seats. Masking rejects any row without a legal action; exclude terminal
rows before constructing policy distributions. No fallback move or action
sampling is provided. The engine remains responsible for move validation.

## Policy and value network

```python
from boardgame_ai.models import TicTacToeActorCritic

# Reuse the initialized game, torch, and encoding/masking imports above.
model = TicTacToeActorCritic()
model.eval()
observation, legal = game.decision_for(0)
encoded = encode_tic_tac_toe_observation(observation)
with torch.inference_mode():
    logits, value = model(encoded)
    probabilities = mask_action_logits(
        logits, encode_tic_tac_toe_legal_actions(legal)
    ).softmax(dim=-1)
assert logits.shape == (9,) and value.shape == ()
```

The shared feed-forward body is 54→64→64 with tanh activations. Linear heads
return nine raw policy logits and one unbounded scalar estimate of future return
for the observation's player. Both heads use only the same visible features.
An untrained model has no demonstrated playing strength.

Inputs shaped `(..., 6, 3, 3)` produce logits `(..., 9)` and values `(...)`.
Construction explicitly uses CPU float32; model and input dtype/device must match.
Use `.to(...)` for explicit conversion and `torch.stack` for batches. Invalid or
empty shapes, nonfinite inputs, and dtype/device mismatches raise ValueError.
Initialization uses standard PyTorch randomness; callers may set a PyTorch seed
before construction for repeatable parameters. Forward consumes no randomness.
Use eval() with no_grad()/inference_mode() for inference.

Terminal/inactive tensors can be evaluated, but predictions remain raw estimates.
Never select actions with their all-false masks. Future trajectories must use
zero bootstrap at true termination rather than relying on the terminal prediction.
Trajectory collection and GAE are described below. A frozen neural Agent
wrapper is available for evaluation; search remains pending.

## Trajectory collection and GAE

```python
from boardgame_ai.training import (
    collect_tic_tac_toe_rollout,
    generalized_advantage_estimation,
)

# Reuse the model and game above; collection resets the engine.
rollout = collect_tic_tac_toe_rollout(
    game, (model, model), max_steps=9,
    engine_seed=42, sampling_seeds=(100, 101),
)
player = rollout.players[0]
advantages, returns = generalized_advantage_estimation(
    player.rewards, player.values,
    bootstrap_value=player.bootstrap_value,
    gamma=0.99, gae_lambda=0.95,
)
```

Each player's trajectory holds CPU float32 observations `(T,6,3,3)`, bool legal
masks `(T,9)`, int64 action indices/global step indices `(T,)`, and float32 old
log probabilities, values, and rewards `(T,)`. Models may be shared between
seats. Inference uses only filtered views and leaves weights, gradients, and
model modes unchanged; inputs are explicitly moved to the model dtype/device.
Sampling uses independent per-seat CPU generators, without global RNG changes.
Fixed sampling seeds reproduce with the same weights, engine configuration,
action ordering, code, and runtime; model initialization seeds are separate.

A record's reward sums its player's incremental rewards until their next action
or rollout end, including opponent steps. Discounting and GAE use the player's
**decision clock**, not the global atomic-step clock. Intervening rewards are
summed without internal discounting. Do not mix player seats or concatenate
separate episodes when calling GAE. With delta=r+gamma*next_value-value, GAE uses
A=delta+gamma*lambda*A_next and return=A+value. Targets are detached; coefficients
must be in [0,1]. Empty segments are valid.

True termination sets zero bootstrap for both seats without terminal inference.
A live step-limit cutoff is truncation, not a draw, and uses each nonempty
player segment's filtered cutoff value, even when that seat is inactive.
Empty segments bootstrap zero. max_steps=0 is valid; every collection call
resets, so this API does not resume an earlier segment. Rewards before a seat's
first decision appear only in rollout.total_rewards, not policy credit.

These records belong to trusted learning code, not public replays or other
players' inputs. Errors propagate without retry. PPO updates are described below;
self-play scheduling is available below; external baseline training is excluded
by the self-play-only methodology.

## PPO updates and checkpoints

```python
from boardgame_ai.training import (
    PPOConfig, PPOTrainer, prepare_ppo_batch,
    save_ppo_checkpoint, load_ppo_checkpoint,
)

# The rollout above used this same model in both seats; both segments are eligible.
config = PPOConfig(epochs=2, minibatch_size=32)
trainer = PPOTrainer(model, config, shuffle_seed=123)
batch = prepare_ppo_batch(rollout.players, config)
report = trainer.update(batch)
save_ppo_checkpoint(trainer, "ppo.pt")  # overwrites this explicit path
restored = load_ppo_checkpoint("ppo.pt")
```

PPO batches compute GAE within each player segment before combining decisions.
Include only segments generated by the learner policy, not a separate opponent's
samples. Inputs and targets are cloned/detached CPU tensors; an entirely empty
batch rejects. The trainer supports CPU or CUDA float32 models and explicitly
transfers batches to the model device. Collection and GAE retain CPU storage.

The objective combines clipped policy loss, weighted half-MSE value loss, and a
legal-action entropy bonus. Defaults are clip .2, value coefficient .5, entropy
coefficient .01, Adam learning rate 3e-4, four epochs, minibatches 64, and gradient
norm limit .5. Full-batch advantage normalization uses population standard
deviation; singleton/constant advantages become zero. Gamma .99 and lambda .95
keep the per-player decision clock. No value clipping is applied.

A private shuffle generator makes minibatch order repeatable without global RNG
changes. Partial minibatches are included. Reports contain losses, entropy,
approximate KL, clip fraction, and sample/step counts; these are not playing-strength
measurements. Invalid samples reject; errors after successful minibatches do not
roll back those updates. Model train/eval mode stays unchanged.

Version-1 checkpoints save model/config, Adam moments, update/step counters,
model mode, and shuffle RNG state. The parent directory must exist. Saving
validates then atomically replaces the chosen file; loading uses restricted
weights-only loading, validates compatibility, and returns a fresh trainer on
the requested device (CPU by default; pass device="cuda" for CUDA). Adam moments
follow the model device; the serialized weights and optimizer tensors remain CPU.
No engine/rollout data, sampling seeds, gradients, or global RNG state is saved.
Restoring reproduces the next update with the same batch and runtime. It does not
resume an interrupted minibatch, episode, or future training experiment.

The self-play runner and independent evaluation are implemented. The completed
run and its limited strength results are documented below; Phase 4.5 follow-ups
remain planned.

## Run training on CUDA (Task 6 implementation ready)

The `train` command defaults to CUDA and never silently falls back to CPU.
Verified on 2026-10-14: Windows, Python 3.12.14, RTX 4070 Ti (12 GB), NVIDIA
driver 610.88, and PyTorch 2.14.1+cu130. The previous 2.14.1+cpu installation
was the cause of unavailable CUDA. To repair that installation while preserving
the PyTorch release and other dependencies, use the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m boardgame_ai info
.\.venv\Scripts\python.exe -m boardgame_ai train --device cuda --check-device
```

The CUDA wheel provides its runtime libraries; a separate CUDA Toolkit is not
needed for this project (no custom CUDA extensions or source builds).
`nvidia-smi` reports CUDA driver support (13.3 here), not an installed Toolkit.
No `nvcc` or CUDA Toolkit directory was found in the checked standard locations.
Driver 610.88 supports the CUDA 13.0 wheel; no driver change was needed. See
[PyTorch Windows installation](https://pytorch.org/get-started/locally/) and
[NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

The preflight prints the selected GPU and exits without training or output files.
If CUDA is unavailable, check that this interpreter has a CUDA PyTorch build and
that the NVIDIA driver supports it. The installer links supported options; use
an appropriate CUDA wheel index if your setup differs. No dependency or driver
installation happens automatically. Development tests use a CPU wheel in CI;
CUDA-specific tests run conditionally when a CUDA build/GPU is available.

Launch the experiment yourself:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai train --device cuda --output runs/tic-tac-toe-cuda --updates 1000 --games-per-update 64 --seed 42
```

This means **1000 PPO updates**, each collecting 64 complete self-play games
before updating. Both seats use one frozen current-policy snapshot; all recorded
samples are on-policy. Game simulation and inference use CPU; PPO batch forward,
backward, Adam moments, and model parameters use the selected CUDA device.
This small game may spend substantial time in CPU collection; CUDA support does
not imply a measured speedup. Choose `--device cuda:1` for another GPU or explicitly
`--device cpu` for CPU runs. `train --help` lists PPO tuning flags.

Each update prints progress and saves:

- `training.json`: immutable seed, rollout schedule, and PPO configuration.
- `latest.pt`: portable model/Adam/shuffle checkpoint at the last saved update.
- `updates.jsonl`: losses/entropy/KL and **training self-play** outcomes, not
  independent evaluation scores.

New output directories must be absent or empty. Ctrl+C stops the run; resume
from the latest saved update with the same seed, games-per-update, and PPO flags:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai train --device cuda --output runs/tic-tac-toe-cuda --updates 2000 --games-per-update 64 --seed 42 --resume
```

`--updates` is the total budget, so this continues toward 2000 total updates.
Resume derives future game/seat seeds from absolute update/game indices. It
repeats an interrupted unsaved update and preserves optimizer/shuffle state.
Changing seed, rollout size, or PPO configuration rejects. Device changes are
allowed but numerical identity across devices/versions is not promised. A crash
after saving but before logging can leave a missing log row; `latest.pt` is the
authoritative progress marker. No mid-minibatch/episode resume is attempted.

The user completed the 1,000-update CUDA experiment on 2026-10-15. The agent ran
only tiny training verification workloads; the full experiment was user-launched.
Fixed-opponent evaluation and Phase 4 results are documented below. Training
continues to use latest shared-policy self-play; baseline opponents remain
evaluation tools. The current runner retains latest.pt rather than historical
snapshots and does not evaluate automatically during training.

## Fixed-opponent evaluation (Task 7)

After training finishes, evaluate its frozen checkpoint on CPU:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai evaluate runs/tic-tac-toe-cuda/latest.pt --output runs/tic-tac-toe-evaluation-update1000-sampled.json --games-per-seat 1000 --seed 20261015 --mode sampled
```

This evaluates 4,000 games: 1,000 each against random and heuristic as X and O.
The original checkpoint/training files are never written. One byte snapshot is
hashed and loaded on CPU, then the existing engine/runner/outcome adapter plays
fixed matches. No optimizer step, CUDA computation, or search occurs. The report
identifies checkpoint hash/update count and includes scores, lengths, outcomes,
seeds, and fixed error categories. Its parent directory must already exist; the
explicit output JSON is overwritten. CLI guards prevent overwriting known
training artifacts or the source checkpoint.

Sampled play is the default, matching training's masked action distribution;
`--mode greedy` selects maximum logits with lowest-cell ties and is a separate
protocol. A deterministic greedy policy versus the deterministic heuristic
repeats the same game per seat; seed count does not make those independent games.
The held-out seed defaults to 20261015 with an independent evaluation namespace;
`--start-index` supports deterministic shards. Seeds reserve RNG streams, not
necessarily unseen board positions. Repeat comparison protocols consistently,
and reserve fresh final-test seeds if these results guide further tuning.

`TicTacToePolicyAgent` lives in `boardgame_ai.agents.policy_agent`; it implements
existing select_action/reset interfaces using only detached player observations
and supplied choices. It owns a frozen CPU float32 model copy and private sampling
RNG. Baseline engine/Agent interfaces and training logic remain unchanged.

The completed checkpoint's measured [results](POLICY_EVALUATION.md) show 77.0%
X and 48.4% O wins against random, but essentially all losses against the heuristic.
There were no errors/truncations in 4,000 games. This exposes weakness hidden by
self-play outcomes; no claim of robust or optimal play is made. Task 8 results and matched controls are available below.

## Training curves and baseline controls (Task 8)

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[analysis]"
.\.venv\Scripts\python.exe -m boardgame_ai analyze runs/tic-tac-toe-cuda/latest.pt --output runs/phase4-analysis-rerun --games-per-seat 1000 --seed 20261015
```

The optional Matplotlib extra renders headless PNG/SVG charts. Analysis reads
updates.jsonl beside the checkpoint (override with --training-log), freezes the
sampled policy, then replaces the learner with random and heuristic controls
using identical case/opponent/engine seeds. There are 12 cases total (12,000 games
at the default count). No training or checkpoint modification occurs. Output must
be a separate absent/empty directory; reports include input hashes, raw match data,
normal approximate paired intervals, and a checkpoint-level assessment.

Raw diagnostics and labeled 25-update trailing averages show loss, entropy, KL,
clipping, self-play outcome rates, and game length. Missing rows are not interpolated;
smoothing restarts after gaps. These curves are not held-out strength over time.
Mean outcome score bars show the policy and controls by opponent/seat.

The [measured result](PHASE4_RESULTS.md) supports the minimal Phase 4 criterion:
+0.3160 balanced score improvement over a random control versus random opponents,
approximate 95% interval [+0.2644,+0.3676]. The policy still loses nearly every
heuristic matchup and is worse there than the random control. This is evidence
for one checkpoint, not robustness or training-seed generalization. Tracked charts
are in docs/phase4; generated reports/checkpoints remain ignored under runs.

## Random baseline

`RandomAgent` selects uniformly from the supplied legal-action tuple, ignoring
the observation payload. Its private RNG is separate from engine and global RNGs.
Use one agent per seat and reset each before every game:

```python
from boardgame_ai.agents import RandomAgent
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation

game = TicTacToeEngine()
game.reset(seed=42)
agents = (RandomAgent[TicTacToeObservation](), RandomAgent[TicTacToeObservation]())
for seat, agent in enumerate(agents):
    agent.reset(seed=100 + seat)

while not game.is_terminal():
    player = game.current_player()
    assert player is not None
    observation, legal = game.decision_for(player)
    game.step(agents[player].select_action(observation, legal))
print(game.render(player_id=0))
```

An integer seed restarts the agent sequence; None uses fresh system entropy.
Reproduction requires the same seeds, ordered legal choices, selection calls,
and Python version. Empty choices raise ValueError; invalid seeds reject without
replacing the current generator. The reusable runner below handles this lifecycle.

## Tic-Tac-Toe heuristic baseline

`TicTacToeHeuristicAgent` implements `Agent[TicTacToeObservation]` and uses only
the public board and supplied legal placements. It takes an immediate win,
otherwise blocks an immediate loss, then prefers center, corners, and edges.
Tactical ties use the lowest cell index; positional ties use row-major order.
It is a simple tactical policy without fork handling or an optimal-play guarantee.

```python
from boardgame_ai.agents import TicTacToeHeuristicAgent

# In the complete-game example above, use the heuristic for X:
agents = (TicTacToeHeuristicAgent(), RandomAgent[TicTacToeObservation]())
```

Use the same `decision_for`/`select_action`/`step` flow and reset each agent before
play. The heuristic has no RNG or memory; its inherited reset is a no-op and
choices do not depend on seeds. It returns a supplied action and raises ValueError
for empty or incompatible decision inputs. The engine still validates moves.

## Single-game simulation

`run_match` resets the engine and each agent, then plays one bounded game:

```python
from boardgame_ai.agents import RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.simulation import run_match

game = TicTacToeEngine()
result = run_match(
    game,
    (TicTacToeHeuristicAgent(), RandomAgent[TicTacToeObservation]()),
    engine_seed=42,
    agent_seeds=(100, 101),
    max_steps=9,
)
assert result.terminated and not result.truncated
print(result.num_steps, result.total_rewards)
print(game.render(player_id=0))
```

Provide a distinct agent instance for every seat. Agent seeds are per-seat and
separate from the engine seed; omitted seeds use None. Configuration and seed
values are validated before resets. Custom reset failures are not transactional.
The runner follows engine actors, including repeated actors, and passes only
detached player inputs. Agent and engine exceptions propagate without retries.

`play_game(game, agents, max_steps=...)` uses the same loop without resetting;
initialize engine and agents first, or use it to resume a truncated match.
The required nonnegative step budget counts atomic decisions. Reaching it while
the engine is live returns `truncated=True`, leaving the position resumable.
Termination on the last allowed step is a completed game; zero permits no steps.

`MatchResult` contains executed `num_steps`, summed incremental `total_rewards`,
`terminated`, and `truncated`. Totals include only steps executed by that call,
not earlier moves or rewards at reset. They are not game scores or winner labels;
truncation is not a draw. Results belong to trusted orchestration and contain no
observations or events. Outcome evaluation is described below; replays remain Task 7.

## Reproducible match batches

```python
from boardgame_ai.agents import RandomAgent, TicTacToeHeuristicAgent
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine, TicTacToeObservation
from boardgame_ai.simulation import run_batch, run_match

game = TicTacToeEngine()
agents = (TicTacToeHeuristicAgent(), RandomAgent[TicTacToeObservation]())
batch = run_batch(game, agents, num_matches=1000, seed=20261014, max_steps=9)
assert all(entry.result.terminated for entry in batch)

entry = batch[10]
assert run_match(
    game, agents, max_steps=9,
    engine_seed=entry.engine_seed, agent_seeds=entry.agent_seeds,
) == entry.result

subset = run_batch(
    game, agents, num_matches=10, start_index=100, seed=20261014, max_steps=9,
)
assert subset == batch[100:110]
```

Batches run sequentially with fixed seats. An integer batch seed determines
separate engine and per-seat agent seeds for every absolute match index using
the versioned SHA-256 schedule documented in ARCHITECTURE.md. Game length does
not affect later seeds. Reproduction also requires matching configuration, code,
runtime, action ordering, and engines/agents that fully reset and honor seeds.

Each immutable SeededMatchResult includes match_index, engine_seed, agent_seeds,
and the existing MatchResult. Seeds are trusted metadata and can reconstruct
hidden setup; keep them out of player-facing output. No observations or events
are retained. Zero matches performs no resets. Truncated matches are returned
as truncated, then the next match resets; exceptions stop the batch immediately
without returning a partial tuple. Use evaluate_batch below for metrics and error records.

Run the repeatable baseline batch verification (6,000 total executions):

```powershell
.\.venv\Scripts\python.exe scripts/verify_batches.py
```

## Outcome evaluation and metric records

```python
from pathlib import Path
from boardgame_ai.games.tic_tac_toe.evaluation import tic_tac_toe_outcome
from boardgame_ai.simulation import evaluate_batch, save_evaluation

# Reuse game and agents from the batch example above.
report = evaluate_batch(
    game, agents, outcome_for=tic_tac_toe_outcome,
    agent_labels=("heuristic", "random"), num_matches=1000,
    seed=20261014, max_steps=9,
)
print(report.summary)
Path("runs").mkdir(exist_ok=True)
save_evaluation(report, "runs/evaluation.json")
```

An outcome adapter defines terminal scores, winner seats, and draws for each game;
generic rewards are not scores or inferred outcomes. Tic-Tac-Toe uses +1/-1/0
result scores and reads its terminal public board. Reports include per-match
seeds, reward totals, game scores, outcomes, lengths, and fixed error categories.
Completed/classified games alone contribute to wins, draws, scores, and length
averages; empty denominators produce None. Truncations and errors count separately.

Unlike run_batch, evaluate_batch records ordinary failures and resets for the next
match. Reset/play errors have unknown length/rewards; outcome adapter errors retain
the runner result but are excluded from outcome aggregates. Interrupts propagate.
No exception text or private views/events are recorded. Agent labels and seeds
are trusted metadata, so summary JSON belongs in a trusted evaluation destination.
save_evaluation overwrites its chosen file and requires an existing parent folder.
It saves metrics, not a move replay.

Run the benchmark and save three reproducible metric reports:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_baselines.py
```

See [baseline measurements](BASELINE_BENCHMARKS.md) for 1,000 games per fixed-seat
matchup, scoring, lengths, and limitations. Phase 3 is complete.

## Human-versus-agent play

```powershell
.\.venv\Scripts\boardgame-ai.exe play
.\.venv\Scripts\boardgame-ai.exe play --game tic-tac-toe --starts ai
.\.venv\Scripts\python.exe -m boardgame_ai play --game tic-tac-toe --opponent random --agent-seed 42
.\.venv\Scripts\python.exe -m boardgame_ai play --help
```

Select the game with `--game tic-tac-toe` (the current default and only supported
game). Choose the starter with `--starts human` (default) or `--starts ai`.
The starter is X and the other player O; your mark is printed in the opening
message. Separate `--human` and X/O starter options have been removed.
`--opponent` accepts `random` or `heuristic`.
`--agent-seed` controls the random opponent independently of `--engine-seed`;
the heuristic is deterministic, and Tic-Tac-Toe setup does not depend on seed.
Random play reproduces with the same seed, human actions, code, and runtime.

Every prompt shows the public board beside a board of available cell numbers;
occupied cells are replaced by dashes in the number board. The final display
uses the same layout. Enter row-major
indices 0–8: top-left is 0, center is 4, bottom-right is 8. Invalid text or occupied
cells prompt again without advancing the game. AI moves use the same engine
validation path as human moves. The final board shows a win, loss, or draw.

Enter `q`, `quit`, or `exit`, send EOF, or press Ctrl+C to stop cleanly without
declaring an outcome. A game is saved only when --save-replay is supplied.

Draws are plausible when you avoid tactical mistakes against the heuristic:
it takes wins and blocks immediate threats. It is deterministic, so repeating
your moves reproduces the game. It is not unbeatable and does not defend all
forks. Try the random opponent for more varied and easier games.

## Save and inspect replays

```powershell
.\.venv\Scripts\boardgame-ai.exe play --game tic-tac-toe --starts ai --save-replay game.json
.\.venv\Scripts\boardgame-ai.exe replay game.json
```

Saving writes a version-1 public Tic-Tac-Toe JSON log to the chosen file,
overwriting it; its parent folder must exist. It records the starting seat,
successful placements with actor/cell, and completed/cancelled/truncated status.
Cancellation saves the played prefix, including an empty game. Rejected input
and unexecuted moves are not recorded. Engine/agent failures propagate and do
not produce a misleading completed log.

Inspection validates the entire log before displaying anything, then rebuilds
the initial board and each placement through a fresh engine and prints the
ending status/outcome. It does not run agents, resume the original game, or trust
saved board snapshots. Unknown schemas/games, invalid actors/cells, post-terminal
moves, and inconsistent status are rejected.

For automated matches, pass `TicTacToeRecorder().record` as the optional on_step
callback to run_match or play_game, then use recorder.finish(starting_player,
status) and save_replay. These helpers live in
`boardgame_ai.games.tic_tac_toe.replay`. A recorder is for one game; create a new
one for each reset. No recording happens by default.

This format contains only public Tic-Tac-Toe placements/configuration, no seeds,
observations, private events, rewards, or authoritative snapshots. Future hidden
games require their own filtered replay adapter; generic on_step is a trusted
hook and does not automatically make game-specific actions public.

See [the roadmap](BOARDGAME_AI_PROJECT_PLAN.md), [architecture decisions](ARCHITECTURE.md),
and [current progress](PROGRESS.md).
