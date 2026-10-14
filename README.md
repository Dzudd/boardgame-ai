# General-Purpose Board Game AI

A Python platform for board game engines and, later, PyTorch agents. Games will
support hidden information and multiple atomic decisions per turn.

## Development setup

Use Python 3.12 or newer. Python 3.12 is the initial development baseline.
From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
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
and strict type checking. This initial setup uses version ranges, not a lockfile.
NumPy and PyTorch are runtime dependencies. The first install command selects a
CPU build of PyTorch; the editable install then installs the remaining dependencies.
For GPU training later, select the appropriate build using the
[official PyTorch installer](https://pytorch.org/get-started/locally/).

## Command line

```powershell
.\.venv\Scripts\boardgame-ai.exe --help
.\.venv\Scripts\boardgame-ai.exe info
.\.venv\Scripts\python.exe -m boardgame_ai info
```

`info` reports Python, NumPy, and PyTorch versions and whether CUDA is available.
CUDA availability is diagnostic; it does not select a training device.
The CLI uses standard-library `argparse`. Game and training commands will be
added in their respective phases.

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
PyTorch. It triggers on pushes, pull requests, and manual dispatch once this
repository is hosted on GitHub. No remote is currently configured, so hosted CI
has not run. These checks can also be run locally using the commands above.

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
Phases 1–3 are complete; next is Phase 4, Task 1: observation tensor encoding.

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
