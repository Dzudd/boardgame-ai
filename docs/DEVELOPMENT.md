# Programmatic development guide

Runnable API examples for contributors using the installed package. Run commands
from the repository root after the [development setup](../README.md#development-setup).
Source lives in `src/boardgame_ai/`, tests in `tests/`; tests use the installed
package without test-only import path overrides.

[Architecture](../ARCHITECTURE.md) owns implemented contracts and tensor shapes;
[information boundaries](../INFORMATION_BOUNDARIES.md) owns visibility and
ownership requirements. This guide illustrates those contracts, not an additional
specification. The roadmap retains the future task of a complete new-game guide.

## Engine lifecycle

Implement the generic `GameEngine[ObservationData]` and `Agent[ObservationData]`
interfaces for new games/agents. Reset before play, deliver `decision_for` inputs,
and follow the returned actor, which can be the same player again. Full state,
step results, reconstructive seeds, and unfiltered events belong to trusted
orchestration, not players. See the [engine contract](../ARCHITECTURE.md#engine-records-and-lifecycle).

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

## Guildworks matches

Use one agent per seat and the shared runner. Rendering is a filtered seat view;
full [rules and transitions](../ARCHITECTURE.md#guildworks-rules-and-transitions)
and [history visibility](../INFORMATION_BOUNDARIES.md#guildworks-player-views-and-history)
are shared contracts.

```python
from boardgame_ai.agents import RandomAgent
from boardgame_ai.games.guildworks import GuildworksEngine, GuildworksObservation
from boardgame_ai.games.guildworks.evaluation import guildworks_outcome
from boardgame_ai.simulation import run_match

game = GuildworksEngine()
result = run_match(
    game,
    (RandomAgent[GuildworksObservation](), RandomAgent[GuildworksObservation]()),
    engine_seed=42, agent_seeds=(100, 101), max_steps=32,
)
assert result.terminated and result.num_steps == 32
print(game.render(player_id=0))  # Only seat 0's filtered view.
print(guildworks_outcome(game))
```

For a saved recurrent policy, construct a separate
`RecurrentGuildworksAgent(load_checkpoint(path).model)` per seat using
`models.guildworks` and `training.guildworks_run`, then use the same `run_match`
lifecycle. The [recurrent contract](../ARCHITECTURE.md#guildworks-recurrent-learning)
defines tokens, masks, full-episode updates, and checkpoint compatibility.

## Tic-Tac-Toe tensor and PPO walkthrough

Run the following examples in order in one Python session. They demonstrate
mechanics with an untrained model, not playing strength. Encoding consumes only
player-visible inputs. See [tensor contracts](../ARCHITECTURE.md#observation-actions-and-network)
for perspective, inactive/terminal handling, shape/dtype checks, and legal masking.

### Observation encoding

```python
from boardgame_ai.games.tic_tac_toe import TicTacToeEngine
from boardgame_ai.models.encoders import encode_tic_tac_toe_observation

game = TicTacToeEngine()
game.reset()
observation, _ = game.decision_for(0)
tensor = encode_tic_tac_toe_observation(observation)
assert tensor.shape == (6, 3, 3)
```

### Action encoding and masking

The decoder returns a supplied legal action. Exclude inactive/terminal rows
before selecting actions: an all-false mask is rejected, not replaced by a move.

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

### Policy and value inference

Use explicit device/dtype conversion and inference mode. Outputs are raw logits
and observer-relative values; terminal predictions do not replace zero bootstrap.

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

### Rollouts and GAE

Collection resets the game. Each seat owns a trajectory; GAE never crosses seat
or episode boundaries. See [credit assignment and PPO](../ARCHITECTURE.md#collection-credit-assignment-and-ppo)
for reward timing, truncation, private sampling RNG, and objective definitions.

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

### PPO updates and checkpoints

The same policy generated both seats' samples, so both segments are eligible.
The example overwrites the explicit `ppo.pt` path in the current directory.
Checkpoints restore optimizer/shuffle state for the same next batch/runtime;
experiment-level resume uses the [training command](../README.md#training-and-resume).
PPO updates preserve the model's train/eval mode. `load_ppo_checkpoint` defaults
to CPU; pass `device="cuda"` explicitly to restore on CUDA.

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

## Baseline agents and simulation

`RandomAgent` samples the supplied choices with its own RNG. The deterministic
Tic-Tac-Toe heuristic prioritizes win, block, center, corners, edges; it does not
guarantee optimal play or fork defense. [Baseline contracts](../ARCHITECTURE.md#tic-tac-toe-and-baseline-agents)
define reset, action ordering, and input validation.

For frozen learned-policy evaluation, `TicTacToePolicyAgent` in
`boardgame_ai.agents.policy_agent` implements the same `select_action`/`reset`
interface using only detached observations and supplied legal choices. It owns
a frozen CPU float32 model copy and private sampling RNG; see
[evaluation contracts](../ARCHITECTURE.md#evaluation-and-analysis).

### Manual decision loop

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

The heuristic can replace either random seat in that loop:

```python
from boardgame_ai.agents import TicTacToeHeuristicAgent

# In the complete-game example above, use the heuristic for X:
agents = (TicTacToeHeuristicAgent(), RandomAgent[TicTacToeObservation]())
```

### Single-game runner

`run_match` performs resets; `play_game` continues an already initialized game.
A live cutoff is truncation, not a draw. Reward totals describe executed steps,
not generic game scores. [Simulation contracts](../ARCHITECTURE.md#simulation-metrics-human-play-and-replays)
cover errors, lifecycle, and outcome adapters.

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

### Reproducible batches

An absolute match index determines independent engine/seat streams under the
[versioned seed schedule](../ARCHITECTURE.md#seeds-checkpoints-and-resume).
Reproduction also requires matching code, runtime, configuration, and action
ordering. Seeds are trusted metadata that can reconstruct hidden setup.

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

### Outcome metrics

Continue with the game and agents from the batch example. `evaluate_batch`
records ordinary failures and resets for the next match; completed/classified
games alone enter outcome aggregates. A report is trusted evaluation data,
not a player replay. The selected JSON file is overwritten.

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

## Connect Four self-play

This collects one complete game with the same frozen current policy in both
seats. It performs no optimization, evaluation or artifact writes. Returned
trajectories use Connect Four shapes; do not pass them to the existing
Tic-Tac-Toe PPO batch builder. Use the dedicated Connect Four adapter below.

```python
from boardgame_ai.games.connect_four import ConnectFourEngine
from boardgame_ai.models.connect_four import initial_connect_four_model
from boardgame_ai.training.connect_four import (
    collect_connect_four_self_play,
    derive_connect_four_seed,
)

root, game_index = 100, 0
model = initial_connect_four_model(
    derive_connect_four_seed(root, 0, "initialization")
)
rollout = collect_connect_four_self_play(
    ConnectFourEngine(), model,
    engine_seed=derive_connect_four_seed(root, game_index, "engine"),
    sampling_seeds=(
        derive_connect_four_seed(root, game_index, "seat:0"),
        derive_connect_four_seed(root, game_index, "seat:1"),
    ),
)
assert 7 <= rollout.num_steps <= 42
assert all(seat.bootstrap_value == 0 for seat in rollout.players)
```

Each seat owns detached CPU observations `(T,6,6,7)`, legal masks `(T,7)`,
sampled indices, old log probabilities/values, own interval rewards and global
step indices. A live step cutoff raises; an optional absolute `deadline` from
`time.monotonic()` raises `TimeoutError` between decisions. Partial samples do
not return. The model/collector preserve source parameters, gradients, modes and
global RNG. See the [architecture](../ARCHITECTURE.md#connect-four) and
[functional protocol](PHASE6_CONNECT_FOUR_PROTOCOL.md) for contracts and remaining
run-layer work. The example seed is separate from the declared functional run.

For a programmatic optimizer step on that current-policy rollout:

```python
from boardgame_ai.training.connect_four_ppo import (
    ConnectFourPPOTrainer, prepare_connect_four_batch,
)

trainer = ConnectFourPPOTrainer(
    model, shuffle_seed=derive_connect_four_seed(root, 0, "shuffle")
)
report = trainer.update(prepare_connect_four_batch(rollout.players, trainer.config))
assert trainer.update_count == 1
```

`training.connect_four_checkpoint.save_connect_four_checkpoint(trainer, path)`
requires an existing parent and a successful update boundary; an existing file
must be a compatible Connect Four checkpoint. The corresponding
`load_connect_four_checkpoint(path, device="cpu", expected_config=trainer.config)`
returns a fresh trainer with model/Adam, counters, mode and shuffle state restored.
Neither function writes or reads trajectories, run schedules or protocol
manifests. A failed/expired update may have performed minibatches; reload a saved
boundary rather than saving or continuing the partial trainer. Exact CPU
continuation requires the same next batches/runtime. Run-level resume, cumulative
elapsed caps, metrics recovery and evaluation still need a separate bounded
runner; these helpers do not execute the declared functional campaign.

## Recording automated Tic-Tac-Toe matches

Pass `TicTacToeRecorder().record` as the optional `on_step` callback to `run_match`
or `play_game`. Then call `recorder.finish(starting_player, status)` and
`save_replay`. These helpers are in `boardgame_ai.games.tic_tac_toe.replay`.
Use a new recorder for every reset; recording is opt-in. See the
[replay contract](../ARCHITECTURE.md#simulation-metrics-human-play-and-replays)
and [public replay boundary](../INFORMATION_BOUNDARIES.md#public-tic-tac-toe-replays).

The callback is trusted orchestration; other games need their own filtered replay
adapter. Public Tic-Tac-Toe placement logs do not expose seeds, private events,
observations, rewards, or authoritative snapshots.
