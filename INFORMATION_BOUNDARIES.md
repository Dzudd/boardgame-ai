# Player visibility and engine boundaries

## Self-play learning and evaluation separation

Training experience is generated only by self-play, currently one latest shared
policy frozen for both seats. Baseline and perfect-play opponents, expert move
labels, and historical training opponent pools are outside the current plan.
Existing random/heuristic evaluation matches remain measurement only: their
trajectories, rewards, and tactical labels must not enter optimizer batches.
Public-board diagnostic checks of immediate wins/blocks are reports, not teacher
targets. No exact solver is required. Evaluation-guided tuning/checkpoint
selection must be declared and followed by a fresh reserved final-test protocol.

Future search targets may come from the learner's own bounded search, using only
simulation reconstructed from permitted observations/history. Future replay may
contain self-generated player sequences and reward-derived targets in trusted
learning storage; it must never make another player's secrets model features.
It needs explicit visibility, sampling, and sequence contracts before implementation.
An authoritative hidden-state clone remains forbidden as an agent/search input.
Pending checkpoint/evaluation monitoring must preserve training RNG, model and
optimizer state, and resume behavior; evaluation reports remain trusted outputs.

## Ownership and player inputs

The engine owns full state, hidden hands, deck order, randomness, rewards, and
complete events. An agent receives only `Observation` and its legal actions.
Never pass an engine, `GameState`, full `StepResult`, RNG, seed, or debug snapshot
as an agent input. Training policy and value inputs follow the same visibility
rules; privileged critics require a separate, explicitly scoped research design.

Use `engine.decision_for(player_id)` to obtain a deep-copied observation and legal
actions for agent selection. Engines must already filter these inputs correctly:
copying removes shared mutable references but does not remove secret fields.
Observation payloads must support deep copying. `observe` must itself return an
independent view; the helper provides an additional defensive copy.

Visible information may include the player's own hand, public board, hand sizes,
discard pile, legitimate public deductions, and that player's visible history.
It must exclude opponent private cards, deck order, secret objectives, and
probabilities computed from inaccessible engine state. Terminal observations
retain these restrictions unless the game's rules explicitly reveal information.

Legal action names, parameters, ordering, and availability are also information.
Two worlds indistinguishable to a player must expose identical decision inputs.
Do not encode an unknown card in a draw choice. Return the draw outcome only
after the engine resolves it, and only to players allowed to learn it.

## Events, rewards, and diagnostics

`encode_tic_tac_toe_observation` in `boardgame_ai.models.encoders` consumes only
the filtered Observation and its public board. It accepts no engine, GameState,
legal actions, rewards, seed, or RNG. Observer-relative marks and public
actor/phase metadata become six float32 planes. Finished views keep the same
visibility restrictions; no reward or privileged terminal data is added.
Fresh tensors share no mutable storage with observations or other calls.
Schema validation cannot detect secret data incorrectly included in a future
game's observation; each game still owns filtering and visibility tests.
The reachable-board encoding audit constructs only public observation records
and uses an independent rules oracle; it never reads authoritative state.
Seat relabeling preserves observer-relative tensors. Tic-Tac-Toe's fully public
board audit does not establish hidden-world equivalence for future games.

Full `StepResult` records stay in trusted simulation/training orchestration.
Call `result.events_for(player_id)` before delivering events. A public event has
`recipients=None`; private events name their seats, and an empty tuple is internal.
The engine must validate recipient seats and keep private fields out of public
payloads. Filtering cannot fix a private card mistakenly marked public.

The reward tuple is indexed by seat and intended for trusted orchestration.
It is not automatically public or part of an observation. Give learning code
only the rewards needed for its player trajectory; games with private scores
must define when those scores become observable.

Do not place secrets in exceptions or public logs. Full replay/debug exports
require an explicitly trusted destination; future player replays must be filtered.
No automatic full-state logging is provided. Python object privacy is a coding
boundary, not protection against malicious in-process code inspecting an engine.

## Engine failure and randomness policy

- Call `reset` before gameplay queries, steps, or cloning. Otherwise raise
  `GameNotReadyError`. Metadata is available before reset.
- Invalid player seats raise `InvalidPlayerError`. Boolean IDs are rejected.
  Inactive valid seats have no legal actions; they may still observe the game.
- Well-formed but illegal actions raise `IllegalActionError`. Steps after
  termination raise its subclass `GameOverError`. Invalid action record construction
  raises `ValueError` before the engine is called.
- Rejection must preserve state, phase, rewards, history/events, and RNG state.
  Validate before consuming randomness or mutating data. Do not substitute a
  random action or silently penalize the player. Error text must not reveal secrets.
- Use a private `random.Random` created with `create_rng(seed)`. An integer seed
  restarts the full setup and chance sequence; `None` creates a fresh generator
  using system entropy. Validate seed/configuration before replacing live state.
- Keep agent randomness separate. Never seed or draw from global Python,
  NumPy, or PyTorch generators in engine operations.
- Queries do not consume randomness. `clone` copies RNG state independently.
  Reproducibility assumes the same game configuration, action order, engine code,
  Python version, and dependencies; it is not a cross-version replay guarantee.

These are requirements for every engine implementation. The abstract base class
cannot enforce game-specific visibility or transactional mutation by itself.

## RandomAgent boundaries and seeds

`RandomAgent` accepts only the existing observation and legal-action inputs and
does not inspect the observation payload. It chooses uniformly from tuple entries
using its own private generator; engine RNGs and global generators are never used.
Use a separate instance per seat and call `reset(agent_seed)` before each game.
Agent seeds configure the agent lifecycle separately from engine seeds; they are
not fields in player observations. Replays require the same agent seeds, ordered
legal choices, call sequence, and runtime versions. `reset(None)` uses fresh system
entropy. Invalid seeds and empty legal tuples raise ValueError without advancing
or replacing the existing sequence. Engines continue to validate selected actions.

## Tic-Tac-Toe heuristic boundaries

`TicTacToeHeuristicAgent` follows the same input boundary: it reads only the public
board and acting seat from the observation and ranks supplied legal placements.
It does not receive or query an engine, authoritative state, rewards, or RNG.
Its deterministic win/block/position policy has no memory or randomness;
inherited `reset(seed)` is a no-op. Invalid decision inputs raise ValueError,
and the engine remains responsible for final legality validation.

## Simulation orchestration boundary

The simulation runner is trusted orchestration. `run_match` resets engine and
agent lifecycle seeds separately; `play_game` continues an initialized game.
Both deliver only the acting player's detached `decision_for` inputs. They do
not deliver full StepResult events or reward tuples to agents. MatchResult is a
trusted summary of reward totals and execution status, not automatically public
player data; games with private scores still require filtered presentation.
The runner retains no observations or private events; recording is opt-in.
Exceptions propagate without retry, substitution, or automatic error logging.

`run_batch` reuses run_match's input boundary and resets each match separately.
Derived engine and per-seat agent seeds configure lifecycle only, never player
observations. SeededMatchResult records are trusted orchestration metadata:
engine seeds can reconstruct hidden setup and must not be published to players
without a game-specific disclosure policy. Batches retain no observations,
authoritative snapshots, or private events and perform no automatic logging.

`evaluate_batch` uses the same detached agent inputs and keeps its outcome
adapter in trusted orchestration. Outcome scores and summary reports are not
automatically player-visible. It records only fixed error categories, never
exception messages, exception class names, or tracebacks, which might reveal
private data. Ordinary failures are followed by a fresh reset for the next match;
interrupts propagate. Explicit save_evaluation exports only summary records to a
caller-chosen trusted destination. Reports include reconstructive seed metadata;
caller-supplied labels must also be appropriate for that destination. No private
events, observations, authoritative snapshots, or action histories are exported.

## Human CLI presentation

The Tic-Tac-Toe human adapter receives only detached observation/legal inputs
through run_match, like any agent. It prints the public board and legal cells,
and returns a supplied placement. Its opponent wrapper prints only the public
chosen cell and uses separate agent randomness. Neither adapter receives engine
state, full StepResults, rewards, or engine RNGs. Final rendering and outcome
classification remain in trusted CLI orchestration and use the human's public
view. Tic-Tac-Toe outcomes are public; this does not authorize exposing private
outcomes in future games. Input retries consume no engine step, and cancellation
submits no further action. Replay saving is opt-in and game-specific.

The CLI choice board is derived from supplied legal actions and printed beside
the public game board on every prompt and final display. Occupied/unavailable
cells show dashes. Starter selection is explicit engine configuration, independent
of player mark and randomness; it does not expose hidden information. X remains
the default starter, and optional O-first configuration persists across resets
and clones.

## Public Tic-Tac-Toe replays

The CLI now assigns the human/AI starter X and the other player O; it supports
explicit --game selection. Optional --save-replay exports only public placements,
starting seat, and ending status, including a cancelled prefix. No seeds, rewards,
observations, private events, or authoritative snapshots are saved. Inspection
validates the entire supported log through an engine before printing public boards.

The optional runner on_step hook receives successful actor/action pairs in trusted
orchestration. It is not an agent input and does not certify that actions are public
in other games. Only Tic-Tac-Toe has a replay export adapter at this stage; hidden
games require explicit filtering and chance-reconstruction policies. Existing
trusted evaluation reports still contain seeds and have stricter disclosure limits.

## Required tests for a new game

Reuse `assert_engine_contract` and `assert_rejection_preserves_game` from
`tests/contracts.py`, with a game factory and a bounded number of steps.
They check a first-legal-action path, reset, queries, reward consistency,
termination, seeds, clone independence, and rejected-action outcomes. Games with
longer episodes need an appropriate bound or a scripted terminating trajectory.

Also add game-specific tests for rules and information boundaries: change only
unseen state and compare player inputs; mutate returned payloads; inspect event
visibility; compare seeded chance trajectories; and try invalid actions at every
important phase. The Phase 1 hidden-card fixture demonstrates these checks but
does not replace validation of each future game.

## Tic-Tac-Toe player views

Tic-Tac-Toe has no hidden board information. Both valid seats, including inactive
seats and after termination, receive the public board in `TicTacToeObservation`.
The Observation wrapper identifies the observer, current actor, and phase.
Views omit authoritative GameState records, rewards, seeds, and RNG data.
The engine constructs fresh frozen observation records; their board tuples
contain only immutable cells and remain stable across moves and reset.
Rendering consumes these same public observations and returns text without
printing or inspecting private engine data. Agent inputs still come through
`decision_for`; shared full-state and reward boundaries continue to apply.

The completed engine passes the shared engine/rejection conformance helpers and
game-specific visibility tests. Clones preserve authoritative state and private
RNG state independently; cloning is a trusted engine operation and does not make
the cloned engine an agent input or expose RNG data through observations.

## Legal-action tensor boundaries

The Tic-Tac-Toe action encoder consumes only supplied Actions, never an engine,
board, seed, reward, or authoritative state. Legal choices become a fixed
row-major bool mask; decoding returns the supplied matching legal Action.
The caller must supply the correct player's filtered choices. Schema validation
cannot certify their legality, completeness, or information safety.

`mask_action_logits` reads only supplied logits and masks. It neither infers
additional moves nor samples an action. Empty legal tuples remain all-false
terminal/inactive masks, and probability masking rejects all-false rows. Filter
terminal rows before policy selection; do not invent a fallback action.
Engine validation still applies to every submitted move. These operations do
not relax the visibility rules for future hidden-information games.

## Policy and value network inputs

`TicTacToeActorCritic` receives only the encoded player-visible observation
tensor. Both policy and value heads share these features; neither accepts an
engine, authoritative state, rewards, seeds, or privileged critic input. The
value perspective is the observation's player, not automatically the current
actor. Identical encoded inputs give identical predictions for fixed parameters.

Model initialization uses caller-controlled PyTorch randomness. Forward neither
reads nor advances engine/agent RNG and does not sample actions. Apply legal
masking separately using that player's supplied choices. Terminal views keep
visibility restrictions; their predictions do not authorize action selection,
and future trajectory code must use zero bootstrap at true termination.
Tensor validation cannot repair information leaked by an incorrect observation.

## Trajectory collection and learning targets

The Tic-Tac-Toe collector is trusted orchestration: it can execute engine steps
and read per-seat incremental rewards. Models receive only encoded detached
player views; neither receives full StepResults, private events, state, seeds,
reward tuples, or the other player's observation. Cutoff value inference uses
that seat's filtered observe, including inactive seats. Terminal inference is
skipped and zero bootstrap is supplied independently of predicted values.

PlayerTrajectory stores only that seat's encoded decisions, legal masks,
action indices, old log probabilities/values, and own interval rewards. The
combined TicTacToeRollout and total_rewards stay in trusted learning code;
they are not public exports or inputs to other players. Frozen records contain
mutable tensors, independently owned by the collection result.

Each seat has an independent sampling generator, separate from engine and
global RNGs. GAE consumes one player's episode segment and bootstrap scalar;
it must never mix player perspectives or cross episode boundaries. This does
not introduce a privileged critic or change reward/event visibility rules.

## PPO updates and checkpoint boundaries

PPO uses stored player-visible tensors and their supplied legal masks as model
inputs. Old log probabilities, GAE advantages/returns, and reward-derived targets
are detached learning signals in trusted orchestration; they are not extra policy
or critic features. GAE runs within a seat/episode segment before batches combine.
Only samples collected with the learner policy belong in that policy's PPO batch;
caller orchestration owns this provenance and must exclude other-policy opponents.

PPO checkpoints contain model weights, Adam state, configuration, counters, model
mode, and private minibatch shuffle RNG state. No engine, authoritative state,
observations, rewards, rollout history, or engine/sampling seeds are exported.
Checkpoint loading uses restricted weights-only loading and format validation;
it returns a fresh trainer on the requested CPU/CUDA device without modifying
an existing model or global RNG.
These helpers do not change player observation or privileged-critic boundaries.

## Self-play schedule and CUDA placement

Training orchestration snapshots the current policy for CPU inference and uses
only filtered observations/legal choices to collect each seat's samples. Both
self-play seats share those weights; no privileged critic or engine reference
reaches the model. Device transfers copy only already-permitted tensors and
learning targets. Engines and sampling stay on CPU, while PPO can run on CUDA.

Portable checkpoints continue to exclude engine/rollout data; model/Adam tensors
serialize on CPU and restore to explicitly requested hardware. Training manifests
contain a reconstructive root seed and logs contain trusted learning diagnostics
and self-play outcomes. These files belong in a trusted training output directory;
they are not player inputs or a generic public export for future hidden games.
No action reveals an unresolved chance outcome and no new full-state API exists.

## Frozen policy evaluation

The evaluation policy adapter receives only the existing observation/legal-action
inputs. It snapshots supplied CPU model weights, uses no engine reference, and
keeps private sampling RNG separate from engine/global generators. Player-relative
encoding and fixed row-major action indices retain their existing boundaries.

Checkpoint evaluation is trusted orchestration, with one immutable byte snapshot
for all cases. Existing outcome classification uses the terminal public board.
Reports contain trusted checkpoint identity, sampling mode, RNG seeds, lengths,
outcomes, and fixed error categories; they omit observations, private events,
authoritative state, model weights, and exception text. Separate evaluation seed
namespaces do not authorize sampling inaccessible hidden worlds or leaking state.
No source checkpoint, training configuration, or training log is modified.

## Analysis artifacts and controls

Analysis uses frozen player-visible policy inputs and the existing fixed-opponent
runner/outcome adapter. Baseline replacement changes only the learner, preserving
case/engine/opponent seeds; it grants no engine access to agents. Statistics read
classified per-seat outcomes, never private state or privileged value features.
Plots use trusted training diagnostics and result summaries. Raw reports contain
reconstructive seed metadata and belong in trusted output directories. Public
figure copies contain aggregate diagnostics/results only, no observations,
private events, engine snapshots, or model weights. Source files are read-only.
