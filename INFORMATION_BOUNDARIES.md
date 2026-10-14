# Player visibility and engine boundaries

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
