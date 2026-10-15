# Architecture

Current implemented contracts and design decisions. Task status belongs in the
[roadmap](BOARDGAME_AI_PROJECT_PLAN.md); evidence and concerns belong in
[progress](PROGRESS.md). [Information boundaries](INFORMATION_BOUNDARIES.md)
are normative for every engine, agent, model, and future planner.

## Package and dependencies

Python 3.12 is the development baseline; package metadata permits newer versions.
Use the src/boardgame_ai namespace, setuptools, venv/pip, NumPy 2.x and PyTorch 2.x
(minimum 2.6). Dependency ranges are not a lockfile or cross-version determinism
guarantee. Ship py.typed and use strict mypy, Ruff, pytest, and Windows/Linux CI.
The core engine layer uses the standard library and stays independent of PyTorch.
Numerical imports are deferred where possible so package imports/CLI help stay light.
Hardware-specific Torch wheel selection is an environment choice.

## Engine, records, and lifecycle

GameEngine and Agent are generic abstract interfaces. Core records are frozen,
keyword-only dataclasses with game-specific typed payloads; freezing is shallow.
Engines own copying/filtering. No full-state accessor is part of the interface.

| Contract | Meaning |
|---|---|
| GameSpec | Stable game ID/name and supported player counts, including solo games |
| GameState | Authoritative state; never an agent input |
| Observation | Observer seat, actor, phase, and independently constructed filtered data |
| Action | Kind and named scalar parameters; no unresolved chance outcome |
| StepResult | Next actor, termination, incremental seat-indexed rewards, and events |
| GameEvent | Public recipients=None; private tuple of seats; empty tuple internal |
| Agent | select_action(observation, legal_actions), optional reset(seed) |

Action parameters reject duplicate names and nonfinite floats. StepResult enforces
finite rewards and consistent terminal/actor metadata. These schema checks cannot
certify game rules, reachability, or correct filtering.

- Reset before gameplay queries, steps, or cloning; otherwise GameNotReadyError.
  Metadata is available before reset. Validate configuration before resetting state.
- Player seats are zero-based integers; booleans/invalid IDs raise InvalidPlayerError.
- step executes one atomic decision and resolves required chance before returning.
  The next actor may be the same player. At termination the actor is None.
- Every live actor has legal choices; inactive/terminal seats have an empty tuple.
  Any valid seat can observe/render after termination. Queries do not consume RNG.
- get_rewards returns the most recent step's incremental rewards, zero at reset.
  Rewards are neither cumulative scores nor generic winner/draw classifications.
- Reject malformed Action construction with ValueError; well-formed illegal moves
  raise IllegalActionError, post-terminal steps GameOverError. Rejection preserves
  state, phase, rewards, history/events, and RNG; no silent replacement or penalty.
- decision_for deep-copies observation/actions before agent delivery. Copying
  prevents mutable aliasing, not secrets incorrectly included by an engine.
- clone preserves concrete type and copies mutable/RNG state independently without
  drawing or reseeding. Tic-Tac-Toe implements this with deepcopy.

Use create_rng(seed) for a private random.Random. Integers restart its sequence;
None uses system entropy. Engine and agent RNG are separate from global Python,
NumPy, and Torch randomness. Reproducibility also requires fixed runtime, code,
action ordering, configuration, reset behavior, and call sequence.

## Tic-Tac-Toe and baseline agents

The board is a nine-cell row-major tuple of None or seat IDs. Seat 0 is X and
starts by default; starting_player=1 permits O-first and persists through reset
and clone. Placements have kind place and one integer cell parameter in 0–8.
Terminal wins reward winner +1 and loser -1; draws and live steps reward zero.
Both seats see the public board; returned observations remain stable after moves.
Rendering uses the same public views and has no print side effect.

RandomAgent samples uniformly over supplied tuple entries with a private seeded
RNG, ignores observation data, and rejects empty choices without advancing RNG.
Invalid reset seeds preserve the existing generator. Use a separate agent per seat.
TicTacToeHeuristicAgent deterministically prioritizes win, block, center, corners,
then edges; ties use row-major/lowest indices. It has no memory or RNG and does
not guarantee optimal play or fork defense. Both return supplied legal actions;
the engine still validates execution.

## Simulation, metrics, human play, and replays

run_match resets the engine and distinct per-seat agents with separate lifecycle
seeds; play_game continues initialized state. Both follow engine actors and pass
only decision_for inputs. max_steps is a required nonnegative atomic-decision
budget: zero executes nothing, live cutoff is truncation, and termination on the
last allowed step takes precedence. MatchResult stores steps, summed executed
rewards, and termination/truncation, without observations/events. Exceptions
propagate without retry; configuration is checked before resets, but custom
reset failures and executed steps are not transactional across a whole match.

run_batch resets fixed seats each match, returns immutable SeededMatchResult
records, and fails at the first exception without returning a partial tuple.
Zero matches leaves instances untouched; a subrange reproduces matching absolute
indices. evaluate_batch instead records ordinary failures and resets the next
match; interrupts propagate. An explicit game outcome adapter maps terminal games
to finite seat scores, winners, and draw status, supporting ties/cooperative games.
Tic-Tac-Toe classification uses the terminal public board and +1/-1/0 scores.

Evaluation records use fixed illegal_action/match_failed/outcome_failed categories,
never exception text/types/tracebacks. Reset/play failure leaves length/rewards
unknown; outcome failure retains the runner result. Only completed/classified
matches contribute to outcome/length means; undefined means/extrema are None.
Errors/truncations are not draws. Explicit version-1 JSON exports require existing
parents and overwrite chosen files; labels and reconstructive seeds are trusted data.

CLI play chooses human or AI starter; starter is X. It displays the board beside
available cell numbers, retries invalid human input without a step, and cancels
cleanly on quit/EOF/Ctrl+C. Baseline opponents are random/heuristic. Trained-policy
human play is not implemented. Final presentation uses the human's public view.

Opt-in version-1 Tic-Tac-Toe replay logs contain starting seat, successful public
actor/cell placements, and completed/cancelled/truncated status. Cancelled prefixes
are valid. Validation reconstructs through an engine before displaying anything;
unknown schemas/games, illegal actors/cells, post-terminal moves, and inconsistent
status reject. No seeds, snapshots, rewards, or private events are exported.
The optional on_step callback is trusted orchestration, not proof that other
games' actions are public. A recorder belongs to one game. Replays do not resume
saved games or simulate hidden chance; other games need their own adapter.

## Observation, actions, and network

| Tensor | Shape | dtype / placement |
|---|---|---|
| Observation | (6, 3, 3) | Fresh contiguous CPU float32, no autograd history |
| Legal mask | (9,) | Fresh CPU bool |
| Policy/value input | (..., 6, 3, 3) | Nonempty floating tensor matching model dtype/device |
| Raw policy output | (..., 9) | Same model dtype/device |
| Raw value output | (...) | Unbounded observer-relative estimate |

Observation channels: own marks, opponent marks, empty cells, own turn, opponent
turn, terminal. All binary; last three are constant planes. The observing seat
sets perspective, even if inactive. Coordinates never rotate: cell i maps to
(i // 3, i % 3). Live phase play requires actor 0/1; finished requires None.
Terminal views retain the board, clear both turn planes, and fill terminal with
ones. No reward/result channel. Encoding consumes only the observation, never
engine/state/actions/rewards/seeds/RNG, and shares no mutable output storage.
Schema/phase/seat/cell failures reject with sanitized ValueError; schema checking
is not reachability checking. Stack/transfer explicitly for batching/device changes.

Action encoding maps placements to indices 0–8; decoding returns the supplied
matching Action. Duplicate indices, invalid parameters/indices, and unavailable
choices reject. The legal encoder consumes supplied actions without querying rules.
Empty terminal/inactive tuples encode all-false, including boards with empty cells.

mask_action_logits requires finite floating logits and bool masks of identical
nonempty shape/device, with at least one legal entry per row. No broadcasting or
automatic transfer. Illegal entries become -inf, with zero softmax probability
and zero logit gradient; legal gradients remain connected. Inputs are unchanged.
Filter terminal/inactive rows before selecting actions; no fallback move.

TicTacToeActorCritic flattens 54 features into shared 64→64 tanh layers and separate
linear nine-logit/scalar-value heads. Both heads use the same permitted features.
There is no recurrence, dropout, running statistics, convolution, or privileged
critic. Arbitrary leading batch dimensions and noncontiguous inputs are supported.
Initialization uses caller-controlled Torch randomness and explicit CPU float32;
forward consumes no RNG and validates shape/finiteness/dtype/device. Use explicit
.to and no_grad/inference_mode for inference. Terminal predictions stay raw;
collectors independently impose zero terminal bootstrap. Masking/sampling are separate.

## Collection, credit assignment, and PPO

Training experience comes only from latest-policy self-play, without external
teachers, expert/solver labels, baseline training opponents, or a historical
opponent pool. Historical self-generated replay is a different, future mechanism.
Fixed baselines measure strength outside training; their samples never enter PPO.
No perfect solver is required. Public tactical audits are diagnostics, not labels.

Defaults: collect 64 complete empty-board X-first games using one frozen CPU
snapshot for both seats. No weights change during collection. Each player's
categorical legal-action sampling has a private CPU generator seeded with
create_rng(sampling_seed).getrandbits(64). No epsilon-random exploration exists.
Per-game model inference preserves weights, modes, and existing gradients.

PlayerTrajectory stores detached observations (T,6,3,3), bool masks (T,9), int64
actions/global step indices (T,), float32 old log probabilities/values/rewards (T,),
and scalar bootstrap. Empty segments retain these shapes. Interval rewards sum
own incremental rewards after a player's action until their next action/end,
including opponent steps; both seats receive terminal rewards. Rewards before
the first decision enter rollout totals only. Records own mutable tensors and
remain trusted learning data, not public exports.

Discount advances per player's decision, not per global atomic step. Intervening
rewards sum without internal discounting. GAE uses delta=r+gamma*next_value-value,
A=delta+gamma*lambda*A_next, returns=A+value. Never interleave seats or cross episode
boundaries. Matching finite 1-D floating inputs produce detached same-dtype/device
outputs; empty segments work and invalid coefficients/overflow reject.
True termination bootstraps zero without a model call; live cutoffs bootstrap
from that player's filtered observation, including inactive views. Empty segments
bootstrap zero. Collection resets every call; it does not resume a partial rollout.
Time-duration discounting needs an explicit future contract.

Prepare per-seat GAE before combining all on-policy decisions. Clone/detach
inputs/targets; reject empty batches, nonfinite data, positive old log probabilities,
or actions outside their saved legal masks. CPU storage transfers explicitly to
colocated CPU/CUDA float32 parameters for optimization. Old targets stay frozen.
Train on every decision in shuffled minibatches each epoch; batching is not
subsampling one or two moves per game. Longer games contribute more decisions.

| PPO setting | Default |
|---|---:|
| Adam learning rate | 0.0003 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| Policy clip | 0.2 |
| Value / entropy coefficient | 0.5 / 0.01 |
| Epochs / minibatch decisions | 4 / 64 |
| Gradient norm limit | 0.5 |
| Advantage normalization | Full-batch population std, floor 1e-8 |

Policy loss is -mean(min(r*A, clamp(r,1-epsilon,1+epsilon)*A)), with
r=exp(new_log_prob-old_log_prob). Value loss is half-MSE to detached returns, with
no value clipping. Total combines policy + weighted value - legal entropy bonus;
illegal log terms are zeroed before entropy multiplication to avoid 0 * -inf.
Singleton/constant advantages normalize to zero. Private CPU randperm shuffles
minibatches, including partial final batches. Gradient clipping rejects nonfinite
norms; failed later minibatches do not roll back successful earlier updates.
Diagnostics are sample-weighted means of pre-step losses, legal entropy,
approximate KL mean((r-1)-log(r)), clip fraction, and sample/optimizer counts.
There is no KL early stop or learning-rate schedule. These are not strength metrics.

## Seeds, checkpoints, and resume

All schedules hash UTF-8 decimal fields using SHA-256 and decode big-endian integers.
They avoid Python hash() and do not consume gameplay RNG.

| Schedule | Hashed text | Bytes used |
|---|---|---|
| Batch | boardgame-ai/batch/v1:{seed}:{index}:{stream} | All 32 |
| Training | boardgame-ai/training/v1:{seed}:{index}:{stream} | First 8 |
| Evaluation case | boardgame-ai/evaluation/v1:{root}:{opponent}:{learner_seat} | All 32 |

Batch streams are engine and agent:{seat}. Training initialization/shuffle use
index zero; game index is update_index*games_per_update+i, with engine and
seat:0/seat:1 sampling streams. Evaluation derives case seeds before the batch
schedule. Initialization uses fork_rng to preserve global CPU/CUDA state.

Version-1 PPO checkpoints fix game/model/encoding identity and store CPU model
weights, Adam state/configuration, model mode, update/step counters, and private
shuffle RNG. No engine, rollout, gradients, global RNG, or engine/sampling seeds.
Save validates then atomically replaces a temporary sibling in an existing
parent. Load uses restricted weights_only loading/strict schema validation on
CPU and restores a fresh trainer on explicit CPU/CUDA hardware, preserving global
RNG. Adam moments follow parameters; scalar step bookkeeping may stay CPU.

A new run requires absent/empty output. training.json fixes seed, collection size,
and PPO configuration; latest.pt is authoritative resume state. Save each complete
update before appending updates.jsonl/progress callback. Crashes can leave missing
log rows; resume restarts unsaved work and derives future game seeds from absolute
indices. Total requested updates cannot precede the saved count. Seed, collection
size, and PPO config must match; device changes are allowed without numerical
identity promises. Same-runtime/device uninterrupted-versus-resumed state is tested.
No partial-minibatch/episode resume, historical checkpoint archive, or automatic
training-time evaluation exists. CUDA is CLI default and fails if unavailable;
there is no silent CPU fallback. CPU execution must be requested explicitly.

## Evaluation and analysis

TicTacToePolicyAgent snapshots a frozen CPU float32 model and owns sampling RNG.
Sampled mode matches training; greedy mode picks lowest-cell maximum-logit ties.
Fixed evaluation uses random-v1/heuristic-v1, equal X/O counts, X-first rules,
nine-step bounds, root seed 20261015 by default, and shardable absolute indices.
Greedy versus deterministic heuristic repeats one path per seat, not independent trials.

Checkpoint evaluation hashes/loads one byte snapshot without modifying source
weights, logs, or configuration. Reports identify hash/counters/mode/protocol,
seeds, outcomes, lengths, and fixed errors; source and known training artifact
overwrite paths are guarded. Parents must exist; explicit JSON reports overwrite.
Errors/truncations remain visible and yield nonzero CLI status. No optimizer/search.
Held-out RNG namespaces do not imply unseen board positions.

Analysis replaces only the learner with random/heuristic controls under matching
engine/opponent/index schedules. Reject incomplete records or mismatched protocols.
The primary effect is paired policy-minus-random score against random, equal
seat weighting; SE=0.5*sqrt(SE_X^2+SE_O^2), approximate 95% interval ±1.96 SE.
Positive lower bound and at least 30 games per seat meet the original Phase 4
criterion for a checkpoint, not training-seed generalization or optimal play.

Validate strictly increasing finite log rows; missing updates break plotted lines
and restart the labeled 25-update trailing average. Self-play curves are not
held-out strength over time. Optional Matplotlib renders headless PNG/SVG loss,
entropy/KL/clipping/outcome/length curves and mean-score comparisons. Analysis
requires separate absent/empty output; preserve input hashes/raw trusted reports.
Aggregate figures/results live in docs/phase4 and dedicated result documents;
checkpoints/logs/raw reports remain ignored under runs. Once evaluation guides
tuning or checkpoint selection, reserve fresh final-test seeds.

## Planned extensions and constraints

The roadmap owns the six Phase 4.5 tasks. Diagnosis, monitoring, and throughput
optimization precede controlled learning experiments and the search design.
Profile repeated warmed-up CPU/CUDA stages and end-to-end time at equal budgets;
consolidate scalar reads/redundant checks without removing failure detection, then
assess batched inference with frozen weights and private per-game/seat streams.
Preserve targets and documented resume guarantees. Larger minibatches or changed
sampling schedules require learning-quality comparisons. CPU is valid if faster;
one timing probe is not a speed guarantee (measurements are in PROGRESS.md).

Long-term: permitted rule-based MCTS → search-guided self-play targets/replay →
learned representation/dynamics/prediction. Planning selects branches; evaluators
estimate player-relative outcomes. A future observation evaluator may consume
history; a simulation evaluator may consume an authorized simulation context
and budget. A heuristic policy or terminal outcome adapter is not a general
position evaluator. Keep PPO as a comparison, and do not treat search-selected
moves as raw-policy PPO samples with incorrect behavior probabilities.

Simulation must be reconstructed from permitted views/history, never supplied as
an authoritative hidden-state clone. sample_hidden_state is only a future design;
it must sample compatible worlds rather than expose actual secrets. Independent
simulation RNG alternatives require an explicit contract. Naive sampled-world
search risks strategy fusion. Chance, hidden information, recurrent context,
same-player backups, and multiplayer/vector returns require explicit designs.
Self-play replay must specify visibility, sequence length/context, sampling/reuse,
game-length weighting, and compute budgets. Stale trajectories are not ordinary
on-policy PPO. Shared backbones/candidate-action scoring and transfer experiments
remain future multi-game work; negative transfer must be measured.

## Verification

Engine/rejection conformance helpers are in tests/contracts.py. New games must
also test unseen-world input equivalence, observation mutation, event filtering,
seeded chance, independent clones, and rejected-action RNG preservation.
The test-only hidden-card engine is a fixture, not a production Phase 5 game.

Tic-Tac-Toe rules are audited against a coordinate-based oracle over 5,478 boards
(626 X wins, 316 O wins, 16 draws). The encoding audit checks both perspectives,
seat relabeling, terminal behavior, and lossless board recovery without state access.
Learning tests cover hand-computed GAE/returns/PPO, illegal gradients, storage/RNG
isolation, optimizer changes, corruption rejection, exact checkpoint continuation,
and CPU/CUDA resume. CUDA tests check real inputs/parameters/gradients/Adam device
placement; portable files serialize tensors on CPU. Current validation results
belong in PROGRESS.md rather than historical per-task test counts here.
