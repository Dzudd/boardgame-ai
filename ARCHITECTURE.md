# Architecture

Current implemented contracts and design decisions. Task status belongs in the
[roadmap](BOARDGAME_AI_PROJECT_PLAN.md); evidence and concerns belong in
[progress](PROGRESS.md). [Information boundaries](INFORMATION_BOUNDARIES.md)
are normative for every engine, agent, model, and future planner.

## Package and dependencies

Python 3.12 is the development baseline; package metadata permits newer versions.
The package uses the src/boardgame_ai namespace, setuptools, venv/pip, NumPy 2.x
and PyTorch 2.x (minimum 2.6). Dependency ranges are not a lockfile or cross-version
determinism guarantee. It ships py.typed and uses strict mypy, Ruff, pytest, and
Windows/Linux CI.
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

## Connect Four

`games.connect_four` implements a deterministic 7×6 gravity board through the
shared GameEngine contract. Each atomic `drop(column)` fills the lowest free
cell; turns alternate from configurable starting_player (default 0). Four in
any straight direction wins immediately, including on the last cell; otherwise
42 filled cells draw. Win rewards are +1/−1, all other steps zero. No chance,
pass or pop-out mechanics exist. Reset validates the seed before mutation.

Immutable public observations expose the top-to-bottom row-major board and
executed column history for active, inactive and terminal seats. Supplied legal
columns are 0–6 left to right, empty for inactive/terminal seats. Human CLI labels
and input are 1–7; its renderer draws the board above the labels. The random
opponent receives the same structured contract. Outcome adaptation reads only
the terminal public board; cloning is independent. No training/search/replay
adapter is introduced. Rule and CLI tests include shared engine conformance.

## Guildworks rules and transitions

`games.guildworks.GuildworksEngine` is a standard-library, two-seat engine.
Four rounds each contain four turns (two workers per seat); round r starts with
seat (r-1)%2 and turns alternate. Each turn is `place` then `craft`, with the
same actor for both decisions. After crafting/passing, advance to the opponent,
or clear placements/refill workers at the next round. The fourth round ends
after its fourth craft/pass: exactly 32 decisions, phase `finished`, actor None.

Setup shuffles twelve cards (four each of values 1, 2, 3) with a private seeded
Random, then deals two to seat 0 and two to seat 1. Both start with two ink and
zero points. Cards have no identity beyond value; hands are unordered multisets.
Each public space accepts one worker per round:

| Space | Immediate engine-resolved effect |
|---|---|
| inkwell | Gain two ink |
| archive | Draw up to two remaining cards privately |
| festival | Gain one point |
| lottery | Uniformly gain one, two or three ink |

Craft one owned card: pay ink equal to its value, gain value+1 points, and reveal
it in the ordered public discard. Passing is always available in `craft`, even
with affordable cards. No hand limit, reshuffle, income, interrupts or extra turns.
Empty draws take zero cards and still proceed to craft; a partial draw takes the
remaining cards. No placement is offered on an occupied space. Worker counts
guarantee a free space at every live placement decision.

Actions are `place(space: str)`, `craft(card: int)`, or parameterless `pass`.
Placement ordering follows the table; crafting lists distinct affordable own
values ascending, then pass. No action identifies a future chance outcome.
Incremental rewards are each seat's points gained on that decision, zero for
other effects; there is no terminal reward bonus or loser penalty. Summed episode
rewards equal final public scores. Higher score wins; equal scores draw with no
tie-breaker. `guildworks_outcome` returns actual scores and this classification.
`render_guildworks` consumes only a filtered observation; engine rendering delegates
to it. Existing runner/batch/evaluation interfaces need no changes.

`cli_guildworks` adds human-versus-random terminal play via the existing runner.
Human agents select numbered supplied actions; AI reporting exposes only public
chosen actions. Round starts still follow the engine; `--starts` assigns the human
to seat 0 or 1. Default opponents are game-specific (Tic-Tac-Toe heuristic,
Guildworks random). Guildworks rejects heuristic opponents and replay saving.

## Guildworks recurrent learning

`models.guildworks` defines `guildworks-visible-sequence-v1`: each seat starts
from zero memory, consumes newly visible history events in order, then one current
snapshot on each own decision. No events are consumed twice; there is no history
truncation. Two seats have independent cursors and GRU state, including consecutive
decisions and repeated actors across round boundaries. Reset clears both memory
and history; changing seat or replacing a consumed history prefix rejects.

Every float32 token has 52 fields: two token-kind flags, 34 snapshot fields and
16 event fields, with the unused block zero. Snapshot fields, in order, are round
(/4), place/craft/finished one-hot, actor free/self/other one-hot, four placements
in space order (each free/self/other), own/other workers (/2), ink (/32), scores
(/36), own counts of values 1/2/3 (/4), own/other hand counts (/12), deck count
(/12), and discard value counts (/4). Discard order is retained by craft history.
Event fields are draw/card/place/points/ink/craft/pass/round one-hot, owner
free/self/other one-hot, numeric value (/4, zero for a string), and space one-hot.
Private card receipts imply the observer as owner. All scales are fixed, with
no clipping or normalization using hidden state. Inactive/terminal snapshots
encode normally, but are never selected for action or terminal value bootstrap.

Indices 0–3 place at inkwell/archive/festival/lottery, 4–6 craft values 1/2/3,
and 7 passes. Masks encode only supplied legal actions; decoding returns the
supplied object. Empty choices produce an all-false mask; sampling them rejects.
This small fixed mapping is game-specific, not Phase 6 candidate-action scoring.

`GuildworksRecurrentPolicy` is GRU(52,32), a linear eight-logit policy head and
an unbounded own-points value head. No dropout or privileged critic. The frozen
CPU `RecurrentGuildworksAgent` implements the existing Agent contract with a
private sampling generator. Forward inference carries detached hidden state;
models receive only visible tokens and that state. `initial_model` seeds CPU
initialization inside fork_rng and preserves global RNG. Rule modules remain
independent of Torch; the observation record has its own framework-free module.

`training.guildworks` collects complete 32-decision games, 16 decisions per seat,
with one latest frozen policy shared by both seats. Own incremental rewards
accumulate until the next own decision, even if actors repeat. GAE uses a player
decision clock (default gamma=1, lambda=.95) and terminal bootstrap=0. Rewards
target own public point return; they are neither zero-sum nor win-probability
targets. Episode records contain only that seat's tokens, decision positions,
supplied masks, sampled indices, behavior log probabilities/values and own rewards.

Recurrent PPO uses whole seat episodes as minibatch units (default eight, two
epochs). Every episode is recomputed from zero with current weights on every
epoch; gradients flow through the full visible stream and losses apply only at
own decision positions. There are no stored hidden-state shortcuts, episode
concatenation, padding, truncated BPTT, replay or burn-in. Full-batch advantage
normalization and clipped PPO/value/entropy losses match the existing conventions.
Behavior checks use a current CPU copy matching collection; CPU/CUDA arithmetic
can differ during optimization. Exact CPU update-boundary continuation is tested;
cross-device numerical identity and speed are not promised.

`training.guildworks_run` provides a separate bounded CPU/CUDA command, preserving
the Tic-Tac-Toe train command and checkpoint schema. Its SHA-256 seed text is
`boardgame-ai/guildworks/v1:{root}:{index}:{stream}` (first eight bytes). Streams
are initialization/shuffle at index zero, engine/seat:0/seat:1 at absolute game
index, and evaluation at seat index; evaluation has a separate declared root.
Portable atomic checkpoints identify game/model/encoding and validate CPU model,
Adam, configuration, counters, mode and private shuffle RNG. They exclude episodes,
hidden states, history, engines and global RNG. New output must be absent/empty;
resume requires identical seed, games/update and PPO config. Latest checkpoint
is authoritative; interrupted work restarts from its update boundary. Metrics are
saved after the checkpoint, so a crash may leave a missing metrics row. There is
no mid-episode/update resume. Frozen initial/final random evaluations are separate
from PPO; manifests and metrics are trusted outputs. See
[Phase 5 results](PHASE5_RESULTS.md).

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
Gamma and GAE lambda must each be in [0, 1].
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
No partial-minibatch/episode resume exists. Optional checkpoint monitoring adds
immutable bounded snapshots and isolated development evaluation at update boundaries. CUDA is CLI default and fails if unavailable;
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
Errors/truncations remain visible and yield nonzero CLI status. No optimizer in policy evaluation. Separate bounded search evaluation is described below.
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

## Development monitoring and performance

`MonitoringConfig` is an optional `run_self_play_training(..., monitoring=...)`
argument, disabled by default. A separate immutable monitoring manifest preserves
legacy training manifests and requires identical settings on resume. At configured
boundaries, complete checkpoint bytes are published exclusively using a temporary
file and a same-directory hard link. Existing mismatched snapshots reject.
`latest.pt` remains the authoritative resume marker. New development reports are
atomic and reused by checkpoint hash on retry; the newest configured number of
snapshots plus a separately pinned best are retained. Reports are retained even
when their non-selected checkpoints age out. `best.json` selects equal-weighted
random/heuristic × X/O mean score, earliest tie, without affecting collection.
Evaluation failures preserve completed updates and can retry at the current
boundary. Training is single-writer; no concurrent run-directory writers supported.

Evaluation loads only copies on CPU and isolates/restores Python, NumPy and global
CPU/initialized-CUDA RNG, including failure paths. Training shuffle/model/Adam and
per-game/seat sampling schedules are untouched. Reports include seat-specific
outcomes and approximate game-level intervals, census/visited tactics, entropy,
position/action diversity and game/sample/optimizer counters. Missing historical
log rows make cumulative sample counts unknown. New isolated experiments never
modify the original completed run. Reserve fresh final seeds after tuning.

PPO validates/clones its incoming batch once; internal minibatches inherit checked
inputs. Public loss calls still validate independently, and every minibatch checks
model outputs, losses and clipped gradients for nonfinite values. Finite flags and
loss diagnostics are consolidated; float64 device accumulators retain the previous
Python-float weighted summation arithmetic with one host transfer per update.
Repeated reference comparisons found exact same-device model/Adam/shuffle equality.
The collector remains sequential: the fixed-corpus batched inference probe has
small numerical differences and does not prove identical sampling or end-to-end
speed. Protocol/results: [Phase 4.5](PHASE45_RESULTS.md).

## Frozen policy diagnosis

`python -m boardgame_ai.policy_audit` is a separate bounded CPU diagnostic, using
existing observation/action encoders, collector, per-seat GAE, checkpoint loader
and PPO interfaces. No training runner behavior changes. Public one-ply tactical
checks use acting-seat observations and supplied actions; all immediate wins count,
win takes precedence, blocks must remove every immediate threat, and unavoidable
threats are separate. A census counts each reachable live X-first board once;
seeded frozen-policy self-play separately reports visit and distinct-board counts.
No symmetry reduction or solver. Undefined metrics are null.

Trusted public-action replay independently verifies observations, masks, behavior
probabilities/values, both-seat interval rewards and scalar GAE/targets. The first
observed outcome traces retain seat perspective; absent outcome classes are explicit.
The first configured number of unselected self-play games supplies every decision
to a disposable update check. Two checkpoint copies retain Adam/shuffle state;
independent loss/backward/clipping/Adam arithmetic is compared with production PPO,
including parameters, moments, RNG, counters and weighted loss reports. Reports map
batch indices back to game/seat/step. No updated checkpoint is saved. Tactical
labels never enter batches. Source checkpoint/config/log bytes are checked unchanged;
new report output must be outside the source directory. Requires stopped training,
not concurrent-writer locking. Protocol and limitations: [diagnosis](POLICY_DIAGNOSIS.md).

## Extension boundaries

The [roadmap](BOARDGAME_AI_PROJECT_PLAN.md) owns phase tasks and acceptance criteria;
[progress](PROGRESS.md) identifies the current next milestone.
[Public search evaluation](#public-search-evaluation) implements bounded
Tic-Tac-Toe simulation and parent-seat vector backups. The separate self-generated
visit/outcome learner and replay learning in the
[search-learning design](SEARCH_LEARNING_DESIGN.md) remain unimplemented.
Future batched collection needs per-game/seat RNG and numerical/sampling
comparisons; changes in minibatch,
collection size or sampling schedule require learning-quality experiments.

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
belong in [progress](PROGRESS.md) rather than historical per-task test counts here.


## Public search evaluation

`boardgame_ai.search` implements the public Tic-Tac-Toe adapter, observation-only
frozen neural evaluator and bounded PUCT Agent. It does not wrap or clone an
engine. Constructor reachability/root-choice validation and independent exhaustive
engine comparison cover both starters. Public branch states are immutable.

Backups carry two seat returns; for leaf actor s, G[s]=v and G[1-s]=-v,
then each edge adds its incremental rule rewards with gamma=1. Terminal continuation
is zero. Selection reads Q for the parent actor, without turn-based sign flips.
Limits, tie-breaking, private exploration namespaces and node accounting are in
[the search-learning design](SEARCH_LEARNING_DESIGN.md). PPO values are clipped
heuristic estimates.

`python -m boardgame_ai.search.evaluation CHECKPOINT NEW_OUTPUT_DIRECTORY`
executes the [Phase 7 protocol](PHASE7_PROTOCOL.md) on frozen CPU copies with
global RNG isolation.
It reports fixed-opponent matches, full tactical census, evaluator counts and
latency; hashes verify source artifacts. No optimizer, replay or training targets.


SearchAgent's optional `on_simulation` observer receives immutable public paths,
rewards, leaf continuation, stop reason and per-edge vector returns after backup.
It is diagnostic only and does not change selection; callback exceptions propagate.
`boardgame_ai.search.diagnosis` replays prior heuristic-O match-zero records,
independently reconstructs N/W/rewards, and exports public trees and leaf predictions.
Post-decision one-step probes are separate from search calls; no diagnostic labels
or baseline trajectories enter training. The destination must not exist.
