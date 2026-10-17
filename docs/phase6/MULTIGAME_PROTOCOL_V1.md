# Phase 6 shared learning and unseen transfer: frozen protocol v1

Declared 2026-10-17, before operator execution. Functionality is implemented;
quantitative learning quality is unmeasured. This protocol supersedes the shared
learning proposals in the earlier independent functional milestone, without changing
its settings, results or saved artifacts. Phase 6 remains open until its strength,
cost and adaptation evidence has been collected and reviewed, including failures.

The [numeric declaration](multigame-v1.json) is byte-copied to package protocols.
The production command rejects changes, including type changes. A different study
requires a new protocol identity, fresh output and fresh final evaluation roots.
Disposable tests call lower-level APIs with separately identified tiny settings;
they are never experiment evidence. No operator jobs were executed by Codex.

## Games and information

Development games A/B/D are Tic-Tac-Toe, Guildworks and Connect Four. All three
participate in shared pretraining. The unseen game C is **Breakthrough 5x5 v1**,
selected and frozen here before shared fitting. None of the development games
counts as unseen. Developing C's rules/encoders and testing them with disposable
models is permitted; no C trajectory, loss, evaluation result or optimizer state
enters development-game training or model selection.

Breakthrough has five rows/columns; seat 0 occupies the bottom two ranks and
moves upward; seat 1 occupies the top two and moves downward. Both have ten pawns.
A pawn moves one rank forward: straight only into an empty square, diagonally
into an empty or enemy square. Capture replaces an enemy. Reach the opposite
home rank or leave the opponent without a move (including elimination) to win.
Seat alternates, no draws/chance/pass. Only termination rewards +1/-1; no shaping.
At most 80 moves: each move advances a surviving pawn, and captures remove future
advances. Observations expose the immutable public board and actor, including
inactive/terminal seats; supplied source/target actions contain no future outcome.
Starting seats alternate across absolute game indices. Guildworks keeps its
existing round/actor rules rather than inventing a starter permutation.

Policy and value receive only each player's filtered observation tokens, supplied
candidate features and private recurrent memory. Guildworks consumes newly visible
history events in order followed by a snapshot at each own decision, using the
existing 52-feature visibility-tested encoder. The public games use their six
observer-relative board planes; Breakthrough rotates seat 1's board and action
coordinates by 180 degrees. No other seat's hidden hand, deck, seed, reward tuple,
engine, authoritative clone, teacher, solver or baseline reaches forward inference.
Full events/returns remain trusted orchestration. Evaluation is separate code and
cannot provide trajectories to this optimizer.

## Architecture and baselines

Every arm uses `multigame-visible-gru-candidates-v1`: per-game linear observation
projection to 32 tanh features, one shared GRU(32,32), per-game candidate projection
to 32 tanh features, per-game query projection and own-return value head. For each
supplied candidate, score `tanh(candidate_projection) dot query(context)/sqrt(32)`.
Softmax ranges only over the supplied tuple; selection returns that tuple entry.
Reordering candidates reorders logits; empty tuples remain empty and cannot act.
Features are one-hot cell (9), Guildworks kind/parameter (8), column (7), or
concatenated one-hot rotated source/target (50). No padding or inferred moves.

Independent comparison arms each have their own entire model, optimizer and
memory, trained on one development game. Their active adapter and backbone have
the same capacity as shared. All four adapters are allocated in every arm from
the outset, so report allocated parameter counts and the three-model total;
unused adapters receive no gradient or Adam history. Existing original feedforward
Tic-Tac-Toe/Connect Four and recurrent Guildworks learners and their results remain
available unchanged. They are historical functional baselines, not substituted
for the matched architecture arms in this study.

Both adaptation arms start from the same seed's initialization. Pretrained copies
only shared-pretraining weights; fresh keeps its original random weights. The C
adapter has identical untouched random weights in both. Both reset Adam; neither
carries pretraining optimizer moments. Adaptation updates the shared GRU and C
adapter, retains the original A/B/D adapters, and collects C latest-policy
self-play only. There is no rehearsal. This makes before/after A/B/D evaluation
an explicit forgetting test. Fresh A/B/D performance is descriptive only.

## Learning, sampling and scales

One complete latest-policy self-play game per active game per update, in fixed
order Tic-Tac-Toe/Guildworks/Connect Four. Each update freezes one CPU policy copy
for both seats; every seat has separate memory and private sampling RNG. No
historical opponents, teachers, baseline training opponents, replay, tactical
bonuses or privileged critic. After C adaptation begins, the sampler uses C only.
Two complete seat episodes/game; full BPTT from zero at every PPO epoch, with no
stored hidden-state shortcut or cross-seat/episode concatenation. No live cutoff
can return a learning record. Same-player decisions and opponent-terminal losses
accumulate into the last own-decision reward interval correctly.

Fixed reward/value divisors: 1 for Tic-Tac-Toe, Connect Four and Breakthrough;
36 for Guildworks. Its target is own points/36, not win probability or score
difference. Report raw Guildworks points separately from W/D/L; never pool raw
rewards across games. Gamma=1 on the own-decision clock, lambda=.95, true terminal
bootstrap=0 without terminal inference. Advantages normalize within each game.
Each game contributes equal mean loss regardless of its decision count, followed
by one accumulated optimizer step/epoch. Two epochs/update, Adam lr=.0003,
clip=.2, half-MSE value coefficient=.5, entropy coefficient=.01, grad norm=.5.
No shuffling is needed: all complete episodes in that update participate in each
epoch, in fixed order. Thus no shuffle RNG state is omitted from checkpoints.
Replay reuse=0; presentations are twice accepted decisions.

## Seeds, budgets and matching

Training seeds: **661101, 661102, 661103**. Paired final evaluation roots:
**662101, 662102, 662103**. No reuse of the old functional/CUDA/search roots.
Derived unsigned seeds use the first eight SHA-256 bytes, big endian, of
`boardgame-ai/multigame/v1:{root}:{index}:{stream}`. Initialization uses index 0,
stream `initialization`. Per-game absolute index is its saved accepted game count;
engine and sampling streams are `{game}/engine` and `{game}/seat:0|1`.
Separate/shared arms share these schedules; different policies can produce
different lengths and legal choices. Evaluation uses a separate root and namespace.

All jobs use CUDA, one Torch CPU thread, CPU acting/engines and GPU optimization.
Unavailable CUDA fails before output creation. Source/runtime/settings and training
hardware changes reject resume; frozen evaluation restores portably on CPU.
No CPU/GPU speed claim is assumed. Run sequentially on the same machine, without
other training jobs competing for resources. Report device/runtime identities.

| Regime | Shared pretraining | Each of 3 separate learners | Each adaptation arm |
|---|---|---|---|
| wall | 900 cumulative seconds | 300 cumulative seconds | 300 cumulative seconds |
| samples | 10,000 accepted decisions/game, cap 1,800 s total | 10,000 accepted decisions, cap 600 s | 5,000 accepted C decisions, cap 600 s |

Wall jobs have a 10,000,000-decision/game safety ceiling; reaching it early does
not count as completing the wall experiment. Every cap includes initialization,
restore, self-play, optimizer work and checkpoint publication; final budget/status
metadata overhead is reported to the final settlement boundary, not recursively
charged. Indivisible operations can overrun; report this. A partial update never
publishes. At the wall cap retain the last complete saved update, publish p100
from it, and report all failed/unsaved work as cost. No cap extensions for weak
play. Sample-arm cap stops remain incomplete and cannot be used as completed
sample comparisons.

Sample ceilings preserve complete games. If the next full game would exceed its
game's decision ceiling, log/discard that game and exhaust that game allocation;
never train on a prefix. Other games continue until exhausted. Accepted fill is
within one game's length of each ceiling (maximum gaps 8/31/41/79 respectively).
These are **matched decision ceilings**, not a claim of exactly identical realized
samples or wall cost. The separate `environment.jsonl` logs every completed
attempt before optimization, including rejected/unsaved games; interrupted partial
games have explicitly unknown decision cost. Report accepted decisions,
presentations, attempted complete-game decisions and their difference by game.

Compare shared total wall seconds with the sum of all three separate learners;
a >5% realized difference invalidates the phrase matched wall cost. Also report
the per-game decision/presentation differences in both regimes. Equal game counts,
loss, model size and GPU occupancy are not strength. Full maximum training caps
across 36 jobs sum to **21,600 seconds (6 hours)**; they are allowances, not timing
predictions. Evaluation has an independent 900-second cap/job (9 hours maximum
across 36 jobs); actual time and completeness must be reported.

## Checkpoints, curves and recovery

Fresh root: `runs/phase6-multigame-v1`. Each named job has settings/prose copies,
Git/dirty/source/runtime provenance, a hash-pinned Python source archive, immutable
initial/curve snapshots, validated CPU-portable model/Adam checkpoints, per-game
counters, environment attempts, metrics, status and cumulative budget ledger.
Never reuse an old experiment directory. Restrict Torch loading to weights-only.
No engines, observations, private history, hidden states or global RNG serialize.

`latest.pt` is atomic and authoritative. Publish it after a complete PPO update,
then append metrics. Existing tail recovery backs up torn bytes; historical gaps
remain gaps. Curve files include update numbers, so an orphan from interrupted
publication cannot overwrite another checkpoint. Single writer only. Resume uses
the same future game indices, weights and Adam; exact CPU continuation is tested.
No mid-episode/update restore. Healthy completed/pause-boundary no-op resumes
preserve bytes. Every invocation reserves the full remaining cumulative time;
hard termination conservatively spends it. Initialization crashes may leave an
unresumable directory: preserve it and report failure; do not reset its ledger.
Ctrl+C/ordinary exceptions settle actual attempted cost and retain the last
durable boundary. An exhausted reservation is not evidence of completed training.

Freeze checkpoints p0, p25, p50, p75 and p100 at the first saved complete update
crossing each declared cost fraction. Use elapsed wall fraction in wall jobs and
minimum per-game accepted decision fraction in sample jobs. p100 is the fixed
terminal budget boundary; there is no best-checkpoint selection. Store actual
costs and counts rather than pretending each point landed exactly on its target.
Curves can lack an intermediate point after an early cap/failure; report that.

## Evaluation and analysis

Explicit evaluation is legal only after a job's fixed completion. Every curve
uses 64 sampled-policy games/seat/game against RandomAgent behavior, balanced
across alternating starting seats. Shared pretraining evaluates A/B/D; each
separate arm evaluates its own game; adaptation evaluates **all four games at
every saved curve point**, including p0 and p100 for retention. Include a paired
random-vs-random control per game. No greedy sweep, search or tuning by these
results. Evaluation costs and failures never enter training budgets or targets.

Case index is `seat*1000000+index`. Streams are
`evaluation/{game}/engine|policy|opponent`; the control substitutes a private
`evaluation/{game}/control` stream for only the designated seat. All arms/curve
points use the same paired schedules. Memory resets every match. Record raw
seat/index/starter/seed/outcome/return/length, snapshot hashes, errors, partial
counts and costs, without observations or private state. Reports publish
exclusively; partial reports cannot resume or overwrite. A new final study after
tuning requires new reserved roots, not rerunning this final evaluation in place.

Report W/D/L and scores (1/.5/0), raw own returns, per-seat and balanced scores,
per-game costs, allocated parameters and optimizer presentations. Compare
shared-minus-separate per-game balanced score for negative transfer and
after-minus-before C adaptation on each old game for forgetting. Training-seed
means are the replication unit: paired effects across three seeds have descriptive
approximate 95% Student intervals `mean +/- 4.303*SD/sqrt(3)`. Fewer seeds give
incomplete evidence; never treat evaluation games as independent training seeds.
These small-seed intervals are descriptive, without multiple-comparison correction.

Predeclared hypotheses: final balanced score lower interval >.55 against random;
shared and retained score differences lower interval >-.05 for noninferiority.
Negative transfer/forgetting point estimate below -.05 is flagged; an upper
interval below -.05 supports a harmful effect. Report every failing game.
Adaptation target=.60 balanced score. Report the first **observed** curve point
meeting it, in both seconds and accepted decisions, and full curves. Fresh divided
by pretrained cost >=1.20 in both axes for every seed supports the adaptation
speed hypothesis. A missed target is censored; a p0 crossing has zero cost and no
finite ratio. Do not interpolate, substitute zero for missing data, or claim
speedup from a censored/undefined ratio. Include pretraining cost separately;
adaptation-only speed is not an end-to-end compute saving.

Quantitative Phase 6 acceptance is separate from those performance hypotheses:
all scheduled per-game strength/cost/transfer/retention results, failures and
uncertainty must be available and reviewed. A negative outcome can fulfill an
honest comparison but cannot support positive transfer. Partial/missing jobs and
censored adaptation must be resolved or explicitly reported as failed evidence;
implementation/tests alone do not complete Phase 6. `--report` reports availability
and effects without changing roadmap completion. Preserve independent baselines
and every original artifact. [Operator commands](MULTIGAME_RUNBOOK_V1.md).
