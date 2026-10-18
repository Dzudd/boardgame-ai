# Search-learning bridge — Phase 4.5 Task 5

This document retains the search/learner design rationale, prospective learning
contracts and required tests. Implemented search details are identified separately;
the shared [architecture](../../ARCHITECTURE.md) describes the current system.

**Phase 4.5 design; bounded Phase 7 evaluation and CPU search-target learner implemented.**
The learner's [functional evidence and limits](../reviews/2026-10-16-search-target-implementation.md)
do not establish learning quality. Phase 7 as a whole remains incomplete.
The first milestone is bounded policy/value-guided search for deterministic,
fully public, two-seat, zero-sum Tic-Tac-Toe. PPO stays a comparison baseline.
Hidden information, chance, general-sum games and learned dynamics require
separate contracts; none is silently approximated by this design.

This project-specific proposal draws on self-play search/visit targets in
[AlphaZero](https://arxiv.org/abs/1712.01815), the representation/dynamics/prediction
separation in [MuZero](https://arxiv.org/abs/1911.08265), and the behavior-policy
ratio requirement in [PPO](https://arxiv.org/abs/1707.06347). These sources motivate
the separation; the budgets, interfaces and acceptance tests below are our choices.

## Inputs and ownership

Preserve `Agent.select_action(observation, legal_actions)` and per-seat reset.
The search agent receives no live engine, authoritative state, seed, RNG,
reward tuple, opponent view or clone. `PublicTicTacToeSimulator.from_observation`
is the game-specific adapter constructed from the supplied public board,
actor and phase. It validates seat counts, reachability, absence of post-terminal
moves, and equality of root legal choices with the supplied tuple. A small
immutable public simulator can share rule helpers with the engine, but must not
receive or inspect the real engine. Any constructor internals belong to trusted
rules code, not model features. The adapter is implemented in `boardgame_ai.search.simulator`.

The simulator owns only reconstructed public board data and implements legal
placements, public next observations, terminal status and seat-indexed incremental
rule rewards. Simulated legal actions come from those rules, never a privileged
real-state mask. It contains no opponent strategy or solved values. Branch copies
copy reconstructed data only. Rule consistency must be tested against the engine
over all reachable boards, including terminal roots. Public construction does not
generalize to hidden games: reject unsupported game IDs rather than clone secrets.

The policy/value evaluator accepts the leaf actor's encoded permitted observation
and supplied simulated legal actions. The value head retains observer perspective.
The initial PPO checkpoint may be used as a **heuristic** evaluator: its per-own-
decision discounted GAE value is not an exact undiscounted minimax value. Record
this mismatch; clip finite heuristic values to [-1,1] for the first search probe.
Reject nonfinite outputs. No terminal network inference or privileged critic.
The later dedicated search learner predicts undiscounted own-seat game outcomes.

## Bounded search contract

Initial development settings: 32 simulations per real decision, depth at most
nine remaining placements, node cap 289, one frozen CPU model per collection
batch, no tree reuse across decisions. Compare 0 (raw policy), 16 and 32 simulations
at matched games and report actual evaluator calls and latency. The 0 case uses
the existing raw-policy adapter, not a tree with undefined visit targets.
Reject invalid/zero search budgets for the search-target learner.

Expand the live root first. For each simulation, select legal edges by
`Q(parent_actor, a) + 1.5 * P(a) * sqrt(1 + sum_b N(b)) / (1 + N(a))`.
Unvisited Q is zero; exact ties use the lowest cell index. Store N, a two-seat
return sum W, mean Q, masked normalized prior and public child metadata per edge.
Legal priors may be floored at 1e-8 then renormalized to avoid underflow starving
legal edges; report that this is a search prior transform, not a PPO policy change.
No transposition table initially: history/visit semantics stay explicit.

One simulation traverses at least one root edge, then stops at a new live leaf
or terminal node. A live leaf is expanded once from a policy/value prediction;
terminal leaves use zero continuation and their transition reward. Count exactly
one backup per traversed edge per simulation. At a depth/resource cutoff, use
the evaluated live leaf continuation and mark the cutoff; do not call it terminal.
The node cap is a hard allocation limit; evaluate the frontier without expansion
when exhausted. Depth bounds decrease with public placements, so cycles reject.

Use a **two-seat return vector** throughout backup. A live leaf value v for seat s
becomes G[s]=v, G[1-s]=-v. Moving backward along an edge applies
`G = reward_vector + gamma_search * G` with gamma_search=1. Edge Q uses the
component for its **parent actor**. A terminal edge has the game's (+1,-1) or
(0,0) reward and zero continuation: do not also add a terminal value/outcome.
This gives correct win/loss signs without assuming actors always alternate;
same-actor decisions never trigger an automatic negation. The first adapter still
rejects unsupported game mechanics; a general simulator must define reward timing,
duration discounting and multiplayer value vectors before extending this rule.

Self-play only: mix root prior with .25 Dirichlet noise (alpha=1 per legal action),
then search; sample actions from visits with temperature 1 for all nine plies.
Targets are normalized root **visit counts**, with all legal moves represented,
zero mass for unavailable moves. All root edge visits sum to the simulation budget.
Noise is applied once at root and is never itself a target. Evaluation uses no
noise, temperature 0, highest visit count and lowest-cell ties. Record raw-policy,
sampled-search and deterministic-search protocols separately.

Use private generators with versioned SHA-256 namespaces derived from root seed,
absolute game index, seat, real decision index and purpose (`root-noise` or
`action-sampling`). Neither simulation length nor opponent actions consume another
seat's stream. Never use engine/global RNG. Replay trace tests must reconstruct
noise, action, root visits and backups from these inputs on a fixed runtime.

## Separate search-target learning

Introduce a new `SearchSample`/learner, not a PPOBatch with relabeled actions.
For each completed self-play game store public encoded observations, legal masks,
seat, global/own decision indices, root visit distribution, selected action,
search budget/settings, policy version/hash and final own-seat outcome z.
Retain incremental own rewards and termination metadata in trusted learning storage
to verify z. Only completed learner-generated games enter the initial replay;
never use evaluation trajectories, tactical categories or solver moves.

The first learner minimizes mean `-sum_a pi[a]*log p[a] + .5*(v-z)^2`, over
supplied legal choices, with finite/normalization checks and detached targets.
Terminal outcome z is the undiscounted sum of future own incremental rewards;
Tic-Tac-Toe has only one terminal increment. No GAE, PPO ratio, old raw-policy
log probability or tactical bonus belongs in this objective. Search selects moves
with a different behavior distribution; treating them as raw-policy PPO samples
would give incorrect ratios. Preserve the old PPO learner and checkpoint IDs.

Implemented functional default replay configuration: FIFO last 10,000 completed self-play
games, uniform game sampling followed by uniform valid decision sampling, batch
256, four optimizer steps per 64 newly generated games. This deliberately weights
games equally rather than all historical decisions equally; log the sampling
distribution and age. Expose capacity, game/position weighting, batch size and
optimizer steps in an immutable manifest. No historical **opponent pool**: replay
targets may be older self-generated data, while both acting seats always use the
current frozen policy/search configuration. Log policy version and replay age.

Keep a sequence-sampling interface for later learned dynamics: sample a game and
start position, unroll K actions without crossing its terminal boundary, supply
valid-step masks and zero-weight padding. Initially K=0 for the feed-forward
public-rule learner. Future MuZero representation h(observation/history), dynamics
g(latent, action) and prediction f(latent) add reward/value/policy sequence losses,
with explicit actor/perspective and imagined legality contracts. Do not infer
hidden opponent cards or chance outcomes from true engine snapshots. Reanalysis,
prioritized replay and learned beliefs are separate extensions outside this
design's scope.

## Required tests and first implementation milestone

1. Public simulator agrees with the existing engine on every reachable position,
   legal transition, terminal reward and root supplied-action set. Mutation of
   a reconstructed branch cannot change another branch or the real game.
2. Hand-built trees verify parent-seat Q, two-seat vector backups, same-actor
   paths, terminal zero bootstrap and no double counting. A forced one-ply win
   must back up +1 for the mover and -1 for the other seat. No exact solver needed.
3. Budgets/node counts/depth caps hold, visits sum to budget, illegal mass is zero,
   multiple winning/legal ties work, nonfinite evaluations reject, and terminal
   roots return no action through the agent's documented rejection behavior.
4. Repeated seeds reproduce visits/noise/actions and preserve global RNG. Changing
   only inaccessible information in any later supported game must not change
   decision inputs or search; hidden/chance support remains disabled until tested.
5. The new learner uses only self-generated search distributions/outcomes; replay
   sampling weights/sequence masks and perspective are tested independently.
   Checkpoint replay/RNG/optimizer resume must match an uninterrupted tiny run.

**Implemented milestone:** public simulator and bounded search evaluation, with
raw greedy policy versus 16/32 simulations. See the [Phase 7 protocol](../../PHASE7_PROTOCOL.md)
and [results](../../PHASE7_RESULTS.md). Separate CPU search-target replay learning is
implemented with `unroll_steps=0`, private replay RNG and exact tiny-run boundary
resume; its frozen learning-quality comparison remains a later task.


## Implementation details

`SearchAgent` implements the shared Agent contract; `Evaluator` takes only an
acting-seat public observation and simulated legal tuple, returning legal-order
priors and an observer-relative scalar. `NeuralEvaluator` snapshots CPU float32
weights. Reconstructed immutable states have no engine reference. The constructor
validates both possible starting seats by public forward reachability, stopping at
first terminal; it checks complete root choices (order may differ).

The node cap counts retained live `Node` objects including root; edge statistics
are allocated at expansion (at most nine per node). Terminal/frontier states are
transient immutable public values, not retained nodes. No frontier `Node` is
allocated when the cap is exhausted. Depth counts traversed placements from root.
SearchResult exposes actual calls, node count, maximum depth and cutoff counts,
and root edges with legal priors, vector W and N. Normalize N by simulations for
visit probabilities; unavailable cells have no edge and zero mass.

Per-game reset seed comes from existing batch-v1 absolute game/seat schedules.
It initializes a private Random to derive 256 bits; search-v1 hashes those bits,
seat, public occupied-cell count (real decision index), and purpose using SHA256.
Noise and action sampling use independent private Random instances. Evaluation
uses neither. Same-runtime replay is supported; cross-version RNG identity is not.
Exploration also serves the separate `training.search_learning` collector. Its
root visits become targets; simulation diagnostic traces remain outside training.
Generic same-actor backup arithmetic is tested, but the
simulator deliberately supports only deterministic alternating Tic-Tac-Toe.
