# Bounded public search-target learner

Date: 2026-10-16. This implements the separate learner from
[the bridge design](../reference/SEARCH_LEARNING_DESIGN.md), following the
[reliability milestone](2026-10-16-reliability-resolution.md).
It is functional implementation evidence, not an experiment or strength result.

## Implemented scope

CPU-only, feed-forward Tic-Tac-Toe with exact public-rule PUCT. Both seats share
one frozen evaluator throughout each collection batch. Root noise and visit
sampling use existing private search streams, seeded from absolute game indices
in a separate `search-learning/v1` namespace. No tree reuse, authoritative engine
clones, hidden search, teachers, baseline training opponents, historical opponent
pool, shared backbone or learned dynamics are introduced.

The collector records permitted encoded observations, legal masks, integer root
visits, selected action, seat/global/own decision indices and complete own-seat
outcomes. Games retain public incremental rewards, policy version/hash, absolute
index, search settings and evaluator-call count. Public trajectory reconstruction
checks legality, completion, visit totals and outcome perspective. Every real
decision contributes exactly the configured simulation budget.

The objective is mean search-visit cross entropy plus half-MSE to undiscounted
own-seat terminal outcome. Illegal log terms are masked before multiplication;
inputs/targets must be finite and detached, with normalized legal target mass.
Gradients and updated parameters must be finite. PPO loss, rewards, models and
checkpoint formats are unchanged. The existing architecture is reused with a
distinct search-outcome learner/model checkpoint identity.

Replay contains only completed learner-generated games. FIFO capacity counts
games; each optimization position samples a game uniformly, then a decision
uniformly, with replacement. This differs from uniform sampling of all decisions.
Defaults match the bridge proposal: 10,000 games, batch 256, four Adam steps per
64 collected games, 32 simulations per decision. Adam learning rate is .0003 and
gradient norm limit .5. These are functional defaults, not a frozen scientific
protocol or approved run budget. Immutable `unroll_steps=0` rejects unsupported
sequence unrolling.

`latest.pt` atomically stores model, Adam, private replay RNG, replay games,
configuration, counters, model mode and latest report. Loading uses
`weights_only=True` and validates identity, tensors, Adam, public replay,
absolute schedules and boundary counters before restoring a fresh trainer.
PPO/search formats reject each other. Exact continuation requires the same
runtime, CPU/thread configuration and source implementation.

After checkpoint publication, `update-NNNNNNNN.json` is published immutably.
Resume retries that report from the checkpoint without repeating training.
Existing mismatched reports reject without overwrite. A partial collection or
optimizer failure requires reloading the authoritative saved boundary. There is
no partial-game/minibatch resume or concurrent-writer support.

## Files and validation

New modules under `src/boardgame_ai/training/` are `search_learning.py`,
`search_checkpoint.py`, and `search_run.py`; regressions are in
`tests/test_search_learning.py`. README, architecture, information boundaries,
bridge design, roadmap and progress describe the implemented scope.

Initial focused validation: **133 passed, 1 skipped** across search learning,
public search and reliability tests. Final learner-specific validation: **50 passed**.
Final full suite: **924 passed, 1 skipped in 52.00 seconds**, including existing
CPU/CUDA regressions. Ruff lint/format, strict mypy (120 files), `pip check`,
new CLI help and whitespace checks pass. The skip is the existing Windows
symlink-privilege check; it does not affect hard-link or learner tests. No hosted
CI is claimed. All 168 existing run artifacts remain unchanged by hash.

Tests use at most six collected games per run, four simulations per
decision (the CLI smoke uses two), and disposable outputs. Sampling-weight
diagnostics draw positions from two fixed scripted games; they do not train or
evaluate an agent against external opponents.

Coverage includes hand-computed loss/illegal gradients; X/O wins and draws;
public trajectory/visit/outcome validation; frozen policy identity; private/global
RNG isolation; uniform-game/position weighting; FIFO/replay age; exact
uninterrupted/resumed model, Adam, replay, RNG, counters and metrics; interrupted
Adam, checkpoint replacement and report publication; malformed checkpoints;
format separation; CLI, zero-budget and incompatible-schedule rejection.

## Limits and next experiment

No full experiment, strength evaluation, throughput benchmark, CUDA learner,
generic checkpoint evaluation integration, sequence learning or phase-wide
completion is claimed. The trained CPU model supports `TicTacToePolicyAgent`;
generic `evaluate` still accepts PPO files only. Replay checkpoints intentionally
contain trusted public training records and reconstructive configuration; they
are not public exports or policy inputs. Full replay validation at every save
prioritizes correctness and may be costly at default capacity.

The next task is to freeze the comparison protocol proposed in the reliability
resolution: independent search-target learning versus unchanged PPO, at least
three paired training seeds, both-seat random/heuristic metrics and fresh final
seeds after development selection. Short, separately capped throughput measurement
must establish numerical game/decision/search/optimizer/evaluation and time
budgets first. Stop at those caps; require a positive paired primary interval
and predeclared heuristic seat-regression limits to advance. No such measurement
or campaign was executed here; no experiment budget is inferred from tiny tests.
