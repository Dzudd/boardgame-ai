# Repository audit and architecture feasibility assessment

Date: 2026-10-16.
Reviewed branch: `codex/documentation-reorganization`.
Reviewed commit: `e2b896d2fb94892ed78aa1bb260d87fb26be3485`.

This is a historical review of that checkout, followed by recommendations. It
does not change the roadmap, approve new experiments, or mark fixes complete.
Source line numbers below refer to the reviewed commit. The initial audit was
read-only; this document was subsequently added at the user's request.

## Confirmed findings, ordered by impact

### 1. P2: Evaluation can overwrite monitoring artifacts and retained checkpoints

**Location:** `src/boardgame_ai/cli_evaluate.py:16-21`; the eventual overwrite is
in `src/boardgame_ai/policy_evaluation.py`, `save_policy_evaluation`.

The guard protects the input checkpoint and the filenames `latest.pt`,
`training.json`, and `updates.jsonl`. It omits `monitoring.json`, `best.json`,
development reports, and other retained checkpoint snapshots.

**Trigger:** invoke `evaluate` with an output path naming one of those existing
artifacts. A disposable reproduction successfully replaced `monitoring.json`
with an evaluation report. No existing project artifact was used as the target.

**Impact:** an accidental output path can prevent training resume, destroy
checkpoint-selection evidence, or replace retained weights with JSON.
`tests/test_policy_evaluation.py:265` tests protection of the older filenames,
but does not cover the later monitoring artifacts.

**Fix:** centralize managed-artifact protection, reject checkpoint destinations,
and default to exclusive report creation. If report replacement remains supported,
make its scope explicit. Test monitoring manifests, best pointers, development
reports, and retained snapshots.

### 2. P2: An interrupted metrics append prevents valid checkpoint recovery

**Location:** `src/boardgame_ai/training/run.py:124-138` parses every log line
on resume. Lines 178-189 save the checkpoint before appending the metrics row.

**Trigger:** a process interruption during that append leaves an incomplete final
JSON line. This is different from a completely missing row, which the code handles.

**Evidence:** after a disposable one-update run, appending incomplete JSON caused
resume to raise `JSONDecodeError`. The checkpoint remained valid and unchanged.
The failure occurs even with monitoring disabled.

**Impact:** the documented checkpoint-authoritative recovery contract fails on
a normal interrupted-write case. Clean uninterrupted-versus-resumed comparisons
do not exercise this failure mode.

**Fix:** distinguish an incomplete final row from corruption elsewhere. Preserve
the damaged log, recover complete records, mark cumulative counts unknown when
necessary, and establish a safe append boundary before continuing. Add fault
injection between checkpoint publication and completion of the metrics append.
Do not silently discard arbitrary malformed history.

### 3. P2: Editing protocol prose broke the recorded experiment's resume path

**Location:** `src/boardgame_ai/training/experiments.py:40-59`.

The experiment hashes the current `PHASE45_PROTOCOL.md` and requires exact
equality with the original experiment manifest on resume.

**Trigger:** resume `runs/phase45/entropy-comparison` from this checkout after
the documentation reorganization.

**Evidence:** the saved protocol hash is
`9d55a0f8ae4c4a34fb69d7ce44508139390d5b9290fd846a79f8ef0b706f834f`;
the current document hash is
`f845ad9e1aad9f245256e3e7f3a6cbd1a67d22ff2c663dbfa0cf8ae3c7347b78`.
The resume entry point rejected immediately with
`Experiment resume protocol mismatch`, before training or writes.

**Impact:** editorial changes prevent experiment continuation and result
regeneration despite unchanged numerical settings. The mismatch does not itself
invalidate the old measurements.

**Fix:** save an immutable protocol snapshot with each experiment and validate
a versioned machine-readable configuration. Provide an explicit compatibility
path for this historical run. Do not silently replace its recorded protocol hash.

## Audit evidence, sound paths, and limits

The audit read local guidance, core project documents, relevant protocols/results,
and source/tests for engine ownership, observations, collection, GAE, PPO,
recurrent memory, search, checkpointing, monitoring, and evaluation.

No confirmed hidden-information leak, reward-seat inversion, terminal double
counting, recurrent cross-seat contamination, or evaluation-to-training sample
contamination was found in the inspected paths. Per-seat GAE, PPO clipping and
entropy arithmetic, full-episode recurrent recomputation, and the current public
search backups appeared consistent with their declared contracts. This is a
bounded audit conclusion, not a proof of general correctness.

Validation: 288 focused tests passed, including available CUDA checks. The files
were `test_rollout_and_gae`, `test_ppo`, `test_ppo_checkpoint`, `test_training_run`,
`test_guildworks`, `test_guildworks_recurrent`, `test_guildworks_training`,
`test_search`, `test_policy_evaluation`, and `test_monitoring` under `tests/`.
Initial sandbox temporary-directory setup failures disappeared on an approved
unrestricted rerun. Bytecode and pytest repository-cache writes were disabled.
The first two findings also have small disposable reproductions; the third was
verified against the existing manifest through its read-only rejection path.

The principal recorded checkpoint hashes matched the reports:

| Checkpoint under `runs/` | SHA-256 |
|---|---|
| `tic-tac-toe-cuda/latest.pt` | `5a86eced197ef40dbe1029641ed685f0d1f6acbc5ce4ce8b75416effc9fc3709` |
| `guildworks-phase5/latest.pt` | `ec27d0033a2d3d0f5360d43d269c8aaf5a185bd948a7237ca1d7df4b6411e237` |
| `monitored-seed45/latest.pt` | `b2f8e1c4c7b53f24646e13cba92e4d8ebd6ae32f60dcffa2b4f436cc1f274a80` |

No lengthy training or complete historical experiment reproduction was performed.
The full test suite was not rerun. Clean Git status was verified before and after
the read-only audit. No existing project files/artifacts were modified, and no
commit or push was made during that audit.

The completion claims are generally bounded appropriately: Phase 4 uses a minimal
random-control criterion, Phase 5 establishes functioning mechanics and learning,
and shared multi-game learning and broader search remain incomplete. No serious
unsupported phase-completion claim was identified. Functional completion should
not be mistaken for evidence that a learning method will scale.

## Does the overall idea make sense?

**Yes, as a reusable experimental platform. A strong shared multi-game agent is
a plausible research objective, but the current results do not establish a
reliable path to it. Automatically playing arbitrary unseen board games is a
much larger goal that the current architecture does not address.**

These goals need distinct success criteria:

| Goal | Assessment of this repository |
|---|---|
| Reuse engines, simulation, evaluation, and training machinery across implemented games | Sensible and already partly demonstrated |
| Train a separate competent agent for each supported game | Plausible; current strength evidence is weak and inconsistent |
| Train one set of weights across several supported games | Feasible to investigate; benefit over separate models is unproven |
| Transfer those weights to learn a new implemented game faster | A testable research hypothesis, not a consequence of code reuse |
| Play a previously unseen game from its written rules without a new adapter | Outside the current design; no rule interpretation or general game-description mechanism exists |

The distinction matters when using AlphaZero as motivation: its paper supports
the viability of a general self-play algorithm across games. It does not establish
that this repository's shared-weight transfer proposal will work.
[AlphaZero paper](https://arxiv.org/abs/1712.01815).

My judgment is that the project is worth continuing, provided the next milestones
test learning hypotheses instead of only adding features. Failure is not certain;
success should not be inferred from a growing count of working games and tests.

## Architecture worth keeping

The engine owns rules, chance, and secrets, while the agent receives a filtered
observation and legal choices. This is the right reusable boundary. Atomic
decisions with an explicit next actor accommodate Guildworks' consecutive actions
without forcing every game into an alternating-turn assumption.

Seat-indexed rewards and observer-relative values are also good foundations.
The separate evaluation path, immutable behavior snapshots during collection,
private random streams, and update-boundary checkpoints make experiments much
easier to reason about. The audit defects are repairable reliability problems;
they do not call for replacing the engine architecture.

Keeping PPO and search-target learning separate is a sound choice. A move chosen
by search does not have the raw policy's behavior probability, so feeding it into
the existing PPO batch would violate the current learning contract.

There is precedent for reusable game research infrastructure: OpenSpiel provides
a framework spanning multiple game types and learning/planning methods. That
supports the platform idea, without implying one learner will suit every game.
[OpenSpiel paper](https://arxiv.org/abs/1908.09453).

## Where the present direction is most likely to fall short

### 1. Latest-policy self-play can become a narrow training distribution

The strongest warning comes from this project's own evidence. The original
Tic-Tac-Toe policy became nearly deterministic in self-play, while losing almost
every match against the heuristic. The diagnostic sample visited only 29 distinct
acting-seat boards. Correct gradients on that distribution do not establish
competence on other reachable positions. See [policy diagnosis](../../POLICY_DIAGNOSIS.md)
and [Phase 4 results](../../PHASE4_RESULTS.md).

This does not prove that latest-policy self-play or PPO must fail. It does show
that adding capacity, games, or training time is not yet an evidence-backed fix.
A shared model could learn several narrow conventions at once without acquiring
robust strategy.

Within the current latest-policy-only constraint, use a bounded comparison of a
clearly specified learning change, such as the already-designed search-target
learner, with independent evaluation and coverage measurements. Historical-policy
evaluation is another possible diagnostic; it need not provide training samples.
An opponent pool would be a separate research choice requiring a deliberate
change to the current plan, not a prerequisite asserted by this review.

### 2. The reusable engine does not yet imply a reusable learner

`training/ppo.py:91-98` fixes observation/action shapes to Tic-Tac-Toe.
`training/guildworks.py:57-63` fixes each seat episode to 16 decisions;
line 212 reshapes advantages using that count. These are reasonable bounded
implementations, but are not a general multi-game learning interface.

Before adding a third largely independent training stack, define the smallest
shared boundary needed by two games: model outputs, legal masks, variable-length
seat sequences, bootstrap/termination metadata, behavior-policy provenance, and
game/encoding identity. Keep encoders and rule-specific validation in adapters.
Avoid a speculative universal framework; prove the boundary with concrete games.

A shared backbone needs an explicit game identity and interpretable per-game
inputs, plus either per-game policy heads or candidate-action scoring. Merely
padding incompatible action vectors to a common width does not give actions
shared meaning. Candidate scoring is useful for variable actions, but is not
necessary to prove an initial shared model with small fixed-action games.

### 3. Reward objectives must be defined before sharing optimization

Tic-Tac-Toe trains terminal win/loss return. Guildworks trains accumulated own
points, although matches are won by outscoring the opponent. That distinction is
documented and is not a reward-ownership bug.

However, maximizing expected points need not maximize win probability. A move
that denies an opponent points may improve the chance of winning without improving
own point return. Decide explicitly whether each agent is intended to maximize
points, score difference, or outcome. Do not silently change rule rewards while
claiming continuity with old experiments.

For shared optimization, track per-game losses and gradients as well as sampling
rates. Guildworks' return scale and sequence length differ substantially from
Tic-Tac-Toe's. Raw concatenation can weight games unintentionally. Declare task
weighting and value-target scaling, preserve raw evaluation scores, and compare
against matched independent models.

### 4. Hidden-information search is a distinct research problem

A GRU can summarize permitted history; that alone does not provide a calibrated
belief over unseen cards or an information-set-aware planner. Likewise, a sampled
hidden world must not let later simulated decisions exploit secrets unavailable
to the simulated player.

The current PUCT conversion at `search/puct.py:129-130` clips a scalar value and
assigns its negative to the other seat. That is appropriate for the declared
zero-sum Tic-Tac-Toe probe. Applying it to Guildworks' own-point values would
invent opponent returns. Hidden/chance search needs explicit beliefs, information
sets, chance semantics, and per-seat utilities before implementation.

ReBeL illustrates that combining search and learning for imperfect-information
games uses a substantially different formulation; its results are not a license
to apply ordinary public-state PUCT to hidden cards.
[ReBeL paper](https://arxiv.org/abs/2007.13544).

### 5. MuZero adds an unsolved component before its need is established

The current supported games already have cheap exact transition rules. Learning
dynamics would introduce model error and new training/resume machinery before
the project has established robust gains from its simpler search-learning design.

MuZero predicts planning-relevant rewards, policies, and values through learned
dynamics. That is a meaningful research direction, but its published results do
not show that it is the best next engineering step here.
[MuZero paper](https://arxiv.org/abs/1911.08265).

My recommendation is to keep it as a later comparison. First determine whether
search-guided self-play with the available exact public simulator improves the
chosen strength metrics at an acceptable cost. This sequencing recommendation
does not change the roadmap by itself.

### 6. Functional milestones are necessary but insufficient learning gates

The documents correctly avoid claiming that Guildworks' 256-game run establishes
strength or a benefit from memory. Nevertheless, proceeding through many such
functional milestones can leave the central hypothesis untested.

Before a substantial shared-model investment, require a bounded independent-model
baseline with repeatable external evaluation. It need not solve Tic-Tac-Toe or
beat every opponent. It should show enough stable competence to make a shared
versus separate comparison meaningful. If both models are weak, equal performance
does not establish useful transfer.

Memory benefit needs a matched history-versus-current-observation comparison.
Transfer needs adaptation on a held-out implemented game against training from
scratch, with pretraining cost reported separately. Using three hand-selected
games would provide an initial result, not evidence about arbitrary board games.

## Recommended bounded sequence

1. **Repair experiment reliability.** Close the three confirmed findings and add
   targeted failure-path tests. Preserve existing artifacts and protocol identities.
2. **Declare one near-term learning question.** For strength, a bounded public
   search-target learner versus PPO is well motivated by the existing design.
   For multi-game reuse, an independent Connect Four baseline is reasonable, but
   should exercise a reusable training boundary and include a strength assessment.
   Do not introduce shared weights, learned dynamics, and hidden search together.
3. **Freeze a small comparison protocol before running it.** Use several training
   seeds, both seats, fixed evaluation opponents, fresh final seeds, and separately
   reported environment steps, optimizer work, and wall time. Predeclare the primary
   metric and what result would stop or redirect the experiment. Numerical budgets
   should follow a short throughput measurement rather than an invented estimate.
4. **Test a minimal shared model only after independent references exist.** Begin
   with small compatible adapters and explicit task balancing. Compare per-game
   strength and learning curves, not just an aggregate score. Record negative
   transfer and forgetting. A large Transformer is not an acceptance criterion.
5. **Treat hidden search and MuZero as separate extensions.** Require dedicated
   correctness and empirical gates for each. Do not use success in deterministic
   public games as evidence that those gates have been met.

These are recommendations for a future protocol decision, not approved experiments
or replacements for the current task lists. Teachers, baseline training opponents,
and historical opponent pools remain outside the existing training plan.

## Final assessment

The project has a credible software foundation and a worthwhile experimental goal.
The main risk is accumulating implemented mechanisms without demonstrating that
the learning process produces robust, transferable behavior. The current evidence
supports a functioning research platform; it does not yet support a strong general
board-game agent.

Keep the engine and information boundaries. Fix artifact safety and recovery.
Then choose a small experiment that can falsify the next learning hypothesis.
If shared learning fails a fair comparison, retaining a common platform with
separate agents is still a useful outcome; it should be reported as such.
