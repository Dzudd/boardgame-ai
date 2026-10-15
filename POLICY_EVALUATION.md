# Fixed-opponent checkpoint evaluation

Measured on 2026-10-15 after the user completed CUDA self-play training.

These baseline opponents are evaluation only. Their actions/rewards do not
enter training; the project's methodology uses latest-policy self-play without
external teachers or a historical opponent pool. The unchanged results below
motivate the planned Phase 4.5 diagnostics in BOARDGAME_AI_PROJECT_PLAN.md.

The checkpoint at `runs/tic-tac-toe-cuda/latest.pt` contains 1,000 completed PPO
updates and 29,020 optimizer steps. The training configuration used 64 games per
update and seed 42, for 64,000 self-play games. No additional training was run
by this evaluation, and the checkpoint hash remained unchanged.

Checkpoint SHA-256:
`5a86eced197ef40dbe1029641ed685f0d1f6acbc5ce4ce8b75416effc9fc3709`

## Protocol

- Frozen CPU policy, sampled from masked logits, matching training action selection.
- Fixed existing random-v1 and heuristic-v1 opponents, X-first engine rules.
- 1,000 games per opponent per learner seat (4,000 games total).
- Held-out root seed 20261015, independent versioned evaluation/batch RNG schedules,
  absolute match indices starting at zero. No self-play rewards or training outcomes
  are substituted for evaluation results.
- All games completed; no errors, illegal actions, or truncations were recorded.
- Game-defined +1/-1/0 result scores, wins, draws, and losses come from the terminal
  public board via the existing outcome adapter. Win percentages use completed games.

## Results

| Opponent | Learner | Wins | Draws | Losses | Win rate | Mean result score |
|---|---|---:|---:|---:|---:|---:|
| Random | X (first) | 770 | 66 | 164 | 77.0% | +0.606 |
| Random | O (second) | 484 | 73 | 443 | 48.4% | +0.041 |
| Heuristic | X (first) | 0 | 3 | 997 | 0.0% | -0.997 |
| Heuristic | O (second) | 0 | 0 | 1,000 | 0.0% | -1.000 |

The policy wins more games than it loses against random in both seats, with a
large first-player advantage. It fails almost universally against the tactical
heuristic, despite the final self-play update reporting 64 X wins in 64 games.
Self-play success therefore does not establish a robust policy. The mechanism
behind this weakness has not been diagnosed; no training configuration was changed.

These Task 7 results alone do not establish optimal play or matched-control
improvement. The completed Task 8 comparison and curves are in PHASE4_RESULTS.md.
The heuristic is the existing win/block/position agent, not an optimal solver.
No confidence intervals or multiple independent training runs are included.
Held-out RNG schedules do not imply disjoint board positions in deterministic
Tic-Tac-Toe. If this suite guides future tuning, reserve fresh seeds for final testing.

## Reproduction and artifacts

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m boardgame_ai evaluate runs/tic-tac-toe-cuda/latest.pt --output runs/tic-tac-toe-evaluation-update1000-sampled.json --games-per-seat 1000 --seed 20261015 --mode sampled
```

The generated trusted JSON report contains checkpoint identity/counters, policy
mode, seed schedule, match seeds, results, summary scores/lengths, and fixed error
categories. It contains no model weights, observations, boards, private events,
or exception text. Generated run artifacts are ignored by Git; retain them with
the checkpoint for future comparison. Use the same checkpoint bytes/runtime for
reproduction. `--start-index` permits deterministic shards; `--mode greedy` is a
separate protocol and must be labeled separately. Against the deterministic
heuristic, greedy replay repeats one path per seat rather than independent trials.
