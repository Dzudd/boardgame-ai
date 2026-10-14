# Tic-Tac-Toe baseline metrics

Recorded on 2026-10-14 for Phase 3, Task 5. Each fixed-seat matchup has 1,000
matches, batch seed 20261014, indices 0–999, and a nine-decision limit. Seat 0
is X and starts. Python 3.12.14 was used. Every complete report reproduced on
rerun. All 3,000 evaluated matches (6,000 executions including reruns) terminated
with no errors or truncations.

| X agent | O agent | X wins | O wins | Draws | Mean X score | Mean O score | Mean decisions | Min–max decisions |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Random | Random | 599 | 273 | 128 | 0.326 | -0.326 | 7.657 | 5–9 |
| Heuristic | Random | 961 | 0 | 39 | 0.961 | -0.961 | 5.612 | 5–9 |
| Random | Heuristic | 10 | 842 | 148 | -0.832 | 0.832 | 6.720 | 6–9 |

The game-specific outcome adapter classifies the terminal public board. Result
scores are +1 for the winner, -1 for the loser, and zero for draws; this convention
matches Tic-Tac-Toe rewards but is explicitly separate from generic reward totals.
Means use completed, successfully classified matches only. A decision is one
atomic placement, not a round. The heuristic uses win/block/position tactics and
has no fork defense or optimal-play guarantee; it lost 10 games as O here.

These are fixed-seed baseline measurements, not multi-seed estimates with
confidence intervals, and seat advantage must be considered when comparing them.

Reproduce and save trusted JSON reports:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_baselines.py
```

The script evaluates each matchup twice, checks full report equality, and writes
`runs/baseline_metrics/random_random.json`, `heuristic_random.json`, and
`random_heuristic.json`. These generated reports are ignored by Git. They retain
per-match seeds, result scores, reward totals, errors, and lengths; they contain
no observations, private events, action histories, or replay snapshots. Seed
metadata is for trusted evaluation and must not be exposed during hidden games.
