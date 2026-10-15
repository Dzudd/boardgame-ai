# Current progress

Updated: 2026-10-15. This file describes current state; the
[roadmap](BOARDGAME_AI_PROJECT_PLAN.md) owns task lists and acceptance criteria.

## Status

| Area | State |
|---|---|
| Phases 1–3 | Complete: contracts, Tic-Tac-Toe, simulation, baselines, human CLI, replays |
| Phase 4 | Complete under its original random-baseline criterion |
| Phase 4.5 | Six tasks planned; no implementation started |
| Phases 5–8 | Remaining mechanics, multi-game, search, and product work in the roadmap |

**Next task: Phase 4.5, Task 1.** Audit the frozen policy's public-board immediate
wins/blocks and trace representative per-seat rewards, GAE, value targets, and PPO
updates. Diagnose before changing training. No weight changes or solver teacher.

## Evidence and concerns

- The user completed 1,000 CUDA updates × 64 games, seed 42 (64,000 self-play
  games; 29,020 optimizer steps). Latest shared policy plays both seats.
- The matched comparison completed 12,000 games without errors. Balanced score
  improvement over random versus random opponents is +0.3160, approximate 95%
  interval [+0.2644, +0.3676]. See [Phase 4 results](PHASE4_RESULTS.md).
- The policy loses almost every heuristic matchup and is worse there than the
  random control. Final-25-update self-play X wins reached 99.81%; entropy fell
  to 0.1455. The cause is undiagnosed. Low loss and X dominance are not strength.
- Evidence covers one training seed. No claim of robust or optimal play.
- Search, learned dynamics, replay learning, checkpoint archives, automatic
  training-time evaluation, and hidden-game learning are not implemented.
- Training stays self-play only. Baselines are evaluation tools; no expert
  labels, perfect-play training, or historical opponent pool is planned.

## Throughput observation

CUDA selection is verified on RTX 4070 Ti with PyTorch 2.14.1+cu130. Games and
inference run sequentially on CPU; PPO can use CUDA. A temporary probe used two
fresh-policy updates per device, 64 games/update, default PPO, seed 42, eight CPU
threads, CUDA first. Second-update measurements (483 decisions, 32 optimizer steps):

| Stage | CUDA run | CPU run |
|---|---:|---:|
| Collection including inference | 0.176 s | 0.208 s |
| GAE / batch preparation | 0.042 s | 0.040 s |
| PPO optimization | 0.183 s | 0.068 s |
| Checkpoint saving | 0.007 s | 0.004 s |
| End-to-end update | 0.413 s | 0.325 s |

First-update totals were 2.180 s CUDA / 0.313 s CPU, including startup. This is
one post-warm-up sample per device, with synchronization instrumentation and
order/warm-up effects; it does not establish sustained speed or a bottleneck.
Temporary outputs were removed; the existing training checkpoint was untouched.
Phase 4.5 Task 3 covers repeated profiling and semantics-preserving optimization.

## Validation and latest change

Last code validation: **617 tests passed**, including CUDA checks; Ruff lint and
format, strict mypy, and pip check passed. The encoding audit checks 5,478 reachable
boards, both perspectives/seat relabeling, and 958 terminal boards.

Latest change: consolidated documentation into current contracts, status, and
checklists; removed superseded status commentary and duplicated plans while preserving
useful contracts, decisions, examples, and measured reports in their relevant files. No application,
training configuration, checkpoint, or test changes. Documentation links,
headings, task counts, and whitespace checked; tests were not rerun for this edit.

CI configuration checks Windows/Linux CPU environments on pushes/PRs. See
[GitHub Actions](https://github.com/Dzudd/boardgame-ai/actions) for hosted status;
local validation is not a claim that the current PR's hosted checks passed.
