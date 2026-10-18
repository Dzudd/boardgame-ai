# Connect Four overnight 2m v1

Operator-requested schedule, 2026-10-16. Fresh CUDA training from new weights,
under runs/connect-four-cuda-overnight-2m-v1. Existing runs and budgets remain
unchanged. The operator starts campaigns; Codex performs read-only preflight only.

Exactly 2,000,000 scheduled self-play games: 250,000 updates of eight complete
games. Stop at the target or 32,400 cumulative seconds (nine hours), whichever
comes first. A time-cap stop is incomplete; retain the last atomic boundary,
report discarded work and overruns, and never reset or extend its ledger.

Model, CPU latest-policy self-play, private absolute schedules, GPU PPO settings,
checkpoint portability, recovery and isolated CPU evaluation follow
CONNECT_FOUR_CUDA_100K_V1.md. Fresh training/evaluation/control roots are
633001/633002/633003. Fixed initial/final/control evaluation has 256 games per
seat per stage (1,536 total) and a separate 600-second cap. Evaluation is requested
separately and requires completed update 250,000 and final.pt. A capped incomplete
run retains latest.pt as a run envelope, not a directly playable trainer snapshot.

Only one writer per output; stop any old campaign with Ctrl+C once and wait for
the console prompt before starting. Resume this schedule with --resume using the
same settings/output; the nine-hour allowance remains cumulative. Keep the machine
awake. No completion-time, GPU speedup or reliable-strength claim is made.
