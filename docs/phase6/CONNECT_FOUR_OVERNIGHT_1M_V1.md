# Connect Four overnight 1m v1

Operator-requested overnight schedule, 2026-10-16. A fresh CUDA run, initialized
from scratch, under runs/connect-four-cuda-overnight-1m-v1. Existing 100k run files
are preserved; this is not a continuation of their learned weights or budgets.

Exactly 1,000,000 scheduled self-play games: 125,000 updates of eight complete
games. Training stops at that target or 25,200 cumulative seconds (seven hours),
whichever comes first. The cap can leave an incomplete schedule, with only the
last saved update retained. Do not extend or reset the ledger. Absolute/private
seed scheduling and all collection, PPO, recovery, device and atomic-publication
contracts follow CONNECT_FOUR_CUDA_100K_V1.md. Only the protocol ID, fresh roots,
update target and training cap change. Training root is 632001; evaluation and
control roots are 632002 and 632003. All PPO coefficients remain unchanged.

Evaluation is separately requested, sampled on CPU, 256 games per seat for each
initial/final/control stage (1,536 total), with a separate 600-second cap. It
requires completed update 125,000 and final.pt. A time-cap stop is incomplete and
has no final.pt; latest.pt remains the authoritative run boundary and cannot be
passed directly to human play. No evaluation is launched automatically.

One writer only. Press Ctrl+C once in the old console and wait for its prompt
before starting this new run. Use --resume only with this new output/settings to
resume an interrupted overnight attempt; it does not reset the seven-hour cap.
The operator launches all campaigns. No speedup, completion-time or reliable
strength claim is made. The machine must remain awake for training to continue.
