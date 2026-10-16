"""Read-only checkpoint evaluation; training is never resumed or updated."""

from argparse import Namespace
from pathlib import Path

from boardgame_ai.artifact_safety import validate_evaluation_destination
from boardgame_ai.policy_evaluation import (
    FixedOpponentCase,
    evaluate_policy_checkpoint,
    save_policy_evaluation,
)


def evaluate_from_args(args: Namespace) -> None:
    source = Path(args.checkpoint).resolve()
    output = validate_evaluation_destination(args.output)

    def progress(case: FixedOpponentCase) -> None:
        summary = case.report.summary
        seat = case.learner_seat
        print(
            f"{case.opponent}, learner {'X' if seat == 0 else 'O'}: "
            f"W/D/L={summary.wins[seat]}/{summary.draws}/{summary.wins[1 - seat]} "
            f"completed={summary.completed}/{summary.attempted} "
            f"errors={summary.errors} truncated={summary.truncated}",
            flush=True,
        )

    print("Evaluating frozen checkpoint on CPU against random and heuristic opponents.")
    result = evaluate_policy_checkpoint(
        source,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        start_index=args.start_index,
        mode=args.mode,
        on_case=progress,
    )
    save_policy_evaluation(result, output)
    print(f"Saved evaluation for update {result.checkpoint_updates} to {output}")
    if any(c.report.summary.errors or c.report.summary.truncated for c in result.cases):
        raise ValueError("Report saved, but evaluation contains errors or truncations")
