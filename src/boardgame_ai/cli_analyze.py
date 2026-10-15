"""Task 8 analysis writes separate reports/figures, never model or training files."""

import json
from argparse import Namespace
from hashlib import sha256
from pathlib import Path

from boardgame_ai.phase4_analysis import (
    compare_checkpoint,
    load_training_log,
    save_comparison,
)


def analyze_from_args(args: Namespace) -> None:
    source = Path(args.checkpoint).resolve()
    log = (
        Path(args.training_log).resolve()
        if args.training_log
        else source.parent / "updates.jsonl"
    )
    output = Path(args.output).resolve()
    if (
        output == source.parent
        or output == log.parent
        or (output.exists() and (not output.is_dir() or any(output.iterdir())))
    ):
        raise ValueError("Analysis output must be a separate absent or empty directory")
    try:
        from boardgame_ai.phase4_plots import (
            plot_baseline_comparison,
            plot_training_curves,
        )
    except ImportError as error:
        raise ValueError(
            'Install optional plotting dependencies: pip install -e ".[analysis]"'
        ) from error
    rows = load_training_log(log)
    print("Comparing frozen sampled policy and baseline controls on CPU...", flush=True)
    comparison = compare_checkpoint(
        source,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        start_index=args.start_index,
    )
    if rows[-1]["update"] > comparison.policy.checkpoint_updates:
        raise ValueError("Training log extends beyond the checkpoint being analyzed")
    output.mkdir(parents=True, exist_ok=True)
    save_comparison(comparison, output / "comparison.json")
    metadata = {
        "schema_version": 1,
        "training_log_sha256": sha256(log.read_bytes()).hexdigest(),
        "logged_updates": len(rows),
        "first_update": rows[0]["update"],
        "last_update": rows[-1]["update"],
        "checkpoint_updates": comparison.policy.checkpoint_updates,
        "smoothing_window": 25,
        "gaps": sum(
            b["update"] != a["update"] + 1 for a, b in zip(rows, rows[1:], strict=False)
        ),
    }
    (output / "training-summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    plot_training_curves(rows, output)
    plot_baseline_comparison(comparison, output)
    effect = comparison.random_opponent_effect
    if effect.ci_low is not None and effect.ci_high is not None:
        print(
            f"Balanced score improvement vs random control: {effect.mean:+.4f} "
            f"(approx. 95% CI {effect.ci_low:+.4f} to {effect.ci_high:+.4f})"
        )
    else:
        print("Insufficient games to estimate comparison uncertainty.")
    print(f"Checkpoint-level improvement supported: {comparison.improvement_supported}")
    print(f"Saved plots and comparison reports to {output}")
