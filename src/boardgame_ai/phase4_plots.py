"""Headless standalone PNG/SVG scientific plots (optional analysis dependency)."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from boardgame_ai.phase4_analysis import Phase4Comparison, trailing_mean


def _gaps(
    updates: tuple[int, ...], values: tuple[float, ...]
) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    for i, value in enumerate(values):
        if i and updates[i] != updates[i - 1] + 1:
            x.append((updates[i] + updates[i - 1]) / 2)
            y.append(float("nan"))
        x.append(float(updates[i]))
        y.append(value)
    return x, y


def _line(
    ax: Axes,
    updates: tuple[int, ...],
    values: tuple[float, ...],
    label: str,
    color: str,
    window: int,
) -> None:
    ax.plot(*_gaps(updates, values), color=color, alpha=0.2, linewidth=0.8)
    ax.plot(
        *_gaps(updates, trailing_mean(updates, values, window)),
        color=color,
        label=label,
        linewidth=1.8,
    )


def _save(figure: Figure, directory: Path, name: str) -> None:
    figure.savefig(directory / f"{name}.png", dpi=160)
    figure.savefig(directory / f"{name}.svg", metadata={"Date": None})
    plt.close(figure)


def plot_training_curves(
    rows: tuple[dict[str, Any], ...], directory: Path, *, window: int = 25
) -> None:
    if not rows:
        raise ValueError("Training plots require nonempty diagnostics")
    updates = tuple(int(r["update"]) for r in rows)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = ("#2166ac", "#b35806", "#4d9221")
    for key, label, color in zip(
        ("total_loss", "policy_loss", "value_loss"),
        ("Total", "Policy", "Value"),
        colors,
        strict=True,
    ):
        _line(
            axes[0, 0],
            updates,
            tuple(float(r["ppo"][key]) for r in rows),
            label,
            color,
            window,
        )
    axes[0, 0].set_title("PPO update losses")
    axes[0, 0].legend(fontsize=9)
    for ax, key, title in (
        (axes[0, 1], "entropy", "Legal-action entropy"),
        (axes[0, 2], "approximate_kl", "Approximate KL"),
        (axes[1, 0], "clip_fraction", "Clipped sample fraction"),
    ):
        _line(
            ax,
            updates,
            tuple(float(r["ppo"][key]) for r in rows),
            title,
            colors[0],
            window,
        )
        ax.set_title(title)
    for key, label, color in zip(
        ("x_wins", "o_wins", "draws"),
        ("X wins", "O wins", "Draws"),
        colors,
        strict=True,
    ):
        _line(
            axes[1, 1],
            updates,
            tuple(r[key] / r["games"] for r in rows),
            label,
            color,
            window,
        )
    axes[1, 1].set_title("Training self-play outcome rates")
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].legend(fontsize=9)
    _line(
        axes[1, 2],
        updates,
        tuple(r["ppo"]["num_samples"] / r["games"] for r in rows),
        "Decisions",
        colors[0],
        window,
    )
    axes[1, 2].set_title("Mean decisions/game (self-play)")
    axes[1, 2].set_ylim(4.5, 9.5)
    for ax in axes.flat:
        ax.set_xlabel("Completed PPO update")
        ax.grid(alpha=0.2)
    figure.suptitle(
        f"Training diagnostics — {len(rows)} logged updates, not held-out evaluation\n"
        f"Faint: raw; bold: {window}-update trailing mean; gaps are not interpolated",
        fontsize=13,
    )
    _save(figure, directory, "training-curves")


def plot_baseline_comparison(result: Phase4Comparison, directory: Path) -> None:
    figure, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    groups = [(c.opponent, c.learner_seat) for c in result.policy.cases]
    width = 0.24
    for series, label, color, offset in (
        ("policy", "Trained policy", "#2166ac", -width),
        ("random-v1", "Random control", "#777777", 0.0),
        ("heuristic-v1", "Heuristic control", "#b35806", width),
    ):
        estimates = []
        for opponent, seat in groups:
            match = next(
                c
                for c in result.controls
                if c.opponent == opponent
                and c.learner_seat == seat
                and c.control == ("random-v1" if series == "policy" else series)
            )
            estimates.append(
                match.policy_score if series == "policy" else match.control_score
            )
        ax.bar(
            [i + offset for i in range(len(groups))],
            [e.mean for e in estimates],
            width,
            label=label,
            color=color,
            yerr=[
                1.96 * e.standard_error if e.standard_error is not None else 0.0
                for e in estimates
            ],
            capsize=4,
        )
    ax.set_xticks(
        range(len(groups)),
        [
            f"{'Random' if o == 'random-v1' else 'Heuristic'} / "
            f"learner {'X' if s == 0 else 'O'}"
            for o, s in groups
        ],
    )
    ax.set_ylabel("Mean result score (+1 win / 0 draw / −1 loss)")
    ax.set_ylim(-1.15, 1.15)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    ax.set_title(
        f"Checkpoint update {result.policy.checkpoint_updates}: "
        "fixed-opponent baseline comparison\n"
        "Equal seats; same opponent/engine seeds; sampled policy; "
        "error bars: normal 95% match-mean intervals",
        fontsize=12,
    )
    _save(figure, directory, "baseline-comparison")
