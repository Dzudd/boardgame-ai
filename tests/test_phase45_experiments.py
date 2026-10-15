"""Budget/statistical checks, not assertions that a learner becomes stronger."""

from pathlib import Path

import pytest
import torch

from boardgame_ai.training.benchmark import benchmark, inference_probe
from boardgame_ai.training.experiments import paired_interval


def test_training_seed_uncertainty_is_paired_and_small_sample() -> None:
    result = paired_interval([-1, 0, 1])
    assert result["mean"] == 0
    assert result["se"] == pytest.approx(1 / 3**0.5)
    assert result["approximate_95_interval"] == pytest.approx(
        [-4.303 / 3**0.5, 4.303 / 3**0.5]
    )
    assert paired_interval([0.2] * 3)["approximate_95_interval"] == pytest.approx(
        [0.2, 0.2]
    )
    with pytest.raises(ValueError):
        paired_interval([0.2])


def test_probe_uses_visible_corpus_without_changing_training_or_rng() -> None:
    before = torch.random.get_rng_state().clone()
    result = inference_probe()
    assert torch.equal(before, torch.random.get_rng_state())
    assert result["adopted"] is False
    assert result["positions"] > 0
    assert len(result["timings"]) == 5
    assert result["max_probability_difference"] < 2e-6


def test_benchmark_rejects_budget_before_creating_outputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        benchmark(tmp_path / "missing.py", tmp_path / "out", repeats=0)
    assert not (tmp_path / "out").exists()
