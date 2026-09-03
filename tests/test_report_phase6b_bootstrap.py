import numpy as np
import pytest

from pulsedb_fewshot.report_phase6b_bootstrap import (
    _paired_bootstrap,
    _parse_comparisons,
)


def test_parse_comparisons_requires_three_named_fields() -> None:
    assert _parse_comparisons(["gain|candidate|reference"]) == [
        ("gain", "candidate", "reference")
    ]
    with pytest.raises(ValueError, match="comparison must be"):
        _parse_comparisons(["candidate|reference"])


def test_paired_bootstrap_is_deterministic_and_preserves_direction() -> None:
    differences = np.asarray([-1.0, -0.5, -0.2, -0.1, -0.7])
    first = _paired_bootstrap(differences, repetitions=2000, seed=17)
    second = _paired_bootstrap(differences, repetitions=2000, seed=17)

    assert first == second
    assert first["mean_delta_mmHg"] == pytest.approx(-0.5)
    assert first["ci95_high_mmHg"] < 0.0
    assert first["bootstrap_fraction_improved"] == pytest.approx(1.0)


def test_paired_bootstrap_rejects_too_few_repetitions() -> None:
    with pytest.raises(ValueError, match="at least 1000"):
        _paired_bootstrap(np.asarray([0.1, -0.1]), repetitions=999, seed=1)
