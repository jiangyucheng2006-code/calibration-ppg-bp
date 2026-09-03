import numpy as np
import pandas as pd

from pulsedb_fewshot.analyze_similarity_error import (
    BIN_ORDER,
    _similarity_bin,
    _spearman,
    _within_participant_contrast,
)


def test_similarity_bins_include_expected_edges() -> None:
    result = _similarity_bin(pd.Series([0.49, 0.50, 0.80, 0.90, 0.95]))
    assert list(result.astype(str)) == BIN_ORDER


def test_spearman_detects_inverse_monotonic_relation() -> None:
    similarity = pd.Series([0.4, 0.6, 0.8, 0.9, 0.99])
    error = pd.Series([10.0, 8.0, 6.0, 4.0, 2.0])
    assert np.isclose(_spearman(similarity, error), -1.0)


def test_within_participant_contrast_handles_decimal_threshold_label() -> None:
    frame = pd.DataFrame(
        {
            "subject_uid": ["a", "a", "b", "b"],
            "source": ["MIMIC"] * 4,
            "pairwise_corr_median": [0.8, 0.95, 0.7, 0.96],
            "sbp_abs_error": [8.0, 4.0, 10.0, 5.0],
            "dbp_abs_error": [4.0, 2.0, 6.0, 3.0],
            "mean_abs_error": [6.0, 3.0, 8.0, 4.0],
        }
    )
    result = _within_participant_contrast(
        frame, "test", seed=1, repetitions=100
    )
    overall = result.loc[
        result.scope.eq("Overall") & result.outcome.eq("mean_abs_error")
    ].iloc[0]
    assert overall.paired_participants == 2
    assert np.isclose(overall.low_minus_high_mmHg, 3.5)
