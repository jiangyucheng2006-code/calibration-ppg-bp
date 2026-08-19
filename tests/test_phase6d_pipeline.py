import numpy as np
import pandas as pd
import pytest

from pulsedb_fewshot.report_phase6d_pipeline import (
    assign_true_tail_by_source,
    binary_identification_metrics,
)


def test_true_tail_is_exact_and_source_stratified() -> None:
    frame = pd.DataFrame(
        {
            "source": ["MIMIC"] * 10 + ["VitalDB"] * 7,
            "subject_uid": [f"m{i}" for i in range(10)] + [f"v{i}" for i in range(7)],
            "sbp_mae": np.arange(17, dtype=float),
            "dbp_mae": np.arange(17, dtype=float),
            "mean_mae": list(range(10)) + list(range(7)),
            "queries": 5,
        }
    )

    labelled = assign_true_tail_by_source(frame, fraction=0.30)

    assert labelled.groupby("source")["true_hard"].sum().to_dict() == {
        "MIMIC": 3,
        "VitalDB": 3,
    }
    assert set(labelled.loc[labelled["true_hard"], "subject_uid"]) == {
        "m7",
        "m8",
        "m9",
        "v4",
        "v5",
        "v6",
    }


def test_identification_metrics_are_paired_to_fixed_predictions() -> None:
    labels = np.asarray([False, False, True, True])
    scores = np.asarray([0.1, 0.8, 0.7, 0.9])
    predicted = scores >= 0.75

    metrics = binary_identification_metrics(labels, scores, predicted)

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["predicted_high_risk_fraction"] == pytest.approx(0.5)
    assert metrics["true_tail_fraction"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
