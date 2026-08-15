from pathlib import Path

import numpy as np
import pandas as pd

from pulsedb_fewshot.report_first_run import (
    bhs_grade,
    compute_extended_metrics,
    load_first_run_predictions,
)


def test_bhs_grade_uses_cumulative_thresholds() -> None:
    assert bhs_grade(60, 85, 95) == "A"
    assert bhs_grade(50, 75, 90) == "B"
    assert bhs_grade(40, 65, 85) == "C"
    assert bhs_grade(39.99, 100, 100) == "D"


def test_extended_metrics_use_prediction_minus_reference_and_inclusive_limits() -> None:
    frame = pd.DataFrame(
        {
            "method": ["m0"] * 4,
            "k": [1] * 4,
            "subject_uid": ["a", "a", "b", "b"],
            "event_id": ["e1", "e2", "e3", "e4"],
            "target_sbp": [100.0, 110.0, 120.0, 130.0],
            "target_dbp": [60.0, 70.0, 80.0, 90.0],
            "pred_sbp": [100.0, 115.0, 110.0, 145.0],
            "pred_dbp": [60.0, 75.0, 70.0, 105.0],
        }
    )
    metrics = compute_extended_metrics(frame).set_index("BP")
    assert metrics.loc["SBP", "ME"] == 2.5
    assert metrics.loc["SBP", "MAE"] == 7.5
    assert metrics.loc["SBP", "≤5 mmHg (%)"] == 50.0
    assert metrics.loc["SBP", "≤10 mmHg (%)"] == 75.0
    assert metrics.loc["SBP", "≤15 mmHg (%)"] == 100.0
    assert metrics.loc["SBP", "Participant-macro MAE"] == 7.5
    assert metrics.loc["SBP", "AAMI"] == "FAIL*"
    assert metrics.loc["SBP", "BHS"] == "PASS (Grade B)*"
    expected_r2 = 1 - (0**2 + 5**2 + (-10) ** 2 + 15**2) / 500
    assert np.isclose(metrics.loc["SBP", "R²"], expected_r2)


def _write(path: Path, method: str | None, target_shift: float = 0.0) -> None:
    frame = pd.DataFrame(
        {
            "k": [1],
            "subject_uid": ["s"],
            "event_id": ["e"],
            "target_sbp": [120.0 + target_shift],
            "target_dbp": [70.0],
            "pred_sbp": [121.0],
            "pred_dbp": [71.0],
        }
    )
    if method is not None:
        frame.insert(0, "method", method)
    frame.to_parquet(path, index=False)


def test_loader_rejects_target_disagreement(tmp_path: Path) -> None:
    controls = tmp_path / "controls.parquet"
    m0 = tmp_path / "m0.parquet"
    m1 = tmp_path / "m1.parquet"
    m2 = tmp_path / "m2.parquet"
    siamese = tmp_path / "siamese.parquet"
    _write(controls, "population")
    _write(m0, None)
    _write(m1, None)
    _write(m2, None, target_shift=1.0)
    _write(siamese, None)

    try:
        load_first_run_predictions(
            controls=controls, m0=m0, m1=m1, m2=m2, siamese=siamese
        )
    except AssertionError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("target disagreement should fail")
