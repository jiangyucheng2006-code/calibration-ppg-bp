"""Shared metrics for the development CalBased analogue."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .training import participant_macro_metrics


SCOPES = ("Overall", "MIMIC", "VitalDB")
POOLED_COLUMNS = (
    "Setting",
    "BP",
    "MAE",
    "R²",
    "ME",
    "STD",
    "≤5 mmHg",
    "≤10 mmHg",
    "≤15 mmHg",
    "AAMI",
    "BHS",
)


def _scope(predictions: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "Overall":
        return predictions
    return predictions.loc[predictions["source"].eq(scope)]


def _bhs_grade(within_5: float, within_10: float, within_15: float) -> str:
    if within_5 >= 60.0 and within_10 >= 85.0 and within_15 >= 95.0:
        return "A"
    if within_5 >= 50.0 and within_10 >= 75.0 and within_15 >= 90.0:
        return "B"
    if within_5 >= 40.0 and within_10 >= 65.0 and within_15 >= 85.0:
        return "C"
    return "D"


def _r_squared(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((prediction - target) ** 2)) / denominator


def validate_prediction_views(predictions: pd.DataFrame) -> None:
    required = {
        "subject_uid",
        "event_id",
        "source",
        "target_sbp",
        "target_dbp",
        "pred_sbp",
        "pred_dbp",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    sources = set(predictions["source"].astype(str))
    if sources != {"MIMIC", "VitalDB"}:
        raise ValueError(
            "formal CalBased views require both internal PulseDB strata: "
            f"MIMIC and VitalDB, got {sorted(sources)}"
        )
    if predictions["event_id"].astype(str).duplicated().any():
        raise ValueError("prediction table contains duplicate event_id values")


def participant_macro_views(
    predictions: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Compute all three views from one unchanged prediction table."""

    validate_prediction_views(predictions)
    return {
        scope: participant_macro_metrics(_scope(predictions, scope))
        for scope in SCOPES
    }


def pooled_diagnostics(predictions: pd.DataFrame, setting: str) -> pd.DataFrame:
    """Return event-pooled numerical diagnostics for all three views.

    AAMI/BHS fields are retrospective numerical screens only. They do not
    establish formal device or protocol compliance.
    """

    validate_prediction_views(predictions)
    rows: list[dict[str, object]] = []
    for scope in SCOPES:
        frame = _scope(predictions, scope)
        for bp in ("SBP", "DBP"):
            lower = bp.lower()
            target = frame[f"target_{lower}"].to_numpy(dtype=float)
            prediction = frame[f"pred_{lower}"].to_numpy(dtype=float)
            error = prediction - target
            absolute = np.abs(error)
            within_5 = float(np.mean(absolute <= 5.0) * 100.0)
            within_10 = float(np.mean(absolute <= 10.0) * 100.0)
            within_15 = float(np.mean(absolute <= 15.0) * 100.0)
            mean_error = float(error.mean())
            error_std = float(error.std(ddof=1)) if len(error) > 1 else float("nan")
            grade = _bhs_grade(within_5, within_10, within_15)
            rows.append(
                {
                    "Scope": scope,
                    "Setting": setting,
                    "BP": bp,
                    "MAE": float(absolute.mean()),
                    "R²": _r_squared(target, prediction),
                    "ME": mean_error,
                    "STD": error_std,
                    "≤5 mmHg": within_5,
                    "≤10 mmHg": within_10,
                    "≤15 mmHg": within_15,
                    "AAMI": (
                        "PASS*"
                        if abs(mean_error) <= 5.0 and error_std <= 8.0
                        else "FAIL*"
                    ),
                    "BHS": (
                        f"{'PASS' if grade in {'A', 'B'} else 'FAIL'} "
                        f"(Grade {grade})*"
                    ),
                }
            )
    return pd.DataFrame(rows)[["Scope", *POOLED_COLUMNS]]
