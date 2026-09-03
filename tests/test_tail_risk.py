from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from pulsedb_fewshot.tail_risk import (  # noqa: E402
    FEATURE_COLUMNS,
    _exact_tail,
    _risk_metrics,
    assign_source_stratified_folds,
    build_risk_features,
)


def test_source_stratified_folds_are_deterministic_and_disjoint() -> None:
    participant = pd.DataFrame(
        {
            "subject_uid": [f"m{i}" for i in range(11)]
            + [f"v{i}" for i in range(8)],
            "source": ["MIMIC"] * 11 + ["VitalDB"] * 8,
        }
    )

    first = assign_source_stratified_folds(participant, n_folds=5, seed=17)
    second = assign_source_stratified_folds(participant, n_folds=5, seed=17)

    pd.testing.assert_frame_equal(first, second)
    assert first["subject_uid"].is_unique
    assert set(first["fold"]) == set(range(5))
    counts = first.groupby(["source", "fold"]).size()
    assert counts.groupby(level=0).agg(lambda values: values.max() - values.min()).eq(1).all()


def test_exact_tail_is_thirty_percent_within_each_source() -> None:
    participant = pd.DataFrame(
        {
            "subject_uid": [f"m{i}" for i in range(10)]
            + [f"v{i}" for i in range(7)],
            "source": ["MIMIC"] * 10 + ["VitalDB"] * 7,
            "participant_mean_mae": list(range(10)) + list(range(7)),
        }
    )

    labelled, summary = _exact_tail(participant, fraction=0.30)

    hard = labelled.groupby("source")["hard_oof"].sum().to_dict()
    assert hard == {"MIMIC": 3, "VitalDB": 3}
    assert set(labelled.loc[labelled["hard_oof"], "subject_uid"]) == {
        "m7",
        "m8",
        "m9",
        "v4",
        "v5",
        "v6",
    }
    assert summary.set_index("source")["hard_participants"].to_dict() == hard
    assert set(labelled["label_split"]) == {"meta_train_crossfit_oof"}


def _write_synthetic_store(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for subject_index, (subject, source, split) in enumerate(
        [("m1", "MIMIC", "meta_train"), ("v1", "VitalDB", "meta_train")]
    ):
        for event_index in range(1, 7):
            rows.append(
                {
                    "subject_uid": subject,
                    "event_id": f"{subject}-e{event_index}",
                    "event_index": event_index,
                    "record_order": event_index,
                    "time_bin": event_index * 2,
                    "n_segments_in_bin": event_index + 1,
                    "ppg_f_mean": 0.1 * event_index + subject_index,
                    "ppg_f_std": 1.0 + 0.05 * event_index,
                    "source": source,
                    "split": split,
                    "support_candidate": event_index <= 5,
                    "common_query": event_index == 6,
                    "sbp": 110.0 + subject_index * 10 + event_index,
                    "dbp": 60.0 + subject_index * 5 + event_index,
                }
            )
    metadata = pd.DataFrame(rows)
    metadata.to_parquet(root / "development_metadata_000.parquet", index=False)
    predictions = metadata.loc[metadata["common_query"], ["subject_uid", "event_id", "source"]].copy()
    predictions["k"] = 5
    predictions["pred_sbp"] = [120.0, 130.0]
    predictions["pred_dbp"] = [70.0, 75.0]
    labels = pd.DataFrame(
        {
            "subject_uid": ["m1", "v1"],
            "fold": [0, 1],
            "hard_oof": [False, True],
            "participant_mean_mae": [4.0, 12.0],
            "label_split": ["meta_train_crossfit_oof"] * 2,
        }
    )
    return predictions, labels


def test_risk_features_use_frozen_source_without_query_bp(tmp_path: Path) -> None:
    predictions, labels = _write_synthetic_store(tmp_path)

    features = build_risk_features(tmp_path, predictions, labels)

    assert set(features["source"]) == {"MIMIC", "VitalDB"}
    assert "source_x" not in features and "source_y" not in features
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert "target_sbp" not in FEATURE_COLUMNS
    assert "target_dbp" not in FEATURE_COLUMNS
    assert "source" not in FEATURE_COLUMNS
    assert "record_order" not in FEATURE_COLUMNS
    assert "time_bin" not in FEATURE_COLUMNS
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy(dtype=float)).all()
    assert set(features["label_split"]) == {"meta_train_crossfit_oof"}


def test_risk_metrics_use_fixed_threshold() -> None:
    metrics = _risk_metrics(
        np.asarray([False, False, True, True]),
        np.asarray([0.1, 0.8, 0.7, 0.9]),
        threshold=0.75,
    )

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["predicted_high_risk_fraction"] == pytest.approx(0.5)
