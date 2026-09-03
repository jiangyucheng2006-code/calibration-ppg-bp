from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulsedb_fewshot.calbased_protocol import (
    PROTOCOL_ID,
    build_calbased_analogue,
    load_frozen_meta_train_segments,
    run_calbased_analogue,
)


def _subject_rows(
    subject_uid: str,
    source: str,
    *,
    n_windows: int = 405,
    split: str = "meta_train",
) -> pd.DataFrame:
    subject_id = subject_uid.split(":", 1)[-1]
    starts = np.arange(n_windows, dtype=float) * 10.0
    return pd.DataFrame(
        {
            "dataset_id": "PulseDB_v2",
            "source": source,
            "source_directory": (
                "PulseDB_MIMIC" if source == "MIMIC" else "PulseDB_Vital"
            ),
            "subject_id": subject_id,
            "subject_uid": subject_uid,
            "record_id": "case-1",
            "record_order": 0,
            "segment_row": np.arange(n_windows),
            "segment_uid": [f"{subject_uid}:segment:{index:06d}" for index in range(n_windows)],
            "segment_id": np.arange(n_windows),
            "win_id": np.arange(n_windows),
            "win_seq_id": np.arange(n_windows),
            "start_time_s": starts,
            "end_time_s": starts + 9.992,
            "duration_s": 10.0,
            "sample_interval_s": 0.008,
            "sampling_rate_hz": 125.0,
            "n_samples": 1250,
            "sbp": 110.0 + np.arange(n_windows) % 30,
            "dbp": 60.0 + np.arange(n_windows) % 20,
            "pulse_pressure": 50.0,
            "include_flag": 1,
            "ppg_abp_corr": 0.95,
            "abp_lag_samples": 2.0,
            "age": 50.0,
            "gender": "F",
            "raw_file": f"/raw/{source}/{subject_id}.mat",
            "raw_file_relative_path": f"{source}/{subject_id}.mat",
            "raw_file_sha256": ("a" if source == "MIMIC" else "b") * 64,
            "ppg_field": "PPG_F",
            "ppg_storage_mode": "references",
            "ppg_reference_index": np.arange(n_windows),
            "ppg_f_mean": 0.0,
            "ppg_f_std": 1.0,
            "segment_schema_valid": True,
            "split": split,
        }
    )


def _splits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_uid": [
                "MIMIC:p000001",
                "VitalDB:p000002",
                "MIMIC:p999998",
                "VitalDB:p999999",
            ],
            "source": ["MIMIC", "VitalDB", "MIMIC", "VitalDB"],
            "split": ["meta_train", "meta_train", "meta_validation", "meta_test"],
            "split_seed": 20260809,
        }
    )


def _meta_train_segments() -> pd.DataFrame:
    return pd.concat(
        [
            _subject_rows("MIMIC:p000001", "MIMIC"),
            _subject_rows("VitalDB:p000002", "VitalDB"),
        ],
        ignore_index=True,
    )


def test_random_disjoint_is_deterministic_and_target_isolated() -> None:
    segments = _meta_train_segments()
    first = build_calbased_analogue(segments, _splits(), seed=7)
    second = build_calbased_analogue(
        segments.sample(frac=1.0, random_state=99).reset_index(drop=True),
        _splits(),
        seed=7,
    )

    left = first.role_manifest.sort_values("segment_uid").reset_index(drop=True)
    right = second.role_manifest.sort_values("segment_uid").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert set(first.role_manifest["role"]) == {
        "train",
        "internal_validation",
        "heldout_test",
    }
    counts = first.role_manifest.groupby(["subject_uid", "role"]).size().unstack()
    assert (counts["train"] == 320).all()
    assert (counts["internal_validation"] == 40).all()
    assert (counts["heldout_test"] == 40).all()
    assert set(first.development_fit_manifest["role"]) == {
        "train",
        "internal_validation",
    }
    assert {"sbp", "dbp"}.issubset(first.development_fit_manifest.columns)
    assert not {"sbp", "dbp", "pulse_pressure"} & set(
        first.heldout_test_inputs.columns
    )
    assert not {"age", "gender", "ppg_abp_corr", "abp_lag_samples"} & set(
        first.development_fit_manifest.columns
    )
    assert set(first.heldout_test_targets["role"]) == {"heldout_test"}
    assert first.audit["status"] == "pass"
    assert first.audit["protocol_id"] == PROTOCOL_ID
    assert first.audit["subject_overlap_counts"] == {
        "train__internal_validation": 2,
        "train__heldout_test": 2,
        "internal_validation__heldout_test": 2,
    }
    assert not any(first.audit["window_overlap_counts"].values())
    assert not any(first.audit["raw_interval_overlap_counts"].values())
    assert not any(first.audit["waveform_locator_overlap_counts"].values())
    assert first.audit["exact_waveform_content_overlap_audited"] is False
    assert first.audit["heldout_test_used_for_early_stopping"] is False
    assert first.audit["meta_validation_windows_accessed"] is False
    assert first.audit["locked_meta_test_windows_accessed"] is False


def test_chronological_blocked_uses_early_320_then_40_then_40() -> None:
    segments = _subject_rows("MIMIC:p000001", "MIMIC")
    artifacts = build_calbased_analogue(
        segments,
        _splits(),
        split_mode="chronological_blocked",
    )
    roles = artifacts.role_manifest.set_index("selection_rank")["role"]

    assert set(roles.loc[1:320]) == {"train"}
    assert set(roles.loc[321:360]) == {"internal_validation"}
    assert set(roles.loc[361:400]) == {"heldout_test"}
    assert artifacts.audit["adjacent_time_gap_audit"][
        "train__internal_validation"
    ]["minimum_gap_seconds"] == pytest.approx(0.0)
    assert artifacts.audit["adjacent_time_gap_audit"][
        "internal_validation__heldout_test"
    ]["adjacent_segment_row_pairs"] == 1


def test_random_disjoint_repairs_cross_role_raw_interval_overlap() -> None:
    segments = _subject_rows("MIMIC:p000001", "MIMIC")
    probe = build_calbased_analogue(segments, _splits(), seed=17)
    roles = probe.role_manifest.set_index("segment_uid")["role"]
    train_uid = roles.loc[roles.eq("train")].index[0]
    validation_uid = roles.loc[roles.eq("internal_validation")].index[0]
    train_start = float(
        segments.loc[segments["segment_uid"].eq(train_uid), "start_time_s"].iloc[0]
    )
    segments.loc[
        segments["segment_uid"].eq(validation_uid), "start_time_s"
    ] = train_start + 5.0

    repaired = build_calbased_analogue(segments, _splits(), seed=17)

    assert repaired.audit["status"] == "pass"
    assert repaired.audit["random_interval_role_swap_repairs"] >= 1
    assert not any(repaired.audit["raw_interval_overlap_counts"].values())
    counts = repaired.role_manifest.groupby("role").size()
    assert counts.to_dict() == {
        "heldout_test": 40,
        "internal_validation": 40,
        "train": 320,
    }


def test_protected_participant_rows_are_rejected() -> None:
    protected = _subject_rows(
        "MIMIC:p999998", "MIMIC", split="meta_validation"
    )
    with pytest.raises(AssertionError, match="meta_validation or locked meta_test"):
        build_calbased_analogue(protected, _splits())


def test_cross_role_raw_interval_overlap_fails_audit() -> None:
    segments = _subject_rows("MIMIC:p000001", "MIMIC")
    # Chronological ranks 320 and 321 fall in different roles.  Make their raw
    # intervals overlap while keeping their segment identities distinct.
    segments.loc[segments["segment_row"].eq(320), "start_time_s"] = 3195.0
    artifacts = build_calbased_analogue(
        segments,
        _splits(),
        split_mode="chronological_blocked",
        strict=False,
    )

    assert artifacts.audit["status"] == "fail"
    assert "raw_interval_overlap_across_roles" in artifacts.audit["failures"]
    with pytest.raises(AssertionError, match="raw_interval_overlap"):
        build_calbased_analogue(
            segments,
            _splits(),
            split_mode="chronological_blocked",
        )


def test_parquet_loader_applies_meta_train_predicate_and_writes_quarantine(
    tmp_path: Path,
) -> None:
    all_development = pd.concat(
        [
            _subject_rows("MIMIC:p000001", "MIMIC"),
            _subject_rows("MIMIC:p999998", "MIMIC", split="meta_validation"),
        ],
        ignore_index=True,
    )
    segment_path = tmp_path / "development_segments.parquet"
    split_path = tmp_path / "subject_splits.csv"
    output = tmp_path / "protocol"
    all_development.to_parquet(segment_path, index=False)
    _splits().to_csv(split_path, index=False)

    loaded, _, description = load_frozen_meta_train_segments(segment_path, split_path)
    assert description == "split == meta_train"
    assert set(loaded["subject_uid"]) == {"MIMIC:p000001"}

    result = run_calbased_analogue(
        segment_path, split_path, output, expected_subjects=1
    )
    assert result["audit"]["status"] == "pass"
    assert result["audit"]["input_loading_filter"] == "split == meta_train"
    fit = pd.read_parquet(output / "development_fit_manifest.parquet")
    heldout_inputs = pd.read_parquet(output / "heldout" / "heldout_test_inputs.parquet")
    heldout_targets = pd.read_parquet(output / "locked" / "heldout_test_targets.parquet")
    assert set(fit["role"]) == {"train", "internal_validation"}
    assert set(heldout_inputs["role"]) == {"heldout_test"}
    assert not {"sbp", "dbp"} & set(heldout_inputs.columns)
    assert {"sbp", "dbp"}.issubset(heldout_targets.columns)
