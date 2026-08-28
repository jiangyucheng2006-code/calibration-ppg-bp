from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pulsedb_fewshot.calbased_prepare import prepare_calbased_data


def _write_ppg_file(path: Path, n_windows: int = 405) -> None:
    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("Subj_Wins")
        references = group.create_dataset(
            "PPG_F", shape=(1, n_windows), dtype=h5py.ref_dtype
        )
        sample = np.arange(1250, dtype=float)
        for index in range(n_windows):
            waveform = (
                np.sin(sample / (15.0 + (index % 37)))
                + index * 0.001
            )[None, :]
            target = h5file.create_dataset(f"wave_{index}", data=waveform)
            references[0, index] = target.ref


def _segment_index(raw: Path, n_windows: int = 405) -> pd.DataFrame:
    starts = np.arange(n_windows, dtype=float) * 10.0
    frame = pd.DataFrame(
        {
            "source": "MIMIC",
            "subject_uid": "MIMIC:p000001",
            "record_id": "case-1",
            "record_order": 0,
            "segment_row": np.arange(n_windows),
            "segment_uid": [f"MIMIC:p000001:{index:06d}" for index in range(n_windows)],
            "start_time_s": starts,
            "duration_s": 10.0,
            "sampling_rate_hz": 125.0,
            "n_samples": 1250,
            "sbp": 110.0 + np.arange(n_windows) % 30,
            "dbp": 60.0 + np.arange(n_windows) % 20,
            "raw_file": str(raw),
            "raw_file_sha256": "a" * 64,
            "ppg_field": "PPG_F",
            "ppg_storage_mode": "references",
            "ppg_reference_index": np.arange(n_windows),
            "segment_schema_valid": True,
            "split": "meta_train",
        }
    )
    # This row deliberately points at a nonexistent raw file. Predicate loading
    # must remove it before protocol construction or materialization.
    protected = frame.iloc[[0]].copy()
    protected["subject_uid"] = "MIMIC:p999998"
    protected["segment_uid"] = "MIMIC:p999998:000000"
    protected["raw_file"] = str(raw.parent / "must-not-be-read.mat")
    protected["split"] = "meta_validation"
    return pd.concat([frame, protected], ignore_index=True)


def test_prepare_builds_both_modes_without_protected_window_access(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "subject.mat"
    _write_ppg_file(raw)
    segments_path = tmp_path / "development.parquet"
    splits_path = tmp_path / "subject_splits.csv"
    _segment_index(raw).to_parquet(segments_path, index=False)
    pd.DataFrame(
        {
            "subject_uid": ["MIMIC:p000001", "MIMIC:p999998", "VitalDB:p999999"],
            "source": ["MIMIC", "MIMIC", "VitalDB"],
            "split": ["meta_train", "meta_validation", "meta_test"],
        }
    ).to_csv(splits_path, index=False)

    protocol_root = tmp_path / "protocol"
    store_root = tmp_path / "store"
    report = prepare_calbased_data(
        segments_path,
        splits_path,
        protocol_root,
        store_root,
        expected_subjects=1,
        train_shards=1,
        validation_shards=1,
        heldout_shards=1,
        workers=1,
    )

    assert report["status"] == "pass"
    assert report["split_modes"] == ["random_disjoint", "chronological_blocked"]
    assert report["input_loading_filter"] == "split == meta_train"
    assert report["meta_validation_windows_accessed"] is False
    assert report["locked_meta_test_windows_accessed"] is False
    assert report["heldout_test_targets_accessed_by_materializer"] is False
    assert (protocol_root / "data_preparation.json").is_file()
    for split_mode in ("random_disjoint", "chronological_blocked"):
        assert (
            protocol_root / split_mode / "development_fit_manifest.parquet"
        ).is_file()
        assert (
            protocol_root
            / split_mode
            / "locked"
            / "heldout_test_targets.parquet"
        ).is_file()
        assert (store_root / split_mode / "train_metadata_000.parquet").is_file()
        assert (
            store_root / split_mode / "internal_validation_metadata_000.parquet"
        ).is_file()
        heldout = pd.read_parquet(
            store_root / split_mode / "heldout_test_metadata_000.parquet"
        )
        assert not {"sbp", "dbp"} & set(heldout.columns)

    materialization = report["materialization"]
    assert materialization["split_modes_materialized"] == [
        "chronological_blocked",
        "random_disjoint",
    ]
    assert materialization["subject_count"] == 1
    assert materialization["windows_per_subject"] == {
        "train": 320,
        "internal_validation": 40,
        "heldout_test": 40,
    }
