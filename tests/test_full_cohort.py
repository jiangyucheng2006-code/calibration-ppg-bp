from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pulsedb_fewshot.development_events import run_development_event_audit
from pulsedb_fewshot.full_cohort import _audit_one_file


def _char_array(text: str) -> np.ndarray:
    return np.asarray([ord(char) for char in text], dtype=np.uint16).reshape(-1, 1)


def _write_reference_field(
    h5file: h5py.File,
    group: h5py.Group,
    refs_group: h5py.Group,
    name: str,
    values: list[np.ndarray],
) -> None:
    references = np.empty((1, len(values)), dtype=h5py.ref_dtype)
    for index, value in enumerate(values):
        target = refs_group.create_dataset(f"{name}_{index}", data=value)
        references[0, index] = target.ref
    group.create_dataset(name, data=references, dtype=h5py.ref_dtype)


def _synthetic_file(path: Path, subject: str = "p000001", n_windows: int = 3) -> None:
    with h5py.File(path, "w", userblock_size=512) as h5file:
        refs_group = h5file.create_group("#refs#")
        group = h5file.create_group("Subj_Wins")
        times = [
            (index * 60.0 + np.arange(1, 11) * 0.1).reshape(1, -1)
            for index in range(n_windows)
        ]
        waves = [
            (np.sin(np.arange(10) / 2.0) + index).reshape(1, -1)
            for index in range(n_windows)
        ]
        for field in ("PPG_Raw", "PPG_F"):
            _write_reference_field(h5file, group, refs_group, field, waves)
        for field in ("ABP_Raw", "ABP_F"):
            _write_reference_field(h5file, group, refs_group, field, waves)
        _write_reference_field(h5file, group, refs_group, "T", times)
        _write_reference_field(
            h5file, group, refs_group, "SubjectID", [_char_array(subject)] * n_windows
        )
        _write_reference_field(
            h5file, group, refs_group, "CaseID", [_char_array("case-1")] * n_windows
        )
        _write_reference_field(
            h5file, group, refs_group, "Gender", [_char_array("F")] * n_windows
        )
        values = {
            "Age": [50.0] * n_windows,
            "SegmentID": list(range(1, n_windows + 1)),
            "WinID": list(range(1, n_windows + 1)),
            "WinSeqID": list(range(1, n_windows + 1)),
            "IncludeFlag": [1] * n_windows,
            "SegSBP": [120.0 + index for index in range(n_windows)],
            "SegDBP": [70.0 + index for index in range(n_windows)],
            "PPG_ABP_Corr": [0.9] * n_windows,
            "ABP_Lag": [2.0] * n_windows,
        }
        for field, field_values in values.items():
            arrays = [np.asarray([[value]]) for value in field_values]
            _write_reference_field(h5file, group, refs_group, field, arrays)
    with path.open("r+b") as handle:
        header = b"MATLAB 7.3 MAT-file, synthetic test"
        handle.write(header + b" " * (128 - len(header)))


def test_one_file_shard_is_traceable_and_resumable(tmp_path: Path) -> None:
    root = tmp_path / "Segment_Files"
    path = root / "PulseDB_MIMIC" / "p000001.mat"
    path.parent.mkdir(parents=True)
    _synthetic_file(path)
    shard_root = tmp_path / "shards"

    first = _audit_one_file(str(path), str(root), str(shard_root))
    second = _audit_one_file(str(path), str(root), str(shard_root))

    assert first["audit_complete"] is True
    assert first["file_schema_valid_pre_duplicate_check"] is True
    assert first["subject_uid"] == "MIMIC:p000001"
    assert first["n_segments"] == 3
    assert first["n_valid_segments"] == 3
    assert second["resumed_from_shard"] is True
    frame = pd.read_parquet(first["segment_shard"])
    assert frame["segment_uid"].is_unique
    assert frame["segment_schema_valid"].all()
    assert set(frame["ppg_field"]) == {"PPG_F"}
    assert frame["ppg_f_all_finite"].all()
    assert (frame["ppg_f_std"] > 0).all()


def test_direct_single_window_storage_is_supported(tmp_path: Path) -> None:
    root = tmp_path / "Segment_Files"
    path = root / "PulseDB_MIMIC" / "p000049.mat"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w", userblock_size=512) as h5file:
        group = h5file.create_group("Subj_Wins")
        wave = np.sin(np.arange(100) / 10.0).reshape(1, -1)
        time = (np.arange(100) * 0.01).reshape(1, -1)
        for field in ("PPG_Raw", "PPG_F", "ABP_Raw", "ABP_F"):
            group.create_dataset(field, data=wave)
        group.create_dataset("T", data=time)
        group.create_dataset("SubjectID", data=_char_array("p000049"))
        group.create_dataset("CaseID", data=_char_array("case-1"))
        group.create_dataset("Gender", data=_char_array("M"))
        for field, value in {
            "Age": 60.0,
            "SegmentID": 1.0,
            "WinID": 1.0,
            "WinSeqID": 1.0,
            "IncludeFlag": 1.0,
            "SegSBP": 120.0,
            "SegDBP": 70.0,
            "PPG_ABP_Corr": 0.9,
            "ABP_Lag": 2.0,
        }.items():
            group.create_dataset(field, data=np.asarray([[value]]))
    with path.open("r+b") as handle:
        header = b"MATLAB 7.3 MAT-file, synthetic direct storage"
        handle.write(header + b" " * (128 - len(header)))

    result = _audit_one_file(str(path), str(root), str(tmp_path / "shards"))

    assert result["file_schema_valid_pre_duplicate_check"] is True
    assert result["storage_mode"] == "direct_single_window"
    assert result["n_segments"] == 1
    frame = pd.read_parquet(result["segment_shard"])
    assert frame.loc[0, "ppg_storage_mode"] == "direct_single_window"
    assert bool(frame.loc[0, "segment_schema_valid"])


def _development_rows(subject_uid: str, split: str) -> pd.DataFrame:
    source, subject_id = subject_uid.split(":")
    return pd.DataFrame(
        {
            "dataset_id": "PulseDB_v2",
            "source": source,
            "source_directory": "PulseDB_MIMIC" if source == "MIMIC" else "PulseDB_Vital",
            "subject_id": subject_id,
            "subject_uid": subject_uid,
            "record_id": "case-1",
            "record_order": 0,
            "segment_row": range(10),
            "segment_uid": [f"{subject_uid}:p:00000{i}" for i in range(10)],
            "segment_id": range(10),
            "win_id": range(10),
            "win_seq_id": range(10),
            "start_time_s": [60.0 * index for index in range(10)],
            "end_time_s": [60.0 * index + 9.9 for index in range(10)],
            "duration_s": 10.0,
            "sample_interval_s": 0.1,
            "sampling_rate_hz": 10.0,
            "n_samples": 100,
            "sbp": [120.0 + index for index in range(10)],
            "dbp": [70.0 + index for index in range(10)],
            "pulse_pressure": 50.0,
            "include_flag": 1,
            "ppg_abp_corr": 0.9,
            "abp_lag_samples": 2.0,
            "age": 50.0,
            "gender": "F",
            "raw_file": "/raw/file.mat",
            "raw_file_relative_path": "PulseDB_MIMIC/file.mat",
            "raw_file_sha256": "a" * 64,
            "ppg_field": "PPG_F",
            "ppg_reference_index": range(10),
            "segment_schema_valid": True,
            "segment_exclusion_reasons": "",
            "split": split,
        }
    )


def test_development_event_audit_excludes_locked_test(tmp_path: Path) -> None:
    development = pd.concat(
        [
            _development_rows("MIMIC:p000001", "meta_train"),
            _development_rows("VitalDB:p000002", "meta_validation"),
        ],
        ignore_index=True,
    )
    segments_path = tmp_path / "development.parquet"
    development.to_parquet(segments_path, index=False)
    splits_path = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "subject_uid": ["MIMIC:p000001", "VitalDB:p000002", "MIMIC:p999999"],
            "source": ["MIMIC", "VitalDB", "MIMIC"],
            "split": ["meta_train", "meta_validation", "meta_test"],
            "split_seed": [7, 7, 7],
        }
    ).to_csv(splits_path, index=False)

    report = run_development_event_audit(
        segments_path, splits_path, tmp_path / "events", widths=(60, 120)
    )

    assert report["locked_test_access"].startswith("not eventized")
    assert report["selection_status"].startswith("not selected")
    assert report["results"]["60"]["n_eligible_development_subjects"] == 2
    assert report["results"]["120"]["n_eligible_development_subjects"] == 0


def test_development_event_audit_rejects_locked_rows(tmp_path: Path) -> None:
    locked = _development_rows("MIMIC:p999999", "meta_test")
    segments_path = tmp_path / "locked.parquet"
    locked.to_parquet(segments_path, index=False)
    splits_path = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "subject_uid": ["MIMIC:p999999"],
            "source": ["MIMIC"],
            "split": ["meta_test"],
            "split_seed": [7],
        }
    ).to_csv(splits_path, index=False)

    with pytest.raises(AssertionError, match="locked meta-test"):
        run_development_event_audit(segments_path, splits_path, tmp_path / "events")
