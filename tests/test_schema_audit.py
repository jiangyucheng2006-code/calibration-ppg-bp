from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pulsedb_fewshot.schema_audit import _waveform_summary, audit_pulsedb_file
from pulsedb_fewshot.pilot_audit import audit_pulsedb_pilot


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
    for index, values_for_window in enumerate(values):
        dataset = refs_group.create_dataset(f"{name}_{index}", data=values_for_window)
        references[0, index] = dataset.ref
    group.create_dataset(name, data=references, dtype=h5py.ref_dtype)


def _synthetic_file(path: Path) -> None:
    n_windows = 3
    n_samples = 10
    with h5py.File(path, "w") as h5file:
        refs_group = h5file.create_group("#refs#")
        group = h5file.create_group("Subj_Wins")
        time_values = [
            (index * 20.0 + np.arange(1, n_samples + 1) * 0.1).reshape(1, -1)
            for index in range(n_windows)
        ]
        ppg_values = [
            (np.sin(np.arange(n_samples) / 2.0) + index).reshape(1, -1)
            for index in range(n_windows)
        ]
        abp_values = [
            (80.0 + 20.0 * np.sin(np.arange(n_samples) / 2.0) + index).reshape(1, -1)
            for index in range(n_windows)
        ]
        for field in ("PPG_Raw", "PPG_F", "PPG_Record", "PPG_Record_F"):
            _write_reference_field(h5file, group, refs_group, field, ppg_values)
        for field in ("ABP_Raw", "ABP_F"):
            _write_reference_field(h5file, group, refs_group, field, abp_values)
        _write_reference_field(h5file, group, refs_group, "T", time_values)
        _write_reference_field(h5file, group, refs_group, "SubjectID", [_char_array("p000001")] * n_windows)
        _write_reference_field(h5file, group, refs_group, "CaseID", [_char_array("case-1")] * n_windows)
        _write_reference_field(h5file, group, refs_group, "Gender", [_char_array("M")] * n_windows)
        scalar_data = {
            "Age": [60.0] * n_windows,
            "SegmentID": [1.0, 3.0, 5.0],
            "WinID": [1.0, 3.0, 5.0],
            "WinSeqID": [1.0, 3.0, 5.0],
            "IncludeFlag": [1, 1, 1],
            "SegSBP": [120.0, 121.0, 122.0],
            "SegDBP": [70.0, 71.0, 72.0],
            "PPG_ABP_Corr": [0.9, 0.91, 0.92],
            "ABP_Lag": [2.0, 2.0, 3.0],
        }
        for field, values in scalar_data.items():
            dtype = np.uint8 if field == "IncludeFlag" else float
            arrays = [np.asarray([[value]], dtype=dtype) for value in values]
            _write_reference_field(h5file, group, refs_group, field, arrays)


def test_comprehensive_audit_writes_traceable_index(tmp_path: Path) -> None:
    input_path = tmp_path / "synthetic.mat"
    output_dir = tmp_path / "output"
    _synthetic_file(input_path)

    report = audit_pulsedb_file(input_path, output_dir, source="MIMIC")

    assert report["status"] in {"pass", "pass_with_warnings"}
    assert report["n_windows"] == 3
    assert report["time"]["sampling_rate_hz"]["median"] == pytest.approx(10.0)
    assert report["time"]["overlap_count"] == 0
    assert report["required_failures"] == []

    index = pd.read_parquet(output_dir / "synthetic_segment_index.parquet")
    assert list(index["subject_id"].unique()) == ["p000001"]
    assert list(index["segment_id"]) == [1, 3, 5]
    assert index["segment_uid"].is_unique
    assert list(index["sbp"]) == [120.0, 121.0, 122.0]
    assert index["ppg_raw_hdf5_path"].str.startswith("/#refs#/").all()


def test_waveform_summary_groups_duplicate_windows_deterministically() -> None:
    first = np.asarray([[1.0, 2.0, 3.0]])
    second = np.asarray([[4.0, 5.0, 6.0]])
    summary = _waveform_summary([first, second, first.copy(), second.copy(), first.copy()])

    assert summary["n_windows"] == 5
    assert summary["unique_waveform_hashes"] == 2
    assert sorted(summary["duplicate_waveform_groups_zero_based"].values()) == [
        [0, 2, 4],
        [1, 3],
    ]


def test_pilot_audit_builds_source_qualified_combined_index(tmp_path: Path) -> None:
    data_root = tmp_path / "Segment_Files"
    mimic_path = data_root / "PulseDB_MIMIC" / "p000001.mat"
    vital_path = data_root / "PulseDB_Vital" / "p000001.mat"
    mimic_path.parent.mkdir(parents=True)
    vital_path.parent.mkdir(parents=True)
    _synthetic_file(mimic_path)
    _synthetic_file(vital_path)
    with h5py.File(vital_path, "a") as h5file:
        h5file.attrs["synthetic_source"] = "VitalDB"

    import hashlib

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    download_manifest = tmp_path / "download_manifest.csv"
    pd.DataFrame(
        [
            {
                "source": "PulseDB_MIMIC",
                "file_name": mimic_path.name,
                "sha256": sha256(mimic_path),
            },
            {
                "source": "PulseDB_Vital",
                "file_name": vital_path.name,
                "sha256": sha256(vital_path),
            },
        ]
    ).to_csv(download_manifest, index=False)

    report = audit_pulsedb_pilot(
        data_root,
        tmp_path / "pilot_output",
        download_manifest=download_manifest,
        expected_per_source=1,
    )

    assert report["status"] in {"pass", "pass_with_warnings"}
    assert report["n_files"] == 2
    assert report["n_segments"] == 6
    assert report["n_subject_uids"] == 2
    assert report["required_failures"] == []

    combined = pd.read_parquet(report["artifacts"]["segment_index_parquet"])
    assert set(combined["dataset_source_directory"]) == {
        "PulseDB_MIMIC",
        "PulseDB_Vital",
    }
    assert set(combined["subject_uid"]) == {"MIMIC:p000001", "VitalDB:p000001"}
    assert combined["segment_uid"].is_unique
    assert combined["ppg_raw_hdf5_path"].isna().all()
    assert combined["ppg_f_hdf5_path"].isna().all()
    assert combined["abp_raw_hdf5_path"].isna().all()
    assert combined["abp_f_hdf5_path"].isna().all()
    assert combined["t_hdf5_path"].isna().all()

    per_file_report_path = (
        tmp_path
        / "pilot_output"
        / "per_file"
        / "PulseDB_MIMIC"
        / "p000001"
        / "p000001_full_audit.json"
    )
    per_file_report = json.loads(per_file_report_path.read_text(encoding="utf-8"))
    assert per_file_report["input"]["field_scope"] == "project_fields_only"
    assert per_file_report["reference_summaries"]["PPG_Raw"]["targets_loaded"] is True
    assert per_file_report["reference_summaries"]["PPG_Record"]["targets_loaded"] is False


def test_duplicate_source_identifiers_fail_cleanly_but_event_preview_is_written(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "duplicate_ids.mat"
    output_dir = tmp_path / "duplicate_output"
    _synthetic_file(input_path)
    with h5py.File(input_path, "a") as h5file:
        for field in ("SegmentID", "WinID", "WinSeqID"):
            h5file[f"#refs#/{field}_1"][...] = 1.0

    report = audit_pulsedb_file(
        input_path,
        output_dir,
        source="MIMIC",
        project_fields_only=True,
    )

    assert report["status"] == "fail"
    assert "segment_ids_are_unique_within_record" in report["required_failures"]
    assert "win_ids_are_unique_within_record" in report["required_failures"]
    assert "win_seq_ids_are_unique_within_record" in report["required_failures"]
    index = pd.read_parquet(output_dir / "duplicate_ids_segment_index.parquet")
    assert index["segment_id"].duplicated().any()
    assert index["segment_uid"].is_unique
    assert (output_dir / "duplicate_ids_events_preview_60s.csv").is_file()
