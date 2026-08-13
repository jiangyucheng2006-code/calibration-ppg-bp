from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pulsedb_fewshot.materialize import materialize_event_waveforms


def _write_mat(path: Path) -> None:
    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("Subj_Wins")
        references = group.create_dataset("PPG_F", shape=(1, 10), dtype=h5py.ref_dtype)
        for index in range(10):
            waveform = h5file.create_dataset(
                f"wave_{index}", data=np.linspace(index, index + 1, 1250)[None, :]
            )
            references[0, index] = waveform.ref


def _manifest(path: Path, subject: str, split: str, with_targets: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "event_id": [f"{subject}::{i:04d}" for i in range(1, 11)],
            "subject_uid": subject,
            "split": split,
            "event_index": range(1, 11),
            "raw_file": str(path),
            "ppg_storage_mode": "references",
            "ppg_reference_index": range(10),
            "n_samples": 1250,
        }
    )
    if with_targets:
        frame["sbp"] = 120.0
        frame["dbp"] = 70.0
    return frame


def test_materialization_preserves_rows_and_excludes_locked_targets(tmp_path: Path) -> None:
    raw = tmp_path / "subject.mat"
    _write_mat(raw)
    development = _manifest(raw, "MIMIC:s1", "meta_train", True)
    locked = _manifest(raw, "VitalDB:s2", "meta_test", False)
    development_path = tmp_path / "development.parquet"
    locked_path = tmp_path / "locked.parquet"
    development.to_parquet(development_path, index=False)
    locked.to_parquet(locked_path, index=False)
    output = tmp_path / "store"
    report = materialize_event_waveforms(
        development_path,
        locked_path,
        output,
        development_shards=1,
        locked_shards=1,
        workers=1,
    )
    assert report["status"] == "pass"
    assert report["materialized_rows"] == 20
    assert not report["locked_query_bp_present"]
    signal = np.load(output / "development_signals_000.npy", mmap_mode="r")
    assert signal.shape == (10, 1250)
