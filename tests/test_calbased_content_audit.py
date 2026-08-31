from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pulsedb_fewshot.calbased_content_audit import (
    audit_calbased_candidate_content,
)


def _write_waveforms(path: Path) -> None:
    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("Subj_Wins")
        references = group.create_dataset("PPG_F", shape=(1, 4), dtype=h5py.ref_dtype)
        sample = np.arange(1250, dtype=float)
        waves = [np.sin(sample / (18.0 + index))[None, :] for index in range(4)]
        for index, waveform in enumerate(waves):
            target = h5file.create_dataset(f"wave_{index}", data=waveform)
            references[0, index] = target.ref


def _mode_frame(raw: Path, split_mode: str, references: tuple[int, int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split_mode": split_mode,
            "role": ["train", "internal_validation", "heldout_test"],
            "subject_uid": "MIMIC:p000001",
            "source": "MIMIC",
            "segment_uid": [f"{split_mode}-{index}" for index in range(3)],
            "raw_file": str(raw),
            "ppg_storage_mode": "references",
            "ppg_reference_index": references,
            "n_samples": 1250,
        }
    )


def test_content_audit_does_not_compare_independent_split_modes(tmp_path: Path) -> None:
    raw = tmp_path / "waves.mat"
    _write_waveforms(raw)
    mode_inputs = {
        "random_disjoint": _mode_frame(raw, "random_disjoint", (0, 1, 2)),
        "chronological_blocked": _mode_frame(
            raw, "chronological_blocked", (0, 1, 2)
        ),
    }

    report, excluded = audit_calbased_candidate_content(mode_inputs, workers=1)

    assert report["status"] == "pass"
    assert report["split_modes_compared_independently"] is True
    assert report["bp_target_columns_loaded"] is False
    assert report["heldout_test_targets_accessed"] is False
    assert excluded == set()


def test_content_audit_excludes_subject_with_within_mode_duplicate(tmp_path: Path) -> None:
    raw = tmp_path / "waves.mat"
    _write_waveforms(raw)
    mode_inputs = {
        "random_disjoint": _mode_frame(raw, "random_disjoint", (0, 0, 2)),
        "chronological_blocked": _mode_frame(
            raw, "chronological_blocked", (0, 1, 2)
        ),
    }

    report, excluded = audit_calbased_candidate_content(mode_inputs, workers=1)

    assert report["status"] == "exclude_then_rebuild"
    assert report["excluded_subject_count"] == 1
    assert excluded == {"MIMIC:p000001"}
    random = report["modes"]["random_disjoint"]
    assert random["duplicate_hash_groups"] == 1
    assert random["cross_role_overlap_counts"]["train__internal_validation"] == 1
