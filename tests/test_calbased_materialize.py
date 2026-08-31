from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pulsedb_fewshot.calbased_materialize import materialize_calbased_ppg
from pulsedb_fewshot.calbased_protocol import PROTOCOL_ID


def _write_mat(path: Path, *, duplicate_across_roles: bool = False) -> None:
    with h5py.File(path, "w") as h5file:
        group = h5file.create_group("Subj_Wins")
        references = group.create_dataset("PPG_F", shape=(1, 6), dtype=h5py.ref_dtype)
        waves = [
            (np.sin(np.arange(1250) / (20.0 + index)) + index / 10.0)[None, :]
            for index in range(6)
        ]
        if duplicate_across_roles:
            waves[4] = waves[0].copy()
        for index, waveform in enumerate(waves):
            target = h5file.create_dataset(f"wave_{index}", data=waveform)
            references[0, index] = target.ref


def _manifests(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = {
        "protocol_id": PROTOCOL_ID,
        "split_mode": "random_disjoint",
        "protocol_seed": 7,
        "original_split": "meta_train",
        "subject_uid": "MIMIC:p000001",
        "source": "MIMIC",
        "raw_file": str(raw),
        "ppg_field": "PPG_F",
        "ppg_storage_mode": "references",
        "n_samples": 1250,
    }
    development = pd.DataFrame(
        [
            {
                **common,
                "role": role,
                "segment_uid": f"segment-{index}",
                "selection_rank": index + 1,
                "ppg_reference_index": index,
                "sbp": 120.0 + index,
                "dbp": 70.0 + index,
            }
            for index, role in enumerate(
                ["train", "train", "internal_validation", "internal_validation"]
            )
        ]
    )
    heldout = pd.DataFrame(
        [
            {
                **common,
                "role": "heldout_test",
                "segment_uid": f"segment-{index}",
                "selection_rank": index + 1,
                "ppg_reference_index": index,
            }
            for index in (4, 5)
        ]
    )
    return development, heldout


def test_materialize_calbased_store_layout_and_exact_hash_audit(tmp_path: Path) -> None:
    raw = tmp_path / "subject.mat"
    _write_mat(raw)
    development, heldout = _manifests(raw)
    development_path = tmp_path / "development_fit.parquet"
    heldout_path = tmp_path / "heldout_inputs.parquet"
    development.to_parquet(development_path, index=False)
    heldout.to_parquet(heldout_path, index=False)

    output = tmp_path / "store"
    report = materialize_calbased_ppg(
        development_path,
        heldout_path,
        output,
        train_shards=1,
        validation_shards=1,
        heldout_shards=1,
        workers=1,
    )

    store = output / "random_disjoint"
    assert (store / "train_metadata_000.parquet").is_file()
    assert (store / "train_signals_000.npy").is_file()
    assert (store / "internal_validation_metadata_000.parquet").is_file()
    assert (store / "heldout_test_metadata_000.parquet").is_file()
    assert (output / "materialization.json").is_file()
    assert report["status"] == "pass"
    assert report["source_subject_split"] == "frozen meta_train only"
    assert report["heldout_test_targets_accessed"] is False
    assert report["screen_loader_roles"] == ["train", "internal_validation"]
    assert report["screen_loader_includes_heldout_test"] is False
    exact_audit = report["exact_ppg_content_overlap_audits"]["random_disjoint"]
    assert exact_audit["status"] == "pass"
    assert not any(
        exact_audit["cross_role_overlap_counts"].values()
    )

    train = pd.read_parquet(store / "train_metadata_000.parquet")
    heldout_metadata = pd.read_parquet(store / "heldout_test_metadata_000.parquet")
    required = {
        "protocol_id",
        "split_mode",
        "role",
        "subject_uid",
        "source",
        "segment_uid",
        "waveform_file",
        "waveform_row",
        "n_samples",
        "ppg_content_sha256",
    }
    assert required.issubset(train.columns)
    assert required.issubset(heldout_metadata.columns)
    assert {"sbp", "dbp"}.issubset(train.columns)
    assert not {"sbp", "dbp"} & set(heldout_metadata.columns)
    signals = np.load(output / train.loc[0, "waveform_file"], mmap_mode="r")
    assert signals.shape == (2, 1250)

    chronological_development = development.assign(
        split_mode="chronological_blocked"
    )
    chronological_heldout = heldout.assign(split_mode="chronological_blocked")
    chronological_development_path = tmp_path / "chronological_development.parquet"
    chronological_heldout_path = tmp_path / "chronological_heldout.parquet"
    chronological_development.to_parquet(chronological_development_path, index=False)
    chronological_heldout.to_parquet(chronological_heldout_path, index=False)
    combined_report = materialize_calbased_ppg(
        chronological_development_path,
        chronological_heldout_path,
        output,
        train_shards=1,
        validation_shards=1,
        heldout_shards=1,
        workers=1,
    )
    assert combined_report["split_modes_materialized"] == [
        "chronological_blocked",
        "random_disjoint",
    ]
    assert set(combined_report["materializations"]) == {
        "random_disjoint",
        "chronological_blocked",
    }
    assert (
        output
        / "chronological_blocked"
        / "heldout_test_metadata_000.parquet"
    ).is_file()


def test_materialization_rejects_exact_ppg_content_shared_across_roles(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "duplicate.mat"
    _write_mat(raw, duplicate_across_roles=True)
    development, heldout = _manifests(raw)
    development_path = tmp_path / "development_fit.parquet"
    heldout_path = tmp_path / "heldout_inputs.parquet"
    development.to_parquet(development_path, index=False)
    heldout.to_parquet(heldout_path, index=False)

    with pytest.raises(AssertionError, match="duplicated within the accepted protocol"):
        materialize_calbased_ppg(
            development_path,
            heldout_path,
            tmp_path / "duplicate_store",
            train_shards=1,
            validation_shards=1,
            heldout_shards=1,
            workers=1,
        )
