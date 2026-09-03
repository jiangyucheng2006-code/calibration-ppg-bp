import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pulsedb_fewshot.calbased_train as calbased_train_module
from pulsedb_fewshot.calbased_report import aggregate_screen_runs
from pulsedb_fewshot.calbased_screen import (
    PROTOCOL_ID,
    ROLE_WINDOWS_PER_SUBJECT,
    SUBJECT_COUNT,
    SUBJECT_SET_SHA256,
)
from pulsedb_fewshot.calbased_train import (
    EXECUTABLE_CANDIDATES,
    load_screen_metadata,
    train_candidate,
)


def _write_manifest(store: Path) -> None:
    store.mkdir(parents=True)
    (store / "materialization.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "source_parent_split": "meta_train",
                "source_parent_splits": ["meta_train"],
                "status": "pass",
                "subject_count": SUBJECT_COUNT,
                "subject_set_sha256": SUBJECT_SET_SHA256,
                "windows_per_subject": ROLE_WINDOWS_PER_SUBJECT,
                "split_modes_materialized": [
                    "random_disjoint",
                    "chronological_blocked",
                ],
                "exact_ppg_content_overlap_audits": {
                    mode: {
                        "status": "pass",
                        "global_exact_content_unique": True,
                        "cross_role_overlap_counts": {
                            "train__internal_validation": 0,
                            "train__heldout_test": 0,
                            "internal_validation__heldout_test": 0,
                        },
                        "within_role_duplicate_counts": {
                            "train": 0,
                            "internal_validation": 0,
                            "heldout_test": 0,
                        },
                    }
                    for mode in ("random_disjoint", "chronological_blocked")
                },
                "meta_validation_windows_accessed": False,
                "locked_meta_test_windows_accessed": False,
                "heldout_test_targets_accessed": False,
                "heldout_test_targets_path_accepted_by_entrypoint": False,
                "screen_loader_includes_heldout_test": False,
            }
        ),
        encoding="utf-8",
    )


def _role_frame(role: str, waveform_file: str, offset: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protocol_id": [PROTOCOL_ID, PROTOCOL_ID],
            "split_mode": ["random_disjoint", "random_disjoint"],
            "role": [role, role],
            "subject_uid": ["a", "b"],
            "source": ["MIMIC", "VitalDB"],
            "segment_uid": [f"a-{offset}", f"b-{offset}"],
            "waveform_file": [waveform_file, waveform_file],
            "waveform_row": [0, 1],
            "n_samples": [1250, 1250],
            "sbp": [110.0 + offset, 130.0 + offset],
            "dbp": [65.0 + offset, 80.0 + offset],
        }
    )


def test_role_loader_never_opens_heldout_metadata(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _write_manifest(store)
    split = store / "random_disjoint"
    split.mkdir()
    signals = np.stack(
        [
            np.sin(np.linspace(0, 20, 1250)),
            np.cos(np.linspace(0, 20, 1250)),
        ]
    ).astype(np.float32)
    np.save(split / "train_signals_000.npy", signals)
    np.save(split / "internal_validation_signals_000.npy", signals)
    _role_frame(
        "train", "random_disjoint/train_signals_000.npy", 0
    ).to_parquet(split / "train_metadata_000.parquet", index=False)
    _role_frame(
        "internal_validation",
        "random_disjoint/internal_validation_signals_000.npy",
        1,
    ).to_parquet(split / "internal_validation_metadata_000.parquet", index=False)
    # The deliberately invalid file proves that the first-round loader does
    # not glob or parse held-out metadata.
    (split / "heldout_test_metadata_000.parquet").write_bytes(b"sealed-target-placeholder")
    train, validation = load_screen_metadata(
        store, "random_disjoint", enforce_full_cohort=False
    )
    assert len(train) == len(validation) == 2
    assert set(train["role"]) == {"train"}
    assert set(validation["role"]) == {"internal_validation"}


def test_only_safe_first_round_candidates_are_executable() -> None:
    assert "subject_train_mean" in EXECUTABLE_CANDIDATES
    assert "subject_mean_residual_ppg" in EXECUTABLE_CANDIDATES
    assert "compact_resnet" in EXECUTABLE_CANDIDATES
    assert "inception_time_wide" in EXECUTABLE_CANDIDATES
    assert "patch_transformer" in EXECUTABLE_CANDIDATES
    assert "compact_resnet_qgh" not in EXECUTABLE_CANDIDATES
    assert "compact_resnet_calibration_relative" not in EXECUTABLE_CANDIDATES


def _metric(mean_mae: float) -> dict[str, object]:
    return {
        "n_participants": 2,
        "n_events": 4,
        "sbp_mae": mean_mae + 1,
        "dbp_mae": mean_mae - 1,
        "mean_mae": mean_mae,
        "sbp_bias": 0.1,
        "dbp_bias": -0.1,
        "worst_30_mean_mae": mean_mae + 3,
        "retained_70_mean_mae": mean_mae - 1,
    }


def _write_fake_run(run_dir: Path, candidate: str, mean_mae: float) -> None:
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol_id": PROTOCOL_ID,
                "candidate": candidate,
                "runner": "population_regression",
                "backbone": "resnet_small",
                "split_mode": "random_disjoint",
                "source_parent_split": "meta_train",
                "read_roles": ["train", "internal_validation"],
                "selection_role": "internal_validation",
                "heldout_test_accessed": False,
                "seed": 7,
                "metrics": {
                    "Overall": _metric(mean_mae),
                    "MIMIC": _metric(mean_mae + 0.2),
                    "VitalDB": _metric(mean_mae - 0.2),
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "event_id": ["m1", "m2", "v1", "v2"],
            "subject_uid": ["m", "m", "v", "v"],
            "source": ["MIMIC", "MIMIC", "VitalDB", "VitalDB"],
            "target_sbp": [110.0, 120.0, 130.0, 140.0],
            "target_dbp": [65.0, 70.0, 75.0, 80.0],
            "pred_sbp": [111.0, 119.0, 132.0, 138.0],
            "pred_dbp": [66.0, 69.0, 76.0, 79.0],
        }
    ).to_parquet(
        run_dir / "best_internal_validation_predictions.parquet", index=False
    )


def test_report_selects_only_by_internal_validation(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_fake_run(first, "compact_resnet", 8.0)
    _write_fake_run(second, "subject_mean_residual_ppg", 7.0)
    result = aggregate_screen_runs([first, second], tmp_path / "report")
    assert result["winner"]["candidate"] == "subject_mean_residual_ppg"
    assert result["heldout_test_accessed"] is False
    assert (tmp_path / "report" / "internal_validation_summary.csv").is_file()
    for suffix in ("overall", "mimic", "vitaldb"):
        table = pd.read_csv(tmp_path / "report" / f"event_pooled_{suffix}.csv")
        assert table.columns.tolist() == [
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
        ]


def test_report_accepts_float_serialization_noise_and_row_reordering(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_fake_run(first, "compact_resnet", 8.0)
    _write_fake_run(second, "subject_mean_residual_ppg", 7.0)
    path = second / "best_internal_validation_predictions.parquet"
    predictions = pd.read_parquet(path).iloc[::-1].reset_index(drop=True)
    predictions["target_sbp"] = predictions["target_sbp"].astype("float32")
    predictions["target_dbp"] = predictions["target_dbp"].astype("float32")
    predictions.loc[0, "target_sbp"] += 5e-5
    predictions.to_parquet(path, index=False)

    result = aggregate_screen_runs([first, second], tmp_path / "report")

    assert result["winner"]["candidate"] == "subject_mean_residual_ppg"


def test_report_rejects_material_target_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_fake_run(first, "compact_resnet", 8.0)
    _write_fake_run(second, "subject_mean_residual_ppg", 7.0)
    path = second / "best_internal_validation_predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions.loc[0, "target_sbp"] += 0.01
    predictions.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="identical internal-validation"):
        aggregate_screen_runs([first, second], tmp_path / "report")


def test_report_rejects_any_claim_of_heldout_access(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_fake_run(run_dir, "compact_resnet", 8.0)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["heldout_test_accessed"] = True
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="accessed heldout_test"):
        aggregate_screen_runs([run_dir], tmp_path / "report")


def test_runner_refuses_deferred_candidate_before_loading_data(tmp_path: Path) -> None:
    args = argparse.Namespace(
        candidate="compact_resnet_qgh",
        patience=8,
        epochs=0,
        examples_per_epoch=1,
        seed=1,
        store_root=tmp_path / "missing",
        split_mode="random_disjoint",
        output=tmp_path / "out",
        require_cuda=False,
        batch_size=1,
        workers=0,
        learning_rate=3e-4,
        weight_decay=1e-4,
        huber_delta=0.5,
    )
    with pytest.raises(ValueError, match="not executable"):
        train_candidate(args)


def _runner_args(tmp_path: Path, candidate: str, output_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        candidate=candidate,
        patience=8,
        epochs=0,
        examples_per_epoch=2,
        seed=1,
        store_root=tmp_path / "runner_store",
        split_mode="random_disjoint",
        output=tmp_path / output_name,
        require_cuda=False,
        batch_size=2,
        workers=0,
        learning_rate=0.0,
        weight_decay=0.0,
        huber_delta=0.5,
    )


def _small_runner_roles(store: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    _write_manifest(store)
    waveforms = np.stack(
        [
            np.sin(np.linspace(0, 10, 250)),
            np.cos(np.linspace(0, 10, 250)),
        ]
    ).astype(np.float32)
    np.save(store / "signals.npy", waveforms)
    common = {
        "subject_uid": ["a", "b"],
        "source": ["MIMIC", "VitalDB"],
        "waveform_file": ["signals.npy", "signals.npy"],
        "waveform_row": [0, 1],
    }
    train = pd.DataFrame(
        common
        | {
            "role": ["train", "train"],
            "segment_uid": ["a-train", "b-train"],
            "sbp": [110.0, 140.0],
            "dbp": [65.0, 85.0],
        }
    )
    validation = pd.DataFrame(
        common
        | {
            "role": ["internal_validation", "internal_validation"],
            "segment_uid": ["a-val", "b-val"],
            "sbp": [112.0, 138.0],
            "dbp": [66.0, 84.0],
        }
    )
    return train, validation


def test_analytic_and_neural_runners_write_validation_only_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation = _small_runner_roles(tmp_path / "runner_store")
    monkeypatch.setattr(
        calbased_train_module,
        "load_screen_metadata",
        lambda *_args, **_kwargs: (train, validation),
    )
    baseline = train_candidate(
        _runner_args(tmp_path, "subject_train_mean", "baseline")
    )
    neural = train_candidate(
        _runner_args(tmp_path, "subject_mean_residual_ppg", "neural")
    )
    for result, output_name in ((baseline, "baseline"), (neural, "neural")):
        assert result["selection_role"] == "internal_validation"
        assert result["heldout_test_accessed"] is False
        assert (tmp_path / output_name / "run.json").is_file()
        assert (
            tmp_path
            / output_name
            / "best_internal_validation_predictions.parquet"
        ).is_file()
    assert baseline["stop_reason"] == "analytic_baseline"
    assert neural["stop_reason"] == "early_stopping"
    assert neural["epochs_completed"] == 9
