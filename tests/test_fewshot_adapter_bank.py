from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulsedb_fewshot.fewshot_adapter_bank import (
    BASIS_COUNTS,
    EARLY_FOLD,
    EXPECTED_FOLDS,
    FIT_FOLDS,
    KS,
    ROUTING_MODES,
    SELECTION_FOLD,
    CachedEpisodeDataset,
    PreparedAdapterCache,
    _artifact_hashes,
)
from pulsedb_fewshot.training import save_json


def _make_prepared_cache(root: Path) -> PreparedAdapterCache:
    root.mkdir()
    rows: list[dict[str, object]] = []
    support_index: list[dict[str, object]] = []
    query_features: list[np.ndarray] = []
    support_features: list[np.ndarray] = []
    support_bp: list[np.ndarray] = []
    targets: dict[str, list[dict[str, object]]] = {
        "fit": [],
        "early": [],
        "selection": [],
    }
    for fold in sorted(EXPECTED_FOLDS):
        subject = f"MIMIC:s{fold}"
        support_features.append(np.full((5, 8), fold, dtype=np.float32))
        support_bp.append(np.full((5, 2), fold / 10.0, dtype=np.float32))
        for position in range(5):
            support_index.append(
                {
                    "subject_uid": subject,
                    "event_id": f"{subject}:e{position + 1}",
                    "source": "MIMIC",
                    "split": "meta_train",
                    "fold": fold,
                    "event_index": position + 1,
                    "support_position": position,
                    "support_row": fold,
                }
            )
        query_row = len(query_features)
        event_id = f"{subject}:e6"
        rows.append(
            {
                "subject_uid": subject,
                "event_id": event_id,
                "source": "MIMIC",
                "split": "meta_train",
                "fold": fold,
                "event_index": 6,
                "record_order": fold,
                "time_bin": fold,
                "support_row": fold,
                "query_row": query_row,
            }
        )
        query_features.append(np.full(8, fold + 0.5, dtype=np.float32))
        role = "fit" if fold in FIT_FOLDS else "early" if fold == EARLY_FOLD else "selection"
        targets[role].append(
            {
                "subject_uid": subject,
                "event_id": event_id,
                "target_sbp": 120.0 + fold,
                "target_dbp": 70.0 + fold,
            }
        )

    pd.DataFrame(rows).to_parquet(root / "query_inputs.parquet", index=False)
    pd.DataFrame(support_index).to_parquet(root / "support_index.parquet", index=False)
    for role, values in targets.items():
        pd.DataFrame(values).to_parquet(root / f"{role}_targets.parquet", index=False)
    np.save(root / "query_embeddings.npy", np.stack(query_features))
    np.save(root / "support_embeddings.npy", np.stack(support_features))
    np.save(root / "support_bp_norm.npy", np.stack(support_bp))
    record = {
        "status": "complete",
        "screen_id": "fewshot-adapter-bank-v1",
        "stage": "feature_cache",
        "split": "meta_train_internal_cache",
        "backbone": "resnet_small",
        "feature_dim": 256,
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_FOLD,
        "selection_fold": SELECTION_FOLD,
        "support_policy": "fixed_first",
        "ks": list(KS),
        "target_scaler": {"mean": [120.0, 70.0], "std": [20.0, 10.0]},
        "meta_validation_used_for_training": False,
        "meta_validation_used_for_early_stopping": False,
        "meta_validation_used_for_candidate_ranking": False,
        "meta_validation_predictions_generated": False,
        "locked_test_accessed": False,
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "participant_identity_model_input": False,
        "source_model_input": False,
    }
    record["artifact_sha256"] = _artifact_hashes(root)
    save_json(root / "run.json", record)
    return PreparedAdapterCache(root)


def test_matrix_has_twelve_jobs_but_eleven_unique_routing_functions() -> None:
    matrix = [(count, mode) for count in BASIS_COUNTS for mode in ROUTING_MODES]
    assert len(matrix) == 12
    assert len(set(matrix)) == 12
    # With only five bases, Top-5 keeps every basis and is therefore the same
    # routing function as dense.  It remains duplicated as a consistency job.
    functional = {
        (count, "all" if count == 5 else mode) for count, mode in matrix
    }
    assert len(functional) == 11


def test_prepared_inputs_exclude_query_bp_and_keep_roles_disjoint(tmp_path: Path) -> None:
    cache = _make_prepared_cache(tmp_path / "prepared")
    assert not {"sbp", "dbp", "target_sbp", "target_dbp"} & set(cache.inputs)
    roles = {
        role: set(cache.input_role(role)["subject_uid"])
        for role in ("fit", "early", "selection")
    }
    assert roles["fit"].isdisjoint(roles["early"])
    assert roles["fit"].isdisjoint(roles["selection"])
    assert roles["early"].isdisjoint(roles["selection"])


@pytest.mark.parametrize("role", ["fit", "early", "selection"])
def test_cached_dataset_uses_fixed_first_k_supports(tmp_path: Path, role: str) -> None:
    cache = _make_prepared_cache(tmp_path / f"prepared_{role}")
    frame = (
        cache.input_role(role)
        if role == "selection"
        else cache.labelled_role(role)
    )
    dataset = CachedEpisodeDataset(
        frame,
        cache,
        include_target=role != "selection",
    )
    assert len(dataset) == len(frame) * 4
    assert [int(dataset[index]["support_mask"].sum()) for index in range(4)] == [1, 2, 3, 5]
    assert all("target" not in dataset[index] for index in range(4)) if role == "selection" else all(
        "target" in dataset[index] for index in range(4)
    )


def test_cache_hash_detects_artifact_mutation(tmp_path: Path) -> None:
    cache = _make_prepared_cache(tmp_path / "prepared")
    path = cache.root / "query_inputs.parquet"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(AssertionError, match="artifact hash mismatch"):
        PreparedAdapterCache(cache.root)
