"""Leakage-safe variable-K support-conditioned adapter-bank screen.

This development screen uses only participant-disjoint internal ``meta_train``
folds: folds 0--2 fit the model, fold 3 controls early stopping, and fold 4 is
scored only after the selected checkpoint is frozen.  The primary
``meta_validation`` split and locked meta-test remain inaccessible.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .models import VariableKPersonalizer, model_parameter_counts
from .report_round8 import _diagnostic_rows
from .train import _load_population_checkpoint
from .training import (
    WaveformAccessor,
    file_sha256,
    load_store_metadata,
    participant_macro_metrics,
    save_json,
    seed_everything,
    source_tree_sha256,
)


FIT_FOLDS = (0, 1, 2)
EARLY_FOLD = 3
SELECTION_FOLD = 4
EXPECTED_FOLDS = set(range(5))
KS = (1, 2, 3, 5)
BASIS_COUNTS = (5, 10, 15, 20, 25, 30)
ROUTING_MODES = ("top5", "dense")
QUERY_KEYS = ["subject_uid", "event_id", "k"]
TARGET_COLUMNS = ["target_sbp", "target_dbp"]
SCOPES = ("Overall", "MIMIC", "VitalDB")
SCREEN_ID = "fewshot-adapter-bank-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )


def _require_false(record: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        if record.get(key) is not False:
            raise AssertionError(f"{label} has unsafe or missing flag {key}")


def _artifact_hashes(root: Path) -> dict[str, str]:
    names = (
        "query_inputs.parquet",
        "fit_targets.parquet",
        "early_targets.parquet",
        "selection_targets.parquet",
        "support_index.parquet",
        "query_embeddings.npy",
        "support_embeddings.npy",
        "support_bp_norm.npy",
    )
    return {name: file_sha256(root / name) for name in names}


def _validate_population_run(
    run_root: Path,
    *,
    folds_path: Path,
    store_root: Path,
) -> tuple[dict[str, Any], Path]:
    record = _read_json(run_root / "run.json")
    expected = {
        "status": "complete",
        "method": "population",
        "backbone": "resnet_small",
        "feature_dim": 256,
        "crossfit_fit_folds": list(FIT_FOLDS),
        "crossfit_validation_fold": EARLY_FOLD,
        "crossfit_excluded_folds": [SELECTION_FOLD],
        "locked_test_accessed": False,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"population run has unexpected {key}")
    arguments = record.get("arguments")
    if not isinstance(arguments, dict):
        raise AssertionError("population run lacks arguments")
    if Path(str(arguments.get("store_root"))) != store_root:
        raise AssertionError("population run used a different event store")
    if Path(str(arguments.get("crossfit_folds"))) != folds_path:
        raise AssertionError("population run used a different fold table")
    if record.get("store_manifest_sha256") != file_sha256(
        store_root / "materialization.json"
    ):
        raise AssertionError("population run store hash differs")
    if record.get("crossfit_folds_sha256") != file_sha256(folds_path):
        raise AssertionError("population run fold hash differs")
    checkpoint = run_root / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if record.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise AssertionError("population checkpoint hash differs from run record")
    return record, checkpoint


def _validate_folds(folds: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_uid", "source", "fold"}
    missing = required - set(folds)
    if missing:
        raise ValueError(f"fold table missing {sorted(missing)}")
    result = folds[list(required)].copy()
    result["subject_uid"] = result["subject_uid"].astype(str)
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    if result["subject_uid"].duplicated().any():
        raise AssertionError("fold table contains duplicate participants")
    if set(result["fold"].unique()) != EXPECTED_FOLDS:
        raise AssertionError("adapter-bank screen requires exactly folds 0--4")
    subjects = set(metadata["subject_uid"].astype(str))
    if set(result["subject_uid"]) != subjects:
        raise AssertionError("fold table does not exactly cover meta-train participants")
    source = metadata[["subject_uid", "source"]].drop_duplicates().assign(
        subject_uid=lambda frame: frame["subject_uid"].astype(str)
    )
    checked = result.merge(
        source,
        on="subject_uid",
        suffixes=("_fold", "_metadata"),
        validate="one_to_one",
    )
    if not checked["source_fold"].astype(str).equals(
        checked["source_metadata"].astype(str)
    ):
        raise AssertionError("fold and event-store source labels differ")
    return result


def _encode_rows(
    frame: pd.DataFrame,
    *,
    accessor: WaveformAccessor,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dimension = int(model.encoder.feature_dim)
    result = np.empty((len(frame), dimension), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            part = frame.iloc[start : start + batch_size]
            waveforms = torch.stack(
                [
                    accessor.get(str(row.waveform_file), int(row.waveform_row))
                    for row in part.itertuples(index=False)
                ]
            ).to(device)
            with _autocast(device):
                features = model.encoder(waveforms)
            result[start : start + len(part)] = (
                features.float().cpu().numpy().astype(np.float32, copy=False)
            )
    return result


def prepare_cache(
    *,
    store_root: Path,
    folds_path: Path,
    population_run: Path,
    output: Path,
    batch_size: int = 1024,
    require_cuda: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if batch_size < 1:
        raise ValueError("cache batch size must be positive")
    population_record, population_checkpoint = _validate_population_run(
        population_run,
        folds_path=folds_path,
        store_root=store_root,
    )
    metadata = load_store_metadata(store_root, "development")
    metadata["subject_uid"] = metadata["subject_uid"].astype(str)
    if metadata["split"].eq("meta_test").any():
        raise AssertionError("development event store unexpectedly contains meta-test")
    selected = metadata.loc[metadata["split"].eq("meta_train")].copy()
    if selected.empty or selected["split"].ne("meta_train").any():
        raise AssertionError("adapter cache must contain meta-train only")
    folds = _validate_folds(pd.read_parquet(folds_path), selected)
    selected = selected.merge(
        folds[["subject_uid", "fold"]],
        on="subject_uid",
        validate="many_to_one",
    )

    query = selected.loc[
        selected["common_query"].astype(bool) & selected["event_index"].gt(max(KS))
    ].copy()
    query.sort_values(
        ["subject_uid", "event_index", "event_id"], kind="mergesort", inplace=True
    )
    query.reset_index(drop=True, inplace=True)
    participants = set(query["subject_uid"])
    if not participants:
        raise AssertionError("adapter cache has no eligible queries")

    support = selected.loc[
        selected["subject_uid"].isin(participants)
        & selected["event_index"].le(max(KS))
    ].copy()
    support.sort_values(
        ["subject_uid", "event_index", "event_id"], kind="mergesort", inplace=True
    )
    support.reset_index(drop=True, inplace=True)
    counts = support.groupby("subject_uid", sort=False).size()
    if not counts.eq(max(KS)).all() or set(counts.index.astype(str)) != participants:
        raise AssertionError("every adapter-cache participant requires five supports")
    support["support_position"] = support.groupby("subject_uid").cumcount()
    ordered_subjects = sorted(participants)
    support_map = {subject: index for index, subject in enumerate(ordered_subjects)}
    support["support_row"] = support["subject_uid"].map(support_map).astype(np.int64)
    query["support_row"] = query["subject_uid"].map(support_map).astype(np.int64)
    query["query_row"] = np.arange(len(query), dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("adapter cache generation requires CUDA")
    population, scaler = _load_population_checkpoint(population_checkpoint, device)
    population.to(device).eval()
    accessor = WaveformAccessor(store_root)
    support_features_flat = _encode_rows(
        support,
        accessor=accessor,
        model=population,
        device=device,
        batch_size=batch_size,
    )
    query_features = _encode_rows(
        query,
        accessor=accessor,
        model=population,
        device=device,
        batch_size=batch_size,
    )
    n_subjects = len(ordered_subjects)
    support_features = support_features_flat.reshape(n_subjects, max(KS), -1)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    support_bp = support[["sbp", "dbp"]].to_numpy(np.float32).reshape(
        n_subjects, max(KS), 2
    )
    support_bp_norm = (support_bp - mean) / std

    input_columns = [
        "subject_uid",
        "event_id",
        "source",
        "split",
        "fold",
        "event_index",
        "record_order",
        "time_bin",
        "support_row",
        "query_row",
    ]
    missing_inputs = set(input_columns) - set(query)
    if missing_inputs:
        raise ValueError(f"query metadata missing {sorted(missing_inputs)}")
    query_inputs = query[input_columns].copy()
    if {"sbp", "dbp", *TARGET_COLUMNS} & set(query_inputs):
        raise AssertionError("query input table exposes query BP")

    targets = query[["subject_uid", "event_id", "fold", "sbp", "dbp"]].rename(
        columns={"sbp": "target_sbp", "dbp": "target_dbp"}
    )
    role_folds = {
        "fit": set(FIT_FOLDS),
        "early": {EARLY_FOLD},
        "selection": {SELECTION_FOLD},
    }
    target_tables = {
        role: targets.loc[targets["fold"].isin(role_values)].drop(columns="fold")
        for role, role_values in role_folds.items()
    }
    participant_roles = {
        role: set(
            query_inputs.loc[
                query_inputs["fold"].isin(role_values), "subject_uid"
            ].astype(str)
        )
        for role, role_values in role_folds.items()
    }
    if (
        participant_roles["fit"] & participant_roles["early"]
        or participant_roles["fit"] & participant_roles["selection"]
        or participant_roles["early"] & participant_roles["selection"]
    ):
        raise AssertionError("adapter cache participant roles overlap")

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "query_embeddings.npy", query_features)
    np.save(output / "support_embeddings.npy", support_features)
    np.save(output / "support_bp_norm.npy", support_bp_norm.astype(np.float32))
    query_inputs.to_parquet(output / "query_inputs.parquet", index=False)
    for role, frame in target_tables.items():
        frame.to_parquet(output / f"{role}_targets.parquet", index=False)
    support[
        [
            "subject_uid",
            "event_id",
            "source",
            "split",
            "fold",
            "event_index",
            "support_position",
            "support_row",
        ]
    ].to_parquet(output / "support_index.parquet", index=False)

    artifacts = _artifact_hashes(output)
    fold_counts = {
        str(fold): {
            "participants": int(group["subject_uid"].nunique()),
            "queries": int(len(group)),
        }
        for fold, group in query_inputs.groupby("fold")
    }
    record: dict[str, Any] = {
        "status": "complete",
        "screen_id": SCREEN_ID,
        "stage": "feature_cache",
        "split": "meta_train_internal_cache",
        "population_seed": int(population_record["seed"]),
        "backbone": "resnet_small",
        "feature_dim": 256,
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_FOLD,
        "selection_fold": SELECTION_FOLD,
        "support_policy": "fixed_first",
        "ks": list(KS),
        "participants": int(query_inputs["subject_uid"].nunique()),
        "queries": int(len(query_inputs)),
        "fold_counts": fold_counts,
        "target_scaler": scaler,
        "population_run_json_sha256": file_sha256(population_run / "run.json"),
        "population_checkpoint_sha256": file_sha256(population_checkpoint),
        "folds_sha256": file_sha256(folds_path),
        "store_manifest_sha256": file_sha256(store_root / "materialization.json"),
        "artifact_sha256": artifacts,
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "meta_validation_used_for_training": False,
        "meta_validation_used_for_early_stopping": False,
        "meta_validation_used_for_candidate_ranking": False,
        "meta_validation_predictions_generated": False,
        "locked_test_accessed": False,
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "participant_identity_model_input": False,
        "source_model_input": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output / "run.json", record)
    return record


class PreparedAdapterCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.record = _read_json(root / "run.json")
        expected = {
            "status": "complete",
            "screen_id": SCREEN_ID,
            "stage": "feature_cache",
            "split": "meta_train_internal_cache",
            "backbone": "resnet_small",
            "feature_dim": 256,
            "fit_folds": list(FIT_FOLDS),
            "early_stopping_fold": EARLY_FOLD,
            "selection_fold": SELECTION_FOLD,
            "support_policy": "fixed_first",
            "ks": list(KS),
        }
        for key, value in expected.items():
            if self.record.get(key) != value:
                raise AssertionError(f"adapter cache has unexpected {key}")
        _require_false(
            self.record,
            (
                "meta_validation_used_for_training",
                "meta_validation_used_for_early_stopping",
                "meta_validation_used_for_candidate_ranking",
                "meta_validation_predictions_generated",
                "locked_test_accessed",
                "query_bp_model_input",
                "future_query_model_input",
                "participant_identity_model_input",
                "source_model_input",
            ),
            "adapter cache",
        )
        if self.record.get("artifact_sha256") != _artifact_hashes(root):
            raise AssertionError("adapter cache artifact hash mismatch")
        self.inputs = pd.read_parquet(root / "query_inputs.parquet")
        if {"sbp", "dbp", *TARGET_COLUMNS} & set(self.inputs):
            raise AssertionError("adapter cache input table exposes query BP")
        if self.inputs["split"].astype(str).ne("meta_train").any():
            raise AssertionError("adapter cache contains non-meta-train query inputs")
        if set(pd.to_numeric(self.inputs["fold"]).astype(int)) != EXPECTED_FOLDS:
            raise AssertionError("adapter cache does not contain folds 0--4")
        if self.inputs.duplicated(["subject_uid", "event_id"]).any():
            raise AssertionError("adapter cache has duplicate query keys")
        self.query_features = np.load(root / "query_embeddings.npy", mmap_mode="r")
        self.support_features = np.load(root / "support_embeddings.npy", mmap_mode="r")
        self.support_bp = np.load(root / "support_bp_norm.npy", mmap_mode="r")
        if len(self.query_features) != len(self.inputs):
            raise AssertionError("adapter cache query features are misaligned")
        if self.support_features.shape[:2] != self.support_bp.shape[:2]:
            raise AssertionError("adapter cache support arrays are misaligned")

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.record["target_scaler"]["mean"], dtype=np.float32)

    @property
    def std(self) -> np.ndarray:
        return np.asarray(self.record["target_scaler"]["std"], dtype=np.float32)

    def input_role(self, role: str) -> pd.DataFrame:
        fold_values = {
            "fit": set(FIT_FOLDS),
            "early": {EARLY_FOLD},
            "selection": {SELECTION_FOLD},
        }
        if role not in fold_values:
            raise KeyError(role)
        frame = self.inputs.loc[self.inputs["fold"].isin(fold_values[role])].copy()
        frame.sort_values(
            ["subject_uid", "event_index", "event_id"], kind="mergesort", inplace=True
        )
        frame.reset_index(drop=True, inplace=True)
        return frame

    def targets(self, role: str) -> pd.DataFrame:
        path = self.root / f"{role}_targets.parquet"
        frame = pd.read_parquet(path)
        required = {"subject_uid", "event_id", *TARGET_COLUMNS}
        if missing := required - set(frame):
            raise ValueError(f"{role} target table missing {sorted(missing)}")
        if frame.duplicated(["subject_uid", "event_id"]).any():
            raise AssertionError(f"{role} target table has duplicate keys")
        return frame

    def labelled_role(self, role: str) -> pd.DataFrame:
        inputs = self.input_role(role)
        return inputs.merge(
            self.targets(role),
            on=["subject_uid", "event_id"],
            how="left",
            validate="one_to_one",
        )


class CachedEpisodeDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        cache: PreparedAdapterCache,
        *,
        include_target: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cache = cache
        self.include_target = include_target
        if include_target and not set(TARGET_COLUMNS).issubset(self.frame):
            raise ValueError("labelled cached episodes require target columns")

    def __len__(self) -> int:
        return len(self.frame) * len(KS)

    def __getitem__(self, index: int) -> dict[str, Any]:
        query_position, k_position = divmod(index, len(KS))
        row = self.frame.iloc[query_position]
        k = KS[k_position]
        query_row = int(row.query_row)
        support_row = int(row.support_row)
        mask = np.zeros(max(KS), dtype=np.bool_)
        mask[:k] = True
        item: dict[str, Any] = {
            "query_features": torch.from_numpy(
                np.asarray(self.cache.query_features[query_row], dtype=np.float32).copy()
            ),
            "support_features": torch.from_numpy(
                np.asarray(self.cache.support_features[support_row], dtype=np.float32).copy()
            ),
            "support_bp": torch.from_numpy(
                np.asarray(self.cache.support_bp[support_row], dtype=np.float32).copy()
            ),
            "support_mask": torch.from_numpy(mask),
            "subject_uid": str(row.subject_uid),
            "event_id": str(row.event_id),
            "source": str(row.source),
            "k": k,
        }
        if self.include_target:
            target = np.asarray(
                [float(row.target_sbp), float(row.target_dbp)], dtype=np.float32
            )
            item["target"] = torch.from_numpy(
                ((target - self.cache.mean) / self.cache.std).astype(np.float32)
            )
        return item


def _participant_balanced_sampler(
    dataset: CachedEpisodeDataset,
    *,
    seed: int,
    num_samples: int,
) -> WeightedRandomSampler:
    counts = dataset.frame["subject_uid"].astype(str).value_counts()
    row_weights = dataset.frame["subject_uid"].astype(str).map(
        lambda value: 1.0 / counts[value]
    ).to_numpy(dtype=np.float64)
    weights = np.repeat(row_weights, len(KS))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _build_model(
    population_checkpoint: Path,
    *,
    device: torch.device,
    basis_count: int,
    routing_mode: str,
) -> tuple[VariableKPersonalizer, dict[str, list[float]]]:
    population, scaler = _load_population_checkpoint(population_checkpoint, device)
    if basis_count == 0:
        if routing_mode != "none":
            raise ValueError("M0 reference requires routing_mode=none")
        top_k = None
    else:
        if basis_count not in BASIS_COUNTS:
            raise ValueError(f"basis_count must be one of {BASIS_COUNTS}")
        if routing_mode not in ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of {ROUTING_MODES}")
        top_k = 5 if routing_mode == "top5" else None
    model = VariableKPersonalizer(
        population,
        use_film=False,
        query_conditioned_weights=False,
        anchor_mode="mean",
        adapter_basis_count=basis_count,
        adapter_rank=4,
        adapter_top_k=top_k,
    )
    for parameter in model.population.parameters():
        parameter.requires_grad = False
    return model.to(device), scaler


@torch.no_grad()
def _predict(
    model: VariableKPersonalizer,
    loader: DataLoader,
    *,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    include_targets: bool,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        prediction_norm = model.forward_from_features(
            batch["query_features"].to(device, non_blocking=True),
            batch["support_features"].to(device, non_blocking=True),
            batch["support_bp"].to(device, non_blocking=True),
            batch["support_mask"].to(device, non_blocking=True),
        )
        prediction = prediction_norm.float().cpu().numpy() * std + mean
        target = None
        if include_targets:
            target = batch["target"].numpy() * std + mean
        for index in range(len(batch["event_id"])):
            row: dict[str, Any] = {
                "subject_uid": batch["subject_uid"][index],
                "event_id": batch["event_id"][index],
                "source": batch["source"][index],
                "k": int(batch["k"][index]),
                "pred_sbp": float(prediction[index, 0]),
                "pred_dbp": float(prediction[index, 1]),
            }
            if target is not None:
                row["target_sbp"] = float(target[index, 0])
                row["target_dbp"] = float(target[index, 1])
            rows.append(row)
    return pd.DataFrame(rows)


def _four_k_score(predictions: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    by_k = {
        str(k): participant_macro_metrics(group)
        for k, group in predictions.groupby("k", sort=True)
    }
    if set(map(int, by_k)) != set(KS):
        raise AssertionError("prediction table does not cover all four K budgets")
    score = float(np.mean([item["mean_mae"] for item in by_k.values()]))
    return score, by_k


@torch.no_grad()
def _basis_usage(
    model: VariableKPersonalizer,
    cache: PreparedAdapterCache,
    selection_inputs: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    if model.adapter_basis_count == 0:
        return pd.DataFrame()
    support_rows = np.sort(selection_inputs["support_row"].unique().astype(np.int64))
    records: list[dict[str, Any]] = []
    model.eval()
    for k in KS:
        weights_parts: list[np.ndarray] = []
        for start in range(0, len(support_rows), batch_size):
            rows = support_rows[start : start + batch_size]
            features = torch.as_tensor(
                np.asarray(cache.support_features[rows], dtype=np.float32), device=device
            )
            support_bp = torch.as_tensor(
                np.asarray(cache.support_bp[rows], dtype=np.float32), device=device
            )
            mask = torch.zeros(
                len(rows), max(KS), dtype=torch.bool, device=device
            )
            mask[:, :k] = True
            population_support = model.population.predict_from_features(
                features.reshape(len(rows) * max(KS), -1)
            ).reshape(len(rows), max(KS), 2)
            residual = support_bp - population_support
            tokens = model.support_token(
                torch.cat([features, support_bp, residual], dim=-1)
            )
            support_weights = mask.to(tokens.dtype)
            support_weights = support_weights / support_weights.sum(
                dim=1, keepdim=True
            )
            context = torch.sum(support_weights[..., None] * tokens, dim=1)
            weights_parts.append(
                model.adapter_bank.routing_weights(context, mask).cpu().numpy()
            )
        weights = np.concatenate(weights_parts)
        top1 = weights.argmax(axis=1)
        entropy = -np.sum(
            np.where(weights > 0, weights * np.log(np.clip(weights, 1e-12, 1.0)), 0.0),
            axis=1,
        )
        active = (weights > 0).sum(axis=1)
        for basis in range(weights.shape[1]):
            records.append(
                {
                    "k": k,
                    "basis": basis,
                    "mean_weight": float(weights[:, basis].mean()),
                    "active_participant_percent": float(
                        (weights[:, basis] > 0).mean() * 100.0
                    ),
                    "top1_participant_percent": float((top1 == basis).mean() * 100.0),
                    "mean_entropy": float(entropy.mean()),
                    "mean_active_bases": float(active.mean()),
                    "n_participants": int(len(weights)),
                }
            )
    return pd.DataFrame(records)


def train_candidate(
    *,
    prepared: Path,
    population_run: Path,
    output: Path,
    basis_count: int,
    routing_mode: str,
    seed: int,
    batch_size: int = 2048,
    evaluation_batch_size: int = 8192,
    episodes_per_epoch: int = 99968,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    require_cuda: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if min(batch_size, evaluation_batch_size, episodes_per_epoch, patience) < 1:
        raise ValueError("training sizes and patience must be positive")
    cache = PreparedAdapterCache(prepared)
    population_checkpoint = population_run / "best.pt"
    if not population_checkpoint.is_file():
        raise FileNotFoundError(population_checkpoint)
    if file_sha256(population_checkpoint) != cache.record.get(
        "population_checkpoint_sha256"
    ):
        raise AssertionError("candidate population checkpoint differs from cache")

    fit = cache.labelled_role("fit")
    early = cache.labelled_role("early")
    selection_inputs = cache.input_role("selection")
    role_subjects = [
        set(frame["subject_uid"].astype(str))
        for frame in (fit, early, selection_inputs)
    ]
    if any(
        role_subjects[left] & role_subjects[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise AssertionError("candidate participant roles overlap")

    fit_dataset = CachedEpisodeDataset(fit, cache, include_target=True)
    early_dataset = CachedEpisodeDataset(early, cache, include_target=True)
    selection_dataset = CachedEpisodeDataset(
        selection_inputs, cache, include_target=False
    )
    sampler = _participant_balanced_sampler(
        fit_dataset, seed=seed, num_samples=episodes_per_epoch
    )
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("adapter-bank training requires CUDA")
    model, scaler = _build_model(
        population_checkpoint,
        device=device,
        basis_count=basis_count,
        routing_mode=routing_mode,
    )
    if scaler != cache.record["target_scaler"]:
        raise AssertionError("candidate target scaler differs from feature cache")
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.MSELoss()
    loader_kwargs = {
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        **loader_kwargs,
    )
    early_loader = DataLoader(
        early_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    mean = cache.mean
    std = cache.std
    for epoch in itertools.count(1):
        model.train()
        model.population.eval()
        total_loss = 0.0
        total_examples = 0
        for batch in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                prediction = model.forward_from_features(
                    batch["query_features"].to(device, non_blocking=True),
                    batch["support_features"].to(device, non_blocking=True),
                    batch["support_bp"].to(device, non_blocking=True),
                    batch["support_mask"].to(device, non_blocking=True),
                )
                target = batch["target"].to(device, non_blocking=True)
                loss = loss_function(prediction.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total_examples += len(target)
        early_predictions = _predict(
            model,
            early_loader,
            device=device,
            mean=mean,
            std=std,
            include_targets=True,
        )
        score, by_k = _four_k_score(early_predictions)
        record = {
            "epoch": epoch,
            "train_mse": total_loss / max(total_examples, 1),
            "fold3_four_k_participant_macro_mean_mae": score,
            "fold3_by_k": by_k,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("adapter-bank training produced no checkpoint")
    model.load_state_dict(best_state)

    # Predict the quarantined fold-4 inputs before opening its target table.
    selection_without_targets = _predict(
        model,
        selection_loader,
        device=device,
        mean=mean,
        std=std,
        include_targets=False,
    )
    if set(TARGET_COLUMNS) & set(selection_without_targets):
        raise AssertionError("selection predictions unexpectedly contain targets")
    selection_targets = cache.targets("selection")
    predictions = selection_without_targets.merge(
        selection_targets,
        on=["subject_uid", "event_id"],
        how="left",
        validate="many_to_one",
    )
    if predictions[TARGET_COLUMNS].isna().any().any():
        raise AssertionError("selection target join is incomplete")
    selection_score, selection_by_k = _four_k_score(predictions)
    metrics = {
        scope: {
            str(k): participant_macro_metrics(k_group)
            for k, k_group in scope_group.groupby("k", sort=True)
        }
        for scope, scope_group in [("Overall", predictions)]
        + [
            (source, predictions.loc[predictions["source"].eq(source)])
            for source in sorted(predictions["source"].unique())
        ]
    }
    usage = _basis_usage(
        model,
        cache,
        selection_inputs,
        device=device,
        batch_size=evaluation_batch_size,
    )

    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "selection_predictions.parquet", index=False)
    if not usage.empty:
        usage.to_csv(output / "basis_usage.csv", index=False)
    torch.save(
        {
            "screen_id": SCREEN_ID,
            "basis_count": basis_count,
            "routing_mode": routing_mode,
            "rank": 4,
            "model_state": best_state,
            "target_scaler": cache.record["target_scaler"],
            "best_epoch": best_epoch,
        },
        output / "best.pt",
    )
    save_json(output / "history.json", history)
    setting = (
        "m0_reference"
        if basis_count == 0
        else f"bank{basis_count:02d}_{routing_mode}"
    )
    payload: dict[str, Any] = {
        "status": "complete",
        "screen_id": SCREEN_ID,
        "stage": "candidate",
        "setting": setting,
        "seed": seed,
        "basis_count": basis_count,
        "routing_mode": routing_mode,
        "rank": 4 if basis_count else 0,
        "top_k": 5 if routing_mode == "top5" else None,
        "split": "meta_train_internal_fold4",
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_FOLD,
        "selection_fold": SELECTION_FOLD,
        "support_policy": "fixed_first",
        "ks": list(KS),
        "loss": "mse",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stop_reason": f"early_stopping_patience_{patience}_no_epoch_cap",
        "fold3_best_four_k_mean_mae": best_score,
        "fold4_four_k_mean_mae": selection_score,
        "fold4_by_k": selection_by_k,
        "metrics": metrics,
        "train_participants": int(fit["subject_uid"].nunique()),
        "early_participants": int(early["subject_uid"].nunique()),
        "selection_participants": int(selection_inputs["subject_uid"].nunique()),
        "selection_queries_per_k": int(len(selection_inputs)),
        "episodes_per_epoch": episodes_per_epoch,
        "batch_size": batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "parameter_counts": model_parameter_counts(model),
        "population_run_json_sha256": file_sha256(population_run / "run.json"),
        "population_checkpoint_sha256": file_sha256(population_checkpoint),
        "cache_run_sha256": file_sha256(prepared / "run.json"),
        "cache_artifact_sha256": cache.record["artifact_sha256"],
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "selection_predictions_sha256": file_sha256(
            output / "selection_predictions.parquet"
        ),
        "checkpoint_sha256": file_sha256(output / "best.pt"),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "meta_validation_used_for_training": False,
        "meta_validation_used_for_early_stopping": False,
        "meta_validation_used_for_candidate_ranking": False,
        "meta_validation_predictions_generated": False,
        "locked_test_accessed": False,
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "participant_identity_model_input": False,
        "source_model_input": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output / "run.json", payload)
    return payload


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.6f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join(rows)


def build_report(
    *,
    runs: dict[str, Path],
    output: Path,
    expected_seed: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    expected_settings = {"m0_reference"} | {
        f"bank{count:02d}_{mode}"
        for count in BASIS_COUNTS
        for mode in ROUTING_MODES
    }
    if set(runs) != expected_settings:
        raise AssertionError(
            f"report settings differ; missing={sorted(expected_settings-set(runs))}, "
            f"extra={sorted(set(runs)-expected_settings)}"
        )
    frames: list[pd.DataFrame] = []
    run_records: dict[str, dict[str, Any]] = {}
    canonical: pd.DataFrame | None = None
    common_cache_sha: str | None = None
    usage_frames: list[pd.DataFrame] = []
    safety_keys = (
        "meta_validation_used_for_training",
        "meta_validation_used_for_early_stopping",
        "meta_validation_used_for_candidate_ranking",
        "meta_validation_predictions_generated",
        "locked_test_accessed",
        "query_bp_model_input",
        "future_query_model_input",
        "participant_identity_model_input",
        "source_model_input",
    )
    for setting, root in runs.items():
        record = _read_json(root / "run.json")
        if (
            record.get("status") != "complete"
            or record.get("screen_id") != SCREEN_ID
            or record.get("stage") != "candidate"
            or record.get("setting") != setting
            or int(record.get("seed", -1)) != expected_seed
        ):
            raise AssertionError(f"{setting} has an invalid run record")
        if record.get("split") != "meta_train_internal_fold4":
            raise AssertionError(f"{setting} has the wrong evaluation split")
        _require_false(record, safety_keys, setting)
        cache_sha = str(record.get("cache_run_sha256"))
        if common_cache_sha is None:
            common_cache_sha = cache_sha
        elif cache_sha != common_cache_sha:
            raise AssertionError("candidates used different feature caches")
        predictions = pd.read_parquet(root / "selection_predictions.parquet")
        required = set(QUERY_KEYS + TARGET_COLUMNS + ["source", "pred_sbp", "pred_dbp"])
        if missing := required - set(predictions):
            raise ValueError(f"{setting} predictions missing {sorted(missing)}")
        if predictions.duplicated(QUERY_KEYS).any():
            raise AssertionError(f"{setting} contains duplicate query/K keys")
        checked = predictions[QUERY_KEYS + TARGET_COLUMNS].sort_values(
            QUERY_KEYS, kind="mergesort"
        ).reset_index(drop=True)
        if canonical is None:
            canonical = checked
        elif not canonical.equals(checked):
            raise AssertionError(f"{setting} uses different queries or targets")
        frames.append(predictions.assign(Setting=setting))
        usage_path = root / "basis_usage.csv"
        if usage_path.is_file():
            usage_frames.append(pd.read_csv(usage_path).assign(Setting=setting))
        run_records[setting] = record

    full = pd.concat(frames, ignore_index=True, sort=False)
    by_k_rows: list[dict[str, Any]] = []
    average_rows: list[dict[str, Any]] = []
    for setting in runs:
        setting_frame = full.loc[full["Setting"].eq(setting)]
        for scope in SCOPES:
            scope_frame = (
                setting_frame
                if scope == "Overall"
                else setting_frame.loc[setting_frame["source"].eq(scope)]
            )
            k_metrics = []
            for k in KS:
                group = scope_frame.loc[scope_frame["k"].eq(k)]
                metric = participant_macro_metrics(group)
                k_metrics.append(metric)
                by_k_rows.append(
                    {
                        "Setting": setting,
                        "Scope": scope,
                        "K": k,
                        "N participants": metric["n_participants"],
                        "N queries": metric["n_events"],
                        "SBP participant-macro MAE": metric["sbp_mae"],
                        "DBP participant-macro MAE": metric["dbp_mae"],
                        "Mean participant-macro MAE": metric["mean_mae"],
                    }
                )
            average_rows.append(
                {
                    "Setting": setting,
                    "Scope": scope,
                    "SBP four-K average participant-macro MAE": float(
                        np.mean([item["sbp_mae"] for item in k_metrics])
                    ),
                    "DBP four-K average participant-macro MAE": float(
                        np.mean([item["dbp_mae"] for item in k_metrics])
                    ),
                    "Mean four-K average participant-macro MAE": float(
                        np.mean([item["mean_mae"] for item in k_metrics])
                    ),
                }
            )
    by_k = pd.DataFrame(by_k_rows)
    averages = pd.DataFrame(average_rows)
    for setting, group in by_k.groupby("Setting"):
        for k, k_group in group.groupby("K"):
            overall = k_group.loc[k_group["Scope"].eq("Overall")].iloc[0]
            sources = k_group.loc[k_group["Scope"].isin(["MIMIC", "VitalDB"])]
            if int(overall["N participants"]) != int(sources["N participants"].sum()):
                raise AssertionError(f"{setting} K={k} source participants do not sum")
            if int(overall["N queries"]) != int(sources["N queries"].sum()):
                raise AssertionError(f"{setting} K={k} source queries do not sum")

    reference_average = averages.loc[
        averages["Setting"].eq("m0_reference")
    ].set_index("Scope")
    reference_by_k = by_k.loc[
        by_k["Setting"].eq("m0_reference")
        & by_k["Scope"].eq("Overall")
    ].set_index("K")
    gate_rows: list[dict[str, Any]] = []
    for setting in sorted(expected_settings - {"m0_reference"}):
        candidate_average = averages.loc[averages["Setting"].eq(setting)].set_index(
            "Scope"
        )
        candidate_by_k = by_k.loc[
            by_k["Setting"].eq(setting) & by_k["Scope"].eq("Overall")
        ].set_index("K")
        gains = {
            scope: float(
                reference_average.loc[
                    scope, "Mean four-K average participant-macro MAE"
                ]
                - candidate_average.loc[
                    scope, "Mean four-K average participant-macro MAE"
                ]
            )
            for scope in SCOPES
        }
        k_gains = {
            k: float(
                reference_by_k.loc[k, "Mean participant-macro MAE"]
                - candidate_by_k.loc[k, "Mean participant-macro MAE"]
            )
            for k in KS
        }
        endpoint_worsening = {
            bp: float(
                candidate_average.loc[
                    "Overall", f"{bp} four-K average participant-macro MAE"
                ]
                - reference_average.loc[
                    "Overall", f"{bp} four-K average participant-macro MAE"
                ]
            )
            for bp in ("SBP", "DBP")
        }
        passes = bool(
            gains["Overall"] >= 0.15
            and gains["MIMIC"] > 0.0
            and gains["VitalDB"] > 0.0
            and sum(value > 0.0 for value in k_gains.values()) >= 3
            and min(k_gains.values()) >= -0.10
            and max(endpoint_worsening.values()) <= 0.05
        )
        gate_rows.append(
            {
                "Setting": setting,
                "Overall gain": gains["Overall"],
                "MIMIC gain": gains["MIMIC"],
                "VitalDB gain": gains["VitalDB"],
                **{f"K={k} gain": k_gains[k] for k in KS},
                "Improved K count": sum(value > 0.0 for value in k_gains.values()),
                "Worst K gain": min(k_gains.values()),
                "Overall SBP worsening": endpoint_worsening["SBP"],
                "Overall DBP worsening": endpoint_worsening["DBP"],
                "Pass": passes,
            }
        )
    gates = pd.DataFrame(gate_rows)
    overall_order = averages.loc[averages["Scope"].eq("Overall")].sort_values(
        ["Mean four-K average participant-macro MAE", "Setting"],
        kind="mergesort",
    )
    numerical_winner = str(overall_order.iloc[0]["Setting"])
    passers = set(gates.loc[gates["Pass"], "Setting"].astype(str))
    promoted = None
    if passers:
        promoted = str(
            overall_order.loc[overall_order["Setting"].isin(passers)].iloc[0][
                "Setting"
            ]
        )

    diagnostic_input = full.copy()
    diagnostic_input["DiagnosticSetting"] = diagnostic_input.apply(
        lambda row: f"{row.Setting} K={int(row.k)}", axis=1
    )
    diagnostic_input.drop(columns="Setting", inplace=True)
    diagnostic_input.rename(columns={"DiagnosticSetting": "Setting"}, inplace=True)
    diagnostic_settings = [
        f"{setting} K={k}" for setting in runs for k in KS
    ]
    diagnostics = _diagnostic_rows(diagnostic_input, diagnostic_settings)

    dense5 = full.loc[full["Setting"].eq("bank05_dense")].sort_values(QUERY_KEYS)
    top5 = full.loc[full["Setting"].eq("bank05_top5")].sort_values(QUERY_KEYS)
    max_difference = float(
        np.abs(
            dense5[["pred_sbp", "pred_dbp"]].to_numpy(float)
            - top5[["pred_sbp", "pred_dbp"]].to_numpy(float)
        ).max()
    )
    equivalence = {
        "mathematically_equivalent": True,
        "max_absolute_prediction_difference_mmHg": max_difference,
        "numerically_identical_within_1e-5": max_difference <= 1e-5,
        "interpretation": (
            "M=5 Top-5 and dense use all five bases; the duplicated jobs are an "
            "implementation/hardware consistency control, not distinct candidates."
        ),
    }

    output.mkdir(parents=True, exist_ok=False)
    by_k.to_csv(output / "participant_macro_by_k.csv", index=False)
    averages.to_csv(output / "four_k_average.csv", index=False)
    gates.to_csv(output / "candidate_gate.csv", index=False)
    diagnostics.to_csv(output / "pooled_diagnostics.csv", index=False)
    if usage_frames:
        pd.concat(usage_frames, ignore_index=True).to_csv(
            output / "basis_usage.csv", index=False
        )
    summary: dict[str, Any] = {
        "status": "complete",
        "screen_id": SCREEN_ID,
        "stage": "internal_report",
        "seed": expected_seed,
        "split": "meta_train_internal_fold4",
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_FOLD,
        "selection_fold": SELECTION_FOLD,
        "reference": "m0_reference",
        "candidate_job_count": 12,
        "unique_candidate_configuration_count": 11,
        "numerical_winner": numerical_winner,
        "promoted_candidate": promoted,
        "passing_candidates": sorted(passers),
        "passes_internal_gate": promoted is not None,
        "m5_equivalence_control": equivalence,
        "cache_run_sha256": common_cache_sha,
        "run_json_sha256": {
            setting: file_sha256(root / "run.json") for setting, root in runs.items()
        },
        "meta_validation_used_for_candidate_ranking": False,
        "locked_test_accessed": False,
        "aami_bhs_interpretation": (
            "retrospective numerical screens only; no compliance claim"
        ),
    }
    save_json(output / "selection.json", summary)
    lines = [
        "# Few-shot support-conditioned adapter-bank internal screen",
        "",
        "This prospective screen compares six shared-basis counts and two routing modes. Folds 0--2 fit the model, fold 3 controls patience-8 early stopping with no epoch cap, and fold 4 is scored only after the checkpoint is frozen. Meta-validation and the locked meta-test were not accessed.",
        "",
        "## Four-K participant-macro primary result",
        "",
        _markdown_table(overall_order),
        "",
        "## Prespecified promotion gate",
        "",
        _markdown_table(gates),
        "",
        "## Per-K and source-stratified participant-macro result",
        "",
        _markdown_table(by_k),
        "",
        "## Event-pooled diagnostics",
        "",
        "AAMI/BHS entries are retrospective numerical screens only and do not establish standards or device compliance.",
        "",
        _markdown_table(diagnostics),
        "",
        f"Numerical winner: **{numerical_winner}**.",
        f"Candidate passing the frozen development gate: **{promoted}**.",
        f"M=5 equivalence-control maximum prediction difference: **{max_difference:.8f} mmHg**.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def smoke_test(*, output: Path, seed: int, require_cuda: bool) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("adapter-bank smoke test requires CUDA")
    rows: list[dict[str, Any]] = []
    query = torch.randn(16, 256, device=device)
    support = torch.randn(16, 5, 256, device=device)
    support_bp = torch.randn(16, 5, 2, device=device)
    mask = torch.tensor(
        [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        * 4,
        dtype=torch.bool,
        device=device,
    )
    for basis_count in BASIS_COUNTS:
        for routing_mode in ROUTING_MODES:
            seed_everything(seed)
            model = VariableKPersonalizer(
                use_film=False,
                query_conditioned_weights=False,
                adapter_basis_count=basis_count,
                adapter_rank=4,
                adapter_top_k=5 if routing_mode == "top5" else None,
            ).to(device)
            for parameter in model.population.parameters():
                parameter.requires_grad = False
            optimizer = torch.optim.AdamW(
                [value for value in model.parameters() if value.requires_grad], lr=1e-3
            )
            router_grad = 0.0
            # The zero-initialized M0 correction head learns first, the
            # zero-initialized adapter output factors learn on the next step,
            # and only then can the support router receive a nonzero gradient.
            # Four steps make that intended delayed-gradient path explicit.
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_from_features(query, support, support_bp, mask)
                loss = prediction.square().mean()
                loss.backward()
                optimizer.step()
                gradient = model.adapter_bank.router[-1].weight.grad
                if gradient is not None:
                    router_grad = max(router_grad, float(gradient.abs().sum()))
            if not torch.isfinite(prediction).all():
                raise AssertionError("adapter smoke prediction is non-finite")
            weights = model.adapter_bank.routing_weights(
                torch.randn(16, 256, device=device), mask
            )
            expected_active = 5 if routing_mode == "top5" else basis_count
            observed_active = int((weights > 0).sum(dim=1).min().item())
            if observed_active != expected_active:
                raise AssertionError("adapter smoke active-basis count differs")
            if router_grad <= 0:
                raise AssertionError("adapter router received no gradient after four steps")
            rows.append(
                {
                    "basis_count": basis_count,
                    "routing_mode": routing_mode,
                    "active_bases": observed_active,
                    "router_gradient_l1": router_grad,
                    "finite": True,
                }
            )
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / "smoke.csv", index=False)
    record = {
        "status": "complete",
        "screen_id": SCREEN_ID,
        "stage": "cuda_smoke",
        "seed": seed,
        "candidate_count": 12,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "locked_test_accessed": False,
        "meta_validation_accessed": False,
    }
    save_json(output / "run.json", record)
    return record


def _parse_runs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--run must use SETTING=PATH")
        setting, raw_path = value.split("=", 1)
        if not setting or setting in result:
            raise ValueError(f"invalid or duplicate setting {setting}")
        result[setting] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    cache = commands.add_parser("prepare-cache")
    cache.add_argument("--store-root", type=Path, required=True)
    cache.add_argument("--folds", type=Path, required=True)
    cache.add_argument("--population-run", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--batch-size", type=int, default=1024)
    cache.add_argument("--require-cuda", action="store_true")

    train = commands.add_parser("train")
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--population-run", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--basis-count", type=int, required=True)
    train.add_argument("--routing-mode", choices=["none", *ROUTING_MODES], required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--batch-size", type=int, default=2048)
    train.add_argument("--evaluation-batch-size", type=int, default=8192)
    train.add_argument("--episodes-per-epoch", type=int, default=99968)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--require-cuda", action="store_true")

    report = commands.add_parser("report")
    report.add_argument("--run", action="append", required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--expected-seed", type=int, required=True)

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--seed", type=int, required=True)
    smoke.add_argument("--require-cuda", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare-cache":
        result = prepare_cache(
            store_root=args.store_root,
            folds_path=args.folds,
            population_run=args.population_run,
            output=args.output,
            batch_size=args.batch_size,
            require_cuda=args.require_cuda,
        )
    elif args.command == "train":
        result = train_candidate(
            prepared=args.prepared,
            population_run=args.population_run,
            output=args.output,
            basis_count=args.basis_count,
            routing_mode=args.routing_mode,
            seed=args.seed,
            batch_size=args.batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            episodes_per_epoch=args.episodes_per_epoch,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            require_cuda=args.require_cuda,
        )
    elif args.command == "report":
        result = build_report(
            runs=_parse_runs(args.run),
            output=args.output,
            expected_seed=args.expected_seed,
        )
    else:
        result = smoke_test(
            output=args.output,
            seed=args.seed,
            require_cuda=args.require_cuda,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
