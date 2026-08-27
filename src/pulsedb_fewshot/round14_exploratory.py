"""Round-14 leakage-safe exploratory calibration-relative method screen.

The module deliberately leaves the historical Round-8 and Round-9 scientific
definitions unchanged.  It rebuilds a cache from an ``inception_time_wide``
population/QGH pair, then trains three complete causal calibration-relative
methods on meta-train folds 0--2, uses fold 3 only for patience-8 early
stopping, and emits predictions only for fold 4.  C2 adds bounded physical-unit
error surrogates to the complete C1 method.  C3 contains complete C2 and adds a
training-only GroupDRO-inspired smooth worst-group penalty over source x
query-SBP relation to the fixed first-five support range.  It is not canonical
GroupDRO because it does not maintain persistent adversarial group weights.

Source and query BP may define supervised training targets/groups, but neither
is a model input.  Meta-validation and the locked test are rejected.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .models import VariableKPersonalizer, model_parameter_counts
from .phase6e_residual import KEYS
from .report_round8 import _diagnostic_rows
from .round8_calibration_relative import (
    PHYSIOLOGY_NAMES,
    SUPPORT_K,
    PreparedRound8,
    _encode_rows,
    _physiology_features,
)
from .round9_refinement import (
    Candidate,
    Round9Model,
    _forward_sequences,
    _loss as _calibration_relative_loss,
    _make_arrays,
    _markdown_table,
    _predict,
    _scored_predictions,
    _sequence_batches,
)
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


METHODS = (
    "calibration_relative",
    "calibration_relative_standards",
    "calibration_relative_groupdro",
)
BACKBONE = "inception_time_wide"
FIT_FOLDS = (0, 1, 2)
EARLY_STOPPING_FOLD = 3
SELECTION_FOLD = 4
EXPECTED_FOLDS = set(range(5))
SOURCE_INDEX = {"MIMIC": 0, "VitalDB": 1}
ANCHOR_SETTING = "inception_time_wide_qgh"
EXPLORATORY_SEED = 20260828

STANDARDS_HUBER_DELTA_MMHG = 5.0
STANDARDS_HUBER_WEIGHT = 0.10
STANDARDS_THRESHOLD_WEIGHT = 0.25
STANDARDS_THRESHOLD_TEMPERATURE_MMHG = 1.0
STANDARDS_THRESHOLDS_MMHG = (5.0, 10.0, 15.0)
STANDARDS_THRESHOLD_WEIGHTS = (0.50, 0.30, 0.20)

GROUPDRO_TEMPERATURE = 0.25
GROUPDRO_WEIGHT = 0.25


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _checkpoint_path(run: Path, record: dict[str, Any]) -> Path:
    candidate = run / "best.pt"
    if not candidate.is_file():
        configured = record.get("checkpoint")
        if configured:
            configured_path = Path(str(configured))
            if configured_path.is_file():
                candidate = configured_path
    if not candidate.is_file():
        raise FileNotFoundError(f"missing checkpoint for {run}")
    return candidate


def _require_false(record: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        if record.get(key) is not False:
            raise AssertionError(f"{label} has unsafe flag {key}")


def _validate_training_run(
    run: Path,
    *,
    expected_method: str,
    expected_seed: int | None = None,
) -> tuple[dict[str, Any], Path]:
    record = _read_json(run / "run.json")
    if record.get("status") != "complete":
        raise AssertionError(f"{run} is incomplete")
    if record.get("method") != expected_method:
        raise AssertionError(f"{run} has the wrong training method")
    if record.get("backbone") != BACKBONE:
        raise AssertionError(f"{run} is not an {BACKBONE} run")
    if expected_seed is not None and int(record.get("seed", -1)) != expected_seed:
        raise AssertionError(f"{run} has the wrong seed")
    if record.get("crossfit_fit_folds") != list(FIT_FOLDS):
        raise AssertionError(f"{run} has the wrong fit folds")
    if record.get("crossfit_validation_fold") != EARLY_STOPPING_FOLD:
        raise AssertionError(f"{run} has the wrong early-stopping fold")
    if record.get("crossfit_excluded_folds") != [SELECTION_FOLD]:
        raise AssertionError(f"{run} did not quarantine fold 4 during fitting")
    if record.get("locked_test_accessed") is not False:
        raise AssertionError(f"{run} accessed the locked test")
    checkpoint = _checkpoint_path(run, record)
    actual_checkpoint_hash = file_sha256(checkpoint)
    recorded_checkpoint_hash = record.get("checkpoint_sha256")
    if recorded_checkpoint_hash and recorded_checkpoint_hash != actual_checkpoint_hash:
        raise AssertionError(f"{run} checkpoint SHA-256 mismatch")
    return record, checkpoint


def validate_base_run_pair(
    population_record: dict[str, Any],
    qgh_record: dict[str, Any],
    *,
    expected_seed: int,
    store_manifest_sha256: str,
    folds_sha256: str,
) -> None:
    """Validate the immutable inputs used to create a Round-14 cache."""

    for label, record, method in (
        ("population", population_record, "population"),
        ("qgh", qgh_record, "m0"),
    ):
        if record.get("status") != "complete" or record.get("method") != method:
            raise AssertionError(f"{label} run is incomplete or has the wrong method")
        if record.get("backbone") != BACKBONE:
            raise AssertionError(f"{label} run has the wrong backbone")
        if int(record.get("seed", -1)) != expected_seed:
            raise AssertionError(f"{label} run has the wrong seed")
        if record.get("crossfit_fit_folds") != list(FIT_FOLDS):
            raise AssertionError(f"{label} run has the wrong fit folds")
        if record.get("crossfit_validation_fold") != EARLY_STOPPING_FOLD:
            raise AssertionError(f"{label} run has the wrong early-stopping fold")
        if record.get("crossfit_excluded_folds") != [SELECTION_FOLD]:
            raise AssertionError(f"{label} run did not exclude fold 4")
        if record.get("locked_test_accessed") is not False:
            raise AssertionError(f"{label} run accessed the locked test")
        if record.get("store_manifest_sha256") != store_manifest_sha256:
            raise AssertionError(f"{label} run store manifest differs from the cache input")
        if record.get("crossfit_folds_sha256") != folds_sha256:
            raise AssertionError(f"{label} run fold assignment differs from the cache input")
    if population_record.get("source_tree_sha256") != qgh_record.get(
        "source_tree_sha256"
    ):
        raise AssertionError("population and QGH were trained from different source trees")
    arguments = qgh_record.get("arguments")
    if not isinstance(arguments, dict):
        raise AssertionError("QGH run lacks an argument audit")
    expected_arguments = {
        "loss": "huber",
        "huber_delta": 0.5,
        "use_quality_gate": True,
        "train_support_policy": "fixed_first",
        "ks": [SUPPORT_K],
    }
    for key, expected in expected_arguments.items():
        if arguments.get(key) != expected:
            raise AssertionError(f"QGH run has unexpected {key}")


def _assert_checkpoint_pair(
    population_checkpoint: Path,
    qgh_checkpoint: Path,
) -> None:
    population = torch.load(
        population_checkpoint, map_location="cpu", weights_only=False
    )
    qgh = torch.load(qgh_checkpoint, map_location="cpu", weights_only=False)
    if population.get("backbone") != BACKBONE or qgh.get("backbone") != BACKBONE:
        raise AssertionError("checkpoint backbone mismatch")
    if population.get("target_scaler") != qgh.get("target_scaler"):
        raise AssertionError("population and QGH target scalers differ")
    population_state = population.get("model_state")
    qgh_state = qgh.get("model_state")
    if not isinstance(population_state, dict) or not isinstance(qgh_state, dict):
        raise AssertionError("checkpoint lacks a model state")
    for name, tensor in population_state.items():
        qgh_name = f"population.{name}"
        if qgh_name not in qgh_state or not torch.equal(tensor, qgh_state[qgh_name]):
            raise AssertionError("QGH does not contain the supplied population checkpoint")


def _load_qgh_model(
    population: nn.Module,
    checkpoint: Path,
    device: torch.device,
) -> VariableKPersonalizer:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("method") != "m0" or payload.get("backbone") != BACKBONE:
        raise AssertionError("QGH checkpoint metadata mismatch")
    model = VariableKPersonalizer(
        population,
        use_film=False,
        query_conditioned_weights=False,
        anchor_mode="mean",
        use_quality_gate=True,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _validate_folds(folds: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_uid", "source", "fold"}
    if missing := required - set(folds.columns):
        raise ValueError(f"fold table missing {sorted(missing)}")
    result = folds.copy()
    result["subject_uid"] = result["subject_uid"].astype(str)
    result["fold"] = pd.to_numeric(result["fold"]).astype(int)
    if result["subject_uid"].duplicated().any():
        raise AssertionError("fold table contains duplicate participants")
    if set(result["fold"].unique()) != EXPECTED_FOLDS:
        raise AssertionError("Round-14 requires folds 0--4")
    if set(result["source"].astype(str)) != set(SOURCE_INDEX):
        raise AssertionError("Round-14 requires both PulseDB source strata")
    if "split" in result and result["split"].astype(str).ne("meta_train").any():
        raise AssertionError("fold table includes a non-meta-train participant")
    return result


def _artifact_hashes(root: Path) -> dict[str, str]:
    names = (
        "queries.parquet",
        "support_index.parquet",
        "query_embeddings.npy",
        "query_physiology.npy",
        "support_embeddings.npy",
        "support_physiology.npy",
        "support_population_bp.npy",
        "support_bp.npy",
    )
    return {name: file_sha256(root / name) for name in names}


def prepare_cache(
    *,
    store_root: Path,
    folds_path: Path,
    population_run: Path,
    qgh_run: Path,
    output: Path,
    seed: int,
    batch_size: int = 256,
    require_cuda: bool = False,
) -> dict[str, Any]:
    """Rebuild a meta-train-only cache from a matched wide-Inception/QGH pair."""

    if seed != EXPLORATORY_SEED:
        raise AssertionError(
            f"Round-14 exploratory cache requires seed {EXPLORATORY_SEED}"
        )
    if output.exists():
        raise FileExistsError(output)
    store_manifest = store_root / "materialization.json"
    if not store_manifest.is_file():
        raise FileNotFoundError(store_manifest)
    store_sha = file_sha256(store_manifest)
    folds_sha = file_sha256(folds_path)
    population_record, population_checkpoint = _validate_training_run(
        population_run, expected_method="population", expected_seed=seed
    )
    qgh_record, qgh_checkpoint = _validate_training_run(
        qgh_run, expected_method="m0", expected_seed=seed
    )
    validate_base_run_pair(
        population_record,
        qgh_record,
        expected_seed=seed,
        store_manifest_sha256=store_sha,
        folds_sha256=folds_sha,
    )
    _assert_checkpoint_pair(population_checkpoint, qgh_checkpoint)

    folds = _validate_folds(pd.read_parquet(folds_path))
    metadata = load_store_metadata(store_root, "development").copy()
    metadata["subject_uid"] = metadata["subject_uid"].astype(str)
    if metadata["split"].eq("meta_test").any():
        raise AssertionError("development metadata unexpectedly contains locked-test rows")
    selected = metadata.loc[
        metadata["split"].eq("meta_train")
        & metadata["subject_uid"].isin(set(folds["subject_uid"]))
    ].copy()
    if selected.empty or selected["split"].ne("meta_train").any():
        raise AssertionError("Round-14 cache must contain meta-train only")
    selected = selected.merge(
        folds[["subject_uid", "source", "fold"]].rename(
            columns={"source": "fold_source"}
        ),
        on="subject_uid",
        validate="many_to_one",
    )
    if not selected["source"].astype(str).equals(
        selected["fold_source"].astype(str)
    ):
        raise AssertionError("store and fold source labels disagree")
    selected.drop(columns="fold_source", inplace=True)

    query = selected.loc[
        selected["common_query"].astype(bool)
        & selected["event_index"].gt(SUPPORT_K)
    ].copy()
    query.sort_values(
        ["subject_uid", "event_index", "event_id"],
        kind="mergesort",
        inplace=True,
    )
    query.reset_index(drop=True, inplace=True)
    participants = set(query["subject_uid"])
    if not participants:
        raise AssertionError("Round-14 cache has no eligible queries")
    if participants != set(folds["subject_uid"]):
        raise AssertionError("fold table and eligible query participants differ")

    support = selected.loc[
        selected["subject_uid"].isin(participants)
        & selected["event_index"].le(SUPPORT_K)
    ].copy()
    support.sort_values(
        ["subject_uid", "event_index", "event_id"],
        kind="mergesort",
        inplace=True,
    )
    support.reset_index(drop=True, inplace=True)
    counts = support.groupby("subject_uid", sort=False).size()
    if not counts.eq(SUPPORT_K).all() or set(counts.index.astype(str)) != participants:
        raise AssertionError("every Round-14 participant requires five fixed supports")
    support["support_position"] = support.groupby("subject_uid").cumcount()
    support_subjects = sorted(participants)
    support_row_map = {subject: index for index, subject in enumerate(support_subjects)}
    query["support_row"] = query["subject_uid"].map(support_row_map).astype(np.int64)
    query["query_embedding_row"] = np.arange(len(query), dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Round-14 cache generation requires CUDA")
    population, scaler = _load_population_checkpoint(population_checkpoint, device)
    population.to(device).eval()
    qgh = _load_qgh_model(population, qgh_checkpoint, device)
    target_mean = np.asarray(scaler["mean"], dtype=np.float32)
    target_std = np.asarray(scaler["std"], dtype=np.float32)
    accessor = WaveformAccessor(store_root)

    support_embeddings, support_physiology_flat, support_population_flat = _encode_rows(
        support,
        accessor=accessor,
        population=population,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        batch_size=batch_size,
        include_embeddings=True,
    )
    assert support_embeddings is not None and support_population_flat is not None
    n_participants = len(support_subjects)
    support_embeddings = support_embeddings.reshape(
        n_participants, SUPPORT_K, -1
    )
    support_physiology = support_physiology_flat.reshape(
        n_participants, SUPPORT_K, -1
    )
    support_population = support_population_flat.reshape(
        n_participants, SUPPORT_K, 2
    )
    support_bp = support[["sbp", "dbp"]].to_numpy(np.float32).reshape(
        n_participants, SUPPORT_K, 2
    )

    query_embeddings: list[np.ndarray] = []
    query_physiology: list[np.ndarray] = []
    qgh_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(query), batch_size):
            part = query.iloc[start : start + batch_size]
            waveform = torch.stack(
                [
                    accessor.get(str(row.waveform_file), int(row.waveform_row))
                    for row in part.itertuples(index=False)
                ]
            ).to(device)
            query_feature = population.encoder(waveform)
            query_embeddings.append(query_feature.cpu().numpy().astype(np.float32))
            query_physiology.append(
                _physiology_features(waveform).cpu().numpy().astype(np.float32)
            )
            support_rows = part["support_row"].to_numpy(np.int64)
            support_feature = torch.as_tensor(
                support_embeddings[support_rows], device=device
            )
            support_bp_norm = torch.as_tensor(
                (support_bp[support_rows] - target_mean) / target_std,
                device=device,
            )
            support_mask = torch.ones(
                len(part), SUPPORT_K, dtype=torch.bool, device=device
            )
            prediction_norm = qgh.forward_from_features(
                query_feature,
                support_feature,
                support_bp_norm,
                support_mask,
            )
            qgh_predictions.append(
                (prediction_norm.cpu().numpy() * target_std + target_mean).astype(
                    np.float32
                )
            )

    query_embedding_array = np.concatenate(query_embeddings)
    query_physiology_array = np.concatenate(query_physiology)
    qgh_prediction_array = np.concatenate(qgh_predictions)
    query["k"] = SUPPORT_K
    query["target_sbp"] = query["sbp"].astype(float)
    query["target_dbp"] = query["dbp"].astype(float)
    query["pred_sbp"] = qgh_prediction_array[:, 0]
    query["pred_dbp"] = qgh_prediction_array[:, 1]
    query["events_since_calibration"] = query["event_index"] - SUPPORT_K
    query["round8_role"] = "train"
    query["cache_split"] = "meta_train"

    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "query_embeddings.npy", query_embedding_array)
    np.save(output / "query_physiology.npy", query_physiology_array)
    np.save(output / "support_embeddings.npy", support_embeddings)
    np.save(output / "support_physiology.npy", support_physiology)
    np.save(output / "support_population_bp.npy", support_population)
    np.save(output / "support_bp.npy", support_bp)
    query.to_parquet(output / "queries.parquet", index=False)
    support[
        [
            "subject_uid",
            "event_id",
            "source",
            "split",
            "event_index",
            "support_position",
        ]
    ].assign(
        support_row=lambda frame: frame["subject_uid"].map(support_row_map)
    ).to_parquet(output / "support_index.parquet", index=False)

    artifacts = _artifact_hashes(output)
    record: dict[str, Any] = {
        "status": "complete",
        "round": 14,
        "stage": "exploratory_cache",
        "split": "meta_train_internal_cache",
        "seed": seed,
        "backbone": BACKBONE,
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_STOPPING_FOLD,
        "selection_fold": SELECTION_FOLD,
        "support_policy": "fixed_first",
        "k": SUPPORT_K,
        "participants": n_participants,
        "queries": int(len(query)),
        "embedding_dimension": int(query_embedding_array.shape[1]),
        "physiology_features": list(PHYSIOLOGY_NAMES),
        "population_run_json_sha256": file_sha256(population_run / "run.json"),
        "qgh_run_json_sha256": file_sha256(qgh_run / "run.json"),
        "population_checkpoint_sha256": file_sha256(population_checkpoint),
        "qgh_checkpoint_sha256": file_sha256(qgh_checkpoint),
        "folds_sha256": folds_sha,
        "store_manifest_sha256": store_sha,
        "training_source_tree_sha256": population_record["source_tree_sha256"],
        "cache_source_tree_sha256": source_tree_sha256(
            Path(__file__).resolve().parents[2]
        ),
        "artifact_sha256": artifacts,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "meta_validation_used_for_training": False,
        "meta_validation_used_for_early_stopping": False,
        "meta_validation_used_for_candidate_ranking": False,
        "meta_validation_predictions_generated": False,
        "locked_test_accessed": False,
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "source_model_input": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output / "run.json", record)
    return record


def validate_cache(root: Path) -> dict[str, Any]:
    record = _read_json(root / "run.json")
    expected = {
        "status": "complete",
        "round": 14,
        "stage": "exploratory_cache",
        "split": "meta_train_internal_cache",
        "backbone": BACKBONE,
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_STOPPING_FOLD,
        "selection_fold": SELECTION_FOLD,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"cache has unexpected {key}")
    _require_false(
        record,
        (
            "meta_validation_used_for_training",
            "meta_validation_used_for_early_stopping",
            "meta_validation_used_for_candidate_ranking",
            "meta_validation_predictions_generated",
            "locked_test_accessed",
            "query_bp_model_input",
            "future_query_model_input",
            "source_model_input",
        ),
        "cache",
    )
    recorded_artifacts = record.get("artifact_sha256")
    if not isinstance(recorded_artifacts, dict):
        raise AssertionError("cache lacks artifact hashes")
    if recorded_artifacts != _artifact_hashes(root):
        raise AssertionError("cache artifact SHA-256 mismatch")
    queries = pd.read_parquet(root / "queries.parquet")
    if queries["split"].astype(str).ne("meta_train").any():
        raise AssertionError("cache contains a non-meta-train query")
    if queries["cache_split"].astype(str).ne("meta_train").any():
        raise AssertionError("cache split marker is unsafe")
    if set(pd.to_numeric(queries["fold"]).astype(int)) != EXPECTED_FOLDS:
        raise AssertionError("cache does not contain folds 0--4")
    if queries["round8_role"].astype(str).ne("train").any():
        raise AssertionError("cache exposes validation rows to Round-14")
    if queries.duplicated(KEYS).any():
        raise AssertionError("cache has duplicate query keys")
    return record


def standards_surrogate(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return a physical-unit Huber plus smooth 5/10/15-mmHg penalty.

    This is a training surrogate only.  It cannot establish AAMI, BHS, or
    device-validation compliance.
    """

    error_mm = (prediction.float() - target.float()) * target_std
    valid = mask.bool()[..., None].expand_as(error_mm)
    if not valid.any():
        raise ValueError("standards surrogate received no valid query")
    huber = F.huber_loss(
        error_mm,
        torch.zeros_like(error_mm),
        reduction="none",
        delta=STANDARDS_HUBER_DELTA_MMHG,
    )
    # Divide by delta squared so this term is dimensionless and comparable
    # with the standardized C1 objective.
    huber_value = huber[valid].mean() / (STANDARDS_HUBER_DELTA_MMHG**2)
    absolute = error_mm.abs()
    threshold_value = torch.zeros((), dtype=error_mm.dtype, device=error_mm.device)
    for threshold, weight in zip(
        STANDARDS_THRESHOLDS_MMHG,
        STANDARDS_THRESHOLD_WEIGHTS,
        strict=True,
    ):
        # A bounded probability-like surrogate prevents extreme errors from
        # dominating the threshold term.  This is still only an optimisation
        # target, not a standards-compliance criterion.
        exceedance = torch.sigmoid(
            (absolute - threshold) / STANDARDS_THRESHOLD_TEMPERATURE_MMHG
        )
        threshold_value = threshold_value + weight * exceedance[valid].mean()
    total = (
        STANDARDS_HUBER_WEIGHT * huber_value
        + STANDARDS_THRESHOLD_WEIGHT * threshold_value
    )
    return total, {
        "physical_huber": float(huber_value.detach()),
        "threshold_surrogate": float(threshold_value.detach()),
    }


def smooth_worst_group_sbp_risk(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    groups: torch.Tensor,
    *,
    temperature: float = GROUPDRO_TEMPERATURE,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return a GroupDRO-inspired batch-local smooth worst-group penalty."""

    if temperature <= 0:
        raise ValueError("GroupDRO temperature must be positive")
    valid = mask.bool() & groups.ge(0)
    sbp = F.huber_loss(
        prediction[..., 0].float(),
        target[..., 0].float(),
        reduction="none",
        delta=0.5,
    )
    risks: list[torch.Tensor] = []
    labels: list[int] = []
    for group in range(len(SOURCE_INDEX) * 3):
        selected = valid & groups.eq(group)
        if selected.any():
            risks.append(sbp[selected].mean())
            labels.append(group)
    if not risks:
        raise ValueError("GroupDRO batch contains no valid SBP target")
    values = torch.stack(risks)
    smooth_worst = temperature * (
        torch.logsumexp(values / temperature, dim=0) - math.log(len(values))
    )
    ordinary = values.mean()
    excess = (smooth_worst - ordinary).clamp_min(0.0)
    diagnostics = {
        "groupdro_mean_sbp_risk": float(ordinary.detach()),
        "groupdro_smooth_worst_sbp_risk": float(smooth_worst.detach()),
        "groupdro_excess_sbp_risk": float(excess.detach()),
    }
    diagnostics.update(
        {
            f"group_{label}_sbp_risk": float(value.detach())
            for label, value in zip(labels, values, strict=True)
        }
    )
    return excess, diagnostics


def _attach_group_ids(
    output: dict[str, torch.Tensor],
    arrays: dict[str, np.ndarray],
    sequences: list[np.ndarray],
    device: torch.device,
) -> None:
    batch, length = output["mask"].shape
    padded = np.full((batch, length), -1, dtype=np.int64)
    for row, indexes in enumerate(sequences):
        padded[row, : len(indexes)] = arrays["group"][indexes]
    output["group"] = torch.as_tensor(padded, device=device)


def candidate_loss(
    output: dict[str, torch.Tensor],
    *,
    method: str,
    target_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if method not in METHODS:
        raise ValueError(f"unknown Round-14 method {method}")
    base = _calibration_relative_loss(output, Candidate())
    diagnostics: dict[str, float] = {"calibration_relative_loss": float(base.detach())}
    if method in {
        "calibration_relative_standards",
        "calibration_relative_groupdro",
    }:
        extra, extra_diagnostics = standards_surrogate(
            output["prediction"],
            output["target"],
            output["mask"],
            target_std,
        )
        diagnostics.update(extra_diagnostics)
        base = base + extra
        if method == "calibration_relative_standards":
            return base, diagnostics
    if method == "calibration_relative_groupdro":
        if "group" not in output:
            raise KeyError("GroupDRO output lacks training group identifiers")
        extra, extra_diagnostics = smooth_worst_group_sbp_risk(
            output["prediction"],
            output["target"],
            output["mask"],
            output["group"],
        )
        diagnostics.update(extra_diagnostics)
        return base + GROUPDRO_WEIGHT * extra, diagnostics
    return base, diagnostics


def _groupdro_preflight(
    frame: pd.DataFrame,
    groups: np.ndarray,
    *,
    minimum_participants: int = 50,
    minimum_events: int = 1000,
) -> dict[str, Any]:
    """Audit the six prespecified source x relative-support-range groups."""

    audit = frame[["subject_uid", "source"]].copy()
    audit["group"] = groups.astype(np.int64)
    rows: list[dict[str, Any]] = []
    labels = ("below_support", "within_support", "above_support")
    for source, source_index in SOURCE_INDEX.items():
        for relative_range, label in enumerate(labels):
            group = source_index * 3 + relative_range
            selected = audit.loc[audit["group"].eq(group)]
            rows.append(
                {
                    "group": group,
                    "source": source,
                    "support_range_relation": label,
                    "participants": int(selected["subject_uid"].nunique()),
                    "events": int(len(selected)),
                }
            )
    eligible = all(
        row["participants"] >= minimum_participants
        and row["events"] >= minimum_events
        for row in rows
    )
    return {
        "eligible": eligible,
        "minimum_participants": minimum_participants,
        "minimum_events": minimum_events,
        "groups": rows,
    }


def _arrays_for_frame(
    data: PreparedRound8,
    frame: pd.DataFrame,
    *,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    physiology_mean: np.ndarray,
    physiology_std: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = _make_arrays(
        data,
        frame,
        target_mean=target_mean,
        target_std=target_std,
        physiology_mean=physiology_mean,
        physiology_std=physiology_std,
    )
    sources = frame["source"].astype(str).map(SOURCE_INDEX)
    if sources.isna().any():
        raise AssertionError("Round-14 encountered an unknown source stratum")
    arrays["group"] = sources.to_numpy(np.int64) * 3 + arrays["range"][:, 0]
    return arrays


def _split_cache_queries(data: PreparedRound8) -> tuple[pd.DataFrame, ...]:
    queries = data.queries.copy()
    if queries["split"].astype(str).ne("meta_train").any():
        raise AssertionError("Round-14 training cache contains non-meta-train rows")
    fold = pd.to_numeric(queries["fold"]).astype(int)
    fit = queries.loc[fold.isin(FIT_FOLDS)].copy()
    early = queries.loc[fold.eq(EARLY_STOPPING_FOLD)].copy()
    selection = queries.loc[fold.eq(SELECTION_FOLD)].copy()
    for frame in (fit, early, selection):
        frame.sort_values(
            ["subject_uid", "event_index", "event_id"],
            kind="mergesort",
            inplace=True,
        )
        frame.reset_index(drop=True, inplace=True)
        if frame.empty:
            raise AssertionError("Round-14 internal role is empty")
    participant_sets = [set(frame["subject_uid"].astype(str)) for frame in (fit, early, selection)]
    if any(
        participant_sets[left] & participant_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise AssertionError("Round-14 internal participant roles overlap")
    return fit, early, selection


def train_candidate(
    *,
    prepared: Path,
    output: Path,
    method: str,
    seed: int,
    batch_size: int = 12,
    require_cuda: bool = False,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown Round-14 method {method}")
    if seed != EXPLORATORY_SEED:
        raise AssertionError(
            f"Round-14 exploratory training requires seed {EXPLORATORY_SEED}"
        )
    if output.exists():
        raise FileExistsError(output)
    cache_record = validate_cache(prepared)
    if int(cache_record.get("seed", -1)) != seed:
        raise AssertionError("candidate seed must match the regenerated cache seed")
    data = PreparedRound8(prepared)
    fit, early, selection = _split_cache_queries(data)

    target_mean = fit[["target_sbp", "target_dbp"]].to_numpy(np.float32).mean(0)
    target_std = fit[["target_sbp", "target_dbp"]].to_numpy(np.float32).std(0)
    target_std[target_std <= 1e-6] = 1.0
    fit_qrows = fit["query_embedding_row"].to_numpy(np.int64)
    fit_srows = fit["support_row"].to_numpy(np.int64)
    physiology_pool = np.concatenate(
        [
            np.asarray(data.query_physiology[fit_qrows]),
            np.asarray(data.support_physiology[np.unique(fit_srows)]).reshape(
                -1, data.support_physiology.shape[-1]
            ),
        ]
    ).astype(np.float32)
    physiology_mean = physiology_pool.mean(0)
    physiology_std = physiology_pool.std(0)
    physiology_std[physiology_std <= 1e-6] = 1.0
    arrays_fit = _arrays_for_frame(
        data,
        fit,
        target_mean=target_mean,
        target_std=target_std,
        physiology_mean=physiology_mean,
        physiology_std=physiology_std,
    )
    arrays_early = _arrays_for_frame(
        data,
        early,
        target_mean=target_mean,
        target_std=target_std,
        physiology_mean=physiology_mean,
        physiology_std=physiology_std,
    )
    arrays_selection = _arrays_for_frame(
        data,
        selection,
        target_mean=target_mean,
        target_std=target_std,
        physiology_mean=physiology_mean,
        physiology_std=physiology_std,
    )
    groupdro_preflight = _groupdro_preflight(fit, arrays_fit["group"])
    if method == "calibration_relative_groupdro" and not groupdro_preflight[
        "eligible"
    ]:
        raise RuntimeError(
            "Group-robust preflight failed: every folds0-2 source x support-range "
            "group requires at least 50 participants and 1000 events"
        )

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Round-14 candidate training requires CUDA")
    model = Round9Model(
        data.query_embeddings.shape[1],
        Candidate(),
        physiology_dim=data.query_physiology.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    target_std_tensor = torch.as_tensor(target_std, device=device)
    best_score = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    batches = _sequence_batches(fit, batch_size=batch_size)
    training_sequences = [sequence for batch in batches for sequence in batch]
    rng = np.random.default_rng(seed)
    for epoch in itertools.count(1):
        model.train()
        if method == "calibration_relative_groupdro":
            # Recompose the participant batches every epoch.  The historical
            # helper length-sorts sequences for efficient padding; retaining
            # those fixed batches can accidentally isolate a source/range
            # group.  Random recomposition exposes the training-only groups to
            # the smooth worst-group objective without adding them to the
            # model inputs.
            order = rng.permutation(len(training_sequences))
            epoch_sequences = [training_sequences[index] for index in order]
            epoch_batches = [
                epoch_sequences[start : start + batch_size]
                for start in range(0, len(epoch_sequences), batch_size)
            ]
        else:
            epoch_batches = list(batches)
            rng.shuffle(epoch_batches)
        total = 0.0
        diagnostic_totals: dict[str, float] = {}
        for sequences in epoch_batches:
            optimizer.zero_grad(set_to_none=True)
            full = _forward_sequences(model, arrays_fit, sequences, device)
            if method == "calibration_relative_groupdro":
                _attach_group_ids(full, arrays_fit, sequences, device)
            loss, diagnostics = candidate_loss(
                full, method=method, target_std=target_std_tensor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            for key, value in diagnostics.items():
                diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + value
        early_prediction = _predict(model, arrays_early, early, device)
        early_scored = _scored_predictions(
            early, early_prediction, target_mean, target_std
        )
        early_score = float(participant_macro_metrics(early_scored)["mean_mae"])
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_batch_loss_sum": total,
            "fold3_participant_macro_mean_mae": early_score,
        }
        record.update(
            {
                f"train_{key}_mean": value / max(len(batches), 1)
                for key, value in diagnostic_totals.items()
            }
        )
        history.append(record)
        if early_score < best_score:
            best_score = early_score
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= 8:
            break
    if best_state is None:
        raise RuntimeError("Round-14 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    selection_prediction = _predict(model, arrays_selection, selection, device)
    predictions = _scored_predictions(
        selection, selection_prediction, target_mean, target_std
    )
    metrics = {
        scope: participant_macro_metrics(group)
        for scope, group in [("Overall", predictions)]
        + [
            (source, predictions.loc[predictions["source"].eq(source)])
            for source in sorted(predictions["source"].unique())
        ]
    }

    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "selection_predictions.parquet", index=False)
    checkpoint = output / "best.pt"
    torch.save(
        {
            "model_state": best_state,
            "round": 14,
            "method": method,
            "seed": seed,
            "backbone": BACKBONE,
            "target_mean": target_mean,
            "target_std": target_std,
            "physiology_mean": physiology_mean,
            "physiology_std": physiology_std,
            "cache_run_sha256": file_sha256(prepared / "run.json"),
        },
        checkpoint,
    )
    save_json(output / "history.json", history)
    record = {
        "status": "complete",
        "round": 14,
        "stage": "exploratory_method",
        "method": method,
        "seed": seed,
        "backbone": BACKBONE,
        "k": SUPPORT_K,
        "split": "meta_train_internal_fold4",
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_STOPPING_FOLD,
        "selection_fold": SELECTION_FOLD,
        "participants": int(predictions["subject_uid"].nunique()),
        "queries": int(len(predictions)),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stop_reason": "early_stopping_patience_8_no_epoch_cap",
        "training_configuration": {
            "batch_size_participants": batch_size,
            "optimizer": "AdamW",
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 5.0,
            "patience": 8,
            "epoch_cap": None,
            "standards_physical_huber_weight": STANDARDS_HUBER_WEIGHT,
            "standards_physical_huber_delta_mmhg": STANDARDS_HUBER_DELTA_MMHG,
            "standards_threshold_weight": STANDARDS_THRESHOLD_WEIGHT,
            "standards_threshold_temperature_mmhg": STANDARDS_THRESHOLD_TEMPERATURE_MMHG,
            "standards_thresholds_mmhg": list(STANDARDS_THRESHOLDS_MMHG),
            "standards_threshold_relative_weights": list(
                STANDARDS_THRESHOLD_WEIGHTS
            ),
            "groupdro_weight": GROUPDRO_WEIGHT,
            "groupdro_temperature": GROUPDRO_TEMPERATURE,
        },
        "components": {
            "pairwise_query_support_delta": True,
            "causal_time_decay_gru": True,
            "support_range_auxiliary": True,
            "physical_mmHg_huber": method
            in {
                "calibration_relative_standards",
                "calibration_relative_groupdro",
            },
            "differentiable_5_10_15_threshold_penalty": method
            in {
                "calibration_relative_standards",
                "calibration_relative_groupdro",
            },
            "groupdro_inspired_smooth_source_by_support_range_penalty": method
            == "calibration_relative_groupdro",
        },
        "standards_surrogate_optimized": method
        in {
            "calibration_relative_standards",
            "calibration_relative_groupdro",
        },
        "aami_bhs_compliance_claim": False,
        "source_training_group": method == "calibration_relative_groupdro",
        "groupdro_preflight": groupdro_preflight,
        "canonical_groupdro": False,
        "group_objective_interpretation": (
            "GroupDRO-inspired batch-local smooth worst-group penalty; no persistent adversarial group weights"
            if method == "calibration_relative_groupdro"
            else None
        ),
        "query_bp_training_target": True,
        "meta_validation_used_for_training": False,
        "meta_validation_used_for_early_stopping": False,
        "meta_validation_used_for_candidate_ranking": False,
        "meta_validation_predictions_generated": False,
        "locked_test_accessed": False,
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "source_model_input": False,
        "cache_run_sha256": file_sha256(prepared / "run.json"),
        "cache_bindings": {
            key: cache_record[key]
            for key in (
                "population_checkpoint_sha256",
                "qgh_checkpoint_sha256",
                "folds_sha256",
                "store_manifest_sha256",
                "training_source_tree_sha256",
                "cache_source_tree_sha256",
            )
        },
        "checkpoint_sha256": file_sha256(checkpoint),
        "selection_predictions_sha256": file_sha256(
            output / "selection_predictions.parquet"
        ),
        "history_sha256": file_sha256(output / "history.json"),
        "source_tree_sha256": source_tree_sha256(
            Path(__file__).resolve().parents[2]
        ),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "parameter_counts": model_parameter_counts(model),
        "metrics": metrics,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output / "run.json", record)
    return record


def _validate_candidate_run(
    root: Path, *, expected_seed: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    record = _read_json(root / "run.json")
    expected = {
        "status": "complete",
        "round": 14,
        "stage": "exploratory_method",
        "seed": expected_seed,
        "backbone": BACKBONE,
        "split": "meta_train_internal_fold4",
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_STOPPING_FOLD,
        "selection_fold": SELECTION_FOLD,
        "aami_bhs_compliance_claim": False,
        "source_model_input": False,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"{root} has unexpected {key}")
    if record.get("method") not in METHODS:
        raise AssertionError(f"{root} has an unknown Round-14 method")
    _require_false(
        record,
        (
            "meta_validation_used_for_training",
            "meta_validation_used_for_early_stopping",
            "meta_validation_used_for_candidate_ranking",
            "meta_validation_predictions_generated",
            "locked_test_accessed",
            "query_bp_model_input",
            "future_query_model_input",
            "source_model_input",
        ),
        str(root),
    )
    predictions = pd.read_parquet(root / "selection_predictions.parquet")
    required = set(
        KEYS
        + ["source", "target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]
    )
    if missing := required - set(predictions.columns):
        raise ValueError(f"{root} predictions missing {sorted(missing)}")
    if predictions.duplicated(KEYS).any():
        raise AssertionError(f"{root} has duplicate prediction keys")
    numeric = predictions[
        ["target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise AssertionError(f"{root} has non-finite predictions")
    recorded_prediction_hash = record.get("selection_predictions_sha256")
    if recorded_prediction_hash and recorded_prediction_hash != file_sha256(
        root / "selection_predictions.parquet"
    ):
        raise AssertionError(f"{root} prediction SHA-256 mismatch")
    return record, predictions


def _validate_qgh_anchor(
    root: Path, *, expected_seed: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load the same-seed wide-Inception QGH fold-4 reference predictions."""

    record = _read_json(root / "run.json")
    expected = {
        "status": "complete",
        "stage": "backbone_evaluation",
        "backbone": BACKBONE,
        "seed": expected_seed,
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_STOPPING_FOLD,
        "selection_fold": SELECTION_FOLD,
        "support_policy": "fixed_first",
        "k": SUPPORT_K,
        "meta_validation_accessed": False,
        "locked_test_accessed": False,
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "source_model_input": False,
        "meta_validation_used_for_training": False,
        "meta_validation_used_for_early_stopping": False,
        "meta_validation_used_for_candidate_ranking": False,
        "meta_validation_predictions_generated": False,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"QGH anchor has unexpected {key}")
    if int(record.get("round", -1)) not in {13, 14}:
        raise AssertionError("QGH anchor has an unsupported round number")
    predictions = pd.read_parquet(root / "queries.parquet")
    predictions = predictions.loc[
        pd.to_numeric(predictions["fold"]).eq(SELECTION_FOLD)
    ].copy()
    predictions.reset_index(drop=True, inplace=True)
    if predictions.empty:
        raise AssertionError("QGH anchor has no fold-4 predictions")
    if predictions["split"].astype(str).ne("meta_train").any():
        raise AssertionError("QGH anchor fold 4 is not meta-train")
    if predictions["k"].astype(int).ne(SUPPORT_K).any():
        raise AssertionError("QGH anchor does not use fixed K=5")
    required = set(
        KEYS
        + ["source", "target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]
    )
    if missing := required - set(predictions.columns):
        raise ValueError(f"QGH anchor predictions missing {sorted(missing)}")
    if predictions.duplicated(KEYS).any():
        raise AssertionError("QGH anchor has duplicate fold-4 query keys")
    return record, predictions


def _candidate_gate_table(
    table: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    reference: str = ANCHOR_SETTING,
) -> pd.DataFrame:
    """Evaluate each complete candidate independently against the QGH anchor."""

    reference_table = table.loc[table["Setting"].eq(reference)].set_index("Scope")
    reference_diagnostics = diagnostics.loc[
        diagnostics["Setting"].eq(reference)
    ].set_index(["Scope", "BP"])
    threshold_columns = ("≤5 mmHg", "≤10 mmHg", "≤15 mmHg")
    rows: list[dict[str, Any]] = []
    for setting in table["Setting"].drop_duplicates():
        if setting == reference:
            continue
        candidate = table.loc[table["Setting"].eq(setting)].set_index("Scope")
        candidate_diagnostics = diagnostics.loc[
            diagnostics["Setting"].eq(setting)
        ].set_index(["Scope", "BP"])
        gains = {
            scope: float(
                reference_table.loc[scope, "Mean participant-macro MAE"]
                - candidate.loc[scope, "Mean participant-macro MAE"]
            )
            for scope in ("Overall", "MIMIC", "VitalDB")
        }
        endpoint_worsening = {
            bp: float(
                candidate.loc["Overall", f"{bp} participant-macro MAE"]
                - reference_table.loc[
                    "Overall", f"{bp} participant-macro MAE"
                ]
            )
            for bp in ("SBP", "DBP")
        }
        primary_gate = bool(
            gains["Overall"] >= 0.15
            and gains["MIMIC"] > 0.0
            and gains["VitalDB"] > 0.0
            and max(endpoint_worsening.values()) <= 0.05
        )

        std_deltas = {
            f"{scope}_{bp}": float(
                candidate_diagnostics.loc[(scope, bp), "STD"]
                - reference_diagnostics.loc[(scope, bp), "STD"]
            )
            for scope in ("Overall", "MIMIC", "VitalDB")
            for bp in ("SBP", "DBP")
        }
        std_decreases_all_scopes = all(
            np.isfinite(value) and value < 0.0 for value in std_deltas.values()
        )
        threshold_deltas: list[float] = []
        endpoint_average_improvements: dict[str, float] = {}
        for scope in ("Overall", "MIMIC", "VitalDB"):
            for bp in ("SBP", "DBP"):
                values = [
                    float(
                        candidate_diagnostics.loc[(scope, bp), column]
                        - reference_diagnostics.loc[(scope, bp), column]
                    )
                    for column in threshold_columns
                ]
                threshold_deltas.extend(values)
                if scope == "Overall":
                    endpoint_average_improvements[bp] = float(np.mean(values))
        minimum_threshold_change = float(min(threshold_deltas))
        tail_gate = bool(
            gains["Overall"] >= -0.05
            and gains["MIMIC"] >= -0.10
            and gains["VitalDB"] >= -0.10
            and std_decreases_all_scopes
            and endpoint_average_improvements["SBP"] >= 1.0
            and endpoint_average_improvements["DBP"] >= 1.0
            and minimum_threshold_change >= -0.5
        )
        rows.append(
            {
                "Setting": setting,
                "Overall mean-MAE gain": gains["Overall"],
                "MIMIC mean-MAE gain": gains["MIMIC"],
                "VitalDB mean-MAE gain": gains["VitalDB"],
                "Overall SBP MAE worsening": endpoint_worsening["SBP"],
                "Overall DBP MAE worsening": endpoint_worsening["DBP"],
                "All-scope SBP/DBP pooled STD decreases": std_decreases_all_scopes,
                "Overall SBP mean within-threshold gain (pp)": endpoint_average_improvements[
                    "SBP"
                ],
                "Overall DBP mean within-threshold gain (pp)": endpoint_average_improvements[
                    "DBP"
                ],
                "Minimum individual within-threshold change (pp)": minimum_threshold_change,
                "Primary gate": primary_gate,
                "Tail gate": tail_gate,
                "Advance": primary_gate or tail_gate,
            }
        )
    return pd.DataFrame(rows)


def build_internal_report(
    *,
    runs: dict[str, Path],
    anchor_run: Path,
    output: Path,
    expected_seed: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if expected_seed != EXPLORATORY_SEED:
        raise AssertionError(
            f"Round-14 exploratory report requires seed {EXPLORATORY_SEED}"
        )
    if ANCHOR_SETTING in runs:
        raise KeyError(f"{ANCHOR_SETTING} is reserved for the matched QGH anchor")
    records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    candidate_artifacts: dict[str, dict[str, str]] = {}
    anchor_record, anchor_predictions = _validate_qgh_anchor(
        anchor_run, expected_seed=expected_seed
    )
    canonical = anchor_predictions[
        KEYS + ["target_sbp", "target_dbp"]
    ].sort_values(KEYS, kind="mergesort")
    canonical.reset_index(drop=True, inplace=True)
    frames.append(anchor_predictions.assign(Setting=ANCHOR_SETTING))
    for scope, group in [("Overall", anchor_predictions)] + [
        (
            source,
            anchor_predictions.loc[anchor_predictions["source"].eq(source)],
        )
        for source in sorted(anchor_predictions["source"].unique())
    ]:
        metrics = participant_macro_metrics(group)
        records.append(
            {
                "Setting": ANCHOR_SETTING,
                "Scope": scope,
                "N participants": int(group["subject_uid"].nunique()),
                "N queries": int(len(group)),
                "SBP participant-macro MAE": metrics["sbp_mae"],
                "DBP participant-macro MAE": metrics["dbp_mae"],
                "Mean participant-macro MAE": metrics["mean_mae"],
            }
        )
    common_cache_sha: str | None = None
    for setting, root in runs.items():
        run, predictions = _validate_candidate_run(root, expected_seed=expected_seed)
        if setting != run.get("method"):
            raise AssertionError(
                f"report setting {setting} does not match the candidate method"
            )
        candidate_artifacts[setting] = {
            "run_json_sha256": file_sha256(root / "run.json"),
            "selection_predictions_sha256": file_sha256(
                root / "selection_predictions.parquet"
            ),
        }
        if common_cache_sha is None:
            common_cache_sha = str(run["cache_run_sha256"])
        elif run.get("cache_run_sha256") != common_cache_sha:
            raise AssertionError("Round-14 candidates used different caches")
        checked = predictions[KEYS + ["target_sbp", "target_dbp"]].sort_values(
            KEYS, kind="mergesort"
        )
        checked.reset_index(drop=True, inplace=True)
        if not canonical.equals(checked):
            raise AssertionError("Round-14 candidates have different queries or targets")
        bindings = run.get("cache_bindings")
        if not isinstance(bindings, dict):
            raise AssertionError(f"{root} lacks immutable cache bindings")
        expected_bindings = {
            "population_checkpoint_sha256": anchor_record.get(
                "population_checkpoint_sha256"
            ),
            "qgh_checkpoint_sha256": anchor_record.get("qgh_checkpoint_sha256"),
            "folds_sha256": anchor_record.get("folds_sha256"),
            "store_manifest_sha256": anchor_record.get("store_manifest_sha256"),
        }
        for key, value in expected_bindings.items():
            if bindings.get(key) != value:
                raise AssertionError(
                    f"{setting} cache binding differs from the matched QGH anchor: {key}"
                )
        training_audit = anchor_record.get("training_audit")
        if isinstance(training_audit, dict) and bindings.get(
            "training_source_tree_sha256"
        ) != training_audit.get("source_tree_sha256"):
            raise AssertionError(
                f"{setting} source tree differs from the matched QGH anchor"
            )
        frames.append(predictions.assign(Setting=setting))
        for scope, group in [("Overall", predictions)] + [
            (source, predictions.loc[predictions["source"].eq(source)])
            for source in sorted(predictions["source"].unique())
        ]:
            metrics = participant_macro_metrics(group)
            records.append(
                {
                    "Setting": setting,
                    "Scope": scope,
                    "N participants": int(group["subject_uid"].nunique()),
                    "N queries": int(len(group)),
                    "SBP participant-macro MAE": metrics["sbp_mae"],
                    "DBP participant-macro MAE": metrics["dbp_mae"],
                    "Mean participant-macro MAE": metrics["mean_mae"],
                }
            )
    table = pd.DataFrame(records)
    for setting, group in table.groupby("Setting", sort=False):
        if set(group["Scope"]) != {"Overall", "MIMIC", "VitalDB"}:
            raise AssertionError(f"{setting} lacks one of the three report scopes")
        overall_row = group.loc[group["Scope"].eq("Overall")].iloc[0]
        sources = group.loc[group["Scope"].isin(["MIMIC", "VitalDB"])]
        if int(overall_row["N participants"]) != int(
            sources["N participants"].sum()
        ) or int(overall_row["N queries"]) != int(sources["N queries"].sum()):
            raise AssertionError(f"{setting} source counts do not sum to Overall")
    settings = [ANCHOR_SETTING, *runs]
    diagnostics = _diagnostic_rows(
        pd.concat(frames, ignore_index=True, sort=False), settings
    )
    overall = table.loc[table["Scope"].eq("Overall")].sort_values(
        ["Mean participant-macro MAE", "Setting"], kind="mergesort"
    )
    winner = str(overall.iloc[0]["Setting"])
    reference = ANCHOR_SETTING
    reference_table = table.loc[table["Setting"].eq(reference)].set_index("Scope")
    winner_table = table.loc[table["Setting"].eq(winner)].set_index("Scope")
    gains = {
        scope: float(
            reference_table.loc[scope, "Mean participant-macro MAE"]
            - winner_table.loc[scope, "Mean participant-macro MAE"]
        )
        for scope in ("Overall", "MIMIC", "VitalDB")
    }
    candidate_gates = _candidate_gate_table(
        table, diagnostics, reference=reference
    )
    primary_passers = candidate_gates.loc[
        candidate_gates["Primary gate"], "Setting"
    ].astype(str).tolist()
    tail_passers = candidate_gates.loc[
        candidate_gates["Tail gate"], "Setting"
    ].astype(str).tolist()
    advancing_candidates = candidate_gates.loc[
        candidate_gates["Advance"], "Setting"
    ].astype(str).tolist()
    passes = bool(advancing_candidates)
    if primary_passers:
        eligible = primary_passers
    else:
        eligible = tail_passers
    promoted = None
    if eligible:
        promoted = str(
            overall.loc[overall["Setting"].isin(eligible)].iloc[0]["Setting"]
        )

    comparison_rows: list[dict[str, Any]] = []
    for setting in settings:
        candidate = table.loc[table["Setting"].eq(setting)].set_index("Scope")
        for scope in ("Overall", "MIMIC", "VitalDB"):
            comparison_rows.append(
                {
                    "Setting": setting,
                    "Scope": scope,
                    "Reference mean participant-macro MAE": float(
                        reference_table.loc[scope, "Mean participant-macro MAE"]
                    ),
                    "Candidate mean participant-macro MAE": float(
                        candidate.loc[scope, "Mean participant-macro MAE"]
                    ),
                    "Candidate minus reference": float(
                        candidate.loc[scope, "Mean participant-macro MAE"]
                        - reference_table.loc[scope, "Mean participant-macro MAE"]
                    ),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "participant_macro_internal.csv", index=False)
    diagnostics.to_csv(output / "pooled_diagnostics_internal.csv", index=False)
    comparison.to_csv(output / "comparison_vs_reference_internal.csv", index=False)
    candidate_gates.to_csv(output / "candidate_gate_internal.csv", index=False)
    summary: dict[str, Any] = {
        "status": "complete",
        "round": 14,
        "stage": "exploratory_method_screen",
        "seed": expected_seed,
        "split": "meta_train_internal_fold4",
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_STOPPING_FOLD,
        "selection_fold": SELECTION_FOLD,
        "meta_validation_used_for_candidate_ranking": False,
        "locked_test_accessed": False,
        "reference": reference,
        "anchor_run_json_sha256": file_sha256(anchor_run / "run.json"),
        "anchor_queries_sha256": file_sha256(anchor_run / "queries.parquet"),
        "candidate_artifacts": candidate_artifacts,
        "winner": winner,
        "gain_vs_reference": gains,
        "primary_gate_passers": primary_passers,
        "tail_gate_passers": tail_passers,
        "advancing_candidates": advancing_candidates,
        "promoted_candidate": promoted,
        "candidate_gates": candidate_gates.to_dict(orient="records"),
        "passes_internal_gate": passes,
        "candidate_count": len(runs),
        "aami_bhs_interpretation": "retrospective numerical screens only; no compliance claim",
    }
    save_json(output / "selection.json", summary)
    lines = [
        "# Round-14 exploratory calibration-relative internal screen",
        "",
        "All candidates fit folds 0--2, use fold 3 only for patience-8 early stopping with no epoch cap, and rank on fold 4. Meta-validation and the locked test were not accessed.",
        "",
        "## Participant-macro primary results",
        "",
        _markdown_table(table.sort_values(["Scope", "Mean participant-macro MAE"])),
        "",
        "## Change versus matched same-seed wide-Inception + QGH anchor",
        "",
        _markdown_table(comparison),
        "",
        "## Prespecified candidate gates",
        "",
        "Each complete candidate is evaluated independently. Passing the primary or tail-focused development gate permits only later confirmation; it does not authorize meta-validation or locked-test access.",
        "",
        _markdown_table(candidate_gates),
        "",
        "## Event-pooled secondary diagnostics",
        "",
        "AAMI/BHS fields are retrospective numerical screens only. The standards-oriented candidate optimizes differentiable error-threshold surrogates but does not establish standards or device compliance.",
        "The optional group-robust candidate uses a GroupDRO-inspired batch-local smooth worst-group penalty; it is not canonical GroupDRO and maintains no persistent adversarial group weights.",
        "",
        _markdown_table(diagnostics),
        "",
        f"Internal numerical winner: **{winner}**.",
        f"Candidate selected by the prespecified gates: **{promoted}**.",
        f"At least one internal development gate passed: **{passes}**.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _parse_runs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--run must use SETTING=PATH")
        setting, path = value.split("=", 1)
        if not setting or setting in result:
            raise ValueError(f"invalid or duplicate setting {setting}")
        result[setting] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-cache")
    prepare.add_argument("--store-root", type=Path, required=True)
    prepare.add_argument("--folds", type=Path, required=True)
    prepare.add_argument("--population-run", type=Path, required=True)
    prepare.add_argument("--qgh-run", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--batch-size", type=int, default=256)
    prepare.add_argument("--require-cuda", action="store_true")

    train = commands.add_parser("train-candidate")
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--method", choices=METHODS, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--batch-size", type=int, default=12)
    train.add_argument("--require-cuda", action="store_true")

    report = commands.add_parser("report-internal")
    report.add_argument("--run", action="append", required=True)
    report.add_argument("--anchor-run", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--expected-seed", type=int, required=True)

    args = parser.parse_args()
    if args.command == "prepare-cache":
        result = prepare_cache(
            store_root=args.store_root,
            folds_path=args.folds,
            population_run=args.population_run,
            qgh_run=args.qgh_run,
            output=args.output,
            seed=args.seed,
            batch_size=args.batch_size,
            require_cuda=args.require_cuda,
        )
    elif args.command == "train-candidate":
        result = train_candidate(
            prepared=args.prepared,
            output=args.output,
            method=args.method,
            seed=args.seed,
            batch_size=args.batch_size,
            require_cuda=args.require_cuda,
        )
    else:
        result = build_internal_report(
            runs=_parse_runs(args.run),
            anchor_run=args.anchor_run,
            output=args.output,
            expected_seed=args.expected_seed,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
