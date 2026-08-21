"""Round-10 leakage-safe partial end-to-end calibration screen.

The round rebuilds its population and Quality-Gate + Huber base using only
meta-train folds 0--2, uses fold 3 for early stopping, and reserves fold 4 for
candidate ranking.  Candidate models load raw PPG waveforms and may update a
prespecified suffix of the population encoder.  Meta-validation and the locked
meta-test are never read by this module.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import VariableKPersonalizer
from .phase6e_residual import KEYS
from .report_round8 import _diagnostic_rows
from .round8_calibration_relative import SUPPORT_K, _range_labels, _sequence_batches
from .round9_refinement import Candidate as HeadCandidate
from .round9_refinement import (
    Round9Model,
    TimeDecayGRU,
    _loss as _round9_loss,
    _markdown_table,
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


FIT_FOLDS = (0, 1, 2)
EARLY_FOLD = 3
SELECTION_FOLD = 4


@dataclass(frozen=True)
class Round10Candidate:
    encoder_mode: str
    encoder_learning_rate: float
    pair_direction_weight: float = 0.0
    temporal_delta_penalty: float = 0.0
    prediction_mode: str = "correction"


METHODS: dict[str, Round10Candidate] = {
    "frozen_reference": Round10Candidate("frozen", 0.0),
    "projection_only": Round10Candidate("projection", 1e-4),
    "last_block": Round10Candidate("last_block", 5e-5),
    "last_two_blocks": Round10Candidate("last_two_blocks", 3e-5),
    "full_encoder": Round10Candidate("full", 1e-5),
    "last_block_direction": Round10Candidate(
        "last_block", 5e-5, pair_direction_weight=0.05
    ),
    "last_block_temporal": Round10Candidate(
        "last_block", 5e-5, temporal_delta_penalty=0.10
    ),
    "last_block_adaptive": Round10Candidate(
        "last_block", 5e-5, prediction_mode="adaptive_fusion"
    ),
    "last_block_joint": Round10Candidate(
        "last_block",
        5e-5,
        pair_direction_weight=0.05,
        temporal_delta_penalty=0.10,
    ),
}


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )


def _load_qgh_model(
    population_checkpoint: Path,
    qgh_checkpoint: Path,
    device: torch.device,
) -> tuple[VariableKPersonalizer, dict[str, list[float]]]:
    population, scaler = _load_population_checkpoint(population_checkpoint, device)
    model = VariableKPersonalizer(
        population,
        use_film=False,
        query_conditioned_weights=False,
        anchor_mode="mean",
        use_quality_gate=True,
    )
    checkpoint = torch.load(qgh_checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("method") != "m0":
        raise AssertionError("Round-10 base checkpoint must be an M0 model")
    other = checkpoint.get("target_scaler")
    if other is None or not (
        np.allclose(other["mean"], scaler["mean"], atol=0.0, rtol=0.0)
        and np.allclose(other["std"], scaler["std"], atol=0.0, rtol=0.0)
    ):
        raise AssertionError("population and QGH target scalers differ")
    model.load_state_dict(checkpoint["model_state"])
    return model, scaler


def _validate_internal_base_run(path: Path, expected_method: str) -> dict[str, object]:
    payload = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("method") != expected_method:
        raise AssertionError(f"invalid Round-10 {expected_method} base run")
    if payload.get("locked_test_accessed") is not False:
        raise AssertionError("Round-10 base accessed locked test")
    if payload.get("crossfit_fit_folds") != list(FIT_FOLDS):
        raise AssertionError("Round-10 base has wrong fit folds")
    if payload.get("crossfit_validation_fold") != EARLY_FOLD:
        raise AssertionError("Round-10 base has wrong early-stopping fold")
    if payload.get("crossfit_excluded_folds") != [SELECTION_FOLD]:
        raise AssertionError("Round-10 base did not quarantine selection fold")
    return payload


def _fold_table(path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    folds = pd.read_parquet(path)
    required = {"subject_uid", "source", "fold"}
    if required - set(folds):
        raise ValueError(f"fold table missing {sorted(required - set(folds))}")
    folds = folds[list(required)].copy()
    folds["subject_uid"] = folds["subject_uid"].astype(str)
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
    if folds["subject_uid"].duplicated().any():
        raise AssertionError("fold table contains duplicate participants")
    if set(folds["fold"].unique()) != set(range(5)):
        raise AssertionError("Round-10 requires exactly five meta-train folds")
    subjects = set(metadata["subject_uid"].astype(str))
    if set(folds["subject_uid"]) != subjects:
        raise AssertionError("fold table does not exactly cover eligible meta-train subjects")
    source = metadata[["subject_uid", "source"]].drop_duplicates().assign(
        subject_uid=lambda frame: frame["subject_uid"].astype(str)
    )
    checked = folds.merge(source, on="subject_uid", suffixes=("_fold", "_data"))
    if not checked["source_fold"].astype(str).equals(checked["source_data"].astype(str)):
        raise AssertionError("fold-table source labels differ from event metadata")
    return folds


def _encode_metadata(
    metadata: pd.DataFrame,
    *,
    accessor: WaveformAccessor,
    model: VariableKPersonalizer,
    scaler: dict[str, list[float]],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = int(model.population.encoder.feature_dim)
    features = np.empty((len(metadata), dimension), dtype=np.float32)
    population_bp = np.empty((len(metadata), 2), dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    model.population.eval()
    with torch.no_grad():
        for start in range(0, len(metadata), batch_size):
            part = metadata.iloc[start : start + batch_size]
            waveform = torch.stack(
                [
                    accessor.get(str(row.waveform_file), int(row.waveform_row))
                    for row in part.itertuples()
                ]
            ).to(device)
            with _autocast(device):
                encoded = model.population.encoder(waveform)
                prediction = model.population.predict_from_features(encoded)
            values = encoded.float().cpu().numpy()
            features[start : start + len(part)] = values
            population_bp[start : start + len(part)] = (
                prediction.float().cpu().numpy() * std + mean
            )
    return features, population_bp


def prepare_round10(
    *,
    store_root: Path,
    folds_path: Path,
    population_checkpoint: Path,
    qgh_checkpoint: Path,
    output: Path,
    batch_size: int = 1024,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    population_run = _validate_internal_base_run(population_checkpoint, "population")
    qgh_run = _validate_internal_base_run(qgh_checkpoint, "m0")
    if population_run.get("target_scaler") != qgh_run.get("target_scaler"):
        raise AssertionError("Round-10 base run scalers differ")

    metadata = load_store_metadata(store_root, "development")
    metadata = metadata.loc[metadata["split"].eq("meta_train")].copy()
    if metadata.empty or metadata["split"].ne("meta_train").any():
        raise AssertionError("Round-10 preparation must contain meta-train only")
    folds = _fold_table(folds_path, metadata)
    metadata["subject_uid"] = metadata["subject_uid"].astype(str)
    metadata = metadata.merge(
        folds[["subject_uid", "fold"]], on="subject_uid", validate="many_to_one"
    ).sort_values(["subject_uid", "event_index", "event_id"], kind="mergesort")
    metadata.reset_index(drop=True, inplace=True)
    metadata["feature_row"] = np.arange(len(metadata), dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Round-10 preparation requires CUDA")
    qgh, scaler = _load_qgh_model(
        population_checkpoint, qgh_checkpoint, device
    )
    qgh.to(device).eval()
    accessor = WaveformAccessor(store_root)
    event_features, event_population = _encode_metadata(
        metadata,
        accessor=accessor,
        model=qgh,
        scaler=scaler,
        device=device,
        batch_size=batch_size,
    )

    support = metadata.loc[metadata["event_index"].le(SUPPORT_K)].copy()
    support.sort_values(
        ["subject_uid", "event_index", "event_id"], kind="mergesort", inplace=True
    )
    counts = support.groupby("subject_uid").size()
    if not counts.eq(SUPPORT_K).all():
        raise AssertionError("every Round-10 participant requires five supports")
    support["support_position"] = support.groupby("subject_uid").cumcount()
    subjects = sorted(support["subject_uid"].unique())
    support_row = {subject: index for index, subject in enumerate(subjects)}
    support["support_row"] = support["subject_uid"].map(support_row).astype(np.int64)
    support_feature_rows = support["feature_row"].to_numpy(np.int64).reshape(-1, SUPPORT_K)
    support_features = event_features[support_feature_rows]
    support_bp = support[["sbp", "dbp"]].to_numpy(np.float32).reshape(-1, SUPPORT_K, 2)
    support_population = event_population[support_feature_rows]

    queries = metadata.loc[metadata["common_query"].astype(bool)].copy()
    queries["support_row"] = queries["subject_uid"].map(support_row)
    if queries["support_row"].isna().any():
        raise AssertionError("query/support mapping is incomplete")
    queries["support_row"] = queries["support_row"].astype(np.int64)
    target_mean = np.asarray(scaler["mean"], dtype=np.float32)
    target_std = np.asarray(scaler["std"], dtype=np.float32)
    predictions = np.empty((len(queries), 2), dtype=np.float32)
    query_feature_rows = queries["feature_row"].to_numpy(np.int64)
    query_support_rows = queries["support_row"].to_numpy(np.int64)
    with torch.no_grad():
        for start in range(0, len(queries), batch_size * 4):
            stop = min(start + batch_size * 4, len(queries))
            rows = slice(start, stop)
            q = torch.from_numpy(event_features[query_feature_rows[rows]]).to(device)
            s = torch.from_numpy(support_features[query_support_rows[rows]]).to(device)
            bp = torch.from_numpy(
                (support_bp[query_support_rows[rows]] - target_mean) / target_std
            ).to(device)
            mask = torch.ones((stop - start, SUPPORT_K), dtype=torch.bool, device=device)
            with _autocast(device):
                pred = qgh.forward_from_features(q, s, bp, mask)
            predictions[rows] = pred.float().cpu().numpy() * target_std + target_mean
    queries["pred_sbp"] = predictions[:, 0]
    queries["pred_dbp"] = predictions[:, 1]
    queries["k"] = SUPPORT_K
    queries.rename(columns={"sbp": "target_sbp", "dbp": "target_dbp"}, inplace=True)
    keep = [
        "subject_uid", "event_id", "k", "source", "split", "fold", "event_index",
        "record_order", "time_bin", "waveform_file", "waveform_row", "support_row",
        "target_sbp", "target_dbp", "pred_sbp", "pred_dbp",
    ]
    missing = set(keep) - set(queries)
    if missing:
        raise ValueError(f"Round-10 query metadata missing {sorted(missing)}")
    queries = queries[keep].sort_values(
        ["fold", "subject_uid", "event_index", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if queries.duplicated(KEYS).any():
        raise AssertionError("Round-10 query keys are not unique")

    output.mkdir(parents=True)
    queries.to_parquet(output / "queries.parquet", index=False)
    support[
        [
            "subject_uid", "event_id", "source", "split", "fold", "event_index",
            "waveform_file", "waveform_row", "support_position", "support_row",
        ]
    ].to_parquet(output / "support_index.parquet", index=False)
    np.save(output / "support_bp.npy", support_bp)
    np.save(output / "support_population_bp.npy", support_population)
    fold_counts = {
        str(fold): {
            "participants": int(group["subject_uid"].nunique()),
            "queries": int(len(group)),
        }
        for fold, group in queries.groupby("fold")
    }
    payload = {
        "status": "complete",
        "round": 10,
        "split": "meta_train_internal_only",
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_FOLD,
        "selection_fold": SELECTION_FOLD,
        "meta_validation_accessed": False,
        "locked_test_accessed": False,
        "support_policy": "fixed_first",
        "k": SUPPORT_K,
        "participants": int(queries["subject_uid"].nunique()),
        "queries": int(len(queries)),
        "fold_counts": fold_counts,
        "target_scaler": scaler,
        "population_checkpoint_sha256": file_sha256(population_checkpoint),
        "qgh_checkpoint_sha256": file_sha256(qgh_checkpoint),
        "folds_sha256": file_sha256(folds_path),
        "store_manifest_sha256": file_sha256(store_root / "materialization.json"),
        "store_root": str(store_root),
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "source_model_input": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output / "run.json", payload)
    return payload


class PreparedRound10:
    def __init__(self, root: Path) -> None:
        self.root = root
        payload = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("round") != 10:
            raise AssertionError("invalid Round-10 prepared cache")
        if payload.get("meta_validation_accessed") is not False:
            raise AssertionError("prepared cache accessed meta-validation")
        if payload.get("locked_test_accessed") is not False:
            raise AssertionError("prepared cache accessed locked test")
        if payload.get("fit_folds") != list(FIT_FOLDS):
            raise AssertionError("prepared cache has wrong fit folds")
        if payload.get("early_stopping_fold") != EARLY_FOLD:
            raise AssertionError("prepared cache has wrong early fold")
        if payload.get("selection_fold") != SELECTION_FOLD:
            raise AssertionError("prepared cache has wrong selection fold")
        self.payload = payload
        self.queries = pd.read_parquet(root / "queries.parquet")
        self.support = pd.read_parquet(root / "support_index.parquet")
        self.support_bp = np.load(root / "support_bp.npy", mmap_mode="r")
        self.support_population = np.load(
            root / "support_population_bp.npy", mmap_mode="r"
        )
        if self.queries["split"].ne("meta_train").any():
            raise AssertionError("Round-10 cache contains a non-meta-train query")
        if self.queries.duplicated(KEYS).any():
            raise AssertionError("Round-10 cache has duplicate query keys")
        ordered = self.support.sort_values(
            ["support_row", "support_position"], kind="mergesort"
        ).reset_index(drop=True)
        expected = np.tile(np.arange(SUPPORT_K), len(self.support_bp))
        if not np.array_equal(ordered["support_position"].to_numpy(), expected):
            raise AssertionError("support positions are not complete and ordered")
        self.support = ordered

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.payload["target_scaler"]["mean"], dtype=np.float32)

    @property
    def std(self) -> np.ndarray:
        return np.asarray(self.payload["target_scaler"]["std"], dtype=np.float32)

    def arrays(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        support_row = frame["support_row"].to_numpy(np.int64)
        event_index = frame["event_index"].to_numpy(np.float32)
        gaps = np.maximum(
            event_index[:, None] - np.arange(1, SUPPORT_K + 1, dtype=np.float32)[None],
            0.0,
        )
        return {
            "base": (
                frame[["pred_sbp", "pred_dbp"]].to_numpy(np.float32) - self.mean
            ) / self.std,
            "target": (
                frame[["target_sbp", "target_dbp"]].to_numpy(np.float32) - self.mean
            ) / self.std,
            "support_bp": (
                np.asarray(self.support_bp[support_row], dtype=np.float32) - self.mean
            ) / self.std,
            "support_population": (
                np.asarray(self.support_population[support_row], dtype=np.float32)
                - self.mean
            ) / self.std,
            "event_gap": np.log1p(gaps) / np.float32(math.log1p(300.0)),
            "sequence_gap": np.log1p(
                frame.groupby("subject_uid", sort=False)["event_index"]
                .diff()
                .fillna(frame["event_index"] - SUPPORT_K)
                .clip(lower=0)
                .to_numpy(np.float32)
            ) / np.float32(math.log1p(300.0)),
            "range": _range_labels(
                frame[["target_sbp", "target_dbp"]].to_numpy(np.float32),
                np.asarray(self.support_bp[support_row], dtype=np.float32),
            ),
        }


def configure_encoder_trainability(encoder: nn.Module, mode: str) -> list[str]:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    modules: list[nn.Module]
    if mode == "frozen":
        modules = []
    elif mode == "projection":
        modules = [encoder.projection]
    elif mode == "last_block":
        modules = [encoder.blocks[-1], encoder.projection]
    elif mode == "last_two_blocks":
        modules = [encoder.blocks[-2], encoder.blocks[-1], encoder.projection]
    elif mode == "full":
        modules = [encoder]
    else:
        raise ValueError(f"unknown encoder mode {mode}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [name for name, value in encoder.named_parameters() if value.requires_grad]


class Round10Model(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        candidate: Round10Candidate,
    ) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.candidate = candidate
        self.trainable_encoder_parameters = configure_encoder_trainability(
            self.encoder, candidate.encoder_mode
        )
        head_candidate = HeadCandidate(
            prediction_mode=candidate.prediction_mode,
            temporal_mode="gru",
            temporal_delta_penalty=candidate.temporal_delta_penalty,
        )
        self.head = Round9Model(
            int(self.encoder.feature_dim), head_candidate, physiology_dim=0
        )
        self.direction_head = (
            nn.Linear(self.head.hidden_dim, 2)
            if candidate.pair_direction_weight > 0
            else None
        )

    def keep_encoder_statistics_frozen(self) -> None:
        # The internal participant batches are small.  Preserve the population
        # encoder's BatchNorm running statistics while allowing selected
        # convolution/projection weights to receive gradients.
        self.encoder.eval()


def _padded_numeric(
    arrays: dict[str, np.ndarray], sequences: list[np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    batch = len(sequences)
    length = max(map(len, sequences))
    shapes = {
        "base": (batch, length, 2),
        "target": (batch, length, 2),
        "support_bp": (batch, length, SUPPORT_K, 2),
        "support_population": (batch, length, SUPPORT_K, 2),
        "event_gap": (batch, length, SUPPORT_K),
        "sequence_gap": (batch, length),
        "range": (batch, length, 2),
    }
    padded = {
        key: np.zeros(shape, dtype=np.int64 if key == "range" else np.float32)
        for key, shape in shapes.items()
    }
    mask = np.zeros((batch, length), dtype=bool)
    for row, indexes in enumerate(sequences):
        for key in padded:
            padded[key][row, : len(indexes)] = arrays[key][indexes]
        mask[row, : len(indexes)] = True
    return padded, mask


def _load_waveforms(
    data: PreparedRound10,
    frame: pd.DataFrame,
    sequences: list[np.ndarray],
    accessor: WaveformAccessor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(sequences)
    length = max(map(len, sequences))
    query = torch.zeros(batch, length, 1, 1250, dtype=torch.float32)
    support = torch.zeros(batch, SUPPORT_K, 1, 1250, dtype=torch.float32)
    for row, indexes in enumerate(sequences):
        part = frame.iloc[indexes]
        support_rows = part["support_row"].unique()
        if len(support_rows) != 1:
            raise AssertionError("one participant sequence maps to multiple support rows")
        support_row = int(support_rows[0])
        supports = data.support.loc[data.support["support_row"].eq(support_row)]
        if len(supports) != SUPPORT_K:
            raise AssertionError("support waveform row is incomplete")
        for position, item in enumerate(supports.itertuples()):
            support[row, position] = accessor.get(
                str(item.waveform_file), int(item.waveform_row)
            )
        for position, item in enumerate(part.itertuples()):
            query[row, position] = accessor.get(
                str(item.waveform_file), int(item.waveform_row)
            )
    return query, support


def _tensor(array: np.ndarray, device: torch.device, dtype=None) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(array))
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value.to(device)


def _forward_sequences(
    model: Round10Model,
    data: PreparedRound10,
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    sequences: list[np.ndarray],
    accessor: WaveformAccessor,
    device: torch.device,
    initial_hidden: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    padded, mask = _padded_numeric(arrays, sequences)
    query_ppg, support_ppg = _load_waveforms(data, frame, sequences, accessor)
    query_ppg = query_ppg.to(device, non_blocking=True)
    support_ppg = support_ppg.to(device, non_blocking=True)
    batch, length = mask.shape
    with _autocast(device):
        query_feature = model.encoder(query_ppg.reshape(batch * length, 1, 1250))
        query_feature = query_feature.reshape(batch, length, -1)
        support_feature = model.encoder(
            support_ppg.reshape(batch * SUPPORT_K, 1, 1250)
        ).reshape(batch, SUPPORT_K, -1)
        expanded_support = support_feature[:, None].expand(
            batch, length, SUPPORT_K, support_feature.shape[-1]
        )
        empty_query_physiology = torch.empty(
            batch * length, 0, device=device, dtype=query_feature.dtype
        )
        empty_support_physiology = torch.empty(
            batch * length, SUPPORT_K, 0,
            device=device, dtype=query_feature.dtype,
        )
        static = model.head.static_forward(
            query_feature.reshape(batch * length, -1),
            expanded_support.reshape(batch * length, SUPPORT_K, -1),
            _tensor(padded["support_bp"], device).reshape(
                batch * length, SUPPORT_K, 2
            ),
            _tensor(padded["support_population"], device).reshape(
                batch * length, SUPPORT_K, 2
            ),
            _tensor(padded["base"], device).reshape(batch * length, 2),
            _tensor(padded["event_gap"], device).reshape(
                batch * length, SUPPORT_K
            ),
            empty_query_physiology,
            empty_support_physiology,
        )
        valid = _tensor(mask, device, dtype=torch.bool)
        hidden = static["hidden"].reshape(batch, length, -1)
        temporal, final_hidden = _time_decay_gru_forward(
            model.head.temporal_core,
            hidden,
            _tensor(padded["sequence_gap"], device),
            valid,
            initial_hidden=initial_hidden,
        )
        prediction = static["prediction"].reshape(batch, length, 2)
        prediction = prediction + model.head.temporal_correction(temporal)
        pair_hidden = static["pair_hidden"].reshape(
            batch, length, SUPPORT_K, -1
        )
        direction_logits = (
            model.direction_head(pair_hidden)
            if model.direction_head is not None
            else None
        )
    return {
        "prediction": prediction,
        "pair_delta": static["pair_delta"].reshape(batch, length, SUPPORT_K, 2),
        "pair_hidden": pair_hidden,
        "range_logits": static["range_logits"].reshape(batch, length, 6),
        "target": _tensor(padded["target"], device),
        "support_bp": _tensor(padded["support_bp"], device),
        "range": _tensor(padded["range"], device, dtype=torch.long),
        "mask": valid,
        "direction_logits": direction_logits,
        "final_hidden": final_hidden,
    }


def _time_decay_gru_forward(
    core: nn.Module,
    inputs: torch.Tensor,
    gaps: torch.Tensor,
    mask: torch.Tensor,
    *,
    initial_hidden: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Round-9 GRU while exposing state for exact chunked inference."""

    if not isinstance(core, TimeDecayGRU):
        raise TypeError("Round-10 chunked execution requires TimeDecayGRU")
    batch, length, dimension = inputs.shape
    if initial_hidden is None:
        hidden = torch.zeros(
            batch, dimension, device=inputs.device, dtype=inputs.dtype
        )
    else:
        if initial_hidden.shape != (batch, dimension):
            raise ValueError("initial temporal hidden state has the wrong shape")
        hidden = initial_hidden.to(device=inputs.device, dtype=inputs.dtype)
    outputs: list[torch.Tensor] = []
    rate = torch.nn.functional.softplus(core.log_decay)
    for position in range(length):
        decayed = hidden * torch.exp(-rate * gaps[:, position : position + 1])
        proposed = core.cell(inputs[:, position], decayed)
        active = mask[:, position : position + 1]
        hidden = torch.where(active, proposed, hidden)
        outputs.append(hidden)
    return torch.stack(outputs, dim=1), hidden


def _training_sequence_batches(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    batch_size: int,
    maximum_length: int = 256,
) -> list[list[np.ndarray]]:
    """Sample one causal contiguous TBPTT segment per participant and epoch."""

    if maximum_length < 2:
        raise ValueError("maximum training sequence length must be at least two")
    sequences: list[np.ndarray] = []
    for _, group in frame.groupby("subject_uid", sort=True):
        indexes = group.sort_values(
            ["event_index", "event_id"], kind="mergesort"
        ).index.to_numpy(np.int64)
        if len(indexes) > maximum_length:
            start = int(rng.integers(0, len(indexes) - maximum_length + 1))
            indexes = indexes[start : start + maximum_length]
        sequences.append(indexes)
    sequences.sort(key=len)
    batches = [
        sequences[start : start + batch_size]
        for start in range(0, len(sequences), batch_size)
    ]
    rng.shuffle(batches)
    return batches


def _loss(output: dict[str, torch.Tensor], candidate: Round10Candidate) -> torch.Tensor:
    # Encoder/head forward passes may use CUDA bfloat16 autocast, while the
    # prepared BP targets and support values are float32.  Compute all loss
    # terms in float32 so autograd does not receive a float gradient for a
    # bfloat16 loss output.  The casts remain differentiable and preserve the
    # intended mixed-precision forward pass.
    loss_output = dict(output)
    for name in ("prediction", "pair_delta", "range_logits"):
        loss_output[name] = output[name].float()
    if output["direction_logits"] is not None:
        loss_output["direction_logits"] = output["direction_logits"].float()

    head_candidate = HeadCandidate(
        prediction_mode=candidate.prediction_mode,
        temporal_mode="gru",
        temporal_delta_penalty=candidate.temporal_delta_penalty,
    )
    total = _round9_loss(loss_output, head_candidate)
    if candidate.pair_direction_weight:
        logits = loss_output["direction_logits"]
        if logits is None:
            raise AssertionError("direction candidate has no direction logits")
        labels = (
            loss_output["target"][..., None, :] - loss_output["support_bp"] > 0
        ).to(logits.dtype)
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        ).mean((-1, -2))
        total = total + candidate.pair_direction_weight * raw[loss_output["mask"]].mean()
    return total


def _predict(
    model: Round10Model,
    data: PreparedRound10,
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    accessor: WaveformAccessor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    model.keep_encoder_statistics_frozen()
    predictions = np.zeros((len(frame), 2), dtype=np.float32)
    # Participant sequences can exceed 2,000 queries.  Evaluate them in causal
    # chunks while carrying the GRU state so the numerical result is equivalent
    # to one full chronological pass without the full-waveform memory spike.
    chunk_length = 256
    sequences = _sequence_batches(frame, batch_size=1)
    with torch.no_grad():
        for sequence_group in sequences:
            indexes = sequence_group[0]
            hidden = None
            for start in range(0, len(indexes), chunk_length):
                chunk = indexes[start : start + chunk_length]
                output = _forward_sequences(
                    model,
                    data,
                    frame,
                    arrays,
                    [chunk],
                    accessor,
                    device,
                    initial_hidden=hidden,
                )
                values = output["prediction"].float().cpu().numpy()[0, : len(chunk)]
                predictions[chunk] = values
                hidden = output["final_hidden"].detach()
    return predictions


def _scored_predictions(
    frame: pd.DataFrame,
    prediction_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> pd.DataFrame:
    prediction = prediction_norm * std + mean
    result = frame[KEYS + ["source", "target_sbp", "target_dbp"]].copy()
    result["pred_sbp"] = prediction[:, 0]
    result["pred_dbp"] = prediction[:, 1]
    return result


def train_candidate(
    *,
    prepared: Path,
    population_checkpoint: Path,
    output: Path,
    method: str,
    seed: int,
    batch_size: int = 6,
) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unknown Round-10 method {method}")
    if output.exists():
        raise FileExistsError(output)
    data = PreparedRound10(prepared)
    candidate = METHODS[method]
    frames = {
        "fit": data.queries.loc[data.queries["fold"].isin(FIT_FOLDS)].copy(),
        "early": data.queries.loc[data.queries["fold"].eq(EARLY_FOLD)].copy(),
        "selection": data.queries.loc[data.queries["fold"].eq(SELECTION_FOLD)].copy(),
    }
    participant_sets = []
    for frame in frames.values():
        frame.sort_values(["subject_uid", "event_index", "event_id"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        participant_sets.append(set(frame["subject_uid"].astype(str)))
    if any(
        participant_sets[left] & participant_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise AssertionError("Round-10 internal participant roles overlap")

    arrays = {name: data.arrays(frame) for name, frame in frames.items()}
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Round-10 candidate training requires CUDA")
    population, scaler = _load_population_checkpoint(population_checkpoint, device)
    if not (
        np.allclose(scaler["mean"], data.mean, atol=0.0, rtol=0.0)
        and np.allclose(scaler["std"], data.std, atol=0.0, rtol=0.0)
    ):
        raise AssertionError("population checkpoint and prepared scaler differ")
    model = Round10Model(population.encoder, candidate).to(device)
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    groups: list[dict[str, object]] = [
        {"params": head_parameters, "lr": 5e-4, "weight_decay": 1e-4}
    ]
    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": candidate.encoder_learning_rate,
                "weight_decay": 1e-4,
            }
        )
    optimizer = torch.optim.AdamW(groups)
    if "store_root" not in data.payload:
        raise AssertionError("Round-10 prepared cache does not record store_root")
    store_root = Path(str(data.payload["store_root"]))
    accessor = WaveformAccessor(store_root)

    rng = np.random.default_rng(seed)
    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in itertools.count(1):
        model.train()
        model.keep_encoder_statistics_frozen()
        fit_batches = _training_sequence_batches(
            frames["fit"], rng=rng, batch_size=batch_size
        )
        total_loss = 0.0
        for sequences in fit_batches:
            optimizer.zero_grad(set_to_none=True)
            output_batch = _forward_sequences(
                model,
                data,
                frames["fit"],
                arrays["fit"],
                sequences,
                accessor,
                device,
            )
            loss = _loss(output_batch, candidate)
            loss.backward()
            parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
        early_norm = _predict(
            model,
            data,
            frames["early"],
            arrays["early"],
            accessor,
            device,
            batch_size,
        )
        early = _scored_predictions(
            frames["early"], early_norm, data.mean, data.std
        )
        metrics = participant_macro_metrics(early)
        score = float(metrics["mean_mae"])
        history.append(
            {
                "epoch": epoch,
                "train_batch_loss_sum": total_loss,
                "fold3_participant_macro_mean_mae": score,
            }
        )
        print(json.dumps(history[-1]), flush=True)
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
        if stale >= 8:
            break
    if best_state is None:
        raise RuntimeError("Round-10 training produced no checkpoint")
    model.load_state_dict(best_state)
    selection_norm = _predict(
        model,
        data,
        frames["selection"],
        arrays["selection"],
        accessor,
        device,
        batch_size,
    )
    predictions = _scored_predictions(
        frames["selection"], selection_norm, data.mean, data.std
    )
    metrics = {
        scope: participant_macro_metrics(group)
        for scope, group in [("Overall", predictions)]
        + [
            (source, predictions.loc[predictions["source"].eq(source)])
            for source in sorted(predictions["source"].unique())
        ]
    }
    output.mkdir(parents=True)
    predictions.to_parquet(output / "selection_predictions.parquet", index=False)
    torch.save(
        {
            "round": 10,
            "method": method,
            "candidate": candidate.__dict__,
            "model_state": best_state,
            "target_scaler": data.payload["target_scaler"],
            "best_epoch": best_epoch,
        },
        output / "best.pt",
    )
    save_json(output / "history.json", history)
    payload = {
        "status": "complete",
        "round": 10,
        "method": method,
        "seed": seed,
        "k": SUPPORT_K,
        "split": "meta_train_internal_fold4",
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_FOLD,
        "selection_fold": SELECTION_FOLD,
        "meta_validation_used_for_training": False,
        "meta_validation_used_for_early_stopping": False,
        "meta_validation_used_for_candidate_ranking": False,
        "meta_validation_predictions_generated": False,
        "locked_test_accessed": False,
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "source_model_input": False,
        "participants": int(predictions["subject_uid"].nunique()),
        "queries": int(len(predictions)),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stop_reason": "early_stopping_patience_8",
        "candidate": candidate.__dict__,
        "trainable_encoder_parameters": model.trainable_encoder_parameters,
        "trainable_parameter_count": int(
            sum(value.numel() for value in model.parameters() if value.requires_grad)
        ),
        "population_checkpoint_sha256": file_sha256(population_checkpoint),
        "prepared_run_sha256": file_sha256(prepared / "run.json"),
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "metrics": metrics,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output / "run.json", payload)
    return payload


def build_internal_report(
    *, runs: dict[str, Path], reference: str, output: Path, expected_seed: int
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    canonical: pd.DataFrame | None = None
    for setting, root in runs.items():
        payload = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("round") != 10:
            raise AssertionError(f"{setting} is not a complete Round-10 run")
        if payload.get("split") != "meta_train_internal_fold4":
            raise AssertionError(f"{setting} has the wrong selection split")
        if payload.get("seed") != expected_seed:
            raise AssertionError(f"{setting} has the wrong seed")
        for key in (
            "meta_validation_used_for_training",
            "meta_validation_used_for_early_stopping",
            "meta_validation_used_for_candidate_ranking",
            "meta_validation_predictions_generated",
        ):
            if payload.get(key) is not False:
                raise AssertionError(f"{setting} has unsafe flag {key}")
        if payload.get("locked_test_accessed") is not False:
            raise AssertionError(f"{setting} accessed locked test")
        predictions = pd.read_parquet(root / "selection_predictions.parquet")
        prediction_frames.append(predictions.assign(Setting=setting))
        check = predictions[KEYS + ["target_sbp", "target_dbp"]].sort_values(KEYS)
        check.reset_index(drop=True, inplace=True)
        if canonical is None:
            canonical = check
        elif not canonical.equals(check):
            raise AssertionError(f"{setting} has different query keys or targets")
        for scope, group in [("Overall", predictions)] + [
            (source, predictions.loc[predictions["source"].eq(source)])
            for source in sorted(predictions["source"].unique())
        ]:
            metric = participant_macro_metrics(group)
            records.append(
                {
                    "Setting": setting,
                    "Scope": scope,
                    "N participants": int(group["subject_uid"].nunique()),
                    "N queries": int(len(group)),
                    "SBP participant-macro MAE": metric["sbp_mae"],
                    "DBP participant-macro MAE": metric["dbp_mae"],
                    "Mean participant-macro MAE": metric["mean_mae"],
                }
            )
    table = pd.DataFrame(records)
    settings = list(runs)
    diagnostics = _diagnostic_rows(
        pd.concat(prediction_frames, ignore_index=True, sort=False), settings
    )
    if reference not in runs:
        raise KeyError(reference)
    overall = table.loc[table["Scope"].eq("Overall")].sort_values(
        ["Mean participant-macro MAE", "Setting"], kind="mergesort"
    )
    winner = str(overall.iloc[0]["Setting"])
    ref = table.loc[table["Setting"].eq(reference)].set_index("Scope")
    win = table.loc[table["Setting"].eq(winner)].set_index("Scope")
    gains = {
        scope: float(
            ref.loc[scope, "Mean participant-macro MAE"]
            - win.loc[scope, "Mean participant-macro MAE"]
        )
        for scope in ("Overall", "MIMIC", "VitalDB")
    }
    passes = gains["Overall"] >= 0.15 and gains["MIMIC"] > 0 and gains["VitalDB"] > 0
    comparison_rows = []
    for setting in settings:
        candidate = table.loc[table["Setting"].eq(setting)].set_index("Scope")
        for scope in ("Overall", "MIMIC", "VitalDB"):
            comparison_rows.append(
                {
                    "Setting": setting,
                    "Scope": scope,
                    "Reference mean participant-macro MAE": float(
                        ref.loc[scope, "Mean participant-macro MAE"]
                    ),
                    "Candidate mean participant-macro MAE": float(
                        candidate.loc[scope, "Mean participant-macro MAE"]
                    ),
                    "Candidate minus reference": float(
                        candidate.loc[scope, "Mean participant-macro MAE"]
                        - ref.loc[scope, "Mean participant-macro MAE"]
                    ),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "participant_macro_internal.csv", index=False)
    diagnostics.to_csv(output / "pooled_diagnostics_internal.csv", index=False)
    comparison.to_csv(output / "comparison_vs_reference_internal.csv", index=False)
    summary = {
        "status": "complete",
        "round": 10,
        "seed": expected_seed,
        "split": "meta_train_internal_fold4",
        "meta_validation_used_for_candidate_ranking": False,
        "locked_test_accessed": False,
        "reference": reference,
        "winner": winner,
        "gain_vs_reference": gains,
        "passes_internal_gate": passes,
        "candidate_count": len(runs),
    }
    save_json(output / "selection.json", summary)
    lines = [
        "# Round-10 partial end-to-end internal screen",
        "",
        "Folds 0--2 fit the models, fold 3 controls patience-8 early stopping, and fold 4 ranks candidates. Meta-validation and the locked meta-test were not accessed.",
        "",
        _markdown_table(table.sort_values(["Scope", "Mean participant-macro MAE"])),
        "",
        "## Change versus the frozen-encoder reference",
        "",
        "Negative candidate-minus-reference values are better.",
        "",
        _markdown_table(comparison),
        "",
        "## Event-pooled diagnostics",
        "",
        "Participant-macro MAE is primary. AAMI/BHS entries are retrospective numerical screens only.",
        "",
        _markdown_table(diagnostics),
        "",
        f"Internal winner: **{winner}**.",
        f"Internal promotion gate passed: **{passes}**.",
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
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--store-root", type=Path, required=True)
    prepare.add_argument("--folds", type=Path, required=True)
    prepare.add_argument("--population-checkpoint", type=Path, required=True)
    prepare.add_argument("--qgh-checkpoint", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--batch-size", type=int, default=1024)
    train = commands.add_parser("train")
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--population-checkpoint", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--method", choices=sorted(METHODS), required=True)
    train.add_argument("--seed", type=int, default=20260824)
    train.add_argument("--batch-size", type=int, default=6)
    report = commands.add_parser("report")
    report.add_argument("--run", action="append", required=True)
    report.add_argument("--reference", required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--expected-seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_round10(
            store_root=args.store_root,
            folds_path=args.folds,
            population_checkpoint=args.population_checkpoint,
            qgh_checkpoint=args.qgh_checkpoint,
            output=args.output,
            batch_size=args.batch_size,
        )
    elif args.command == "train":
        result = train_candidate(
            prepared=args.prepared,
            population_checkpoint=args.population_checkpoint,
            output=args.output,
            method=args.method,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    else:
        result = build_internal_report(
            runs=_parse_runs(args.run),
            reference=args.reference,
            output=args.output,
            expected_seed=args.expected_seed,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
