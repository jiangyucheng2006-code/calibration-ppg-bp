"""Round-8 calibration-relative and causal temporal BP correction models.

The module consumes participant-disjoint, cross-fitted Quality Gate + Huber
predictions.  Query BP is used only as a meta-train loss target or a
meta-validation score.  Model inputs contain only fixed-first support PPG/BP,
the current query PPG representation, the frozen base prediction, and causal
event history.  The locked meta-test is never read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .phase6e_residual import KEYS, _validate_frames
from .train import _load_population_checkpoint
from .training import (
    WaveformAccessor,
    load_store_metadata,
    participant_macro_metrics,
    save_json,
    seed_everything,
)


SUPPORT_K = 5
PHYSIOLOGY_NAMES = (
    "robust_range",
    "first_diff_rms",
    "second_diff_rms",
    "skewness",
    "kurtosis",
    "turning_fraction",
    "high_excursion_fraction",
    "lag80_correlation",
)
METHODS = {
    "pair_delta": (False, False, False),
    "pair_delta_temporal": (True, False, False),
    "pair_delta_range": (False, True, False),
    "pair_delta_temporal_range": (True, True, False),
    "pair_delta_temporal_range_physiology": (True, True, True),
}


def _physiology_features(waveforms: torch.Tensor) -> torch.Tensor:
    """Return generic, target-free 10-second PPG morphology summaries."""

    x = waveforms.squeeze(1)
    x = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)
    q05 = torch.quantile(x, 0.05, dim=1)
    q95 = torch.quantile(x, 0.95, dim=1)
    d1 = torch.diff(x, dim=1)
    d2 = torch.diff(d1, dim=1)
    skew = torch.mean(x**3, dim=1)
    kurt = torch.mean(x**4, dim=1) - 3.0
    turning = (d1[:, 1:] * d1[:, :-1] < 0).float().mean(1)
    excursion = (x.abs() >= 2.5).float().mean(1)
    lag = 80
    lag_corr = torch.mean(x[:, lag:] * x[:, :-lag], dim=1)
    return torch.stack(
        [
            q95 - q05,
            torch.sqrt(torch.mean(d1**2, dim=1)),
            torch.sqrt(torch.mean(d2**2, dim=1)),
            skew,
            kurt,
            turning,
            excursion,
            lag_corr,
        ],
        dim=1,
    )


def _assert_query_tables(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    _validate_frames(train, validation)
    if set(train["k"].unique()) != {SUPPORT_K}:
        raise AssertionError("Round-8 training requires K=5 OOF queries")
    if set(validation["k"].unique()) != {SUPPORT_K}:
        raise AssertionError("Round-8 validation requires K=5 queries")
    if "fold" not in train or set(train["fold"].unique()) != set(range(5)):
        raise AssertionError("Round-8 OOF table requires all five cross-fit folds")


def _encode_rows(
    metadata: pd.DataFrame,
    *,
    accessor: WaveformAccessor,
    population: nn.Module,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
    batch_size: int,
    include_embeddings: bool,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    embeddings: list[np.ndarray] = []
    physiology: list[np.ndarray] = []
    population_bp: list[np.ndarray] = []
    population.eval()
    with torch.no_grad():
        for start in range(0, len(metadata), batch_size):
            part = metadata.iloc[start : start + batch_size]
            batch = torch.stack(
                [
                    accessor.get(str(row.waveform_file), int(row.waveform_row))
                    for row in part.itertuples()
                ]
            ).to(device)
            physiology.append(_physiology_features(batch).cpu().numpy().astype(np.float32))
            if include_embeddings:
                encoded = population.encoder(batch)
                embeddings.append(encoded.cpu().numpy().astype(np.float32))
                standardized = population.predict_from_features(encoded).cpu().numpy()
                population_bp.append(
                    (standardized * target_std + target_mean).astype(np.float32)
                )
    embedding_array = np.concatenate(embeddings) if embeddings else None
    physiology_array = np.concatenate(physiology)
    population_array = np.concatenate(population_bp) if population_bp else None
    return embedding_array, physiology_array, population_array


def prepare_round8(
    *,
    store_root: Path,
    population_checkpoint: Path,
    train_features: Path,
    validation_features: Path,
    query_embeddings: Path,
    output: Path,
    batch_size: int = 1024,
) -> dict[str, object]:
    train = pd.read_parquet(train_features)
    validation = pd.read_parquet(validation_features)
    _assert_query_tables(train, validation)
    queries = pd.concat(
        [train.assign(round8_role="train"), validation.assign(round8_role="validation")],
        ignore_index=True,
        sort=False,
    )
    if queries.duplicated(KEYS).any():
        raise AssertionError("Round-8 combined query table has duplicate keys")

    embedding_frame = pd.read_parquet(query_embeddings)
    embedding_columns = [c for c in embedding_frame if c.startswith("embedding_")]
    if not embedding_columns:
        raise ValueError("query embedding table has no embedding columns")
    embedding_keys = embedding_frame[["subject_uid", "event_id"]].copy()
    if embedding_keys.duplicated().any():
        raise AssertionError("query embedding table has duplicate event keys")
    query_index = queries[["subject_uid", "event_id"]].merge(
        embedding_keys.assign(query_embedding_row=np.arange(len(embedding_frame))),
        on=["subject_uid", "event_id"],
        validate="one_to_one",
    )
    if len(query_index) != len(queries):
        raise AssertionError("query embeddings do not cover every Round-8 query")
    queries = queries.merge(
        query_index,
        on=["subject_uid", "event_id"],
        validate="one_to_one",
    )

    metadata = load_store_metadata(store_root, "development")
    participants = set(queries["subject_uid"].astype(str))
    selected = metadata.loc[metadata["subject_uid"].astype(str).isin(participants)].copy()
    support = (
        selected.loc[selected["event_index"].le(SUPPORT_K)]
        .sort_values(["subject_uid", "event_index", "event_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    counts = support.groupby("subject_uid").size()
    if not counts.eq(SUPPORT_K).all() or set(counts.index.astype(str)) != participants:
        raise AssertionError("every Round-8 participant must have exactly five fixed supports")
    support["support_position"] = support.groupby("subject_uid").cumcount()
    support_subjects = sorted(participants)
    support_row_map = {subject: row for row, subject in enumerate(support_subjects)}
    queries["support_row"] = queries["subject_uid"].astype(str).map(support_row_map)
    if queries["support_row"].isna().any():
        raise AssertionError("support row mapping is incomplete")
    queries["support_row"] = queries["support_row"].astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    population, scaler = _load_population_checkpoint(population_checkpoint, device)
    population.to(device).eval()
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

    ordered_support = support.sort_values(
        ["subject_uid", "support_position"], kind="mergesort"
    ).reset_index(drop=True)
    if not ordered_support.equals(support.reset_index(drop=True)):
        raise AssertionError("support ordering changed unexpectedly")
    n_participants = len(support_subjects)
    support_embeddings = support_embeddings.reshape(n_participants, SUPPORT_K, -1)
    support_physiology = support_physiology_flat.reshape(
        n_participants, SUPPORT_K, -1
    )
    support_population = support_population_flat.reshape(n_participants, SUPPORT_K, 2)
    support_bp = support[["sbp", "dbp"]].to_numpy(np.float32).reshape(
        n_participants, SUPPORT_K, 2
    )

    query_meta = embedding_frame[["subject_uid", "event_id"]].copy()
    query_meta["query_embedding_row"] = np.arange(len(query_meta), dtype=np.int64)
    query_meta = query_meta.merge(
        metadata[
            ["subject_uid", "event_id", "waveform_file", "waveform_row"]
        ],
        on=["subject_uid", "event_id"],
        validate="one_to_one",
    ).sort_values("query_embedding_row", kind="mergesort").reset_index(drop=True)
    if not query_meta["query_embedding_row"].equals(
        pd.Series(np.arange(len(query_meta), dtype=np.int64))
    ):
        raise AssertionError("query waveform rows no longer align with embeddings")
    _, query_physiology, _ = _encode_rows(
        query_meta,
        accessor=accessor,
        population=population,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        batch_size=batch_size,
        include_embeddings=False,
    )

    output.mkdir(parents=True, exist_ok=False)
    np.save(
        output / "query_embeddings.npy",
        embedding_frame[embedding_columns].to_numpy(np.float32),
    )
    np.save(output / "query_physiology.npy", query_physiology)
    np.save(output / "support_embeddings.npy", support_embeddings)
    np.save(output / "support_physiology.npy", support_physiology)
    np.save(output / "support_population_bp.npy", support_population)
    np.save(output / "support_bp.npy", support_bp)
    keep = list(dict.fromkeys(list(queries.columns)))
    queries[keep].to_parquet(output / "queries.parquet", index=False)
    support[
        ["subject_uid", "event_id", "source", "split", "event_index", "support_position"]
    ].assign(
        support_row=lambda frame: frame["subject_uid"].astype(str).map(support_row_map)
    ).to_parquet(output / "support_index.parquet", index=False)

    payload = {
        "status": "complete",
        "split": "development_only",
        "locked_test_accessed": False,
        "support_policy": "fixed_first",
        "k": SUPPORT_K,
        "participants": n_participants,
        "train_queries": int(train.shape[0]),
        "validation_queries": int(validation.shape[0]),
        "embedding_dimension": int(len(embedding_columns)),
        "physiology_features": list(PHYSIOLOGY_NAMES),
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "source_model_input": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output / "run.json", payload)
    return payload


class TimeDecayGRU(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(dimension, dimension)
        self.log_decay = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self, inputs: torch.Tensor, gaps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, length, dimension = inputs.shape
        hidden = torch.zeros(batch, dimension, device=inputs.device, dtype=inputs.dtype)
        outputs = []
        rate = torch.nn.functional.softplus(self.log_decay)
        for position in range(length):
            decayed = hidden * torch.exp(-rate * gaps[:, position : position + 1])
            proposed = self.cell(inputs[:, position], decayed)
            active = mask[:, position : position + 1]
            hidden = torch.where(active, proposed, hidden)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


class CalibrationRelativeModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        temporal: bool,
        range_auxiliary: bool,
        physiology_dim: int = 0,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.temporal = temporal
        self.range_auxiliary = range_auxiliary
        self.physiology_dim = physiology_dim
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        pair_dim = hidden_dim * 4 + 2 + 2 + 1
        if physiology_dim:
            pair_dim += physiology_dim * 4
        self.pair = nn.Sequential(
            nn.Linear(pair_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
        )
        self.pair_delta = nn.Linear(hidden_dim, 2)
        self.attention = nn.Linear(hidden_dim, 1)
        static_dim = hidden_dim * 2 + 6
        self.static = nn.Sequential(
            nn.Linear(static_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.static_correction = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.static_correction.weight)
        nn.init.zeros_(self.static_correction.bias)
        if temporal:
            self.temporal_core = TimeDecayGRU(hidden_dim)
            self.temporal_correction = nn.Linear(hidden_dim, 2)
            nn.init.zeros_(self.temporal_correction.weight)
            nn.init.zeros_(self.temporal_correction.bias)
        if range_auxiliary:
            self.range_head = nn.Linear(hidden_dim, 6)

    def static_forward(
        self,
        query_embedding: torch.Tensor,
        support_embedding: torch.Tensor,
        support_bp_norm: torch.Tensor,
        support_population_norm: torch.Tensor,
        base_norm: torch.Tensor,
        event_gap: torch.Tensor,
        query_physiology: torch.Tensor | None = None,
        support_physiology: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        query = self.project(query_embedding)
        support = self.project(support_embedding)
        expanded = query[:, None, :].expand_as(support)
        pair_inputs = [
            expanded,
            support,
            expanded - support,
            (expanded - support).abs(),
            support_bp_norm,
            base_norm[:, None, :] - support_population_norm,
            event_gap[..., None],
        ]
        if self.physiology_dim:
            if query_physiology is None or support_physiology is None:
                raise ValueError("physiology-enabled model requires physiology inputs")
            query_phys = query_physiology[:, None, :].expand_as(support_physiology)
            pair_inputs.extend(
                [
                    query_phys,
                    support_physiology,
                    query_phys - support_physiology,
                    (query_phys - support_physiology).abs(),
                ]
            )
        pair_hidden = self.pair(torch.cat(pair_inputs, dim=-1))
        pair_delta = self.pair_delta(pair_hidden)
        weights = torch.softmax(self.attention(pair_hidden).squeeze(-1), dim=1)
        pair_prediction = torch.sum(
            weights[..., None] * (support_bp_norm + pair_delta), dim=1
        )
        context = torch.sum(weights[..., None] * pair_hidden, dim=1)
        static_hidden = self.static(
            torch.cat(
                [
                    query,
                    context,
                    base_norm,
                    pair_prediction,
                    pair_prediction - base_norm,
                ],
                dim=-1,
            )
        )
        static_prediction = base_norm + self.static_correction(static_hidden)
        range_logits = self.range_head(static_hidden) if self.range_auxiliary else None
        return static_prediction, pair_delta, static_hidden, range_logits


def _range_labels(target: np.ndarray, support: np.ndarray) -> np.ndarray:
    low = support.min(axis=1)
    high = support.max(axis=1)
    labels = np.ones_like(target, dtype=np.int64)
    labels[target < low] = 0
    labels[target > high] = 2
    return labels


class PreparedRound8:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.queries = pd.read_parquet(root / "queries.parquet")
        self.query_embeddings = np.load(root / "query_embeddings.npy", mmap_mode="r")
        self.query_physiology = np.load(root / "query_physiology.npy", mmap_mode="r")
        self.support_embeddings = np.load(root / "support_embeddings.npy", mmap_mode="r")
        self.support_physiology = np.load(root / "support_physiology.npy", mmap_mode="r")
        self.support_population = np.load(root / "support_population_bp.npy", mmap_mode="r")
        self.support_bp = np.load(root / "support_bp.npy", mmap_mode="r")
        if self.queries.duplicated(KEYS).any():
            raise AssertionError("prepared Round-8 queries contain duplicate keys")
        if self.queries["query_embedding_row"].max() >= len(self.query_embeddings):
            raise AssertionError("query embedding pointer exceeds array")
        if self.queries["support_row"].max() >= len(self.support_embeddings):
            raise AssertionError("support pointer exceeds array")

    def arrays(
        self,
        frame: pd.DataFrame,
        *,
        mean: np.ndarray,
        std: np.ndarray,
        physiology_mean: np.ndarray,
        physiology_std: np.ndarray,
    ) -> dict[str, np.ndarray]:
        qrow = frame["query_embedding_row"].to_numpy(np.int64)
        srow = frame["support_row"].to_numpy(np.int64)
        event_index = frame["event_index"].to_numpy(np.float32)
        gaps = np.maximum(
            event_index[:, None] - np.arange(1, SUPPORT_K + 1, dtype=np.float32)[None],
            0.0,
        )
        return {
            "query": np.asarray(self.query_embeddings[qrow], dtype=np.float32),
            "support": np.asarray(self.support_embeddings[srow], dtype=np.float32),
            "support_bp": (np.asarray(self.support_bp[srow]) - mean) / std,
            "support_population": (np.asarray(self.support_population[srow]) - mean) / std,
            "base": (frame[["pred_sbp", "pred_dbp"]].to_numpy(np.float32) - mean) / std,
            "target": (frame[["target_sbp", "target_dbp"]].to_numpy(np.float32) - mean) / std,
            "event_gap": np.log1p(gaps) / np.float32(math.log1p(300.0)),
            "sequence_gap": np.log1p(
                frame.groupby("subject_uid", sort=False)["event_index"]
                .diff()
                .fillna(frame["events_since_calibration"])
                .clip(lower=0)
                .to_numpy(np.float32)
            )
            / np.float32(math.log1p(300.0)),
            "query_physiology": (
                np.asarray(self.query_physiology[qrow]) - physiology_mean
            )
            / physiology_std,
            "support_physiology": (
                np.asarray(self.support_physiology[srow]) - physiology_mean
            )
            / physiology_std,
            "range": _range_labels(
                frame[["target_sbp", "target_dbp"]].to_numpy(np.float32),
                np.asarray(self.support_bp[srow]),
            ),
        }


def _tensor(array: np.ndarray, device: torch.device, dtype=None) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(array))
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value.to(device)


def _losses(
    *,
    prediction: torch.Tensor,
    pair_delta: torch.Tensor,
    range_logits: torch.Tensor | None,
    target: torch.Tensor,
    support_bp: torch.Tensor,
    range_labels: torch.Tensor,
    range_auxiliary: bool,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    main = torch.nn.functional.huber_loss(
        prediction, target, reduction="none", delta=0.5
    ).mean(-1)
    pair_target = target[..., None, :] - support_bp
    pair = torch.nn.functional.huber_loss(
        pair_delta, pair_target, reduction="none", delta=0.5
    ).mean((-1, -2))
    loss = main + 0.25 * pair
    if range_auxiliary:
        assert range_logits is not None
        logits = range_logits.reshape(*range_logits.shape[:-1], 2, 3)
        ce = 0.5 * (
            torch.nn.functional.cross_entropy(
                logits[..., 0, :].reshape(-1, 3),
                range_labels[..., 0].reshape(-1),
                reduction="none",
            )
            + torch.nn.functional.cross_entropy(
                logits[..., 1, :].reshape(-1, 3),
                range_labels[..., 1].reshape(-1),
                reduction="none",
            )
        )
        loss = loss + 0.10 * ce.reshape(loss.shape)
    if mask is not None:
        loss = loss * mask
        participant_loss = loss.sum(1) / mask.sum(1).clamp_min(1)
        return participant_loss.mean()
    return loss.mean()


def _static_batches(frame: pd.DataFrame, seed: int, epoch: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + epoch)
    selected = []
    for _, group in frame.groupby("subject_uid", sort=True):
        indexes = group.index.to_numpy(np.int64)
        size = min(32, len(indexes))
        selected.append(rng.choice(indexes, size=size, replace=False))
    order = np.concatenate(selected)
    rng.shuffle(order)
    return [order[start : start + 1024] for start in range(0, len(order), 1024)]


def _sequence_batches(frame: pd.DataFrame, batch_size: int = 12) -> list[list[np.ndarray]]:
    sequences = [
        group.sort_values(["event_index", "event_id"], kind="mergesort").index.to_numpy(np.int64)
        for _, group in frame.groupby("subject_uid", sort=True)
    ]
    sequences.sort(key=len)
    return [sequences[start : start + batch_size] for start in range(0, len(sequences), batch_size)]


def _forward_static_arrays(
    model: CalibrationRelativeModel,
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    device: torch.device,
    use_physiology: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return model.static_forward(
        _tensor(arrays["query"][indexes], device),
        _tensor(arrays["support"][indexes], device),
        _tensor(arrays["support_bp"][indexes], device),
        _tensor(arrays["support_population"][indexes], device),
        _tensor(arrays["base"][indexes], device),
        _tensor(arrays["event_gap"][indexes], device),
        _tensor(arrays["query_physiology"][indexes], device)
        if use_physiology
        else None,
        _tensor(arrays["support_physiology"][indexes], device)
        if use_physiology
        else None,
    )


def _forward_sequence_batch(
    model: CalibrationRelativeModel,
    arrays: dict[str, np.ndarray],
    sequences: list[np.ndarray],
    device: torch.device,
    use_physiology: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(map(len, sequences))
    width = arrays["query"].shape[1]
    physiology_width = arrays["query_physiology"].shape[1]
    batch = len(sequences)
    shapes = {
        "query": (batch, length, width),
        "support": (batch, length, SUPPORT_K, width),
        "support_bp": (batch, length, SUPPORT_K, 2),
        "support_population": (batch, length, SUPPORT_K, 2),
        "base": (batch, length, 2),
        "target": (batch, length, 2),
        "event_gap": (batch, length, SUPPORT_K),
        "sequence_gap": (batch, length),
        "range": (batch, length, 2),
        "query_physiology": (batch, length, physiology_width),
        "support_physiology": (batch, length, SUPPORT_K, physiology_width),
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
    flat = np.arange(batch * length)
    static, pair, hidden, logits = model.static_forward(
        _tensor(padded["query"].reshape(batch * length, width), device),
        _tensor(padded["support"].reshape(batch * length, SUPPORT_K, width), device),
        _tensor(padded["support_bp"].reshape(batch * length, SUPPORT_K, 2), device),
        _tensor(padded["support_population"].reshape(batch * length, SUPPORT_K, 2), device),
        _tensor(padded["base"].reshape(batch * length, 2), device),
        _tensor(padded["event_gap"].reshape(batch * length, SUPPORT_K), device),
        _tensor(padded["query_physiology"].reshape(batch * length, physiology_width), device)
        if use_physiology
        else None,
        _tensor(
            padded["support_physiology"].reshape(
                batch * length, SUPPORT_K, physiology_width
            ),
            device,
        )
        if use_physiology
        else None,
    )
    del flat
    hidden = hidden.reshape(batch, length, -1)
    mask_tensor = _tensor(mask, device, dtype=torch.bool)
    temporal_hidden = model.temporal_core(
        hidden, _tensor(padded["sequence_gap"], device), mask_tensor
    )
    prediction = static.reshape(batch, length, 2) + model.temporal_correction(
        temporal_hidden
    )
    logits = None if logits is None else logits.reshape(batch, length, 6)
    return (
        prediction,
        pair.reshape(batch, length, SUPPORT_K, 2),
        logits,
        _tensor(padded["target"], device),
        _tensor(padded["support_bp"], device),
        _tensor(padded["range"], device, dtype=torch.long),
        mask_tensor.float(),
    )


def _predict(
    model: CalibrationRelativeModel,
    arrays: dict[str, np.ndarray],
    frame: pd.DataFrame,
    device: torch.device,
    *,
    temporal: bool,
    use_physiology: bool,
) -> np.ndarray:
    model.eval()
    output = np.zeros((len(frame), 2), dtype=np.float32)
    with torch.no_grad():
        if temporal:
            for sequences in _sequence_batches(frame, batch_size=12):
                prediction, _, _, _, _, _, mask = _forward_sequence_batch(
                    model, arrays, sequences, device, use_physiology
                )
                values = prediction.cpu().numpy()
                for row, indexes in enumerate(sequences):
                    output[indexes] = values[row, : len(indexes)]
        else:
            indexes = np.arange(len(frame))
            for start in range(0, len(indexes), 4096):
                chosen = indexes[start : start + 4096]
                prediction, _, _, _ = _forward_static_arrays(
                    model, arrays, chosen, device, use_physiology
                )
                output[chosen] = prediction.cpu().numpy()
    return output


def _mean_mae(frame: pd.DataFrame, prediction_mm_hg: np.ndarray) -> float:
    scored = frame[KEYS + ["target_sbp", "target_dbp"]].copy()
    scored["pred_sbp"] = prediction_mm_hg[:, 0]
    scored["pred_dbp"] = prediction_mm_hg[:, 1]
    return float(participant_macro_metrics(scored)["mean_mae"])


def train_round8(
    *,
    prepared: Path,
    output: Path,
    method: str,
    seed: int,
) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unknown Round-8 method {method}")
    temporal, range_auxiliary, use_physiology = METHODS[method]
    data = PreparedRound8(prepared)
    train = data.queries.loc[data.queries["round8_role"].eq("train")].copy()
    validation = data.queries.loc[data.queries["round8_role"].eq("validation")].copy()
    train = train.sort_values(["subject_uid", "event_index", "event_id"]).reset_index(drop=True)
    validation = validation.sort_values(["subject_uid", "event_index", "event_id"]).reset_index(drop=True)
    fit = train.loc[train["fold"].ne(4)].reset_index(drop=True)
    internal = train.loc[train["fold"].eq(4)].reset_index(drop=True)
    if set(fit["subject_uid"].astype(str)) & set(internal["subject_uid"].astype(str)):
        raise AssertionError("Round-8 fit and internal participants overlap")
    if set(train["subject_uid"].astype(str)) & set(validation["subject_uid"].astype(str)):
        raise AssertionError("Round-8 train and validation participants overlap")

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

    arrays_fit = data.arrays(
        fit,
        mean=target_mean,
        std=target_std,
        physiology_mean=physiology_mean,
        physiology_std=physiology_std,
    )
    arrays_internal = data.arrays(
        internal,
        mean=target_mean,
        std=target_std,
        physiology_mean=physiology_mean,
        physiology_std=physiology_std,
    )
    arrays_validation = data.arrays(
        validation,
        mean=target_mean,
        std=target_std,
        physiology_mean=physiology_mean,
        physiology_std=physiology_std,
    )

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Round-8 training requires CUDA")
    model = CalibrationRelativeModel(
        data.query_embeddings.shape[1],
        temporal=temporal,
        range_auxiliary=range_auxiliary,
        physiology_dim=data.query_physiology.shape[1] if use_physiology else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)

    for epoch in itertools.count(1):
        model.train()
        total = 0.0
        batches = (
            _sequence_batches(fit, batch_size=12)
            if temporal
            else _static_batches(fit, seed, epoch)
        )
        if temporal:
            rng.shuffle(batches)
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            if temporal:
                prediction, pair, logits, target, support_bp, labels, mask = _forward_sequence_batch(
                    model, arrays_fit, batch, device, use_physiology
                )
                loss = _losses(
                    prediction=prediction,
                    pair_delta=pair,
                    range_logits=logits,
                    target=target,
                    support_bp=support_bp,
                    range_labels=labels,
                    range_auxiliary=range_auxiliary,
                    mask=mask,
                )
            else:
                indexes = batch
                prediction, pair, _, logits = _forward_static_arrays(
                    model, arrays_fit, indexes, device, use_physiology
                )
                loss = _losses(
                    prediction=prediction,
                    pair_delta=pair,
                    range_logits=logits,
                    target=_tensor(arrays_fit["target"][indexes], device),
                    support_bp=_tensor(arrays_fit["support_bp"][indexes], device),
                    range_labels=_tensor(
                        arrays_fit["range"][indexes], device, dtype=torch.long
                    ),
                    range_auxiliary=range_auxiliary,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())

        internal_norm = _predict(
            model,
            arrays_internal,
            internal,
            device,
            temporal=temporal,
            use_physiology=use_physiology,
        )
        internal_mm_hg = internal_norm * target_std + target_mean
        score = _mean_mae(internal, internal_mm_hg)
        history.append(
            {
                "epoch": float(epoch),
                "train_batch_loss_sum": total,
                "internal_participant_macro_mean_mae": score,
            }
        )
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
        raise RuntimeError("Round-8 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    prediction_norm = _predict(
        model,
        arrays_validation,
        validation,
        device,
        temporal=temporal,
        use_physiology=use_physiology,
    )
    prediction_mm_hg = prediction_norm * target_std + target_mean
    predictions = validation[
        KEYS + ["source", "target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]
    ].copy()
    predictions.rename(
        columns={"pred_sbp": "base_pred_sbp", "pred_dbp": "base_pred_dbp"},
        inplace=True,
    )
    predictions["pred_sbp"] = prediction_mm_hg[:, 0]
    predictions["pred_dbp"] = prediction_mm_hg[:, 1]
    metrics = {
        scope: participant_macro_metrics(group)
        for scope, group in [("Overall", predictions)]
        + [
            (source, predictions.loc[predictions["source"].eq(source)])
            for source in sorted(predictions["source"].unique())
        ]
    }
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    torch.save(
        {
            "model_state": best_state,
            "method": method,
            "target_mean": target_mean,
            "target_std": target_std,
            "physiology_mean": physiology_mean,
            "physiology_std": physiology_std,
        },
        output / "best.pt",
    )
    save_json(output / "history.json", history)
    payload = {
        "status": "complete",
        "method": method,
        "seed": seed,
        "split": "meta_validation",
        "locked_test_accessed": False,
        "support_policy": "fixed_first",
        "k": SUPPORT_K,
        "query_policy": "event_6_and_later",
        "training_labels": "meta_train_crossfit_oof",
        "selection_split": "meta_train_crossfit_fold_4",
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "source_model_input": False,
        "temporal": temporal,
        "range_auxiliary": range_auxiliary,
        "physiology_features": list(PHYSIOLOGY_NAMES) if use_physiology else [],
        "participants": int(predictions["subject_uid"].nunique()),
        "queries": int(len(predictions)),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stop_reason": "early_stopping_patience_8",
        "metrics": metrics,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output / "run.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--store-root", type=Path, required=True)
    prepare.add_argument("--population-checkpoint", type=Path, required=True)
    prepare.add_argument("--train-features", type=Path, required=True)
    prepare.add_argument("--validation-features", type=Path, required=True)
    prepare.add_argument("--query-embeddings", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--method", choices=sorted(METHODS), required=True)
    train.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_round8(
            store_root=args.store_root,
            population_checkpoint=args.population_checkpoint,
            train_features=args.train_features,
            validation_features=args.validation_features,
            query_embeddings=args.query_embeddings,
            output=args.output,
        )
    else:
        result = train_round8(
            prepared=args.prepared,
            output=args.output,
            method=args.method,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
