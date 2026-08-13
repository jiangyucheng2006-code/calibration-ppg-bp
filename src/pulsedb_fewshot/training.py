"""Shared data, metrics, and training utilities for calibrated PPG-BP models."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


KS = (1, 2, 3, 5)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".sh", ".toml", ".txt"}
        and "__pycache__" not in path.parts
        and not any(part.endswith(".egg-info") for part in path.parts)
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_store_metadata(store_root: Path, prefix: str) -> pd.DataFrame:
    paths = sorted(store_root.glob(f"{prefix}_metadata_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no metadata shards for {prefix} under {store_root}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["metadata_file"] = str(path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result["event_id"].duplicated().any():
        raise AssertionError(f"duplicate event IDs in {prefix} store")
    return result


def fit_target_scaler(metadata: pd.DataFrame) -> dict[str, list[float]]:
    train = metadata.loc[metadata["split"].eq("meta_train"), ["sbp", "dbp"]]
    if train.empty:
        raise ValueError("meta-train target set is empty")
    mean = train.mean().to_numpy(dtype=np.float32)
    standard_deviation = train.std(ddof=0).to_numpy(dtype=np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(standard_deviation).all():
        raise ValueError("nonfinite target scaler")
    if not (standard_deviation > 0).all():
        raise ValueError("target standard deviation must be positive")
    return {"mean": mean.tolist(), "std": standard_deviation.tolist()}


def participant_macro_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    required = {"subject_uid", "target_sbp", "target_dbp", "pred_sbp", "pred_dbp"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    frame = predictions.copy()
    for target in ("sbp", "dbp"):
        error = frame[f"pred_{target}"] - frame[f"target_{target}"]
        frame[f"abs_error_{target}"] = error.abs()
        frame[f"sq_error_{target}"] = error.pow(2)
        frame[f"error_{target}"] = error
    participant = frame.groupby("subject_uid", as_index=False).agg(
        sbp_mae=("abs_error_sbp", "mean"),
        dbp_mae=("abs_error_dbp", "mean"),
        sbp_mse=("sq_error_sbp", "mean"),
        dbp_mse=("sq_error_dbp", "mean"),
        sbp_bias=("error_sbp", "mean"),
        dbp_bias=("error_dbp", "mean"),
        n_events=("event_id", "size"),
    )
    return {
        "n_participants": int(len(participant)),
        "n_events": int(len(frame)),
        "sbp_mae": float(participant["sbp_mae"].mean()),
        "dbp_mae": float(participant["dbp_mae"].mean()),
        "mean_mae": float((participant["sbp_mae"] + participant["dbp_mae"]).mean() / 2),
        "sbp_rmse": float(np.sqrt(participant["sbp_mse"]).mean()),
        "dbp_rmse": float(np.sqrt(participant["dbp_mse"]).mean()),
        "sbp_bias": float(participant["sbp_bias"].mean()),
        "dbp_bias": float(participant["dbp_bias"].mean()),
    }


class WaveformAccessor:
    def __init__(self, store_root: Path) -> None:
        self.store_root = store_root
        self._arrays: dict[str, np.ndarray] = {}

    def get(self, waveform_file: str, row: int) -> torch.Tensor:
        if waveform_file not in self._arrays:
            self._arrays[waveform_file] = np.load(
                self.store_root / waveform_file, mmap_mode="r"
            )
        values = np.asarray(self._arrays[waveform_file][int(row)], dtype=np.float32).copy()
        mean = float(values.mean())
        standard_deviation = float(values.std())
        if not np.isfinite(standard_deviation) or standard_deviation <= 1e-8:
            raise ValueError(f"invalid waveform standard deviation in {waveform_file}:{row}")
        values = (values - mean) / standard_deviation
        return torch.from_numpy(values[None, :])


class PopulationDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        store_root: Path,
        scaler: dict[str, list[float]],
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.accessor = WaveformAccessor(store_root)
        self.mean = torch.tensor(scaler["mean"], dtype=torch.float32)
        self.std = torch.tensor(scaler["std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        bp = torch.tensor([row.sbp, row.dbp], dtype=torch.float32)
        return {
            "ppg": self.accessor.get(row.waveform_file, int(row.waveform_row)),
            "target": (bp - self.mean) / self.std,
            "subject_uid": row.subject_uid,
            "event_id": row.event_id,
        }


class EpisodicDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        store_root: Path,
        scaler: dict[str, list[float]],
        *,
        ks: tuple[int, ...] = KS,
        rolling_support: bool = False,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.accessor = WaveformAccessor(store_root)
        self.mean = torch.tensor(scaler["mean"], dtype=torch.float32)
        self.std = torch.tensor(scaler["std"], dtype=torch.float32)
        self.ks = tuple(ks)
        self.rolling_support = rolling_support
        support_rows: dict[str, list[int]] = defaultdict(list)
        all_rows: dict[str, list[int]] = defaultdict(list)
        query_rows: list[int] = []
        for index, row in self.metadata.iterrows():
            all_rows[str(row.subject_uid)].append(index)
            if int(row.event_index) <= max(KS):
                support_rows[str(row.subject_uid)].append(index)
            elif bool(row.common_query):
                query_rows.append(index)
        self.support_rows = {
            subject: sorted(rows, key=lambda value: int(self.metadata.iloc[value].event_index))
            for subject, rows in support_rows.items()
        }
        self.all_rows = {
            subject: sorted(rows, key=lambda value: int(self.metadata.iloc[value].event_index))
            for subject, rows in all_rows.items()
        }
        self.query_rows = query_rows
        if not self.query_rows:
            raise ValueError("episodic dataset has no query rows")
        if any(len(rows) < max(KS) for rows in self.support_rows.values()):
            raise AssertionError("an eligible subject lacks five support candidates")

    def __len__(self) -> int:
        return len(self.query_rows) * len(self.ks)

    @property
    def participant_ids(self) -> list[str]:
        return [
            str(self.metadata.iloc[self.query_rows[index // len(self.ks)]].subject_uid)
            for index in range(len(self))
        ]

    def _waveform(self, row: pd.Series) -> torch.Tensor:
        return self.accessor.get(row.waveform_file, int(row.waveform_row))

    def __getitem__(self, index: int) -> dict[str, object]:
        query_position, k_position = divmod(index, len(self.ks))
        k = self.ks[k_position]
        query = self.metadata.iloc[self.query_rows[query_position]]
        if self.rolling_support:
            eligible_prior = [
                row_index
                for row_index in self.all_rows[str(query.subject_uid)]
                if int(self.metadata.iloc[row_index].event_index) < int(query.event_index)
            ]
            candidate_pool = eligible_prior[-max(KS) :]
            if len(candidate_pool) < max(KS):
                raise AssertionError("rolling episode has fewer than five prior events")
            support_indexes = candidate_pool[-k:]
        else:
            support_indexes = self.support_rows[str(query.subject_uid)][:k]
        support_ppg = torch.zeros(max(KS), 1, int(query.n_samples), dtype=torch.float32)
        support_bp = torch.zeros(max(KS), 2, dtype=torch.float32)
        support_mask = torch.zeros(max(KS), dtype=torch.bool)
        for position, support_index in enumerate(support_indexes):
            support = self.metadata.iloc[support_index]
            support_ppg[position] = self._waveform(support)
            bp = torch.tensor([support.sbp, support.dbp], dtype=torch.float32)
            support_bp[position] = (bp - self.mean) / self.std
            support_mask[position] = True
        query_bp = torch.tensor([query.sbp, query.dbp], dtype=torch.float32)
        return {
            "query_ppg": self._waveform(query),
            "support_ppg": support_ppg,
            "support_bp": support_bp,
            "support_mask": support_mask,
            "target": (query_bp - self.mean) / self.std,
            "subject_uid": query.subject_uid,
            "event_id": query.event_id,
            "k": k,
        }


@torch.no_grad()
def predict_population(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler: dict[str, list[float]],
) -> pd.DataFrame:
    model.eval()
    mean = torch.tensor(scaler["mean"], device=device)
    std = torch.tensor(scaler["std"], device=device)
    rows: list[dict[str, object]] = []
    for batch in loader:
        inputs = batch["ppg"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        predictions = model(inputs)
        predictions = predictions * std + mean
        targets = targets * std + mean
        for index in range(len(batch["event_id"])):
            rows.append(
                {
                    "subject_uid": batch["subject_uid"][index],
                    "event_id": batch["event_id"][index],
                    "target_sbp": float(targets[index, 0]),
                    "target_dbp": float(targets[index, 1]),
                    "pred_sbp": float(predictions[index, 0]),
                    "pred_dbp": float(predictions[index, 1]),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def predict_episodic(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler: dict[str, list[float]],
    *,
    siamese: bool = False,
) -> pd.DataFrame:
    model.eval()
    mean = torch.tensor(scaler["mean"], device=device)
    std = torch.tensor(scaler["std"], device=device)
    rows: list[dict[str, object]] = []
    for batch in loader:
        query = batch["query_ppg"].to(device, non_blocking=True)
        support = batch["support_ppg"].to(device, non_blocking=True)
        support_bp = batch["support_bp"].to(device, non_blocking=True)
        mask = batch["support_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if siamese:
            predictions = model(query, support[:, 0], support_bp[:, 0])
        else:
            predictions = model(query, support, support_bp, mask)
        predictions = predictions * std + mean
        targets = targets * std + mean
        for index in range(len(batch["event_id"])):
            rows.append(
                {
                    "subject_uid": batch["subject_uid"][index],
                    "event_id": batch["event_id"][index],
                    "k": int(batch["k"][index]),
                    "target_sbp": float(targets[index, 0]),
                    "target_dbp": float(targets[index, 1]),
                    "pred_sbp": float(predictions[index, 0]),
                    "pred_dbp": float(predictions[index, 1]),
                }
            )
    return pd.DataFrame(rows)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
