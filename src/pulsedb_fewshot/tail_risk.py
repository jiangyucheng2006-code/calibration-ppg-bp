"""Cross-fitted risk identification and hard-participant expert utilities.

This module deliberately separates three roles:

1. generate participant-disjoint meta-train folds;
2. derive out-of-fold hard-participant labels and deployment-visible features;
3. train/evaluate a risk model without using meta-validation labels for fitting.

The locked meta-test is never read.  Oracle error labels are used only for
supervised learning inside cross-fitted meta-train data and for explicitly
labelled development evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .training import load_store_metadata, participant_macro_metrics, save_json, seed_everything


FEATURE_COLUMNS = [
    "pred_sbp",
    "pred_dbp",
    "support_sbp_mean",
    "support_sbp_std",
    "support_sbp_range",
    "support_sbp_mad",
    "support_dbp_mean",
    "support_dbp_std",
    "support_dbp_range",
    "support_dbp_mad",
    "pred_support_sbp_distance",
    "pred_support_dbp_distance",
    "query_ppg_mean",
    "query_ppg_std",
    "support_ppg_mean_mean",
    "support_ppg_mean_range",
    "support_ppg_std_mean",
    "support_ppg_std_range",
    "query_support_ppg_mean_distance",
    "query_support_ppg_std_distance",
    "event_index",
    "events_since_calibration",
]


def _exact_tail(
    participant: pd.DataFrame,
    *,
    fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("tail fraction must be strictly between zero and one")
    required = {"subject_uid", "source", "participant_mean_mae"}
    missing = required - set(participant.columns)
    if missing:
        raise ValueError(f"participant table missing {sorted(missing)}")
    labelled: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for source, group in participant.groupby("source", sort=True):
        ordered = group.sort_values(
            ["participant_mean_mae", "subject_uid"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        tail_n = int(math.ceil(fraction * len(ordered)))
        ordered["source_error_rank"] = np.arange(1, len(ordered) + 1)
        ordered["hard_oof"] = ordered["source_error_rank"].le(tail_n)
        labelled.append(ordered)
        summary_rows.append(
            {
                "source": str(source),
                "participants": int(len(ordered)),
                "hard_participants": tail_n,
                "easy_participants": int(len(ordered) - tail_n),
                "hard_threshold_mae": float(
                    ordered.iloc[tail_n - 1]["participant_mean_mae"]
                ),
            }
        )
    result = pd.concat(labelled, ignore_index=True)
    result["label_split"] = "meta_train_crossfit_oof"
    return result, pd.DataFrame(summary_rows)


def assign_source_stratified_folds(
    participant: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError("at least two cross-fit folds are required")
    required = {"subject_uid", "source"}
    missing = required - set(participant.columns)
    if missing:
        raise ValueError(f"participant table missing {sorted(missing)}")
    frame = participant[["subject_uid", "source"]].drop_duplicates().copy()
    if frame["subject_uid"].duplicated().any():
        raise AssertionError("a participant is associated with multiple sources")
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for source, group in frame.groupby("source", sort=True):
        ordered = group.sort_values("subject_uid", kind="mergesort").reset_index(drop=True)
        order = rng.permutation(len(ordered))
        shuffled = ordered.iloc[order].reset_index(drop=True)
        shuffled["fold"] = np.arange(len(shuffled)) % n_folds
        rows.append(shuffled)
    result = pd.concat(rows, ignore_index=True).sort_values(
        ["fold", "source", "subject_uid"], kind="mergesort"
    )
    if result["subject_uid"].duplicated().any():
        raise AssertionError("cross-fit assignment duplicated a participant")
    result["split"] = "meta_train"
    result["n_folds"] = n_folds
    result["seed"] = seed
    return result.reset_index(drop=True)


def prepare_folds(
    store_root: Path,
    output_dir: Path,
    *,
    n_folds: int,
    seed: int,
) -> dict[str, object]:
    metadata = load_store_metadata(store_root, "development")
    if metadata["split"].eq("meta_test").any():
        raise AssertionError("development store unexpectedly contains meta-test rows")
    train = metadata.loc[metadata["split"].eq("meta_train")]
    participant = train[["subject_uid", "source"]].drop_duplicates()
    folds = assign_source_stratified_folds(participant, n_folds=n_folds, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "meta_train_crossfit_folds.parquet"
    folds.to_parquet(path, index=False)
    counts = (
        folds.groupby(["fold", "source"], as_index=False)
        .size()
        .rename(columns={"size": "participants"})
    )
    counts.to_csv(output_dir / "fold_counts.csv", index=False)
    payload = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "meta_train_only",
        "locked_test_accessed": False,
        "participants": int(len(folds)),
        "n_folds": n_folds,
        "seed": seed,
        "folds_file": path.name,
        "counts": counts.to_dict(orient="records"),
    }
    save_json(output_dir / "folds.json", payload)
    return payload


def _load_oof_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata_path = run_dir / "run.json"
    prediction_path = run_dir / "best_validation_predictions.parquet"
    if not metadata_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"incomplete cross-fit run: {run_dir}")
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        raise AssertionError(f"cross-fit run is not complete: {run_dir}")
    if record.get("locked_test_accessed") is not False:
        raise AssertionError("cross-fit run does not explicitly deny locked-test access")
    fold = record.get("crossfit_heldout_fold")
    if record.get("split") != f"meta_train_crossfit_fold_{fold}":
        raise AssertionError("cross-fit run split metadata are inconsistent")
    if record.get("method") != "m0":
        raise AssertionError("OOF label generation requires M0 fold runs")
    predictions = pd.read_parquet(prediction_path)
    if set(pd.to_numeric(predictions["k"]).astype(int)) != {5}:
        raise AssertionError("OOF risk-label runs must contain K=5 only")
    predictions["fold"] = int(fold)
    return predictions, record


def aggregate_oof_labels(
    store_root: Path,
    fold_runs: Sequence[Path],
    folds_path: Path,
    output_dir: Path,
    *,
    tail_fraction: float,
) -> dict[str, object]:
    folds = pd.read_parquet(folds_path)
    if folds["subject_uid"].duplicated().any():
        raise AssertionError("cross-fit fold table contains duplicate participants")
    frames: list[pd.DataFrame] = []
    seen_folds: set[int] = set()
    for run_dir in fold_runs:
        frame, record = _load_oof_run(run_dir)
        fold = int(record["crossfit_heldout_fold"])
        if fold in seen_folds:
            raise AssertionError(f"duplicate OOF run for fold {fold}")
        seen_folds.add(fold)
        expected = set(
            folds.loc[pd.to_numeric(folds["fold"]).astype(int).eq(fold), "subject_uid"]
            .astype(str)
        )
        observed = set(frame["subject_uid"].astype(str))
        if observed != expected:
            raise AssertionError(f"OOF predictions do not match held-out fold {fold}")
        frames.append(frame)
    expected_folds = set(pd.to_numeric(folds["fold"]).astype(int))
    if seen_folds != expected_folds:
        raise AssertionError("OOF runs do not cover every cross-fit fold")
    oof = pd.concat(frames, ignore_index=True)
    if oof.duplicated(["subject_uid", "event_id", "k"]).any():
        raise AssertionError("OOF prediction rows are not unique")
    expected_subjects = set(folds["subject_uid"].astype(str))
    if set(oof["subject_uid"].astype(str)) != expected_subjects:
        raise AssertionError("OOF predictions do not cover all meta-train participants")
    fold_source = folds[["subject_uid", "source", "fold"]].assign(
        subject_uid=lambda frame: frame["subject_uid"].astype(str)
    )
    oof = oof.assign(subject_uid=oof["subject_uid"].astype(str)).merge(
        fold_source,
        on=["subject_uid", "fold"],
        how="left",
        validate="many_to_one",
    )
    if oof["source"].isna().any():
        raise AssertionError("OOF predictions could not be assigned a source")
    oof["abs_error_sbp"] = (oof["pred_sbp"] - oof["target_sbp"]).abs()
    oof["abs_error_dbp"] = (oof["pred_dbp"] - oof["target_dbp"]).abs()
    participant = oof.groupby(
        ["subject_uid", "source", "fold"], as_index=False
    ).agg(
        sbp_mae=("abs_error_sbp", "mean"),
        dbp_mae=("abs_error_dbp", "mean"),
        n_query_events=("event_id", "size"),
    )
    participant["participant_mean_mae"] = (
        participant["sbp_mae"] + participant["dbp_mae"]
    ) / 2.0
    labels, source_summary = _exact_tail(participant, fraction=tail_fraction)
    feature_rows = build_risk_features(store_root, oof, labels)

    output_dir.mkdir(parents=True, exist_ok=False)
    labels.to_parquet(output_dir / "participant_risk_labels.parquet", index=False)
    oof.to_parquet(output_dir / "oof_k5_predictions.parquet", index=False)
    feature_rows.to_parquet(output_dir / "oof_risk_features.parquet", index=False)
    source_summary.to_csv(output_dir / "oof_tail_source_summary.csv", index=False)
    payload = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "meta_train_crossfit_oof",
        "locked_test_accessed": False,
        "folds": sorted(seen_folds),
        "participants": int(len(labels)),
        "queries": int(len(oof)),
        "tail_fraction": tail_fraction,
        "tail_definition": (
            "within-source OOF participant mean MAE descending, subject_uid "
            "ascending tie-break, first ceil(fraction*N)"
        ),
        "risk_features": FEATURE_COLUMNS,
        "source_summary": source_summary.to_dict(orient="records"),
    }
    save_json(output_dir / "oof_labels.json", payload)
    return payload


def _mad(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def build_risk_features(
    store_root: Path,
    predictions: pd.DataFrame,
    labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required_predictions = {
        "subject_uid",
        "event_id",
        "pred_sbp",
        "pred_dbp",
    }
    missing = required_predictions - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction table missing {sorted(missing)}")
    metadata = load_store_metadata(store_root, "development")
    subjects = set(predictions["subject_uid"].astype(str))
    metadata = metadata.loc[metadata["subject_uid"].astype(str).isin(subjects)].copy()
    support = metadata.loc[metadata["support_candidate"] & metadata["event_index"].le(5)]
    support = support.sort_values(["subject_uid", "event_index"], kind="mergesort")
    support = support.groupby("subject_uid", as_index=False, group_keys=False).head(5)
    counts = support.groupby("subject_uid").size()
    if not counts.eq(5).all() or set(counts.index.astype(str)) != subjects:
        raise AssertionError("every risk-feature participant requires exactly five supports")
    support_features = support.groupby("subject_uid", as_index=False).agg(
        support_sbp_mean=("sbp", "mean"),
        support_sbp_std=("sbp", lambda values: float(values.std(ddof=0))),
        support_sbp_min=("sbp", "min"),
        support_sbp_max=("sbp", "max"),
        support_sbp_mad=("sbp", _mad),
        support_dbp_mean=("dbp", "mean"),
        support_dbp_std=("dbp", lambda values: float(values.std(ddof=0))),
        support_dbp_min=("dbp", "min"),
        support_dbp_max=("dbp", "max"),
        support_dbp_mad=("dbp", _mad),
        support_ppg_mean_mean=("ppg_f_mean", "mean"),
        support_ppg_mean_min=("ppg_f_mean", "min"),
        support_ppg_mean_max=("ppg_f_mean", "max"),
        support_ppg_std_mean=("ppg_f_std", "mean"),
        support_ppg_std_min=("ppg_f_std", "min"),
        support_ppg_std_max=("ppg_f_std", "max"),
    )
    support_features["support_sbp_range"] = (
        support_features.pop("support_sbp_max")
        - support_features.pop("support_sbp_min")
    )
    support_features["support_dbp_range"] = (
        support_features.pop("support_dbp_max")
        - support_features.pop("support_dbp_min")
    )
    support_features["support_ppg_mean_range"] = (
        support_features.pop("support_ppg_mean_max")
        - support_features.pop("support_ppg_mean_min")
    )
    support_features["support_ppg_std_range"] = (
        support_features.pop("support_ppg_std_max")
        - support_features.pop("support_ppg_std_min")
    )
    query_columns = [
        "subject_uid",
        "event_id",
        "event_index",
        "record_order",
        "time_bin",
        "n_segments_in_bin",
        "ppg_f_mean",
        "ppg_f_std",
        "source",
        "split",
    ]
    queries = metadata.loc[metadata["common_query"], query_columns]
    # Source and split are recovered from the frozen store, not trusted from a
    # prediction artifact.  Dropping possible copies also prevents merge
    # suffixes from silently removing the canonical ``source`` column.
    prediction_frame = predictions.drop(
        columns=[column for column in ("source", "split") if column in predictions]
    ).assign(subject_uid=predictions["subject_uid"].astype(str))
    frame = prediction_frame.merge(
        queries.assign(subject_uid=queries["subject_uid"].astype(str)),
        on=["subject_uid", "event_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        support_features.assign(
            subject_uid=support_features["subject_uid"].astype(str)
        ),
        on="subject_uid",
        how="left",
        validate="many_to_one",
    )
    if frame[["source", "event_index", "support_sbp_mean"]].isna().any().any():
        raise AssertionError("risk features could not be joined to frozen metadata")
    frame = frame.rename(
        columns={"ppg_f_mean": "query_ppg_mean", "ppg_f_std": "query_ppg_std"}
    )
    frame["pred_support_sbp_distance"] = (
        frame["pred_sbp"] - frame["support_sbp_mean"]
    ).abs()
    frame["pred_support_dbp_distance"] = (
        frame["pred_dbp"] - frame["support_dbp_mean"]
    ).abs()
    frame["query_support_ppg_mean_distance"] = (
        frame["query_ppg_mean"] - frame["support_ppg_mean_mean"]
    ).abs()
    frame["query_support_ppg_std_distance"] = (
        frame["query_ppg_std"] - frame["support_ppg_std_mean"]
    ).abs()
    frame["events_since_calibration"] = frame["event_index"] - 5
    if labels is not None:
        label_columns = [
            "subject_uid",
            "fold",
            "hard_oof",
            "participant_mean_mae",
            "label_split",
        ]
        frame = frame.merge(
            labels[label_columns].assign(
                subject_uid=labels["subject_uid"].astype(str)
            ),
            on="subject_uid",
            how="left",
            validate="many_to_one",
            suffixes=("", "_label"),
        )
        if frame["hard_oof"].isna().any():
            raise AssertionError("risk feature row is missing an OOF label")
    numeric = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("risk feature matrix contains nonfinite values")
    return frame


class RiskMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.risk_head = nn.Linear(32, 1)
        self.error_head = nn.Linear(32, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.risk_head(hidden).squeeze(-1), self.error_head(hidden).squeeze(-1)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def _risk_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(bool)
    predicted = scores >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    return {
        "average_precision": _average_precision(labels, scores),
        "brier": float(np.mean((scores - labels.astype(float)) ** 2)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "predicted_high_risk_fraction": float(predicted.mean()),
    }


def train_risk_model(
    feature_path: Path,
    output_dir: Path,
    *,
    validation_fold: int,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> dict[str, object]:
    seed_everything(seed)
    frame = pd.read_parquet(feature_path)
    required = set(FEATURE_COLUMNS) | {
        "subject_uid",
        "source",
        "fold",
        "hard_oof",
        "participant_mean_mae",
        "label_split",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"risk feature table missing {sorted(missing)}")
    if set(frame["label_split"].astype(str)) != {"meta_train_crossfit_oof"}:
        raise AssertionError("risk model accepts cross-fitted meta-train labels only")
    train = frame.loc[pd.to_numeric(frame["fold"]).astype(int).ne(validation_fold)].copy()
    validation = frame.loc[
        pd.to_numeric(frame["fold"]).astype(int).eq(validation_fold)
    ].copy()
    if train.empty or validation.empty:
        raise ValueError("risk train or validation fold is empty")
    if set(train["subject_uid"]) & set(validation["subject_uid"]):
        raise AssertionError("participant leakage in risk train/validation split")
    mean = train[FEATURE_COLUMNS].mean().to_numpy(dtype=np.float32)
    std = train[FEATURE_COLUMNS].std(ddof=0).to_numpy(dtype=np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("invalid risk-feature scaler")
    constant_features = [
        feature for feature, value in zip(FEATURE_COLUMNS, std) if value <= 1e-8
    ]
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)

    def tensors(data: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = ((data[FEATURE_COLUMNS].to_numpy(np.float32) - mean) / std).astype(np.float32)
        hard = data["hard_oof"].to_numpy(np.float32)
        error = np.log1p(data["participant_mean_mae"].to_numpy(np.float32))
        return torch.from_numpy(x), torch.from_numpy(hard), torch.from_numpy(error)

    x_train, y_train, e_train = tensors(train)
    x_val, y_val, e_val = tensors(validation)
    validation_identity = validation[["subject_uid", "hard_oof"]].copy()
    subject_counts = train["subject_uid"].astype(str).value_counts()
    sample_weights = train["subject_uid"].astype(str).map(
        lambda value: 1.0 / subject_counts[value]
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights.to_numpy(), dtype=torch.double),
        num_samples=min(200000, len(train)),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        TensorDataset(x_train, y_train, e_train),
        batch_size=batch_size,
        sampler=sampler,
    )
    model = RiskMLP(len(FEATURE_COLUMNS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    huber = nn.HuberLoss(delta=0.25)
    best_ap = -math.inf
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, float]] = []
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "risk_model.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        examples = 0
        for x, hard, error in loader:
            optimizer.zero_grad(set_to_none=True)
            logit, expected_error = model(x)
            loss = bce(logit, hard) + 0.25 * huber(expected_error, error)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(x)
            examples += len(x)
        model.eval()
        with torch.no_grad():
            val_logit, val_error = model(x_val)
            val_scores = torch.sigmoid(val_logit).numpy()
            validation_identity["risk_score"] = val_scores
            participant_epoch = validation_identity.groupby(
                ["subject_uid", "hard_oof"], as_index=False
            )["risk_score"].mean()
            val_ap = _average_precision(
                participant_epoch["hard_oof"].to_numpy(bool),
                participant_epoch["risk_score"].to_numpy(float),
            )
            val_brier = float(
                np.mean(
                    (
                        participant_epoch["risk_score"].to_numpy(float)
                        - participant_epoch["hard_oof"].to_numpy(float)
                    )
                    ** 2
                )
            )
            val_error_mae = float(torch.mean(torch.abs(val_error - e_val)))
        record = {
            "epoch": float(epoch),
            "train_loss": total / max(examples, 1),
            "validation_average_precision": val_ap,
            "validation_brier": val_brier,
            "validation_log_error_mae": val_error_mae,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            no_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_columns": FEATURE_COLUMNS,
                    "feature_mean": mean.tolist(),
                    "feature_std": std.tolist(),
                    "validation_fold": validation_fold,
                    "seed": seed,
                    "best_epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            no_improvement += 1
        if no_improvement >= patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        training_scores = torch.sigmoid(model(x_train)[0]).numpy()
        scores = torch.sigmoid(model(x_val)[0]).numpy()
    participant_training = train[["subject_uid", "source", "hard_oof"]].copy()
    participant_training["risk_score"] = training_scores
    participant_training = participant_training.groupby(
        ["subject_uid", "source", "hard_oof"], as_index=False
    )["risk_score"].mean()
    participant_validation = validation[["subject_uid", "source", "hard_oof"]].copy()
    participant_validation["risk_score"] = scores
    participant_validation = participant_validation.groupby(
        ["subject_uid", "source", "hard_oof"], as_index=False
    )["risk_score"].mean()
    participant_threshold = float(
        participant_training["risk_score"].quantile(0.70)
    )
    event_threshold = float(np.quantile(training_scores, 0.70))
    checkpoint["participant_threshold"] = participant_threshold
    checkpoint["event_threshold"] = event_threshold
    torch.save(checkpoint, checkpoint_path)
    metric_rows: list[dict[str, object]] = []
    for scope, group in [("Overall", participant_validation)] + [
        (source, participant_validation.loc[participant_validation["source"].eq(source)])
        for source in sorted(participant_validation["source"].unique())
    ]:
        metric_rows.append(
            {"scope": scope, **_risk_metrics(
                group["hard_oof"].to_numpy(bool),
                group["risk_score"].to_numpy(float),
                participant_threshold,
            )}
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "risk_validation_metrics.csv", index=False)
    participant_validation.to_csv(
        output_dir / "risk_validation_participant_scores.csv", index=False
    )
    save_json(output_dir / "risk_history.json", history)
    payload = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "meta_train_crossfit_risk_validation",
        "locked_test_accessed": False,
        "validation_fold": validation_fold,
        "train_participants": int(train["subject_uid"].nunique()),
        "validation_participants": int(validation["subject_uid"].nunique()),
        "best_epoch": best_epoch,
        "participant_threshold": participant_threshold,
        "event_threshold": event_threshold,
        "metrics": metrics.to_dict(orient="records"),
        "feature_columns": FEATURE_COLUMNS,
        "constant_features": constant_features,
    }
    save_json(output_dir / "risk_run.json", payload)
    return payload


def _score_features(feature_frame: pd.DataFrame, checkpoint_path: Path) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["feature_columns"] != FEATURE_COLUMNS:
        raise AssertionError("risk checkpoint feature schema differs")
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    x = (feature_frame[FEATURE_COLUMNS].to_numpy(np.float32) - mean) / std
    model = RiskMLP(len(FEATURE_COLUMNS))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        tensor = torch.from_numpy(x)
        for start in range(0, len(tensor), 4096):
            scores.append(torch.sigmoid(model(tensor[start : start + 4096])[0]).numpy())
    return np.concatenate(scores)


def evaluate_routing(
    store_root: Path,
    reference_run: Path,
    expert_run: Path,
    risk_checkpoint: Path,
    output_dir: Path,
) -> dict[str, object]:
    def load_run(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
        record = json.loads((path / "run.json").read_text(encoding="utf-8"))
        if record.get("status") != "complete" or record.get("split") != "meta_validation":
            raise AssertionError(f"routing input is not a complete meta-validation run: {path}")
        if record.get("locked_test_accessed") is not False:
            raise AssertionError("routing input does not deny locked-test access")
        frame = pd.read_parquet(path / "best_validation_predictions.parquet")
        return frame.loc[pd.to_numeric(frame["k"]).astype(int).eq(5)].copy(), record

    reference, reference_record = load_run(reference_run)
    expert, expert_record = load_run(expert_run)
    keys = ["subject_uid", "event_id", "k"]
    reference = reference.sort_values(keys, kind="mergesort").reset_index(drop=True)
    expert = expert.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if not reference[keys].equals(expert[keys]):
        raise AssertionError("reference and expert query keys differ")
    for target in ("target_sbp", "target_dbp"):
        if not np.allclose(reference[target], expert[target], rtol=0.0, atol=1e-4):
            raise AssertionError("reference and expert targets differ")
    metadata = load_store_metadata(store_root, "development")
    source_lookup = metadata.loc[
        metadata["split"].eq("meta_validation") & metadata["common_query"],
        ["subject_uid", "event_id", "source"],
    ]
    reference = reference.merge(
        source_lookup,
        on=["subject_uid", "event_id"],
        how="left",
        validate="one_to_one",
    )
    features = build_risk_features(store_root, reference)
    scores = _score_features(features, risk_checkpoint)
    checkpoint = torch.load(risk_checkpoint, map_location="cpu", weights_only=False)
    threshold = float(checkpoint["event_threshold"])
    routed = reference.copy()
    routed["risk_score"] = scores
    routed["high_risk"] = routed["risk_score"].ge(threshold)
    for bp in ("sbp", "dbp"):
        routed[f"pred_{bp}"] = np.where(
            routed["high_risk"], expert[f"pred_{bp}"], reference[f"pred_{bp}"]
        )
    # Soft fusion is retained as a separate exploratory comparator.  It is
    # deployment-valid because the mixing weight is the input-visible risk
    # score; it does not use query BP or query error.
    soft_routed = reference.copy()
    soft_routed["risk_score"] = scores
    for bp in ("sbp", "dbp"):
        soft_routed[f"pred_{bp}"] = (
            (1.0 - scores) * reference[f"pred_{bp}"].to_numpy(dtype=float)
            + scores * expert[f"pred_{bp}"].to_numpy(dtype=float)
        )
    rows: list[dict[str, object]] = []
    for setting, frame in (
        ("M0", reference),
        ("Tail expert", expert),
        ("Risk hard-routed", routed),
        ("Risk soft-fused", soft_routed),
    ):
        frame = frame.merge(
            source_lookup,
            on=["subject_uid", "event_id"],
            how="left",
            validate="one_to_one",
        ) if "source" not in frame.columns else frame
        for scope, selected in [("Overall", frame)] + [
            (source, frame.loc[frame["source"].eq(source)])
            for source in sorted(frame["source"].dropna().unique())
        ]:
            metrics = participant_macro_metrics(selected)
            rows.append(
                {
                    "setting": setting,
                    "scope": scope,
                    "participants": metrics["n_participants"],
                    "queries": metrics["n_events"],
                    "sbp_mae": metrics["sbp_mae"],
                    "dbp_mae": metrics["dbp_mae"],
                    "mean_mae": metrics["mean_mae"],
                    "worst_30_mean_mae": metrics["worst_30_mean_mae"],
                    "retained_70_mean_mae": metrics["retained_70_mean_mae"],
                }
            )
    results = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    results.to_csv(output_dir / "routing_metrics.csv", index=False)
    routed[["subject_uid", "event_id", "k", "risk_score", "high_risk"]].to_parquet(
        output_dir / "routing_scores.parquet", index=False
    )
    payload = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "meta_validation",
        "locked_test_accessed": False,
        "risk_threshold_source": "meta_train_crossfit_risk_validation",
        "event_risk_threshold": threshold,
        "reference_job": reference_record.get("slurm_job_id"),
        "expert_job": expert_record.get("slurm_job_id"),
        "high_risk_event_fraction": float(routed["high_risk"].mean()),
        "metrics": results.to_dict(orient="records"),
        "claim_limit": (
            "development-only input-visible routing; locked meta-test and external "
            "performance are not established"
        ),
    }
    save_json(output_dir / "routing_run.json", payload)
    return payload


def _paths(values: Iterable[str]) -> list[Path]:
    return [Path(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    folds = subparsers.add_parser("prepare-folds")
    folds.add_argument("--store-root", type=Path, required=True)
    folds.add_argument("--output-dir", type=Path, required=True)
    folds.add_argument("--n-folds", type=int, default=5)
    folds.add_argument("--seed", type=int, default=20260818)

    aggregate = subparsers.add_parser("aggregate-oof")
    aggregate.add_argument("--store-root", type=Path, required=True)
    aggregate.add_argument("--folds", type=Path, required=True)
    aggregate.add_argument("--fold-run", action="append", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--tail-fraction", type=float, default=0.30)

    risk = subparsers.add_parser("train-risk")
    risk.add_argument("--features", type=Path, required=True)
    risk.add_argument("--output-dir", type=Path, required=True)
    risk.add_argument("--validation-fold", type=int, default=4)
    risk.add_argument("--seed", type=int, default=20260818)
    risk.add_argument("--epochs", type=int, default=100)
    risk.add_argument("--patience", type=int, default=10)
    risk.add_argument("--batch-size", type=int, default=512)

    route = subparsers.add_parser("evaluate-routing")
    route.add_argument("--store-root", type=Path, required=True)
    route.add_argument("--reference-run", type=Path, required=True)
    route.add_argument("--expert-run", type=Path, required=True)
    route.add_argument("--risk-checkpoint", type=Path, required=True)
    route.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare-folds":
        payload = prepare_folds(
            args.store_root,
            args.output_dir,
            n_folds=args.n_folds,
            seed=args.seed,
        )
    elif args.command == "aggregate-oof":
        payload = aggregate_oof_labels(
            args.store_root,
            _paths(args.fold_run),
            args.folds,
            args.output_dir,
            tail_fraction=args.tail_fraction,
        )
    elif args.command == "train-risk":
        payload = train_risk_model(
            args.features,
            args.output_dir,
            validation_fold=args.validation_fold,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
    else:
        payload = evaluate_routing(
            args.store_root,
            args.reference_run,
            args.expert_run,
            args.risk_checkpoint,
            args.output_dir,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
