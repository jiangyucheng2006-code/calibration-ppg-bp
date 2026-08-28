"""Train one executable candidate on the development CalBased analogue.

The runner is fail-closed: it loads exactly ``train`` and
``internal_validation`` metadata, performs early stopping on internal
validation participant-macro MAE, and has no argument or code path for the
held-out test targets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .calbased_screen import (
    CANDIDATES,
    EARLY_STOPPING_PATIENCE,
    PROTOCOL_ID,
    ROLE_WINDOWS_PER_SUBJECT,
    SCREENING_READ_ROLES,
    SPLIT_MODES,
    SUBJECT_COUNT,
    fit_subject_train_means,
    predict_subject_train_mean,
    _validate_store_manifest,
)
from .calbased_metrics import participant_macro_views, pooled_diagnostics
from .models import (
    PopulationRegressor,
    SubjectMeanResidualRegressor,
    build_ppg_encoder,
    model_parameter_counts,
)
from .training import (
    WaveformAccessor,
    file_sha256,
    save_json,
    seed_everything,
    source_tree_sha256,
)


EXECUTABLE_CANDIDATES = tuple(
    candidate.name
    for candidate in CANDIDATES.values()
    if candidate.executable_first_round
)


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )


def _load_role_metadata(
    store_root: Path, split_mode: str, role: str
) -> pd.DataFrame:
    """Load one allowed role; held-out metadata is intentionally unreachable."""

    if role not in SCREENING_READ_ROLES:
        raise ValueError(
            f"screening runner may load only {SCREENING_READ_ROLES}, got {role!r}"
        )
    split_root = store_root / split_mode
    paths = sorted(split_root.glob(f"{role}_metadata_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no {role} metadata shards under CalBased split {split_root}"
        )
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
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
        "sbp",
        "dbp",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{role} metadata missing columns: {sorted(missing)}")
    if set(frame["protocol_id"].astype(str)) != {PROTOCOL_ID}:
        raise ValueError(f"{role} metadata has a wrong protocol_id")
    if set(frame["split_mode"].astype(str)) != {split_mode}:
        raise ValueError(f"{role} metadata has a wrong split_mode")
    if set(frame["role"].astype(str)) != {role}:
        raise ValueError(f"{role} metadata contains a different role")
    if frame["segment_uid"].astype(str).duplicated().any():
        raise ValueError(f"{role} metadata contains duplicate segment_uid values")
    if not pd.to_numeric(frame["n_samples"], errors="raise").eq(1250).all():
        raise ValueError(f"{role} metadata must contain 1250-sample PPG windows")
    targets = frame[["sbp", "dbp"]].to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError(f"{role} labels must be finite")
    root = store_root.resolve()
    for relative in frame["waveform_file"].astype(str).unique():
        waveform_path = (store_root / relative).resolve()
        if not waveform_path.is_relative_to(root):
            raise ValueError("waveform_file escapes the CalBased store root")
        if not waveform_path.is_file():
            raise FileNotFoundError(f"missing waveform shard: {waveform_path}")
    frame = frame.copy()
    frame["subject_uid"] = frame["subject_uid"].astype(str)
    frame["segment_uid"] = frame["segment_uid"].astype(str)
    frame["waveform_row"] = pd.to_numeric(
        frame["waveform_row"], errors="raise"
    ).astype(np.int64)
    return frame


def load_screen_metadata(
    store_root: Path, split_mode: str, *, enforce_full_cohort: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and jointly validate train/internal-validation roles only."""

    if split_mode not in SPLIT_MODES:
        raise ValueError(f"unsupported split_mode {split_mode!r}")
    _validate_store_manifest(store_root)
    train = _load_role_metadata(store_root, split_mode, "train")
    validation = _load_role_metadata(store_root, split_mode, "internal_validation")
    train_subjects = set(train["subject_uid"])
    validation_subjects = set(validation["subject_uid"])
    if train_subjects != validation_subjects:
        raise ValueError("train and internal_validation must contain the same subjects")
    if set(train["segment_uid"]) & set(validation["segment_uid"]):
        raise ValueError("train/internal_validation segment leakage detected")
    subject_sources = pd.concat(
        [
            train[["subject_uid", "source"]],
            validation[["subject_uid", "source"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    if subject_sources["subject_uid"].duplicated().any():
        raise ValueError("a participant changes source across roles")
    if enforce_full_cohort:
        if len(train_subjects) != SUBJECT_COUNT:
            raise ValueError(f"expected {SUBJECT_COUNT} same-subject participants")
        for role, metadata in (
            ("train", train),
            ("internal_validation", validation),
        ):
            counts = metadata.groupby("subject_uid").size()
            expected = ROLE_WINDOWS_PER_SUBJECT[role]
            if not counts.eq(expected).all():
                raise ValueError(
                    f"every participant must have exactly {expected} {role} windows"
                )
    return train, validation


def _fit_scaler(train: pd.DataFrame) -> dict[str, list[float]]:
    values = train[["sbp", "dbp"]].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("invalid train-only target scaler")
    return {"mean": mean.tolist(), "std": std.tolist()}


class CalBasedDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        store_root: Path,
        scaler: dict[str, list[float]],
        *,
        subject_means: pd.DataFrame | None = None,
    ) -> None:
        frame = metadata.copy()
        if subject_means is not None:
            frame = frame.merge(
                subject_means,
                on="subject_uid",
                how="left",
                validate="many_to_one",
            )
            columns = ["subject_train_sbp", "subject_train_dbp"]
            if frame[columns].isna().any().any():
                raise ValueError("subject train means do not cover every dataset row")
        self.metadata = frame.reset_index(drop=True)
        self.accessor = WaveformAccessor(store_root)
        self.mean = torch.tensor(scaler["mean"], dtype=torch.float32)
        self.std = torch.tensor(scaler["std"], dtype=torch.float32)
        self.use_subject_mean = subject_means is not None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        target = torch.tensor([row.sbp, row.dbp], dtype=torch.float32)
        item: dict[str, object] = {
            "ppg": self.accessor.get(str(row.waveform_file), int(row.waveform_row)),
            "target": (target - self.mean) / self.std,
            "subject_uid": str(row.subject_uid),
            "event_id": str(row.segment_uid),
            "source": str(row.source),
        }
        if self.use_subject_mean:
            subject_mean = torch.tensor(
                [row.subject_train_sbp, row.subject_train_dbp], dtype=torch.float32
            )
            item["subject_train_mean"] = (subject_mean - self.mean) / self.std
        return item


def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: dict[str, list[float]],
    *,
    subject_mean_residual: bool,
) -> pd.DataFrame:
    mean = torch.tensor(scaler["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(scaler["std"], dtype=torch.float32, device=device)
    rows: list[pd.DataFrame] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ppg = batch["ppg"].to(device, non_blocking=True)
            with _autocast(device):
                if subject_mean_residual:
                    standardized = model(
                        ppg,
                        batch["subject_train_mean"].to(device, non_blocking=True),
                    )
                else:
                    standardized = model(ppg)
            prediction = standardized.float() * std + mean
            target = batch["target"].to(device).float() * std + mean
            rows.append(
                pd.DataFrame(
                    {
                        "subject_uid": list(batch["subject_uid"]),
                        "event_id": list(batch["event_id"]),
                        "source": list(batch["source"]),
                        "target_sbp": target[:, 0].cpu().numpy(),
                        "target_dbp": target[:, 1].cpu().numpy(),
                        "pred_sbp": prediction[:, 0].cpu().numpy(),
                        "pred_dbp": prediction[:, 1].cpu().numpy(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _baseline_predictions(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = fit_subject_train_means(train)
    predicted = predict_subject_train_mean(validation, means)
    predictions = validation[
        ["subject_uid", "segment_uid", "source", "sbp", "dbp"]
    ].copy()
    predictions = predictions.rename(
        columns={
            "segment_uid": "event_id",
            "sbp": "target_sbp",
            "dbp": "target_dbp",
        }
    )
    predictions[["pred_sbp", "pred_dbp"]] = predicted[
        ["sbp_pred", "dbp_pred"]
    ].to_numpy()
    return predictions, means


def train_candidate(args: argparse.Namespace) -> dict[str, object]:
    if args.candidate not in EXECUTABLE_CANDIDATES:
        candidate = CANDIDATES.get(args.candidate)
        reason = candidate.deferred_reason if candidate else "unknown candidate"
        raise ValueError(f"candidate is not executable in the first round: {reason}")
    if args.patience != EARLY_STOPPING_PATIENCE:
        raise ValueError("first-round screening requires patience=8")
    if args.epochs != 0:
        raise ValueError("first-round screening requires epochs=0 (no epoch cap)")
    if args.examples_per_epoch < 1:
        raise ValueError("examples-per-epoch must be positive")
    seed_everything(args.seed)
    train, validation = load_screen_metadata(args.store_root, args.split_mode)
    args.output.mkdir(parents=True, exist_ok=False)
    candidate = CANDIDATES[args.candidate]
    scaler = _fit_scaler(train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if candidate.runner != "analysis_baseline" and args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required but is unavailable")

    history: list[dict[str, object]] = []
    checkpoint_path: Path | None = None
    if candidate.runner == "analysis_baseline":
        predictions, means = _baseline_predictions(train, validation)
        metrics = participant_macro_views(predictions)
        means.to_parquet(args.output / "subject_train_means.parquet", index=False)
        best_epoch = 0
        epochs_completed = 0
        stop_reason = "analytic_baseline"
        parameter_counts = {"total": 0, "trainable": 0}
    else:
        subject_mean_residual = candidate.runner == "subject_mean_residual"
        encoder = build_ppg_encoder(
            str(candidate.backbone), feature_dim=int(candidate.feature_dim or 256)
        )
        if subject_mean_residual:
            model: nn.Module = SubjectMeanResidualRegressor(encoder)
            subject_means = fit_subject_train_means(train)
        else:
            model = PopulationRegressor(encoder)
            subject_means = None
        model = model.to(device)
        training_dataset = CalBasedDataset(
            train, args.store_root, scaler, subject_means=subject_means
        )
        validation_dataset = CalBasedDataset(
            validation, args.store_root, scaler, subject_means=subject_means
        )
        sampler = RandomSampler(
            training_dataset,
            replacement=True,
            num_samples=min(args.examples_per_epoch, len(training_dataset)),
            generator=torch.Generator().manual_seed(args.seed),
        )
        loader_options = {
            "num_workers": args.workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": args.workers > 0,
        }
        training_loader = DataLoader(
            training_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            drop_last=True,
            **loader_options,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_options,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        objective = nn.HuberLoss(delta=args.huber_delta)
        best_score = float("inf")
        best_epoch = 0
        best_metrics: dict[str, dict[str, object]] | None = None
        best_predictions: pd.DataFrame | None = None
        without_improvement = 0
        checkpoint_path = args.output / "best.pt"
        for epoch in itertools.count(1):
            model.train()
            running_loss = 0.0
            examples = 0
            for batch in training_loader:
                optimizer.zero_grad(set_to_none=True)
                ppg = batch["ppg"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                with _autocast(device):
                    if subject_mean_residual:
                        output = model(
                            ppg,
                            batch["subject_train_mean"].to(
                                device, non_blocking=True
                            ),
                        )
                    else:
                        output = model(ppg)
                    loss = objective(output.float(), target.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                running_loss += float(loss.detach()) * len(target)
                examples += len(target)
            predictions = _predict(
                model,
                validation_loader,
                device,
                scaler,
                subject_mean_residual=subject_mean_residual,
            )
            metrics = participant_macro_views(predictions)
            score = float(metrics["Overall"]["mean_mae"])
            record = {
                "epoch": epoch,
                "train_huber": running_loss / max(examples, 1),
                "internal_validation": metrics,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = metrics
                best_predictions = predictions
                without_improvement = 0
                torch.save(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "candidate": candidate.name,
                        "backbone": candidate.backbone,
                        "feature_dim": candidate.feature_dim,
                        "model_state": model.state_dict(),
                        "target_scaler": scaler,
                        "epoch": epoch,
                        "internal_validation": metrics,
                        "seed": args.seed,
                        "subject_mean_residual": subject_mean_residual,
                    },
                    checkpoint_path,
                )
            else:
                without_improvement += 1
            save_json(args.output / "history.json", history)
            if without_improvement >= args.patience:
                break
        if best_metrics is None or best_predictions is None:
            raise RuntimeError("training ended without a validation checkpoint")
        predictions = best_predictions
        metrics = best_metrics
        epochs_completed = len(history)
        stop_reason = "early_stopping"
        parameter_counts = model_parameter_counts(model)

    predictions.to_parquet(
        args.output / "best_internal_validation_predictions.parquet", index=False
    )
    pooled_diagnostics(predictions, candidate.name).to_csv(
        args.output / "internal_validation_pooled_diagnostics.csv", index=False
    )
    result = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "track": "development_only_same_subject_analogue",
        "official_pulsedb_calbased_reproduction": False,
        "candidate": candidate.name,
        "runner": candidate.runner,
        "backbone": candidate.backbone,
        "split_mode": args.split_mode,
        "source_parent_split": "meta_train",
        "read_roles": list(SCREENING_READ_ROLES),
        "selection_role": "internal_validation",
        "heldout_test_accessed": False,
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "train_participants": int(train["subject_uid"].nunique()),
        "internal_validation_participants": int(
            validation["subject_uid"].nunique()
        ),
        "train_windows": int(len(train)),
        "internal_validation_windows": int(len(validation)),
        "target_scaler": scaler,
        "parameter_counts": parameter_counts,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "stop_reason": stop_reason,
        "metrics": metrics,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": (
            file_sha256(checkpoint_path) if checkpoint_path else None
        ),
        "store_manifest_sha256": file_sha256(
            args.store_root / "materialization.json"
        ),
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    save_json(args.output / "run.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=list(CANDIDATES), required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--split-mode", choices=list(SPLIT_MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--examples-per-epoch", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=0.5)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train_candidate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
