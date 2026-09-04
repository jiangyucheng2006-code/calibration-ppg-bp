"""Train one isolated component on the same-subject PulseDB development track."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
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

from .calbased_metrics import participant_macro_views, pooled_diagnostics
from .calbased_screen import (
    EARLY_STOPPING_PATIENCE,
    PROTOCOL_ID,
    SCREENING_READ_ROLES,
    fit_subject_train_means,
)
from .calbased_train import _autocast, _fit_scaler, load_screen_metadata
from .models import model_parameter_counts
from .same_subject_components import (
    COMPONENTS,
    DESCRIPTOR_DIM,
    SCREEN_ID,
    SUPPORT_COUNT,
    SUPPORT_RESERVE,
    SameSubjectComponentRegressor,
    waveform_descriptor,
)
from .training import (
    WaveformAccessor,
    file_sha256,
    save_json,
    seed_everything,
    source_tree_sha256,
)


DEMOGRAPHIC_COLUMNS = [
    "age_z",
    "age_valid",
    "sex_female",
    "sex_male",
    "sex_unknown",
]


def fit_quality_proxy(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Fit a conservative raw-PPG amplitude proxy using train-role rows only."""

    frame = train.copy()
    values = pd.to_numeric(frame["ppg_f_std"], errors="raise").astype(float)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("ppg_f_std must be finite and positive")
    frame["_log_ppg_std"] = np.log(values)
    parameters: dict[str, dict[str, float]] = {}
    scores = pd.Series(index=frame.index, dtype=float)
    for source, indices in frame.groupby("source", sort=True).groups.items():
        source_values = frame.loc[indices, "_log_ppg_std"].to_numpy(dtype=float)
        center = float(np.median(source_values))
        mad = float(np.median(np.abs(source_values - center)))
        scale = max(1.4826 * mad, 1e-6)
        score = np.exp(-np.abs(source_values - center) / (3.0 * scale))
        scores.loc[indices] = np.clip(score, 0.05, 1.0)
        parameters[str(source)] = {"log_std_median": center, "robust_scale": scale}
    frame["quality_weight"] = scores.to_numpy(dtype=np.float32)
    frame = frame.drop(columns="_log_ppg_std")
    return frame, parameters


def apply_training_rule(
    train: pd.DataFrame,
    rule: str,
    *,
    beat_similarity_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    original_rows = int(len(train))
    result = train.copy()
    details: dict[str, object] = {
        "rule": rule,
        "original_train_rows": original_rows,
        "validation_coverage_changed": False,
    }
    if rule == "quality_filter":
        keep = pd.Series(False, index=result.index)
        thresholds: dict[str, float] = {}
        for source, indices in result.groupby("source", sort=True).groups.items():
            threshold = float(result.loc[indices, "quality_weight"].quantile(0.10))
            thresholds[str(source)] = threshold
            keep.loc[indices] = result.loc[indices, "quality_weight"] > threshold
        result = result.loc[keep].copy()
        details["source_thresholds"] = thresholds
    elif rule == "beat_similarity_filter":
        if beat_similarity_path is None or not beat_similarity_path.is_file():
            raise FileNotFoundError("beat-similarity training rule requires its PPG-only table")
        similarity = pd.read_parquet(beat_similarity_path)
        required = {"subject_uid", "segment_uid", "median_pairwise_similarity", "valid"}
        missing = required - set(similarity.columns)
        if missing:
            raise ValueError(f"beat-similarity table missing columns: {sorted(missing)}")
        if similarity[["subject_uid", "segment_uid"]].duplicated().any():
            raise ValueError("beat-similarity table contains duplicate train windows")
        result = result.merge(
            similarity[list(required)],
            on=["subject_uid", "segment_uid"],
            how="left",
            validate="one_to_one",
        )
        if result["valid"].isna().any():
            raise ValueError("beat-similarity table does not cover every train window")
        result = result.loc[
            result["valid"].astype(bool)
            & pd.to_numeric(result["median_pairwise_similarity"], errors="coerce").ge(0.90)
        ].copy()
        details["threshold"] = 0.90
    elif rule not in {"standard", "quality_weighted_loss"}:
        raise ValueError(f"unsupported training rule {rule!r}")
    details["retained_train_rows"] = int(len(result))
    details["retained_fraction"] = float(len(result) / original_rows)
    if result.empty:
        raise ValueError("training rule removed every train window")
    return result, details


def _even_support_rows(subject_rows: pd.DataFrame) -> pd.DataFrame:
    ordered = subject_rows.sort_values(
        ["selection_rank", "segment_uid"], kind="mergesort"
    ).reset_index(drop=True)
    required = SUPPORT_COUNT + SUPPORT_RESERVE
    if len(ordered) < required:
        raise ValueError("participant has too few train-role windows for support bank")
    positions = np.linspace(0, len(ordered) - 1, num=required)
    positions = np.rint(positions).astype(int)
    if len(set(positions.tolist())) != required:
        raise ValueError("deterministic support positions are not unique")
    return ordered.iloc[positions].reset_index(drop=True)


def build_support_bank(
    train: pd.DataFrame,
    store_root: Path,
    scaler: dict[str, list[float]],
) -> dict[str, dict[str, object]]:
    """Build six input-selected train supports; each query receives five non-self supports."""

    accessor = WaveformAccessor(store_root)
    mean = torch.tensor(scaler["mean"], dtype=torch.float32)
    std = torch.tensor(scaler["std"], dtype=torch.float32)
    bank: dict[str, dict[str, object]] = {}
    for subject_uid, rows in train.groupby("subject_uid", sort=True):
        selected = _even_support_rows(rows)
        descriptors = []
        for row in selected.itertuples(index=False):
            ppg = accessor.get(str(row.waveform_file), int(row.waveform_row))
            descriptors.append(waveform_descriptor(ppg).squeeze(0))
        bp = torch.tensor(selected[["sbp", "dbp"]].to_numpy(), dtype=torch.float32)
        bank[str(subject_uid)] = {
            "segment_uids": selected["segment_uid"].astype(str).tolist(),
            "descriptors": torch.stack(descriptors),
            "bp": (bp - mean) / std,
        }
    return bank


class SameSubjectComponentDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        store_root: Path,
        scaler: dict[str, list[float]],
        subject_means: pd.DataFrame,
        subject_indices: dict[str, int],
        *,
        support_bank: dict[str, dict[str, object]] | None = None,
        demographics: pd.DataFrame | None = None,
    ) -> None:
        frame = metadata.copy()
        frame["subject_uid"] = frame["subject_uid"].astype(str)
        frame = frame.merge(subject_means, on="subject_uid", how="left", validate="many_to_one")
        if frame[["subject_train_sbp", "subject_train_dbp"]].isna().any().any():
            raise ValueError("train-role subject means do not cover component dataset")
        if demographics is not None:
            demo = demographics[["subject_uid", *DEMOGRAPHIC_COLUMNS]].copy()
            demo["subject_uid"] = demo["subject_uid"].astype(str)
            frame = frame.merge(demo, on="subject_uid", how="left", validate="many_to_one")
            if frame[DEMOGRAPHIC_COLUMNS].isna().any().any():
                raise ValueError("cleaned demographics do not cover component dataset")
        self.metadata = frame.reset_index(drop=True)
        self.accessor = WaveformAccessor(store_root)
        self.mean = torch.tensor(scaler["mean"], dtype=torch.float32)
        self.std = torch.tensor(scaler["std"], dtype=torch.float32)
        self.subject_indices = subject_indices
        self.support_bank = support_bank
        self.use_demographics = demographics is not None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        subject_uid = str(row.subject_uid)
        ppg = self.accessor.get(str(row.waveform_file), int(row.waveform_row))
        target = torch.tensor([row.sbp, row.dbp], dtype=torch.float32)
        anchor = torch.tensor(
            [row.subject_train_sbp, row.subject_train_dbp], dtype=torch.float32
        )
        item: dict[str, object] = {
            "ppg": ppg,
            "query_descriptor": waveform_descriptor(ppg).squeeze(0),
            "target": (target - self.mean) / self.std,
            "subject_train_mean": (anchor - self.mean) / self.std,
            "quality_weight": torch.tensor(float(row.quality_weight), dtype=torch.float32),
            "subject_index": torch.tensor(self.subject_indices[subject_uid], dtype=torch.long),
            "subject_uid": subject_uid,
            "event_id": str(row.segment_uid),
            "source": str(row.source),
        }
        if self.support_bank is not None:
            support = self.support_bank[subject_uid]
            identifiers = support["segment_uids"]
            indices = [i for i, value in enumerate(identifiers) if value != str(row.segment_uid)]
            indices = indices[:SUPPORT_COUNT]
            if len(indices) != SUPPORT_COUNT:
                raise ValueError("failed to construct five non-self train supports")
            item["support_descriptors"] = support["descriptors"][indices]
            item["support_bp"] = support["bp"][indices]
        if self.use_demographics:
            item["demographics"] = torch.tensor(
                row[DEMOGRAPHIC_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
            )
        return item


def _model_call(model: nn.Module, batch: dict[str, object], device: torch.device) -> torch.Tensor:
    optional: dict[str, torch.Tensor] = {}
    for key in (
        "query_descriptor",
        "support_descriptors",
        "support_bp",
        "subject_index",
        "demographics",
    ):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            optional[key] = value.to(device, non_blocking=True)
    return model(
        batch["ppg"].to(device, non_blocking=True),
        batch["subject_train_mean"].to(device, non_blocking=True),
        **optional,
    )


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: dict[str, list[float]],
) -> pd.DataFrame:
    mean = torch.tensor(scaler["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(scaler["std"], dtype=torch.float32, device=device)
    rows: list[pd.DataFrame] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with _autocast(device):
                standardized = _model_call(model, batch, device)
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


def train_component(
    args: argparse.Namespace,
    *,
    components: Mapping[str, object] = COMPONENTS,
    model_factory: Callable[..., nn.Module] = SameSubjectComponentRegressor,
    screen_id: str = SCREEN_ID,
    runner: str = "single_component_residual",
    allowed_split_modes: Sequence[str] = ("random_disjoint",),
) -> dict[str, object]:
    """Train one residual candidate under an explicitly supplied registry.

    The defaults preserve the frozen single-component screen.  A separate
    combination runner supplies its own registry/model/screen identifier while
    reusing the same audited data, optimization, prediction, and artifact path.
    """

    if args.candidate not in components:
        raise ValueError(f"unknown component candidate {args.candidate!r}")
    if args.split_mode not in set(allowed_split_modes):
        raise ValueError(
            f"split_mode {args.split_mode!r} is not allowed for screen {screen_id!r}"
        )
    if args.epochs != 0 or args.patience != EARLY_STOPPING_PATIENCE:
        raise ValueError("component screen requires no epoch cap and patience=8")
    seed_everything(args.seed)
    full_train, validation = load_screen_metadata(args.store_root, args.split_mode)
    scaler = _fit_scaler(full_train)
    subject_means = fit_subject_train_means(full_train)
    train_with_quality, quality_parameters = fit_quality_proxy(full_train)
    validation_with_quality, _ = fit_quality_proxy_for_validation(
        validation, quality_parameters
    )
    spec = components[args.candidate]
    filtered_train, training_rule = apply_training_rule(
        train_with_quality,
        spec.training_rule,
        beat_similarity_path=args.beat_similarity_path,
    )
    subject_ids = sorted(full_train["subject_uid"].astype(str).unique())
    subject_indices = {subject: index for index, subject in enumerate(subject_ids)}
    demographics = None
    if spec.uses_demographics:
        if args.demographics_path is None or not args.demographics_path.is_file():
            raise FileNotFoundError("demographic component requires cleaned demographics")
        demographics = pd.read_parquet(args.demographics_path)
        required = {"subject_uid", "split", *DEMOGRAPHIC_COLUMNS}
        missing = required - set(demographics.columns)
        if missing:
            raise ValueError(f"demographic table missing columns: {sorted(missing)}")
        demographics = demographics.loc[demographics["split"].eq("meta_train")].copy()
        if not set(subject_ids).issubset(set(demographics["subject_uid"].astype(str))):
            raise ValueError("demographic table does not cover all seen participants")
    support_bank = (
        build_support_bank(full_train, args.store_root, scaler) if spec.uses_support else None
    )

    args.output.mkdir(parents=True, exist_ok=False)
    subject_index_path = args.output / "subject_index.json"
    save_json(
        subject_index_path,
        {
            "protocol_id": PROTOCOL_ID,
            "screen_id": screen_id,
            "split_mode": args.split_mode,
            "source_parent_split": "meta_train",
            "read_roles": list(SCREENING_READ_ROLES),
            "heldout_test_accessed": False,
            "subject_count": len(subject_ids),
            "subject_to_index": subject_indices,
        },
    )
    participant_profile = subject_means.copy()
    participant_profile["subject_uid"] = participant_profile["subject_uid"].astype(str)
    participant_profile["subject_index"] = participant_profile["subject_uid"].map(
        subject_indices
    )
    if participant_profile["subject_index"].isna().any():
        raise ValueError("participant profile could not be mapped to subject indices")
    participant_profile["subject_index"] = participant_profile["subject_index"].astype(int)
    if support_bank is not None:
        participant_profile["support_segment_uids"] = participant_profile[
            "subject_uid"
        ].map(
            lambda subject: json.dumps(
                support_bank[str(subject)]["segment_uids"], separators=(",", ":")
            )
        )
    participant_profile = participant_profile.sort_values(
        "subject_index", kind="mergesort"
    ).reset_index(drop=True)
    participant_profile_path = args.output / "participant_profile_index.parquet"
    participant_profile.to_parquet(participant_profile_path, index=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required but is unavailable")
    model = model_factory(spec, subject_count=len(subject_ids)).to(device)
    train_dataset = SameSubjectComponentDataset(
        filtered_train,
        args.store_root,
        scaler,
        subject_means,
        subject_indices,
        support_bank=support_bank,
        demographics=demographics,
    )
    validation_dataset = SameSubjectComponentDataset(
        validation_with_quality,
        args.store_root,
        scaler,
        subject_means,
        subject_indices,
        support_bank=support_bank,
        demographics=demographics,
    )
    sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=min(args.examples_per_epoch, len(train_dataset)),
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
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
    objective = nn.HuberLoss(delta=args.huber_delta, reduction="none")
    best_score = float("inf")
    best_epoch = 0
    best_predictions: pd.DataFrame | None = None
    best_metrics: dict[str, dict[str, object]] | None = None
    without_improvement = 0
    history: list[dict[str, object]] = []
    checkpoint_path = args.output / "best.pt"
    for epoch in itertools.count(1):
        model.train()
        running_loss = 0.0
        examples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = batch["target"].to(device, non_blocking=True)
            with _autocast(device):
                output = _model_call(model, batch, device)
                per_example = objective(output.float(), target.float()).mean(dim=1)
                if spec.training_rule == "quality_weighted_loss":
                    weights = batch["quality_weight"].to(device, non_blocking=True)
                    loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-6)
                else:
                    loss = per_example.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(target)
            examples += len(target)
        predictions = predict(model, validation_loader, device, scaler)
        metrics = participant_macro_views(predictions)
        score = float(metrics["Overall"]["mean_mae"])
        record = {
            "epoch": epoch,
            "train_objective": running_loss / max(examples, 1),
            "internal_validation": metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        save_json(args.output / "history.json", history)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_predictions = predictions
            best_metrics = metrics
            without_improvement = 0
            torch.save(
                {
                    "protocol_id": PROTOCOL_ID,
                    "screen_id": screen_id,
                    "candidate": spec.name,
                    "backbone": spec.backbone,
                    "adapter": spec.adapter,
                    "modules": list(getattr(spec, "modules", (spec.adapter,))),
                    "model_state": model.state_dict(),
                    "target_scaler": scaler,
                    "epoch": epoch,
                    "seed": args.seed,
                    "metrics": metrics,
                    "subject_to_index": subject_indices,
                },
                checkpoint_path,
            )
        else:
            without_improvement += 1
        if without_improvement >= args.patience:
            break
    if best_predictions is None or best_metrics is None:
        raise RuntimeError("component training ended without a checkpoint")
    best_predictions.to_parquet(
        args.output / "best_internal_validation_predictions.parquet", index=False
    )
    pooled_diagnostics(best_predictions, spec.name).to_csv(
        args.output / "internal_validation_pooled_diagnostics.csv", index=False
    )
    run = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "screen_id": screen_id,
        "track": "development_only_same_subject_analogue",
        "official_pulsedb_calbased_reproduction": False,
        "candidate": spec.name,
        "runner": runner,
        "backbone": spec.backbone,
        "adapter": spec.adapter,
        "modules": list(getattr(spec, "modules", (spec.adapter,))),
        "training_rule": training_rule,
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
        "train_participants": int(full_train["subject_uid"].nunique()),
        "train_windows_available": int(len(full_train)),
        "train_windows_used": int(len(filtered_train)),
        "internal_validation_participants": int(validation["subject_uid"].nunique()),
        "internal_validation_windows": int(len(validation)),
        "quality_proxy_fit": quality_parameters,
        "support_count": SUPPORT_COUNT if spec.uses_support else 0,
        "support_selection": (
            "train-role selection_rank evenly spaced, six reserved, five non-self used"
            if spec.uses_support
            else None
        ),
        "participant_trainable_parameters": int(
            getattr(spec, "participant_trainable_parameters", 0)
        ),
        "personal_state_storage": {
            "subject_index": str(subject_index_path),
            "subject_index_sha256": file_sha256(subject_index_path),
            "participant_profile_index": str(participant_profile_path),
            "participant_profile_index_sha256": file_sha256(
                participant_profile_path
            ),
            "participant_profile_contains_train_role_bp_anchor": True,
            "participant_profile_contains_support_event_ids": support_bank is not None,
            "checkpoint_contains_subject_to_index": True,
            "checkpoint_contains_subject_specific_state": bool(
                getattr(spec, "uses_subject_index", False)
            ),
        },
        "parameter_counts": model_parameter_counts(model),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stop_reason": "early_stopping",
        "metrics": best_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "store_manifest_sha256": file_sha256(args.store_root / "materialization.json"),
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    save_json(args.output / "run.json", run)
    return run


def fit_quality_proxy_for_validation(
    validation: pd.DataFrame,
    parameters: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, None]:
    frame = validation.copy()
    values = pd.to_numeric(frame["ppg_f_std"], errors="raise").astype(float)
    weights = np.empty(len(frame), dtype=np.float32)
    for source, indices in frame.groupby("source", sort=True).groups.items():
        if str(source) not in parameters:
            raise ValueError("validation source was absent from train quality fit")
        center = parameters[str(source)]["log_std_median"]
        scale = parameters[str(source)]["robust_scale"]
        score = np.exp(-np.abs(np.log(values.loc[indices]) - center) / (3.0 * scale))
        weights[frame.index.get_indexer(indices)] = np.clip(score, 0.05, 1.0)
    frame["quality_weight"] = weights
    return frame, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=list(COMPONENTS), required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--split-mode", default="random_disjoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demographics-path", type=Path)
    parser.add_argument("--beat-similarity-path", type=Path)
    parser.add_argument("--seed", type=int, default=20260831)
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
    print(json.dumps(train_component(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
