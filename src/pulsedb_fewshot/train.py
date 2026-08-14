"""Train the population, Siamese, or variable-K personalized model."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import platform

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from .models import PopulationRegressor, SiameseDeltaRegressor, VariableKPersonalizer
from .training import (
    EpisodicDataset,
    PopulationDataset,
    file_sha256,
    fit_target_scaler,
    load_store_metadata,
    participant_macro_metrics,
    predict_episodic,
    predict_population,
    save_json,
    seed_everything,
    source_tree_sha256,
)


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )


def _participant_balanced_sampler(
    metadata: pd.DataFrame, seed: int, num_samples: int
) -> WeightedRandomSampler:
    counts = metadata["subject_uid"].value_counts()
    weights = metadata["subject_uid"].map(lambda value: 1.0 / counts[value]).to_numpy()
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=min(num_samples, len(metadata)),
        replacement=True,
        generator=generator,
    )


def _episodic_balanced_sampler(
    dataset: EpisodicDataset, seed: int, num_samples: int
) -> WeightedRandomSampler:
    identities = pd.Series(dataset.participant_ids)
    counts = identities.value_counts()
    weights = identities.map(lambda value: 1.0 / counts[value]).to_numpy()
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=min(num_samples, len(dataset)),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _load_population_checkpoint(
    path: Path, device: torch.device
) -> tuple[PopulationRegressor, dict[str, list[float]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PopulationRegressor()
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint["target_scaler"]


def _epoch_numbers(max_epochs: int) -> Iterable[int]:
    """Yield one-based epoch numbers; zero means early-stopping-only training."""

    if max_epochs < 0:
        raise ValueError("epochs must be nonnegative; use 0 for no epoch-count cap")
    if max_epochs == 0:
        return itertools.count(1)
    return range(1, max_epochs + 1)


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.patience < 1:
        raise ValueError("patience must be positive")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required but is unavailable")
    metadata = load_store_metadata(args.store_root, "development")
    train_metadata = metadata.loc[metadata["split"].eq("meta_train")].copy()
    validation_metadata = metadata.loc[metadata["split"].eq("meta_validation")].copy()
    if set(train_metadata["subject_uid"]) & set(validation_metadata["subject_uid"]):
        raise AssertionError("participant leakage between training and validation")
    if args.max_train_examples is not None:
        train_metadata = train_metadata.head(args.max_train_examples).copy()
    if args.max_validation_examples is not None:
        validation_metadata = validation_metadata.head(args.max_validation_examples).copy()
    # fit_target_scaler explicitly selects meta_train rows. Passing the full
    # development metadata preserves a single audited entry point while
    # keeping meta-validation targets out of preprocessing parameters.
    scaler = fit_target_scaler(metadata)
    validation_metadata = validation_metadata.loc[
        validation_metadata["common_query"]
        | validation_metadata["support_candidate"]
    ].copy()

    siamese = args.method == "siamese"
    if args.method == "population":
        model: nn.Module = PopulationRegressor()
        train_dataset = PopulationDataset(train_metadata, args.store_root, scaler)
        validation_dataset = PopulationDataset(
            validation_metadata.loc[validation_metadata["common_query"]].copy(),
            args.store_root,
            scaler,
        )
        sampler = _participant_balanced_sampler(
            train_metadata, args.seed, args.episodes_per_epoch
        )
    else:
        if args.population_checkpoint is None:
            raise ValueError("personalized methods require --population-checkpoint")
        population, checkpoint_scaler = _load_population_checkpoint(
            args.population_checkpoint, device
        )
        if checkpoint_scaler != scaler:
            raise AssertionError("target scaler differs from population checkpoint")
        if siamese:
            model = SiameseDeltaRegressor(population.encoder)
            episodic_ks = (1,)
        else:
            configuration = {
                "m0": (False, False),
                "m1": (True, False),
                "m2": (True, True),
            }
            use_film, attention = configuration[args.method]
            model = VariableKPersonalizer(
                population, use_film=use_film, query_conditioned_weights=attention
            )
            for parameter in model.population.parameters():
                parameter.requires_grad = False
            episodic_ks = (1, 2, 3, 5)
        train_dataset = EpisodicDataset(
            train_metadata,
            args.store_root,
            scaler,
            ks=episodic_ks,
            rolling_support=not siamese,
        )
        validation_dataset = EpisodicDataset(
            validation_metadata,
            args.store_root,
            scaler,
            ks=episodic_ks,
            rolling_support=False,
        )
        sampler = _episodic_balanced_sampler(
            train_dataset, args.seed, args.episodes_per_epoch
        )

    model = model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.MSELoss()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, object]] = []
    best_score = float("inf")
    best_epoch: int | None = None
    epochs_without_improvement = 0
    stop_reason = "epoch_cap"
    checkpoint_path = args.output / "best.pt"
    for epoch in _epoch_numbers(args.epochs):
        model.train()
        if isinstance(model, VariableKPersonalizer):
            # The population mapping is a frozen reference for residual
            # anchoring. Keep BatchNorm statistics and dropout frozen too.
            model.population.eval()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                if args.method == "population":
                    prediction = model(batch["ppg"].to(device, non_blocking=True))
                elif siamese:
                    prediction = model(
                        batch["query_ppg"].to(device, non_blocking=True),
                        batch["support_ppg"][:, 0].to(device, non_blocking=True),
                        batch["support_bp"][:, 0].to(device, non_blocking=True),
                    )
                else:
                    prediction = model(
                        batch["query_ppg"].to(device, non_blocking=True),
                        batch["support_ppg"].to(device, non_blocking=True),
                        batch["support_bp"].to(device, non_blocking=True),
                        batch["support_mask"].to(device, non_blocking=True),
                    )
                loss = loss_function(prediction.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total_examples += len(target)

        if args.method == "population":
            predictions = predict_population(
                model, validation_loader, device, scaler
            )
            metrics = participant_macro_metrics(predictions)
        else:
            predictions = predict_episodic(
                model, validation_loader, device, scaler, siamese=siamese
            )
            by_k = {
                str(k): participant_macro_metrics(group)
                for k, group in predictions.groupby("k")
            }
            metrics = {
                "mean_mae": float(
                    sum(item["mean_mae"] for item in by_k.values()) / len(by_k)
                ),
                "by_k": by_k,
            }
        record = {
            "epoch": epoch,
            "train_mse": total_loss / max(total_examples, 1),
            "validation": metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        score = float(metrics["mean_mae"])
        if score < best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "method": args.method,
                    "model_state": model.state_dict(),
                    "target_scaler": scaler,
                    "epoch": epoch,
                    "validation": metrics,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
            predictions.to_parquet(args.output / "best_validation_predictions.parquet", index=False)
        else:
            epochs_without_improvement += 1
        save_json(args.output / "history.json", history)
        if epochs_without_improvement >= args.patience:
            stop_reason = "early_stopping"
            break

    if best_epoch is None:
        raise RuntimeError("training completed without a valid checkpoint")

    result = {
        "status": "complete",
        "method": args.method,
        "split": "meta_validation",
        "locked_test_accessed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "train_participants": int(train_metadata["subject_uid"].nunique()),
        "validation_participants": int(validation_metadata["subject_uid"].nunique()),
        "train_events": int(len(train_metadata)),
        "validation_events": int(len(validation_metadata)),
        "target_scaler": scaler,
        "best_validation_mean_mae": best_score,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "store_manifest_sha256": file_sha256(args.store_root / "materialization.json"),
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "arguments": vars(args) | {
            "store_root": str(args.store_root),
            "output": str(args.output),
            "population_checkpoint": str(args.population_checkpoint)
            if args.population_checkpoint
            else None,
        },
    }
    save_json(args.output / "run.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["population", "siamese", "m0", "m1", "m2"], required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="maximum epochs; use 0 for no epoch-count cap and stop by patience",
    )
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--episodes-per-epoch", type=int, default=200000)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
