"""Evaluate simple calibration baselines and per-subject adaptation on meta-validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from .models import PopulationRegressor, configure_personal_adaptation
from .training import (
    PopulationDataset,
    WaveformAccessor,
    load_store_metadata,
    participant_macro_metrics,
    save_json,
    seed_everything,
)


KS = (1, 2, 3, 5)


def _load_population(
    path: Path, device: torch.device
) -> tuple[PopulationRegressor, dict[str, list[float]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PopulationRegressor().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["target_scaler"]


@torch.no_grad()
def _predict_rows(
    model: nn.Module,
    frame: pd.DataFrame,
    accessor: WaveformAccessor,
    device: torch.device,
    scaler: dict[str, list[float]],
) -> torch.Tensor:
    mean = torch.tensor(scaler["mean"], device=device)
    std = torch.tensor(scaler["std"], device=device)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(frame), 256):
        batch = frame.iloc[start : start + 256]
        waveforms = torch.stack(
            [
                accessor.get(row.waveform_file, int(row.waveform_row))
                for row in batch.itertuples(index=False)
            ]
        ).to(device)
        chunks.append((model(waveforms) * std + mean).cpu())
    return torch.cat(chunks, dim=0)


def _fine_tune_subject(
    population: PopulationRegressor,
    support: pd.DataFrame,
    accessor: WaveformAccessor,
    device: torch.device,
    scaler: dict[str, list[float]],
    *,
    mode: str,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    lora_rank: int,
) -> PopulationRegressor:
    model = configure_personal_adaptation(
        population, mode, lora_rank=lora_rank
    ).to(device)
    # K is as small as one independent event. Keep dropout disabled and batch-
    # normalization running statistics frozen; only parameters explicitly
    # exposed by the adaptation mode may change.
    model.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    inputs = torch.stack(
        [
            accessor.get(row.waveform_file, int(row.waveform_row))
            for row in support.itertuples(index=False)
        ]
    ).to(device)
    mean = torch.tensor(scaler["mean"], device=device)
    std = torch.tensor(scaler["std"], device=device)
    targets_mm = torch.tensor(
        support[["sbp", "dbp"]].to_numpy(), dtype=torch.float32, device=device
    )
    targets = (targets_mm - mean) / std
    for _ in range(steps):
        prediction = model(inputs)
        loss = nn.functional.mse_loss(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
        optimizer.step()
    return model.eval()


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required")
    metadata = load_store_metadata(args.store_root, "development")
    validation = metadata.loc[metadata["split"].eq("meta_validation")].copy()
    population, scaler = _load_population(args.population_checkpoint, device)
    accessor = WaveformAccessor(args.store_root)
    rows: list[dict[str, object]] = []

    for subject_uid, subject in validation.groupby("subject_uid", sort=False):
        subject = subject.sort_values("event_index", kind="mergesort")
        query = subject.loc[subject["common_query"]].copy()
        if query.empty:
            continue
        population_query = _predict_rows(
            population, query, accessor, device, scaler
        ).cpu()
        targets = torch.tensor(
            query[["sbp", "dbp"]].to_numpy(), dtype=torch.float32
        )
        for k in KS:
            support = subject.loc[subject[f"role_k{k}"].eq("support")].copy()
            population_support = _predict_rows(
                population, support, accessor, device, scaler
            ).cpu()
            support_targets = torch.tensor(
                support[["sbp", "dbp"]].to_numpy(), dtype=torch.float32
            )
            simple = {
                "population_mean": torch.tensor(
                    scaler["mean"], dtype=torch.float32
                )[None, :].repeat(len(query), 1),
                "population": population_query,
                "last_cuff": support_targets[-1:].repeat(len(query), 1),
                "support_mean": support_targets.mean(dim=0, keepdim=True).repeat(len(query), 1),
                "residual_offset": population_query
                + (support_targets - population_support).mean(dim=0, keepdim=True),
            }
            adapted: dict[str, torch.Tensor] = {}
            for mode in ("head", "full", "lora"):
                adapted_model = _fine_tune_subject(
                    population,
                    support,
                    accessor,
                    device,
                    scaler,
                    mode=mode,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    lora_rank=args.lora_rank,
                )
                adapted[mode] = _predict_rows(
                    adapted_model, query, accessor, device, scaler
                )
                del adapted_model
            all_predictions = {**simple, **adapted}
            for method, predictions in all_predictions.items():
                for index, event in enumerate(query.itertuples(index=False)):
                    rows.append(
                        {
                            "method": method,
                            "k": k,
                            "subject_uid": subject_uid,
                            "event_id": event.event_id,
                            "target_sbp": float(targets[index, 0]),
                            "target_dbp": float(targets[index, 1]),
                            "pred_sbp": float(predictions[index, 0]),
                            "pred_dbp": float(predictions[index, 1]),
                        }
                    )

    predictions = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(args.output / "validation_predictions.parquet", index=False)
    metrics = {
        method: {
            str(k): participant_macro_metrics(group)
            for k, group in method_frame.groupby("k")
        }
        for method, method_frame in predictions.groupby("method")
    }
    result = {
        "status": "complete",
        "split": "meta_validation",
        "locked_test_accessed": False,
        "population_checkpoint": str(args.population_checkpoint),
        "adaptation": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lora_rank": args.lora_rank,
        },
        "metrics": metrics,
    }
    save_json(args.output / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--population-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
