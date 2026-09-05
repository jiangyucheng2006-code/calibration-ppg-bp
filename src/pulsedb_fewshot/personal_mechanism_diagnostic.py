"""Frozen-model diagnostics of waveform alignment and persistent personal state.

Only train/internal_validation roles are readable. Disturbance maps use IDs,
source and seeded random numbers, never BP. Cached encoder features avoid
repeated waveform I/O; natural predictions must reproduce the saved run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .calbased_train import load_screen_metadata, _autocast
from .calbased_screen import fit_subject_train_means
from .calbased_metrics import participant_macro_views, pooled_diagnostics
from .same_subject_component_train import SameSubjectComponentDataset, _model_call
from .same_subject_personal_profiles import PERSONAL_PROFILES, SameSubjectPersonalProfileRegressor
from .training import file_sha256, save_json, seed_everything


def grouped_derangement(groups, seed: int) -> np.ndarray:
    """Return donor indices with identical group and no fixed points."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    result = np.arange(len(groups))
    for group in sorted(set(groups)):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            raise ValueError("derangement group must contain at least two entries")
        order = rng.permutation(indices)
        result[order] = np.roll(order, int(rng.integers(1, len(order))))
    assert np.all(result != np.arange(len(result)))
    assert np.all(groups[result] == groups)
    return result


def run(args) -> dict:
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("frozen encoder diagnostics require a scheduled GPU")
    device = torch.device("cuda")
    metadata = json.loads((args.run / "run.json").read_text())
    if metadata["heldout_test_accessed"] or metadata["selection_role"] != "internal_validation":
        raise ValueError("only development checkpoints are supported")
    candidate = metadata["candidate"]
    if candidate not in {"subject_lora_rank4", "personal_profile_code32_no_support"}:
        raise ValueError("this frozen diagnostic supports the declared no-support pair only")
    checkpoint_path = args.run / "best.pt"
    assert file_sha256(checkpoint_path) == metadata["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scaler = checkpoint["target_scaler"]
    indices = checkpoint["subject_to_index"]
    train, validation = load_screen_metadata(args.store_root, metadata["split_mode"])
    means = fit_subject_train_means(train)
    validation = validation.copy()
    validation["quality_weight"] = 1.0
    assert sorted(indices) == sorted(train.subject_uid.unique())
    dataset = SameSubjectComponentDataset(validation, args.store_root, scaler, means, indices)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model = SameSubjectPersonalProfileRegressor(PERSONAL_PROFILES[candidate], subject_count=len(indices)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    encoder = model.reference.encoder if hasattr(model, "reference") else model.encoder
    parts = {k: [] for k in ["features", "descriptor", "anchor", "subject_index", "natural"]}
    captured = {}
    handle = encoder.register_forward_hook(lambda _m, _a, output: captured.update(features=output.detach().cpu()))
    mean = torch.tensor(scaler["mean"], device=device)
    std = torch.tensor(scaler["std"], device=device)
    with torch.inference_mode():
        for batch in loader:
            with _autocast(device):
                output = _model_call(model, batch, device)
            parts["features"].append(captured["features"])
            parts["descriptor"].append(batch["query_descriptor"])
            parts["anchor"].append(batch["subject_train_mean"])
            parts["subject_index"].append(batch["subject_index"])
            parts["natural"].append((output.float() * std + mean).cpu())
    handle.remove()
    cached = {key: torch.cat(values) for key, values in parts.items()}
    keys = pd.DataFrame({"subject_uid": validation.subject_uid.astype(str), "event_id": validation.segment_uid.astype(str), "source": validation.source.astype(str)})
    saved = pd.read_parquet(args.run / "best_internal_validation_predictions.parquet")
    joined = keys.merge(saved, on=["subject_uid", "event_id", "source"], how="left", validate="one_to_one")
    assert len(joined) == 82040 and not joined.isna().any().any()
    assert np.allclose(joined[["target_sbp", "target_dbp"]], validation[["sbp", "dbp"]], atol=1e-4)
    baseline_difference = np.abs(cached["natural"].numpy() - joined[["pred_sbp", "pred_dbp"]].to_numpy())
    if baseline_difference.max() > 0.05 or baseline_difference.mean() > 0.005:
        raise ValueError(f"checkpoint reproduction failed: max={baseline_difference.max()}, mean={baseline_difference.mean()}")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = train.groupby("subject_uid").source.first().reindex(sorted(indices, key=indices.get)).to_numpy()
    swapped = grouped_derangement(sources, args.seed + 2)
    maps = {"ppg_permuted_1": grouped_derangement(keys.subject_uid, args.seed),
            "ppg_permuted_2": grouped_derangement(keys.subject_uid, args.seed + 1)}
    cached_forward = encoder.forward
    holder = {}
    encoder.forward = lambda _x: holder["features"]
    states = (model.reference.lora_b,) if hasattr(model, "reference") else (model.subject_code, model.subject_bias)
    original = [module.weight.detach().clone() for module in states]
    summaries, diagnostics = [], []
    conditions = ["cached_identity", "natural", "ppg_permuted_1", "ppg_permuted_2", "personal_state_swapped", "personal_state_zero", "train_bp_mean", "train_bp_median"]
    cache_max_difference = None
    try:
        with torch.inference_mode():
            for condition in conditions:
                if condition == "natural":
                    prediction = cached["natural"].numpy()
                elif condition in {"train_bp_mean", "train_bp_median"}:
                    if condition == "train_bp_mean":
                        prediction = (cached["anchor"] * std.cpu() + mean.cpu()).numpy()
                    else:
                        medians = train.groupby("subject_uid")[["sbp", "dbp"]].median()
                        prediction = medians.reindex(keys.subject_uid).to_numpy(dtype=np.float32)
                else:
                    if condition == "personal_state_zero":
                        for module in states:
                            module.weight.zero_()
                    donor = maps.get(condition, np.arange(len(keys)))
                    outputs = []
                    for start in range(0, len(keys), 64):
                        end = min(start + 64, len(keys))
                        take = torch.tensor(donor[start:end], dtype=torch.long)
                        person = cached["subject_index"][start:end].clone()
                        if condition == "personal_state_swapped":
                            person = torch.tensor(swapped[person.numpy()], dtype=torch.long)
                        holder["features"] = cached["features"][take].to(device)
                        with _autocast(device):
                            y = model(torch.zeros(end-start, 1, 1250, device=device),
                                      cached["anchor"][start:end].to(device),
                                      query_descriptor=cached["descriptor"][take].to(device),
                                      subject_index=person.to(device))
                        outputs.append((y.float() * std + mean).cpu())
                    prediction = torch.cat(outputs).numpy()
                    for module, weights in zip(states, original):
                        module.weight.copy_(weights)
                if condition == "cached_identity":
                    cache_difference = np.abs(prediction - cached["natural"].numpy())
                    cache_max_difference = float(cache_difference.max())
                    if cache_max_difference > 0.05 or cache_difference.mean() > 0.005:
                        raise ValueError("cached-head identity reproduction failed")
                    continue
                frame = joined[["subject_uid", "event_id", "source", "target_sbp", "target_dbp"]].copy()
                frame[["pred_sbp", "pred_dbp"]] = prediction
                frame.to_parquet(args.output / f"{condition}_predictions.parquet", index=False)
                views = participant_macro_views(frame)
                for scope, values in views.items():
                    summaries.append({"candidate": candidate, "split_mode": metadata["split_mode"], "condition": condition, "Scope": scope, **values})
                diag = pooled_diagnostics(frame, condition)
                diag.insert(0, "candidate", candidate)
                diag.insert(0, "split_mode", metadata["split_mode"])
                diagnostics.append(diag)
                print(json.dumps({"condition": condition, "mean_mae": views["Overall"]["mean_mae"]}), flush=True)
    finally:
        encoder.forward = cached_forward
        with torch.no_grad():
            for module, weights in zip(states, original):
                module.weight.copy_(weights)
    pd.DataFrame(summaries).to_csv(args.output / "participant_macro.csv", index=False)
    pd.concat(diagnostics).to_csv(args.output / "pooled_diagnostics.csv", index=False)
    result = {"status": "complete", "candidate": candidate, "split_mode": metadata["split_mode"],
              "seed": args.seed, "heldout_test_accessed": False, "parameters_updated": False,
              "natural_max_absolute_difference": float(baseline_difference.max()),
              "natural_mean_absolute_difference": float(baseline_difference.mean()),
              "cache_max_absolute_difference": cache_max_difference,
              "gpu": torch.cuda.get_device_name(0),
              "source_checkpoint_sha256": metadata["checkpoint_sha256"], "conditions": conditions,
              "ppg_derangements_use_labels": False, "personal_swap_within_source": True,
              "anchor_preserved_during_personal_swap": True,
              "interpretation": "Post-hoc frozen-model sensitivity; not equivalent to retrained ablations or held-out confirmation."}
    save_json(args.output / "diagnostic.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
