"""E1: physical-gap-v2 training donors, with unchanged v1 validation and alpha.

This isolates removal of the training-only 40-window block from the permitted
donor mask. Identity, waveform hashes, physical overlap and chronological
ordering remain audited. Frozen supervised LoRA features are NOT OOF features.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .personal_memory_prepare import FIELDS, audit_metadata, prepare_neighbors, validate_export_manifest


SCREEN_ID = "personal-memory-v2"
CANDIDATE = "E1_matched_relation"
POLICY = "physical-gap-v2"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "q25": float(np.quantile(values, .25)) if len(values) else None,
            "q50": float(np.quantile(values, .50)) if len(values) else None,
            "q75": float(np.quantile(values, .75)) if len(values) else None,
            "q95": float(np.quantile(values, .95)) if len(values) else None}


def donor_summary(query_features, train_features, query_rows, train_rows, indices):
    """Aggregate donor mismatch diagnostics; no BP targets or predictions read."""
    valid = (indices >= 0).all(axis=1)
    cosine, elapsed = [], []
    for start in range(0, len(indices), 2048):
        rows = np.flatnonzero(valid[start:start + 2048]) + start
        if not len(rows):
            continue
        q = np.asarray(query_features[rows], dtype=np.float64)
        r = np.asarray(train_features[indices[rows]], dtype=np.float64)
        distance = 1 - np.einsum("nd,nkd->nk", q, r) / (
            np.linalg.norm(q, axis=1)[:, None] * np.linalg.norm(r, axis=2))
        cosine.extend(np.clip(distance, 0, 2).ravel().tolist())
        for qi in rows:
            query = query_rows[qi]
            for ri in indices[qi]:
                ref = train_rows[ri]
                if (ref["time_axis_uid"] == query["time_axis_uid"] and
                        ref["recording_uid"] == query["recording_uid"]):
                    elapsed.append(max(ref["start_s"] - query["end_s"],
                                       query["start_s"] - ref["end_s"], 0.0))
    return {"n_queries": len(indices), "n_valid": int(valid.sum()),
            "n_fallback": int((~valid).sum()), "cosine_distance_all_donors": quantiles(cosine),
            "same_record_interval_gap_seconds": quantiles(elapsed),
            "query_targets_accessed": False}


def rematch_training(manifest, arrays, metadata):
    """Change two training arrays only, after full global lineage validation."""
    train_rows = metadata["train"][list(FIELDS)].to_dict("records")
    validation_rows = metadata["validation"][list(FIELDS)].to_dict("records")
    audit = audit_metadata(train_rows, validation_rows)
    # Empty validation makes the recomputation training-only. Full cross-role
    # audit above remains mandatory; audit=False is not a leakage bypass.
    fresh = prepare_neighbors(arrays["train_features"], np.empty((0, 256)), train_rows, [],
        mode=manifest["split_mode"], k=5, block_size=1, exclusion_s=0.0, audit=False)
    updated = dict(arrays)
    updated["train_indices"] = fresh["train"]["knn_indices"]
    updated["train_weights"] = fresh["train"]["knn_weights"]
    unchanged = ("train_support", "validation_support", "validation_indices", "validation_weights")
    for key in unchanged:
        np.testing.assert_array_equal(updated[key], arrays[key])
    summary = {
        "policy": POLICY, "old_training_excluded_block_size": 40, "training_excluded_block_size": 1,
        "physical_exclusion_seconds": 0.0, "reference_count": 5, "lineage_audit": audit,
        "alpha_recomputed": False, "validation_references_recomputed": False,
        "changed_train_rows": int(np.any(updated["train_indices"] != arrays["train_indices"], axis=1).sum()),
        "invariant_array_sha256": {key: array_sha256(arrays[key]) for key in unchanged},
        "train_old": donor_summary(arrays["train_features"], arrays["train_features"], train_rows, train_rows, arrays["train_indices"]),
        "train_new": donor_summary(arrays["train_features"], arrays["train_features"], train_rows, train_rows, updated["train_indices"]),
        "validation_unchanged": donor_summary(arrays["validation_features"], arrays["train_features"], validation_rows, train_rows, arrays["validation_indices"]),
    }
    return updated, summary


def load_matched_cache(cache, *, smoke=False):
    from .personal_memory_train import load_cache
    from .training import file_sha256

    digest = file_sha256(cache / "manifest.json")
    manifest, arrays, metadata = load_cache(cache, "pair_distance_blend", smoke)
    if not smoke:
        validate_export_manifest({**manifest, "status": "exported"})
        if manifest.get("screen_id") != "personal-memory-v1" or manifest.get("preparation", {}).get("train_excluded_block_size") != 40:
            raise ValueError("E1 requires the original complete 40-window-block v1 cache")
    arrays, summary = rematch_training(manifest, arrays, metadata)
    if file_sha256(cache / "manifest.json") != digest:
        raise ValueError("cache manifest changed while loading E1")
    return manifest, arrays, metadata, summary, digest


def training_contract(args):
    smoke_check = bool(getattr(args, "smoke_check", False))
    if args.smoke and smoke_check:
        raise ValueError("synthetic smoke and real-cache smoke-check are distinct")
    if smoke_check:
        args.epochs, args.examples_per_epoch, args.batch_size = 1, 1024, 256
    fixed = {"seed": 20260907, "epochs": 0, "patience": 8, "batch_size": 256, "examples_per_epoch": 200000,
             "learning_rate": 3e-4, "weight_decay": 1e-4, "huber_delta": .5}
    if not (args.smoke or smoke_check) and any(getattr(args, key) != value for key, value in fixed.items()):
        raise ValueError("production E1 has a fixed single-intervention training contract")
    if args.batch_size < 1 or args.examples_per_epoch < 1 or args.patience < 1 or args.epochs < 0:
        raise ValueError("invalid training controls")
    return smoke_check


def train(args):
    import torch
    from .calbased_metrics import participant_macro_views, pooled_diagnostics
    from .personal_memory_models import PersonalMemoryRelation, supervised_relation_loss
    from .personal_memory_train import _batch, predict, validate_gate
    from .training import file_sha256, seed_everything, source_tree_sha256

    started = time.monotonic()
    smoke_check = training_contract(args)
    if args.output.exists():
        raise FileExistsError("E1 output must be a new directory")
    raw = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    gate = validate_gate(args.gate_manifest, raw, file_sha256(args.cache_dir / "manifest.json"), args.smoke)
    if gate["status"] == "stop_no_complementarity":
        raise ValueError("v1 diagnostic gate did not authorize a relation screen")
    manifest, arrays, metadata, mismatch, cache_digest = load_matched_cache(args.cache_dir, smoke=args.smoke)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable")
    args.output.mkdir(parents=True)
    save_json(args.output / "reference_mismatch.json", mismatch)
    np.save(args.output / "private_train_indices.npy", arrays["train_indices"], allow_pickle=False)
    np.save(args.output / "private_train_weights.npy", arrays["train_weights"], allow_pickle=False)
    seed_everything(args.seed)
    tensors = {key: torch.as_tensor(np.array(value, copy=True),
        dtype=torch.long if key.endswith("indices") else torch.float32, device=device) for key, value in arrays.items()}
    del arrays
    std = torch.tensor(manifest["target_scaler"]["std"], dtype=torch.float32, device=device)
    model = PersonalMemoryRelation().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_prediction = predict(model, tensors, metadata["validation"], std, args.validation_batch_size)
    initial_metrics = participant_macro_views(best_prediction)
    best_metrics = initial_metrics
    best_score = float(best_metrics["Overall"]["mean_mae"])
    best_epoch = epoch = stale = steps = examples = 0
    best_trained_score = float("inf")
    best_trained_epoch, best_trained_prediction, best_trained_metrics = None, None, None
    best_path = args.output / "best.pt"
    mapping = {person: index for index, person in enumerate(sorted(metadata["train"].subject_uid.unique()))}

    def checkpoint():
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": best_epoch, "seed": args.seed, "candidate": CANDIDATE, "screen_id": SCREEN_ID,
            "target_scaler": manifest["target_scaler"], "subject_to_index": mapping,
            "cache_manifest_sha256": cache_digest, "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
            "frozen_personal_state_is_in_source_checkpoint": True, "metrics": best_metrics,
            "retrieval_training_policy": POLICY}, best_path)

    checkpoint()
    best_prediction.to_parquet(args.output / "initial_internal_validation_predictions.parquet", index=False)
    print(json.dumps({"initial_epoch": 0, "internal_validation": initial_metrics}), flush=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    history = []
    while args.epochs == 0 or epoch < args.epochs:
        epoch += 1
        model.train()
        totals = np.zeros(3, dtype=float)
        seen = 0
        epoch_start = time.monotonic()
        for offset in range(0, args.examples_per_epoch, args.batch_size):
            size = min(args.batch_size, args.examples_per_epoch - offset)
            index = torch.randint(len(metadata["train"]), (size,), generator=generator, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction, delta, reference_bp, legal = _batch(model, tensors, "train", index, std)
            loss, absolute, pair = supervised_relation_loss(prediction, tensors["train_bp"][index],
                delta, reference_bp, legal, std, huber_delta=args.huber_delta, pair_weight=.25)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite E1 training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            totals += np.array([float(value.detach()) for value in (loss, absolute, pair)]) * size
            steps += 1
            seen += size
        examples += seen
        prediction = predict(model, tensors, metadata["validation"], std, args.validation_batch_size)
        metrics = participant_macro_views(prediction)
        score = float(metrics["Overall"]["mean_mae"])
        if score < best_trained_score:
            best_trained_score, best_trained_epoch = score, epoch
            best_trained_prediction, best_trained_metrics = prediction, metrics
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "seed": args.seed,
                "candidate": CANDIDATE, "screen_id": SCREEN_ID, "metrics": metrics,
                "cache_manifest_sha256": cache_digest, "target_scaler": manifest["target_scaler"],
                "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
                "subject_to_index": mapping, "retrieval_training_policy": POLICY}, args.output / "best_trained.pt")
        improved = score < best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            best_prediction, best_metrics = prediction, metrics
            checkpoint()
        else:
            stale += 1
        item = {"epoch": epoch, "train_loss": totals[0] / seen, "train_absolute_loss": totals[1] / seen,
            "train_pair_loss": totals[2] / seen, "examples": seen, "optimizer_steps_total": steps,
            "epoch_seconds": time.monotonic() - epoch_start, "internal_validation": metrics,
            "improved": improved, "stale_epochs": stale}
        history.append(item)
        save_json(args.output / "history.json", history)
        print(json.dumps(item), flush=True)
        if stale >= args.patience:
            break
    best_prediction.to_parquet(args.output / "best_internal_validation_predictions.parquet", index=False)
    best_trained_prediction.to_parquet(args.output / "best_trained_internal_validation_predictions.parquet", index=False)
    pooled_diagnostics(best_prediction, CANDIDATE).to_csv(args.output / "internal_validation_pooled_diagnostics.csv", index=False)
    save_json(args.output / "metrics.json", best_metrics)
    run = {"status": "synthetic_smoke_complete" if args.smoke else "real_smoke" if smoke_check else "complete", "screen_id": SCREEN_ID,
        "candidate": CANDIDATE, "protocol_id": "development-calbased-analogue-v1",
        "split_mode": manifest["split_mode"], "source_parent_split": "meta_train",
        "read_roles": ["train", "internal_validation"], "heldout_test_accessed": False,
        "selection_role": "internal_validation", "official_pulsedb_calbased_reproduction": False,
        "seed": args.seed, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cache_manifest_sha256": cache_digest, "gate_manifest_sha256": gate.get("manifest_sha256"),
        "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"), "source_run": manifest.get("source_run"),
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "feature_provenance": "frozen supervised personal LoRA; not OOF", "persistent_LoRA_preserved": True,
        "personal_label_budget_windows": manifest.get("personal_training_budget", manifest.get("personal_label_budget_windows", 320)),
        "train_participants": metadata["train"].subject_uid.nunique(), "train_windows_available": len(metadata["train"]),
        "internal_validation_participants": metadata["validation"].subject_uid.nunique(),
        "internal_validation_windows": len(best_prediction), "validation_query_coverage": 1.0,
        "fallback_validation_windows": int((tensors["validation_indices"] < 0).all(dim=1).sum()),
        "fallback_train_windows": int((tensors["train_indices"] < 0).all(dim=1).sum()),
        "retrieved_references": 5, "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "retrieval_training_policy": POLICY, "fixed_v1_alpha_preserved": True,
        "fixed_v1_validation_references_preserved": True, "reference_mismatch": mismatch,
        "initial_metrics": initial_metrics, "metrics": best_metrics, "best_epoch": best_epoch,
        "best_trained_epoch": best_trained_epoch, "best_trained_metrics": best_trained_metrics,
        "best_trained_checkpoint_sha256": file_sha256(args.output / "best_trained.pt"),
        "epochs_completed": epoch, "optimizer_steps": steps, "examples_processed": examples,
        "runtime_seconds": time.monotonic() - started,
        "stop_reason": "early_stopping" if stale >= args.patience else "real_smoke_four_steps" if smoke_check else "synthetic_smoke_epoch_cap",
        "checkpoint_sha256": file_sha256(best_path),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "standard_claim": "retrospective numerical screens only, not clinical certification"}
    save_json(args.output / "run.json", run)
    return run


def synthetic_cache(root, mode="random_disjoint"):
    """Explicit synthetic fixture, including canonical lineage and both sources."""
    from .personal_memory_prepare import sha256
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(17)
    metadata, features = {}, {}
    for role, count in (("train", 48), ("validation", 2)):
        rows = []
        for person, source in enumerate(("MIMIC", "VitalDB")):
            for i in range(count):
                uid = f"{source}:{role}:{i}"
                start = 20 * i if role == "train" else 1100 + 20 * i
                rows.append({"subject_uid": source + ":person", "event_id": uid, "window_uid": uid,
                    "source": source, "waveform_sha256": hashlib.sha256(uid.encode()).hexdigest(),
                    "recording_uid": source + ":record", "time_axis_uid": source + ":record",
                    "start_s": float(start), "end_s": float(start + 10),
                    "role": "train" if role == "train" else "internal_validation",
                    "target_sbp": 115.0 + person * 10 + i / 10, "target_dbp": 65.0 + person * 5 + i / 20})
        metadata[role] = rows
        features[role] = rng.normal(size=(len(rows), 256)).astype(np.float32)
    neighbors = prepare_neighbors(features["train"], features["validation"], metadata["train"], metadata["validation"], mode=mode)
    for role in metadata:
        frame = pd.DataFrame(metadata[role])
        frame.to_parquet(root / f"{role}_metadata.parquet", index=False)
        bp = frame[["target_sbp", "target_dbp"]].to_numpy(np.float32)
        for name, value in {"features": features[role], "bp": bp, "base": bp + [3., -2.],
                            **{key: neighbors[role][key] for key in ("knn_indices", "knn_weights", "support_weight")}}.items():
            np.save(root / f"{role}_{name}.npy", value, allow_pickle=False)
    manifest = {"status": "complete", "screen_id": "personal-memory-v1", "synthetic_smoke": True,
        "protocol_id": "development-calbased-analogue-v1", "split_mode": mode, "source_parent_split": "meta_train",
        "read_roles": ["train", "internal_validation"], "heldout_test_accessed": False,
        "target_scaler": {"mean": [125., 75.], "std": [20., 10.]}, "personal_label_budget_windows": 48,
        "source_checkpoint_sha256": "a" * 64, "source_run": "synthetic-only-no-real-checkpoint",
        "files": {path.name: {"path": path.name, "sha256": sha256(path)} for path in root.iterdir()}}
    save_json(root / "manifest.json", manifest)
    return root


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gate-manifest", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--validation-batch-size", type=int, default=1024)
    p.add_argument("--examples-per-epoch", type=int, default=200000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=.5)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-check", action="store_true", help="real cache technical check: four optimizer steps, full validation, not a scientific result")
    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--split-mode", choices=("random_disjoint", "chronological_blocked"), default="random_disjoint")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.synthetic_smoke:
        if args.cache_dir is not None or args.gate_manifest is not None:
            p.error("synthetic smoke never accepts a real cache or gate")
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        args.cache_dir = synthetic_cache(args.output / "synthetic_cache", args.split_mode)
        args.output = args.output / "synthetic_run"
        args.smoke, args.epochs, args.examples_per_epoch, args.batch_size = True, 1, 16, 8
    elif args.cache_dir is None:
        p.error("--cache-dir is required for an existing-cache run")
    print(json.dumps(train(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
