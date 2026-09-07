"""Train a finite local-relation candidate from an immutable, audited cache.

Production requires a two-mode diagnostic gate. Features were learned using
ordinary supervised train labels; neither cached features nor train residuals
are called out-of-fold. Validation labels select checkpoints only. No cache
file or model path for a held-out role is supported here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from .calbased_metrics import participant_macro_views, pooled_diagnostics
from .personal_memory_models import (
    CANDIDATES, SCREEN_ID, PersonalMemoryRelation, reference_prediction,
    supervised_relation_loss,
)
from .training import file_sha256, seed_everything, source_tree_sha256


PROTOCOL_ID = "development-calbased-analogue-v1"
SPLIT_MODES = ("random_disjoint", "chronological_blocked")
READ_ROLES = ["train", "internal_validation"]


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_gate(gate_path: Path | None, manifest: dict, digest: str, smoke: bool) -> dict:
    if smoke:
        if not manifest.get("synthetic_smoke", False):
            raise ValueError("--smoke is only allowed for explicitly synthetic caches")
        return {"status": "synthetic_smoke_only", "not_scientific_results": True}
    if gate_path is None:
        raise ValueError("production requires --gate-manifest; diagnostics cannot be bypassed")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("heldout_test_accessed") is not False:
        raise ValueError("gate must declare heldout_test_accessed=false")
    hashes = gate.get("cache_manifest_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(SPLIT_MODES):
        raise ValueError("gate must identify both independently audited split caches")
    if hashes.get(manifest["split_mode"]) != digest:
        raise ValueError("gate/cache manifest SHA-256 mismatch")
    if gate.get("status") not in {"proceed_to_train", "stop_no_complementarity"}:
        raise ValueError("unrecognized diagnostic gate decision")
    return {**gate, "manifest_path": str(gate_path), "manifest_sha256": file_sha256(gate_path)}


def _read_file(cache: Path, manifest: dict, name: str) -> Path:
    spec = manifest.get("files", {}).get(name)
    if not isinstance(spec, dict) or not {"path", "sha256"}.issubset(spec):
        raise ValueError(f"cache manifest lacks a hashed file entry for {name}")
    path = (cache / spec["path"]).resolve()
    if path.parent != cache.resolve() or path.name != name:
        raise ValueError(f"cache path must be its exact top-level filename: {name}")
    if file_sha256(path) != spec["sha256"]:
        raise ValueError(f"cache content changed: {name}")
    return path


def load_cache(cache: Path, candidate: str, smoke: bool = False) -> tuple[dict, dict, dict]:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "status": "complete", "protocol_id": PROTOCOL_ID,
        "source_parent_split": "meta_train", "read_roles": READ_ROLES,
        "heldout_test_accessed": False,
    }
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError("cache is not a completed, development-only seen-user artifact")
    if manifest.get("split_mode") not in SPLIT_MODES:
        raise ValueError("unrecognized split mode")
    if bool(manifest.get("synthetic_smoke", False)) != bool(smoke):
        raise ValueError("synthetic and real caches must never be mixed")
    scaler = manifest.get("target_scaler", {})
    mean, std = np.asarray(scaler.get("mean")), np.asarray(scaler.get("std"))
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() \
            or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("invalid source train-only BP scaler")
    family = "uniform" if candidate == "pair_uniform" else "knn"
    arrays, metadata = {}, {}
    for role in ("train", "validation"):
        table = pd.read_parquet(_read_file(cache, manifest, f"{role}_metadata.parquet"))
        required_columns = {"subject_uid", "event_id", "source", "target_sbp", "target_dbp"}
        if not required_columns.issubset(table.columns) or table.empty:
            raise ValueError(f"invalid {role} metadata schema")
        if table[list(required_columns)].isna().any().any() or table.event_id.astype(str).duplicated().any():
            raise ValueError(f"null or duplicated {role} query identifiers")
        table["subject_uid"] = table.subject_uid.astype(str)
        table["event_id"] = table.event_id.astype(str)
        if set(table.source.astype(str)) != {"MIMIC", "VitalDB"}:
            raise ValueError("each role must contain both internal PulseDB sources")
        expected_role = "train" if role == "train" else "internal_validation"
        if "role" in table.columns:
            if not table["role"].eq(expected_role).all():
                raise ValueError(f"metadata role must be exactly {expected_role}")
        elif not smoke:
            raise ValueError("production metadata requires explicit role column")
        if not smoke:
            expected = 320 if role == "train" else 40
            if not table.groupby("subject_uid").size().eq(expected).all():
                raise ValueError(f"production {role} requires {expected} rows per participant")
            if table.subject_uid.nunique() != 2051:
                raise ValueError("production cache must retain all 2051 registered participants")
            per_source = table.groupby("source")["subject_uid"].nunique().to_dict()
            if per_source != {"MIMIC": 1011, "VitalDB": 1040}:
                raise ValueError("production cache must retain 1011 MIMIC and 1040 VitalDB people")
        metadata[role] = table
        for suffix, shape in (("features", (len(table), 256)), ("bp", (len(table), 2)),
                              ("base", (len(table), 2))):
            value = np.load(_read_file(cache, manifest, f"{role}_{suffix}.npy"),
                            mmap_mode="r", allow_pickle=False)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid {role}_{suffix} shape/values")
            arrays[f"{role}_{suffix}"] = value
        if not np.allclose(table[["target_sbp", "target_dbp"]].to_numpy(float),
                           arrays[f"{role}_bp"], rtol=0, atol=1e-4):
            raise ValueError(f"{role} target array/metadata row alignment mismatch")
        neighbours = np.load(_read_file(cache, manifest, f"{role}_{family}_indices.npy"),
                             allow_pickle=False)
        if neighbours.shape != (len(table), 5) or not np.issubdtype(neighbours.dtype, np.integer):
            raise ValueError("expected five integer absolute train-bank indices per query")
        if (neighbours < -1).any() or (neighbours >= len(metadata["train"])).any():
            raise ValueError("reference index outside train memory")
        valid = neighbours >= 0
        if ((valid.sum(axis=1) != 0) & (valid.sum(axis=1) != 5)).any():
            raise ValueError("incomplete neighbours require all-minus-one fallback rows")
        # Five distinct donors, not repeated copies of one conveniently close BP.
        sorted_indices = np.sort(neighbours, axis=1)
        if ((np.diff(sorted_indices, axis=1) == 0) & valid[:, :1]).any():
            raise ValueError("duplicate reference indices within query")
        safe = np.maximum(neighbours, 0)
        for column in ("subject_uid", "source"):
            bank = metadata["train"][column].astype(str).to_numpy()
            query = table[column].astype(str).to_numpy()
            if ((bank[safe] != query[:, None]) & valid).any():
                raise ValueError(f"references must belong to the same {column}")
        bank_ids = metadata["train"].event_id.to_numpy()
        if ((bank_ids[safe] == table.event_id.to_numpy()[:, None]) & valid).any():
            raise ValueError("self-reference is forbidden")
        weights = np.full(neighbours.shape, 0.2, dtype=np.float32)
        if family == "knn":
            weights = np.load(_read_file(cache, manifest, f"{role}_knn_weights.npy"), allow_pickle=False)
        if weights.shape != neighbours.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("invalid PPG-only reference weights")
        if not np.allclose(weights[valid.any(axis=1)].sum(axis=1), 1, atol=1e-5):
            raise ValueError("reference weights must sum to one for covered queries")
        support = np.load(_read_file(cache, manifest, f"{role}_support_weight.npy"), allow_pickle=False)
        if support.shape != (len(table),) or not np.isfinite(support).all() or \
                ((support < 0) | (support > 1)).any():
            raise ValueError("invalid fixed support-distance weights")
        if candidate == "pair_single":
            neighbours, weights = neighbours[:, :1], np.ones((len(table), 1), dtype=np.float32)
        arrays[f"{role}_indices"] = neighbours
        arrays[f"{role}_weights"] = weights
        arrays[f"{role}_support"] = support if candidate == "pair_distance_blend" \
            else np.ones(len(table), dtype=np.float32)
    if set(metadata["train"].event_id) & set(metadata["validation"].event_id):
        raise ValueError("train/validation query identifiers overlap")
    if set(metadata["train"].subject_uid) != set(metadata["validation"].subject_uid):
        raise ValueError("seen-user roles must contain identical registered participants")
    return manifest, arrays, metadata


def _batch(model, tensors: dict, role: str, index: torch.Tensor, std: torch.Tensor):
    references = tensors[f"{role}_indices"][index]
    legal = references >= 0
    safe = references.clamp_min(0)
    reference_bp = tensors["train_bp"][safe]
    prediction, delta = reference_prediction(
        model, tensors[f"{role}_features"][index], tensors["train_features"][safe],
        reference_bp, tensors[f"{role}_base"][index], tensors[f"{role}_weights"][index],
        legal, tensors[f"{role}_support"][index], std,
    )
    return prediction, delta, reference_bp, legal


@torch.no_grad()
def predict(model, tensors: dict, metadata: pd.DataFrame, std: torch.Tensor,
            batch_size: int) -> pd.DataFrame:
    model.eval()
    chunks = []
    for start in range(0, len(metadata), batch_size):
        index = torch.arange(start, min(start + batch_size, len(metadata)), device=std.device)
        prediction, _, _, _ = _batch(model, tensors, "validation", index, std)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("non-finite validation prediction")
        chunks.append(prediction.cpu().numpy())
    result = metadata[["subject_uid", "event_id", "source", "target_sbp", "target_dbp"]].copy()
    values = np.concatenate(chunks)
    result["pred_sbp"], result["pred_dbp"] = values[:, 0], values[:, 1]
    return result


def train(args: argparse.Namespace) -> dict:
    start_time = time.monotonic()
    if args.num_workers != 0:
        raise ValueError("cached-feature trainer is serial; --num-workers must be 0")
    if args.epochs != 0 and not args.smoke:
        raise ValueError("production has no epoch cap; use --epochs 0 with patience 8")
    if args.patience < 1 or args.batch_size < 1 or args.examples_per_epoch < 1:
        raise ValueError("positive training controls required")
    if args.patience != 8 and not args.smoke:
        raise ValueError("production screen requires patience 8")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty experiment output")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_dir / "manifest.json"
    manifest_digest = file_sha256(manifest_path)
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("split_mode") not in SPLIT_MODES:
        raise ValueError("invalid split mode before gate inspection")
    gate = validate_gate(args.gate_manifest, raw_manifest, manifest_digest, args.smoke)
    if gate["status"] == "stop_no_complementarity":
        run = {"status": "skipped_no_complementarity", "candidate": args.candidate,
               "screen_id": SCREEN_ID, "split_mode": raw_manifest["split_mode"],
               "heldout_test_accessed": False, "gate": gate,
               "cache_manifest_sha256": manifest_digest,
               "gate_manifest_sha256": gate.get("manifest_sha256")}
        _json(args.output / "run.json", run)
        return run
    manifest, arrays, metadata = load_cache(args.cache_dir, args.candidate, args.smoke)
    if file_sha256(manifest_path) != manifest_digest:
        raise ValueError("cache manifest changed while loading")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable")
    seed_everything(args.seed)
    # <1 GiB for the production 656320x256 bank and 82040 validation features.
    # No waveform DataLoader, worker shared memory, or batch-normalization update.
    tensors = {key: torch.as_tensor(np.array(value, copy=True),
               dtype=torch.long if key.endswith("indices") else torch.float32, device=device)
               for key, value in arrays.items()}
    del arrays
    std = torch.tensor(manifest["target_scaler"]["std"], dtype=torch.float32, device=device)
    model = PersonalMemoryRelation().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_path = args.output / "best.pt"
    best_predictions = predict(model, tensors, metadata["validation"], std, args.validation_batch_size)
    best_metrics = participant_macro_views(best_predictions)
    best_score = float(best_metrics["Overall"]["mean_mae"])
    best_epoch, stale, epoch, optimizer_steps = 0, 0, 0, 0
    history = []
    mapping = {person: i for i, person in enumerate(sorted(metadata["train"].subject_uid.unique()))}

    def save_checkpoint():
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "epoch": best_epoch, "seed": args.seed, "candidate": args.candidate,
                    "target_scaler": manifest["target_scaler"], "subject_to_index": mapping,
                    "cache_manifest_sha256": manifest_digest, "metrics": best_metrics,
                    "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
                    "source_run": manifest.get("source_run"),
                    "frozen_personal_state_is_in_source_checkpoint": True}, best_path)

    save_checkpoint()
    best_predictions.to_parquet(args.output / "initial_internal_validation_predictions.parquet", index=False)
    print(json.dumps({"initial_epoch": 0, "internal_validation": best_metrics}), flush=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    while args.epochs == 0 or epoch < args.epochs:
        epoch += 1
        model.train()
        total_loss, total_absolute, total_pair, seen = 0.0, 0.0, 0.0, 0
        epoch_start = time.monotonic()
        for offset in range(0, args.examples_per_epoch, args.batch_size):
            size = min(args.batch_size, args.examples_per_epoch - offset)
            index = torch.randint(len(metadata["train"]), (size,), generator=generator, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction, delta, reference_bp, legal = _batch(model, tensors, "train", index, std)
            loss, absolute, pair = supervised_relation_loss(
                prediction, tensors["train_bp"][index], delta, reference_bp, legal, std,
                huber_delta=args.huber_delta, pair_weight=0.25,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("non-finite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.detach()) * size
            total_absolute += float(absolute.detach()) * size
            total_pair += float(pair.detach()) * size
            seen += size
        predictions = predict(model, tensors, metadata["validation"], std, args.validation_batch_size)
        metrics = participant_macro_views(predictions)
        score = float(metrics["Overall"]["mean_mae"])
        improved = score < best_score
        if improved:
            best_score, best_epoch = score, epoch
            best_predictions, best_metrics, stale = predictions, metrics, 0
            save_checkpoint()
        else:
            stale += 1
        item = {"epoch": epoch, "train_loss": total_loss / seen,
                "train_absolute_loss": total_absolute / seen, "train_pair_loss": total_pair / seen,
                "examples": seen, "optimizer_steps_total": optimizer_steps,
                "epoch_seconds": time.monotonic() - epoch_start,
                "internal_validation": metrics, "improved": improved, "stale_epochs": stale}
        history.append(item)
        _json(args.output / "history.json", history)
        print(json.dumps(item), flush=True)
        if stale >= args.patience:
            break
    best_predictions.to_parquet(args.output / "best_internal_validation_predictions.parquet", index=False)
    pooled = pooled_diagnostics(best_predictions, args.candidate)
    pooled.to_csv(args.output / "internal_validation_pooled_diagnostics.csv", index=False)
    _json(args.output / "metrics.json", best_metrics)
    base_predictions = metadata["validation"][["subject_uid", "event_id", "source", "target_sbp", "target_dbp"]].copy()
    base_values = tensors["validation_base"].cpu().numpy()
    base_predictions["pred_sbp"], base_predictions["pred_dbp"] = base_values[:, 0], base_values[:, 1]
    baseline_metrics = participant_macro_views(base_predictions)
    run = {
        "status": "synthetic_smoke_complete" if args.smoke else "complete",
        "protocol_id": PROTOCOL_ID, "screen_id": SCREEN_ID, "candidate": args.candidate,
        "runner": "personal_memory_cached_relation", "split_mode": manifest["split_mode"],
        "source_parent_split": "meta_train", "track": "development_only_same_subject_analogue",
        "official_pulsedb_calbased_reproduction": False, "read_roles": READ_ROLES,
        "selection_role": "internal_validation", "heldout_test_accessed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gate": gate,
        "gate_manifest_sha256": gate.get("manifest_sha256"),
        "cache_manifest_sha256": manifest_digest, "cache_manifest": manifest,
        "feature_provenance": "frozen_features_from_supervised_personal_train_fit_not_OOF",
        "relation_initialization": "zero_final_layer_delta_equals_zero_reference_interpolation",
        "fusion": "fixed_PPG_distance_only" if args.candidate == "pair_distance_blend" else "none",
        "train_participants": int(metadata["train"].subject_uid.nunique()),
        "train_windows_available": len(metadata["train"]),
        "internal_validation_participants": int(metadata["validation"].subject_uid.nunique()),
        "internal_validation_windows": len(metadata["validation"]),
        "personal_label_budget_windows": manifest.get(
            "personal_training_budget", manifest.get("personal_label_budget_windows", 320)),
        "personal_training_budget": manifest.get(
            "personal_training_budget", manifest.get("personal_label_budget_windows", 320)),
        "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
        "source_run": manifest.get("source_run"),
        "retrieved_references": 1 if args.candidate == "pair_single" else 5,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "new_participant_trainable_parameters": 0, "persistent_LoRA_preserved": True,
        "fallback_validation_windows": int((tensors["validation_indices"] < 0).all(dim=1).sum()),
        "fallback_train_windows": int((tensors["train_indices"] < 0).all(dim=1).sum()),
        "validation_query_coverage": 1.0, "best_epoch": best_epoch, "epochs_completed": epoch,
        "optimizer_steps": optimizer_steps, "examples_processed": epoch * args.examples_per_epoch,
        "stop_reason": "early_stopping" if stale >= args.patience else "synthetic_smoke_epoch_cap",
        "runtime_seconds": time.monotonic() - start_time, "metrics": best_metrics,
        "frozen_baseline_metrics": baseline_metrics, "checkpoint": str(best_path),
        "checkpoint_sha256": file_sha256(best_path),
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "standard_claim": "pooled retrospective AAMI/BHS numerical screens, not clinical certification",
    }
    _json(args.output / "run.json", run)
    return run


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--candidate", choices=CANDIDATES, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gate-manifest", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--validation-batch-size", type=int, default=1024)
    p.add_argument("--examples-per-epoch", type=int, default=200000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=0.5)
    p.add_argument("--smoke", action="store_true", help="synthetic caches only; never bypass real-data gate")
    return p


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(train(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
