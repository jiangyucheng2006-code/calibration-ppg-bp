"""Fit official-CalBased history methods without reading official test data.

Commands operate on explicitly exported caches, not glob-discovered datasets.
The new protocol has its own provenance/cohort gates; older frozen experiments
and their gates remain unchanged. Reused neighbour mathematics keeps the v1
40-window block, cosine temperature, distance alpha and five donors fixed.

Trust candidates require three encoder-level crossfit caches. Every gate
training query was excluded from the corresponding backbone, personal state,
anchor, scaler and memory-bank fit. Full-train frozen features used for v1/E1
are ordinary supervised features and are explicitly NOT called OOF.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from .personal_memory_prepare import audit_metadata, prepare_neighbors, macro_metrics


PROTOCOL = "pulsedb-official-calbased-v1"
METHODS = ("fixed", "v1", "e1", "trust_scalar", "trust_bp")
CACHE_FILES = ("metadata.parquet", "features.npy", "base_bp.npy", "bp.npy")
TRUST_FEATURE_NAMES = (
    "nearest_cosine_distance", "mean_cosine_distance", "max_cosine_distance",
    "effective_reference_fraction", "reference_sbp_sd_scaled", "reference_dbp_sd_scaled",
    "memory_minus_lora_sbp_scaled", "memory_minus_lora_dbp_scaled",
    "abs_memory_minus_lora_sbp_scaled", "abs_memory_minus_lora_dbp_scaled",
    "fixed_alpha", "has_five_legal_references",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_digest(values):
    """Cross-module ID hash contract, including one trailing newline."""
    return hashlib.sha256(("\n".join(sorted(map(str, values))) + "\n").encode()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_role(root: Path, expected_role: str, *, synthetic=False, allow_test_inputs=False):
    """Verify permitted provenance and file hashes before loading any arrays."""
    root = Path(root)
    manifest = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL:
        raise ValueError("wrong official cache protocol")
    if manifest.get("official_test_accessed") is not False and not allow_test_inputs:
        raise ValueError("official test access must explicitly be false")
    if allow_test_inputs and (manifest.get("stage") != "final" or manifest.get("official_test_targets_accessed") is not False):
        raise ValueError("final input access requires final-stage provenance and sealed targets")
    allowed_roles = {"train", "validation", "source_fit", "excluded_fold"} | ({"test_inputs"} if allow_test_inputs else set())
    if expected_role not in allowed_roles:
        raise ValueError("test/held-out cache roles are not accepted by this trainer")
    if bool(manifest.get("synthetic_smoke", False)) != bool(synthetic):
        raise ValueError("synthetic and real evidence may not be mixed")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("checkpoint_sha256", ""))):
        raise ValueError("source checkpoint hash required")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("official_contract_sha256", ""))):
        raise ValueError("official data membership contract hash required")
    input_only = expected_role == "test_inputs"
    needed_files = CACHE_FILES[:-1] if input_only else CACHE_FILES
    if input_only and (manifest.get("labels_present") is not False or (root / "bp.npy").exists() or "bp.npy" in manifest.get("files", {})):
        raise ValueError("test input cache must not contain BP targets")
    for filename in needed_files:
        digest = manifest.get("files", {}).get(filename)
        if isinstance(digest, dict):
            digest = digest.get("sha256")
        path = root / filename
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"cache checksum mismatch: {filename}")
    frame = pd.read_parquet(root / "metadata.parquet")
    if frame.empty or "role" not in frame:
        raise ValueError("nonempty metadata with an explicit role required")
    actual = frame["oof_role"] if expected_role in {"source_fit", "excluded_fold"} else frame["role"]
    if not actual.eq(expected_role).all():
        raise ValueError("metadata role disagrees with requested cache role")
    forbidden = {"bp", "sbp", "dbp", "SegSBP", "SegDBP", "ABP"}
    if any(c in forbidden or str(c).lower().startswith(("target_", "abp", "segsbp", "segdbp")) for c in frame):
        raise ValueError("canonical metadata must not contain BP targets")
    features = np.load(root / "features.npy", mmap_mode="r", allow_pickle=False)
    base = np.load(root / "base_bp.npy", mmap_mode="r", allow_pickle=False)
    bp = None if input_only else np.load(root / "bp.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != (len(frame), 256) or base.shape != (len(frame), 2) or (bp is not None and bp.shape != base.shape):
        raise ValueError("cache must contain aligned 256-D features and 2-D raw-mmHg BP")
    if not all(np.isfinite(v).all() for v in (features, base) + (() if bp is None else (bp,))):
        raise ValueError("non-finite cache input")
    if manifest.get("bp_units") != "mmHg":
        raise ValueError("raw mmHg cache units must be declared")
    if manifest.get("fit_transforms_on_source_fit_only") is not True:
        raise ValueError("scaler, anchor and personal state must use source-fit labels only")
    return {"root": root, "metadata": frame, "features": features,
            "base": base, "bp": bp, "manifest": manifest,
            "manifest_sha256": sha256_file(root / "provenance.json")}


def canonical_rows(frame, role):
    """Map identical role semantics to the shared *pure lineage audit* API.

    This does not invoke or disable any old experiment/protocol/count checker.
    All source windows become train, all excluded/evaluation windows become
    internal_validation for interval/hash auditing, without changing membership.
    """
    columns = ["subject_uid", "event_id", "source", "recording_uid", "time_axis_uid",
               "start_s", "end_s", "waveform_sha256"]
    if any(name not in frame for name in columns):
        raise ValueError("missing canonical cache lineage")
    rows = frame[columns].to_dict("records")
    for row in rows:
        row["window_uid"] = row["event_id"]
        row["role"] = role
    return rows


def audit_roles(bank, query, *, expected_counts=None):
    bank_rows = canonical_rows(bank["metadata"], "train")
    query_rows = canonical_rows(query["metadata"], "internal_validation")
    audit = audit_metadata(bank_rows, query_rows)
    if set(bank["metadata"].subject_uid) != set(query["metadata"].subject_uid):
        raise ValueError("all official registered participants must have both roles")
    if bank["manifest"]["checkpoint_sha256"] != query["manifest"]["checkpoint_sha256"]:
        raise ValueError("feature spaces come from different checkpoints")
    if bank["manifest"]["official_contract_sha256"] != query["manifest"]["official_contract_sha256"]:
        raise ValueError("official source membership contracts disagree")
    if expected_counts is not None:
        people, n_bank, n_query = expected_counts
        if (bank["metadata"].subject_uid.nunique() != people or
                not bank["metadata"].groupby("subject_uid").size().eq(n_bank).all() or
                not query["metadata"].groupby("subject_uid").size().eq(n_query).all()):
            raise ValueError(f"official inner cohort requires {people} people and {n_bank}/{n_query} windows")
    return audit, bank_rows, query_rows


def prepare_pair(bank, query, *, matched=False, expected_counts=None):
    audit, bank_rows, query_rows = audit_roles(bank, query, expected_counts=expected_counts)
    result = prepare_neighbors(bank["features"], query["features"], bank_rows, query_rows,
                               mode="random_disjoint", k=5, block_size=40)
    if matched:
        # Only training donor indices/weights change. The v1 fixed alpha and
        # all validation donors stay exactly equal to the paired v1 comparator.
        fresh = prepare_neighbors(bank["features"], query["features"], bank_rows, query_rows,
                                  mode="random_disjoint", k=5, block_size=1)
        result["train"]["knn_indices"] = fresh["train"]["knn_indices"]
        result["train"]["knn_weights"] = fresh["train"]["knn_weights"]
        result["train"]["valid"] = fresh["train"]["valid"]
    return result, audit


def training_std(bank):
    return np.maximum(np.asarray(bank["bp"], dtype=np.float64).std(axis=0), 1.0).astype(np.float32)


def memory_inputs(bank, query, neighbors, target_std):
    """Compute trust inputs without accessing query['bp'] or target columns."""
    n = len(query["features"])
    index = neighbors["knn_indices"]
    legal = (index >= 0).all(axis=1)
    refs = np.asarray(bank["bp"])[np.maximum(index, 0)]
    weights = neighbors["knn_weights"] * legal[:, None]
    memory = (refs * weights[:, :, None]).sum(axis=1)
    base = np.asarray(query["base"])
    memory[~legal] = base[~legal]
    variance = (weights[:, :, None] * (refs - memory[:, None, :]) ** 2).sum(axis=1)
    distances = np.zeros((n, index.shape[1]), dtype=np.float32)
    # Chunked gathers avoid allocating N x K x 256 for the complete cohort.
    for start in range(0, n, 4096):
        stop = min(n, start + 4096)
        q = np.asarray(query["features"][start:stop], dtype=np.float32)
        r = np.asarray(bank["features"][np.maximum(index[start:stop], 0)], dtype=np.float32)
        norms = np.linalg.norm(q, axis=1)[:, None] * np.linalg.norm(r, axis=2)
        if (norms <= 0).any():
            raise ValueError("zero vectors cannot define cosine trust features")
        distances[start:stop] = np.clip(1 - np.einsum("nd,nkd->nk", q, r) / norms, 0, 2)
    distances[~legal] = 0
    disagreement = (memory - base) / target_std
    ess = np.divide(1, (weights ** 2).sum(axis=1), out=np.zeros(n), where=legal) / index.shape[1]
    features = np.column_stack((distances.min(axis=1), distances.mean(axis=1), distances.max(axis=1),
                                ess, np.sqrt(np.maximum(variance, 0)) / target_std,
                                disagreement, np.abs(disagreement),
                                neighbors["support_weight"], legal.astype(np.float32))).astype(np.float32)
    if features.shape != (n, len(TRUST_FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError("invalid target-free trust inputs")
    return {"features": features, "base": base.astype(np.float32), "memory": memory.astype(np.float32),
            "alpha": neighbors["support_weight"].astype(np.float32), "legal": legal}


def audit_crossfit(folds, outer_train):
    """Fail closed on model-level provenance, fold reuse or incomplete coverage."""
    if len(folds) != 3:
        raise ValueError("exactly three distinct encoder-level OOF folds required")
    universe = set(outer_train["metadata"].event_id.astype(str))
    excluded_seen, checkpoints, fold_ids = set(), set(), set()
    evidence = []
    for fit, held in folds:
        audit_roles(fit, held)
        fit_ids = set(fit["metadata"].event_id.astype(str))
        held_ids = set(held["metadata"].event_id.astype(str))
        if fit_ids & held_ids or fit_ids | held_ids != universe:
            raise ValueError("OOF must partition the exact outer training windows")
        if held_ids & excluded_seen:
            raise ValueError("a gate training query occurs in more than one excluded fold")
        excluded_seen.update(held_ids)
        for cache in (fit, held):
            info = cache["manifest"]
            if info.get("official_contract_sha256") != outer_train["manifest"]["official_contract_sha256"]:
                raise ValueError("OOF official membership contract differs from outer training")
            expected_epoch_policy = "prespecified_smoke" if info.get("synthetic_smoke") else "prespecified_25_epochs"
            if (info.get("encoder_protocol") != "model-level-exclusion" or
                    info.get("encoder_initialization") != "scratch" or
                    info.get("model_selection_excluded_fold_labels_used") is not False or
                    info.get("epoch_policy") != expected_epoch_policy):
                raise ValueError("OOF requires scratch encoder exclusion with no excluded-label stopping")
            if info.get("encoder_fit_event_ids_sha256") != event_digest(fit_ids):
                raise ValueError("source-fit hash does not match encoder fit declaration")
            if info.get("excluded_event_ids_sha256") != event_digest(held_ids):
                raise ValueError("excluded-query hash does not match encoder declaration")
        if fit["manifest"].get("fold") != held["manifest"].get("fold"):
            raise ValueError("OOF paired cache fold IDs disagree")
        fold = fit["manifest"].get("fold")
        checkpoint = fit["manifest"]["checkpoint_sha256"]
        if fold not in {0, 1, 2} or fold in fold_ids or checkpoint in checkpoints:
            raise ValueError("OOF fold IDs and checkpoints must be distinct")
        fold_ids.add(fold)
        checkpoints.add(checkpoint)
        # IDs alone do not ensure the same labels/physical rows were exported.
        reference = outer_train["metadata"].set_index("event_id")
        for cache in (fit, held):
            rows = cache["metadata"].set_index("event_id")
            for key in ("subject_uid", "source", "recording_uid", "time_axis_uid", "start_s", "end_s", "waveform_sha256"):
                if not rows[key].equals(reference.loc[rows.index, key]):
                    raise ValueError(f"OOF lineage differs from outer training: {key}")
            lookup = pd.Series(np.arange(len(outer_train["metadata"])), index=outer_train["metadata"].event_id)
            np.testing.assert_array_equal(cache["bp"], outer_train["bp"][lookup.loc[rows.index].to_numpy()])
        evidence.append({"fold": fold, "checkpoint_sha256": checkpoint,
                         "n_source_fit": len(fit_ids), "n_excluded": len(held_ids),
                         "source_fit_digest": event_digest(fit_ids), "excluded_digest": event_digest(held_ids)})
    if excluded_seen != universe or fold_ids != {0, 1, 2}:
        raise ValueError("OOF targets do not cover each outer training window exactly once")
    return evidence


def build_oof_examples(roots, outer_train, *, synthetic=False):
    folds = [(load_role(root / "source_fit", "source_fit", synthetic=synthetic),
              load_role(root / "excluded_fold", "excluded_fold", synthetic=synthetic)) for root in roots]
    evidence = audit_crossfit(folds, outer_train)
    arrays, targets, subjects = [], [], []
    for fit, held in folds:
        neighbors, _ = prepare_pair(fit, held)
        arrays.append(memory_inputs(fit, held, neighbors["validation"], training_std(fit)))
        targets.append(np.asarray(held["bp"]))
        subjects.extend(held["metadata"].subject_uid.astype(str))
    merged = {key: np.concatenate([item[key] for item in arrays]) for key in arrays[0]}
    merged["bp"] = np.concatenate(targets).astype(np.float32)
    merged["subjects"] = np.asarray(subjects)
    mean = merged["features"].mean(axis=0)
    std = np.maximum(merged["features"].std(axis=0), 1e-6)
    merged["features"] = (merged["features"] - mean) / std
    return merged, {"mean": mean.tolist(), "std": std.tolist(), "fit_role": "train_encoder_OOF_only"}, evidence


def metrics(target, prediction, frame):
    return {scope: macro_metrics(target[mask], prediction[mask], frame.subject_uid.to_numpy()[mask])
            for scope, mask in (("Overall", np.ones(len(frame), dtype=bool)),
                                ("MIMIC", frame.source.eq("MIMIC").to_numpy()),
                                ("VitalDB", frame.source.eq("VitalDB").to_numpy()))}


def prediction_frame(frame, bp, prediction):
    result = frame[["subject_uid", "event_id", "source"]].copy()
    for index, name in enumerate(("sbp", "dbp")):
        if bp is not None:
            result[f"target_{name}"] = bp[:, index]
        result[f"pred_{name}"] = prediction[:, index]
    return result


def fit(args):
    import torch
    from .official_memory_models import PersonalMemoryRelation, reference_prediction, TrustFusion
    from .personal_memory_models import supervised_relation_loss

    started = time.monotonic()
    final = args.stage == "final"
    if args.output.exists():
        raise FileExistsError("output must be a new directory")
    if args.epochs < 0 or args.patience < 1 or args.batch_size < 1 or args.examples_per_epoch < 1:
        raise ValueError("invalid training controls")
    if not final and args.method.startswith("trust") != bool(args.oof_roots):
        raise ValueError("only learned trust methods require encoder-level OOF caches")
    selected, selected_path, selected_checkpoint, fixed_epochs = None, None, None, None
    if final:
        if args.oof_roots or args.selection_run is None or args.epochs != 0:
            raise ValueError("final refit requires --selection-run, no new OOF caches, and frozen selected epochs")
        selected_path = args.selection_run / "run.json" if args.selection_run.is_dir() else args.selection_run
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        if (selected.get("protocol_id") != PROTOCOL or selected.get("stage") != "official_training_internal_selection" or
                selected.get("method") != args.method or selected.get("official_test_accessed") is not False or
                selected.get("status") != ("synthetic_smoke_complete" if args.synthetic_smoke else "complete")):
            raise ValueError("final refit requires the matching completed inner-selection result")
        fixed_epochs = selected.get("best_epoch")
        if not isinstance(fixed_epochs, int) or fixed_epochs < 0:
            raise ValueError("selected epoch must be a nonnegative integer")
        if args.seed != selected.get("seed"):
            raise ValueError("final seed must match frozen inner configuration")
        for key in ("learning_rate", "batch_size", "examples_per_epoch"):
            if getattr(args, key) != selected.get("arguments", {}).get(key):
                raise ValueError(f"final hyperparameter differs from frozen selection: {key}")
        selected_checkpoint = selected_path.parent / "best.pt"
        if sha256_file(selected_checkpoint) != selected.get("checkpoint_sha256"):
            raise ValueError("selected checkpoint checksum mismatch")
    elif args.selection_run is not None:
        raise ValueError("inner fit may not initialize from a selected/final run")
    bank = load_role(args.cache_root / "train", "train", synthetic=args.synthetic_smoke, allow_test_inputs=final)
    query_role = "test_inputs" if final else "validation"
    val = load_role(args.cache_root / query_role, query_role, synthetic=args.synthetic_smoke, allow_test_inputs=final)
    if final and selected.get("official_contract_sha256") != bank["manifest"]["official_contract_sha256"]:
        raise ValueError("official membership contract changed after inner selection")
    expected = None if args.synthetic_smoke else (2506, 360 if final else 320, 40)
    neighbors, lineage = prepare_pair(bank, val, matched=args.method == "e1", expected_counts=expected)
    target_std = training_std(bank)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tensor = lambda value, dtype=torch.float32: torch.as_tensor(np.asarray(value).copy(), dtype=dtype, device=device)
    scale = tensor(target_std)
    gate_scaler, crossfit = None, None
    trust = args.method.startswith("trust")
    if trust:
        if final:
            saved_gate = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
            gate_scaler, crossfit = saved_gate["feature_scaler"], saved_gate["crossfit_evidence"]
            if not gate_scaler or gate_scaler.get("fit_role") != "train_encoder_OOF_only" or len(crossfit or []) != 3:
                raise ValueError("final frozen gate requires verified inner OOF state")
            training = None
        else:
            training, gate_scaler, crossfit = build_oof_examples(args.oof_roots, bank, synthetic=args.synthetic_smoke)
        validation = memory_inputs(bank, val, neighbors["validation"], target_std)
        validation["features"] = ((validation["features"] - np.asarray(gate_scaler["mean"])) /
                                  np.asarray(gate_scaler["std"]))
        for values in (training, validation):
            if values is None:
                continue
            for key in ("features", "base", "memory", "alpha", "legal", "bp"):
                if key in values:
                    values[key] = tensor(values[key], torch.bool if key == "legal" else torch.float32)
        model = TrustFusion(len(TRUST_FEATURE_NAMES), per_bp=args.method == "trust_bp").to(device)
        if final:
            model.load_state_dict(saved_gate["model_state"], strict=True)
        train_count = 0 if final else len(training["bp"])

        def batch(role, indices):
            data = training if role == "train" else validation
            prediction, alpha = model(*(data[key][indices] for key in ("features", "base", "memory", "alpha", "legal")))
            return prediction, alpha

        def loss_batch(indices):
            prediction, _ = batch("train", indices)
            return torch.nn.functional.huber_loss(prediction / scale, training["bp"][indices] / scale, delta=.5)
    else:
        model = PersonalMemoryRelation().to(device)
        tensors = {"features": tensor(bank["features"]), "bp": tensor(bank["bp"]), "base": tensor(bank["base"]),
                   "validation_features": tensor(val["features"]), "validation_base": tensor(val["base"])}
        donor = {role: {"index": tensor(neighbors[role]["knn_indices"], torch.long),
                        "weight": tensor(neighbors[role]["knn_weights"]),
                        "alpha": tensor(neighbors[role]["support_weight"])} for role in ("train", "validation")}
        train_count = len(bank["bp"])

        def batch(role, indices):
            item = donor[role]
            refs = item["index"][indices]
            legal = refs >= 0
            refs = refs.clamp_min(0)
            features = tensors["features"] if role == "train" else tensors["validation_features"]
            base = tensors["base"] if role == "train" else tensors["validation_base"]
            prediction, delta = reference_prediction(model, features[indices], tensors["features"][refs],
                tensors["bp"][refs], base[indices], item["weight"][indices], legal, item["alpha"][indices], scale)
            return prediction, (delta, tensors["bp"][refs], legal)

        def loss_batch(indices):
            prediction, (delta, reference_bp, legal) = batch("train", indices)
            return supervised_relation_loss(prediction, tensors["bp"][indices], delta,
                                              reference_bp, legal, scale, pair_weight=.25)[0]

    def evaluate():
        model.eval()
        chunks = []
        with torch.no_grad():
            for offset in range(0, len(val["metadata"]), args.validation_batch_size):
                index = torch.arange(offset, min(len(val["metadata"]), offset + args.validation_batch_size), device=device)
                chunks.append(batch("validation", index)[0].cpu().numpy())
        return np.concatenate(chunks)

    if args.method == "fixed" or (final and trust):
        model.requires_grad_(False)
    args.output.mkdir(parents=True)
    freeze_record = {"protocol_id": PROTOCOL, "stage": args.stage, "method": args.method,
                     "selected_run_sha256": sha256_file(selected_path) if final else None,
                     "selected_checkpoint_sha256": selected.get("checkpoint_sha256") if final else None,
                     "source_checkpoint_sha256": bank["manifest"]["checkpoint_sha256"],
                     "official_contract_sha256": bank["manifest"]["official_contract_sha256"],
                     "fixed_refit_epochs": 0 if final and trust else fixed_epochs,
                     "test_targets_accessed": False, "frozen_before_optimizer": True,
                     "created_utc": datetime.now(timezone.utc).isoformat()}
    save_json(args.output / "frozen_configuration.json", freeze_record)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_prediction = None if final else evaluate()
    initial_metrics = None if final else metrics(val["bp"], best_prediction, val["metadata"])
    best_metrics = initial_metrics
    best_score = float("inf") if final else best_metrics["Overall"]["mean_mae"]
    best_epoch = epoch = stale = optimizer_steps = 0
    best_trained_score, best_trained_epoch = None, None
    history = []
    checkpoint_path = args.output / "best.pt"

    def checkpoint(path, checkpoint_epoch, checkpoint_metrics):
        torch.save({"protocol_id": PROTOCOL, "method": args.method, "model_state": model.state_dict(),
                    "epoch": checkpoint_epoch, "metrics": checkpoint_metrics, "seed": args.seed,
                    "feature_scaler": gate_scaler, "target_std": target_std.tolist(),
                    "subject_to_index": {s: i for i, s in enumerate(sorted(bank["metadata"].subject_uid.unique()))},
                    "source_checkpoint_sha256": bank["manifest"]["checkpoint_sha256"],
                    "crossfit_evidence": crossfit, "official_test_accessed": final,
                    "official_test_inputs_accessed": final, "official_test_targets_accessed": False,
                    "retrieved_references": 5, "reference_bank_role": "train",
                    "inference_query_bp_input": False}, path)

    if not final:
        checkpoint(checkpoint_path, 0, best_metrics)
        prediction_frame(val["metadata"], val["bp"], best_prediction).to_parquet(args.output / "initial_validation_predictions.parquet", index=False)
    limit = fixed_epochs if final else args.epochs
    while args.method != "fixed" and not (final and trust) and ((not final and limit == 0) or epoch < limit):
        epoch += 1
        model.train()
        total_loss = seen = 0
        for offset in range(0, args.examples_per_epoch, args.batch_size):
            size = min(args.batch_size, args.examples_per_epoch - offset)
            index = tensor(rng.integers(train_count, size=size), torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_batch(index)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite memory fit loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * size
            seen += size
            optimizer_steps += 1
        if final:
            history.append({"epoch": epoch, "train_loss": total_loss / seen, "optimizer_steps": optimizer_steps,
                            "selection": "none_fixed_inner_selected_epochs", "test_targets_accessed": False})
            save_json(args.output / "history.json", history)
            print(json.dumps(history[-1]), flush=True)
            continue
        prediction = evaluate()
        views = metrics(val["bp"], prediction, val["metadata"])
        score = views["Overall"]["mean_mae"]
        if best_trained_score is None or score < best_trained_score:
            best_trained_score, best_trained_epoch = score, epoch
            checkpoint(args.output / "best_trained.pt", epoch, views)
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_prediction, best_metrics = prediction, views
            checkpoint(checkpoint_path, epoch, views)
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "metrics": views,
                        "stale_epochs": stale, "optimizer_steps": optimizer_steps})
        save_json(args.output / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
        if stale >= args.patience:
            break
    if final:
        best_epoch = epoch
        best_prediction = evaluate()
        checkpoint(checkpoint_path, epoch, None)
    frame = prediction_frame(val["metadata"], val["bp"], best_prediction)
    prediction_file = args.output / ("official_test_input_predictions.parquet" if final else "validation_predictions.parquet")
    frame.to_parquet(prediction_file, index=False)
    if not final:
        save_json(args.output / "metrics.json", best_metrics)
        from .calbased_metrics import pooled_diagnostics
        pooled_diagnostics(frame, "official_inner_" + args.method).to_csv(args.output / "pooled_diagnostics.csv", index=False)
    run = {"status": "synthetic_smoke_complete" if args.synthetic_smoke else "complete",
           "protocol_id": PROTOCOL, "stage": "official_final_frozen_prediction" if final else "official_training_internal_selection", "method": args.method,
           "official_test_accessed": final, "official_test_inputs_accessed": final,
           "official_test_targets_accessed": False, "official_final_test_result": False,
           "read_roles": ["train", query_role] + (["OOF_source_fit", "OOF_excluded_fold"] if trust and not final else []),
           "query_bp_is_prediction_input": False, "validation_updates_personal_state": False,
           "personal_label_budget_windows": 360 if final else 320, "event_is_cuff_measurement": False,
           "gate_supervised_label_budget_windows": 320 if trust else None,
           "gate_frozen_from_inner_320_OOF": final and trust,
           "official_train_budget_after_final_refit": 360, "retrieved_references": 5,
           "source_checkpoint_sha256": bank["manifest"]["checkpoint_sha256"],
           "official_contract_sha256": bank["manifest"]["official_contract_sha256"],
           "cache_provenance_sha256": {"train": bank["manifest_sha256"], query_role: val["manifest_sha256"]},
           "lineage_audit": lineage, "crossfit_evidence": crossfit,
           "feature_provenance": "encoder-level OOF for trust training" if trust else "supervised frozen LoRA; not OOF",
           "trust_feature_names": TRUST_FEATURE_NAMES if trust else None, "trust_scaler": gate_scaler,
           "novelty_claim": "unverified prospective adaptation; generic gating and retrieval are precedents",
           "initial_metrics": initial_metrics, "metrics": best_metrics, "best_epoch": best_epoch,
           "best_trained_epoch": best_trained_epoch, "best_trained_mean_mae": best_trained_score,
           "epochs_completed": epoch, "optimizer_steps": optimizer_steps,
           "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "stored_neural_parameters": sum(p.numel() for p in model.parameters()),
           "stop_reason": "frozen_inner_OOF_gate_no_refit" if final and trust else "fixed_no_optimizer" if args.method == "fixed" else "fixed_inner_selected_epochs" if final else "early_stopping" if stale >= args.patience else "explicit_epoch_limit",
           "frozen_configuration": freeze_record, "predictions_sha256": sha256_file(prediction_file),
           "checkpoint_sha256": sha256_file(checkpoint_path), "seed": args.seed, "device": str(device),
           "torch_version": torch.__version__, "runtime_seconds": time.monotonic() - started,
           "generated_at_utc": datetime.now(timezone.utc).isoformat(),
           "arguments": {key: str(value) if isinstance(value, Path) else [str(x) for x in value] if isinstance(value, list) else value
                         for key, value in vars(args).items()}}
    save_json(args.output / "run.json", run)
    return run


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--stage", choices=("inner", "final"), default="inner")
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--selection-run", type=Path)
    p.add_argument("--oof-roots", type=Path, nargs=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--epochs", type=int, default=0, help="0 means patience-only stopping")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--validation-batch-size", type=int, default=1024)
    p.add_argument("--examples-per-epoch", type=int, default=200000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--synthetic-smoke", action="store_true", help="requires explicitly synthetic caches; never accepts real data")
    return p


def main():
    print(json.dumps(fit(parser().parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
