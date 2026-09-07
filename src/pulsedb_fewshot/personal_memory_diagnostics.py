"""Frozen, full-cohort personal-memory diagnostics and fixed E3 reliability.

No fitting, checkpoint selection, query-label routing, or held-out access occurs.
The original v1 artifacts remain immutable. The source relation must reproduce
its complete saved internal-validation predictions before any new comparison.
Temporal matching is record-local; all uncovered queries return frozen LoRA.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .personal_memory_prepare import (
    PROTOCOL, SCOPES, audit_metadata, diagnostic_predictions, macro_metrics,
    prepare_neighbors, sha256,
)

SCREEN_ID = "personal-memory-v2"
REPLAY_ATOL = 2e-5
MATCHING_SEED = 20260908
NEIGHBOR_NAMES = ("knn_indices", "knn_weights", "support_weight", "valid",
                  "nearest_distance", "nearest_time_index")


def fixed_reliability(base_bp, local_bp, weights, legal, old_alpha, target_std):
    """E3 fixed rule; deliberately no query BP/error argument.

    Scatter is the weighted RMS dispersion of reference-specific predictions,
    normalized by the immutable source TRAIN SBP/DBP standard deviations.
    ESS = 1/sum(w**2); alpha = old_alpha * (ESS/5)/(1+scatter).
    Reference-specific BP includes a relation delta for the trained variant.
    """
    base, local, weight, alpha, std = (
        np.asarray(a, dtype=np.float64)
        for a in (base_bp, local_bp, weights, old_alpha, target_std)
    )
    legal = np.asarray(legal, dtype=bool)
    n = len(base)
    if base.shape != (n, 2) or local.shape != (n, 5, 2) or weight.shape != (n, 5) \
            or legal.shape != weight.shape or alpha.shape != (n,) or std.shape != (2,):
        raise ValueError("E3 requires exactly five reference slots and two BP outputs")
    if not all(np.isfinite(a).all() for a in (base, local, weight, alpha, std)) \
            or (weight < 0).any() or ((alpha < 0) | (alpha > 1)).any() or (std <= 0).any():
        raise ValueError("invalid fixed-reliability inputs")
    effective = weight * legal
    total = effective.sum(axis=1, keepdims=True)
    normalized = effective / np.maximum(total, np.finfo(np.float64).eps)
    present = total[:, 0] > 0
    memory = (normalized[:, :, None] * local).sum(axis=1)
    dispersion = ((local - memory[:, None, :]) / std) ** 2
    scatter = np.sqrt((normalized[:, :, None] * dispersion).sum(axis=1).mean(axis=1))
    square_sum = (normalized ** 2).sum(axis=1)
    ess = np.divide(1, square_sum, out=np.zeros(n), where=present)
    adjusted = alpha * np.clip(ess / 5, 0, 1) / (1 + scatter)
    adjusted[~present] = 0
    result = base + adjusted[:, None] * (memory - base)
    return result, {"alpha": adjusted, "scatter": scatter, "ess": ess}


def frozen_distance_alpha(distance, valid, q95):
    """Keep the source gate scale frozen even when excluding more references."""
    distance, valid = np.asarray(distance), np.asarray(valid, dtype=bool)
    result = np.zeros(len(distance), dtype=np.float32)
    if q95 is not None:
        if not np.isfinite(q95) or q95 < 0:
            raise ValueError("invalid frozen train q95")
        result[valid] = np.clip(1 - distance[valid] / q95, 0, 1) if q95 > 1e-12 \
            else (distance[valid] <= 1e-12).astype(np.float32)
    return result


def reference_time_gaps(bank, queries, indices):
    """Half-open interval gaps; NaN means no comparable physical record clock."""
    indices = np.asarray(indices)
    if indices.ndim != 2 or len(indices) != len(queries):
        raise ValueError("reference/query shape mismatch")
    if (indices < -1).any() or (indices >= len(bank)).any():
        raise ValueError("reference index outside train bank")
    safe = np.maximum(indices, 0)
    def column(rows, key):
        return np.asarray([r[key] for r in rows])
    comparable = indices >= 0
    for key in ("subject_uid", "source", "recording_uid", "time_axis_uid"):
        comparable &= column(bank, key)[safe] == column(queries, key)[:, None]
    start, end = column(bank, "start_s")[safe], column(bank, "end_s")[safe]
    gaps = np.maximum(start - column(queries, "end_s")[:, None],
                      column(queries, "start_s")[:, None] - end).astype(np.float64)
    gaps[~comparable] = np.nan
    return gaps


def fit_temporal_terciles(train, train_indices):
    """Fit on comparable gaps of TRAIN queries' five selected legal references.

    This bounded train-only distribution (at most 5N entries) is explicit, not
    a sample of validation distances or a post-hoc fit to matching success.
    """
    gaps = reference_time_gaps(train, train, train_indices)
    values = gaps[np.isfinite(gaps)]
    if (values < -1e-7).any():
        raise ValueError("training references contain overlapping intervals")
    return {"cuts_s": None if not len(values) else np.quantile(values, [1/3, 2/3]).tolist(),
            "n_train_reference_gaps": int(len(values)),
            "fit_role": "train", "fit_distribution": "legal_selected_train_knn_reference_gaps"}


def time_matched_references(train, validation, feature_indices, temporal_fit,
                            *, gap_s, past_only, seed=MATCHING_SEED):
    """Match each query's feature donors by train-fitted temporal stratum.

    Five distinct random legal references are selected without replacement;
    the original feature donors are not artificially excluded. Each arm uses
    uniform weights and identical all-query fallback, isolating donor choice.
    Cross-record feature sets cannot be clock-matched and fall back in BOTH
    comparison arms; the unrestricted feature-kNN result is separate.
    """
    original = np.asarray(feature_indices, dtype=np.int64)
    if original.shape != (len(validation), 5) or gap_s < 0:
        raise ValueError("invalid time-matching configuration")
    selected = np.full_like(original, -1)
    paired = np.full_like(original, -1)
    selected_strata = np.full_like(original, -1)
    cuts = temporal_fit.get("cuts_s")
    reasons = defaultdict(int)
    if cuts is None:
        return {"random_indices": selected, "feature_indices": paired,
                "strata": selected_strata, "fallback_reasons": {"no_train_clock_gaps": len(validation)}}
    cuts = np.asarray(cuts, dtype=float)
    if cuts.shape != (2,) or not np.isfinite(cuts).all() or cuts[0] > cuts[1]:
        raise ValueError("invalid train-only temporal terciles")
    groups = defaultdict(list)
    for i, row in enumerate(train):
        groups[row["subject_uid"]].append(i)
    grouped = {}
    for person, idx in groups.items():
        ids = np.asarray(idx, dtype=np.int64)
        grouped[person] = (ids, {key: np.asarray([train[i][key] for i in idx]) for key in
            ("start_s", "end_s", "recording_uid", "time_axis_uid", "window_uid", "source")})
    feature_gaps = reference_time_gaps(train, validation, original)
    for i, query in enumerate(validation):
        if (original[i] < 0).any():
            reasons["insufficient_feature_references"] += 1
            continue
        if not np.isfinite(feature_gaps[i]).all():
            reasons["feature_references_cross_record_clock"] += 1
            continue
        ids, values = grouped[query["subject_uid"]]
        legal = ((values["recording_uid"] == query["recording_uid"])
                 & (values["time_axis_uid"] == query["time_axis_uid"])
                 & (values["source"] == query["source"])
                 & (values["window_uid"] != query["window_uid"]))
        gaps = np.maximum(values["start_s"] - query["end_s"], query["start_s"] - values["end_s"])
        legal &= gaps >= gap_s - 1e-7
        if past_only:
            legal &= values["end_s"] <= query["start_s"] - gap_s + 1e-7
        if not set(original[i]).issubset(set(ids[legal])):
            raise ValueError("selected feature reference violates temporal matching legality")
        target_strata = np.searchsorted(cuts, feature_gaps[i], side="right")
        bank_strata = np.searchsorted(cuts, gaps, side="right")
        digest = hashlib.sha256(f"{seed}:{query['window_uid']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        chosen = np.full(5, -1, dtype=np.int64)
        for stratum in range(3):
            slots = np.flatnonzero(target_strata == stratum)
            pool = np.flatnonzero(legal & (bank_strata == stratum))
            if len(pool) < len(slots):
                break
            # Sorting makes a frozen query's selection invariant to bank row order.
            pool = pool[np.argsort(values["window_uid"][pool], kind="stable")]
            if len(slots):
                chosen[slots] = ids[rng.choice(pool, len(slots), replace=False)]
        if (chosen < 0).any():
            reasons["insufficient_stratum_references"] += 1
            continue
        selected[i], paired[i], selected_strata[i] = chosen, original[i], target_strata
    return {"random_indices": selected, "feature_indices": paired, "strata": selected_strata,
            "fallback_reasons": dict(reasons)}


def uniform_bp_prediction(train_bp, base, indices):
    result = np.asarray(base, dtype=float).copy()
    valid = (indices >= 0).all(axis=1)
    result[valid] = np.asarray(train_bp)[indices[valid]].mean(axis=1)
    return result


def describe(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return {"n_finite": int(len(x)), "mean": float(x.mean()) if len(x) else None,
            "p05": float(np.quantile(x, .05)) if len(x) else None,
            "p50": float(np.median(x)) if len(x) else None,
            "p95": float(np.quantile(x, .95)) if len(x) else None}


def assert_reproduction(expected, reproduced, *, atol=REPLAY_ATOL):
    expected, reproduced = np.asarray(expected), np.asarray(reproduced)
    if expected.shape != reproduced.shape or not np.isfinite(expected).all() \
            or not np.isfinite(reproduced).all() or expected.ndim != 2 or expected.shape[1] != 2:
        raise ValueError("invalid full prediction replay arrays")
    error = np.abs(expected.astype(float) - reproduced.astype(float))
    result = {"n_queries": len(expected), "max_absolute_error_mmhg": float(error.max()),
              "mean_absolute_error_mmhg": float(error.mean()), "absolute_tolerance_mmhg": atol,
              "status": "pass" if (error <= atol).all() else "fail"}
    if result["status"] != "pass":
        raise ValueError(f"source full prediction reproduction failed: {result}")
    return result


def _frozen_inference(model, tensors, neighbors, std, *, batch_size, include_reliability=True):
    """GPU feature-only inference; no validation target tensor is accepted."""
    import torch
    from .personal_memory_models import reference_prediction
    reliability_names = ("E3_zero_reliability", "E3_trained_reliability") if include_reliability else ()
    outputs = {name: [] for name in ("fixed_blend", "trained_blend", *reliability_names)}
    diagnostics = {name: defaultdict(list) for name in reliability_names}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(neighbors["valid"]), batch_size):
            stop = min(start + batch_size, len(neighbors["valid"]))
            index = torch.as_tensor(neighbors["knn_indices"][start:stop], device=std.device)
            legal, safe = index >= 0, index.clamp_min(0)
            weights = torch.as_tensor(neighbors["knn_weights"][start:stop], device=std.device)
            alpha = torch.as_tensor(neighbors["support_weight"][start:stop], device=std.device)
            base = tensors["validation_base"][start:stop]
            references = tensors["train_bp"][safe]
            trained, delta = reference_prediction(model, tensors["validation_features"][start:stop],
                tensors["train_features"][safe], references, base, weights, legal, alpha, std)
            effective = weights * legal.to(weights.dtype)
            total = effective.sum(1, keepdim=True)
            normalized = effective / total.clamp_min(torch.finfo(effective.dtype).eps)
            zero_memory = (normalized[:, :, None] * references).sum(1)
            fixed = base + (alpha * (total[:, 0] > 0))[:, None] * (zero_memory - base)
            if not torch.isfinite(trained).all() or not torch.isfinite(fixed).all():
                raise FloatingPointError("nonfinite frozen predictions")
            outputs["fixed_blend"].append(fixed.cpu().numpy())
            outputs["trained_blend"].append(trained.cpu().numpy())
            for name in reliability_names:
                local = references if name == "E3_zero_reliability" else references + delta * std
                pred, extra = fixed_reliability(base.cpu().numpy(), local.cpu().numpy(),
                    weights.cpu().numpy(), legal.cpu().numpy(), alpha.cpu().numpy(), std.cpu().numpy())
                outputs[name].append(pred)
                for key, value in extra.items():
                    diagnostics[name][key].append(value)
    return ({key: np.concatenate(value) for key, value in outputs.items()},
            {name: {key: np.concatenate(value) for key, value in item.items()}
             for name, item in diagnostics.items()})


def _source_checkpoint(source, manifest, cache_digest, device):
    import torch
    from .personal_memory_models import PersonalMemoryRelation
    run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    for key, expected in {"status": "complete", "protocol_id": PROTOCOL,
            "screen_id": "personal-memory-v1", "candidate": "pair_distance_blend",
            "split_mode": manifest["split_mode"], "heldout_test_accessed": False,
            "source_parent_split": "meta_train", "read_roles": ["train", "internal_validation"],
            "cache_manifest_sha256": cache_digest,
            "source_checkpoint_sha256": manifest["source_checkpoint_sha256"]}.items():
        if run.get(key) != expected:
            raise ValueError(f"source relation provenance mismatch: {key}")
    checkpoint_path = source / "best.pt"
    if sha256(checkpoint_path) != run.get("checkpoint_sha256"):
        raise ValueError("source relation checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for key, expected in {"candidate": "pair_distance_blend", "cache_manifest_sha256": cache_digest,
            "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
            "target_scaler": manifest["target_scaler"], "epoch": run["best_epoch"]}.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"source checkpoint provenance mismatch: {key}")
    model = PersonalMemoryRelation().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return run, checkpoint, model


def _load_neighbors(cache, manifest, prefix):
    from .personal_memory_train import _read_file
    return {role: {name: np.load(_read_file(cache, manifest, f"{prefix}{role}_{name}.npy"),
            allow_pickle=False) for name in NEIGHBOR_NAMES} for role in ("train", "validation")}


def _validate_neighbors(neighbors, train, validation, *, gap_s, past_only):
    """Independently verify loaded/reference files against the audited clock."""
    for role, queries in (("train", train), ("validation", validation)):
        values = neighbors[role]
        idx = values["knn_indices"]
        if idx.shape != (len(queries), 5) or not np.issubdtype(idx.dtype, np.integer) \
                or (idx < -1).any() or (idx >= len(train)).any():
            raise ValueError("invalid neighbor index array")
        valid = (idx >= 0).all(axis=1)
        if ((idx >= 0).any(axis=1) != valid).any() or not np.array_equal(valid, values["valid"]):
            raise ValueError("neighbor fallback flags must agree with all-minus-one rows")
        if ((np.diff(np.sort(idx, axis=1), axis=1) == 0) & valid[:, None]).any():
            raise ValueError("neighbor donors must be distinct")
        safe = np.maximum(idx, 0)
        def col(rows, name):
            return np.asarray([row[name] for row in rows])
        for field in ("subject_uid", "source"):
            if ((col(train, field)[safe] != col(queries, field)[:, None]) & valid[:, None]).any():
                raise ValueError(f"neighbor {field} mismatch")
        for field in ("window_uid", "waveform_sha256"):
            if ((col(train, field)[safe] == col(queries, field)[:, None]) & valid[:, None]).any():
                raise ValueError("neighbor self/duplicate waveform reference")
        gaps = reference_time_gaps(train, queries, idx)
        if ((gaps < gap_s - 1e-7) & valid[:, None]).any():
            raise ValueError("neighbor violates physiological exclusion band")
        if past_only:
            ends = col(train, "end_s")[safe]
            starts = col(queries, "start_s")[:, None]
            if ((~np.isfinite(gaps) | (ends > starts - gap_s + 1e-7)) & valid[:, None]).any():
                raise ValueError("chronological references must be same-record past only")
        weights = values["knn_weights"]
        if weights.shape != idx.shape or not np.isfinite(weights).all() or (weights < 0).any() \
                or not np.allclose(weights[valid].sum(1), 1, atol=1e-5) \
                or (weights[~valid] != 0).any():
            raise ValueError("invalid fixed neighbor weights")
        distance = values["nearest_distance"]
        alpha = values["support_weight"]
        if distance.shape != (len(queries),) or not np.isfinite(distance[valid]).all() \
                or alpha.shape != (len(queries),) or not np.isfinite(alpha).all() \
                or ((alpha < 0) | (alpha > 1)).any() or (alpha[~valid] != 0).any():
            raise ValueError("invalid neighbor distance/alpha")
        time_index = values["nearest_time_index"]
        if time_index.shape != (len(queries),) or (time_index < -1).any() or (time_index >= len(train)).any():
            raise ValueError("invalid nearest-time reference index")
        has_time = time_index >= 0
        time_gap = reference_time_gaps(train, queries, time_index[:, None])[:, 0]
        if ((~np.isfinite(time_gap) | (time_gap < gap_s - 1e-7)) & has_time).any():
            raise ValueError("nearest-time donor violates record/interval constraints")
        if past_only:
            if ((col(train, "end_s")[np.maximum(time_index, 0)] > col(queries, "start_s")
                 - gap_s + 1e-7) & has_time).any():
                raise ValueError("nearest-time donor is not past only")


def _scoring_groups(target, train_bp, train_rows, validation_rows):
    """Called for reporting only, after every model prediction is already frozen."""
    grouped = defaultdict(list)
    for index, row in enumerate(train_rows):
        grouped[row["subject_uid"]].append(index)
    medians = {person: np.median(np.asarray(train_bp)[indices], axis=0)
               for person, indices in grouped.items()}
    deviation = np.abs(np.asarray(target) - np.array([medians[row["subject_uid"]] for row in validation_rows]))
    result = {}
    for bp_index, bp in enumerate(("SBP", "DBP")):
        for label, lower, upper in (("0-10", 0, 10), ("10-20", 10, 20), (">20", 20, np.inf)):
            mask = (deviation[:, bp_index] > lower) & (deviation[:, bp_index] <= upper)
            if lower == 0:
                mask = deviation[:, bp_index] <= upper
            result[f"{bp}_train_median_deviation_{label}"] = mask
    return result


def run_diagnostics(args):
    import pandas as pd
    import torch
    from .calbased_metrics import participant_macro_views, pooled_diagnostics
    from .personal_memory_train import _read_file, load_cache
    from .training import seed_everything, source_tree_sha256
    start = time.monotonic()
    cache, source, output = args.cache_dir.resolve(), args.source_relation_run.resolve(), args.output.resolve()
    if output == cache or output == source or cache in output.parents or source in output.parents:
        raise ValueError("diagnostics must not modify original cache/source run directories")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty diagnostics output")
    manifest_digest = sha256(cache / "manifest.json")
    manifest, arrays, metadata = load_cache(cache, "pair_distance_blend")
    rows = {role: table.to_dict("records") for role, table in metadata.items()}
    lineage = audit_metadata(rows["train"], rows["validation"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested GPU is unavailable")
    seed_everything(MATCHING_SEED)
    source_run, checkpoint, model = _source_checkpoint(source, manifest, manifest_digest, device)
    mapping = {person: i for i, person in enumerate(sorted(metadata["train"].subject_uid.unique()))}
    if checkpoint["subject_to_index"] != mapping:
        raise ValueError("source persistent participant mapping mismatch")
    tensors = {key: torch.as_tensor(np.array(arrays[key], copy=True), device=device, dtype=torch.float32)
               for key in ("train_features", "validation_features", "train_bp", "validation_base")}
    std = torch.as_tensor(manifest["target_scaler"]["std"], device=device, dtype=torch.float32)
    batch_size = int(source_run.get("arguments", {}).get("validation_batch_size", 1024))
    if batch_size < 1:
        raise ValueError("invalid original validation batch size")
    neighbors0 = _load_neighbors(cache, manifest, "")
    chronological = manifest["split_mode"] == "chronological_blocked"
    _validate_neighbors(neighbors0, rows["train"], rows["validation"], gap_s=0, past_only=chronological)
    main_predictions, reliability = _frozen_inference(model, tensors, neighbors0["validation"], std,
                                                     batch_size=batch_size)
    old = pd.read_parquet(source / "best_internal_validation_predictions.parquet")
    keys = ["subject_uid", "event_id", "source"]
    target_columns = ["target_sbp", "target_dbp"]
    if old.event_id.duplicated().any() or len(old) != len(metadata["validation"]):
        raise ValueError("source replay requires the exact complete query cohort")
    canonical_targets = metadata["validation"][keys].copy()
    canonical_targets["target_sbp"], canonical_targets["target_dbp"] = arrays["validation_bp"][:, 0], arrays["validation_bp"][:, 1]
    aligned = canonical_targets.merge(old, on=keys, how="left",
                suffixes=("_expected", ""), validate="one_to_one", sort=False)
    if aligned.isna().any().any() or not np.allclose(
        aligned[[f"{c}_expected" for c in target_columns]].to_numpy(dtype=np.float32),
        aligned[target_columns].to_numpy(dtype=np.float32), rtol=0, atol=REPLAY_ATOL):
        raise ValueError("source replay query identity or target mismatch")
    reproduction = assert_reproduction(aligned[["pred_sbp", "pred_dbp"]].to_numpy(),
                                      main_predictions["trained_blend"])
    reproduction.update({"exact_query_keys": True, "canonical_float32_target_tolerance_mmhg": REPLAY_ATOL,
                         "source_best_epoch": source_run["best_epoch"],
                         "source_prediction_sha256": sha256(source / "best_internal_validation_predictions.parquet")})
    probe = json.loads(_read_file(cache, manifest, "probe_summary.json").read_text(encoding="utf-8"))
    frozen_q95 = probe["train_distance_cutpoints"]["q95"]
    for role in ("train", "validation"):
        replay_alpha = frozen_distance_alpha(neighbors0[role]["nearest_distance"], neighbors0[role]["valid"], frozen_q95)
        if not np.allclose(replay_alpha, neighbors0[role]["support_weight"], rtol=0, atol=1e-7):
            raise ValueError("original gate cannot be reconstructed from its recorded train q95")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "screen_id": SCREEN_ID, "protocol_id": PROTOCOL,
        "split_mode": manifest["split_mode"], "source_parent_split": "meta_train",
        "read_roles": ["train", "internal_validation"], "heldout_test_accessed": False,
        "selection_role": "internal_validation", "official_pulsedb_calbased_reproduction": False,
        "parameters_updated": False, "query_labels_used_for_routing": False,
        "cache_manifest_sha256": manifest_digest,
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "source_relation_checkpoint_sha256": source_run["checkpoint_sha256"],
        "source_relation_run_sha256": sha256(source / "run.json"),
        "source_best_epoch": source_run["best_epoch"], "lineage_audit": lineage,
        "personal_training_budget": 320, "retrieval_count": 5,
        "reproduction": reproduction, "prediction_files": {}, "files": {}, "conditions": {},
        "metrics": {}, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "frozen_gate_q95": frozen_q95,
        "fusion_rule": "original_alpha*(ESS/5)/(1+weighted_RMS_scatter_in_train_BP_std_units)",
        "reliability_gate_learned": False, "E3_settings": ["E3_zero_reliability", "E3_trained_reliability"],
        "E3_gap": 0, "future_train_references_in_random_main": True,
        "temporal_matching_weights": "uniform_in_both_coverage_matched_arms",
        "temporal_matching_seed": MATCHING_SEED,
        "claim_limit": "development registered-user diagnostics; not clinical validation or OOF"}
    metric_rows, pooled, subgroups = [], [], []
    target = arrays["validation_bp"]
    subjects = metadata["validation"].subject_uid.to_numpy()
    sources = metadata["validation"].source.to_numpy()
    scoring_groups = _scoring_groups(target, arrays["train_bp"], rows["train"], rows["validation"])
    def register(path):
        summary["files"][path.name] = {"path": path.name, "sha256": sha256(path)}
    def save_predictions(setting, values):
        frame = canonical_targets.copy()
        frame["pred_sbp"], frame["pred_dbp"] = values[:, 0], values[:, 1]
        path = output / f"private_{setting}_predictions.parquet"
        frame.to_parquet(path, index=False)
        summary["prediction_files"][setting] = path.name
        register(path)
        views = participant_macro_views(frame)
        summary["metrics"][setting] = views
        for scope, metrics in views.items():
            metric_rows.append({"Setting": setting, "Scope": scope, **metrics})
        pooled.append(pooled_diagnostics(frame, setting))
        for group, group_mask in scoring_groups.items():
            for scope in SCOPES:
                mask = group_mask & (np.ones(len(frame), dtype=bool) if scope == "Overall" else sources == scope)
                metrics = macro_metrics(target[mask], values[mask], subjects[mask])
                base_metrics = macro_metrics(target[mask], arrays["validation_base"][mask], subjects[mask])
                subgroups.append({"Setting": setting, "Scope": scope, "scoring_only_group": group,
                    **metrics, "delta_mean_mae_vs_frozen_lora": None if not mask.any() else
                    metrics["mean_mae"] - base_metrics["mean_mae"],
                    "delta_sbp_mae_vs_frozen_lora": None if not mask.any() else metrics["sbp_mae"]-base_metrics["sbp_mae"],
                    "delta_dbp_mae_vs_frozen_lora": None if not mask.any() else metrics["dbp_mae"]-base_metrics["dbp_mae"]})
    conditions = [(f"gap{gap}", gap, chronological) for gap in (0, 60, 300)]
    if not chronological:
        conditions += [(f"past_gap{gap}", gap, True) for gap in (0, 60, 300)]
    for condition, gap, past_only in conditions:
        print(json.dumps({"condition": condition, "phase": "frozen_diagnostics"}), flush=True)
        if condition == "gap0":
            neighbors, predictions = neighbors0, main_predictions
        else:
            cached60 = condition == "gap60" and all(f"sensitivity60_{role}_{name}.npy" in manifest["files"]
                    for role in ("train", "validation") for name in NEIGHBOR_NAMES)
            neighbors = _load_neighbors(cache, manifest, "sensitivity60_") if cached60 else prepare_neighbors(
                arrays["train_features"], arrays["validation_features"], rows["train"], rows["validation"],
                mode="chronological_blocked" if past_only else "random_disjoint", exclusion_s=gap, audit=False)
            _validate_neighbors(neighbors, rows["train"], rows["validation"], gap_s=gap, past_only=past_only)
            for role in ("train", "validation"):
                neighbors[role]["support_weight"] = frozen_distance_alpha(
                    neighbors[role]["nearest_distance"], neighbors[role]["valid"], frozen_q95)
            predictions, _ = _frozen_inference(model, tensors, neighbors["validation"], std,
                                              batch_size=batch_size, include_reliability=False)
        # Only two predeclared E3 settings, both at the original main gap zero.
        predictions = dict(predictions)
        old_probes = diagnostic_predictions(arrays["train_bp"], arrays["train_base"], arrays["validation_base"],
                                            neighbors["validation"])
        predictions.update({"D0_frozen_lora": old_probes["D0"], "feature_knn": old_probes["D1"],
                            "nearest_record_time_bp": old_probes["D3"]})
        temporal_fit = fit_temporal_terciles(rows["train"], neighbors["train"]["knn_indices"])
        matched = time_matched_references(rows["train"], rows["validation"], neighbors["validation"]["knn_indices"],
                                          temporal_fit, gap_s=gap, past_only=past_only)
        predictions["time_matched_feature_uniform"] = uniform_bp_prediction(
            arrays["train_bp"], arrays["validation_base"], matched["feature_indices"])
        predictions["time_stratum_random_uniform"] = uniform_bp_prediction(
            arrays["train_bp"], arrays["validation_base"], matched["random_indices"])
        info = {"gap_s": gap, "past_only": past_only, "frozen_train_q95": frozen_q95,
                "all_queries_retained": True, "temporal_fit": temporal_fit,
                "matching_fallback_reasons": matched["fallback_reasons"],
                "n_time_matched_queries": int((matched["random_indices"] >= 0).all(1).sum()), "roles": {}}
        for role in ("train", "validation"):
            n = neighbors[role]
            info["roles"][role] = {"n_queries": len(n["valid"]), "n_memory_fallback": int((~n["valid"]).sum()),
                "n_alpha_zero": int((n["support_weight"] == 0).sum()),
                "n_time_fallback": int((n["nearest_time_index"] < 0).sum()),
                "nearest_feature_distance": describe(n["nearest_distance"]),
                "selected_record_time_gaps_s": describe(reference_time_gaps(rows["train"], rows[role], n["knn_indices"])),
                "alpha": describe(n["support_weight"])}
        evidence_path = output / f"private_{condition}_retrieval.npz"
        np.savez_compressed(evidence_path, **{f"validation_{name}": values for name, values in neighbors["validation"].items()},
                            matched_random_indices=matched["random_indices"], matched_feature_indices=matched["feature_indices"],
                            matched_strata=matched["strata"])
        register(evidence_path)
        if condition == "gap0":
            info["reliability"] = {name: {key: describe(value) for key, value in item.items()}
                                     for name, item in reliability.items()}
            path = output / "private_gap0_reliability.npz"
            np.savez_compressed(path, **{f"{name}_{key}": value for name, item in reliability.items() for key, value in item.items()})
            register(path)
        for setting, prediction in predictions.items():
            name = setting if condition == "gap0" else f"{condition}__{setting}"
            save_predictions(name, prediction)
        summary["conditions"][condition] = info
    for filename, frame in (("metrics.csv", pd.DataFrame(metric_rows)),
                            ("pooled_diagnostics.csv", pd.concat(pooled, ignore_index=True)),
                            ("subgroup_diagnostics.csv", pd.DataFrame(subgroups))):
        path = output / filename
        frame.to_csv(path, index=False)
        register(path)
    # Re-hash immutable sources at completion, not just before inference.
    if sha256(cache / "manifest.json") != manifest_digest or sha256(source / "best.pt") != source_run["checkpoint_sha256"]:
        raise ValueError("source artifacts changed during frozen diagnostics")
    summary.update(status="complete", runtime_seconds=time.monotonic()-start)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-relation-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_diagnostics(args)
    print(json.dumps({"status": result["status"], "split_mode": result["split_mode"],
                      "runtime_seconds": result["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
