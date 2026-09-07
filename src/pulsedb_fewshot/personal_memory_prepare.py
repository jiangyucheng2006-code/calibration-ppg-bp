"""Prepare train-only personal neighbours and frozen diagnostic predictions.

The encoder saw all train labels: leave-block-out retrieval is NOT an OOF
encoder or OOF residual estimate. Internal-validation BP is consulted only for
scoring after retrieval parameters and arrays are fixed; metadata may carry
unused scoring columns earlier. No held-out files are discovered or read.
Export provenance is verified, not inferred.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

PROTOCOL = "development-calbased-analogue-v1"
MODES = {"random_disjoint", "chronological_blocked"}
FIELDS = ("subject_uid", "event_id", "source", "window_uid", "waveform_sha256",
          "recording_uid", "time_axis_uid", "start_s", "end_s", "role")
SCOPES = ("Overall", "MIMIC", "VitalDB")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def audit_metadata(train: list[dict], validation: list[dict]) -> dict:
    """Audit globally before person filtering; clocks are explicit half-open axes."""
    if not train or not validation:
        raise ValueError("both permitted roles require nonempty metadata")
    sets = {}
    sources, recording_axes, recordings = {}, {}, defaultdict(list)
    for rows, role in ((train, "train"), (validation, "internal_validation")):
        keys = {name: set() for name in ("event_id", "window_uid", "waveform_sha256")}
        for row in rows:
            if any(name not in row for name in FIELDS):
                raise ValueError("metadata is missing explicit lineage fields")
            if row["role"] != role:
                raise ValueError("forbidden or incorrect metadata role")
            for name in FIELDS:
                if name not in {"start_s", "end_s"} and (
                    not isinstance(row[name], str) or not row[name].strip()
                ):
                    raise ValueError(f"nonempty string required: {name}")
            if row["source"] not in {"MIMIC", "VitalDB"}:
                raise ValueError("unrecognized source")
            if not re.fullmatch(r"[0-9a-f]{64}", row["waveform_sha256"]):
                raise ValueError("canonical waveform SHA-256 required")
            if not np.isfinite([row["start_s"], row["end_s"]]).all() or row["end_s"] <= row["start_s"]:
                raise ValueError("finite positive half-open interval required")
            for name, seen in keys.items():
                if row[name] in seen:
                    raise ValueError(f"duplicate {name} within role")
                seen.add(row[name])
            person = row["subject_uid"]
            if person in sources and sources[person] != row["source"]:
                raise ValueError("canonical subject has conflicting sources")
            sources[person] = row["source"]
            record = (row["source"], row["recording_uid"])
            if record in recording_axes and recording_axes[record] != row["time_axis_uid"]:
                raise ValueError("recording has inconsistent canonical clocks")
            recording_axes[record] = row["time_axis_uid"]
            recordings[record].append(row)
        sets[role] = keys
    for name in sets["train"]:
        if sets["train"][name] & sets["internal_validation"][name]:
            raise ValueError(f"global cross-role {name} overlap")
    # This is intentionally not grouped by subject: a mislabelled subject must
    # not hide reuse of the same physical recording interval across roles.
    for rows in recordings.values():
        latest_end = {"train": -np.inf, "internal_validation": -np.inf}
        for row in sorted(rows, key=lambda r: (r["start_s"], r["end_s"], r["window_uid"])):
            opposite = "internal_validation" if row["role"] == "train" else "train"
            if latest_end[opposite] > row["start_s"] + 1e-7:
                raise ValueError("global cross-role physiological interval overlap")
            latest_end[row["role"]] = max(latest_end[row["role"]], row["end_s"])
    train_people = {r["subject_uid"] for r in train}
    if not {r["subject_uid"] for r in validation} <= train_people:
        raise ValueError("validation includes unregistered subjects")
    return {"n_train": len(train), "n_validation": len(validation),
            "n_participants": len(train_people), "cross_role_lineage": "pass"}


def _groups(rows: list[dict]) -> dict[str, np.ndarray]:
    result = defaultdict(list)
    for i, row in enumerate(rows):
        result[row["subject_uid"]].append(i)
    return {key: np.asarray(value, dtype=np.int64) for key, value in result.items()}


def _unit(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError("features must be finite nonzero vectors")
    return x / norm


def _empty(n: int, k: int) -> dict[str, np.ndarray]:
    return {"knn_indices": np.full((n, k), -1, dtype=np.int64),
            "uniform_indices": np.full((n, k), -1, dtype=np.int64),
            "knn_weights": np.zeros((n, k), dtype=np.float32),
            "support_weight": np.zeros(n, dtype=np.float32),
            "valid": np.zeros(n, dtype=bool),
            "nearest_distance": np.full(n, np.inf, dtype=np.float32),
            "nearest_time_index": np.full(n, -1, dtype=np.int64)}


def prepare_neighbors(train_features, validation_features, train: list[dict], validation: list[dict],
                      *, mode: str, k: int = 5, block_size: int = 40,
                      exclusion_s: float = 0.0, audit: bool = True) -> dict:
    """Label-free retrieval. All insufficient histories use an explicit fallback.

    Train retrieval excludes the query's complete 40-window time-sorted block
    within a (subject, recording, clock). Chronological retrieval never compares
    different clocks. The distance gate and quartiles use train queries only.
    """
    if mode not in MODES or k < 1 or block_size < 1 or exclusion_s < 0:
        raise ValueError("invalid retrieval configuration")
    if audit:
        audit_metadata(train, validation)
    for x, rows in ((train_features, train), (validation_features, validation)):
        if np.ndim(x) != 2 or x.shape[0] != len(rows) or x.shape[1] < 1:
            raise ValueError("feature/metadata shape mismatch")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("feature spaces must have identical dimensions")
    train_groups, val_groups = _groups(train), _groups(validation)
    output = {"train": _empty(len(train), k), "validation": _empty(len(validation), k)}
    for subject, bank_indices in train_groups.items():
        bank_rows = [train[i] for i in bank_indices]
        bx = _unit(train_features[bank_indices])
        starts = np.array([r["start_s"] for r in bank_rows])
        ends = np.array([r["end_s"] for r in bank_rows])
        axes = np.array([r["time_axis_uid"] for r in bank_rows])
        records = np.array([r["recording_uid"] for r in bank_rows])
        uids = np.array([r["window_uid"] for r in bank_rows])
        blocks = np.full(len(bank_rows), -1, dtype=np.int64)
        block_groups = defaultdict(list)
        for i, row in enumerate(bank_rows):
            block_groups[(row["recording_uid"], row["time_axis_uid"])].append(i)
        next_block = 0
        for key in sorted(block_groups):
            indices = sorted(block_groups[key], key=lambda i: (starts[i], uids[i]))
            for j, i in enumerate(indices):
                blocks[i] = next_block + j // block_size
            next_block += (len(indices) + block_size - 1) // block_size
        stable_order = np.array(sorted(range(len(bank_rows)), key=lambda i:
            (records[i], axes[i], starts[i], uids[i])), dtype=np.int64)
        for role, query_indices, rows, x in (
            ("train", bank_indices, train, train_features),
            ("validation", val_groups.get(subject, np.array([], dtype=np.int64)), validation, validation_features)
        ):
            if not len(query_indices):
                continue
            scores = np.clip(_unit(x[query_indices]) @ bx.T, -1.0, 1.0)
            target = output[role]
            for local_q, global_q in enumerate(query_indices):
                query = rows[global_q]
                comparable = axes == query["time_axis_uid"]
                physical = comparable & (records == query["recording_uid"])
                # Exact query/self, training block, overlapping and near windows
                # are excluded before considering similarities.
                legal = np.ones(len(bank_rows), dtype=bool)
                if role == "train":
                    legal &= blocks != blocks[local_q]
                legal &= ~((uids == query["window_uid"]))
                legal &= ~(physical & (starts < query["end_s"] + exclusion_s - 1e-7)
                           & (ends > query["start_s"] - exclusion_s + 1e-7))
                if mode == "chronological_blocked":
                    legal &= comparable & (ends <= query["start_s"] - exclusion_s + 1e-7)
                donors = np.flatnonzero(legal)
                if len(donors) < k:
                    continue
                chosen = donors[np.lexsort((uids[donors], -scores[local_q, donors]))[:k]]
                weights = np.exp((scores[local_q, chosen] - scores[local_q, chosen].max()) / 0.1)
                weights /= weights.sum()
                target["knn_indices"][global_q] = bank_indices[chosen]
                target["knn_weights"][global_q] = weights
                legal_ordered = stable_order[legal[stable_order]]
                uniform = legal_ordered[np.linspace(0, len(legal_ordered) - 1, k, dtype=int)]
                target["uniform_indices"][global_q] = bank_indices[uniform]
                target["nearest_distance"][global_q] = 1.0 - scores[local_q, chosen[0]]
                target["valid"][global_q] = True
                time_donors = donors[comparable[donors]]
                if len(time_donors):
                    separation = np.maximum(starts[time_donors] - query["end_s"],
                                            query["start_s"] - ends[time_donors])
                    nearest = time_donors[np.lexsort((uids[time_donors], separation))[0]]
                    target["nearest_time_index"][global_q] = bank_indices[nearest]
    train_dist = output["train"]["nearest_distance"][output["train"]["valid"]]
    cuts = dict.fromkeys(("q25", "q50", "q75", "q95"), None)
    if len(train_dist):
        cuts = dict(zip(cuts, [float(v) for v in np.quantile(train_dist, [.25, .5, .75, .95])]))
    q95 = cuts["q95"]
    for role in ("train", "validation"):
        valid = output[role]["valid"]
        distance = output[role]["nearest_distance"]
        if q95 is not None:
            if q95 > 1e-12:
                output[role]["support_weight"][valid] = np.clip(1 - distance[valid] / q95, 0, 1)
            else:
                output[role]["support_weight"][valid] = (distance[valid] <= 1e-12).astype(np.float32)
    output["train_distance_cutpoints"] = cuts
    return output


def diagnostic_predictions(train_bp, train_base, validation_base, neighbors) -> dict[str, np.ndarray]:
    """No validation target argument. D2 uses declared IN-SAMPLE train residuals."""
    bp, fitted, base = (np.asarray(v, dtype=np.float64) for v in (train_bp, train_base, validation_base))
    if bp.shape != fitted.shape or bp.ndim != 2 or bp.shape[1] != 2 or base.shape != (len(neighbors["valid"]), 2):
        raise ValueError("BP/base arrays must have matching (N,2) SBP/DBP shapes")
    if not all(np.isfinite(a).all() for a in (bp, fitted, base)):
        raise ValueError("BP/base arrays contain nonfinite values")
    predictions = {name: base.copy() for name in ("D0", "D1", "D2", "D3")}
    valid = neighbors["valid"]
    index = neighbors["knn_indices"][valid]
    weight = neighbors["knn_weights"][valid, :, None].astype(np.float64)
    if len(weight):
        weight /= weight.sum(axis=1, keepdims=True)
    predictions["D1"][valid] = (weight * bp[index]).sum(axis=1)
    predictions["D2"][valid] += (weight * (bp[index] - fitted[index])).sum(axis=1)
    time = neighbors["nearest_time_index"]
    available = time >= 0
    predictions["D3"][available] = bp[time[available]]
    return predictions


def macro_metrics(target, prediction, subjects) -> dict:
    """Same per-person arithmetic MAE definition as training.participant_macro_metrics."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    subjects = np.asarray(subjects)
    if target.shape != prediction.shape or target.shape != (len(subjects), 2):
        raise ValueError("metric shape mismatch")
    if not len(subjects):
        return {"n_participants": 0, "n_events": 0, "sbp_mae": None, "dbp_mae": None, "mean_mae": None}
    _, inverse = np.unique(subjects, return_inverse=True)
    count = np.bincount(inverse)
    ae = np.abs(prediction - target)
    mae = np.array([np.bincount(inverse, weights=ae[:, i]) / count for i in range(2)]).mean(axis=1)
    return {"n_participants": len(count), "n_events": len(subjects), "sbp_mae": float(mae[0]),
            "dbp_mae": float(mae[1]), "mean_mae": float(mae.mean())}


def score_probe(target, predictions, validation, train_bp, train, neighbors, cutpoints) -> tuple[dict, list[dict]]:
    """Scoring-only target access; groups never influence retrieval or prediction."""
    target = np.asarray(target)
    if target.shape != predictions["D0"].shape or not np.isfinite(target).all():
        raise ValueError("invalid validation targets")
    subjects = np.array([r["subject_uid"] for r in validation])
    source = np.array([r["source"] for r in validation])
    overall = {name: {scope: macro_metrics(target[mask], prediction[mask], subjects[mask])
                     for scope, mask in ((s, np.ones(len(target), bool) if s == "Overall" else source == s) for s in SCOPES)}
               for name, prediction in predictions.items()}
    medians = {s: np.median(np.asarray(train_bp)[indices], axis=0) for s, indices in _groups(train).items()}
    deviation = np.abs(target - np.array([medians[s] for s in subjects]))
    rows = []
    distance = neighbors["nearest_distance"]
    distance_groups = np.full(len(target), "no_eligible_memory", dtype=object)
    if cutpoints["q25"] is not None:
        group = np.searchsorted([cutpoints["q25"], cutpoints["q50"], cutpoints["q75"]], distance, side="right")
        for i in range(4):
            distance_groups[neighbors["valid"] & (group == i)] = f"Q{i+1}"
    for scope in SCOPES:
        source_mask = np.ones(len(target), bool) if scope == "Overall" else source == scope
        for bp_index, bp in enumerate(("SBP", "DBP")):
            bp_groups = np.select([deviation[:, bp_index] <= 10, deviation[:, bp_index] <= 20], ["0-10", "10-20"], default=">20")
            for kind, categories in (("bp_deviation", bp_groups), ("feature_distance", distance_groups)):
                for group_name in sorted(set(categories)):
                    mask = source_mask & (categories == group_name)
                    if not mask.any():
                        continue
                    base = macro_metrics(target[mask], predictions["D0"][mask], subjects[mask])
                    for name in ("D1", "D2", "D3"):
                        metric = macro_metrics(target[mask], predictions[name][mask], subjects[mask])
                        key = bp.lower() + "_mae"
                        rows.append({"method": name, "scope": scope, "bp": bp, "group_type": kind,
                                     "group": str(group_name), "n_participants": metric["n_participants"],
                                     "n_events": metric["n_events"], "base_mae": base[key],
                                     "method_mae": metric[key], "gain_mmhg": base[key] - metric[key]})
    return overall, rows


def stage_evidence(overall, subgroups) -> dict:
    overall_triggers = []
    for method in ("D1", "D2"):
        gain = overall["D0"]["Overall"]["mean_mae"] - overall[method]["Overall"]["mean_mae"]
        if gain >= .02:
            overall_triggers.append({"method": method, "gain_mmhg": gain})
    subgroup_triggers = [r for r in subgroups if r["method"] in {"D1", "D2"}
                         and r["scope"] in {"MIMIC", "VitalDB"} and r["group_type"] == "bp_deviation"
                         and r["n_participants"] >= 100 and r["n_events"] >= 500 and r["gain_mmhg"] >= .15]
    return {"overall_triggers": overall_triggers, "subgroup_triggers": subgroup_triggers,
            "passes_this_mode": bool(overall_triggers or subgroup_triggers),
            "purpose": "diagnostic neural-stage screen, NOT model promotion or independent confirmation"}


def _coverage(result) -> dict:
    return {role: {"n_queries": len(result[role]["valid"]), "n_valid": int(result[role]["valid"].sum()),
                   "n_fallback": int((~result[role]["valid"]).sum()),
                   "n_time_control_fallback": int((result[role]["nearest_time_index"] < 0).sum())}
            for role in ("train", "validation")}


def validate_export_manifest(manifest: dict) -> None:
    """Reject ambiguous provenance before any feature or target file is loaded."""
    if manifest.get("status") != "exported" or manifest.get("protocol_id") != PROTOCOL or manifest.get("split_mode") not in MODES:
        raise ValueError("requires an exported cache under the authorized protocol")
    if manifest.get("heldout_test_accessed") is not False or manifest.get("source_parent_split") != "meta_train":
        raise ValueError("invalid parent split or held-out declaration")
    if set(manifest.get("read_roles", [])) != {"train", "internal_validation"}:
        raise ValueError("cache declares forbidden roles")
    if not isinstance(manifest.get("source_run"), str) or not manifest["source_run"].strip():
        raise ValueError("missing export provenance: source_run")
    for key in ("source_checkpoint_sha256", "source_tree_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(key, ""))):
            raise ValueError(f"missing or invalid export provenance: {key}")
    scaler = manifest.get("target_scaler", {})
    mean, std = np.asarray(scaler.get("mean", [])), np.asarray(scaler.get("std", []))
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("invalid train-only target scaler")


def prepare_cache(cache_dir: Path) -> dict:
    """CLI driver; verify the exact export before creating neighbour artifacts."""
    import pandas as pd

    cache_dir = cache_dir.resolve()
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_export_manifest(manifest)
    needed = [f"{role}_{suffix}" for role in ("train", "validation") for suffix in
              ("features.npy", "bp.npy", "base.npy", "metadata.parquet")]
    for filename in needed:
        item = manifest.get("files", {}).get(filename, {})
        path = cache_dir / filename
        if item.get("path") != filename or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError(f"missing canonical file manifest entry: {filename}")
        if path.is_symlink() or not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"cache file checksum mismatch: {filename}")
    train = pd.read_parquet(cache_dir / "train_metadata.parquet").to_dict("records")
    validation = pd.read_parquet(cache_dir / "validation_metadata.parquet").to_dict("records")
    audit = audit_metadata(train, validation)
    train_groups, validation_groups = _groups(train), _groups(validation)
    expected_sources = {"MIMIC": 1011, "VitalDB": 1040}
    observed_sources = {source: sum(train[indices[0]]["source"] == source for indices in train_groups.values())
                        for source in expected_sources}
    if (len(train_groups) != 2051 or set(train_groups) != set(validation_groups)
            or any(len(indices) != 320 for indices in train_groups.values())
            or any(len(indices) != 40 for indices in validation_groups.values())
            or observed_sources != expected_sources):
        raise ValueError("frozen cohort requires 2051 people, 320 train/40 validation each, MIMIC1011/VitalDB1040")
    audit["cohort_counts"] = observed_sources
    x = np.load(cache_dir / "train_features.npy", mmap_mode="r", allow_pickle=False)
    v = np.load(cache_dir / "validation_features.npy", mmap_mode="r", allow_pickle=False)
    if x.shape[1] != 256 or v.shape[1] != 256:
        raise ValueError("this frozen experiment requires the post-LoRA 256D feature space")
    main = prepare_neighbors(x, v, train, validation, mode=manifest["split_mode"], audit=False)
    sensitivity = prepare_neighbors(x, v, train, validation, mode=manifest["split_mode"], exclusion_s=60, audit=False)
    generated = []
    for prefix, result in (("", main), ("sensitivity60_", sensitivity)):
        for role in ("train", "validation"):
            for name, array in result[role].items():
                filename = f"{prefix}{role}_{name}.npy"
                np.save(cache_dir / filename, array, allow_pickle=False)
                generated.append(filename)
    train_bp = np.load(cache_dir / "train_bp.npy", mmap_mode="r", allow_pickle=False)
    train_base = np.load(cache_dir / "train_base.npy", mmap_mode="r", allow_pickle=False)
    validation_base = np.load(cache_dir / "validation_base.npy", mmap_mode="r", allow_pickle=False)
    predictions = diagnostic_predictions(train_bp, train_base, validation_base, main["validation"])
    sensitivity_predictions = diagnostic_predictions(train_bp, train_base, validation_base, sensitivity["validation"])
    # Labels below this line are for scoring only; retrieval/gating is complete.
    validation_bp = np.load(cache_dir / "validation_bp.npy", mmap_mode="r", allow_pickle=False)
    overall, subgroups = score_probe(validation_bp, predictions, validation, train_bp, train,
                                    main["validation"], main["train_distance_cutpoints"])
    sensitivity_overall, sensitivity_subgroups = score_probe(validation_bp, sensitivity_predictions, validation, train_bp, train,
                                    sensitivity["validation"], sensitivity["train_distance_cutpoints"])
    summary = {"status": "complete", "protocol_id": PROTOCOL, "split_mode": manifest["split_mode"],
               "reference": "D0", "residual_provenance": "in_sample", "heldout_test_accessed": False,
               "coverage": _coverage(main), "overall": overall, "subgroups": subgroups,
               "stage_evidence": stage_evidence(overall, subgroups),
               "train_distance_cutpoints": main["train_distance_cutpoints"], "lineage_audit": audit,
               "sensitivity_60s": {"overall": sensitivity_overall, "subgroups": sensitivity_subgroups,
                                     "coverage": _coverage(sensitivity),
                                     "train_distance_cutpoints": sensitivity["train_distance_cutpoints"]},
               "settings": {"neighbors": 5, "cosine_temperature": .1, "train_excluded_block_size": 40,
                            "main_exclusion_seconds": 0, "sensitivity_exclusion_seconds": 60,
                            "fallback": "unchanged frozen LoRA for every insufficient query",
                            "support_weight": "clip(1-nearest_cosine_distance/train_q95,0,1)",
                            "train_encoder_predictions": "in_sample; NOT OOF or crossfit"}}
    # Reuse the project's formal pooled diagnostic definitions on the server;
    # these imports are deliberately outside NumPy-only unit-test functions.
    from .calbased_metrics import pooled_diagnostics, participant_macro_views
    pooled = []
    for variant, values in (("main", predictions), ("sensitivity60", sensitivity_predictions)):
        for name, prediction in values.items():
            frame = pd.DataFrame(validation)[["subject_uid", "event_id", "source"]].copy()
            for i, bp in enumerate(("sbp", "dbp")):
                frame[f"target_{bp}"] = validation_bp[:, i]
                frame[f"pred_{bp}"] = prediction[:, i]
            formal = participant_macro_views(frame)
            comparison = overall if variant == "main" else sensitivity_overall
            for scope in SCOPES:
                for key in ("sbp_mae", "dbp_mae", "mean_mae"):
                    if not np.isclose(formal[scope][key], comparison[name][scope][key], rtol=0, atol=1e-8):
                        raise ValueError("participant-macro implementation parity failed")
            pooled.append(pooled_diagnostics(frame, f"{variant}_{name}"))
            filename = f"private_{variant}_{name}_predictions.parquet"
            frame.to_parquet(cache_dir / filename, index=False)
            generated.append(filename)
    pd.concat(pooled, ignore_index=True).to_csv(cache_dir / "pooled_diagnostics.csv", index=False)
    pd.DataFrame(subgroups).to_csv(cache_dir / "subgroup_diagnostics.csv", index=False)
    _json(cache_dir / "probe_summary.json", summary)
    generated += ["pooled_diagnostics.csv", "subgroup_diagnostics.csv", "probe_summary.json"]
    for filename in generated:
        manifest["files"][filename] = {"path": filename, "sha256": sha256(cache_dir / filename)}
    manifest["preparation"] = {"status": "complete", "module": __name__, "coverage": _coverage(main),
                               "neighbors": 5, "train_excluded_block_size": 40,
                               "residual_provenance": "in_sample", "not_crossfit": True}
    manifest["status"] = "complete"
    _json(manifest_path, manifest)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = prepare_cache(arguments.cache_dir)
    print(json.dumps({"status": result["status"], "split_mode": result["split_mode"],
                      "coverage": result["coverage"], "stage_evidence": result["stage_evidence"]}, indent=2))
