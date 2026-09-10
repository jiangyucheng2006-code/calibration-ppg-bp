"""Full original cohort, label-isolated materialization and enrollment plans.

No official 400-window membership intersection and no subject/window cap.
Only the saved source-qualified outer assignments determine participant roles.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .legacy_enrollment_protocol import (
    FULL_PROTOCOL, FULL_CONFIG, LEGACY_SPLIT_SHA256, METHODS, KEYS, PARTITIONS,
    build_assignment, reject_labels,
)
from .official_calbased_train import sha256, save_json, event_ids_sha256, REQUIRED

STORE_PROTOCOL = "pulsedb-full-input-pool-v1"
EXPECTED_COUNTS = {"meta_train": 3752, "meta_validation": 805, "meta_test": 804}
INPUT_COLUMNS = ["subject_uid", "segment_uid", "source", "subject_id", "record_id",
                 "start_time_s", "end_time_s", "duration_s", "sample_interval_s",
                 "n_samples", "raw_file", "raw_file_sha256", "ppg_field",
                 "ppg_storage_mode", "ppg_reference_index", "segment_schema_valid",
                 "segment_exclusion_reasons"]
OUTPUT_COLUMNS = sorted((REQUIRED - {"sbp", "dbp", "inner_role"}) | {"full_index_group"})


def original_assignment(path, synthetic=False):
    if not synthetic and sha256(path) != LEGACY_SPLIT_SHA256:
        raise ValueError("original complete subject split checksum changed")
    f = pd.read_csv(path)
    if f.subject_uid.duplicated().any() or f[["subject_uid", "source", "split"]].isna().any().any():
        raise ValueError("ambiguous original subject identity")
    if set(f.split) != set(EXPECTED_COUNTS):
        raise ValueError("original three outer roles required")
    if not synthetic and (len(f) != 5361 or f.split.value_counts().to_dict() != EXPECTED_COUNTS):
        raise ValueError("full original cohort was restricted")
    return f


def _source_path(row, raw_root):
    directory = {"MIMIC": "PulseDB_MIMIC", "VitalDB": "PulseDB_Vital"}[row.source]
    name = Path(row.raw_file).name
    if not name.endswith(".mat"):
        raise ValueError("invalid raw file name")
    root = Path(raw_root).resolve()
    path = (root / directory / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing work-area raw file: {path}")
    return path


def materialize_shard(task):
    import h5py
    import pyarrow.parquet as pq
    index, raw_root, output, shard, shards, synthetic = task
    root = Path(output)
    pf = pq.ParquetFile(index)
    groups = list(range(shard, pf.num_row_groups, shards))
    total = sum(pf.metadata.row_group(i).num_rows for i in groups)
    wave_name = f"full_ppg_{shard:03d}.npy"
    array = np.lib.format.open_memmap(root / wave_name, mode="w+", dtype=np.float32,
                                      shape=(total, 1250))
    frames, excluded, people, raw_receipts = [], [], set(), []
    offset, started = 0, time.monotonic()
    for count, group_id in enumerate(groups, 1):
        f = pf.read_row_group(group_id, columns=INPUT_COLUMNS).to_pandas()
        reject_labels(f)
        if f.subject_uid.nunique() != 1 or f.raw_file.nunique() != 1:
            raise ValueError("full source index must have one file/subject per row group")
        people.update(f.subject_uid)
        source = _source_path(f.iloc[0], raw_root)
        expected = set(f.raw_file_sha256)
        if len(expected) != 1 or sha256(source) != next(iter(expected)):
            raise ValueError("raw work copy changed from the full index")
        valid = np.ones(len(f), dtype=bool)
        hashes = [""] * len(f)
        with h5py.File(source, "r") as h:
            dataset = h["Subj_Wins"]["PPG_F"]
            refs = dataset[()].ravel(order="F") if h5py.check_dtype(ref=dataset.dtype) is not None else None
            direct = dataset[()] if refs is None else None
            for i, row in enumerate(f.itertuples(index=False)):
                reason = ""
                if not bool(row.segment_schema_valid):
                    reason = "original_schema_invalid:" + str(row.segment_exclusion_reasons)
                elif (row.n_samples != 1250
                      or not np.isclose(row.duration_s, 10., rtol=0, atol=1e-4)
                      or not np.isclose(row.sample_interval_s, .008, rtol=0, atol=1e-6)
                      or not np.isclose(row.end_time_s - row.start_time_s + row.sample_interval_s,
                                        row.duration_s, rtol=0, atol=1e-4)):
                    reason = "unsupported_window_length"
                elif row.ppg_field != "PPG_F":
                    raise ValueError("unexpected signal field")
                else:
                    r = int(row.ppg_reference_index)
                    if refs is not None and row.ppg_storage_mode == "references":
                        wave = np.asarray(h[refs[r]][()], np.float32).ravel(order="F")
                    elif refs is None and row.ppg_storage_mode == "direct_single_window" and r == 0:
                        wave = np.asarray(direct, np.float32).ravel(order="F")
                    else:
                        raise ValueError("PPG storage/reference index mismatch")
                    if wave.shape != (1250,) or not np.isfinite(wave).all() or float(wave.std()) <= 1e-8:
                        reason = "invalid_float32_ppg"
                    else:
                        array[offset + i] = wave
                        hashes[i] = hashlib.sha256(wave.tobytes(order="C")).hexdigest()
                if reason:
                    valid[i] = False
                    excluded.append({"subject_uid": row.subject_uid, "segment_uid": row.segment_uid,
                                     "source": row.source, "reason": reason})
        f["waveform_file"] = wave_name
        f["waveform_row"] = np.arange(offset, offset + len(f), dtype=np.int64)
        f["ppg_content_sha256"] = hashes
        f["full_index_group"] = group_id
        frames.append(f.loc[valid, OUTPUT_COLUMNS])
        offset += len(f)
        raw_receipts.append({"subject_uid": f.subject_uid.iloc[0], "file": str(source),
                             "sha256": next(iter(expected)), "rows": len(f)})
        if count % 10 == 0 or count == len(groups):
            save_json(root / f"progress_{shard:03d}.json", {
                "shard": shard, "completed_files": count, "total_files": len(groups),
                "scanned_windows": offset, "elapsed_seconds": time.monotonic() - started})
    array.flush()
    del array
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(root / f"inputs_{shard:03d}.parquet", index=False)
    pd.DataFrame(excluded, columns=["subject_uid", "segment_uid", "source", "reason"]).to_parquet(
        root / f"excluded_{shard:03d}.parquet", index=False)
    receipt = {"shard": shard, "original_subjects": sorted(people), "scanned_windows": offset,
               "valid_windows": len(frame), "invalid_windows": len(excluded), "raw_files": raw_receipts,
               "files": {name: sha256(root / name) for name in
                         (wave_name, f"inputs_{shard:03d}.parquet", f"excluded_{shard:03d}.parquet")}}
    save_json(root / f"receipt_{shard:03d}.json", receipt)
    return receipt


def materialize(args):
    import pyarrow.parquet as pq
    old = original_assignment(args.legacy_split, args.synthetic)
    pf = pq.ParquetFile(args.full_index)
    if not args.synthetic and pf.num_row_groups != 5361:
        raise ValueError("full source index row-group count changed")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = {"protocol_id": STORE_PROTOCOL, "status": "materializing", "synthetic": args.synthetic,
            "full_index": str(args.full_index.resolve()), "full_index_sha256": sha256(args.full_index),
            "legacy_split_sha256": sha256(args.legacy_split), "original_subjects": len(old),
            "original_windows": pf.metadata.num_rows, "window_cap": None, "subject_cap": None,
            "label_values_read": False, "shards": args.shards}
    save_json(root / "materialization.json", plan)
    jobs = [(str(args.full_index), str(args.raw_root), str(root), i, args.shards, args.synthetic)
            for i in range(args.shards)]
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(materialize_shard, j) for j in jobs]
        for future in as_completed(futures):
            receipts.append(future.result())
            print(json.dumps({"completed_shards": len(receipts), "total_shards": args.shards,
                              "windows_scanned": sum(r["scanned_windows"] for r in receipts)}), flush=True)
    subjects = [s for r in receipts for s in r["original_subjects"]]
    if len(subjects) != len(set(subjects)) or set(subjects) != set(old.subject_uid):
        raise ValueError("not every original participant was scanned exactly once")
    if sum(r["scanned_windows"] for r in receipts) != pf.metadata.num_rows:
        raise ValueError("full source window accounting failed")
    plan.update(status="ready", original_subject_ids_sha256=event_ids_sha256(subjects),
                valid_windows=sum(r["valid_windows"] for r in receipts),
                invalid_windows=sum(r["invalid_windows"] for r in receipts),
                files={name: digest for r in receipts for name, digest in r["files"].items()})
    save_json(root / "manifest.json", plan)
    return {k: v for k, v in plan.items() if k != "files"}


def read_grouped_targets(index, keys):
    """Per-row-group exact predicates; never read all query targets then mask."""
    import pyarrow.dataset as ds
    if keys.segment_uid.duplicated().any() or "full_index_group" not in keys:
        raise ValueError("exact unique target keys with source row groups required")
    fragment = next(ds.dataset(index, format="parquet").get_fragments())
    frames = []
    columns = KEYS + ["sbp", "dbp"]
    for group_id, g in keys.groupby("full_index_group", sort=True):
        table = fragment.subset(row_group_ids=[int(group_id)]).to_table(
            columns=columns, filter=ds.field("segment_uid").isin(g.segment_uid.tolist()))
        f = table.to_pandas()
        if len(f) != len(g) or f.segment_uid.duplicated().any():
            raise ValueError("target predicate returned missing/extra rows")
        check = g[KEYS].merge(f, on=KEYS, how="left", validate="one_to_one", indicator=True)
        if not check._merge.eq("both").all() or not np.isfinite(f[["sbp", "dbp"]]).all().all():
            raise ValueError("target identity or finite-reference check failed")
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def prepare(args):
    old = original_assignment(args.legacy_split, args.synthetic)
    store = args.store_root.resolve()
    contract = json.loads((store / "manifest.json").read_text())
    if (contract.get("protocol_id") != STORE_PROTOCOL or contract.get("status") != "ready"
            or bool(contract.get("synthetic")) != args.synthetic
            or contract.get("original_subjects") != len(old)
            or contract.get("original_subject_ids_sha256") != event_ids_sha256(old.subject_uid)
            or contract.get("legacy_split_sha256") != sha256(args.legacy_split)
            or contract.get("full_index_sha256") != sha256(args.full_index)
            or contract.get("subject_cap") is not None or contract.get("window_cap") is not None):
        raise ValueError("unapproved/restricted full waveform pool")
    frames, input_exclusions = [], []
    for name, digest in sorted(contract["files"].items()):
        path = (store / name).resolve()
        if not path.is_relative_to(store) or sha256(path) != digest:
            raise ValueError("full waveform or manifest provenance changed")
        if name.startswith("inputs_"):
            f = pd.read_parquet(path)
            reject_labels(f)
            frames.append(f)
        elif name.startswith("excluded_"):
            input_exclusions.append(pd.read_parquet(path))
    metadata = pd.concat(frames, ignore_index=True)
    if len(metadata) != contract["valid_windows"]:
        raise ValueError("full waveform coverage changed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    f, audit = build_assignment(metadata, old, full_cohort=True)
    # Preserve a private row-level accounting of every unused source window.
    not_used = metadata.loc[~metadata.segment_uid.isin(f.segment_uid), KEYS].copy()
    not_used["reason"] = "same_outer_duplicate_content"
    not_used.loc[not_used.subject_uid.isin(audit["quarantined_subjects"]), "reason"] = "cross_outer_content_or_interval_component"
    for e in audit["insufficient_history_exclusions"]:
        not_used.loc[not_used.subject_uid.eq(e["subject_uid"]), "reason"] = e["reason"]
    exclusions = pd.concat(input_exclusions + [not_used], ignore_index=True)
    exclusions.to_parquet(output / "window_exclusions.parquet", index=False)
    people = old[["subject_uid", "source", "split"]].copy()
    people["original_windows"] = people.subject_uid.map(pd.concat([metadata[KEYS]] + [x[KEYS] for x in input_exclusions]).groupby("subject_uid").size()).fillna(0).astype(int)
    people["eligible_windows"] = people.subject_uid.map(f.groupby("subject_uid").size()).fillna(0).astype(int)
    people["status"] = np.where(people.eligible_windows.gt(0), "included", "excluded")
    reasons = exclusions.groupby("subject_uid").reason.agg(lambda values: ";".join(sorted(set(values))))
    people["exclusion_reasons"] = people.subject_uid.map(reasons).fillna("")
    if (len(f) + len(exclusions) != contract["original_windows"] or people.original_windows.eq(0).any()
            or set(people.subject_uid) != set(old.subject_uid)):
        raise ValueError("all original people/windows must be accounted for")
    people.to_csv(output / "full_cohort_accounting.csv", index=False)
    f[["subject_uid", "source", "split"]].drop_duplicates().to_csv(output / "subject_assignments.csv", index=False)
    f.to_parquet(output / "assignment_inputs.parquet", index=False)
    budget = f.groupby(["subject_uid", "source", "split", "personal_role", "inner_role"]).size().rename("windows").reset_index()
    budget.to_parquet(output / "personal_budgets.parquet", index=False)
    parts = {"population_train": f.loc[f.split.eq("meta_train")].copy()}
    for name, role in (("validation", "meta_validation"), ("test", "meta_test")):
        parts[f"{name}_registration"] = f.loc[f.split.eq(role) & f.personal_role.eq("registration")].copy()
        parts[f"{name}_inputs"] = f.loc[f.split.eq(role) & f.personal_role.eq("query")].copy()
    for name, part in parts.items():
        if part.empty:
            raise ValueError("empty outer partition")
        if not name.endswith("inputs"):
            part = part.merge(read_grouped_targets(args.full_index, part), on=KEYS, validate="one_to_one")
        else:
            reject_labels(part)
        part.to_parquet(output / f"{name}.parquet", index=False)
        audit[name] = {"rows": len(part), "subjects": part.subject_uid.nunique(),
                       "segment_ids_sha256": event_ids_sha256(part.segment_uid),
                       "subject_ids_sha256": event_ids_sha256(part.subject_uid.unique())}
    read_grouped_targets(args.full_index, parts["validation_inputs"]).to_parquet(output / "validation_targets.parquet", index=False)
    audit.update(original_subjects=len(old), original_windows=contract["original_windows"],
                 original_counts=old.groupby(["split", "source"]).size().unstack(fill_value=0).to_dict("index"),
                 excluded_people=int(people.status.eq("excluded").sum()),
                 excluded_windows=len(exclusions), full_source_scanned=True,
                 official_membership_intersection=False, window_cap=None, subject_cap=None)
    files = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    plan = {"protocol_id": FULL_PROTOCOL, "status": "ready", "synthetic": args.synthetic,
            "config": FULL_CONFIG, "source_root": str(store), "full_index": str(args.full_index.resolve()),
            "full_index_sha256": sha256(args.full_index), "source_manifest_sha256": sha256(store / "manifest.json"),
            "legacy_split_path": str(args.legacy_split.resolve()), "legacy_split_sha256": sha256(args.legacy_split),
            "original_subjects": len(old), "all_original_subjects_accounted_for": True,
            "files": files, "audit": audit, "methods": METHODS,
            "validation_subjects": sorted(parts["validation_inputs"].subject_uid.unique()),
            "test_subjects": sorted(parts["test_inputs"].subject_uid.unique()),
            "test_query_labels_read_during_prepare": False,
            "claims": "Exploratory original full-cohort person-disjoint enrollment, random within-person registration/query; not official CalBased, K-shot, future-date or untouched external confirmation."}
    save_json(output / "plan.json", plan)
    save_json(output / "plan_digest.json", {"sha256": sha256(output / "plan.json")})
    print(json.dumps(audit, indent=2), flush=True)
    return plan


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("materialize", "prepare"), required=True)
    p.add_argument("--full-index", type=Path, required=True)
    p.add_argument("--legacy-split", type=Path, required=True)
    p.add_argument("--raw-root", type=Path)
    p.add_argument("--store-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--shards", type=int, default=32)
    p.add_argument("--synthetic", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    result = materialize(args) if args.stage == "materialize" else prepare(args)
    print(json.dumps({"stage": args.stage, "status": result["status"]}), flush=True)
