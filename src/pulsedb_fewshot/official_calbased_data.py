"""Exact official CalBased membership, isolated from the old analogue protocol.

Only the two identity/index fields in official Info files are read. Official
test BP is never loaded or materialized here. The user's 2026-09-08 authorization
starts a separate registered-user benchmark; it does not overwrite old splits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROTOCOL_ID = "pulsedb-official-calbased-v1"
OFFICIAL_COUNTS = {"MIMIC": 1213, "VitalDB": 1293}
INFO_SHA1 = {
    "Train_Info.mat": "c785192a7a860a71e4006e97ac0fd9e7c4b83697",
    "CalBased_Test_Info.mat": "0887f79e4618b19ff0be9ab0190f2a389432a108",
}


def digest_file(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def id_digest(values):
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def parse_subject_name(value):
    """Match the exact first-seven/last-character mapping in official MATLAB."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("unsafe or non-text official subject name")
    if len(value) < 8 or value[-1] not in "01":
        raise ValueError("official subject name lacks source suffix")
    source = "MIMIC" if value[-1] == "0" else "VitalDB"
    return source, value[:7]


def _cell(f, dataset, index):
    flat_shape = dataset.shape
    if len(flat_shape) != 2 or 1 not in flat_shape:
        raise ValueError("expected one-dimensional MATLAB struct field")
    obj = dataset[0, index] if flat_shape[0] == 1 else dataset[index, 0]
    if isinstance(obj, h5py.Reference):
        if not obj:
            raise ValueError("null official identity/index")
        return np.asarray(f[obj][()]).ravel(order="F")
    return np.asarray(obj).ravel(order="F")


def read_official_membership(path, role):
    """Never traverse other Info fields, which can contain target information."""
    if role not in {"official_train", "official_test"}:
        raise ValueError("unknown official role")
    with h5py.File(path, "r") as f:
        # Never enumerate the millions of unrelated MATLAB #refs# names.
        groups = [f[name] for name in f.keys() if not name.startswith("#")
                  and isinstance(f[name], h5py.Group)
                  and all(key in f[name] for key in ("Subj_Name", "Subj_SegIDX"))]
        if len(groups) != 1:
            raise ValueError("cannot identify unique official Info struct")
        g = groups[0]
        if g["Subj_Name"].size != g["Subj_SegIDX"].size:
            raise ValueError("identity and index lengths differ")
        rows = []
        for i in range(g["Subj_Name"].size):
            codes = _cell(f, g["Subj_Name"], i)
            if codes.dtype.kind not in "ui":
                raise ValueError("official name is not a MATLAB character array")
            name = "".join(chr(int(c)) for c in codes if int(c))
            source, file_subject = parse_subject_name(name)
            for value in _cell(f, g["Subj_SegIDX"], i):
                numeric = float(value)
                if not np.isfinite(numeric) or numeric < 1 or numeric != int(numeric):
                    raise ValueError("non-positive/non-integral MATLAB index")
                rows.append((source, file_subject, int(numeric) - 1, name, role))
            if (i + 1) % 100000 == 0:
                print(json.dumps({"phase": "official_index", "role": role, "rows": i + 1}), flush=True)
    return pd.DataFrame(rows, columns=["source", "file_subject", "segment_row", "official_subject_name", "official_role"])


def validate_memberships(train, test, expected_counts=OFFICIAL_COUNTS, train_count=360, test_count=40):
    keys = ["source", "file_subject", "segment_row"]
    for frame, count in ((train, train_count), (test, test_count)):
        if frame.empty or frame.duplicated(keys).any():
            raise ValueError("empty or duplicate official membership")
        per_person = frame.groupby(["source", "file_subject"]).size()
        if not per_person.eq(count).all():
            raise ValueError("official per-participant window count differs")
        counts = frame[["source", "file_subject"]].drop_duplicates().groupby("source").size().to_dict()
        if counts != expected_counts:
            raise ValueError(f"official cohort mismatch: {counts}")
    if set(map(tuple, train[keys].to_numpy())) & set(map(tuple, test[keys].to_numpy())):
        raise ValueError("official train/test share raw segment indices")
    if set(map(tuple, train[keys[:2]].to_numpy())) != set(map(tuple, test[keys[:2]].to_numpy())):
        raise ValueError("official train/test participant sets differ")


def assign_inner_roles(train, seed=20260908, validation_count=40, block_size=40):
    """Deterministic inner split entirely inside official training membership."""
    frame = train.copy()
    frame["inner_role"] = "train"
    frame["inner_fold"] = -1
    for subject, idx in frame.groupby("subject_uid", sort=True).groups.items():
        keys = frame.loc[idx, "segment_uid"]
        ranks = keys.map(lambda uid: hashlib.sha256(f"{seed}|{subject}|{uid}".encode()).hexdigest())
        selected = ranks.sort_values(kind="mergesort").index[:validation_count]
        frame.loc[selected, "inner_role"] = "internal_validation"
        fit = frame.loc[idx].loc[lambda x: x.inner_role.eq("train")]
        fit = fit.sort_values(["record_id", "start_time_s", "segment_uid"], kind="mergesort")
        # Keep each consecutive block in one fold; rotate by participant hash.
        offset = int(hashlib.sha256(str(subject).encode()).hexdigest()[:8], 16) % 3
        folds = (np.arange(len(fit)) // block_size + offset) % 3
        if set(folds) != {0, 1, 2}:
            raise ValueError("too few training blocks for three-fold crossfit")
        frame.loc[fit.index, "inner_fold"] = folds
    return frame


def join_index(membership, metadata):
    keys = ["source", "file_subject", "segment_row"]
    if metadata.duplicated(keys).any():
        raise ValueError("raw index keys are ambiguous")
    joined = membership.merge(metadata, on=keys, how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"official windows missing from raw index: {(joined['_merge'] != 'both').sum()}")
    return joined.drop(columns="_merge")


def interval_conflicts(frame):
    """Count distinct-record-clock physical overlaps, including different IDs."""
    count = 0
    for _, group in frame.groupby(["source", "subject_uid", "record_id"], sort=False):
        ordered = group.sort_values("start_time_s")
        active = []
        for row in ordered.itertuples():
            start, end = float(row.start_time_s), float(row.start_time_s + row.duration_s)
            active = [(b, role) for b, role in active if b > start + 1e-6]
            count += sum(role != row.official_role for _, role in active)
            active.append((end, row.official_role))
    return count


def _read_raw_ppg(f, row):
    g = f["Subj_Wins"]["PPG_F"]
    index = int(row.segment_row)
    if h5py.check_dtype(ref=g.dtype) is not None:
        wave = np.asarray(f[g[0, index] if g.shape[0] == 1 else g[index, 0]][()]).reshape(-1)
    elif g.ndim == 2 and 1250 in g.shape:
        wave = np.asarray(g[:, index] if g.shape[0] == 1250 else g[index, :]).reshape(-1)
    else:
        raise ValueError("unsupported PPG_F storage")
    wave = np.asarray(wave, dtype="<f4")
    if wave.size != 1250 or not np.isfinite(wave).all() or float(wave.std()) <= 1e-8:
        raise ValueError("invalid official PPG window; do not silently drop official members")
    return wave


def materialize_shard(args):
    plan_path, signal_path = map(Path, args)
    frame = pd.read_parquet(plan_path)
    arrays = np.lib.format.open_memmap(signal_path, mode="w+", dtype="<f4", shape=(len(frame), 1250))
    hashes = [None] * len(frame)
    for raw, group in frame.groupby("raw_file", sort=True):
        with h5py.File(raw, "r") as f:
            for row in group.itertuples():
                wave = _read_raw_ppg(f, row)
                arrays[int(row.waveform_row)] = wave
                hashes[row.Index] = hashlib.sha256(wave.tobytes()).hexdigest()
    arrays.flush()
    del arrays
    frame["ppg_content_sha256"] = hashes
    frame.to_parquet(plan_path, index=False)
    return {"file": signal_path.name, "rows": len(frame), "sha256": digest_file(signal_path)}


def prepare(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("output already exists; preserve failed runs and use a new directory")
    args.output.mkdir(parents=True)
    report = {"status": "preparing", "protocol_id": PROTOCOL_ID, "official_test_targets_accessed": False,
              "seed": args.seed, "legacy_models_reused": False, "input_files": {},
              "source_index_sha256": digest_file(args.segment_index),
              "legacy_subject_map_sha256": digest_file(args.legacy_subjects),
              "preparer_sha256": digest_file(Path(__file__)),
              "arguments": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()}}
    save_json(args.output / "manifest.json", report)
    files = [(args.info_root / name, role) for name, role in
             (("Train_Info.mat", "official_train"), ("CalBased_Test_Info.mat", "official_test"))]
    members = {}
    for path, role in files:
        sha1 = digest_file(path, "sha1")
        if sha1 != INFO_SHA1[path.name]:
            raise ValueError(f"official file identity differs: {path.name}")
        report["input_files"][path.name] = {"sha1": sha1, "sha256": digest_file(path), "bytes": path.stat().st_size}
        members[role] = read_official_membership(path, role)
    validate_memberships(members["official_train"], members["official_test"])

    # Load metadata only here. The target predicate below returns only TRAIN
    # rows; Parquet may internally decode mixed-role column pages before filtering.
    available = set(pq.ParquetFile(args.segment_index).schema.names)
    columns = ["source", "subject_uid", "segment_uid", "segment_row", "record_id", "raw_file",
               "start_time_s", "end_time_s", "duration_s", "sample_interval_s", "n_samples"]
    if not set(columns).issubset(available):
        raise ValueError(f"raw index is missing metadata: {set(columns) - available}")
    meta = pd.read_parquet(args.segment_index, columns=columns)
    meta["file_subject"] = meta.raw_file.map(lambda p: Path(p).stem)
    tables = {role: join_index(frame, meta) for role, frame in members.items()}
    del meta
    # Query precisely the allowed official training row IDs before reading BP.
    fit_ids = tables["official_train"].segment_uid.astype(str).tolist()
    targets = pd.read_parquet(args.segment_index, columns=["segment_uid", "sbp", "dbp"],
                              filters=[("segment_uid", "in", fit_ids)])
    if set(targets.segment_uid) != set(fit_ids) or targets.segment_uid.duplicated().any():
        raise ValueError("training target predicate did not return exact official train membership")
    if not np.isfinite(targets[["sbp", "dbp"]].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite training BP")
    tables["official_train"] = tables["official_train"].merge(targets, on="segment_uid", validate="one_to_one")
    all_meta = pd.concat([f.drop(columns=["sbp", "dbp"], errors="ignore") for f in tables.values()], ignore_index=True)
    if interval_conflicts(all_meta):
        raise ValueError("official assignment contains overlapping physical intervals across roles")
    # The raw files are addressed only under the explicitly supplied hot roots.
    for frame in tables.values():
        frame["raw_file"] = [str((args.mimic_root if s == "MIMIC" else args.vital_root) / f"{p}.mat")
                             for s, p in zip(frame.source, frame.file_subject)]
        if not np.allclose(frame.duration_s, 10, atol=1e-4) or not frame.n_samples.eq(1250).all():
            raise ValueError("official window schema does not match 10s/125Hz")
        frame["protocol_id"] = PROTOCOL_ID
    tables["official_train"] = assign_inner_roles(tables["official_train"], args.seed)
    tables["official_test"]["inner_role"] = "official_test"
    tables["official_test"]["inner_fold"] = -1
    if args.legacy_subjects is not None:
        legacy = pd.read_csv(args.legacy_subjects, usecols=["subject_uid", "split"])
        if legacy.subject_uid.duplicated().any():
            raise ValueError("legacy subject map ambiguous")
        exposure = tables["official_train"][["subject_uid", "source"]].drop_duplicates().merge(legacy, on="subject_uid", how="left")
        report["legacy_parent_overlap"] = exposure["split"].fillna("unmapped").value_counts().to_dict()
        report["historical_exposure_scope"] = "subject-parent overlap only; not exhaustive historical-window-use audit"
        report["independent_from_prior_model_development"] = False
        report["new_protocol_may_include_previous_unseen_subjects"] = True
    else:
        raise ValueError("explicit legacy subject overlap accounting is required")
    tasks, plans = [], {role: [] for role in tables}
    for role, frame in tables.items():
        frame = frame.copy()
        frame["_shard"] = frame.subject_uid.map(lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % args.shards)
        for shard, group in frame.groupby("_shard", sort=True):
            group = group.drop(columns="_shard").sort_values(["subject_uid", "segment_row"]).reset_index(drop=True)
            prefix = f"{role}_{shard:03d}"
            group["waveform_file"] = f"{prefix}.npy"
            group["waveform_row"] = np.arange(len(group))
            plan = args.output / f"{prefix}.parquet"
            group.to_parquet(plan, index=False)
            plans[role].append(plan)
            tasks.append((str(plan), str(args.output / f"{prefix}.npy")))
    report["arrays"] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(materialize_shard, task) for task in tasks]):
            record = future.result()
            report["arrays"].append(record)
            print(json.dumps({"phase": "materialize", **record}), flush=True)
    train = pd.concat([pd.read_parquet(p) for p in plans["official_train"]], ignore_index=True)
    test = pd.concat([pd.read_parquet(p) for p in plans["official_test"]], ignore_index=True)
    cross_duplicates = set(train.ppg_content_sha256) & set(test.ppg_content_sha256)
    report["cross_role_exact_ppg_hashes"] = len(cross_duplicates)
    if cross_duplicates:
        save_json(args.output / "manifest.json", report)
        raise ValueError("exact official membership includes duplicate PPG across train/test; requires an explicit documented decision")
    # Keep identical train windows in one inner role/fold rather than training
    # on exact copies of internal validation or OOF queries.
    combined_groups = train.groupby("ppg_content_sha256").agg(roles=("inner_role", "nunique"), folds=("inner_fold", "nunique"))
    if (combined_groups.roles > 1).any() or (combined_groups.folds > 1).any():
        raise ValueError("exact PPG copies cross inner-validation/OOF folds; split must be revised without test outcomes")
    train.to_parquet(args.output / "train_manifest.parquet", index=False)
    test.to_parquet(args.output / "test_inputs.parquet", index=False)
    report.update(status="ready", train_rows=len(train), test_rows=len(test),
                  subjects=train.subject_uid.nunique(), source_counts=train.groupby("source").subject_uid.nunique().to_dict(),
                  official_train_windows_per_subject=360, official_test_windows_per_subject=40,
                  inner_train_windows_per_subject=320, inner_validation_windows_per_subject=40,
                  train_manifest_sha256=digest_file(args.output / "train_manifest.parquet"),
                  test_inputs_sha256=digest_file(args.output / "test_inputs.parquet"),
                  train_membership_sha256=id_digest(train.segment_uid),
                  test_membership_sha256=id_digest(test.segment_uid),
                  elapsed_seconds=time.monotonic() - started)
    save_json(args.output / "manifest.json", report)
    print(json.dumps({"status": "ready", "subjects": int(train.subject_uid.nunique()), "train_rows": len(train), "test_rows": len(test)}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--info-root", type=Path, required=True)
    p.add_argument("--segment-index", type=Path, required=True)
    p.add_argument("--legacy-subjects", type=Path, required=True)
    p.add_argument("--mimic-root", type=Path, required=True)
    p.add_argument("--vital-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--shards", type=int, default=32)
    args = p.parse_args()
    if args.workers < 1 or args.shards < 1:
        p.error("positive workers/shards required")
    try:
        prepare(args)
    except Exception as exc:
        receipt = args.output / "manifest.json"
        if receipt.exists():
            report = json.loads(receipt.read_text(encoding="utf-8"))
            if report.get("status") == "preparing":
                report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                save_json(receipt, report)
        raise


if __name__ == "__main__":
    main()
