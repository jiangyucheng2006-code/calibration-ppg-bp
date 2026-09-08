"""Read-only source verification of duplicate official PPG windows.

Reads identity metadata, PPG_F, PPG_Raw and T only. Never reads BP/ABP.
Does not modify the prepared store, split, gates or training configuration.
An optional new JSON report contains private raw identifiers; do not publish it.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


CACHE_HASHES = {
    "Train_Info": "9fee5cf48e975ae565ee7150a1ffdb6c8400ba68c4929be70117cecaa771c151",
    "CalBased_Test_Info": "5127409ca9727239ee04427f2229747c73ec17bc256f8a3906671bc7839f808f",
}
COLUMNS = ["source", "subject_uid", "segment_uid", "segment_row", "record_id",
           "raw_file", "ppg_content_sha256", "inner_role", "inner_fold", "waveform_row"]


def cell(handle, dataset, index):
    ref = dataset[0, index] if dataset.shape[0] == 1 else dataset[index, 0]
    return np.asarray(handle[ref][()]).ravel(order="F")


def verify_selected_membership(rows, info_root, stem):
    path = info_root / f"{stem}_membership.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != CACHE_HASHES[stem]:
        raise ValueError("identity-only cache hash mismatch")
    cache = pd.read_parquet(path)
    selected = set(zip(rows.subject_uid, rows.segment_row))
    keys = list(zip(cache.source.str.replace("PulseDB_MIMIC", "MIMIC", regex=False)
                    .str.replace("PulseDB_Vital", "VitalDB", regex=False)
                    + ":" + cache.raw_subject_id,
                    cache.matlab_subj_segidx.astype(int) - 1))
    positions = [i for i, key in enumerate(keys) if key in selected]
    if len(positions) != len(selected):
        raise ValueError("duplicate rows do not exactly match official membership")
    with h5py.File(info_root / f"{stem}.mat", "r") as handle:
        groups = [handle[n] for n in handle if not n.startswith("#")
                  and isinstance(handle[n], h5py.Group)
                  and "Subj_Name" in handle[n] and "Subj_SegIDX" in handle[n]]
        if len(groups) != 1 or groups[0]["Subj_Name"].size != len(cache):
            raise ValueError("unexpected original Info layout")
        group = groups[0]
        for i in positions:
            name = "".join(chr(int(c)) for c in cell(handle, group["Subj_Name"], i) if c)
            indices = cell(handle, group["Subj_SegIDX"], i)
            if name != cache.iloc[i].official_subject or indices.size != 1:
                raise ValueError("cached name or original index layout mismatch")
            if int(indices[0]) != int(cache.iloc[i].matlab_subj_segidx):
                raise ValueError("original one-based segment index mismatch")
    return len(positions)


def audit(store, info_root):
    frames = []
    for role in ("official_train", "official_test"):
        for path in sorted(store.glob(f"{role}_*.parquet")):
            frame = pd.read_parquet(path, columns=COLUMNS)
            frame["role"] = role
            frame["shard"] = str(path.with_suffix(".npy"))
            frames.append(frame)
    if not frames:
        raise ValueError("no prepared shards")
    frame = pd.concat(frames, ignore_index=True)
    duplicates = frame.loc[frame.ppg_content_sha256.duplicated(keep=False)].copy()
    verified = {}
    for role, stem in (("official_train", "Train_Info"), ("official_test", "CalBased_Test_Info")):
        verified[role] = verify_selected_membership(duplicates.loc[duplicates.role.eq(role)], info_root, stem)
    groups = []
    with ExitStack() as stack:
        raw = {p: stack.enter_context(h5py.File(p, "r")) for p in sorted(duplicates.raw_file.unique())}
        for content_hash, group in duplicates.groupby("ppg_content_sha256", sort=True):
            values = []
            for row in group.itertuples():
                handle = raw[row.raw_file]
                fields = {name: cell(handle, handle["Subj_Wins"][name], int(row.segment_row))
                          for name in ("PPG_F", "PPG_Raw", "T")}
                wave = np.asarray(fields["PPG_F"], dtype="<f4")
                stored = np.load(row.shard, mmap_mode="r")[int(row.waveform_row)]
                if hashlib.sha256(wave.tobytes()).hexdigest() != content_hash or not np.array_equal(wave, stored):
                    raise ValueError("source-array/prepared-shard/hash mismatch")
                values.append(fields)
            first = values[0]
            groups.append({
                "hash": content_hash, "rows": len(group),
                "cross_official": bool(group.role.nunique() > 1),
                "cross_subject": bool(group.subject_uid.nunique() > 1),
                "different_record": bool(group.record_id.nunique() > 1),
                "float64_equal": all(np.array_equal(first["PPG_F"], v["PPG_F"]) for v in values[1:]),
                "raw_equal": all(np.array_equal(first["PPG_Raw"], v["PPG_Raw"]) for v in values[1:]),
                "time_equal": all(np.array_equal(first["T"], v["T"]) for v in values[1:]),
                "train_inner_cross": bool(group.role.eq("official_train").all() and group.inner_role.nunique() > 1),
                "train_fit_cross_fold": bool(group.inner_role.eq("train").all() and group.inner_fold.nunique() > 1),
                "members": group[["source", "subject_uid", "segment_uid", "segment_row", "record_id", "role"]].to_dict("records"),
            })
    cross = [g for g in groups if g["cross_official"]]
    test = duplicates.loc[duplicates.role.eq("official_test") & duplicates.ppg_content_sha256.isin([g["hash"] for g in cross])]
    def counts(subset):
        return {"groups": len(subset), **{k: sum(g[k] for g in subset) for k in (
            "float64_equal", "raw_equal", "time_equal", "different_record", "cross_subject",
            "train_inner_cross", "train_fit_cross_fold")}}
    return {
        "audit_status": "verified_source_duplicates" if groups else "no_duplicates",
        "created_utc": datetime.now(timezone.utc).isoformat(), "prepared_store": str(store),
        "test_bp_accessed": False, "training_resumed": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts_by_role": frame.groupby("role").size().to_dict(),
        "within_role_duplicate_excess": {role: int(g.ppg_content_sha256.duplicated().sum()) for role, g in frame.groupby("role")},
        "original_info_identity_entries_verified": verified,
        "all_groups": counts(groups), "cross_official_groups": counts(cross),
        "affected_test_rows_by_subject": test.groupby("subject_uid").size().to_dict(),
        "affected_test_rows_by_source": test.groupby("source").size().to_dict(),
        "groups_private": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--info-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError("preserve prior audit report")
    result = audit(args.store, args.info_root)
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "groups_private"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
