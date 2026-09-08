"""Recover complete failed shards under the authorized exact-membership policy.

Makes a new store, never changes or deletes old shards. All official rows and
internal assignments are preserved. Does not open any test BP/ABP fields.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from .official_calbased_data import (PROTOCOL_ID, digest_file, save_json, id_digest,
                                    read_verified_membership_cache, validate_memberships, INFO_SHA1)
from .official_content_policy import (POLICY, AUDIT_SHA256, CLAIM_LIMIT, approved_groups,
                                     assert_allowed_duplicates)

FAILED_MANIFEST_SHA256 = "fff78262ebba75600381099b521cf37097fc127d458fd35a594d7581efc352da"


def check_array(array_path, frame, expected):
    if digest_file(array_path) != expected["sha256"]:
        raise ValueError("completed shard checksum mismatch")
    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(frame), 1250) or array.dtype != np.dtype("<f4") or len(frame) != expected["rows"]:
        raise ValueError("shard shape/count mismatch")
    if sorted(frame.waveform_row.tolist()) != list(range(len(frame))):
        raise ValueError("shard row identity is not a bijection")
    for row in frame.itertuples():
        wave = array[int(row.waveform_row)]
        if hashlib.sha256(wave.tobytes()).hexdigest() != row.ppg_content_sha256:
            raise ValueError("per-window PPG content changed")


def recover(args):
    started = time.monotonic()
    source, output = args.failed_store.resolve(), args.output.resolve()
    if output.exists() or source == output:
        raise FileExistsError("recovery requires a new store, preserving the old failure")
    if digest_file(source / "manifest.json") != FAILED_MANIFEST_SHA256:
        raise ValueError("not the source-audited completed failed materialization")
    old = json.loads((source / "manifest.json").read_text())
    if old.get("status") != "failed" or old.get("protocol_id") != PROTOCOL_ID or len(old.get("arrays", [])) != 64:
        raise ValueError("wrong failed store or incomplete arrays")
    if args.content_policy != POLICY:
        raise ValueError("explicit user-authorized content policy required")
    if digest_file(args.duplicate_audit) != AUDIT_SHA256:
        raise ValueError("source duplicate audit checksum mismatch")
    approved_groups(str(args.duplicate_audit.resolve()))
    output.mkdir(parents=True)
    report = dict(old)
    report.pop("error", None)
    report.update(status="recovering", content_policy=POLICY, content_claim_limit=CLAIM_LIMIT,
                  duplicate_audit_sha256=AUDIT_SHA256, recovery_source=str(source),
                  recovery_source_manifest_sha256=FAILED_MANIFEST_SHA256,
                  recovery_code_sha256=digest_file(Path(__file__)),
                  recovery_started_utc=datetime.now(timezone.utc).isoformat(),
                  official_test_targets_accessed=False, rows_removed=0,
                  official_membership_changed=False, inner_assignments_changed=False,
                  recovery_output=str(output))
    save_json(output / "manifest.json", report)
    try:
        audit_path = output / "official_duplicate_audit.json"
        shutil.copy2(args.duplicate_audit, audit_path)
        if digest_file(audit_path) != AUDIT_SHA256:
            raise ValueError("copied audit mismatch")
        arrays = {r["file"]: r for r in old["arrays"]}
        tables = {}
        for role in ("official_train", "official_test"):
            frames = []
            plans = sorted(source.glob(f"{role}_*.parquet"))
            if len(plans) != 32:
                raise ValueError("expected 32 complete shards per official role")
            for plan in plans:
                frame = pd.read_parquet(plan)
                if role == "official_test" and any(c in {"sbp", "dbp", "bp"} or str(c).lower().startswith(("abp", "target_", "segsbp", "segdbp")) for c in frame):
                    raise ValueError("test input plan contains forbidden target fields")
                filename = plan.with_suffix(".npy").name
                if not frame.waveform_file.eq(filename).all() or filename not in arrays:
                    raise ValueError("unexpected waveform path")
                source_array = source / filename
                check_array(source_array, frame, arrays[filename])
                shutil.copy2(source_array, output / filename)
                if digest_file(output / filename) != arrays[filename]["sha256"]:
                    raise ValueError("recovered signal copy checksum mismatch")
                frame["official_content_policy"] = POLICY
                frame["official_duplicate_audit_path"] = str(audit_path)
                frame.to_parquet(output / plan.name, index=False)
                frames.append(frame)
                print(json.dumps({"phase": "verified_recovered_shard", "file": filename, "rows": len(frame)}), flush=True)
            tables[role] = pd.concat(frames, ignore_index=True)
        train, test = tables["official_train"], tables["official_test"]
        validate_memberships(train, test)
        keys = ["source", "file_subject", "segment_row"]
        for stem, role in (("Train_Info", "official_train"), ("CalBased_Test_Info", "official_test")):
            info = args.info_root / f"{stem}.mat"
            if digest_file(info, "sha1") != INFO_SHA1[info.name]:
                raise ValueError("original official Info checksum mismatch")
            reference = read_verified_membership_cache(info.with_name(f"{stem}_membership.parquet"), role)
            if set(map(tuple, tables[role][keys].to_numpy())) != set(map(tuple, reference[keys].to_numpy())):
                raise ValueError("recovered rows differ from exact official membership")
        retained = assert_allowed_duplicates(pd.concat([train, test], ignore_index=True))
        if retained != 154:
            raise ValueError("full recovered duplicate inventory differs from source audit")
        train.to_parquet(output / "train_manifest.parquet", index=False)
        test.to_parquet(output / "test_inputs.parquet", index=False)
        report.update(status="ready", train_rows=len(train), test_rows=len(test),
                      subjects=int(train.subject_uid.nunique()),
                      source_counts=train.groupby("source").subject_uid.nunique().to_dict(),
                      official_train_windows_per_subject=360, official_test_windows_per_subject=40,
                      inner_train_windows_per_subject=320, inner_validation_windows_per_subject=40,
                      retained_duplicate_groups=retained, cross_role_exact_ppg_hashes=35,
                      within_role_duplicate_ppg_rows={"official_train": 117, "official_test": 2},
                      train_manifest_sha256=digest_file(output / "train_manifest.parquet"),
                      test_inputs_sha256=digest_file(output / "test_inputs.parquet"),
                      train_membership_sha256=id_digest(train.segment_uid),
                      test_membership_sha256=id_digest(test.segment_uid),
                      recovery_completed_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - started)
        save_json(output / "manifest.json", report)
        print("OFFICIAL_EXACT_MEMBERSHIP_RECOVERY=pass", flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_json(output / "manifest.json", report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failed-store", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--info-root", type=Path, required=True)
    p.add_argument("--duplicate-audit", type=Path, required=True)
    p.add_argument("--content-policy", choices=[POLICY], required=True)
    recover(p.parse_args())


if __name__ == "__main__":
    main()
