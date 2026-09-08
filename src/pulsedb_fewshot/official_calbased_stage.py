"""Stage only the pinned official CalBased participants from NAS to hot storage.

Only identity and raw-file checksum columns are read. No BP labels are loaded.
Existing targets are checksum verified and never overwritten on disagreement.
Run as a CPU-only preparation step; the raw NAS master is always read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time


TRAIN_MEMBERSHIP_SHA256 = "9fee5cf48e975ae565ee7150a1ffdb6c8400ba68c4929be70117cecaa771c151"
SOURCE_FOLDERS = {"MIMIC": "PulseDB_MIMIC", "VitalDB": "PulseDB_Vital"}
EXPECTED_SUBJECTS = {"MIMIC": 1213, "VitalDB": 1293}
BUFFER_BYTES = 8 * 1024 * 1024


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    """Crash-safe receipt update, independent of third-party dependencies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only the exact temporary file created above.


def validate_subjects(subjects, expected_counts=EXPECTED_SUBJECTS):
    counts = {}
    clean = set()
    for source, subject in subjects:
        if source not in SOURCE_FOLDERS or not isinstance(subject, str) or not re.fullmatch(r"[a-zA-Z][0-9]{6}", subject):
            raise ValueError("unsafe or invalid official participant identity")
        if (source, subject) in clean:
            raise ValueError("duplicate staging participant")
        clean.add((source, subject))
        counts[source] = counts.get(source, 0) + 1
    if counts != expected_counts:
        raise ValueError(f"official participant counts differ: {counts}")
    return sorted(clean)


def load_subjects(membership):
    if digest_file(membership) != TRAIN_MEMBERSHIP_SHA256:
        raise ValueError("official Train_Info identity-cache checksum differs")
    import pandas as pd
    frame = pd.read_parquet(membership, columns=["source", "raw_subject_id"])
    counts = frame.groupby(["source", "raw_subject_id"]).size()
    if len(frame) != 902160 or not counts.eq(360).all():
        raise ValueError("official identity cache is not 360 training windows per participant")
    return validate_subjects(frame.drop_duplicates().itertuples(index=False, name=None))


def load_expected_hashes(segment_index, subjects):
    """Stream three metadata-only columns, not the 5M-row BP-bearing table."""
    import pyarrow.parquet as pq
    wanted = {f"{source}:{subject}": (source, subject) for source, subject in subjects}
    hashes = {}
    for batch in pq.ParquetFile(segment_index).iter_batches(
            batch_size=65536, columns=["source", "subject_uid", "raw_file_sha256"]):
        frame = batch.to_pandas()
        frame = frame.loc[frame.subject_uid.isin(wanted)].drop_duplicates()
        for source, subject_uid, value in frame.itertuples(index=False, name=None):
            key = wanted[subject_uid]
            if source != key[0] or not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value):
                raise ValueError("invalid raw-file checksum provenance")
            value = value.lower()
            if key in hashes and hashes[key] != value:
                raise ValueError("one participant maps to multiple raw-file checksums")
            hashes[key] = value
    if set(hashes) != set(subjects):
        raise ValueError("official raw-file checksums incomplete")
    return hashes


def build_plan(subjects, hashes, source_root, target_root):
    source_root, target_root = Path(source_root).resolve(), Path(target_root).resolve()
    if source_root == target_root or source_root in target_root.parents or target_root in source_root.parents:
        raise ValueError("source and target roots must be distinct non-nested directories")
    records = []
    for source, subject in subjects:
        relative = Path(SOURCE_FOLDERS[source]) / f"{subject}.mat"
        source_path, target_path = source_root / relative, target_root / relative
        if source_path.resolve().parent != source_root / SOURCE_FOLDERS[source]:
            raise ValueError("source participant path escapes the explicit source directory")
        if target_path.resolve().parent != target_root / SOURCE_FOLDERS[source]:
            raise ValueError("target participant path escapes the explicit target directory")
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(f"source file missing or symbolic: {source_path}")
        if target_path.exists() and (not target_path.is_file() or target_path.is_symlink()):
            raise ValueError(f"target is not an ordinary file: {target_path}")
        records.append({"source": source, "subject": subject, "source_path": str(source_path),
                        "target_path": str(target_path), "bytes": source_path.stat().st_size,
                        "sha256": hashes[source, subject], "existing_target": target_path.exists()})
    return records


def check_free_space(records, target_root, reserve_bytes):
    if reserve_bytes < 0:
        raise ValueError("reserve cannot be negative")
    target = Path(target_root).resolve()
    while not target.exists():
        target = target.parent
    needed = sum(row["bytes"] for row in records if not row["existing_target"])
    free = shutil.disk_usage(target).free
    if free < needed + reserve_bytes:
        raise OSError(f"insufficient hot space: need {needed + reserve_bytes}, available {free}")
    return {"copy_bytes": needed, "free_bytes": free, "reserve_bytes": reserve_bytes}


def copy_verified_file(source, target, expected_sha256):
    """Copy and hash once, read-back verify, then atomically install the target.

    A pre-existing target is never replaced. Failures remove only this call's
    uniquely created partial file; the existing raw master and target survive.
    On rerun, complete targets are re-hashed and skipped without recopying.
    """
    source, target = Path(source), Path(target)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("invalid expected checksum")
    if source.resolve() == target.resolve() or source.is_symlink() or target.is_symlink():
        raise ValueError("source and target must be distinct ordinary files")
    source_stat = source.stat()
    if target.exists():
        if target.stat().st_size != source_stat.st_size or digest_file(target) != expected_sha256:
            raise ValueError(f"existing hot target differs; preserved without overwrite: {target}")
        return "verified_existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, partial = tempfile.mkstemp(prefix=target.name + ".official-stage.", suffix=".partial", dir=target.parent)
    try:
        observed = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as writer, source.open("rb") as reader:
            for chunk in iter(lambda: reader.read(BUFFER_BYTES), b""):
                writer.write(chunk)
                observed.update(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        now = source.stat()
        if (now.st_size, now.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise ValueError("NAS source changed while copying")
        if observed.hexdigest() != expected_sha256 or digest_file(partial) != expected_sha256:
            raise ValueError("copied raw-file checksum differs from full-cohort audit")
        if target.exists():
            raise FileExistsError("target appeared during staging; refusing overwrite")
        # Hard-link installation is atomic and fails if another writer won the
        # target race; the temporary lives on the same hot filesystem.
        os.link(partial, target)
        return "copied_verified"
    finally:
        if os.path.exists(partial):
            os.unlink(partial)  # Exact mkstemp path, never any user raw file.


def stage(args):
    started = time.monotonic()
    source_root, target_root = args.source_root.resolve(), args.target_root.resolve()
    receipt = {"status": "planning", "started_utc": datetime.now(timezone.utc).isoformat(),
               "official_test_targets_accessed": False, "membership_sha256": TRAIN_MEMBERSHIP_SHA256,
               "source_root": str(source_root), "target_root": str(target_root), "files": []}
    try:
        subjects = load_subjects(args.membership)
        receipt["segment_index_sha256"] = digest_file(args.segment_index)
        hashes = load_expected_hashes(args.segment_index, subjects)
        records = build_plan(subjects, hashes, source_root, target_root)
        receipt.update(required_files=len(records), required_bytes=sum(row["bytes"] for row in records),
                       source_counts=EXPECTED_SUBJECTS,
                       space=check_free_space(records, target_root, int(args.reserve_gib * 2**30)))
        receipt["status"] = "planned" if args.plan_only else "copying"
        save_json(args.receipt, receipt)
        print(json.dumps({key: value for key, value in receipt.items() if key != "files"}), flush=True)
        if args.plan_only:
            return receipt
        for position, row in enumerate(records, 1):
            receipt["active_file"] = row["target_path"]
            save_json(args.receipt, receipt)
            action = copy_verified_file(row["source_path"], row["target_path"], row["sha256"])
            receipt["files"].append({**row, "verification": action})
            receipt["completed_files"] = position
            receipt["verified_bytes"] = sum(value["bytes"] for value in receipt["files"])
            receipt["elapsed_seconds"] = time.monotonic() - started
            save_json(args.receipt, receipt)
            print(json.dumps({"phase": "hot_stage", "completed": position, "total": len(records),
                              "file": row["target_path"], "verification": action,
                              "verified_gib": receipt["verified_bytes"] / 2**30}), flush=True)
        receipt.update(status="ready", elapsed_seconds=time.monotonic() - started)
        receipt.pop("active_file", None)
        save_json(args.receipt, receipt)
        return receipt
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       elapsed_seconds=time.monotonic() - started)
        save_json(args.receipt, receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--segment-index", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=20)
    parser.add_argument("--plan-only", action="store_true")
    stage(parser.parse_args())


if __name__ == "__main__":
    main()
