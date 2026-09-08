"""CPU-only post-preparation rehearsal of the official training data contracts.

This opens the permitted official TRAIN manifest (including its training BP),
the separate test INPUT manifest, and NumPy shard headers. It never opens raw
MAT files, the full BP index, test targets, a checkpoint, or a neural model.
The same runtime validators used by LoRA and personal-memory training are
exercised before GPU dependencies can be released.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from pulsedb_fewshot.official_calbased_train import (
    PROTOCOL_ID, canonical_metadata, event_ids_sha256, load_store,
    load_test_inputs, select_stage, sha256,
)
from pulsedb_fewshot.official_memory_train import canonical_rows
from pulsedb_fewshot.personal_memory_prepare import audit_metadata
from pulsedb_fewshot.official_time_policy import TIME_BOUNDARY_POLICY


def save_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit_memory_pair(bank: pd.DataFrame, query: pd.DataFrame, *, stage: str) -> dict:
    """Use the real export-to-retrieval lineage conversion; no fake features."""
    bank_meta = canonical_metadata(bank, "train")
    query_meta = canonical_metadata(query, "validation")
    bank_rows = canonical_rows(bank_meta, "train")
    query_rows = canonical_rows(query_meta, "internal_validation")
    result = audit_metadata(bank_rows, query_rows)
    if set(bank_meta.subject_uid) != set(query_meta.subject_uid):
        raise ValueError(f"{stage}: registered people differ between bank and query")
    result.update(stage=stage, time_boundary_policy=TIME_BOUNDARY_POLICY,
                  bank_ids_sha256=event_ids_sha256(bank_meta.event_id),
                  query_ids_sha256=event_ids_sha256(query_meta.event_id),
                  bank_per_subject_min=int(bank_meta.groupby("subject_uid").size().min()),
                  bank_per_subject_max=int(bank_meta.groupby("subject_uid").size().max()),
                  query_per_subject_min=int(query_meta.groupby("subject_uid").size().min()),
                  query_per_subject_max=int(query_meta.groupby("subject_uid").size().max()))
    return result


def validate_shard_headers(root: Path, train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Validate allocation bounds without loading the full waveform arrays."""
    metadata = pd.concat([train[["waveform_file", "waveform_row"]],
                          test[["waveform_file", "waveform_row"]]], ignore_index=True)
    evidence = []
    for filename, rows in metadata.groupby("waveform_file", sort=True):
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or path.is_symlink():
            raise ValueError("unsafe or aliased waveform shard path")
        indices = pd.to_numeric(rows.waveform_row, errors="raise").to_numpy(dtype=float)
        if (not np.isfinite(indices).all() or (indices < 0).any()
                or not np.equal(indices, np.floor(indices)).all()):
            raise ValueError("waveform shard row index is invalid")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (len(rows), 1250):
            raise ValueError(f"official shard must be complete float32 [N,1250]: {filename}")
        if len(np.unique(indices)) != len(rows) or indices.max() >= len(array):
            raise ValueError(f"waveform shard row reused or out of bounds: {filename}")
        evidence.append({"file": filename, "rows": len(rows), "samples_per_row": 1250,
                         "dtype": str(array.dtype), "bytes": path.stat().st_size})
        del array
    return evidence


def run(args) -> dict:
    root = args.store_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("preserve the previous post-preparation receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial_manifest_hash = sha256(root / "manifest.json")
    report = {"protocol_id": PROTOCOL_ID, "status": "running",
              "time_boundary_policy": TIME_BOUNDARY_POLICY,
              "store_root": str(root), "store_manifest_sha256": initial_manifest_hash,
              "script_sha256": sha256(Path(__file__)),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "official_training_labels_accessed": False,
              "official_test_input_metadata_accessed": False,
              "official_test_targets_accessed": False,
              "neural_training_performed": False,
              "waveform_values_loaded": False,
              "checks": []}
    save_report(output, report)
    try:
        report["official_training_labels_accessed"] = True
        manifest, train = load_store(root)
        train_path, test_path = root / "train_manifest.parquet", root / "test_inputs.parquet"
        if (sha256(train_path) != manifest.get("train_manifest_sha256")
                or sha256(test_path) != manifest.get("test_inputs_sha256")):
            raise ValueError("prepared train/input manifest checksum changed")
        report.update(train_manifest_sha256=sha256(train_path),
                      test_inputs_sha256=sha256(test_path))

        fit, exports = select_stage(train, "inner")
        report["checks"].append(audit_memory_pair(fit, exports["validation"], stage="inner_320_40"))
        inner_ids = set(fit.segment_uid)
        excluded_seen = set()
        del fit, exports
        save_report(output, report)
        print("POSTPREP_INNER_LINEAGE=pass", flush=True)

        for fold in range(3):
            fit, exports = select_stage(train, "oof", fold)
            held = exports["excluded_fold"]
            excluded_ids = set(held.segment_uid)
            if excluded_seen & excluded_ids or set(fit.segment_uid) | excluded_ids != inner_ids:
                raise ValueError("OOF folds fail exact once-only inner-training coverage")
            excluded_seen.update(excluded_ids)
            report["checks"].append(audit_memory_pair(fit, held, stage=f"oof_{fold}_fit_excluded"))
            # Its outer-validation cache is exported without influencing fit.
            if set(fit.segment_uid) & set(exports["outer_validation"].segment_uid):
                raise ValueError("OOF fitting rows overlap internal validation")
            del fit, held, exports, excluded_ids
            gc.collect()
            save_report(output, report)
            print(f"POSTPREP_OOF_{fold}_LINEAGE=pass", flush=True)
        if excluded_seen != inner_ids:
            raise ValueError("OOF excluded rows do not cover all 320 training windows/person")
        del inner_ids, excluded_seen

        full_fit, exports = select_stage(train, "final")
        if len(full_fit) != len(train) or set(full_fit.segment_uid) != set(train.segment_uid):
            raise ValueError("final stage must fit exact official 360-window membership")
        del exports
        report["official_test_input_metadata_accessed"] = True
        test = load_test_inputs(root, train)
        report["checks"].append(audit_memory_pair(full_fit, test, stage="final_360_test_inputs_40"))
        report["shard_headers"] = validate_shard_headers(root, train, test)
        report["source_counts"] = train.groupby("source").subject_uid.nunique().to_dict()
        if report["source_counts"] != {"MIMIC": 1213, "VitalDB": 1293}:
            raise ValueError("official source-specific participant counts differ")
        if sha256(root / "manifest.json") != initial_manifest_hash:
            raise ValueError("official store manifest changed during post-preparation audit")
        report.update(status="pass", train_windows=len(train), test_input_windows=len(test),
                      participants=int(train.subject_uid.nunique()),
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - started,
                      scope_limit="Metadata/runtime-contract and shard-header verification only; not a completed model fit, PPG rehash, or proof of physical sample independence.")
        save_report(output, report)
        print("POSTPREP_FINAL_LINEAGE=pass", flush=True)
        print(f"POSTPREP_AUDIT=pass RECEIPT={output}", flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.monotonic() - started)
        save_report(output, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
