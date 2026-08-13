"""Resumable full PulseDB v2 manifest, schema, index, and split construction.

This module intentionally creates participant splits before event eligibility is
computed.  It reads metadata and waveform shapes but does not copy waveform
samples into the segment index.  Locked-test segments are written to a separate
quarantine artifact and are not summarized by BP outcome.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema_audit import REQUIRED_FIELDS, _decode_matlab_char
from .splits import assign_subject_splits, assert_disjoint_subject_splits


SOURCE_MAP = {"PulseDB_MIMIC": "MIMIC", "PulseDB_Vital": "VitalDB"}
EXPECTED_SOURCE_COUNTS = {"PulseDB_MIMIC": 2423, "PulseDB_Vital": 2938}
FILE_NAME_RE = re.compile(r"p\d{6}\.mat")
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
AUDIT_SCHEMA_VERSION = "2026-08-12.2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _scalar(h5file: h5py.File, reference: h5py.Reference) -> float:
    values = np.asarray(h5file[reference][()]).reshape(-1, order="F")
    if values.size != 1:
        raise ValueError(f"expected scalar target, found {values.shape}")
    return float(values[0])


def _text(h5file: h5py.File, reference: h5py.Reference) -> str:
    return _decode_matlab_char(np.asarray(h5file[reference][()]))


def _logical_json(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _logical_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_logical_json(item) for item in value]
    return value


def _audit_one_file(
    input_path_text: str,
    data_root_text: str,
    shard_root_text: str,
) -> dict[str, Any]:
    """Audit one participant file and atomically write resumable shards."""

    input_path = Path(input_path_text)
    data_root = Path(data_root_text)
    shard_root = Path(shard_root_text)
    relative = input_path.relative_to(data_root)
    source_directory = relative.parts[0]
    source = SOURCE_MAP.get(source_directory, "unknown")
    key = f"{source_directory}/{input_path.stem}"
    summary_path = shard_root / "file_summaries" / source_directory / f"{input_path.stem}.json"
    segment_path = shard_root / "segments" / source_directory / f"{input_path.stem}.parquet"
    size = input_path.stat().st_size
    mtime_ns = input_path.stat().st_mtime_ns

    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            previous.get("audit_schema_version") == AUDIT_SCHEMA_VERSION
            and
            previous.get("raw_file_size") == size
            and previous.get("raw_file_mtime_ns") == mtime_ns
            and previous.get("audit_complete") is True
            and (previous.get("n_segments", 0) == 0 or segment_path.is_file())
        ):
            previous["resumed_from_shard"] = True
            return previous

    digest = _sha256(input_path)
    with input_path.open("rb") as handle:
        header = handle.read(128)
        handle.seek(512)
        signature = handle.read(8)

    base: dict[str, Any] = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "raw_file": str(input_path.resolve()),
        "raw_file_relative_path": relative.as_posix(),
        "raw_file_name": input_path.name,
        "raw_file_stem": input_path.stem,
        "source_directory": source_directory,
        "source": source,
        "raw_file_size": size,
        "raw_file_mtime_ns": mtime_ns,
        "raw_file_sha256": digest,
        "matlab_73_header": header.startswith(b"MATLAB 7.3 MAT-file"),
        "hdf5_signature_at_512": signature == HDF5_SIGNATURE,
        "file_name_valid": FILE_NAME_RE.fullmatch(input_path.name) is not None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "resumed_from_shard": False,
    }

    file_failures: list[str] = []
    rows: list[dict[str, Any]] = []
    subject_ids: list[str] = []
    case_ids: list[str] = []
    genders: list[str] = []
    ages: list[float] = []

    for flag, name in (
        (base["matlab_73_header"], "matlab_73_header"),
        (base["hdf5_signature_at_512"], "hdf5_signature_at_512"),
        (base["file_name_valid"], "file_name_valid"),
        (source in {"MIMIC", "VitalDB"}, "source_directory_valid"),
    ):
        if not flag:
            file_failures.append(name)

    try:
        with h5py.File(input_path, "r") as h5file:
            if "Subj_Wins" not in h5file or not isinstance(h5file["Subj_Wins"], h5py.Group):
                raise ValueError("missing Subj_Wins group")
            group = h5file["Subj_Wins"]
            missing = sorted(REQUIRED_FIELDS.difference(group.keys()))
            if missing:
                file_failures.append("required_fields_present")
                raise ValueError(f"missing required fields: {missing}")

            required_datasets = {field: group[field] for field in REQUIRED_FIELDS}
            reference_modes = {
                field: h5py.check_dtype(ref=dataset.dtype) is not None
                for field, dataset in required_datasets.items()
            }
            if len(set(reference_modes.values())) != 1:
                file_failures.append("required_storage_mode_consistent")
                raise ValueError(f"required fields mix direct and reference storage: {reference_modes}")
            storage_mode = "references" if next(iter(reference_modes.values())) else "direct_single_window"

            def values_for(field: str) -> list[np.ndarray]:
                dataset = group[field]
                if h5py.check_dtype(ref=dataset.dtype) is None:
                    return [np.asarray(dataset[()])]
                references = np.asarray(dataset[()]).reshape(-1, order="F")
                if any(not bool(reference) for reference in references):
                    raise ValueError(f"{field} contains null references")
                return [np.asarray(h5file[reference][()]) for reference in references]

            required_values = {field: values_for(field) for field in REQUIRED_FIELDS}
            counts = {field: len(values) for field, values in required_values.items()}
            n_segments = counts.get("PPG_Raw", 0)
            if not counts or len(set(counts.values())) != 1:
                file_failures.append("required_container_counts_align")
                raise ValueError(f"required window counts differ: {counts}")
            if n_segments <= 0:
                file_failures.append("participant_has_segments")
                raise ValueError("participant has zero segments")

            optional_values = {
                field: values_for(field)
                for field in ("Age", "Gender")
                if field in group
            }
            optional_values = {
                field: values
                for field, values in optional_values.items()
                if len(values) == n_segments
            }

            subject_ids = [_decode_matlab_char(value) for value in required_values["SubjectID"]]
            case_ids = [_decode_matlab_char(value) for value in required_values["CaseID"]]
            genders = (
                [_decode_matlab_char(value) for value in optional_values["Gender"]]
                if "Gender" in optional_values
                else [""] * n_segments
            )
            ages = (
                [float(np.asarray(value).reshape(-1, order="F")[0]) for value in optional_values["Age"]]
                if "Age" in optional_values
                else [float("nan")] * n_segments
            )
            unique_subjects = sorted(set(subject_ids))
            if len(unique_subjects) != 1 or not unique_subjects[0]:
                file_failures.append("one_nonempty_subject_per_file")
            if unique_subjects and unique_subjects[0] != input_path.stem:
                file_failures.append("subject_id_matches_file_name")
            if any(not case_id for case_id in case_ids):
                file_failures.append("case_ids_nonempty")

            record_order_by_case: dict[str, int] = {}
            for case_id in case_ids:
                if case_id not in record_order_by_case:
                    record_order_by_case[case_id] = len(record_order_by_case)

            scalar_fields = (
                "SegmentID",
                "WinID",
                "WinSeqID",
                "IncludeFlag",
                "SegSBP",
                "SegDBP",
                "PPG_ABP_Corr",
                "ABP_Lag",
            )
            scalar_values = {
                field: [float(np.asarray(value).reshape(-1, order="F")[0]) for value in required_values[field]]
                for field in scalar_fields
            }

            for index in range(n_segments):
                invalid: list[str] = []
                time_values = np.asarray(required_values["T"][index], dtype=float).reshape(-1, order="F")
                if time_values.size < 2:
                    invalid.append("time_too_short")
                if not np.isfinite(time_values).all():
                    invalid.append("time_nonfinite")
                differences = np.diff(time_values)
                if differences.size == 0 or not np.all(differences > 0):
                    invalid.append("time_not_strictly_increasing")
                ppg_raw = np.asarray(required_values["PPG_Raw"][index])
                ppg_f = np.asarray(required_values["PPG_F"][index])
                if ppg_raw.size != time_values.size:
                    invalid.append("ppg_raw_time_length_mismatch")
                if ppg_f.size != time_values.size:
                    invalid.append("ppg_f_time_length_mismatch")
                if ppg_f.size:
                    ppg_values = np.asarray(ppg_f, dtype=float).reshape(-1, order="F")
                    ppg_finite = bool(np.isfinite(ppg_values).all())
                    finite_ppg = ppg_values[np.isfinite(ppg_values)]
                    ppg_minimum = float(np.min(finite_ppg)) if finite_ppg.size else float("nan")
                    ppg_maximum = float(np.max(finite_ppg)) if finite_ppg.size else float("nan")
                    ppg_mean = float(np.mean(finite_ppg)) if finite_ppg.size else float("nan")
                    ppg_standard_deviation = (
                        float(np.std(finite_ppg)) if finite_ppg.size else float("nan")
                    )
                    if not ppg_finite:
                        invalid.append("ppg_f_nonfinite")
                    if not math.isfinite(ppg_standard_deviation) or ppg_standard_deviation <= 0:
                        invalid.append("ppg_f_constant_or_empty")
                else:
                    ppg_finite = False
                    ppg_minimum = ppg_maximum = ppg_mean = ppg_standard_deviation = float("nan")
                scalar_row = {field: scalar_values[field][index] for field in scalar_fields}
                if not all(math.isfinite(value) for value in scalar_row.values()):
                    invalid.append("required_scalar_nonfinite")
                sbp = scalar_row["SegSBP"]
                dbp = scalar_row["SegDBP"]
                if math.isfinite(sbp) and math.isfinite(dbp) and sbp <= dbp:
                    invalid.append("sbp_not_greater_than_dbp")
                if not subject_ids[index]:
                    invalid.append("subject_id_empty")
                if not case_ids[index]:
                    invalid.append("case_id_empty")

                if time_values.size >= 2 and np.isfinite(time_values).all() and np.all(differences > 0):
                    sample_interval = float(np.median(differences))
                    start = float(time_values[0])
                    end = float(time_values[-1])
                    duration = end - start + sample_interval
                    sampling_rate = 1.0 / sample_interval
                else:
                    sample_interval = start = end = duration = sampling_rate = float("nan")

                subject_id = subject_ids[index]
                subject_uid = f"{source}:{subject_id}"
                rows.append(
                    {
                        "dataset_id": "PulseDB_v2",
                        "source": source,
                        "source_directory": source_directory,
                        "subject_id": subject_id,
                        "subject_uid": subject_uid,
                        "record_id": case_ids[index],
                        "record_order": record_order_by_case.get(case_ids[index], -1),
                        "segment_row": index,
                        "segment_uid": f"{source}:{subject_id}:{input_path.stem}:{index:06d}",
                        "segment_id": int(round(scalar_row["SegmentID"])) if math.isfinite(scalar_row["SegmentID"]) else None,
                        "win_id": int(round(scalar_row["WinID"])) if math.isfinite(scalar_row["WinID"]) else None,
                        "win_seq_id": int(round(scalar_row["WinSeqID"])) if math.isfinite(scalar_row["WinSeqID"]) else None,
                        "start_time_s": start,
                        "end_time_s": end,
                        "duration_s": duration,
                        "sample_interval_s": sample_interval,
                        "sampling_rate_hz": sampling_rate,
                        "n_samples": int(time_values.size),
                        "sbp": sbp,
                        "dbp": dbp,
                        "pulse_pressure": sbp - dbp,
                        "include_flag": int(round(scalar_row["IncludeFlag"])) if math.isfinite(scalar_row["IncludeFlag"]) else None,
                        "ppg_abp_corr": scalar_row["PPG_ABP_Corr"],
                        "abp_lag_samples": scalar_row["ABP_Lag"],
                        "age": ages[index],
                        "gender": genders[index],
                        "raw_file": str(input_path.resolve()),
                        "raw_file_relative_path": relative.as_posix(),
                        "raw_file_sha256": digest,
                        "ppg_field": "PPG_F",
                        "ppg_storage_mode": storage_mode,
                        "ppg_reference_index": index,
                        "ppg_f_all_finite": ppg_finite,
                        "ppg_f_min": ppg_minimum,
                        "ppg_f_max": ppg_maximum,
                        "ppg_f_mean": ppg_mean,
                        "ppg_f_std": ppg_standard_deviation,
                        "bp_broad_diagnostic_outlier": bool(
                            math.isfinite(sbp)
                            and math.isfinite(dbp)
                            and (sbp < 50 or sbp > 260 or dbp < 20 or dbp > 180)
                        ),
                        "segment_schema_valid": not invalid,
                        "segment_exclusion_reasons": ";".join(sorted(set(invalid))),
                    }
                )

            frame = pd.DataFrame(rows)
            if not frame.empty and frame["segment_uid"].duplicated().any():
                file_failures.append("segment_uid_unique")
            valid_rows = frame[frame["segment_schema_valid"]] if not frame.empty else frame
            if valid_rows.empty:
                file_failures.append("file_has_valid_segments")
            if not valid_rows.empty:
                duplicate_intervals = valid_rows.duplicated(
                    ["record_id", "start_time_s", "end_time_s"], keep=False
                )
                if duplicate_intervals.any():
                    duplicate_uids = set(valid_rows.loc[duplicate_intervals, "segment_uid"])
                    frame.loc[frame["segment_uid"].isin(duplicate_uids), "segment_schema_valid"] = False
                    frame.loc[frame["segment_uid"].isin(duplicate_uids), "segment_exclusion_reasons"] = (
                        frame.loc[frame["segment_uid"].isin(duplicate_uids), "segment_exclusion_reasons"]
                        .replace("", "duplicate_interval")
                        .where(lambda values: values == "duplicate_interval", lambda values: values + ";duplicate_interval")
                    )
            # Greedily preserve the first chronological non-overlapping row in
            # each record.  This is label-independent and prevents the same
            # physiological interval from crossing support/query roles later.
            valid_rows = frame[frame["segment_schema_valid"]].sort_values(
                ["record_order", "start_time_s", "end_time_s", "segment_uid"],
                kind="mergesort",
            )
            overlap_rejections: set[str] = set()
            for _, record in valid_rows.groupby("record_id", sort=False):
                accepted_end: float | None = None
                for row in record.itertuples(index=False):
                    tolerance = row.sample_interval_s / 2.0
                    if accepted_end is not None and row.start_time_s < accepted_end - tolerance:
                        overlap_rejections.add(row.segment_uid)
                    else:
                        accepted_end = row.end_time_s
            if overlap_rejections:
                mask = frame["segment_uid"].isin(overlap_rejections)
                frame.loc[mask, "segment_schema_valid"] = False
                existing = frame.loc[mask, "segment_exclusion_reasons"]
                frame.loc[mask, "segment_exclusion_reasons"] = np.where(
                    existing.eq(""),
                    "positive_overlap_with_previous_accepted",
                    existing + ";positive_overlap_with_previous_accepted",
                )
            rows = frame.to_dict("records")
    except Exception as exc:
        file_failures.append(f"parse_exception:{type(exc).__name__}")
        base["parse_error"] = str(exc)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        _atomic_parquet(segment_path, frame)

    unique_subjects = sorted(set(subject_ids))
    subject_id = unique_subjects[0] if len(unique_subjects) == 1 else None
    valid_count = int(frame["segment_schema_valid"].sum()) if not frame.empty else 0
    if not frame.empty and valid_count == 0:
        file_failures.append("file_has_valid_segments_after_interval_checks")
    summary = {
        **base,
        "audit_complete": True,
        "n_segments": int(len(frame)),
        "n_valid_segments": valid_count,
        "n_invalid_segments": int(len(frame) - valid_count),
        "subject_id": subject_id,
        "subject_uid": f"{source}:{subject_id}" if subject_id else None,
        "case_count": len(set(case_ids)),
        "storage_mode": locals().get("storage_mode"),
        "file_schema_valid_pre_duplicate_check": not file_failures and valid_count > 0,
        "file_failures": sorted(set(file_failures)),
        "segment_shard": str(segment_path) if not frame.empty else None,
        "age_values": sorted({value for value in ages if math.isfinite(value)}),
        "gender_values": sorted({value for value in genders if value}),
    }
    _atomic_json(summary_path, _logical_json(summary))
    return summary


def _write_streamed_parquet(paths: list[Path], output: Path) -> tuple[int, str]:
    writer: pq.ParquetWriter | None = None
    total = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    for path in paths:
        table = pq.read_table(path)
        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
        writer.write_table(table)
        total += table.num_rows
    if writer is None:
        raise ValueError("no segment shards were available")
    writer.close()
    temporary.replace(output)
    return total, _sha256(output)


def _write_partitioned_segments(
    summaries: pd.DataFrame,
    splits: pd.DataFrame,
    output_root: Path,
) -> dict[str, Any]:
    split_by_subject = splits.set_index("subject_uid")["split"].to_dict()
    writers: dict[str, pq.ParquetWriter] = {}
    temporary_paths = {
        "development": output_root / "development_segments.parquet.tmp",
        "locked_meta_test": output_root / "locked" / "locked_meta_test_segments.parquet.tmp",
    }
    final_paths = {
        "development": output_root / "development_segments.parquet",
        "locked_meta_test": output_root / "locked" / "locked_meta_test_segments.parquet",
    }
    counts = {"development": 0, "locked_meta_test": 0}
    for path in temporary_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for row in summaries.itertuples(index=False):
            if not row.subject_schema_valid or not row.segment_shard:
                continue
            frame = pd.read_parquet(row.segment_shard)
            frame = frame[frame["segment_schema_valid"]].copy()
            if frame.empty:
                continue
            split = split_by_subject[row.subject_uid]
            frame["split"] = split
            destination = "locked_meta_test" if split == "meta_test" else "development"
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if destination not in writers:
                writers[destination] = pq.ParquetWriter(
                    temporary_paths[destination], table.schema, compression="zstd"
                )
            writers[destination].write_table(table)
            counts[destination] += len(frame)
    finally:
        for writer in writers.values():
            writer.close()
    artifacts: dict[str, Any] = {}
    for key, temporary in temporary_paths.items():
        if not temporary.is_file():
            raise ValueError(f"no rows written for {key}")
        temporary.replace(final_paths[key])
        artifacts[key] = {
            "path": str(final_paths[key]),
            "rows": counts[key],
            "sha256": _sha256(final_paths[key]),
        }
    return artifacts


def run_full_cohort(
    data_root: Path,
    output_root: Path,
    *,
    workers: int = 4,
    seed: int = 20260809,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    shard_root = output_root / "shards"
    paths = sorted(data_root.glob("PulseDB_*/*.mat"))
    actual_source_counts = {
        source: sum(path.parent.name == source for path in paths)
        for source in EXPECTED_SOURCE_COUNTS
    }
    if actual_source_counts != EXPECTED_SOURCE_COUNTS:
        raise ValueError(
            f"expected source counts {EXPECTED_SOURCE_COUNTS}, found {actual_source_counts}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "full_cohort_progress.json"
    summaries: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_by_path = {
            executor.submit(_audit_one_file, str(path), str(data_root), str(shard_root)): path
            for path in paths
        }
        for completed, future in enumerate(as_completed(future_by_path), start=1):
            path = future_by_path[future]
            try:
                summary = future.result()
            except Exception as exc:
                summary = {
                    "raw_file": str(path),
                    "raw_file_relative_path": str(path.relative_to(data_root)),
                    "raw_file_name": path.name,
                    "raw_file_stem": path.stem,
                    "source_directory": path.parent.name,
                    "source": SOURCE_MAP.get(path.parent.name, "unknown"),
                    "raw_file_size": path.stat().st_size,
                    "raw_file_mtime_ns": path.stat().st_mtime_ns,
                    "raw_file_sha256": None,
                    "matlab_73_header": False,
                    "hdf5_signature_at_512": False,
                    "file_name_valid": FILE_NAME_RE.fullmatch(path.name) is not None,
                    "audit_complete": False,
                    "file_schema_valid_pre_duplicate_check": False,
                    "file_failures": [f"worker_exception:{type(exc).__name__}"],
                    "parse_error": str(exc),
                    "n_segments": 0,
                    "n_valid_segments": 0,
                    "subject_uid": None,
                    "segment_shard": None,
                }
            summaries.append(summary)
            if completed == 1 or completed % 10 == 0 or completed == len(paths):
                progress = {
                    "phase": "file_audit",
                    "completed_files": completed,
                    "total_files": len(paths),
                    "schema_valid_pre_duplicate": sum(
                        bool(item.get("file_schema_valid_pre_duplicate_check")) for item in summaries
                    ),
                    "failed_files": sum(not bool(item.get("audit_complete")) for item in summaries),
                    "last_file": str(path),
                    "started_at_utc": started.isoformat(),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _atomic_json(progress_path, progress)
                print(
                    f"FULL_COHORT_PROGRESS={completed}/{len(paths)} "
                    f"valid_pre_duplicate={progress['schema_valid_pre_duplicate']} "
                    f"last={path.name}",
                    flush=True,
                )

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["source_directory", "raw_file_name"], kind="mergesort"
    ).reset_index(drop=True)
    duplicate_subjects = set(
        summary_frame.loc[
            summary_frame["subject_uid"].notna()
            & summary_frame["subject_uid"].duplicated(keep=False),
            "subject_uid",
        ]
    )
    summary_frame["subject_schema_valid"] = (
        summary_frame["file_schema_valid_pre_duplicate_check"].fillna(False)
        & ~summary_frame["subject_uid"].isin(duplicate_subjects)
    )
    summary_frame["post_consolidation_failures"] = summary_frame["subject_uid"].map(
        lambda value: "subject_uid_in_multiple_raw_files" if value in duplicate_subjects else ""
    )

    manifest_columns = [
        "source_directory",
        "source",
        "raw_file_name",
        "raw_file_relative_path",
        "raw_file",
        "raw_file_size",
        "raw_file_mtime_ns",
        "raw_file_sha256",
        "matlab_73_header",
        "hdf5_signature_at_512",
        "file_name_valid",
    ]
    manifest = summary_frame[manifest_columns].copy()
    manifest_csv = output_root / "pulsedb_v2_raw_file_manifest.csv"
    manifest_parquet = output_root / "pulsedb_v2_raw_file_manifest.parquet"
    manifest.to_csv(manifest_csv, index=False)
    manifest.to_parquet(manifest_parquet, index=False)
    canonical_manifest = "\n".join(
        f"{row.source_directory}\t{row.raw_file_relative_path}\t{row.raw_file_size}\t{row.raw_file_sha256}"
        for row in manifest.itertuples(index=False)
    )
    manifest_digest = hashlib.sha256(canonical_manifest.encode("utf-8")).hexdigest()
    (output_root / "pulsedb_v2_raw_file_manifest.sha256").write_text(
        manifest_digest + "\n", encoding="ascii"
    )

    summary_csv = output_root / "pulsedb_v2_file_summary.csv"
    summary_parquet = output_root / "pulsedb_v2_file_summary.parquet"
    summary_frame.to_csv(summary_csv, index=False)
    summary_frame.to_parquet(summary_parquet, index=False)

    segment_paths = [
        Path(path) for path in summary_frame["segment_shard"].dropna().astype(str)
    ]
    full_index_path = output_root / "pulsedb_v2_full_segment_index.parquet"
    n_full_segments, full_index_sha = _write_streamed_parquet(segment_paths, full_index_path)

    valid_subjects = summary_frame[summary_frame["subject_schema_valid"]].copy()
    subject_table = valid_subjects[
        [
            "subject_uid",
            "subject_id",
            "source",
            "source_directory",
            "raw_file_relative_path",
            "raw_file_sha256",
            "n_segments",
            "n_valid_segments",
            "case_count",
            "age_values",
            "gender_values",
        ]
    ].copy()
    subject_table.to_csv(output_root / "schema_valid_subjects.csv", index=False)
    subject_table.to_parquet(output_root / "schema_valid_subjects.parquet", index=False)

    split_input = subject_table[["subject_uid", "source"]].rename(
        columns={"subject_uid": "subject_id"}
    )
    splits = assign_subject_splits(split_input, seed=seed).rename(
        columns={"subject_id": "subject_uid"}
    )
    assert_disjoint_subject_splits(splits.rename(columns={"subject_uid": "subject_id"}))
    if set(splits["subject_uid"]) != set(subject_table["subject_uid"]):
        raise AssertionError("every and only schema-valid subject must receive one split")
    split_csv = output_root / "subject_splits.csv"
    splits.to_csv(split_csv, index=False)
    split_file_sha = _sha256(split_csv)
    (output_root / "subject_splits.sha256").write_text(split_file_sha + "\n", encoding="ascii")

    partition_artifacts = _write_partitioned_segments(summary_frame, splits, output_root)
    result = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "source_file_counts": actual_source_counts,
        "n_raw_files": len(summary_frame),
        "n_schema_valid_subjects": int(summary_frame["subject_schema_valid"].sum()),
        "n_schema_invalid_files": int((~summary_frame["subject_schema_valid"]).sum()),
        "duplicate_subject_uid_count": len(duplicate_subjects),
        "n_full_segments": n_full_segments,
        "n_valid_segments": int(summary_frame.loc[summary_frame["subject_schema_valid"], "n_valid_segments"].sum()),
        "split_counts": {key: int(value) for key, value in splits["split"].value_counts().items()},
        "seed": seed,
        "protocol_order": "schema validity -> subject split -> development-only event eligibility",
        "locked_test_outcomes_summarized": False,
        "artifacts": {
            "raw_manifest_csv": str(manifest_csv),
            "raw_manifest_parquet": str(manifest_parquet),
            "raw_manifest_canonical_sha256": manifest_digest,
            "file_summary_csv": str(summary_csv),
            "file_summary_parquet": str(summary_parquet),
            "full_segment_index": str(full_index_path),
            "full_segment_index_sha256": full_index_sha,
            "schema_valid_subjects": str(output_root / "schema_valid_subjects.csv"),
            "subject_splits": str(split_csv),
            "subject_splits_file_sha256": split_file_sha,
            **partition_artifacts,
        },
    }
    _atomic_json(output_root / "full_cohort_summary.json", _logical_json(result))
    _atomic_json(
        progress_path,
        {
            "phase": "complete",
            "completed_files": len(paths),
            "total_files": len(paths),
            "summary": str(output_root / "full_cohort_summary.json"),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_full_cohort(
        args.data_root,
        args.output,
        workers=args.workers,
        seed=args.seed,
    )
    print(json.dumps(_logical_json(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
