"""Comprehensive, read-only audit of PulseDB v2 MATLAB 7.3 participant files.

The command validates storage structure, MATLAB object references, identity,
time axes, waveform integrity, BP labels, and quality metadata.  It also emits
a normalized segment index without copying raw waveform samples into the index.

ABP-derived fields are retained for offline auditing only.  They must not be
used as routine PPG-only model inputs, and the exploratory event counts written
by this module must not be used to select a final event spacing from locked-test
participants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .events import eventize_segments


REQUIRED_FIELDS = {
    "PPG_Raw",
    "PPG_F",
    "ABP_Raw",
    "ABP_F",
    "SegSBP",
    "SegDBP",
    "T",
    "SubjectID",
    "CaseID",
    "SegmentID",
    "WinID",
    "WinSeqID",
    "IncludeFlag",
    "PPG_ABP_Corr",
    "ABP_Lag",
}

WAVEFORM_FIELDS = (
    "PPG_Raw",
    "PPG_F",
    "PPG_Record",
    "PPG_Record_F",
    "ABP_Raw",
    "ABP_F",
)

SCALAR_FIELDS = (
    "Age",
    "SegmentID",
    "WinID",
    "WinSeqID",
    "IncludeFlag",
    "SegSBP",
    "SegDBP",
    "PPG_ABP_Corr",
    "ABP_Lag",
)

# The combined pilot needs every field container for alignment checks, but only
# these project-relevant targets must be dereferenced.  The default single-file
# audit remains exhaustive so the first schema specimen still covers all fields.
PROJECT_FIELDS = REQUIRED_FIELDS | {"Age", "Gender"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, Any] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
        "unique_count": int(np.unique(finite).size) if finite.size else 0,
    }
    if finite.size:
        result.update(
            {
                "minimum": float(np.min(finite)),
                "maximum": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "standard_deviation": float(np.std(finite)),
                "median": float(np.median(finite)),
                "q01": float(np.quantile(finite, 0.01)),
                "q99": float(np.quantile(finite, 0.99)),
            }
        )
    return result


def _decode_matlab_char(values: np.ndarray) -> str:
    flat = np.asarray(values).reshape(-1, order="F")
    return "".join(chr(int(code)) for code in flat if int(code) != 0)


def _infer_source(path: Path) -> str:
    upper = str(path).upper()
    if "PULSEDB_MIMIC" in upper or "MIMIC" in upper:
        return "MIMIC"
    if "PULSEDB_VITAL" in upper or "VITALDB" in upper:
        return "VitalDB"
    return "unknown"


def _waveform_summary(windows: list[np.ndarray]) -> dict[str, Any]:
    finite_counts: list[int] = []
    nonfinite_counts: list[int] = []
    minima: list[float] = []
    maxima: list[float] = []
    means: list[float] = []
    standard_deviations: list[float] = []
    constant_indices: list[int] = []
    hashes: list[str] = []

    for index, raw in enumerate(windows):
        values = np.asarray(raw, dtype=float).reshape(-1, order="F")
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        finite_counts.append(int(finite.size))
        nonfinite_counts.append(int(values.size - finite.size))
        if finite.size:
            minima.append(float(np.min(finite)))
            maxima.append(float(np.max(finite)))
            means.append(float(np.mean(finite)))
            std = float(np.std(finite))
            standard_deviations.append(std)
            if std == 0.0:
                constant_indices.append(index)
        else:
            minima.append(float("nan"))
            maxima.append(float("nan"))
            means.append(float("nan"))
            standard_deviations.append(float("nan"))
            constant_indices.append(index)
        contiguous = np.ascontiguousarray(values)
        hashes.append(hashlib.sha256(contiguous.tobytes()).hexdigest())

    indices_by_digest: defaultdict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(hashes):
        indices_by_digest[digest].append(index)
    duplicate_groups = {
        digest: indices_by_digest[digest]
        for digest in sorted(indices_by_digest)
        if len(indices_by_digest[digest]) > 1
    }
    return {
        "n_windows": len(windows),
        "sample_lengths": dict(Counter(int(np.asarray(x).size) for x in windows)),
        "finite_values_per_window": _summary(np.asarray(finite_counts)),
        "nonfinite_values_total": int(sum(nonfinite_counts)),
        "window_minima": _summary(np.asarray(minima)),
        "window_maxima": _summary(np.asarray(maxima)),
        "window_means": _summary(np.asarray(means)),
        "window_standard_deviations": _summary(np.asarray(standard_deviations)),
        "constant_window_indices_zero_based": constant_indices,
        "unique_waveform_hashes": len(set(hashes)),
        "duplicate_waveform_groups_zero_based": duplicate_groups,
    }


def _pair_comparison(left: list[np.ndarray], right: list[np.ndarray]) -> dict[str, Any]:
    maximum_absolute_differences: list[float] = []
    mean_absolute_differences: list[float] = []
    exact_matches = 0
    for left_raw, right_raw in zip(left, right, strict=True):
        left_values = np.asarray(left_raw, dtype=float).reshape(-1, order="F")
        right_values = np.asarray(right_raw, dtype=float).reshape(-1, order="F")
        if left_values.shape != right_values.shape:
            maximum_absolute_differences.append(float("nan"))
            mean_absolute_differences.append(float("nan"))
            continue
        if np.array_equal(left_values, right_values, equal_nan=True):
            exact_matches += 1
        difference = np.abs(left_values - right_values)
        finite = difference[np.isfinite(difference)]
        maximum_absolute_differences.append(float(np.max(finite)) if finite.size else float("nan"))
        mean_absolute_differences.append(float(np.mean(finite)) if finite.size else float("nan"))
    return {
        "n_pairs": len(left),
        "exact_match_count": exact_matches,
        "maximum_absolute_difference_by_window": _summary(
            np.asarray(maximum_absolute_differences)
        ),
        "mean_absolute_difference_by_window": _summary(
            np.asarray(mean_absolute_differences)
        ),
    }


def audit_pulsedb_file(
    input_path: Path,
    output_dir: Path,
    *,
    expected_sha256: str | None = None,
    source: str = "auto",
    project_fields_only: bool = False,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    actual_sha256 = _sha256(input_path)
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    def check(name: str, passed: bool, detail: Any, *, required: bool = True) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "required": bool(required),
                "detail": _jsonable(detail),
            }
        )

    if expected_sha256 is not None:
        check(
            "input_sha256_matches_expected",
            actual_sha256.lower() == expected_sha256.lower(),
            {"expected": expected_sha256.lower(), "actual": actual_sha256.lower()},
        )

    with h5py.File(input_path, "r") as h5file:
        root_keys = sorted(h5file.keys())
        check("matlab_hdf5_root_contains_Subj_Wins", "Subj_Wins" in h5file, root_keys)
        if "Subj_Wins" not in h5file or not isinstance(h5file["Subj_Wins"], h5py.Group):
            raise ValueError("Expected an HDF5 group named Subj_Wins")
        group = h5file["Subj_Wins"]
        fields = sorted(group.keys())
        missing_fields = sorted(REQUIRED_FIELDS.difference(fields))
        check("required_fields_present", not missing_fields, {"missing": missing_fields})

        reference_summaries: dict[str, Any] = {}
        reference_arrays: dict[str, np.ndarray] = {}
        target_paths: dict[str, list[str | None]] = {}
        target_values: dict[str, list[np.ndarray | None]] = {}
        container_sizes: dict[str, int] = {}

        for field in fields:
            dataset = group[field]
            if not isinstance(dataset, h5py.Dataset):
                reference_summaries[field] = {"object_type": type(dataset).__name__}
                continue
            refs = np.asarray(dataset[()]).reshape(-1, order="F")
            reference_arrays[field] = refs
            container_sizes[field] = int(refs.size)
            if project_fields_only and field not in PROJECT_FIELDS:
                reference_summaries[field] = {
                    "container_shape": [int(size) for size in dataset.shape],
                    "container_dtype": str(dataset.dtype),
                    "slot_count": int(refs.size),
                    "targets_loaded": False,
                }
                continue
            shapes: Counter[str] = Counter()
            dtypes: Counter[str] = Counter()
            null_indices: list[int] = []
            paths: list[str | None] = []
            values: list[np.ndarray | None] = []
            for index, reference in enumerate(refs):
                if not reference:
                    null_indices.append(index)
                    paths.append(None)
                    values.append(None)
                    continue
                target = h5file[reference]
                # Resolving ``target.name`` for every MATLAB object reference
                # can repeatedly scan a very large ``#refs#`` group.  The
                # combined pilot audit does not consume these diagnostic paths,
                # so omit them in project-field mode while still dereferencing
                # and validating every required target dataset.
                paths.append(None if project_fields_only else target.name)
                if isinstance(target, h5py.Dataset):
                    shapes[str(tuple(int(size) for size in target.shape))] += 1
                    dtypes[str(target.dtype)] += 1
                    values.append(np.asarray(target[()]))
                else:
                    shapes["group"] += 1
                    dtypes["group"] += 1
                    values.append(None)
            target_paths[field] = paths
            target_values[field] = values
            reference_summaries[field] = {
                "container_shape": [int(size) for size in dataset.shape],
                "container_dtype": str(dataset.dtype),
                "slot_count": int(refs.size),
                "null_count": len(null_indices),
                "null_indices_zero_based": null_indices,
                "target_shape_counts": dict(shapes),
                "target_dtype_counts": dict(dtypes),
                "targets_loaded": True,
            }

        n_windows = container_sizes.get("PPG_Raw", 0)
        check(
            "all_field_containers_have_same_slot_count",
            bool(container_sizes) and len(set(container_sizes.values())) == 1,
            dict(Counter(container_sizes.values())),
        )
        check("participant_file_has_windows", n_windows > 0, {"n_windows": n_windows})
        required_null_counts = {
            field: reference_summaries[field]["null_count"]
            for field in REQUIRED_FIELDS
            if field in reference_summaries and "null_count" in reference_summaries[field]
        }
        check(
            "required_fields_have_no_null_references",
            len(required_null_counts) == len(REQUIRED_FIELDS)
            and all(count == 0 for count in required_null_counts.values()),
            required_null_counts,
        )

        def values_for(field: str) -> list[np.ndarray]:
            raw = target_values[field]
            if any(value is None for value in raw):
                raise ValueError(f"{field} contains a null or non-dataset reference")
            return [np.asarray(value) for value in raw if value is not None]

        def scalar_vector(field: str) -> np.ndarray:
            return np.asarray(
                [np.asarray(value).reshape(-1, order="F")[0] for value in values_for(field)]
            )

        def text_vector(field: str) -> list[str]:
            return [_decode_matlab_char(value) for value in values_for(field)]

        waveforms = {
            field: values_for(field)
            for field in WAVEFORM_FIELDS
            if field in target_values and reference_summaries[field].get("null_count") == 0
        }
        time_windows = [
            np.asarray(value, dtype=float).reshape(-1, order="F")
            for value in values_for("T")
        ]
        scalar_vectors = {
            field: scalar_vector(field)
            for field in SCALAR_FIELDS
            if field in target_values and reference_summaries[field].get("null_count") == 0
        }
        subject_ids = text_vector("SubjectID")
        case_ids = text_vector("CaseID")
        genders = text_vector("Gender") if "Gender" in target_values else [""] * n_windows

        time_lengths = [int(values.size) for values in time_windows]
        time_finite = [bool(np.isfinite(values).all()) for values in time_windows]
        time_increasing = [bool(np.all(np.diff(values) > 0)) for values in time_windows]
        time_steps = np.asarray(
            [float(np.median(np.diff(values))) for values in time_windows], dtype=float
        )
        sampling_rates = 1.0 / time_steps
        starts = np.asarray([values[0] for values in time_windows], dtype=float)
        ends = np.asarray([values[-1] for values in time_windows], dtype=float)
        durations = ends - starts + time_steps
        # Participant files may contain several CaseID records whose time axes
        # and window identifiers restart.  Temporal and identifier integrity
        # must therefore be checked within records, never across the whole
        # participant file.
        record_order_by_case: dict[str, int] = {}
        for case_id in case_ids:
            if case_id not in record_order_by_case:
                record_order_by_case[case_id] = len(record_order_by_case)
        record_indices = {
            case_id: np.asarray(
                [index for index, value in enumerate(case_ids) if value == case_id],
                dtype=int,
            )
            for case_id in record_order_by_case
        }
        record_start_orders: dict[str, np.ndarray] = {}
        start_gap_parts: list[np.ndarray] = []
        inter_window_gap_parts: list[np.ndarray] = []
        duplicate_interval_count = 0
        overlap_count = 0
        touching_interval_count = 0
        record_rows_are_chronological = True
        overlap_tolerance_s = float(np.median(time_steps)) / 2.0
        for case_id, indices in record_indices.items():
            local_order = np.argsort(starts[indices], kind="mergesort")
            ordered_indices = indices[local_order]
            record_start_orders[case_id] = ordered_indices
            local_starts = starts[ordered_indices]
            local_ends = ends[ordered_indices]
            local_start_gaps = np.diff(local_starts)
            local_inter_window_gaps = local_starts[1:] - local_ends[:-1]
            start_gap_parts.append(local_start_gaps)
            inter_window_gap_parts.append(local_inter_window_gaps)
            duplicate_interval_count += len(indices) - len(
                set(zip(starts[indices].tolist(), ends[indices].tolist(), strict=True))
            )
            overlap_count += int(
                np.sum(local_inter_window_gaps < -overlap_tolerance_s)
            )
            touching_interval_count += int(
                np.sum(np.abs(local_inter_window_gaps) <= overlap_tolerance_s)
            )
            record_rows_are_chronological = record_rows_are_chronological and bool(
                np.array_equal(ordered_indices, indices)
            )
        start_gaps = (
            np.concatenate(start_gap_parts) if start_gap_parts else np.asarray([], dtype=float)
        )
        inter_window_gaps = (
            np.concatenate(inter_window_gap_parts)
            if inter_window_gap_parts
            else np.asarray([], dtype=float)
        )

        check("all_time_vectors_are_finite", all(time_finite), {"failed": [i for i, ok in enumerate(time_finite) if not ok]})
        check("all_time_vectors_strictly_increase", all(time_increasing), {"failed": [i for i, ok in enumerate(time_increasing) if not ok]})
        check("time_vector_lengths_are_consistent", len(set(time_lengths)) == 1, dict(Counter(time_lengths)))
        check(
            "sampling_rate_is_consistent_across_windows",
            bool(sampling_rates.size)
            and np.allclose(sampling_rates, np.median(sampling_rates), rtol=1e-8, atol=1e-8),
            _summary(sampling_rates),
        )
        check("window_intervals_are_unique", duplicate_interval_count == 0, {"duplicate_count": duplicate_interval_count})
        check("window_intervals_do_not_overlap", overlap_count == 0, {"overlap_count": overlap_count})
        check(
            "file_row_order_is_chronological_within_record",
            record_rows_are_chronological,
            {case_id: indices.tolist() for case_id, indices in record_start_orders.items()},
        )
        if touching_interval_count:
            warnings.append(
                f"{touching_interval_count} adjacent window pairs touch at a sample boundary; "
                "no positive-duration overlap was counted."
            )

        primary_lengths = {
            field: [int(np.asarray(value).size) for value in values]
            for field, values in waveforms.items()
        }
        aligned_sample_lengths = all(
            primary_lengths[field] == time_lengths
            for field in ("PPG_Raw", "PPG_F", "ABP_Raw", "ABP_F")
            if field in primary_lengths
        )
        check(
            "primary_ppg_abp_time_lengths_align",
            aligned_sample_lengths,
            {field: dict(Counter(lengths)) for field, lengths in primary_lengths.items()},
        )

        identities = {
            "subject_ids": sorted(set(subject_ids)),
            "case_ids": sorted(set(case_ids)),
            "genders": sorted(set(genders)),
            "ages": sorted(set(float(value) for value in scalar_vectors.get("Age", []))),
        }
        check("one_subject_id_per_participant_file", len(identities["subject_ids"]) == 1, identities["subject_ids"])
        check(
            "case_ids_are_nonempty",
            bool(identities["case_ids"]) and all(identities["case_ids"]),
            identities["case_ids"],
        )

        segment_ids = scalar_vectors["SegmentID"].astype(float)
        win_ids = scalar_vectors["WinID"].astype(float)
        win_seq_ids = scalar_vectors["WinSeqID"].astype(float)
        def identifiers_unique_within_record(values: np.ndarray) -> bool:
            return all(np.unique(values[indices]).size == len(indices) for indices in record_indices.values())

        check("segment_ids_are_unique_within_record", identifiers_unique_within_record(segment_ids), _summary(segment_ids))
        check("win_ids_are_unique_within_record", identifiers_unique_within_record(win_ids), _summary(win_ids))
        check("win_seq_ids_are_unique_within_record", identifiers_unique_within_record(win_seq_ids), _summary(win_seq_ids))
        check("segment_win_identifiers_match", bool(np.array_equal(segment_ids, win_ids) and np.array_equal(win_ids, win_seq_ids)), {"all_equal": bool(np.array_equal(segment_ids, win_ids) and np.array_equal(win_ids, win_seq_ids))})
        identifiers_increase_with_time = all(
            bool(np.all(np.diff(win_seq_ids[ordered_indices]) > 0))
            for ordered_indices in record_start_orders.values()
        )
        check("window_identifiers_increase_with_time_within_record", identifiers_increase_with_time, {"first": win_seq_ids[:10].tolist(), "last": win_seq_ids[-10:].tolist()})

        sbp = scalar_vectors["SegSBP"].astype(float)
        dbp = scalar_vectors["SegDBP"].astype(float)
        pulse_pressure = sbp - dbp
        check("bp_labels_are_finite", bool(np.isfinite(sbp).all() and np.isfinite(dbp).all()), {"sbp": _summary(sbp), "dbp": _summary(dbp)})
        check("sbp_is_greater_than_dbp", bool(np.all(sbp > dbp)), {"violating_indices": np.where(sbp <= dbp)[0].tolist()})
        if np.any((sbp < 50) | (sbp > 260) | (dbp < 20) | (dbp > 180)):
            warnings.append("One or more BP labels lie outside the broad diagnostic range used by this audit; inspect them manually.")

        include_flag = scalar_vectors["IncludeFlag"].astype(int)
        ppg_abp_corr = scalar_vectors["PPG_ABP_Corr"].astype(float)
        abp_lag = scalar_vectors["ABP_Lag"].astype(float)
        if np.unique(include_flag).size == 1:
            warnings.append("IncludeFlag is constant in this file; it cannot validate exclusion behavior by itself.")
        median_lag = float(np.median(abp_lag))
        mad_lag = float(np.median(np.abs(abp_lag - median_lag)))
        lag_outlier_indices = (
            np.where(np.abs(abp_lag - median_lag) > 6.0 * mad_lag)[0].tolist()
            if mad_lag > 0
            else []
        )
        if lag_outlier_indices:
            warnings.append(f"ABP_Lag has robust outliers at zero-based rows {lag_outlier_indices}; retain for audit and do not use as a PPG-only feature.")

        waveform_summaries = {
            field: _waveform_summary(values) for field, values in waveforms.items()
        }
        nonfinite_signal_total = sum(
            int(summary["nonfinite_values_total"])
            for summary in waveform_summaries.values()
        )
        constant_signal_windows = {
            field: summary["constant_window_indices_zero_based"]
            for field, summary in waveform_summaries.items()
            if summary["constant_window_indices_zero_based"]
        }
        duplicate_signal_windows = {
            field: summary["duplicate_waveform_groups_zero_based"]
            for field, summary in waveform_summaries.items()
            if summary["duplicate_waveform_groups_zero_based"]
        }
        check("audited_waveforms_have_only_finite_values", nonfinite_signal_total == 0, {"nonfinite_total": nonfinite_signal_total})
        check("audited_waveforms_are_not_constant", not constant_signal_windows, constant_signal_windows)
        check("audited_waveforms_are_not_exact_duplicates", not duplicate_signal_windows, duplicate_signal_windows, required=False)
        if duplicate_signal_windows:
            warnings.append("Exact duplicate waveform arrays were detected; inspect temporal duplication before using these rows.")

        pair_comparisons: dict[str, Any] = {}
        for left, right in (
            ("PPG_Raw", "PPG_Record"),
            ("PPG_F", "PPG_Record_F"),
            ("PPG_Raw", "PPG_F"),
            ("ABP_Raw", "ABP_F"),
        ):
            if left in waveforms and right in waveforms:
                pair_comparisons[f"{left}_vs_{right}"] = _pair_comparison(
                    waveforms[left], waveforms[right]
                )

        abp_f_min = np.asarray(
            [np.nanmin(np.asarray(values, dtype=float)) for values in waveforms["ABP_F"]]
        )
        abp_f_max = np.asarray(
            [np.nanmax(np.asarray(values, dtype=float)) for values in waveforms["ABP_F"]]
        )
        label_abp_diagnostics = {
            "SegSBP_minus_ABP_F_global_max": _summary(sbp - abp_f_max),
            "SegDBP_minus_ABP_F_global_min": _summary(dbp - abp_f_min),
            "note": "Diagnostic only: PulseDB labels may use detected peaks/turns rather than the global waveform extrema.",
        }

        selected_source = _infer_source(input_path) if source == "auto" else source
        rows: list[dict[str, Any]] = []
        for index in range(n_windows):
            rows.append(
                {
                    "dataset_id": "PulseDB_v2",
                    "source": selected_source,
                    "subject_id": subject_ids[index],
                    "record_id": case_ids[index],
                    "record_order": record_order_by_case[case_ids[index]],
                    "segment_row": index,
                    "segment_uid": (
                        f"{selected_source}:{subject_ids[index]}:"
                        f"{input_path.stem}:{index:06d}"
                    ),
                    "segment_id": int(round(segment_ids[index])),
                    "win_id": int(round(win_ids[index])),
                    "win_seq_id": int(round(win_seq_ids[index])),
                    "start_time_s": float(starts[index]),
                    "end_time_s": float(ends[index]),
                    "duration_s": float(durations[index]),
                    "sample_interval_s": float(time_steps[index]),
                    "sampling_rate_hz": float(sampling_rates[index]),
                    "n_samples": int(time_lengths[index]),
                    "sbp": float(sbp[index]),
                    "dbp": float(dbp[index]),
                    "pulse_pressure": float(pulse_pressure[index]),
                    "include_flag": int(include_flag[index]),
                    "ppg_abp_corr": float(ppg_abp_corr[index]),
                    "abp_lag_samples": float(abp_lag[index]),
                    "age": float(scalar_vectors["Age"][index]) if "Age" in scalar_vectors else None,
                    "gender": genders[index],
                    "raw_file": str(input_path),
                    "raw_sha256": actual_sha256,
                    "ppg_raw_hdf5_path": target_paths["PPG_Raw"][index],
                    "ppg_f_hdf5_path": target_paths["PPG_F"][index],
                    "abp_raw_hdf5_path": target_paths["ABP_Raw"][index],
                    "abp_f_hdf5_path": target_paths["ABP_F"][index],
                    "t_hdf5_path": target_paths["T"][index],
                }
            )

        segment_index = pd.DataFrame(rows).sort_values(
            ["subject_id", "record_order", "start_time_s", "segment_id", "segment_row"],
            kind="mergesort",
        ).reset_index(drop=True)

        csv_path = output_dir / f"{stem}_segment_index.csv"
        parquet_path = output_dir / f"{stem}_segment_index.parquet"
        segment_index.to_csv(csv_path, index=False)
        segment_index.to_parquet(parquet_path, index=False)
        roundtrip = pd.read_parquet(parquet_path)
        check("segment_index_parquet_roundtrip", roundtrip.shape == segment_index.shape and list(roundtrip.columns) == list(segment_index.columns), {"written_shape": list(segment_index.shape), "read_shape": list(roundtrip.shape)})

        event_preview: dict[str, Any] = {}
        preview_input = segment_index[
            ["subject_id", "record_id", "record_order", "source", "segment_uid", "start_time_s", "sbp", "dbp"]
        ].copy()
        # ``SegmentID`` is a source field and is not guaranteed to be unique.
        # Event construction needs a stable unique row key, so expose the
        # source-qualified ``segment_uid`` under its generic input contract.
        preview_input = preview_input.rename(columns={"segment_uid": "segment_id"})
        for width in (60, 120, 300):
            events = eventize_segments(preview_input, bin_width_sec=float(width))
            event_path = output_dir / f"{stem}_events_preview_{width}s.csv"
            events.to_csv(event_path, index=False)
            n_events = int(len(events))
            event_preview[str(width)] = {
                "n_events": n_events,
                "n_support_candidates_if_max_k_5": min(n_events, 5),
                "n_common_query_events_after_event_5": max(n_events - 5, 0),
                "initial_eligibility_if_used": n_events >= 10,
                "path": str(event_path),
                "status": "pilot_only_not_for_protocol_selection",
            }

        report: dict[str, Any] = {
            "report_type": "PulseDB_v2_comprehensive_single_file_audit",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(input_path),
                "size_bytes": int(input_path.stat().st_size),
                "sha256": actual_sha256,
                "expected_sha256": expected_sha256,
                "source": selected_source,
                "field_scope": (
                    "project_fields_only" if project_fields_only else "all_fields"
                ),
            },
            "environment": {
                "python": platform.python_version(),
                "h5py": h5py.__version__,
                "hdf5_library": h5py.version.hdf5_version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "root_keys": root_keys,
            "fields": fields,
            "n_windows": n_windows,
            "reference_summaries": reference_summaries,
            "identities": identities,
            "scalar_summaries": {
                field: _summary(values) for field, values in scalar_vectors.items()
            },
            "time": {
                "sample_lengths": dict(Counter(time_lengths)),
                "sample_interval_s": _summary(time_steps),
                "sampling_rate_hz": _summary(sampling_rates),
                "duration_s": _summary(durations),
                "start_time_s": _summary(starts),
                "end_time_s": _summary(ends),
                "start_gap_s": _summary(start_gaps),
                "inter_window_gap_s": _summary(inter_window_gaps),
                "overlap_count": overlap_count,
                "touching_interval_count": touching_interval_count,
                "duplicate_interval_count": duplicate_interval_count,
            },
            "bp": {
                "sbp": _summary(sbp),
                "dbp": _summary(dbp),
                "pulse_pressure": _summary(pulse_pressure),
                "label_abp_diagnostics": label_abp_diagnostics,
            },
            "quality": {
                "include_flag_counts": {str(key): int(value) for key, value in Counter(include_flag.tolist()).items()},
                "ppg_abp_corr": _summary(ppg_abp_corr),
                "abp_lag_samples": _summary(abp_lag),
                "abp_lag_robust_outlier_indices_zero_based": lag_outlier_indices,
            },
            "waveforms": waveform_summaries,
            "waveform_pair_comparisons": pair_comparisons,
            "event_preview": event_preview,
            "checks": checks,
            "warnings": warnings,
            "interpretation_limits": [
                "PulseDB reference BP is ABP-derived; these are pseudo-cuff/reference-BP events, not literal cuff measurements.",
                "ABP, PPG_ABP_Corr, ABP_Lag, SegSBP, and SegDBP are not routine PPG-only query inputs.",
                "The event previews are parser feasibility diagnostics only and must not select the final event spacing from this subject.",
                "Participant splits must be created before development-only event-spacing selection on the full schema-valid cohort.",
            ],
            "artifacts": {
                "segment_index_csv": str(csv_path),
                "segment_index_parquet": str(parquet_path),
            },
        }

    required_failures = [
        item["name"] for item in checks if item["required"] and not item["passed"]
    ]
    optional_failures = [
        item["name"] for item in checks if not item["required"] and not item["passed"]
    ]
    status = "fail" if required_failures else ("pass_with_warnings" if warnings or optional_failures else "pass")
    report["status"] = status
    report["required_failures"] = required_failures
    report["optional_failures"] = optional_failures

    json_path = output_dir / f"{stem}_full_audit.json"
    markdown_path = output_dir / f"{stem}_full_audit.md"
    report["artifacts"].update(
        {"json": str(json_path), "markdown": str(markdown_path)}
    )
    json_path.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    time_report = report["time"]
    bp_report = report["bp"]
    quality_report = report["quality"]
    markdown_lines = [
        f"# {stem} comprehensive PulseDB audit",
        "",
        f"- Status: `{status}`",
        f"- Input: `{input_path}`",
        f"- SHA-256: `{actual_sha256}`",
        f"- Source: `{report['input']['source']}`",
        f"- Participant: `{', '.join(report['identities']['subject_ids'])}`",
        f"- Case: `{', '.join(report['identities']['case_ids'])}`",
        f"- Windows: `{report['n_windows']}`",
        "",
        "## Time and sampling",
        "",
        f"- Samples/window: `{time_report['sample_lengths']}`",
        f"- Median sampling interval: `{time_report['sample_interval_s']['median']}` s",
        f"- Median sampling rate: `{time_report['sampling_rate_hz']['median']}` Hz",
        f"- Median duration: `{time_report['duration_s']['median']}` s",
        f"- Window overlaps: `{time_report['overlap_count']}`",
        f"- Duplicate intervals: `{time_report['duplicate_interval_count']}`",
        "",
        "## BP and quality",
        "",
        f"- SBP range: `{bp_report['sbp'].get('minimum')}` to `{bp_report['sbp'].get('maximum')}` mmHg",
        f"- DBP range: `{bp_report['dbp'].get('minimum')}` to `{bp_report['dbp'].get('maximum')}` mmHg",
        f"- IncludeFlag counts: `{quality_report['include_flag_counts']}`",
        f"- PPG-ABP correlation range: `{quality_report['ppg_abp_corr'].get('minimum')}` to `{quality_report['ppg_abp_corr'].get('maximum')}`",
        f"- ABP lag robust outlier rows: `{quality_report['abp_lag_robust_outlier_indices_zero_based']}`",
        "",
        "## Required checks",
        "",
    ]
    for item in checks:
        marker = "PASS" if item["passed"] else "FAIL"
        level = "required" if item["required"] else "diagnostic"
        markdown_lines.append(f"- `{marker}` ({level}) {item['name']}")
    markdown_lines.extend(["", "## Pilot-only event preview", ""])
    for width, item in report["event_preview"].items():
        markdown_lines.append(
            f"- `{width}s`: {item['n_events']} events; "
            f"{item['n_common_query_events_after_event_5']} common queries after event 5"
        )
    markdown_lines.extend(["", "## Warnings and limits", ""])
    for warning in warnings:
        markdown_lines.append(f"- {warning}")
    for limit in report["interpretation_limits"]:
        markdown_lines.append(f"- {limit}")
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(f"FULL_AUDIT_STATUS={status}")
    print(f"N_WINDOWS={report['n_windows']}")
    print(f"SUBJECT_IDS={','.join(report['identities']['subject_ids'])}")
    print(f"CASE_IDS={','.join(report['identities']['case_ids'])}")
    print(f"SAMPLING_RATE_HZ={time_report['sampling_rate_hz']['median']}")
    print(f"WINDOW_DURATION_S={time_report['duration_s']['median']}")
    print(f"OVERLAP_COUNT={time_report['overlap_count']}")
    print(f"DUPLICATE_INTERVAL_COUNT={time_report['duplicate_interval_count']}")
    print(f"REQUIRED_FAILURES={','.join(required_failures) if required_failures else 'none'}")
    print(f"WARNING_COUNT={len(warnings)}")
    for width, item in report["event_preview"].items():
        print(f"EVENT_PREVIEW_{width}S={item['n_events']}")
    print(f"FULL_AUDIT_JSON={json_path}")
    print(f"FULL_AUDIT_MARKDOWN={markdown_path}")
    print(f"SEGMENT_INDEX_CSV={report['artifacts']['segment_index_csv']}")
    print(f"SEGMENT_INDEX_PARQUET={report['artifacts']['segment_index_parquet']}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--source", default="auto")
    parser.add_argument("--project-fields-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_pulsedb_file(
        args.input,
        args.output,
        expected_sha256=args.expected_sha256,
        source=args.source,
        project_fields_only=args.project_fields_only,
    )
    if report["status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
