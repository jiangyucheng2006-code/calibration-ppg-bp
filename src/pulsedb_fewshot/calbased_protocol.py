"""Development-only same-subject analogue of the PulseDB CalBased protocol.

This module deliberately does not claim to reproduce the official PulseDB
``Train_Info``/``CalBased_Test_Info`` assignment.  It creates a controlled
same-subject benchmark from the project's already frozen ``meta_train``
participants only.  No ``meta_validation`` or locked ``meta_test`` waveform is
permitted to enter this protocol.

Each eligible participant contributes exactly 400 mutually exclusive 10-second
PPG windows: 320 for fitting, 40 for internal validation/model selection, and
40 for one-time held-out evaluation.  The held-out inputs and targets are
written separately so ordinary training code never needs to load held-out
labels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROTOCOL_ID = "development-calbased-analogue-v1"
ROLES = ("train", "internal_validation", "heldout_test")
SPLIT_MODES = ("random_disjoint", "chronological_blocked")
DEFAULT_SEED = 20260828
DEFAULT_WINDOWS = (320, 40, 40)
EXPECTED_META_TRAIN_SUBJECTS = 2058

TARGET_COLUMNS = {"sbp", "dbp", "pulse_pressure", "target_sbp", "target_dbp"}
NON_PPG_MODEL_COLUMNS = {
    "age",
    "gender",
    "sex",
    "height",
    "weight",
    "bmi",
    "ppg_abp_corr",
    "abp_lag_samples",
}
REQUIRED_SEGMENT_COLUMNS = {
    "source",
    "subject_uid",
    "record_id",
    "segment_row",
    "segment_uid",
    "start_time_s",
    "duration_s",
    "sampling_rate_hz",
    "n_samples",
    "sbp",
    "dbp",
    "raw_file",
    "raw_file_sha256",
    "ppg_field",
    "ppg_reference_index",
    "segment_schema_valid",
}


@dataclass(frozen=True)
class CalBasedAnalogueArtifacts:
    """In-memory artifacts with held-out targets physically separated."""

    development_fit_manifest: pd.DataFrame
    heldout_test_inputs: pd.DataFrame
    heldout_test_targets: pd.DataFrame
    role_manifest: pd.DataFrame
    audit: dict[str, Any]


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _stable_digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pair_name(left: str, right: str) -> str:
    order = {role: index for index, role in enumerate(ROLES)}
    first, second = sorted((left, right), key=order.__getitem__)
    return f"{first}__{second}"


def _validate_subject_splits(subject_splits: pd.DataFrame) -> set[str]:
    required = {"subject_uid", "split"}
    missing = required.difference(subject_splits.columns)
    if missing:
        raise ValueError(f"subject split file is missing columns: {sorted(missing)}")
    if subject_splits["subject_uid"].duplicated().any():
        raise AssertionError("subject split assignments are not unique")
    meta_train = set(
        subject_splits.loc[
            subject_splits["split"].eq("meta_train"), "subject_uid"
        ].astype(str)
    )
    if not meta_train:
        raise ValueError("subject split file contains no meta_train participants")
    return meta_train


def load_frozen_meta_train_segments(
    segment_index_path: Path,
    subject_splits_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load only frozen ``meta_train`` rows from a Parquet segment index.

    A Parquet predicate is mandatory.  The function never falls back to reading
    the complete development/full-cohort table, because that would expose
    protected participant windows to this secondary benchmark.
    """

    subject_splits = pd.read_csv(subject_splits_path)
    meta_train = _validate_subject_splits(subject_splits)
    schema_names = set(pq.ParquetFile(segment_index_path).schema.names)
    if "split" in schema_names:
        filters: list[tuple[str, str, object]] = [("split", "==", "meta_train")]
        filter_description = "split == meta_train"
    elif "subject_uid" in schema_names:
        filters = [("subject_uid", "in", sorted(meta_train))]
        filter_description = "subject_uid in frozen meta_train IDs"
    else:
        raise ValueError("segment index has neither split nor subject_uid for safe filtering")

    segments = pd.read_parquet(segment_index_path, filters=filters)
    observed = set(segments["subject_uid"].astype(str))
    protected = observed.difference(meta_train)
    if protected:
        raise AssertionError(
            "predicate returned protected non-meta_train participants: "
            + ", ".join(sorted(protected)[:10])
        )
    return segments, subject_splits, filter_description


def _waveform_locator_keys(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    if "ppg_waveform_sha256" in frame and frame["ppg_waveform_sha256"].notna().all():
        return frame["ppg_waveform_sha256"].astype(str), "ppg_waveform_sha256"

    locator_options = (
        ("raw_file_sha256", "ppg_field", "ppg_reference_index"),
        ("raw_file", "ppg_field", "ppg_reference_index"),
    )
    for columns in locator_options:
        if set(columns).issubset(frame.columns) and frame[list(columns)].notna().all().all():
            values = frame[list(columns)].astype(str).agg("\x1f".join, axis=1)
            keys = values.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
            return keys, "storage_locator_sha256:" + "+".join(columns)
    raise ValueError(
        "cannot audit duplicate waveform locators without ppg_waveform_sha256 "
        "or a complete PPG locator"
    )


def _prepare_candidates(
    meta_train_segments: pd.DataFrame,
    subject_splits: pd.DataFrame,
    *,
    require_include_flag: bool,
    duration_seconds: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = REQUIRED_SEGMENT_COLUMNS.difference(meta_train_segments.columns)
    if missing:
        raise ValueError(f"segment index is missing columns: {sorted(missing)}")
    if meta_train_segments["segment_uid"].duplicated().any():
        raise AssertionError("segment_uid is not unique in the meta_train input")

    meta_train_ids = _validate_subject_splits(subject_splits)
    observed = set(meta_train_segments["subject_uid"].astype(str))
    protected = observed.difference(meta_train_ids)
    if protected:
        raise AssertionError(
            "protocol input contains meta_validation or locked meta_test participants: "
            + ", ".join(sorted(protected)[:10])
        )
    if "split" in meta_train_segments and not meta_train_segments["split"].eq("meta_train").all():
        raise AssertionError("protocol input contains rows whose split is not meta_train")

    candidates = meta_train_segments.copy()
    finite_columns = ["start_time_s", "duration_s", "sampling_rate_hz", "sbp", "dbp"]
    finite = np.isfinite(candidates[finite_columns].astype(float)).all(axis=1)
    duration_ok = np.isclose(
        candidates["duration_s"].astype(float), duration_seconds, atol=0.02, rtol=0.0
    )
    schema_ok = candidates["segment_schema_valid"].fillna(False).astype(bool)
    keep = finite & duration_ok & schema_ok
    exclusion_counts = {
        "nonfinite_required_numeric": int((~finite).sum()),
        "not_ten_second_window": int((~duration_ok).sum()),
        "segment_schema_invalid": int((~schema_ok).sum()),
    }
    if require_include_flag:
        if "include_flag" not in candidates:
            raise ValueError("require_include_flag=True but include_flag is unavailable")
        include_ok = candidates["include_flag"].eq(1)
        keep &= include_ok
        exclusion_counts["include_flag_not_one"] = int((~include_ok).sum())

    candidates = candidates.loc[keep].copy()
    candidates["subject_uid"] = candidates["subject_uid"].astype(str)
    candidates["source"] = candidates["source"].astype(str)
    return candidates, exclusion_counts


def _select_subject_windows(
    candidates: pd.DataFrame,
    *,
    split_mode: str,
    seed: int,
    n_train: int,
    n_validation: int,
    n_test: int,
) -> tuple[pd.DataFrame, int]:
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}")
    total = n_train + n_validation + n_test
    if min(n_train, n_validation, n_test) <= 0:
        raise ValueError("all role window counts must be positive")

    counts = candidates.groupby("subject_uid", sort=True).size()
    eligible_subjects = set(counts[counts >= total].index)
    selected_parts: list[pd.DataFrame] = []
    for subject_uid, subject in candidates.loc[
        candidates["subject_uid"].isin(eligible_subjects)
    ].groupby("subject_uid", sort=True):
        subject = subject.copy()
        if split_mode == "random_disjoint":
            subject["selection_key"] = subject["segment_uid"].map(
                lambda segment_uid: _stable_digest(
                    PROTOCOL_ID, split_mode, seed, subject_uid, segment_uid
                )
            )
            subject = subject.sort_values(
                ["selection_key", "segment_uid"], kind="mergesort"
            ).iloc[:total]
        else:
            sort_columns = [
                column
                for column in (
                    "record_order",
                    "record_id",
                    "start_time_s",
                    "segment_row",
                    "segment_uid",
                )
                if column in subject.columns
            ]
            subject = subject.sort_values(sort_columns, kind="mergesort").iloc[:total]
            subject["selection_key"] = subject["segment_uid"].astype(str)

        subject = subject.reset_index(drop=True)
        subject["selection_rank"] = np.arange(1, total + 1)
        subject["role"] = np.select(
            [
                subject["selection_rank"].le(n_train),
                subject["selection_rank"].le(n_train + n_validation),
            ],
            ["train", "internal_validation"],
            default="heldout_test",
        )
        selected_parts.append(subject)

    if not selected_parts:
        raise ValueError(f"no meta_train participant has at least {total} eligible windows")
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["protocol_id"] = PROTOCOL_ID
    selected["split_mode"] = split_mode
    selected["protocol_seed"] = seed
    selected["original_split"] = "meta_train"
    return selected, int((counts < total).sum())


def _role_sets(selected: pd.DataFrame, column: str) -> dict[str, set[str]]:
    return {
        role: set(selected.loc[selected["role"].eq(role), column].astype(str))
        for role in ROLES
    }


def _pairwise_intersections(sets: dict[str, set[Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for left_index, left in enumerate(ROLES):
        for right in ROLES[left_index + 1 :]:
            result[_pair_name(left, right)] = len(sets[left] & sets[right])
    return result


def _interval_audit(selected: pd.DataFrame) -> tuple[dict[str, int], dict[str, Any]]:
    overlap_counts = {
        _pair_name(left, right): 0
        for index, left in enumerate(ROLES)
        for right in ROLES[index + 1 :]
    }
    gap_values: dict[str, list[float]] = {key: [] for key in overlap_counts}
    boundary_touching = {key: 0 for key in overlap_counts}
    adjacent_segment_rows = {key: 0 for key in overlap_counts}
    tolerance = 1e-9

    group_columns = ["subject_uid", "record_id"]
    for _, group in selected.groupby(group_columns, sort=False, dropna=False):
        work = group.copy()
        work["interval_end_exclusive"] = (
            work["start_time_s"].astype(float) + work["duration_s"].astype(float)
        )
        work = work.sort_values(
            ["start_time_s", "interval_end_exclusive", "segment_uid"], kind="mergesort"
        ).reset_index(drop=True)

        active: list[tuple[float, str]] = []
        for row in work.itertuples(index=False):
            start = float(row.start_time_s)
            active = [(end, role) for end, role in active if end > start + tolerance]
            for _, other_role in active:
                if other_role != row.role:
                    overlap_counts[_pair_name(other_role, row.role)] += 1
            active.append((float(row.interval_end_exclusive), str(row.role)))

        for previous, current in zip(work.iloc[:-1].itertuples(), work.iloc[1:].itertuples()):
            if previous.role == current.role:
                continue
            pair = _pair_name(str(previous.role), str(current.role))
            gap = float(current.start_time_s - previous.interval_end_exclusive)
            if gap >= -tolerance:
                nonnegative_gap = max(0.0, gap)
                gap_values[pair].append(nonnegative_gap)
                if nonnegative_gap <= tolerance:
                    boundary_touching[pair] += 1
            if abs(int(current.segment_row) - int(previous.segment_row)) == 1:
                adjacent_segment_rows[pair] += 1

    summaries: dict[str, Any] = {}
    for pair, values in gap_values.items():
        array = np.asarray(values, dtype=float)
        summaries[pair] = {
            "consecutive_cross_role_pairs": int(array.size),
            "minimum_gap_seconds": float(array.min()) if array.size else None,
            "p05_gap_seconds": float(np.quantile(array, 0.05)) if array.size else None,
            "median_gap_seconds": float(np.median(array)) if array.size else None,
            "boundary_touching_pairs": int(boundary_touching[pair]),
            "adjacent_segment_row_pairs": int(adjacent_segment_rows[pair]),
        }
    return overlap_counts, summaries


def audit_calbased_analogue(
    selected: pd.DataFrame,
    *,
    expected_subjects: int,
    n_train: int,
    n_validation: int,
    n_test: int,
) -> dict[str, Any]:
    """Audit role, target, interval, waveform, and source invariants."""

    failures: list[str] = []
    if selected.groupby("subject_uid")["source"].nunique().gt(1).any():
        failures.append("participant_source_changes_across_rows")
    subjects = _role_sets(selected, "subject_uid")
    subject_overlap = _pairwise_intersections(subjects)
    if any(len(subjects[role]) != expected_subjects for role in ROLES):
        failures.append("not_every_eligible_subject_present_in_every_role")
    if any(value != expected_subjects for value in subject_overlap.values()):
        failures.append("same_subject_role_overlap_not_complete")

    windows = _role_sets(selected, "segment_uid")
    window_overlap = _pairwise_intersections(windows)
    if any(window_overlap.values()):
        failures.append("segment_uid_overlap_across_roles")

    record_sets = {
        role: set(
            zip(
                selected.loc[selected["role"].eq(role), "subject_uid"].astype(str),
                selected.loc[selected["role"].eq(role), "record_id"].astype(str),
            )
        )
        for role in ROLES
    }
    record_overlap = _pairwise_intersections(record_sets)

    locators = _role_sets(selected, "waveform_locator_key")
    locator_overlap = _pairwise_intersections(locators)
    if any(locator_overlap.values()):
        failures.append("waveform_locator_overlap_across_roles")

    interval_overlap, adjacent_gaps = _interval_audit(selected)
    if any(interval_overlap.values()):
        failures.append("raw_interval_overlap_across_roles")

    expected_counts = {
        "train": n_train,
        "internal_validation": n_validation,
        "heldout_test": n_test,
    }
    actual_per_subject = (
        selected.groupby(["subject_uid", "role"]).size().unstack(fill_value=0)
    )
    for role, expected in expected_counts.items():
        if role not in actual_per_subject or not actual_per_subject[role].eq(expected).all():
            failures.append(f"per_subject_window_count_invalid:{role}")

    source_counts = (
        selected.groupby(["source", "role"], sort=True)
        .agg(windows=("segment_uid", "size"), subjects=("subject_uid", "nunique"))
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "status": "pass" if not failures else "fail",
        "protocol_id": PROTOCOL_ID,
        "failures": failures,
        "benchmark_scope": "development-only same-subject analogue; not official PulseDB CalBased",
        "official_train_info_used": False,
        "official_calbased_test_info_used": False,
        "original_subject_split": "meta_train only",
        "meta_validation_windows_accessed": False,
        "locked_meta_test_windows_accessed": False,
        "roles": list(ROLES),
        "model_selection_role": "internal_validation",
        "heldout_test_used_for_early_stopping": False,
        "post_selection_refit_roles": ["train", "internal_validation"],
        "post_selection_refit_windows_per_subject": n_train + n_validation,
        "heldout_test_evaluations_allowed": 1,
        "eligible_subjects": expected_subjects,
        "selected_windows": int(len(selected)),
        "per_subject_windows": expected_counts,
        "subject_overlap_expected": True,
        "subject_overlap_counts": subject_overlap,
        "window_overlap_counts": window_overlap,
        "subject_record_overlap_counts": record_overlap,
        "raw_interval_overlap_counts": interval_overlap,
        "waveform_locator_overlap_counts": locator_overlap,
        "waveform_locator_key_method": str(
            selected["waveform_locator_key_method"].iloc[0]
        ),
        "exact_waveform_content_overlap_audited": False,
        "exact_waveform_content_overlap_audit_stage": "PPG materialization",
        "adjacent_time_gap_audit": adjacent_gaps,
        "source_counts": source_counts,
    }


def _model_input_columns(selected: pd.DataFrame) -> list[str]:
    preferred = [
        "protocol_id",
        "split_mode",
        "protocol_seed",
        "original_split",
        "role",
        "selection_rank",
        "dataset_id",
        "source",
        "source_directory",
        "subject_id",
        "subject_uid",
        "record_id",
        "record_order",
        "segment_row",
        "segment_uid",
        "segment_id",
        "win_id",
        "win_seq_id",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "sample_interval_s",
        "sampling_rate_hz",
        "n_samples",
        "raw_file",
        "raw_file_relative_path",
        "raw_file_sha256",
        "ppg_field",
        "ppg_storage_mode",
        "ppg_reference_index",
        "ppg_f_mean",
        "ppg_f_std",
        "waveform_locator_key",
        "waveform_locator_key_method",
    ]
    columns = [column for column in preferred if column in selected.columns]
    forbidden = TARGET_COLUMNS | NON_PPG_MODEL_COLUMNS
    if forbidden & set(columns) or any(
        column.lower().startswith(("ecg", "abp")) for column in columns
    ):
        raise AssertionError("non-PPG feature or target leaked into model-input projection")
    return columns


def build_calbased_analogue(
    meta_train_segments: pd.DataFrame,
    subject_splits: pd.DataFrame,
    *,
    split_mode: str = "random_disjoint",
    seed: int = DEFAULT_SEED,
    n_train: int = DEFAULT_WINDOWS[0],
    n_validation: int = DEFAULT_WINDOWS[1],
    n_test: int = DEFAULT_WINDOWS[2],
    require_include_flag: bool = False,
    strict: bool = True,
) -> CalBasedAnalogueArtifacts:
    """Build the leakage-audited development-only same-subject protocol."""

    candidates, exclusion_counts = _prepare_candidates(
        meta_train_segments,
        subject_splits,
        require_include_flag=require_include_flag,
        duration_seconds=10.0,
    )
    selected, insufficient_subjects = _select_subject_windows(
        candidates,
        split_mode=split_mode,
        seed=seed,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
    )
    # Compute locator hashes only for the fixed 400-window cohort, not for the
    # millions of otherwise eligible raw windows.
    selected["waveform_locator_key"], locator_method = _waveform_locator_keys(
        selected
    )
    selected["waveform_locator_key_method"] = locator_method
    eligible_subjects = int(selected["subject_uid"].nunique())
    audit = audit_calbased_analogue(
        selected,
        expected_subjects=eligible_subjects,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
    )
    audit.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "split_mode": split_mode,
            "seed": seed,
            "require_include_flag": require_include_flag,
            "meta_train_subjects_present_in_input": int(
                meta_train_segments["subject_uid"].nunique()
            ),
            "candidate_windows": int(len(candidates)),
            "subjects_with_fewer_than_400_eligible_windows": insufficient_subjects,
            "candidate_exclusion_counts": exclusion_counts,
        }
    )
    if strict and audit["status"] != "pass":
        raise AssertionError("CalBased analogue audit failed: " + ", ".join(audit["failures"]))

    input_columns = _model_input_columns(selected)
    fit = selected.loc[
        selected["role"].isin(["train", "internal_validation"]),
        input_columns + ["sbp", "dbp"],
    ].reset_index(drop=True)
    heldout_inputs = selected.loc[
        selected["role"].eq("heldout_test"), input_columns
    ].reset_index(drop=True)
    heldout_targets = selected.loc[
        selected["role"].eq("heldout_test"),
        ["protocol_id", "source", "subject_uid", "segment_uid", "role", "sbp", "dbp"],
    ].reset_index(drop=True)
    role_manifest = selected[
        [
            "protocol_id",
            "split_mode",
            "protocol_seed",
            "source",
            "subject_uid",
            "segment_uid",
            "role",
            "selection_rank",
        ]
    ].reset_index(drop=True)

    if TARGET_COLUMNS & set(heldout_inputs.columns):
        raise AssertionError("heldout input manifest exposes BP targets")
    if set(heldout_targets["segment_uid"]) & set(fit["segment_uid"]):
        raise AssertionError("heldout targets overlap development fitting windows")
    return CalBasedAnalogueArtifacts(fit, heldout_inputs, heldout_targets, role_manifest, audit)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_calbased_analogue(
    artifacts: CalBasedAnalogueArtifacts,
    output_root: Path,
) -> dict[str, str]:
    """Persist development and held-out artifacts into separate locations."""

    output_root = output_root.resolve()
    paths = {
        "development_fit_manifest": output_root / "development_fit_manifest.parquet",
        "role_manifest": output_root / "role_manifest.parquet",
        "heldout_test_inputs": output_root / "heldout" / "heldout_test_inputs.parquet",
        "heldout_test_targets": output_root / "locked" / "heldout_test_targets.parquet",
        "audit": output_root / "protocol_audit.json",
    }
    _atomic_parquet(paths["development_fit_manifest"], artifacts.development_fit_manifest)
    _atomic_parquet(paths["role_manifest"], artifacts.role_manifest)
    _atomic_parquet(paths["heldout_test_inputs"], artifacts.heldout_test_inputs)
    _atomic_parquet(paths["heldout_test_targets"], artifacts.heldout_test_targets)
    _atomic_json(paths["audit"], artifacts.audit)
    return {key: str(path) for key, path in paths.items()}


def run_calbased_analogue(
    segment_index_path: Path,
    subject_splits_path: Path,
    output_root: Path,
    *,
    split_mode: str = "random_disjoint",
    seed: int = DEFAULT_SEED,
    require_include_flag: bool = False,
    expected_subjects: int | None = EXPECTED_META_TRAIN_SUBJECTS,
) -> dict[str, Any]:
    """Safely load frozen meta-train rows, build the protocol, and write it."""

    segments, splits, loading_filter = load_frozen_meta_train_segments(
        segment_index_path, subject_splits_path
    )
    artifacts = build_calbased_analogue(
        segments,
        splits,
        split_mode=split_mode,
        seed=seed,
        require_include_flag=require_include_flag,
    )
    observed_subjects = int(artifacts.audit["eligible_subjects"])
    artifacts.audit["expected_meta_train_subjects"] = expected_subjects
    if expected_subjects is not None and observed_subjects != expected_subjects:
        raise AssertionError(
            "eligible same-subject cohort does not match the frozen expectation: "
            f"{observed_subjects} != {expected_subjects}"
        )
    artifacts.audit["input_loading_filter"] = loading_filter
    artifacts.audit["segment_index_path"] = str(segment_index_path.resolve())
    artifacts.audit["subject_splits_path"] = str(subject_splits_path.resolve())
    paths = write_calbased_analogue(artifacts, output_root)
    return {"audit": artifacts.audit, "artifacts": paths}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-index", required=True, type=Path)
    parser.add_argument("--subject-splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-mode", choices=SPLIT_MODES, default="random_disjoint")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-subjects", type=int, default=EXPECTED_META_TRAIN_SUBJECTS
    )
    parser.add_argument("--require-include-flag", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_calbased_analogue(
        args.segment_index,
        args.subject_splits,
        args.output,
        split_mode=args.split_mode,
        seed=args.seed,
        require_include_flag=args.require_include_flag,
        expected_subjects=args.expected_subjects,
    )
    print(json.dumps(_json_value(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
