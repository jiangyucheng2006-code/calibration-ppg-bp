"""Temporal event construction and K-shot episode assignment."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_SEGMENT_COLUMNS = {
    "subject_id",
    "record_id",
    "record_order",
    "source",
    "segment_id",
    "start_time_s",
    "sbp",
    "dbp",
}


def _validate_segments(segments: pd.DataFrame) -> None:
    missing = REQUIRED_SEGMENT_COLUMNS.difference(segments.columns)
    if missing:
        raise ValueError(f"Missing required segment columns: {sorted(missing)}")
    if segments.empty:
        raise ValueError("Segment table is empty")
    if segments["subject_id"].isna().any():
        raise ValueError("subject_id contains missing values")
    if segments["segment_id"].duplicated().any():
        duplicates = segments.loc[segments["segment_id"].duplicated(), "segment_id"]
        raise ValueError(f"segment_id must be unique; duplicates include {duplicates.iloc[0]!r}")
    numeric = segments[["record_order", "start_time_s", "sbp", "dbp"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("record_order, start_time_s, sbp, and dbp must be finite")
    if (segments["start_time_s"] < 0).any():
        raise ValueError("start_time_s must be non-negative")


def eventize_segments(segments: pd.DataFrame, *, bin_width_sec: float) -> pd.DataFrame:
    """Convert segment rows into deterministic, temporally separated events.

    Bins are anchored to the first available segment in each subject-record pair.
    Exactly one segment, closest to the centre of the bin, represents each event.
    BP labels are copied from that representative segment; labels from other
    segments in the bin are not averaged into the event.
    """

    _validate_segments(segments)
    if not np.isfinite(bin_width_sec) or bin_width_sec <= 0:
        raise ValueError("bin_width_sec must be a positive finite number")

    work = segments.copy()
    group_keys = ["subject_id", "record_id"]
    record_start = work.groupby(group_keys, sort=False)["start_time_s"].transform("min")
    relative_time = work["start_time_s"] - record_start
    work["time_bin"] = np.floor(relative_time / bin_width_sec).astype("int64")
    work["bin_start_s"] = record_start + work["time_bin"] * bin_width_sec
    work["distance_to_bin_center"] = (
        work["start_time_s"] - (work["bin_start_s"] + bin_width_sec / 2.0)
    ).abs()

    event_keys = ["subject_id", "record_id", "time_bin"]
    work = work.sort_values(
        event_keys + ["distance_to_bin_center", "start_time_s", "segment_id"],
        kind="mergesort",
    )
    representative = work.groupby(event_keys, sort=False, as_index=False).first()
    counts = work.groupby(event_keys, sort=False).size().rename("n_segments_in_bin").reset_index()
    events = representative.merge(counts, on=event_keys, how="left", validate="one_to_one")

    events = events.sort_values(
        ["subject_id", "record_order", "start_time_s", "record_id", "time_bin"],
        kind="mergesort",
    ).reset_index(drop=True)
    events["event_index"] = events.groupby("subject_id", sort=False).cumcount() + 1
    events["event_id"] = (
        events["subject_id"].astype(str)
        + "::"
        + events["event_index"].astype(str).str.zfill(4)
    )

    keep = [
        "event_id",
        "subject_id",
        "source",
        "record_id",
        "record_order",
        "event_index",
        "time_bin",
        "bin_start_s",
        "start_time_s",
        "segment_id",
        "sbp",
        "dbp",
        "n_segments_in_bin",
    ]
    return events[keep]


def summarize_eligibility(
    events: pd.DataFrame,
    *,
    max_k: int = 5,
    min_query_events: int = 5,
) -> pd.DataFrame:
    """Summarize event counts and common-query eligibility per subject."""

    if max_k < 1 or min_query_events < 1:
        raise ValueError("max_k and min_query_events must be positive integers")
    required = {"subject_id", "source", "event_id", "sbp", "dbp"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required event columns: {sorted(missing)}")

    summary = (
        events.groupby(["subject_id", "source"], as_index=False)
        .agg(
            n_events=("event_id", "size"),
            sbp_min=("sbp", "min"),
            sbp_max=("sbp", "max"),
            sbp_sd=("sbp", "std"),
            dbp_min=("dbp", "min"),
            dbp_max=("dbp", "max"),
            dbp_sd=("dbp", "std"),
        )
    )
    summary["n_support_candidates"] = np.minimum(summary["n_events"], max_k)
    summary["n_common_query_events"] = np.maximum(summary["n_events"] - max_k, 0)
    summary["eligible"] = summary["n_common_query_events"] >= min_query_events
    summary["required_events"] = max_k + min_query_events
    summary["sbp_range"] = summary["sbp_max"] - summary["sbp_min"]
    summary["dbp_range"] = summary["dbp_max"] - summary["dbp_min"]
    return summary


def build_episode_assignments(
    events: pd.DataFrame,
    *,
    ks: Iterable[int] = (1, 2, 3, 5),
    min_query_events: int = 5,
) -> pd.DataFrame:
    """Assign support, unused-pool, and common-query roles for every K.

    All K values share the query set starting after ``max(ks)``. Subjects with
    fewer than ``max(ks) + min_query_events`` events are excluded.
    """

    ks = tuple(sorted(set(int(k) for k in ks)))
    if not ks or ks[0] < 1:
        raise ValueError("ks must contain positive integers")
    max_k = max(ks)
    eligibility = summarize_eligibility(
        events,
        max_k=max_k,
        min_query_events=min_query_events,
    )
    eligible_subjects = set(eligibility.loc[eligibility["eligible"], "subject_id"])
    eligible_events = events[events["subject_id"].isin(eligible_subjects)].copy()
    if eligible_events.empty:
        columns = list(events.columns) + ["k", "role"]
        return pd.DataFrame(columns=columns)

    assignments: list[pd.DataFrame] = []
    for k in ks:
        frame = eligible_events.copy()
        frame["k"] = k
        frame["role"] = np.select(
            [frame["event_index"] <= k, frame["event_index"] <= max_k],
            ["support", "unused_calibration_pool"],
            default="query",
        )
        assignments.append(frame)
    result = pd.concat(assignments, ignore_index=True)
    return result.sort_values(["subject_id", "k", "event_index"], kind="mergesort")
