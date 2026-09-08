"""Recorded-sample-span contract for the new official CalBased benchmark.

PulseDB's ``end_time_s`` is T[-1], while ``duration_s`` includes one nominal
sample interval. Extending T[-1] by dt manufactured an 8-ms overlap at four
official role boundaries. The official pipeline preserves T[0] and T[-1]
instead. Exact touching timestamps are reported, not called independent
samples. Duplicate IDs/content and every positive-duration overlap above
floating-point tolerance still require their separate gates.

Only this new protocol imports the helper; historical experiments are intact.
The DataFrame is duck-typed so importing this module needs only the stdlib.
"""
from __future__ import annotations

import math


TIME_BOUNDARY_POLICY = "recorded-sample-span-v1"
TIME_TOLERANCE_S = 1e-7


def sample_span_audit(frame, role: str, *, group_cols=("source", "record_id"),
                      start_col="start_time_s", end_col="end_time_s",
                      id_col="segment_uid") -> dict:
    """Count cross-role overlaps/touches using actual first/last sample times.

    Group globally by source and recording, not person: relabelling a person
    must not hide physical recording reuse. The caller supplies canonical
    recording identity and is responsible for checking its clock provenance.
    One-point touching is distinct from a positive recorded-span overlap.
    """
    needed = {*group_cols, role, start_col, end_col, id_col}
    missing = needed - set(frame)
    if missing:
        raise ValueError(f"sample-span audit missing {sorted(missing)}")
    if frame[list(needed)].isna().any().any():
        raise ValueError("sample-span audit requires complete lineage")
    overlaps = touches = 0
    for _, group in frame.groupby(list(group_cols), sort=False):
        active = []
        ordered = group.sort_values([start_col, end_col, id_col], kind="mergesort")
        for start, end, label in ordered[[start_col, end_col, role]].itertuples(index=False, name=None):
            start, end = float(start), float(end)
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError("finite positive recorded sample span required")
            active = [(previous_end, previous_role) for previous_end, previous_role in active
                      if previous_end >= start - TIME_TOLERANCE_S]
            for previous_end, previous_role in active:
                if previous_role == label:
                    continue
                overlap = min(previous_end, end) - start
                if overlap > TIME_TOLERANCE_S:
                    overlaps += 1
                elif abs(overlap) <= TIME_TOLERANCE_S:
                    touches += 1
            active.append((end, label))
    return {"time_boundary_policy": TIME_BOUNDARY_POLICY,
            "time_tolerance_s": TIME_TOLERANCE_S,
            "cross_role_overlap_pairs": overlaps,
            "cross_role_touching_pairs": touches}
