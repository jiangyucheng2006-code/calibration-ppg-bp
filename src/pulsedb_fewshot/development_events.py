"""Development-only event-spacing and eligibility feasibility audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .events import eventize_segments, summarize_eligibility


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def run_development_event_audit(
    development_segments_path: Path,
    subject_splits_path: Path,
    output_root: Path,
    *,
    widths: tuple[int, ...] = (60, 120, 300),
    min_query_events: int = 5,
) -> dict[str, object]:
    segments = pd.read_parquet(development_segments_path)
    splits = pd.read_csv(subject_splits_path)
    if "meta_test" in set(segments["split"]):
        raise AssertionError("development event audit received locked meta-test rows")
    if set(segments["split"]) - {"meta_train", "meta_validation"}:
        raise AssertionError("unexpected split labels in development segments")
    if not segments["segment_schema_valid"].all():
        raise AssertionError("development segments must already be schema-valid")

    development_subjects = set(
        splits.loc[splits["split"].isin(["meta_train", "meta_validation"]), "subject_uid"]
    )
    if set(segments["subject_uid"]) - development_subjects:
        raise AssertionError("development rows include a subject outside development splits")

    before_quality_rows = len(segments)
    quality = segments[segments["include_flag"] == 1].copy()
    subjects_lost_to_quality = development_subjects - set(quality["subject_uid"])
    input_columns = [
        "subject_uid",
        "record_id",
        "record_order",
        "source",
        "segment_uid",
        "start_time_s",
        "sbp",
        "dbp",
        "split",
    ]
    quality = quality[input_columns].rename(
        columns={"subject_uid": "subject_id", "segment_uid": "segment_id"}
    )
    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    split_map = splits.set_index("subject_uid")["split"].to_dict()

    for width in widths:
        events = eventize_segments(quality, bin_width_sec=float(width))
        events["split"] = events["subject_id"].map(split_map)
        if events["split"].eq("meta_test").any() or events["split"].isna().any():
            raise AssertionError("locked or unassigned subjects entered event audit")
        eligibility = summarize_eligibility(
            events,
            max_k=5,
            min_query_events=min_query_events,
        )
        eligibility["split"] = eligibility["subject_id"].map(split_map)
        eligible = eligibility[eligibility["eligible"]].copy()
        events_path = output_root / f"development_events_{width}s.parquet"
        eligibility_path = output_root / f"development_eligibility_{width}s.parquet"
        eligible_path = output_root / f"eligible_development_subjects_{width}s.csv"
        events.to_parquet(events_path, index=False)
        eligibility.to_parquet(eligibility_path, index=False)
        eligible.to_csv(eligible_path, index=False)

        group_counts = (
            eligibility.groupby(["split", "source"], dropna=False)
            .agg(
                n_subjects=("subject_id", "nunique"),
                n_eligible=("eligible", "sum"),
                total_events=("n_events", "sum"),
                total_common_queries=("n_common_query_events", "sum"),
            )
            .reset_index()
        )
        group_counts["retention_fraction"] = (
            group_counts["n_eligible"] / group_counts["n_subjects"]
        )
        results[str(width)] = {
            "n_events": int(len(events)),
            "n_subjects_with_quality_events": int(eligibility["subject_id"].nunique()),
            "n_eligible_development_subjects": int(eligible["subject_id"].nunique()),
            "event_count_distribution": _quantiles(eligibility["n_events"]),
            "common_query_count_distribution": _quantiles(
                eligibility["n_common_query_events"]
            ),
            "eligible_sbp_range_distribution": _quantiles(eligible["sbp_range"]),
            "eligible_dbp_range_distribution": _quantiles(eligible["dbp_range"]),
            "group_counts": group_counts.to_dict("records"),
            "artifacts": {
                "events": str(events_path),
                "events_sha256": _sha256(events_path),
                "eligibility": str(eligibility_path),
                "eligibility_sha256": _sha256(eligibility_path),
                "eligible_subjects": str(eligible_path),
                "eligible_subjects_sha256": _sha256(eligible_path),
            },
        }

    report: dict[str, object] = {
        "status": "development_feasibility_complete_no_spacing_selected",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_segments": str(development_segments_path.resolve()),
        "development_segments_sha256": _sha256(development_segments_path),
        "subject_splits": str(subject_splits_path.resolve()),
        "subject_splits_sha256": _sha256(subject_splits_path),
        "widths_seconds": list(widths),
        "max_k": 5,
        "min_query_events": min_query_events,
        "quality_rule": "segment_schema_valid and IncludeFlag == 1",
        "n_rows_before_include_flag": int(before_quality_rows),
        "n_rows_after_include_flag": int(len(quality)),
        "n_development_subjects": len(development_subjects),
        "n_subjects_without_include_flag_1_rows": len(subjects_lost_to_quality),
        "locked_test_access": "not eventized; eligibility and BP distributions not inspected",
        "selection_status": "not selected; compare development results and freeze before applying to locked test",
        "results": results,
    }
    report_path = output_root / "development_event_feasibility.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-segments", required=True, type=Path)
    parser.add_argument("--subject-splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--widths", nargs="+", type=int, default=[60, 120, 300])
    parser.add_argument("--min-query-events", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_development_event_audit(
        args.development_segments,
        args.subject_splits,
        args.output,
        widths=tuple(args.widths),
        min_query_events=args.min_query_events,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
