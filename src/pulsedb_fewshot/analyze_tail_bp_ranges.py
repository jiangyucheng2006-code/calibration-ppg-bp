"""Analyze BP-level and BP-range associations in a participant error tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BP_BINS = {
    "sbp": ([-np.inf, 90, 110, 130, 150, np.inf], ["<90", "90-109", "110-129", "130-149", ">=150"]),
    "dbp": ([-np.inf, 50, 60, 70, 80, np.inf], ["<50", "50-59", "60-69", "70-79", ">=80"]),
}
PARTICIPANT_MEAN_BINS = {
    "sbp": ([-np.inf, 110, 130, np.inf], ["<110", "110-129", ">=130"]),
    "dbp": ([-np.inf, 60, 70, np.inf], ["<60", "60-69", ">=70"]),
}


def _macro_range_table(events: pd.DataFrame, bp: str) -> pd.DataFrame:
    bins, labels = BP_BINS[bp]
    frame = events.copy()
    frame["bp_range"] = pd.cut(
        frame[f"target_{bp}"], bins=bins, labels=labels, right=False
    )
    frame["absolute_error"] = frame[f"abs_error_{bp}"]
    observed = frame.groupby(
        ["tail_group", "subject_uid", "bp_range"],
        observed=True,
        as_index=False,
    ).agg(events=("event_id", "size"), error_sum=("absolute_error", "sum"))
    subjects = frame[["tail_group", "subject_uid"]].drop_duplicates().copy()
    ranges = pd.DataFrame({"bp_range": labels})
    subjects["_join"] = 1
    ranges["_join"] = 1
    grouped = (
        subjects.merge(ranges, on="_join", how="inner")
        .drop(columns="_join")
        .merge(
            observed,
            on=["tail_group", "subject_uid", "bp_range"],
            how="left",
            validate="one_to_one",
        )
    )
    grouped[["events", "error_sum"]] = grouped[["events", "error_sum"]].fillna(0)
    totals = frame.groupby(["tail_group", "subject_uid"], as_index=False).agg(
        total_events=("event_id", "size"), total_error=("absolute_error", "sum")
    )
    grouped = grouped.merge(
        totals, on=["tail_group", "subject_uid"], how="left", validate="many_to_one"
    )
    grouped["event_fraction"] = grouped["events"] / grouped["total_events"]
    grouped["error_fraction"] = grouped["error_sum"] / grouped["total_error"].replace(0, np.nan)
    grouped["mae_in_range"] = grouped["error_sum"] / grouped["events"].replace(0, np.nan)
    result = grouped.groupby(["tail_group", "bp_range"], observed=False, as_index=False).agg(
        participant_macro_event_fraction=("event_fraction", "mean"),
        participant_macro_error_fraction=("error_fraction", "mean"),
        participant_macro_mae_in_range=("mae_in_range", "mean"),
        participants_with_events=("events", lambda values: int(values.gt(0).sum())),
    )
    result.insert(1, "bp", bp.upper())
    return result


def analyze(events_path: Path, participant_path: Path, output_dir: Path) -> dict[str, object]:
    events = pd.read_parquet(events_path)
    participant = pd.read_csv(participant_path)
    tail_columns = [column for column in participant if column.startswith("high_error_q")]
    if len(tail_columns) != 1:
        raise ValueError(f"expected one high-error flag, found {tail_columns}")
    tail_column = tail_columns[0]
    participant[tail_column] = participant[tail_column].astype(bool)
    participant["tail_group"] = np.where(
        participant[tail_column], "worst_20pct_subjects", "remaining_80pct_subjects"
    )
    events = events.merge(
        participant[["subject_uid", "tail_group"]],
        on="subject_uid",
        how="left",
        validate="many_to_one",
    )
    if events["tail_group"].isna().any():
        raise AssertionError("event table contains a participant without a tail assignment")

    distribution = pd.concat(
        [_macro_range_table(events, bp) for bp in ("sbp", "dbp")],
        ignore_index=True,
    )
    rows: list[dict[str, object]] = []
    for group, frame in participant.groupby("tail_group"):
        for bp in ("sbp", "dbp"):
            values = frame[f"target_{bp}_mean"]
            rows.append(
                {
                    "tail_group": group,
                    "bp": bp.upper(),
                    "participants": int(len(frame)),
                    "participant_mean_bp_mean": float(values.mean()),
                    "participant_mean_bp_q25": float(values.quantile(0.25)),
                    "participant_mean_bp_median": float(values.median()),
                    "participant_mean_bp_q75": float(values.quantile(0.75)),
                }
            )
    level_summary = pd.DataFrame(rows)
    mean_range_rows: list[dict[str, object]] = []
    for bp, (bins, labels) in PARTICIPANT_MEAN_BINS.items():
        ranges = pd.cut(
            participant[f"target_{bp}_mean"],
            bins=bins,
            labels=labels,
            right=False,
        )
        counts = (
            participant.assign(mean_bp_range=ranges)
            .groupby(["tail_group", "mean_bp_range"], observed=False)
            .size()
            .rename("participants")
            .reset_index()
        )
        counts["participant_fraction"] = counts["participants"] / counts.groupby(
            "tail_group"
        )["participants"].transform("sum")
        counts.insert(1, "bp", bp.upper())
        mean_range_rows.extend(counts.to_dict("records"))
    participant_mean_ranges = pd.DataFrame(mean_range_rows)
    correlation_fields = [
        "target_sbp_mean",
        "target_dbp_mean",
        "target_sbp_std",
        "target_dbp_std",
        "abs_delta_support_sbp_mean",
        "abs_delta_support_dbp_mean",
        "event_index_mean",
    ]
    correlations = (
        participant[["mean_mae", *correlation_fields]]
        .corr(method="spearman")["mean_mae"]
        .drop("mean_mae")
        .reset_index(name="spearman_rho")
        .rename(columns={"index": "participant_metric"})
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(output_dir / "bp_range_participant_macro.csv", index=False)
    level_summary.to_csv(output_dir / "participant_mean_bp_summary.csv", index=False)
    participant_mean_ranges.to_csv(
        output_dir / "participant_mean_bp_ranges.csv", index=False
    )
    correlations.to_csv(output_dir / "participant_spearman_correlations.csv", index=False)

    summary = {
        "analysis": "development-only BP-range association in the worst participant tail",
        "locked_meta_test_accessed": False,
        "tail_definition": tail_column,
        "participants": int(participant["subject_uid"].nunique()),
        "events": int(len(events)),
        "bp_bins": {bp.upper(): labels for bp, (_, labels) in BP_BINS.items()},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.events, args.participants, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
