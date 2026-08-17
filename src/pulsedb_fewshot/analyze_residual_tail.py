"""Development-only residual-tail analysis for calibrated PPG-BP predictions.

The coverage analysis is explicitly oracle-only because it ranks participants by
their observed query error. It must not be used as an inference-time filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .training import load_store_metadata


def _participant_table(events: pd.DataFrame) -> pd.DataFrame:
    aggregation: dict[str, tuple[str, str]] = {
        "source": ("source", "first"),
        "n_queries": ("event_id", "size"),
        "sbp_mae": ("abs_error_sbp", "mean"),
        "dbp_mae": ("abs_error_dbp", "mean"),
        "target_sbp_mean": ("target_sbp", "mean"),
        "target_sbp_std": ("target_sbp", "std"),
        "target_dbp_mean": ("target_dbp", "mean"),
        "target_dbp_std": ("target_dbp", "std"),
        "event_index_mean": ("event_index", "mean"),
        "event_index_max": ("event_index", "max"),
        "ppg_f_std_mean": ("ppg_f_std", "mean"),
    }
    for column in ("abs_delta_support_sbp", "abs_delta_support_dbp", "age"):
        if column in events:
            aggregation[f"{column}_mean"] = (column, "mean")
    if "gender" in events:
        aggregation["gender"] = ("gender", "first")
    participant = events.groupby("subject_uid", as_index=False).agg(**aggregation)
    participant["mean_mae"] = (participant["sbp_mae"] + participant["dbp_mae"]) / 2
    return participant


def _coverage_curve(participant: pd.DataFrame) -> pd.DataFrame:
    ordered = participant.sort_values(["mean_mae", "subject_uid"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for raw_coverage in np.arange(0.50, 1.001, 0.05):
        coverage = min(1.0, float(raw_coverage))
        retained_n = min(
            len(ordered), max(1, int(np.ceil(len(ordered) * coverage)))
        )
        retained = ordered.iloc[:retained_n]
        rows.append(
            {
                "coverage": float(coverage),
                "retained_participants": int(retained_n),
                "oracle_mean_mae": float(retained["mean_mae"].mean()),
                "oracle_sbp_mae": float(retained["sbp_mae"].mean()),
                "oracle_dbp_mae": float(retained["dbp_mae"].mean()),
                "selection_rule": "lowest observed participant mean MAE (oracle only)",
            }
        )
    return pd.DataFrame(rows)


def analyze(
    predictions_path: Path,
    store_root: Path,
    output_dir: Path,
    *,
    k: int = 5,
) -> dict[str, object]:
    predictions = pd.read_parquet(predictions_path)
    required = {
        "subject_uid",
        "event_id",
        "k",
        "target_sbp",
        "target_dbp",
        "pred_sbp",
        "pred_dbp",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    predictions = predictions.loc[predictions["k"].eq(k)].copy()
    if predictions.empty:
        raise ValueError(f"no predictions for K={k}")
    if predictions["event_id"].duplicated().any():
        raise AssertionError("prediction event IDs are not unique within K")

    metadata = load_store_metadata(store_root, "development")
    metadata = metadata.loc[
        metadata["split"].eq("meta_validation") & metadata["common_query"]
    ].copy()
    if set(predictions["event_id"]) != set(metadata["event_id"]):
        raise AssertionError("prediction query set differs from frozen meta-validation queries")
    if set(predictions["subject_uid"]) != set(metadata["subject_uid"]):
        raise AssertionError("prediction participant set differs from meta-validation")

    metadata_columns = [
        "event_id",
        "subject_uid",
        "source",
        "event_index",
        "time_bin",
        "start_time_s",
        "ppg_f_std",
    ]
    metadata_columns += [
        column
        for column in (
            f"support_sbp_k{k}",
            f"support_dbp_k{k}",
            "age",
            "gender",
        )
        if column in metadata
    ]
    events = predictions.merge(
        metadata[metadata_columns],
        on=["event_id", "subject_uid"],
        how="left",
        validate="one_to_one",
    )
    for bp in ("sbp", "dbp"):
        events[f"error_{bp}"] = events[f"pred_{bp}"] - events[f"target_{bp}"]
        events[f"abs_error_{bp}"] = events[f"error_{bp}"].abs()
        support_column = f"support_{bp}_k{k}"
        if support_column in events:
            events[f"abs_delta_support_{bp}"] = (
                events[f"target_{bp}"] - events[support_column]
            ).abs()
    events["mean_abs_error"] = (
        events["abs_error_sbp"] + events["abs_error_dbp"]
    ) / 2

    participant = _participant_table(events)
    threshold = float(participant["mean_mae"].quantile(0.90))
    participant["high_error_q90"] = participant["mean_mae"].ge(threshold)

    source_summary = (
        participant.groupby("source", as_index=False)
        .agg(
            participants=("subject_uid", "nunique"),
            mean_mae=("mean_mae", "mean"),
            sbp_mae=("sbp_mae", "mean"),
            dbp_mae=("dbp_mae", "mean"),
            high_error_rate=("high_error_q90", "mean"),
        )
        .sort_values("source")
    )
    comparison_columns = [
        "mean_mae",
        "sbp_mae",
        "dbp_mae",
        "target_sbp_mean",
        "target_sbp_std",
        "target_dbp_mean",
        "target_dbp_std",
        "event_index_mean",
        "event_index_max",
        "ppg_f_std_mean",
    ] + [
        column
        for column in participant.columns
        if column.startswith("abs_delta_support_") or column == "age_mean"
    ]
    high_error_comparison = (
        participant.assign(
            group=np.where(participant["high_error_q90"], "top_10pct_error", "remaining_90pct")
        )
        .groupby("group", as_index=False)[comparison_columns]
        .mean(numeric_only=True)
    )
    coverage = _coverage_curve(participant)

    output_dir.mkdir(parents=True, exist_ok=False)
    events.to_parquet(output_dir / "event_error_enriched.parquet", index=False)
    participant.to_csv(output_dir / f"participant_error_k{k}.csv", index=False)
    source_summary.to_csv(output_dir / f"source_summary_k{k}.csv", index=False)
    high_error_comparison.to_csv(
        output_dir / f"high_error_comparison_k{k}.csv", index=False
    )
    coverage.to_csv(
        output_dir / f"oracle_coverage_error_curve_k{k}.csv", index=False
    )

    quantiles = participant["mean_mae"].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    summary = {
        "analysis": "development-only residual-tail analysis",
        "predictions": str(predictions_path),
        "store_root": str(store_root),
        "k": k,
        "split": "meta_validation",
        "locked_meta_test_accessed": False,
        "participants": int(participant["subject_uid"].nunique()),
        "queries": int(len(events)),
        "mean_mae": float(participant["mean_mae"].mean()),
        "sbp_mae": float(participant["sbp_mae"].mean()),
        "dbp_mae": float(participant["dbp_mae"].mean()),
        "participant_mean_mae_quantiles": {
            str(index): float(value) for index, value in quantiles.items()
        },
        "high_error_threshold_q90": threshold,
        "oracle_warning": (
            "Coverage-error rows rank participants using observed query error. "
            "They are diagnostic only and cannot be used for deployment or headline accuracy."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    summary = analyze(
        args.predictions, args.store_root, args.output_dir, k=args.k
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
