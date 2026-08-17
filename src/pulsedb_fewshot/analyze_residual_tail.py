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


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""

    headers = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for record in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in record]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


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
        "abs_delta_support_sbp_mean": ("abs_delta_support_sbp", "mean"),
        "abs_delta_support_dbp_mean": ("abs_delta_support_dbp", "mean"),
        "outside_support_sbp_rate": ("outside_support_sbp", "mean"),
        "outside_support_dbp_rate": ("outside_support_dbp", "mean"),
        "support_sbp_std": ("support_sbp_std", "first"),
        "support_dbp_std": ("support_dbp_std", "first"),
    }
    for column in ("age_clean", "age_valid"):
        if column in events:
            aggregation[f"{column}_mean"] = (column, "mean")
    if "sex" in events:
        aggregation["sex"] = ("sex", "first")
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
    tail_fraction: float = 0.20,
    demographics_path: Path | None = None,
) -> dict[str, object]:
    if not 0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must be between zero and 0.5")
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

    all_metadata = load_store_metadata(store_root, "development")
    validation_all = all_metadata.loc[all_metadata["split"].eq("meta_validation")].copy()
    support = validation_all.loc[
        validation_all["event_index"].le(k)
    ].sort_values(["subject_uid", "event_index"])
    support_summary = support.groupby("subject_uid", as_index=False).agg(
        support_sbp_mean=("sbp", "mean"),
        support_dbp_mean=("dbp", "mean"),
        support_sbp_min=("sbp", "min"),
        support_sbp_max=("sbp", "max"),
        support_dbp_min=("dbp", "min"),
        support_dbp_max=("dbp", "max"),
        support_sbp_std=("sbp", "std"),
        support_dbp_std=("dbp", "std"),
    ).fillna({"support_sbp_std": 0.0, "support_dbp_std": 0.0})
    metadata = validation_all.loc[validation_all["common_query"]].copy()
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
    events = predictions.merge(
        metadata[metadata_columns],
        on=["event_id", "subject_uid"],
        how="left",
        validate="one_to_one",
    )
    events = events.merge(
        support_summary, on="subject_uid", how="left", validate="many_to_one"
    )
    if events[["support_sbp_mean", "support_dbp_mean"]].isna().any().any():
        raise AssertionError("at least one validation participant lacks K support events")
    if demographics_path is not None:
        demographics = pd.read_parquet(demographics_path)
        demo_columns = [
            column
            for column in ("subject_uid", "age_clean", "age_valid", "sex")
            if column in demographics
        ]
        events = events.merge(
            demographics[demo_columns],
            on="subject_uid",
            how="left",
            validate="many_to_one",
        )
    for bp in ("sbp", "dbp"):
        events[f"error_{bp}"] = events[f"pred_{bp}"] - events[f"target_{bp}"]
        events[f"abs_error_{bp}"] = events[f"error_{bp}"].abs()
        events[f"abs_delta_support_{bp}"] = (
            events[f"target_{bp}"] - events[f"support_{bp}_mean"]
        ).abs()
        events[f"outside_support_{bp}"] = (
            events[f"target_{bp}"].lt(events[f"support_{bp}_min"])
            | events[f"target_{bp}"].gt(events[f"support_{bp}_max"])
        )
    events["mean_abs_error"] = (
        events["abs_error_sbp"] + events["abs_error_dbp"]
    ) / 2

    participant = _participant_table(events)
    quantile = 1.0 - tail_fraction
    threshold = float(participant["mean_mae"].quantile(quantile))
    tail_column = f"high_error_q{int(round(quantile * 100))}"
    participant[tail_column] = participant["mean_mae"].ge(threshold)

    source_summary = (
        participant.groupby("source", as_index=False)
        .agg(
            participants=("subject_uid", "nunique"),
            mean_mae=("mean_mae", "mean"),
            sbp_mae=("sbp_mae", "mean"),
            dbp_mae=("dbp_mae", "mean"),
            high_error_rate=(tail_column, "mean"),
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
        "abs_delta_support_sbp_mean",
        "abs_delta_support_dbp_mean",
        "outside_support_sbp_rate",
        "outside_support_dbp_rate",
        "support_sbp_std",
        "support_dbp_std",
    ] + [column for column in ("age_clean_mean", "age_valid_mean") if column in participant]
    tail_percent = int(round(tail_fraction * 100))
    high_error_comparison = (
        participant.assign(
            group=np.where(
                participant[tail_column],
                f"worst_{tail_percent}pct_subjects",
                f"remaining_{100 - tail_percent}pct_subjects",
            )
        )
        .groupby("group", as_index=False)[comparison_columns]
        .mean(numeric_only=True)
    )
    coverage = _coverage_curve(participant)

    output_dir.mkdir(parents=True, exist_ok=False)
    events.to_parquet(output_dir / "event_error_enriched.parquet", index=False)
    participant.to_csv(output_dir / f"participant_error_k{k}.csv", index=False)
    participant.loc[participant[tail_column]].sort_values(
        "mean_mae", ascending=False
    ).to_csv(output_dir / f"worst_{tail_percent}pct_subjects_k{k}.csv", index=False)
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
        "tail_fraction": tail_fraction,
        "high_error_threshold_quantile": quantile,
        "high_error_threshold_mean_mae": threshold,
        "oracle_warning": (
            "Coverage-error rows rank participants using observed query error. "
            "They are diagnostic only and cannot be used for deployment or headline accuracy."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown = [
        f"# Worst {tail_percent}% participant error analysis (K={k})",
        "",
        "Development split: meta-validation only. Locked meta-test was not accessed.",
        "",
        f"- Participants: {len(participant)}",
        f"- Worst-tail threshold: participant mean MAE >= {threshold:.3f} mmHg",
        f"- Tail participants: {int(participant[tail_column].sum())}",
        "- Support-to-query BP change uses only the first K calibration events as the support anchor.",
        "- PPG standard deviation is a simple amplitude proxy, not a validated signal-quality index.",
        "",
        "## Group comparison",
        "",
        _markdown_table(high_error_comparison),
        "",
        "## Source composition",
        "",
        _markdown_table(source_summary),
        "",
        "Observed query error defines the tail, so this is diagnostic/oracle analysis only; it is not a deployable rejection rule.",
    ]
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tail-fraction", type=float, default=0.20)
    parser.add_argument("--demographics-path", type=Path)
    args = parser.parse_args()
    summary = analyze(
        args.predictions,
        args.store_root,
        args.output_dir,
        k=args.k,
        tail_fraction=args.tail_fraction,
        demographics_path=args.demographics_path,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
