"""Recompute the extended Phase-5 first-run table from saved predictions.

The standard-style columns in this module are retrospective numerical screens.
They are not device validation or regulatory-compliance determinations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "population_mean": "B0 Population BP mean",
    "population": "B3 Population PPG network",
    "last_cuff": "B1 Last-cuff persistence",
    "support_mean": "B2 Support-BP mean",
    "residual_offset": "B4 Residual-offset correction",
    "siamese": "B5 Siamese delta",
    "head": "B6 Head-only adaptation",
    "full": "B7 Full-network adaptation",
    "lora": "B8 LoRA adaptation",
    "m0": "M0 Variable-K residual anchor",
    "m1": "M1 M0 + FiLM",
    "m2": "M2 M1 + reliability weighting",
}

METHOD_ORDER = {name: index for index, name in enumerate(METHOD_LABELS)}
REQUIRED_COLUMNS = {
    "method",
    "k",
    "subject_uid",
    "event_id",
    "target_sbp",
    "target_dbp",
    "pred_sbp",
    "pred_dbp",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bhs_grade(within_5: float, within_10: float, within_15: float) -> str:
    """Return the historical BHS grade from cumulative percentages."""

    if within_5 >= 60 and within_10 >= 85 and within_15 >= 95:
        return "A"
    if within_5 >= 50 and within_10 >= 75 and within_15 >= 90:
        return "B"
    if within_5 >= 40 and within_10 >= 65 and within_15 >= 85:
        return "C"
    return "D"


def _read_predictions(path: Path, method: str | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if method is not None:
        if "method" in frame and set(frame["method"].dropna().astype(str)) != {method}:
            raise AssertionError(f"{path.name} has an unexpected method column")
        frame = frame.copy()
        frame["method"] = method
    return frame


def load_first_run_predictions(
    *,
    controls: Path,
    m0: Path,
    m1: Path,
    m2: Path,
    siamese: Path,
) -> pd.DataFrame:
    """Load and validate the five first-run prediction artifacts."""

    frames = [
        _read_predictions(controls),
        _read_predictions(m0, "m0"),
        _read_predictions(m1, "m1"),
        _read_predictions(m2, "m2"),
        _read_predictions(siamese, "siamese"),
    ]
    result = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"prediction table is missing columns: {sorted(missing)}")
    result = result[list(REQUIRED_COLUMNS)].copy()
    result["method"] = result["method"].astype(str)
    unknown = set(result["method"]) - set(METHOD_LABELS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    result["k"] = pd.to_numeric(result["k"], errors="raise").astype(int)
    if set(result["k"]) - {1, 2, 3, 5}:
        raise ValueError("K must be one of 1, 2, 3, or 5")
    numeric = result[
        ["target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("prediction table contains nonfinite targets or predictions")
    keys = ["method", "k", "subject_uid", "event_id"]
    if result.duplicated(keys).any():
        raise AssertionError("duplicate method/K/participant/event prediction rows")

    targets = result.groupby(["subject_uid", "event_id"], sort=False).agg(
        target_sbp_min=("target_sbp", "min"),
        target_sbp_max=("target_sbp", "max"),
        target_dbp_min=("target_dbp", "min"),
        target_dbp_max=("target_dbp", "max"),
    )
    sbp_spread = targets["target_sbp_max"] - targets["target_sbp_min"]
    dbp_spread = targets["target_dbp_max"] - targets["target_dbp_min"]
    # Saved analytical controls and neural predictions pass through slightly
    # different float32/float64 paths. Their observed target spread is below
    # 8e-6 mmHg, so 1e-4 is a strict numerical-equivalence tolerance.
    if (sbp_spread > 1e-4).any() or (dbp_spread > 1e-4).any():
        raise AssertionError("saved methods disagree on a query target")
    return result


def _pooled_r_squared(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator <= 0:
        return float("nan")
    return 1.0 - float(np.sum((prediction - target) ** 2)) / denominator


def compute_extended_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled diagnostic metrics and participant-macro MAE by BP."""

    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction table is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    grouped = predictions.groupby(["method", "k"], sort=False)
    for (method, k), group in grouped:
        for bp in ("SBP", "DBP"):
            lower = bp.lower()
            target = group[f"target_{lower}"].to_numpy(dtype=float)
            prediction = group[f"pred_{lower}"].to_numpy(dtype=float)
            error = prediction - target
            absolute_error = np.abs(error)
            within_5 = float(np.mean(absolute_error <= 5.0) * 100.0)
            within_10 = float(np.mean(absolute_error <= 10.0) * 100.0)
            within_15 = float(np.mean(absolute_error <= 15.0) * 100.0)
            grade = bhs_grade(within_5, within_10, within_15)
            mean_error = float(np.mean(error))
            error_std = float(np.std(error, ddof=1))
            participant_macro_mae = float(
                pd.DataFrame(
                    {
                        "subject_uid": group["subject_uid"].to_numpy(),
                        "absolute_error": absolute_error,
                    }
                )
                .groupby("subject_uid", sort=False)["absolute_error"]
                .mean()
                .mean()
            )
            aami_pass = abs(mean_error) <= 5.0 and error_std <= 8.0
            bhs_pass = grade in {"A", "B"}
            rows.append(
                {
                    "Setting": f"{METHOD_LABELS[str(method)]} (K={int(k)})",
                    "BP": bp,
                    "MAE": float(np.mean(absolute_error)),
                    "R²": _pooled_r_squared(target, prediction),
                    "ME": mean_error,
                    "STD": error_std,
                    "≤5 mmHg (%)": within_5,
                    "≤10 mmHg (%)": within_10,
                    "≤15 mmHg (%)": within_15,
                    "AAMI": "PASS*" if aami_pass else "FAIL*",
                    "BHS": f"{'PASS' if bhs_pass else 'FAIL'} (Grade {grade})*",
                    "Participant-macro MAE": participant_macro_mae,
                    "N participants": int(group["subject_uid"].nunique()),
                    "N query events": int(len(group)),
                    "Method": str(method),
                    "K": int(k),
                    "Aggregation": "event-pooled diagnostic",
                    "Standards scope": "numerical screen only; formal compliance not established",
                }
            )
    result = pd.DataFrame(rows)
    result["_method_order"] = result["Method"].map(METHOD_ORDER)
    result["_bp_order"] = result["BP"].map({"SBP": 0, "DBP": 1})
    result = result.sort_values(
        ["_method_order", "K", "_bp_order"], kind="mergesort"
    ).drop(columns=["_method_order", "_bp_order"])
    return result.reset_index(drop=True)


def _format_number(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_markdown(metrics: pd.DataFrame, path: Path) -> None:
    columns = [
        "Setting",
        "BP",
        "MAE",
        "R²",
        "ME",
        "STD",
        "≤5 mmHg (%)",
        "≤10 mmHg (%)",
        "≤15 mmHg (%)",
        "AAMI",
        "BHS",
    ]
    lines = [
        "# Phase-5 first-run extended result table",
        "",
        "Last updated: 2026-08-15.",
        "",
        "This is an exploratory, single-seed (`20260813`) `meta_validation` ",
        "report reconstructed from saved event-level predictions. The locked test ",
        "was not accessed. Every row contains 697 participants and 103,564 common ",
        "future query events. `ME` is prediction minus reference; `STD` uses the ",
        "sample standard deviation (`ddof=1`). MAE, R², ME, STD, and threshold ",
        "percentages in this table are event-pooled diagnostics. The project primary ",
        "MAE remains the participant-macro value reported in `RESULTS_PHASE5.md`.",
        "",
        "`AAMI` is only a Criterion-1-style numerical screen: `|ME| <= 5 mmHg` ",
        "and `STD <= 8 mmHg`, evaluated separately for SBP and DBP. The full ",
        "AAMI/ESH/ISO protocol also has design, population, reference-measurement, ",
        "and repeated-measure requirements that this retrospective ML benchmark does ",
        "not satisfy. `BHS` is a historical numerical grade based on cumulative ",
        "percentages within 5/10/15 mmHg; Grade A/B is displayed as PASS and C/D as ",
        "FAIL. Every asterisk therefore means **numerical screen only; formal device ",
        "compliance is not established**.",
        "",
        "Primary references: [AAMI/ESH/ISO collaboration statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC5796427/), [current ISO 81060-2:2018 status](https://www.iso.org/standard/73339.html), and the [original BHS protocol](https://pubmed.ncbi.nlm.nih.gov/2168451/).",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, values in metrics.iterrows():
        rendered = []
        for column in columns:
            value = values[column]
            if column in {"MAE", "R²", "ME", "STD"}:
                rendered.append(_format_number(value, 3))
            elif column.endswith("(%)"):
                rendered.append(_format_number(value, 2))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "The machine-readable CSV also contains participant-macro MAE, participant ",
            "and event counts, method/K identifiers, aggregation scope, and the formal ",
            "standards limitation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _input_record(name: str, path: Path) -> dict[str, object]:
    return {
        "name": name,
        "file_name": path.name,
        "sha256": file_sha256(path),
        "rows": int(len(pd.read_parquet(path, columns=["event_id"]))),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    inputs = {
        "controls": args.controls,
        "m0": args.m0,
        "m1": args.m1,
        "m2": args.m2,
        "siamese": args.siamese,
    }
    predictions = load_first_run_predictions(**inputs)
    metrics = compute_extended_metrics(predictions)
    if len(metrics) != 90:
        raise AssertionError(f"expected 90 BP-specific rows, found {len(metrics)}")
    if set(metrics["N participants"]) != {697}:
        raise AssertionError("first-run settings do not all contain 697 participants")
    if set(metrics["N query events"]) != {103_564}:
        raise AssertionError("first-run settings do not share 103,564 query events")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_csv, index=False, float_format="%.6f")
    write_markdown(metrics, args.output_markdown)
    report = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "exploratory single-seed meta-validation",
        "seed": 20260813,
        "locked_test_accessed": False,
        "error_sign": "prediction_minus_reference",
        "std_ddof": 1,
        "primary_project_metric": "participant-macro MAE",
        "table_metric_scope": "event-pooled diagnostics plus participant-macro MAE",
        "aami_numeric_screen": "abs(ME) <= 5 mmHg and error STD <= 8 mmHg; criterion-1-style only",
        "bhs_numeric_screen": {
            "A": [60, 85, 95],
            "B": [50, 75, 90],
            "C": [40, 65, 85],
            "D": "below Grade C",
            "binary_display": "PASS for A/B; FAIL for C/D",
        },
        "formal_standard_compliance": "not established",
        "inputs": [_input_record(name, path) for name, path in inputs.items()],
        "outputs": {
            "csv_name": args.output_csv.name,
            "markdown_name": args.output_markdown.name,
            "rows": int(len(metrics)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--m0", required=True, type=Path)
    parser.add_argument("--m1", required=True, type=Path)
    parser.add_argument("--m2", required=True, type=Path)
    parser.add_argument("--siamese", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run(args)
    print(f"FIRST_RUN_EXTENDED_REPORT_COMPLETE=yes rows={len(metrics)}")


if __name__ == "__main__":
    main()
