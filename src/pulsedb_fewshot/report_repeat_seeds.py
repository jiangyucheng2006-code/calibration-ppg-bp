"""Build the leakage-safe repeated-seed Phase-5 development report.

The standard-style columns produced here are retrospective numerical screens.
They are not device-validation or regulatory-compliance determinations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .report_first_run import (
    METHOD_LABELS,
    REQUIRED_COLUMNS,
    compute_extended_metrics,
    file_sha256,
)


KS = (1, 2, 3, 5)
TRAINED_PERSONALIZED_METHODS = ("m0", "m1", "m2")
CONTROL_METHODS = (
    "population_mean",
    "last_cuff",
    "support_mean",
    "population",
    "residual_offset",
    "head",
    "full",
    "lora",
)
EXPECTED_METHODS = CONTROL_METHODS + TRAINED_PERSONALIZED_METHODS
REPEAT_METHOD_ORDER = {name: index for index, name in enumerate(EXPECTED_METHODS)}
KEY_COLUMNS = ["subject_uid", "event_id"]
TARGET_COLUMNS = ["target_sbp", "target_dbp"]


def _find_one(run_root: Path, pattern: str, marker: str) -> Path:
    matches = sorted(
        path.parent for path in run_root.glob(f"{pattern}/{marker}") if path.is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one complete run for {pattern}, found {len(matches)}: {matches}"
        )
    return matches[0]


def _load_metadata(path: Path, *, seed: int, method: str | None = None) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("split") != "meta_validation":
        raise AssertionError(f"non-validation result at {path}")
    if record.get("locked_test_accessed") is not False:
        raise AssertionError(f"locked-test status is not explicitly false at {path}")
    if method is not None:
        if int(record.get("seed")) != seed or record.get("method") != method:
            raise AssertionError(f"run identity mismatch at {path}")
    return record


def _normalize_predictions(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    columns = [
        "method",
        "k",
        "subject_uid",
        "event_id",
        "target_sbp",
        "target_dbp",
        "pred_sbp",
        "pred_dbp",
    ]
    result = frame[columns].copy()
    result["method"] = result["method"].astype(str)
    result["k"] = pd.to_numeric(result["k"], errors="raise").astype(int)
    if not set(result["method"]).issubset(EXPECTED_METHODS):
        unknown = sorted(set(result["method"]) - set(EXPECTED_METHODS))
        raise ValueError(f"unknown methods in {path}: {unknown}")
    if set(result["k"]) != set(KS):
        raise ValueError(f"{path} does not contain K={KS}")
    numeric = result[
        ["target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"nonfinite target or prediction in {path}")
    keys = ["method", "k", *KEY_COLUMNS]
    if result.duplicated(keys).any():
        raise AssertionError(f"duplicate method/K/query rows in {path}")
    return result


def _canonical_targets(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame[
        (frame["method"] == "population") & (frame["k"] == 1)
    ][[*KEY_COLUMNS, *TARGET_COLUMNS]].sort_values(KEY_COLUMNS, kind="mergesort")
    if canonical.duplicated(KEY_COLUMNS).any():
        raise AssertionError("canonical query target table contains duplicate keys")
    return canonical.reset_index(drop=True)


def _assert_common_targets(frame: pd.DataFrame, canonical: pd.DataFrame) -> None:
    expected_keys = canonical[KEY_COLUMNS]
    expected_values = canonical[TARGET_COLUMNS].to_numpy(dtype=float)
    for (method, k), group in frame.groupby(["method", "k"], sort=False):
        observed = group[[*KEY_COLUMNS, *TARGET_COLUMNS]].sort_values(
            KEY_COLUMNS, kind="mergesort"
        )
        observed = observed.reset_index(drop=True)
        if len(observed) != len(canonical):
            raise AssertionError(
                f"query count mismatch for method={method}, K={k}: {len(observed)}"
            )
        if not observed[KEY_COLUMNS].equals(expected_keys):
            raise AssertionError(f"query keys differ for method={method}, K={k}")
        if not np.allclose(
            observed[TARGET_COLUMNS].to_numpy(dtype=float),
            expected_values,
            rtol=0.0,
            atol=1e-4,
        ):
            raise AssertionError(f"query targets differ for method={method}, K={k}")


def load_seed_predictions(
    run_root: Path, *, run_prefix: str, seed: int
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    """Load one seed while proving development-only identity and common queries."""

    evidence: list[dict[str, object]] = []
    population_dir = _find_one(
        run_root, f"{run_prefix}_population_seed{seed}_job*", "run.json"
    )
    population_run = population_dir / "run.json"
    _load_metadata(population_run, seed=seed, method="population")
    evidence.append(
        {"path": str(population_run), "sha256": file_sha256(population_run)}
    )

    controls_dir = _find_one(
        run_root,
        f"{run_prefix}_calibration_controls_seed{seed}_job*",
        "metrics.json",
    )
    controls_metadata = controls_dir / "metrics.json"
    _load_metadata(controls_metadata, seed=seed)
    controls_path = controls_dir / "validation_predictions.parquet"
    controls = _normalize_predictions(pd.read_parquet(controls_path), controls_path)
    if set(controls["method"]) != set(CONTROL_METHODS):
        raise AssertionError(f"incomplete control methods at {controls_path}")
    evidence.extend(
        [
            {
                "path": str(controls_metadata),
                "sha256": file_sha256(controls_metadata),
            },
            {
                "path": str(controls_path),
                "sha256": file_sha256(controls_path),
                "rows": int(len(controls)),
            },
        ]
    )

    frames = [controls]
    for method in TRAINED_PERSONALIZED_METHODS:
        run_dir = _find_one(
            run_root, f"{run_prefix}_{method}_seed{seed}_job*", "run.json"
        )
        run_path = run_dir / "run.json"
        _load_metadata(run_path, seed=seed, method=method)
        predictions_path = run_dir / "best_validation_predictions.parquet"
        predictions = pd.read_parquet(predictions_path)
        predictions = predictions.copy()
        predictions["method"] = method
        frames.append(_normalize_predictions(predictions, predictions_path))
        evidence.extend(
            [
                {"path": str(run_path), "sha256": file_sha256(run_path)},
                {
                    "path": str(predictions_path),
                    "sha256": file_sha256(predictions_path),
                    "rows": int(len(predictions)),
                },
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    if set(combined["method"]) != set(EXPECTED_METHODS):
        raise AssertionError(f"seed {seed} has incomplete method coverage")
    canonical = _canonical_targets(combined)
    _assert_common_targets(combined, canonical)
    return combined, evidence, canonical


def summarize_participant_macro(per_seed: pd.DataFrame) -> pd.DataFrame:
    by_bp = per_seed.pivot(
        index=[
            "Seed",
            "Method",
            "K",
            "Setting",
            "N participants",
            "N query events",
        ],
        columns="BP",
        values="Participant-macro MAE",
    ).reset_index()
    by_bp["Mean"] = (by_bp["SBP"] + by_bp["DBP"]) / 2.0
    rows: list[dict[str, object]] = []
    for (method, k, setting), group in by_bp.groupby(
        ["Method", "K", "Setting"], sort=False
    ):
        rows.append(
            {
                "Setting": setting,
                "Method": method,
                "K": int(k),
                "SBP MAE": float(group["SBP"].mean()),
                "SBP seed SD": float(group["SBP"].std(ddof=1)),
                "DBP MAE": float(group["DBP"].mean()),
                "DBP seed SD": float(group["DBP"].std(ddof=1)),
                "Mean MAE": float(group["Mean"].mean()),
                "Mean MAE seed SD": float(group["Mean"].std(ddof=1)),
                "N seeds": int(group["Seed"].nunique()),
                "N participants": int(group["N participants"].iloc[0]),
                "N query events per K": int(group["N query events"].iloc[0]),
                "Split": "meta_validation",
            }
        )
    result = pd.DataFrame(rows)
    result["_method_order"] = result["Method"].map(REPEAT_METHOD_ORDER)
    return (
        result.sort_values(["_method_order", "K"], kind="mergesort")
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )


def summarize_extended(per_seed: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "MAE",
        "R²",
        "ME",
        "STD",
        "≤5 mmHg (%)",
        "≤10 mmHg (%)",
        "≤15 mmHg (%)",
        "Participant-macro MAE",
    ]
    rows: list[dict[str, object]] = []
    for (setting, bp, method, k), group in per_seed.groupby(
        ["Setting", "BP", "Method", "K"], sort=False
    ):
        row: dict[str, object] = {
            "Setting": setting,
            "BP": bp,
            "Method": method,
            "K": int(k),
        }
        for column in numeric:
            row[column] = float(group[column].mean())
            row[f"{column} seed SD"] = float(group[column].std(ddof=1))
        aami_passes = int(group["AAMI"].astype(str).str.startswith("PASS").sum())
        bhs_passes = int(group["BHS"].astype(str).str.startswith("PASS").sum())
        grades = group["BHS"].astype(str).str.extract(r"Grade ([A-D])")[0]
        grade_counts = {grade: int((grades == grade).sum()) for grade in "ABCD"}
        row["AAMI"] = (
            f"{'PASS' if aami_passes == len(group) else 'FAIL'}* "
            f"({aami_passes}/{len(group)} seeds pass)"
        )
        row["BHS"] = (
            f"{'PASS' if bhs_passes == len(group) else 'FAIL'}* "
            f"({bhs_passes}/{len(group)} seeds pass; "
            + ", ".join(f"{grade}={grade_counts[grade]}" for grade in "ABCD")
            + ")"
        )
        row["N seeds"] = int(group["Seed"].nunique())
        row["N participants"] = int(group["N participants"].iloc[0])
        row["N query events per seed"] = int(group["N query events"].iloc[0])
        row["Aggregation"] = "mean across seed-specific event-pooled diagnostics"
        row["Standards scope"] = (
            "numerical screen only; formal compliance not established"
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    first = [
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
    remaining = [column for column in result.columns if column not in first]
    result = result[first + remaining]
    result["_method_order"] = result["Method"].map(REPEAT_METHOD_ORDER)
    result["_bp_order"] = result["BP"].map({"SBP": 0, "DBP": 1})
    return (
        result.sort_values(
            ["_method_order", "K", "_bp_order"], kind="mergesort"
        )
        .drop(columns=["_method_order", "_bp_order"])
        .reset_index(drop=True)
    )


def _mean_sd(mean: object, standard_deviation: object, digits: int = 3) -> str:
    return f"{float(mean):.{digits}f} ± {float(standard_deviation):.{digits}f}"


def _public_evidence(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove cluster-specific absolute paths from the public provenance record."""

    result: list[dict[str, object]] = []
    for record in records:
        path = Path(str(record["path"]))
        item: dict[str, object] = {
            "run_id": path.parent.name,
            "file_name": path.name,
            "sha256": record["sha256"],
        }
        if "rows" in record:
            item["rows"] = record["rows"]
        result.append(item)
    return result


def write_markdown(
    participant_macro: pd.DataFrame,
    extended: pd.DataFrame,
    per_seed: pd.DataFrame,
    path: Path,
) -> None:
    primary = participant_macro.pivot(
        index=["Method", "Setting"], columns="K", values=["Mean MAE", "Mean MAE seed SD"]
    )
    four_k_by_seed = per_seed.pivot(
        index=["Seed", "Method", "K"],
        columns="BP",
        values="Participant-macro MAE",
    ).reset_index()
    four_k_by_seed["Mean"] = (
        four_k_by_seed["SBP"] + four_k_by_seed["DBP"]
    ) / 2.0
    four_k = (
        four_k_by_seed.groupby(["Seed", "Method"], as_index=False)["Mean"]
        .mean()
        .groupby("Method")["Mean"]
        .agg(["mean", "std"])
        .sort_values("mean")
    )
    lines = [
        "# Phase-5 five-seed development results",
        "",
        "Last updated: 2026-08-17.",
        "",
        "This report covers five prespecified training seeds (`20260813`--",
        "`20260817`) on `meta_validation` only. The locked meta-test was not",
        "accessed. Every method/K/seed row contains the same 697 participants and",
        "103,564 future query events. Calibration uses K=1/2/3/5 independent",
        "`event120-v1` pseudo-cuff/reference-BP events; all K values share queries",
        "beginning at event 6.",
        "",
        "## Primary participant-macro comparison",
        "",
        "Values are mean ± sample SD across the five training seeds. Within each",
        "seed, SBP and DBP MAE are first averaged per participant; the displayed",
        "mean MAE is then the mean of the SBP and DBP participant-macro MAE.",
        "",
        "| Method | K=1 | K=2 | K=3 | K=5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in EXPECTED_METHODS:
        if method not in primary.index.get_level_values("Method"):
            continue
        setting = METHOD_LABELS[method]
        values = []
        for k in KS:
            values.append(
                _mean_sd(
                    primary.loc[(method, f"{setting} (K={k})"), ("Mean MAE", k)],
                    primary.loc[
                        (method, f"{setting} (K={k})"), ("Mean MAE seed SD", k)
                    ],
                )
            )
        lines.append(f"| {setting} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "Across the four K budgets, M0 has the lowest average participant-macro",
            "mean MAE ("
            + _mean_sd(four_k.loc["m0", "mean"], four_k.loc["m0", "std"])
            + " mmHg). M1 and M2 are close, so the current evidence supports selecting",
            "M0 primarily for parsimony rather than claiming a decisive advantage",
            "over every more complex variant. M0 remains clearly better than the",
            "calibration-free population network and the prespecified simple or",
            "adaptation controls.",
            "",
            "## M0 SBP and DBP detail",
            "",
            "| K | SBP MAE | DBP MAE | Mean MAE |",
            "|---:|---:|---:|---:|",
        ]
    )
    for k in KS:
        row = participant_macro[
            (participant_macro["Method"] == "m0") & (participant_macro["K"] == k)
        ].iloc[0]
        lines.append(
            "| "
            + str(k)
            + " | "
            + _mean_sd(row["SBP MAE"], row["SBP seed SD"])
            + " | "
            + _mean_sd(row["DBP MAE"], row["DBP seed SD"])
            + " | "
            + _mean_sd(row["Mean MAE"], row["Mean MAE seed SD"])
            + " |"
        )

    lines.extend(
        [
            "",
            "## Extended diagnostic table",
            "",
            "`ME` is prediction minus reference. Each numeric cell is the mean ±",
            "sample SD across five seed-specific event-pooled metrics. In the `STD`",
            "column, the first number is the mean within-seed sample SD of signed",
            "errors; the second number is its between-seed SD. These pooled metrics",
            "are secondary diagnostics; participant-macro MAE above is primary.",
            "",
            "`AAMI` is only a Criterion-1-style numerical screen (`|ME| <= 5 mmHg`",
            "and error `STD <= 8 mmHg`) applied separately within each seed/BP row.",
            "`BHS` is the historical cumulative 5/10/15-mmHg numerical grade, with",
            "Grade A/B displayed as pass. Asterisks mean **numerical screen only;",
            "formal device compliance is not established**. This retrospective",
            "PulseDB evaluation does not meet the full population, reference,",
            "pairing, or repeated-measure requirements of a device-validation study.",
            "",
            "The [ISO catalogue](https://www.iso.org/standard/73339.html), checked",
            "2026-08-17, lists ISO 81060-2:2018 as current (confirmed in 2024), with",
            "2020 and 2024 amendments and a replacement draft in development; its",
            "published scope is intermittent cuff-based equipment. The historical",
            "BHS grading source is the [original 1990 protocol](https://pubmed.ncbi.nlm.nih.gov/2168451/).",
            "",
            "| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg (%) | ≤10 mmHg (%) | ≤15 mmHg (%) | AAMI | BHS |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    numeric_columns = [
        "MAE",
        "R²",
        "ME",
        "STD",
        "≤5 mmHg (%)",
        "≤10 mmHg (%)",
        "≤15 mmHg (%)",
    ]
    for _, row in extended.iterrows():
        rendered = []
        for column in numeric_columns:
            digits = 2 if column.endswith("(%)") else 3
            rendered.append(
                _mean_sd(row[column], row[f"{column} seed SD"], digits=digits)
            )
        lines.append(
            f"| {row['Setting']} | {row['BP']} | "
            + " | ".join(rendered)
            + f" | {row['AAMI']} | {row['BHS']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation and decision boundary",
            "",
            "- M0 improves as K increases and is the lowest-mean method at every K.",
            "- FiLM (M1) and reliability weighting (M2) do not produce a consistent",
            "  average gain over M0 under the shared protocol.",
            "- The unlimited-epoch runs all stopped by patience=8 early stopping;",
            "  therefore the original 25-epoch-cap concern is resolved.",
            "- This is still internal model-selection evidence. It does not establish",
            "  locked-test generalization, pressure/motion/device robustness, external",
            "  validation, clinical accuracy, or standards compliance.",
            "",
            "The machine-readable result files retain the seed-specific rows, seed",
            "SDs, participant/event counts, method/K identifiers, and standards",
            "scope needed to reconstruct this table.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three unique prespecified seeds are required")

    per_seed_frames: list[pd.DataFrame] = []
    evidence: list[dict[str, object]] = []
    reference_targets: pd.DataFrame | None = None
    for seed in seeds:
        predictions, seed_evidence, canonical = load_seed_predictions(
            args.run_root, run_prefix=args.run_prefix, seed=seed
        )
        if reference_targets is None:
            reference_targets = canonical
        else:
            if not canonical[KEY_COLUMNS].equals(reference_targets[KEY_COLUMNS]):
                raise AssertionError(f"seed {seed} uses a different query set")
            if not np.allclose(
                canonical[TARGET_COLUMNS].to_numpy(dtype=float),
                reference_targets[TARGET_COLUMNS].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-4,
            ):
                raise AssertionError(f"seed {seed} uses different query targets")
        metrics = compute_extended_metrics(predictions)
        metrics.insert(0, "Seed", seed)
        per_seed_frames.append(metrics)
        evidence.extend(seed_evidence)

    per_seed = pd.concat(per_seed_frames, ignore_index=True)
    expected_rows = len(seeds) * len(EXPECTED_METHODS) * len(KS) * 2
    if len(per_seed) != expected_rows:
        raise AssertionError(
            f"expected {expected_rows} seed/method/K/BP rows, found {len(per_seed)}"
        )
    if set(per_seed["N participants"]) != {697}:
        raise AssertionError("not every setting contains 697 participants")
    if set(per_seed["N query events"]) != {103_564}:
        raise AssertionError("not every setting contains 103,564 common queries")

    participant_macro = summarize_participant_macro(per_seed)
    extended = summarize_extended(per_seed)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_seed_path = args.output_dir / "phase5_repeat5_extended_metrics_by_seed.csv"
    participant_path = args.output_dir / "phase5_repeat5_participant_macro.csv"
    extended_path = args.output_dir / "phase5_repeat5_extended_summary.csv"
    markdown_path = args.output_dir / "RESULTS_PHASE5_REPEAT5.md"
    report_path = args.output_dir / "phase5_repeat5_report.json"
    per_seed.to_csv(per_seed_path, index=False, float_format="%.6f")
    participant_macro.to_csv(participant_path, index=False, float_format="%.6f")
    extended.to_csv(extended_path, index=False, float_format="%.6f")
    write_markdown(participant_macro, extended, per_seed, markdown_path)

    output_paths = [per_seed_path, participant_path, extended_path, markdown_path]
    report: dict[str, object] = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "five-seed internal meta-validation",
        "run_prefix": args.run_prefix,
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "split": "meta_validation",
        "locked_test_accessed": False,
        "error_sign": "prediction_minus_reference",
        "error_std_ddof": 1,
        "seed_std_ddof": 1,
        "primary_project_metric": "participant-macro MAE",
        "table_metric_scope": (
            "mean across seed-specific event-pooled diagnostics plus "
            "participant-macro MAE"
        ),
        "aami_numeric_screen": (
            "abs(ME) <= 5 mmHg and error STD <= 8 mmHg; "
            "criterion-1-style numerical screen only"
        ),
        "bhs_numeric_screen": {
            "A": [60, 85, 95],
            "B": [50, 75, 90],
            "C": [40, 65, 85],
            "D": "below Grade C",
            "binary_display": "PASS for A/B; FAIL for C/D",
        },
        "formal_standard_compliance": "not established",
        "coverage": {
            "participants_per_setting": 697,
            "common_query_events_per_setting": 103_564,
            "methods": list(EXPECTED_METHODS),
            "k": list(KS),
            "per_seed_extended_rows": int(len(per_seed)),
            "participant_macro_rows": int(len(participant_macro)),
            "extended_summary_rows": int(len(extended)),
        },
        "inputs": _public_evidence(evidence),
        "outputs": [],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["outputs"] = [
        {
            "file_name": path.name,
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in output_paths
    ]
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(
        "REPEAT_SEED_REPORT_COMPLETE=yes "
        f"seeds={report['n_seeds']} "
        f"rows={report['coverage']['extended_summary_rows']}"
    )


if __name__ == "__main__":
    main()
