"""Build a leakage-safe Phase-6 single-seed screening report.

The report restores the PulseDB source from frozen development metadata and
reports the same saved predictions at three scopes: Overall, MIMIC, and
VitalDB.  Participant-macro MAE is primary.  Event-pooled AAMI/BHS columns are
retrospective numerical screens only and are not device-compliance claims.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

KS = (1, 2, 3, 5)
PREDICTION_COLUMNS = {
    "subject_uid",
    "event_id",
    "k",
    "target_sbp",
    "target_dbp",
    "pred_sbp",
    "pred_dbp",
}
KEY_COLUMNS = ["subject_uid", "event_id"]
TARGET_COLUMNS = ["target_sbp", "target_dbp"]
SOURCES = ("MIMIC", "VitalDB")
SCOPES = ("Overall", *SOURCES)
EVENT_TABLE_COLUMNS = [
    "Setting",
    "BP",
    "MAE",
    "R²",
    "ME",
    "STD",
    "≤5 mmHg",
    "≤10 mmHg",
    "≤15 mmHg",
    "AAMI",
    "BHS",
]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_store_metadata(store_root: Path) -> pd.DataFrame:
    paths = sorted(store_root.glob("development_metadata_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no development metadata shards found under {store_root}"
        )
    frames = [pd.read_parquet(path) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    if result["event_id"].duplicated().any():
        raise AssertionError("duplicate event IDs in development metadata")
    return result


def _bhs_grade(within_5: float, within_10: float, within_15: float) -> str:
    if within_5 >= 60.0 and within_10 >= 85.0 and within_15 >= 95.0:
        return "A"
    if within_5 >= 50.0 and within_10 >= 75.0 and within_15 >= 90.0:
        return "B"
    if within_5 >= 40.0 and within_10 >= 65.0 and within_15 >= 85.0:
        return "C"
    return "D"


def _pooled_r_squared(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((prediction - target) ** 2)) / denominator


def _scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "Overall":
        return frame
    return frame.loc[frame["source"].eq(scope)]


def _setting_label(setting: str, k: int) -> str:
    return f"{setting} (K={k})"


def _parse_run_specs(values: Sequence[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"run specification must be SETTING=PATH, found {value!r}")
        setting, raw_path = value.split("=", 1)
        setting = setting.strip()
        raw_path = raw_path.strip()
        if not setting or not raw_path:
            raise ValueError(f"run specification must be SETTING=PATH, found {value!r}")
        if setting in runs:
            raise ValueError(f"duplicate setting in --run: {setting}")
        runs[setting] = Path(raw_path)
    return runs


def _load_run(
    setting: str,
    run_dir: Path,
    *,
    expected_seed: int | None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    run_path = run_dir / "run.json"
    predictions_path = run_dir / "best_validation_predictions.parquet"
    if not run_path.is_file():
        raise FileNotFoundError(f"missing run metadata for {setting}: {run_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"missing validation predictions for {setting}: {predictions_path}"
        )
    record = json.loads(run_path.read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        raise AssertionError(f"{setting} run status is not complete")
    if record.get("split") != "meta_validation":
        raise AssertionError(f"{setting} is not a meta-validation run")
    if record.get("locked_test_accessed") is not False:
        raise AssertionError(f"{setting} does not explicitly deny locked-test access")
    if record.get("method") != "m0":
        raise AssertionError(f"{setting} is not an M0 Phase-6 candidate")
    seed = int(record.get("seed"))
    if expected_seed is not None and seed != expected_seed:
        raise AssertionError(
            f"{setting} seed {seed} differs from expected seed {expected_seed}"
        )
    arguments = record.get("arguments")
    if not isinstance(arguments, dict):
        raise AssertionError(f"{setting} run metadata lacks an arguments mapping")
    if arguments.get("train_support_policy") != "fixed_first":
        raise AssertionError(f"{setting} did not use fixed_first meta-training support")

    predictions = pd.read_parquet(predictions_path)
    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"{setting} predictions are missing {sorted(missing)}")
    predictions = predictions[
        ["subject_uid", "event_id", "k", *TARGET_COLUMNS, "pred_sbp", "pred_dbp"]
    ].copy()
    predictions["k"] = pd.to_numeric(predictions["k"], errors="raise").astype(int)
    if set(predictions["k"]) != set(KS):
        raise AssertionError(f"{setting} does not contain exactly K={KS}")
    if predictions.duplicated(["k", *KEY_COLUMNS]).any():
        raise AssertionError(f"{setting} contains duplicate K/query prediction rows")
    numeric = predictions[[*TARGET_COLUMNS, "pred_sbp", "pred_dbp"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all():
        raise ValueError(f"{setting} contains nonfinite targets or predictions")
    predictions.insert(0, "setting", setting)
    evidence = {
        "setting": setting,
        "run_directory_name": run_dir.name,
        "run_json": {"file_name": run_path.name, "sha256": _file_sha256(run_path)},
        "predictions": {
            "file_name": predictions_path.name,
            "sha256": _file_sha256(predictions_path),
            "rows": int(len(predictions)),
        },
    }
    return predictions, record, evidence


def _restore_and_validate_source(
    predictions: pd.DataFrame,
    *,
    store_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = _load_store_metadata(store_root)
    required = {"subject_uid", "event_id", "split", "common_query", "source", "sbp", "dbp"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"development metadata are missing {sorted(missing)}")
    validation = metadata.loc[
        metadata["split"].eq("meta_validation") & metadata["common_query"],
        ["subject_uid", "event_id", "source", "sbp", "dbp"],
    ].copy()
    if validation.duplicated(KEY_COLUMNS).any():
        raise AssertionError("meta-validation common-query metadata are not unique")
    observed_sources = set(validation["source"].dropna().astype(str))
    if observed_sources != set(SOURCES):
        raise AssertionError(
            f"expected meta-validation sources {SOURCES}, found {sorted(observed_sources)}"
        )
    subject_sources = validation.groupby("subject_uid")["source"].nunique()
    if not subject_sources.eq(1).all():
        raise AssertionError("a participant is associated with multiple sources")

    expected_keys = validation[KEY_COLUMNS].sort_values(KEY_COLUMNS, kind="mergesort")
    expected_keys = expected_keys.reset_index(drop=True)
    metadata_targets = validation[
        [*KEY_COLUMNS, "sbp", "dbp"]
    ].sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    for (setting, k), group in predictions.groupby(["setting", "k"], sort=False):
        observed = group[[*KEY_COLUMNS, *TARGET_COLUMNS]].sort_values(
            KEY_COLUMNS, kind="mergesort"
        ).reset_index(drop=True)
        if not observed[KEY_COLUMNS].equals(expected_keys):
            raise AssertionError(
                f"{setting}, K={k} does not use the frozen meta-validation common queries"
            )
        if not np.allclose(
            observed[TARGET_COLUMNS].to_numpy(dtype=float),
            metadata_targets[["sbp", "dbp"]].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-4,
        ):
            raise AssertionError(
                f"{setting}, K={k} targets differ from frozen development metadata"
            )

    source_lookup = validation[[*KEY_COLUMNS, "source"]]
    enriched = predictions.merge(
        source_lookup,
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    if enriched["source"].isna().any():
        raise AssertionError("at least one prediction could not be assigned a source")
    return enriched, validation


def _assert_common_predictions(
    predictions: pd.DataFrame,
    *,
    reference_setting: str,
) -> None:
    reference = predictions.loc[predictions["setting"].eq(reference_setting)]
    if reference.empty:
        raise KeyError(f"reference setting not found: {reference_setting}")
    for k in KS:
        canonical = reference.loc[reference["k"].eq(k), [*KEY_COLUMNS, *TARGET_COLUMNS]]
        canonical = canonical.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(
            drop=True
        )
        for setting, group in predictions.loc[predictions["k"].eq(k)].groupby(
            "setting", sort=False
        ):
            observed = group[[*KEY_COLUMNS, *TARGET_COLUMNS]].sort_values(
                KEY_COLUMNS, kind="mergesort"
            ).reset_index(drop=True)
            if not observed[KEY_COLUMNS].equals(canonical[KEY_COLUMNS]):
                raise AssertionError(
                    f"{setting}, K={k} differs from the reference common-query set"
                )
            if not np.allclose(
                observed[TARGET_COLUMNS].to_numpy(dtype=float),
                canonical[TARGET_COLUMNS].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-4,
            ):
                raise AssertionError(f"{setting}, K={k} differs from reference targets")


def _participant_errors(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["abs_error_sbp"] = (frame["pred_sbp"] - frame["target_sbp"]).abs()
    frame["abs_error_dbp"] = (frame["pred_dbp"] - frame["target_dbp"]).abs()
    participant = frame.groupby("subject_uid", as_index=False).agg(
        sbp_mae=("abs_error_sbp", "mean"),
        dbp_mae=("abs_error_dbp", "mean"),
        n_query_events=("event_id", "size"),
    )
    participant["mean_mae"] = (participant["sbp_mae"] + participant["dbp_mae"]) / 2.0
    return participant


def _exact_worst_fraction(
    participant: pd.DataFrame,
    *,
    fraction: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("tail fraction must be between zero and one")
    if participant.empty:
        raise ValueError("participant table is empty")
    ordered = participant.sort_values(
        ["mean_mae", "subject_uid"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    tail_n = max(1, int(math.ceil(len(ordered) * fraction)))
    tail = ordered.iloc[:tail_n].copy()
    remaining = ordered.iloc[tail_n:].copy()
    return tail, remaining, float(tail["mean_mae"].iloc[-1])


def _participant_macro_table(
    predictions: pd.DataFrame,
    *,
    setting_order: Mapping[str, int],
    reference_setting: str,
) -> tuple[pd.DataFrame, dict[tuple[str, int, str], pd.DataFrame]]:
    tables: dict[tuple[str, int, str], pd.DataFrame] = {}
    for scope in SCOPES:
        scoped = _scope(predictions, scope)
        for (setting, k), events in scoped.groupby(["setting", "k"], sort=False):
            tables[(scope, int(k), str(setting))] = _participant_errors(events)

    rows: list[dict[str, object]] = []
    for (scope, k, setting), participant in tables.items():
        reference = tables[(scope, k, reference_setting)][
            ["subject_uid", "mean_mae"]
        ].rename(columns={"mean_mae": "reference_mean_mae"})
        paired = participant.merge(
            reference,
            on="subject_uid",
            how="left",
            validate="one_to_one",
        )
        if paired["reference_mean_mae"].isna().any():
            raise AssertionError("candidate and reference participant sets differ")
        rows.append(
            {
                "Setting": setting,
                "Scope": scope,
                "K": k,
                "N participants": int(len(participant)),
                "N query events": int(participant["n_query_events"].sum()),
                "SBP participant-macro MAE": float(participant["sbp_mae"].mean()),
                "DBP participant-macro MAE": float(participant["dbp_mae"].mean()),
                "Mean participant-macro MAE": float(participant["mean_mae"].mean()),
                "Paired delta vs reference": float(
                    (paired["mean_mae"] - paired["reference_mean_mae"]).mean()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["_setting_order"] = result["Setting"].map(setting_order)
    result["_scope_order"] = result["Scope"].map({name: i for i, name in enumerate(SCOPES)})
    result = result.sort_values(
        ["_scope_order", "_setting_order", "K"], kind="mergesort"
    ).drop(columns=["_setting_order", "_scope_order"])
    return result.reset_index(drop=True), tables


def _event_pooled_tables(
    predictions: pd.DataFrame,
    *,
    setting_order: Mapping[str, int],
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for scope in SCOPES:
        scoped = _scope(predictions, scope)
        rows: list[dict[str, object]] = []
        for (setting, k), group in scoped.groupby(["setting", "k"], sort=False):
            for bp in ("SBP", "DBP"):
                lower = bp.lower()
                target = group[f"target_{lower}"].to_numpy(dtype=float)
                prediction = group[f"pred_{lower}"].to_numpy(dtype=float)
                error = prediction - target
                absolute_error = np.abs(error)
                within_5 = float(np.mean(absolute_error <= 5.0) * 100.0)
                within_10 = float(np.mean(absolute_error <= 10.0) * 100.0)
                within_15 = float(np.mean(absolute_error <= 15.0) * 100.0)
                mean_error = float(error.mean())
                error_std = float(error.std(ddof=1))
                grade = _bhs_grade(within_5, within_10, within_15)
                rows.append(
                    {
                        "Setting": _setting_label(str(setting), int(k)),
                        "BP": bp,
                        "MAE": float(absolute_error.mean()),
                        "R²": _pooled_r_squared(target, prediction),
                        "ME": mean_error,
                        "STD": error_std,
                        "≤5 mmHg": within_5,
                        "≤10 mmHg": within_10,
                        "≤15 mmHg": within_15,
                        "AAMI": "PASS*"
                        if abs(mean_error) <= 5.0 and error_std <= 8.0
                        else "FAIL*",
                        "BHS": f"{'PASS' if grade in {'A', 'B'} else 'FAIL'} (Grade {grade})*",
                        "_setting": setting,
                        "_k": int(k),
                        "_bp_order": 0 if bp == "SBP" else 1,
                    }
                )
        result = pd.DataFrame(rows)
        result["_setting_order"] = result["_setting"].map(setting_order)
        result = result.sort_values(
            ["_setting_order", "_k", "_bp_order"], kind="mergesort"
        )
        results[scope] = result[EVENT_TABLE_COLUMNS].reset_index(drop=True)
    return results


def _tail_tables(
    participant_tables: Mapping[tuple[str, int, str], pd.DataFrame],
    *,
    setting_order: Mapping[str, int],
    reference_setting: str,
    tail_fraction: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tail_rows: list[dict[str, object]] = []
    reference_tail_rows: list[dict[str, object]] = []
    for (scope, k, setting), participant in participant_tables.items():
        tail, remaining, threshold = _exact_worst_fraction(
            participant, fraction=tail_fraction
        )
        tail_rows.append(
            {
                "Setting": setting,
                "Scope": scope,
                "K": k,
                "N participants": int(len(participant)),
                "Worst 30% participants": int(len(tail)),
                "Worst 30% threshold": threshold,
                "Worst 30% mean MAE": float(tail["mean_mae"].mean()),
                "Remaining 70% mean MAE": float(remaining["mean_mae"].mean())
                if not remaining.empty
                else float("nan"),
                "P90 participant mean MAE": float(participant["mean_mae"].quantile(0.90)),
                "P95 participant mean MAE": float(participant["mean_mae"].quantile(0.95)),
                "P99 participant mean MAE": float(participant["mean_mae"].quantile(0.99)),
                "Selection": "observed query error; oracle diagnostic only",
            }
        )

    for scope in SCOPES:
        for k in KS:
            reference = participant_tables[(scope, k, reference_setting)]
            reference_tail, _, _ = _exact_worst_fraction(
                reference, fraction=tail_fraction
            )
            tail_ids = set(reference_tail["subject_uid"])
            for setting in setting_order:
                participant = participant_tables[(scope, k, setting)]
                selected = participant.loc[participant["subject_uid"].isin(tail_ids)]
                if len(selected) != len(reference_tail):
                    raise AssertionError("candidate does not cover the reference-tail participants")
                reference_tail_rows.append(
                    {
                        "Setting": setting,
                        "Scope": scope,
                        "K": k,
                        "Reference tail setting": reference_setting,
                        "Reference-tail participants": int(len(selected)),
                        "Mean MAE on reference worst 30%": float(
                            selected["mean_mae"].mean()
                        ),
                        "Selection": "reference observed query error; oracle diagnostic only",
                    }
                )

    def ordered(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["_setting_order"] = result["Setting"].map(setting_order)
        result["_scope_order"] = result["Scope"].map(
            {name: i for i, name in enumerate(SCOPES)}
        )
        return result.sort_values(
            ["_scope_order", "_setting_order", "K"], kind="mergesort"
        ).drop(columns=["_setting_order", "_scope_order"]).reset_index(drop=True)

    return ordered(pd.DataFrame(tail_rows)), ordered(pd.DataFrame(reference_tail_rows))


def _format_cell(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NA"
        return f"{float(value):.3f}"
    return str(value).replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _write_markdown(
    path: Path,
    *,
    participant_macro: pd.DataFrame,
    event_tables: Mapping[str, pd.DataFrame],
    tail: pd.DataFrame,
    reference_tail: pd.DataFrame,
    reference_setting: str,
    seed: int,
) -> None:
    lines = [
        "# Phase-6 fixed-first single-seed screening results",
        "",
        f"- Development seed: `{seed}`",
        "- Split: `meta_validation` only; locked meta-test was not accessed.",
        f"- Reference: `{reference_setting}`.",
        "- Participant-macro MAE is the primary project metric.",
        "- MIMIC and VitalDB are PulseDB source strata, not independent external validation datasets.",
        "- AAMI/BHS entries are retrospective numerical screens only; formal compliance is not established.",
        "- Worst-30% and remaining-70% rows use observed query error and are oracle diagnostics, not deployable filters.",
        "",
        "## Participant-macro primary results",
        "",
        _markdown_table(participant_macro),
    ]
    for scope in SCOPES:
        lines.extend(
            [
                "",
                f"## {scope} event-pooled diagnostics",
                "",
                _markdown_table(event_tables[scope]),
            ]
        )
    lines.extend(
        [
            "",
            "## Method-specific observed-error tail diagnostics",
            "",
            _markdown_table(tail),
            "",
            "## Performance on the reference model's observed-error worst 30%",
            "",
            _markdown_table(reference_tail),
            "",
            "No candidate is automatically promoted by this report. Promotion requires a prespecified development decision and later repeated-seed confirmation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_scope_name(scope: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", scope.lower()).strip("_")


def generate_report(
    *,
    runs: Mapping[str, Path],
    reference_setting: str,
    store_root: Path,
    output_dir: Path,
    expected_seed: int | None = None,
) -> dict[str, object]:
    if len(runs) < 2:
        raise ValueError("at least a reference and one candidate run are required")
    if reference_setting not in runs:
        raise KeyError(f"reference setting not found: {reference_setting}")
    setting_order = {setting: index for index, setting in enumerate(runs)}

    frames: list[pd.DataFrame] = []
    records: dict[str, dict[str, object]] = {}
    evidence: list[dict[str, object]] = []
    inferred_seed: int | None = expected_seed
    for setting, run_dir in runs.items():
        predictions, record, item = _load_run(
            setting,
            Path(run_dir),
            expected_seed=expected_seed,
        )
        seed = int(record["seed"])
        if inferred_seed is None:
            inferred_seed = seed
        elif seed != inferred_seed:
            raise AssertionError("Phase-6 screening runs use different seeds")
        frames.append(predictions)
        records[setting] = record
        evidence.append(item)
    assert inferred_seed is not None

    combined = pd.concat(frames, ignore_index=True)
    _assert_common_predictions(combined, reference_setting=reference_setting)
    combined, validation_metadata = _restore_and_validate_source(
        combined,
        store_root=store_root,
    )
    participant_macro, participant_tables = _participant_macro_table(
        combined,
        setting_order=setting_order,
        reference_setting=reference_setting,
    )
    event_tables = _event_pooled_tables(combined, setting_order=setting_order)
    tail, reference_tail = _tail_tables(
        participant_tables,
        setting_order=setting_order,
        reference_setting=reference_setting,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    participant_path = output_dir / "phase6_participant_macro_by_scope.csv"
    tail_path = output_dir / "phase6_oracle_tail_by_scope.csv"
    reference_tail_path = output_dir / "phase6_reference_oracle_tail_comparison.csv"
    markdown_path = output_dir / "PHASE6_SCREENING_RESULTS.md"
    participant_macro.to_csv(participant_path, index=False, float_format="%.6f")
    tail.to_csv(tail_path, index=False, float_format="%.6f")
    reference_tail.to_csv(reference_tail_path, index=False, float_format="%.6f")
    event_paths: dict[str, Path] = {}
    for scope, table in event_tables.items():
        event_path = output_dir / f"phase6_{_safe_scope_name(scope)}_metrics.csv"
        table.to_csv(event_path, index=False, float_format="%.6f")
        event_paths[scope] = event_path
    _write_markdown(
        markdown_path,
        participant_macro=participant_macro,
        event_tables=event_tables,
        tail=tail,
        reference_tail=reference_tail,
        reference_setting=reference_setting,
        seed=inferred_seed,
    )

    output_paths = [
        participant_path,
        tail_path,
        reference_tail_path,
        markdown_path,
        *event_paths.values(),
    ]
    report: dict[str, object] = {
        "status": "pass",
        "analysis": "Phase-6 fixed-first single-seed development screening",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "meta_validation",
        "locked_test_accessed": False,
        "seed": inferred_seed,
        "reference_setting": reference_setting,
        "settings": list(runs),
        "k": list(KS),
        "scopes": list(SCOPES),
        "sources": list(SOURCES),
        "source_interpretation": "PulseDB source strata; not independent external validation",
        "primary_metric": "participant-macro MAE",
        "event_table_scope": "event-pooled secondary diagnostics",
        "aami_bhs_warning": "retrospective numerical screens only; formal compliance not established",
        "oracle_tail_warning": (
            "Worst-30% membership and remaining-70% accuracy use observed query error. "
            "They are diagnostic upper bounds and cannot be used as deployment filters."
        ),
        "coverage": {
            scope: {
                "participants": int(_scope(validation_metadata, scope)["subject_uid"].nunique()),
                "common_query_events_per_setting_k": int(len(_scope(validation_metadata, scope))),
            }
            for scope in SCOPES
        },
        "inputs": evidence,
        "outputs": [
            {"file_name": path.name, "sha256": _file_sha256(path)}
            for path in output_paths
        ],
    }
    report_path = output_dir / "phase6_screening_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SETTING=RUN_DIR",
        help="repeat for the reference and every Phase-6 candidate",
    )
    parser.add_argument("--reference-setting", required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seed", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = generate_report(
        runs=_parse_run_specs(args.run),
        reference_setting=args.reference_setting,
        store_root=args.store_root,
        output_dir=args.output_dir,
        expected_seed=args.expected_seed,
    )
    print(
        "PHASE6_SCREENING_REPORT_COMPLETE=yes "
        f"settings={len(report['settings'])} scopes={len(report['scopes'])}"
    )


if __name__ == "__main__":
    main()
