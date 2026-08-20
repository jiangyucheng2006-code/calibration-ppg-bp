"""Build the leakage-safe Round-8 development screening report.

Full-coverage candidates are compared on the exact K=5 meta-validation query
set.  The beat-similarity threshold is reported separately as a partial-
coverage sensitivity analysis and is never ranked as a full-coverage model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .report_phase6_screening import _bhs_grade, _markdown_table, _pooled_r_squared
from .training import participant_macro_metrics, save_json


KEYS = ["subject_uid", "event_id"]
TARGETS = ["target_sbp", "target_dbp"]
SCOPES = ("Overall", "MIMIC", "VitalDB")
SIMILARITY_SETTING = "R8-6 similarity >=0.90 sensitivity"


def _parse_candidates(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"candidate must be NAME=RUN_DIR, found {value!r}")
        name, raw_path = value.split("=", 1)
        if not name.strip() or name.strip() in result:
            raise ValueError(f"missing or duplicate candidate name: {name!r}")
        result[name.strip()] = Path(raw_path.strip())
    return result


def _prediction_path(run_dir: Path) -> Path:
    paths = [
        run_dir / "predictions.parquet",
        run_dir / "best_validation_predictions.parquet",
    ]
    found = [path for path in paths if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"expected exactly one prediction table under {run_dir}")
    return found[0]


def _load_run(
    setting: str,
    run_dir: Path,
    *,
    expected_seed: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        raise AssertionError(f"{setting} is not complete")
    if record.get("split") != "meta_validation":
        raise AssertionError(f"{setting} is not a meta-validation run")
    if record.get("locked_test_accessed") is not False:
        raise AssertionError(f"{setting} does not explicitly deny locked-test access")
    if expected_seed is not None and int(record.get("seed")) != expected_seed:
        raise AssertionError(f"{setting} seed differs from the Round-8 seed")
    frame = pd.read_parquet(_prediction_path(run_dir))
    if "k" in frame:
        frame = frame.loc[frame["k"].eq(5)].copy()
    missing = set(KEYS + TARGETS + ["pred_sbp", "pred_dbp"]) - set(frame)
    if missing:
        raise ValueError(f"{setting} prediction table misses {sorted(missing)}")
    if frame.duplicated(KEYS).any():
        raise AssertionError(f"{setting} contains duplicate K=5 queries")
    numeric = frame[TARGETS + ["pred_sbp", "pred_dbp"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{setting} contains non-finite values")
    return frame, record


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[KEYS + TARGETS].sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def _validate_full_coverage(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    expected = _canonical(reference)
    observed = _canonical(candidate)
    if not observed[KEYS].equals(expected[KEYS]):
        raise AssertionError("candidate does not use the exact reference query set")
    if not np.allclose(observed[TARGETS], expected[TARGETS], atol=1e-4, rtol=0.0):
        raise AssertionError("candidate targets differ from the reference")


def _scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    return frame if scope == "Overall" else frame.loc[frame["source"].eq(scope)]


def _participant_table(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["sbp_ae"] = (work["pred_sbp"] - work["target_sbp"]).abs()
    work["dbp_ae"] = (work["pred_dbp"] - work["target_dbp"]).abs()
    participants = work.groupby("subject_uid", as_index=False).agg(
        sbp_mae=("sbp_ae", "mean"),
        dbp_mae=("dbp_ae", "mean"),
        n_queries=("event_id", "size"),
    )
    participants["mean_mae"] = (
        participants["sbp_mae"] + participants["dbp_mae"]
    ) / 2.0
    return participants


def _participant_rows(predictions: pd.DataFrame, settings: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in SCOPES:
        for setting in settings:
            frame = _scope(predictions.loc[predictions["Setting"].eq(setting)], scope)
            participant = _participant_table(frame)
            metric = participant_macro_metrics(frame)
            rows.append(
                {
                    "Setting": setting,
                    "Scope": scope,
                    "N participants": int(len(participant)),
                    "N query events": int(len(frame)),
                    "SBP participant-macro MAE": float(metric["sbp_mae"]),
                    "DBP participant-macro MAE": float(metric["dbp_mae"]),
                    "Mean participant-macro MAE": float(metric["mean_mae"]),
                }
            )
    result = pd.DataFrame(rows)
    order = {setting: index for index, setting in enumerate(settings)}
    result["_scope"] = result["Scope"].map({scope: i for i, scope in enumerate(SCOPES)})
    result["_setting"] = result["Setting"].map(order)
    return result.sort_values(["_scope", "_setting"]).drop(
        columns=["_scope", "_setting"]
    ).reset_index(drop=True)


def _diagnostic_rows(predictions: pd.DataFrame, settings: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in SCOPES:
        for setting in settings:
            frame = _scope(predictions.loc[predictions["Setting"].eq(setting)], scope)
            for bp in ("SBP", "DBP"):
                name = bp.lower()
                target = frame[f"target_{name}"].to_numpy(float)
                prediction = frame[f"pred_{name}"].to_numpy(float)
                error = prediction - target
                absolute = np.abs(error)
                within = [float(np.mean(absolute <= threshold) * 100.0) for threshold in (5, 10, 15)]
                grade = _bhs_grade(*within)
                mean_error = float(error.mean())
                std = float(error.std(ddof=1))
                rows.append(
                    {
                        "Setting": setting,
                        "Scope": scope,
                        "BP": bp,
                        "MAE": float(absolute.mean()),
                        "R²": _pooled_r_squared(target, prediction),
                        "ME": mean_error,
                        "STD": std,
                        "≤5 mmHg": within[0],
                        "≤10 mmHg": within[1],
                        "≤15 mmHg": within[2],
                        "AAMI": "PASS*" if abs(mean_error) <= 5.0 and std <= 8.0 else "FAIL*",
                        "BHS": f"{'PASS' if grade in {'A', 'B'} else 'FAIL'} (Grade {grade})*",
                    }
                )
    return pd.DataFrame(rows)


def build_report(
    *,
    reference_name: str,
    reference_dir: Path,
    candidates: dict[str, Path],
    similarity_path: Path,
    output: Path,
    expected_seed: int,
) -> dict[str, object]:
    reference, reference_record = _load_run(
        reference_name, reference_dir, expected_seed=None
    )
    if set(reference.get("k", pd.Series([5])).astype(int)) not in ({5}, set()):
        raise AssertionError("reference must provide K=5 predictions")
    full_frames = [reference.assign(Setting=reference_name)]
    records: dict[str, object] = {reference_name: reference_record}
    for setting, run_dir in candidates.items():
        frame, record = _load_run(setting, run_dir, expected_seed=expected_seed)
        _validate_full_coverage(reference, frame)
        full_frames.append(frame.assign(Setting=setting))
        records[setting] = record
    full = pd.concat(full_frames, ignore_index=True, sort=False)
    if "source" not in reference:
        source_lookup = full.loc[full["source"].notna(), KEYS + ["source"]].drop_duplicates()
        if source_lookup.empty:
            raise AssertionError("no candidate supplied the PulseDB source field")
        full = full.drop(columns=["source"], errors="ignore").merge(
            source_lookup, on=KEYS, how="left", validate="many_to_one"
        )
    if full["source"].isna().any() or set(full["source"].unique()) != {"MIMIC", "VitalDB"}:
        raise AssertionError("source restoration failed")

    similarity = pd.read_parquet(similarity_path)
    required_similarity = set(KEYS + ["pairwise_corr_median"])
    if missing := required_similarity - set(similarity):
        raise ValueError(f"similarity table misses {sorted(missing)}")
    if similarity.duplicated(KEYS).any():
        raise AssertionError("similarity table contains duplicate queries")
    filtered = reference.merge(
        similarity[KEYS + ["pairwise_corr_median"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if filtered["pairwise_corr_median"].isna().all():
        raise AssertionError("similarity audit does not cover the K=5 reference")
    filtered = filtered.loc[
        np.isfinite(filtered["pairwise_corr_median"])
        & filtered["pairwise_corr_median"].ge(0.90)
    ].copy()
    filtered["Setting"] = SIMILARITY_SETTING
    if "source" not in filtered:
        filtered = filtered.merge(
            full.loc[full["Setting"].eq(reference_name), KEYS + ["source"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

    full_settings = [reference_name, *candidates]
    participant = _participant_rows(full, full_settings)
    diagnostics = _diagnostic_rows(full, full_settings)
    filter_participant = _participant_rows(filtered, [SIMILARITY_SETTING])
    filter_diagnostics = _diagnostic_rows(filtered, [SIMILARITY_SETTING])
    coverage_rows = []
    for scope in SCOPES:
        before = _scope(reference.merge(
            full.loc[full["Setting"].eq(reference_name), KEYS + ["source"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        ) if "source" not in reference else reference, scope)
        after = _scope(filtered, scope)
        coverage_rows.append(
            {
                "Scope": scope,
                "Threshold": ">=0.90 and finite",
                "Queries before": int(len(before)),
                "Queries retained": int(len(after)),
                "Query coverage (%)": float(100.0 * len(after) / len(before)),
                "Participants before": int(before["subject_uid"].nunique()),
                "Participants retained": int(after["subject_uid"].nunique()),
                "Participant coverage (%)": float(
                    100.0 * after["subject_uid"].nunique() / before["subject_uid"].nunique()
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    output.mkdir(parents=True, exist_ok=False)
    participant.to_csv(output / "participant_macro_full_coverage.csv", index=False)
    diagnostics.to_csv(output / "diagnostic_full_coverage.csv", index=False)
    filter_participant.to_csv(output / "similarity_filter_participant_macro.csv", index=False)
    filter_diagnostics.to_csv(output / "similarity_filter_diagnostic.csv", index=False)
    coverage.to_csv(output / "similarity_filter_coverage.csv", index=False)
    best = participant.loc[participant["Scope"].eq("Overall")].sort_values(
        "Mean participant-macro MAE"
    ).iloc[0]
    payload = {
        "status": "complete",
        "split": "meta_validation",
        "locked_test_accessed": False,
        "k": 5,
        "reference": reference_name,
        "seed": expected_seed,
        "full_coverage_settings": full_settings,
        "full_coverage_queries_per_setting": int(reference.shape[0]),
        "full_coverage_participants_per_setting": int(reference["subject_uid"].nunique()),
        "best_full_coverage_setting": str(best["Setting"]),
        "best_full_coverage_mean_mae": float(best["Mean participant-macro MAE"]),
        "similarity_filter_is_partial_coverage_sensitivity": True,
        "similarity_filter_threshold": 0.90,
        "run_records": records,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output / "run.json", payload)
    lines = [
        "# Round-8 calibration-relative screening",
        "",
        "Development-only, single-seed K=5 screening. The locked meta-test was not accessed.",
        "",
        "## Full-coverage participant-macro results",
        "",
        _markdown_table(participant),
        "",
        "## Similarity threshold coverage",
        "",
        _markdown_table(coverage),
        "",
        "The similarity-threshold result is a partial-coverage sensitivity analysis, not a deployable full-coverage model and not eligible to win the model screen.",
        "",
        "MIMIC and VitalDB are internal PulseDB source strata, not independent external validation datasets. AAMI/BHS fields are retrospective numerical screens only and do not establish device compliance.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-name", default="Quality Gate + Huber (256-D)")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--similarity-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seed", type=int, default=20260822)
    args = parser.parse_args()
    result = build_report(
        reference_name=args.reference_name,
        reference_dir=args.reference_dir,
        candidates=_parse_candidates(args.candidate),
        similarity_path=args.similarity_path,
        output=args.output,
        expected_seed=args.expected_seed,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
