"""Development-only fixed-reference worst-tail oracle diagnostics.

This module deliberately uses observed meta-validation query errors to define a
participant tail.  The resulting retained-cohort and routing-mixture metrics are
oracle diagnostics only: they are not deployable screening or accuracy claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "subject_uid",
    "event_id",
    "k",
    "target_sbp",
    "target_dbp",
    "pred_sbp",
    "pred_dbp",
}
SOURCE_SCOPES = ("MIMIC", "VitalDB")


def _percent_label(fraction: float) -> str:
    percent = fraction * 100.0
    rounded = round(percent)
    if math.isclose(percent, rounded, abs_tol=1e-9):
        return str(int(rounded))
    return f"{percent:g}".replace(".", "p")


def _load_development_predictions(
    run_dir: Path, *, k: int
) -> tuple[dict[str, object], pd.DataFrame]:
    run_path = run_dir / "run.json"
    prediction_path = run_dir / "best_validation_predictions.parquet"
    if not run_path.is_file():
        raise FileNotFoundError(f"missing run metadata: {run_path}")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"missing validation predictions: {prediction_path}")

    metadata = json.loads(run_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError(f"run is not complete: {run_dir}")
    if metadata.get("split") != "meta_validation":
        raise ValueError("oracle analysis accepts meta_validation predictions only")
    if metadata.get("locked_test_accessed") is not False:
        raise ValueError("run metadata does not prove locked-test non-access")

    frame = pd.read_parquet(prediction_path)
    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    frame = frame.loc[frame["k"].eq(k)].copy()
    if frame.empty:
        raise ValueError(f"no predictions for K={k}: {prediction_path}")
    if frame.duplicated(["subject_uid", "event_id"]).any():
        raise AssertionError("duplicate participant/query keys within selected K")
    numeric = frame[["target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("prediction table contains non-finite targets or predictions")

    frame["source"] = frame["subject_uid"].str.split(":", n=1).str[0]
    unknown_sources = set(frame["source"]) - set(SOURCE_SCOPES)
    if unknown_sources:
        raise ValueError(f"unexpected PulseDB sources: {sorted(unknown_sources)}")
    if set(frame["source"]) != set(SOURCE_SCOPES):
        raise ValueError("both MIMIC and VitalDB must be present")
    return metadata, frame


def _assert_common_queries(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    order = ["subject_uid", "event_id"]
    reference_ordered = reference.sort_values(order).reset_index(drop=True)
    candidate_ordered = candidate.sort_values(order).reset_index(drop=True)
    if not candidate_ordered[order].equals(reference_ordered[order]):
        raise AssertionError("candidate query keys differ from the fixed reference")
    for target in ("target_sbp", "target_dbp"):
        if not np.allclose(
            candidate_ordered[target].to_numpy(dtype=float),
            reference_ordered[target].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-4,
        ):
            raise AssertionError(f"candidate {target} differs from the fixed reference")


def _participant_metrics(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["abs_error_sbp"] = (frame["pred_sbp"] - frame["target_sbp"]).abs()
    frame["abs_error_dbp"] = (frame["pred_dbp"] - frame["target_dbp"]).abs()
    participant = frame.groupby(
        ["subject_uid", "source"], as_index=False, sort=True
    ).agg(
        n_queries=("event_id", "size"),
        sbp_mae=("abs_error_sbp", "mean"),
        dbp_mae=("abs_error_dbp", "mean"),
    )
    participant["mean_mae"] = (
        participant["sbp_mae"] + participant["dbp_mae"]
    ) / 2.0
    return participant


def _assign_fixed_tail(
    participant: pd.DataFrame, *, tail_fraction: float
) -> tuple[pd.DataFrame, int, str, str, str]:
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must be strictly between zero and one")
    tail_label = _percent_label(tail_fraction)
    retained_label = _percent_label(1.0 - tail_fraction)
    flag = f"is_worst_{tail_label}pct_reference"
    tail_name = f"worst_{tail_label}pct_subjects"
    retained_name = f"oracle_retained_{retained_label}pct_subjects"
    ordered = participant.sort_values(
        ["mean_mae", "subject_uid"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    tail_count = int(math.ceil(tail_fraction * len(ordered)))
    ordered["reference_error_rank"] = np.arange(1, len(ordered) + 1)
    ordered[flag] = ordered["reference_error_rank"].le(tail_count)
    ordered["reference_tail_group"] = np.where(
        ordered[flag], tail_name, retained_name
    )
    if int(ordered[flag].sum()) != tail_count:
        raise AssertionError("deterministic tail assignment returned the wrong size")
    return ordered, tail_count, flag, tail_name, retained_name


def _scoped_frames(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [("Overall", frame)] + [
        (source, frame.loc[frame["source"].eq(source)])
        for source in SOURCE_SCOPES
    ]


def _fixed_tail_summary(
    participant: pd.DataFrame,
    *,
    setting: str,
    flag: str,
    tail_name: str,
    retained_name: str,
    selection_rule: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cohorts = (
        ("overall", participant),
        (tail_name, participant.loc[participant[flag]]),
        (retained_name, participant.loc[~participant[flag]]),
    )
    for source_scope, source_frame in _scoped_frames(participant):
        for cohort, _ in cohorts:
            if cohort == "overall":
                selected = source_frame
            elif cohort == tail_name:
                selected = source_frame.loc[source_frame[flag]]
            else:
                selected = source_frame.loc[~source_frame[flag]]
            rows.append(
                {
                    "setting": setting,
                    "source_scope": source_scope,
                    "cohort": cohort,
                    "participants": int(len(selected)),
                    "queries": int(selected["n_queries"].sum()),
                    "participant_macro_sbp_mae": float(selected["sbp_mae"].mean()),
                    "participant_macro_dbp_mae": float(selected["dbp_mae"].mean()),
                    "participant_macro_mean_mae": float(selected["mean_mae"].mean()),
                    "tail_membership_from_setting": "reference",
                    "selection_rule": selection_rule,
                    "oracle_only": cohort != "overall",
                    "deployable": cohort == "overall",
                }
            )
    return pd.DataFrame(rows)


def _routing_summary(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    candidate_setting: str,
    flag: str,
    tail_name: str,
    retained_name: str,
    selection_rule: str,
) -> pd.DataFrame:
    columns = [
        "subject_uid",
        "source",
        "n_queries",
        "sbp_mae",
        "dbp_mae",
        "mean_mae",
        flag,
    ]
    merged = reference[columns].merge(
        candidate.drop(columns=[flag], errors="ignore"),
        on=["subject_uid", "source"],
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_candidate"),
    )
    if len(merged) != len(reference):
        raise AssertionError("candidate participant set differs from the reference")
    for bp in ("sbp", "dbp"):
        merged[f"routed_{bp}_mae"] = np.where(
            merged[flag],
            merged[f"{bp}_mae_candidate"],
            merged[f"{bp}_mae_reference"],
        )
    merged["routed_mean_mae"] = (
        merged["routed_sbp_mae"] + merged["routed_dbp_mae"]
    ) / 2.0

    rows: list[dict[str, object]] = []
    for source_scope, selected in _scoped_frames(merged):
        reference_mean = float(selected["mean_mae_reference"].mean())
        routed_mean = float(selected["routed_mean_mae"].mean())
        rows.append(
            {
                "candidate_setting": candidate_setting,
                "source_scope": source_scope,
                "participants": int(len(selected)),
                "queries": int(selected["n_queries_reference"].sum()),
                "tail_participants_routed_to_candidate": int(selected[flag].sum()),
                "retained_participants_routed_to_reference": int((~selected[flag]).sum()),
                "participant_macro_sbp_mae": float(selected["routed_sbp_mae"].mean()),
                "participant_macro_dbp_mae": float(selected["routed_dbp_mae"].mean()),
                "participant_macro_mean_mae": routed_mean,
                "reference_mean_mae": reference_mean,
                "delta_mean_mae_vs_reference": routed_mean - reference_mean,
                "improvement_vs_reference_mean_mae": reference_mean - routed_mean,
                "routing_policy": (
                    f"reference on {retained_name}; {candidate_setting} on {tail_name}"
                ),
                "selection_rule": selection_rule,
                "oracle_only": True,
                "deployable": False,
            }
        )
    return pd.DataFrame(rows)


def analyze(
    reference_run: Path,
    candidate_runs: dict[str, Path],
    output_dir: Path,
    *,
    k: int = 5,
    tail_fraction: float = 0.30,
) -> dict[str, object]:
    """Create fixed-reference tail and oracle-routing diagnostics."""

    if not candidate_runs:
        raise ValueError("at least one candidate run is required")
    if len(candidate_runs) != len(set(candidate_runs)):
        raise ValueError("candidate setting names must be unique")

    reference_metadata, reference_events = _load_development_predictions(
        reference_run, k=k
    )
    reference_participant = _participant_metrics(reference_events)
    (
        reference_participant,
        tail_count,
        flag,
        tail_name,
        retained_name,
    ) = _assign_fixed_tail(reference_participant, tail_fraction=tail_fraction)
    selection_rule = (
        "reference participant mean MAE descending, subject_uid ascending; "
        f"first ceil({tail_fraction:.6g} * N) participants"
    )

    fixed_tail_tables = [
        _fixed_tail_summary(
            reference_participant,
            setting="reference",
            flag=flag,
            tail_name=tail_name,
            retained_name=retained_name,
            selection_rule=selection_rule,
        )
    ]
    routing_tables: list[pd.DataFrame] = []
    candidate_metadata: dict[str, dict[str, object]] = {}
    for setting in sorted(candidate_runs):
        run_dir = candidate_runs[setting]
        metadata, events = _load_development_predictions(run_dir, k=k)
        _assert_common_queries(reference_events, events)
        participant = _participant_metrics(events).merge(
            reference_participant[["subject_uid", "source", flag]],
            on=["subject_uid", "source"],
            how="inner",
            validate="one_to_one",
        )
        if len(participant) != len(reference_participant):
            raise AssertionError("candidate participant set differs from the reference")
        fixed_tail_tables.append(
            _fixed_tail_summary(
                participant,
                setting=setting,
                flag=flag,
                tail_name=tail_name,
                retained_name=retained_name,
                selection_rule=selection_rule,
            )
        )
        routing_tables.append(
            _routing_summary(
                reference_participant,
                participant,
                candidate_setting=setting,
                flag=flag,
                tail_name=tail_name,
                retained_name=retained_name,
                selection_rule=selection_rule,
            )
        )
        candidate_metadata[setting] = {
            "run_directory_name": run_dir.name,
            "job_id": metadata.get("slurm_job_id"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        }

    fixed_tail_metrics = pd.concat(fixed_tail_tables, ignore_index=True)
    routing_metrics = pd.concat(routing_tables, ignore_index=True)
    tail_label = _percent_label(tail_fraction)
    retained_label = _percent_label(1.0 - tail_fraction)

    output_dir.mkdir(parents=True, exist_ok=False)
    membership_name = f"reference_worst_{tail_label}pct_membership_k{k}.csv"
    fixed_name = f"fixed_reference_tail_metrics_k{k}.csv"
    routing_name = f"oracle_routing_mixture_k{k}.csv"
    reference_participant.to_csv(output_dir / membership_name, index=False)
    fixed_tail_metrics.to_csv(output_dir / fixed_name, index=False)
    routing_metrics.to_csv(output_dir / routing_name, index=False)

    source_counts = (
        reference_participant.groupby("source")["subject_uid"].nunique().to_dict()
    )
    summary: dict[str, object] = {
        "analysis": "development-only fixed-reference worst-tail oracle diagnostic",
        "split": "meta_validation",
        "locked_meta_test_accessed": False,
        "k": int(k),
        "participants": int(len(reference_participant)),
        "source_participants": {key: int(value) for key, value in source_counts.items()},
        "tail_fraction": float(tail_fraction),
        "tail_percentage_label": tail_label,
        "retained_percentage_label": retained_label,
        "tail_participants": int(tail_count),
        "retained_participants": int(len(reference_participant) - tail_count),
        "tail_group": tail_name,
        "retained_group": retained_name,
        "tail_selection_rule": selection_rule,
        "tail_membership_is_global": True,
        "reference_run": {
            "run_directory_name": reference_run.name,
            "job_id": reference_metadata.get("slurm_job_id"),
            "checkpoint_sha256": reference_metadata.get("checkpoint_sha256"),
        },
        "candidate_runs": candidate_metadata,
        "outputs": {
            "reference_membership": membership_name,
            "fixed_tail_metrics": fixed_name,
            "oracle_routing_mixture": routing_name,
        },
        "oracle_warning": (
            "Worst-tail membership, retained-cohort accuracy, and routing mixtures use "
            "observed meta-validation query error. They are diagnostic, non-deployable, "
            "and must not be reported as achievable screening performance."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def _candidate_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                "candidate must have the form SETTING=/path/to/run"
            )
        setting, path = value.split("=", 1)
        setting = setting.strip()
        if not setting or setting in result:
            raise argparse.ArgumentTypeError("candidate setting is empty or duplicated")
        result[setting] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument(
        "--candidate-run",
        action="append",
        default=[],
        metavar="SETTING=RUN_DIR",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tail-fraction", type=float, default=0.30)
    args = parser.parse_args()
    summary = analyze(
        args.reference_run,
        _candidate_mapping(args.candidate_run),
        args.output_dir,
        k=args.k,
        tail_fraction=args.tail_fraction,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
