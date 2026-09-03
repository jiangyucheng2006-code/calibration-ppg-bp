"""Participant-cluster paired bootstrap for exploratory Phase-6B contrasts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .report_phase6_screening import (
    KS,
    SCOPES,
    _assert_common_predictions,
    _load_run,
    _parse_run_specs,
    _restore_and_validate_source,
    _scope,
)


def _parse_comparisons(values: Sequence[str]) -> list[tuple[str, str, str]]:
    comparisons: list[tuple[str, str, str]] = []
    labels: set[str] = set()
    for value in values:
        parts = [part.strip() for part in value.split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                "comparison must be LABEL|CANDIDATE_SETTING|REFERENCE_SETTING"
            )
        label, candidate, reference = parts
        if label in labels:
            raise ValueError(f"duplicate comparison label: {label}")
        labels.add(label)
        comparisons.append((label, candidate, reference))
    return comparisons


def _paired_bootstrap(
    differences: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired differences must be a finite one-dimensional array")
    if repetitions < 1000:
        raise ValueError("at least 1000 bootstrap repetitions are required")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(repetitions, dtype=np.float64)
    chunk = 1000
    for start in range(0, repetitions, chunk):
        stop = min(start + chunk, repetitions)
        indexes = rng.integers(0, len(values), size=(stop - start, len(values)))
        bootstrap_means[start:stop] = values[indexes].mean(axis=1)
    standard_deviation = float(values.std(ddof=1))
    return {
        "mean_delta_mmHg": float(values.mean()),
        "ci95_low_mmHg": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high_mmHg": float(np.quantile(bootstrap_means, 0.975)),
        "bootstrap_standard_error_mmHg": float(bootstrap_means.std(ddof=1)),
        "paired_difference_sd_mmHg": standard_deviation,
        "paired_standardized_effect": (
            float(values.mean() / standard_deviation)
            if standard_deviation > 0.0
            else float("nan")
        ),
        "bootstrap_fraction_improved": float(np.mean(bootstrap_means < 0.0)),
    }


def _participant_error_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["abs_error_sbp"] = (frame["pred_sbp"] - frame["target_sbp"]).abs()
    frame["abs_error_dbp"] = (frame["pred_dbp"] - frame["target_dbp"]).abs()
    participant = frame.groupby(
        ["setting", "source", "k", "subject_uid"], as_index=False
    ).agg(
        sbp_mae=("abs_error_sbp", "mean"),
        dbp_mae=("abs_error_dbp", "mean"),
    )
    participant["mean_mae"] = (
        participant["sbp_mae"] + participant["dbp_mae"]
    ) / 2.0
    return participant


def generate_bootstrap_report(
    *,
    runs: Mapping[str, Path],
    comparisons: Sequence[tuple[str, str, str]],
    store_root: Path,
    output_dir: Path,
    expected_seed: int,
    repetitions: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    needed = {setting for _, candidate, reference in comparisons for setting in (candidate, reference)}
    if not needed.issubset(runs):
        raise KeyError(f"comparison settings missing from runs: {sorted(needed - set(runs))}")
    frames: list[pd.DataFrame] = []
    for setting, run_dir in runs.items():
        predictions, _, _ = _load_run(
            setting, Path(run_dir), expected_seed=expected_seed
        )
        frames.append(predictions)
    combined = pd.concat(frames, ignore_index=True)
    first_setting = next(iter(runs))
    _assert_common_predictions(combined, reference_setting=first_setting)
    combined, _ = _restore_and_validate_source(combined, store_root=store_root)
    participant = _participant_error_table(combined)

    rows: list[dict[str, object]] = []
    for comparison_index, (label, candidate, reference) in enumerate(comparisons):
        for scope_index, scope in enumerate(SCOPES):
            scoped = _scope(participant, scope)
            for k_index, k in enumerate([*KS, "All K"]):
                selected = scoped if k == "All K" else scoped.loc[scoped["k"].eq(k)]
                if k == "All K":
                    selected = selected.groupby(
                        ["setting", "source", "subject_uid"], as_index=False
                    )["mean_mae"].mean()
                candidate_frame = selected.loc[
                    selected["setting"].eq(candidate), ["subject_uid", "mean_mae"]
                ].rename(columns={"mean_mae": "candidate_mae"})
                reference_frame = selected.loc[
                    selected["setting"].eq(reference), ["subject_uid", "mean_mae"]
                ].rename(columns={"mean_mae": "reference_mae"})
                paired = candidate_frame.merge(
                    reference_frame,
                    on="subject_uid",
                    how="inner",
                    validate="one_to_one",
                )
                if len(paired) != candidate_frame["subject_uid"].nunique() or len(paired) != reference_frame["subject_uid"].nunique():
                    raise AssertionError("candidate and reference participant sets differ")
                difference = (
                    paired["candidate_mae"] - paired["reference_mae"]
                ).to_numpy(dtype=float)
                metrics = _paired_bootstrap(
                    difference,
                    repetitions=repetitions,
                    seed=(
                        bootstrap_seed
                        + comparison_index * 100
                        + scope_index * 10
                        + k_index
                    ),
                )
                rows.append(
                    {
                        "Comparison": label,
                        "Candidate": candidate,
                        "Reference": reference,
                        "Scope": scope,
                        "K": k,
                        "N participants": int(len(paired)),
                        **metrics,
                    }
                )
    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "phase6b_paired_participant_bootstrap.csv"
    result.to_csv(csv_path, index=False, float_format="%.6f")
    report = {
        "status": "pass",
        "analysis": "exploratory participant-cluster paired bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "meta_validation",
        "locked_test_accessed": False,
        "training_seed": expected_seed,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_repetitions": repetitions,
        "metric": "candidate minus reference participant mean MAE; negative favors candidate",
        "multiplicity": "exploratory unadjusted intervals; candidate selection used the same development set",
        "comparisons": [
            {"label": label, "candidate": candidate, "reference": reference}
            for label, candidate, reference in comparisons
        ],
        "output": csv_path.name,
    }
    (output_dir / "phase6b_bootstrap_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--comparison", action="append", required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260819)
    args = parser.parse_args()
    report = generate_bootstrap_report(
        runs=_parse_run_specs(args.run),
        comparisons=_parse_comparisons(args.comparison),
        store_root=args.store_root,
        output_dir=args.output_dir,
        expected_seed=args.expected_seed,
        repetitions=args.repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        "PHASE6B_BOOTSTRAP_COMPLETE=yes "
        f"comparisons={len(report['comparisons'])} repetitions={report['bootstrap_repetitions']}"
    )


if __name__ == "__main__":
    main()
