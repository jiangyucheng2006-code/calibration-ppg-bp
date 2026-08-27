"""Paired-seed confirmation for the Round-13 calibrated PPG-BP winner.

The confirmation compares ``inception_time_wide`` with ``resnet_small`` on
the same fold-4 queries for every seed.  The Round-13 discovery seed is kept
explicitly separate from four prespecified confirmation seeds.  Meta-
validation and the locked meta-test are rejected at load time.

Participant-macro MAE is primary.  Event-pooled AAMI/BHS-style entries are
retrospective numerical diagnostics only; they are not device-validation or
regulatory-compliance claims.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .report_round8 import _diagnostic_rows
from .round10_end_to_end import (
    EARLY_FOLD,
    FIT_FOLDS,
    KEYS,
    SELECTION_FOLD,
    _markdown_table,
)
from .training import file_sha256, participant_macro_metrics, save_json


SCOPES = ("Overall", "MIMIC", "VitalDB")
EXPECTED_SOURCES = {"MIMIC", "VitalDB"}
REFERENCE_BACKBONE = "resnet_small"
CANDIDATE_BACKBONE = "inception_time_wide"
REFERENCE_SETTING = "resnet_small | QGH"
CANDIDATE_SETTING = "inception_time_wide | QGH"
PER_SEED_SETTINGS = (REFERENCE_SETTING, CANDIDATE_SETTING)
REFERENCE_ENSEMBLE_SETTING = "resnet_small | 5-seed equal-weight QGH ensemble"
CANDIDATE_ENSEMBLE_SETTING = (
    "inception_time_wide | 5-seed equal-weight QGH ensemble"
)
ENSEMBLE_SETTINGS = (REFERENCE_ENSEMBLE_SETTING, CANDIDATE_ENSEMBLE_SETTING)
TARGET_COLUMNS = ("target_sbp", "target_dbp")
PREDICTION_COLUMNS = ("pred_sbp", "pred_dbp")
SAFETY_FLAGS = (
    "meta_validation_accessed",
    "meta_validation_used_for_training",
    "meta_validation_used_for_early_stopping",
    "meta_validation_used_for_candidate_ranking",
    "meta_validation_predictions_generated",
    "locked_test_accessed",
    "query_bp_model_input",
    "future_query_model_input",
    "source_model_input",
)
AUDIT_EXPECTED = {
    "population_microbatch": 32,
    "population_accumulation": 4,
    "population_effective_batch": 128,
    "qgh_microbatch": 16,
    "qgh_accumulation": 4,
    "qgh_effective_batch": 64,
    "episodes_per_epoch": 99968,
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_seed_runs(values: list[str], *, label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use SEED=RUN_DIR, found {value!r}")
        seed_text, raw_path = value.split("=", 1)
        try:
            seed = int(seed_text)
        except ValueError as error:
            raise ValueError(f"invalid seed in {label}: {seed_text!r}") from error
        if seed in result or not raw_path.strip():
            raise ValueError(f"duplicate seed or empty path in {label}: {value!r}")
        result[seed] = Path(raw_path.strip())
    return result


def _parse_expected_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("expected seeds must contain at least two unique values")
    return seeds


def _validate_evaluation_run(
    root: Path,
    *,
    expected_seed: int,
    expected_backbone: str,
    expected_round: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    run_path = root / "run.json"
    query_path = root / "queries.parquet"
    payload = _read_json(run_path)
    identity = f"seed={expected_seed}, backbone={expected_backbone}"
    if payload.get("status") != "complete":
        raise AssertionError(f"{identity} is not complete")
    if payload.get("stage") != "backbone_evaluation":
        raise AssertionError(f"{identity} is not a backbone evaluation")
    if payload.get("round") != expected_round:
        raise AssertionError(f"{identity} has the wrong round")
    if payload.get("seed") != expected_seed:
        raise AssertionError(f"{identity} has the wrong seed")
    if payload.get("backbone") != expected_backbone:
        raise AssertionError(f"{identity} has the wrong backbone")
    if payload.get("split") != "meta_train_internal_only":
        raise AssertionError(f"{identity} is outside the meta-train boundary")
    if payload.get("fit_folds") != list(FIT_FOLDS):
        raise AssertionError(f"{identity} has the wrong fit folds")
    if payload.get("early_stopping_fold") != EARLY_FOLD:
        raise AssertionError(f"{identity} has the wrong early-stopping fold")
    if payload.get("selection_fold") != SELECTION_FOLD:
        raise AssertionError(f"{identity} has the wrong selection fold")
    if payload.get("support_policy") != "fixed_first" or payload.get("k") != 5:
        raise AssertionError(f"{identity} has the wrong K=5 support protocol")
    for flag in SAFETY_FLAGS:
        if payload.get(flag) is not False:
            raise AssertionError(f"{identity} has unsafe flag {flag}")

    audit = payload.get("training_audit")
    if not isinstance(audit, dict) or audit.get("status") != "pass":
        raise AssertionError(f"{identity} lacks a passing training audit")
    for key, expected in AUDIT_EXPECTED.items():
        if audit.get(key) != expected:
            raise AssertionError(
                f"{identity} has mismatched {key}: {audit.get(key)!r}"
            )
    for key in (
        "source_tree_sha256",
        "store_manifest_sha256",
        "crossfit_folds_sha256",
    ):
        if not audit.get(key):
            raise AssertionError(f"{identity} lacks training provenance {key}")
    if payload.get("evaluation_source_tree_sha256") != audit["source_tree_sha256"]:
        raise AssertionError(f"{identity} changed source before evaluation")

    frame = pd.read_parquet(query_path)
    required = {
        *KEYS,
        *TARGET_COLUMNS,
        *PREDICTION_COLUMNS,
        "source",
        "split",
        "fold",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{identity} queries miss {sorted(missing)}")
    frame = frame.loc[frame["fold"].eq(SELECTION_FOLD)].copy()
    if frame.empty or frame["split"].ne("meta_train").any():
        raise AssertionError(f"{identity} has invalid fold-4 rows")
    if set(frame["source"].astype(str)) != EXPECTED_SOURCES:
        raise AssertionError(f"{identity} lacks the two PulseDB source strata")
    if "k" in frame and set(frame["k"].astype(int)) != {5}:
        raise AssertionError(f"{identity} contains non-K=5 queries")
    if frame.duplicated(KEYS).any():
        raise AssertionError(f"{identity} contains duplicate queries")
    numeric = frame[[*TARGET_COLUMNS, *PREDICTION_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{identity} contains non-finite targets or predictions")
    frame = frame.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    return frame, payload


def _assert_common_queries(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    label: str,
) -> None:
    expected_keys = expected[[*KEYS, "source"]].reset_index(drop=True)
    observed_keys = observed[[*KEYS, "source"]].reset_index(drop=True)
    if not observed_keys.equals(expected_keys):
        raise AssertionError(f"{label} does not use the common query/source rows")
    if not np.allclose(
        observed[[*TARGET_COLUMNS]].to_numpy(dtype=float),
        expected[[*TARGET_COLUMNS]].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-4,
    ):
        raise AssertionError(f"{label} has different query targets")


def _scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "Overall":
        return frame
    return frame.loc[frame["source"].eq(scope)]


def _participant_rows(
    predictions_by_seed: dict[int, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed, settings in predictions_by_seed.items():
        for setting in PER_SEED_SETTINGS:
            for scope in SCOPES:
                frame = _scope(settings[setting], scope)
                metric = participant_macro_metrics(frame)
                rows.append(
                    {
                        "Seed": seed,
                        "Setting": setting,
                        "Scope": scope,
                        "N participants": int(metric["n_participants"]),
                        "N queries": int(metric["n_events"]),
                        "SBP participant-macro MAE": float(metric["sbp_mae"]),
                        "DBP participant-macro MAE": float(metric["dbp_mae"]),
                        "Mean participant-macro MAE": float(metric["mean_mae"]),
                    }
                )
    return pd.DataFrame(rows)


def _participant_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_names = (
        "SBP participant-macro MAE",
        "DBP participant-macro MAE",
        "Mean participant-macro MAE",
    )
    for (setting, scope), group in per_seed.groupby(
        ["Setting", "Scope"], sort=False
    ):
        participants = set(group["N participants"].astype(int))
        queries = set(group["N queries"].astype(int))
        if len(participants) != 1 or len(queries) != 1:
            raise AssertionError(f"{setting}/{scope} changed query coverage across seeds")
        row: dict[str, object] = {
            "Setting": setting,
            "Scope": scope,
            "N seeds": int(group["Seed"].nunique()),
            "N participants": int(next(iter(participants))),
            "N queries per seed": int(next(iter(queries))),
        }
        for metric in metric_names:
            row[f"{metric} mean"] = float(group[metric].mean())
            row[f"{metric} SD"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_gain_rows(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in sorted(per_seed["Seed"].unique()):
        seed_rows = per_seed.loc[per_seed["Seed"].eq(seed)]
        for scope in SCOPES:
            scoped = seed_rows.loc[seed_rows["Scope"].eq(scope)].set_index("Setting")
            reference = scoped.loc[REFERENCE_SETTING]
            candidate = scoped.loc[CANDIDATE_SETTING]
            rows.append(
                {
                    "Seed": int(seed),
                    "Scope": scope,
                    "Reference SBP MAE": float(
                        reference["SBP participant-macro MAE"]
                    ),
                    "Candidate SBP MAE": float(
                        candidate["SBP participant-macro MAE"]
                    ),
                    "SBP gain": float(
                        reference["SBP participant-macro MAE"]
                        - candidate["SBP participant-macro MAE"]
                    ),
                    "Reference DBP MAE": float(
                        reference["DBP participant-macro MAE"]
                    ),
                    "Candidate DBP MAE": float(
                        candidate["DBP participant-macro MAE"]
                    ),
                    "DBP gain": float(
                        reference["DBP participant-macro MAE"]
                        - candidate["DBP participant-macro MAE"]
                    ),
                    "Reference mean MAE": float(
                        reference["Mean participant-macro MAE"]
                    ),
                    "Candidate mean MAE": float(
                        candidate["Mean participant-macro MAE"]
                    ),
                    "Mean MAE gain": float(
                        reference["Mean participant-macro MAE"]
                        - candidate["Mean participant-macro MAE"]
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["Candidate improves mean MAE"] = result["Mean MAE gain"] > 0.0
    return result


def _paired_gain_summary(gains: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in SCOPES:
        group = gains.loc[gains["Scope"].eq(scope)]
        rows.append(
            {
                "Scope": scope,
                "N seeds": int(group["Seed"].nunique()),
                "SBP gain mean": float(group["SBP gain"].mean()),
                "SBP gain SD": float(group["SBP gain"].std(ddof=1)),
                "DBP gain mean": float(group["DBP gain"].mean()),
                "DBP gain SD": float(group["DBP gain"].std(ddof=1)),
                "Mean MAE gain mean": float(group["Mean MAE gain"].mean()),
                "Mean MAE gain SD": float(group["Mean MAE gain"].std(ddof=1)),
                "Positive mean-gain seeds": int(
                    group["Candidate improves mean MAE"].sum()
                ),
                "Direction consistency": float(
                    group["Candidate improves mean MAE"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _pooled_diagnostics(
    predictions_by_seed: dict[int, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for seed, settings in predictions_by_seed.items():
        combined = pd.concat(
            [frame.assign(Setting=setting) for setting, frame in settings.items()],
            ignore_index=True,
            sort=False,
        )
        diagnostics = _diagnostic_rows(combined, list(PER_SEED_SETTINGS))
        diagnostics.insert(0, "Seed", seed)
        records.append(diagnostics)
    return pd.concat(records, ignore_index=True)


def _cross_seed_ensemble(
    predictions_by_seed: dict[int, dict[str, pd.DataFrame]],
    *,
    source_setting: str,
) -> pd.DataFrame:
    ordered = [
        predictions_by_seed[seed][source_setting]
        for seed in sorted(predictions_by_seed)
    ]
    result = ordered[0].copy()
    for column in PREDICTION_COLUMNS:
        stacked = np.stack(
            [frame[column].to_numpy(dtype=float) for frame in ordered], axis=0
        )
        result[column] = stacked.mean(axis=0)
    return result


def _ensemble_participant_rows(
    ensembles: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for setting in ENSEMBLE_SETTINGS:
        for scope in SCOPES:
            frame = _scope(ensembles[setting], scope)
            metric = participant_macro_metrics(frame)
            rows.append(
                {
                    "Setting": setting,
                    "Scope": scope,
                    "N seeds ensembled": 5,
                    "N participants": int(metric["n_participants"]),
                    "N queries": int(metric["n_events"]),
                    "SBP participant-macro MAE": float(metric["sbp_mae"]),
                    "DBP participant-macro MAE": float(metric["dbp_mae"]),
                    "Mean participant-macro MAE": float(metric["mean_mae"]),
                }
            )
    return pd.DataFrame(rows)


def _ensemble_pooled_diagnostics(
    ensembles: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    combined = pd.concat(
        [frame.assign(Setting=setting) for setting, frame in ensembles.items()],
        ignore_index=True,
        sort=False,
    )
    return _diagnostic_rows(combined, list(ENSEMBLE_SETTINGS))


def build_confirmation_report(
    *,
    reference_runs: dict[int, Path],
    candidate_runs: dict[int, Path],
    expected_seeds: tuple[int, ...],
    discovery_seed: int,
    output: Path,
) -> dict[str, object]:
    """Build a leakage-safe paired-seed confirmation report."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if len(expected_seeds) != 5:
        raise ValueError("Round-14 confirmation requires exactly five seeds")
    expected = set(expected_seeds)
    if set(reference_runs) != expected or set(candidate_runs) != expected:
        raise AssertionError("reference/candidate seed coverage is incomplete")
    if discovery_seed not in expected:
        raise ValueError("discovery seed must be one of the expected seeds")

    predictions_by_seed: dict[int, dict[str, pd.DataFrame]] = {}
    evidence: list[dict[str, object]] = []
    canonical: pd.DataFrame | None = None
    store_hashes: set[str] = set()
    fold_hashes: set[str] = set()
    confirmation_source_hashes: set[str] = set()
    parameter_counts: dict[str, set[str]] = {
        REFERENCE_BACKBONE: set(),
        CANDIDATE_BACKBONE: set(),
    }

    for seed in expected_seeds:
        expected_round = 13 if seed == discovery_seed else 14
        reference, reference_record = _validate_evaluation_run(
            reference_runs[seed],
            expected_seed=seed,
            expected_backbone=REFERENCE_BACKBONE,
            expected_round=expected_round,
        )
        candidate, candidate_record = _validate_evaluation_run(
            candidate_runs[seed],
            expected_seed=seed,
            expected_backbone=CANDIDATE_BACKBONE,
            expected_round=expected_round,
        )
        _assert_common_queries(reference, candidate, label=f"candidate seed {seed}")
        if canonical is None:
            canonical = reference[[*KEYS, "source", *TARGET_COLUMNS]].copy()
        else:
            _assert_common_queries(canonical, reference, label=f"reference seed {seed}")

        reference_audit = reference_record["training_audit"]
        candidate_audit = candidate_record["training_audit"]
        assert isinstance(reference_audit, dict)
        assert isinstance(candidate_audit, dict)
        for key in (
            "source_tree_sha256",
            "store_manifest_sha256",
            "crossfit_folds_sha256",
        ):
            if reference_audit[key] != candidate_audit[key]:
                raise AssertionError(f"seed {seed} differs on paired provenance {key}")
        store_hashes.add(str(reference_audit["store_manifest_sha256"]))
        fold_hashes.add(str(reference_audit["crossfit_folds_sha256"]))
        if seed != discovery_seed:
            confirmation_source_hashes.add(str(reference_audit["source_tree_sha256"]))

        for backbone, record in (
            (REFERENCE_BACKBONE, reference_record),
            (CANDIDATE_BACKBONE, candidate_record),
        ):
            counts = record.get("qgh_parameter_counts")
            if not isinstance(counts, dict) or not counts.get("total"):
                raise AssertionError(f"{backbone} lacks QGH parameter counts")
            parameter_counts[backbone].add(json.dumps(counts, sort_keys=True))
        predictions_by_seed[seed] = {
            REFERENCE_SETTING: reference,
            CANDIDATE_SETTING: candidate,
        }
        for backbone, root in (
            (REFERENCE_BACKBONE, reference_runs[seed]),
            (CANDIDATE_BACKBONE, candidate_runs[seed]),
        ):
            evidence.append(
                {
                    "seed": seed,
                    "backbone": backbone,
                    "run_json_sha256": file_sha256(root / "run.json"),
                    "queries_sha256": file_sha256(root / "queries.parquet"),
                }
            )

    if len(store_hashes) != 1 or len(fold_hashes) != 1:
        raise AssertionError("seeds do not share the frozen store/fold provenance")
    if len(confirmation_source_hashes) != 1:
        raise AssertionError("new confirmation seeds do not share one source tree")
    for backbone, counts in parameter_counts.items():
        if len(counts) != 1:
            raise AssertionError(f"{backbone} parameter counts changed across seeds")

    per_seed = _participant_rows(predictions_by_seed)
    summary = _participant_summary(per_seed)
    gains = _paired_gain_rows(per_seed)
    gain_summary = _paired_gain_summary(gains)
    confirmation_gains = gains.loc[gains["Seed"].ne(discovery_seed)].copy()
    confirmation_gain_summary = _paired_gain_summary(confirmation_gains)
    diagnostics = _pooled_diagnostics(predictions_by_seed)
    ensembles = {
        REFERENCE_ENSEMBLE_SETTING: _cross_seed_ensemble(
            predictions_by_seed, source_setting=REFERENCE_SETTING
        ),
        CANDIDATE_ENSEMBLE_SETTING: _cross_seed_ensemble(
            predictions_by_seed, source_setting=CANDIDATE_SETTING
        ),
    }
    ensemble_participant = _ensemble_participant_rows(ensembles)
    ensemble_diagnostics = _ensemble_pooled_diagnostics(ensembles)

    indexed_gain = gain_summary.set_index("Scope")
    overall_gain = float(indexed_gain.loc["Overall", "Mean MAE gain mean"])
    mimic_gain = float(indexed_gain.loc["MIMIC", "Mean MAE gain mean"])
    vital_gain = float(indexed_gain.loc["VitalDB", "Mean MAE gain mean"])
    positive_overall = int(indexed_gain.loc["Overall", "Positive mean-gain seeds"])
    confirmation_indexed_gain = confirmation_gain_summary.set_index("Scope")
    confirmation_mean_gains = {
        scope: float(
            confirmation_indexed_gain.loc[scope, "Mean MAE gain mean"]
        )
        for scope in SCOPES
    }
    confirmation_positive_overall = int(
        confirmation_indexed_gain.loc["Overall", "Positive mean-gain seeds"]
    )
    gate = {
        "four_new_seeds_overall_mean_gain_at_least_0_15": (
            confirmation_mean_gains["Overall"] >= 0.15
        ),
        "four_new_seeds_mimic_mean_gain_positive": (
            confirmation_mean_gains["MIMIC"] > 0.0
        ),
        "four_new_seeds_vitaldb_mean_gain_positive": (
            confirmation_mean_gains["VitalDB"] > 0.0
        ),
        "four_new_seeds_overall_positive_count_at_least_3_of_4": (
            confirmation_positive_overall >= 3
        ),
    }
    passes = all(gate.values())

    output.mkdir(parents=True, exist_ok=False)
    per_seed.to_csv(output / "per_seed_participant_macro_internal.csv", index=False)
    summary.to_csv(output / "participant_macro_mean_sd_internal.csv", index=False)
    gains.to_csv(output / "paired_gains_internal.csv", index=False)
    gain_summary.to_csv(output / "paired_gain_summary_internal.csv", index=False)
    confirmation_gain_summary.to_csv(
        output / "four_new_seed_gain_summary_internal.csv", index=False
    )
    diagnostics.to_csv(output / "per_seed_pooled_diagnostics_internal.csv", index=False)
    ensemble_participant.to_csv(
        output / "cross_seed_ensemble_participant_macro_internal.csv", index=False
    )
    ensemble_diagnostics.to_csv(
        output / "cross_seed_ensemble_diagnostics_internal.csv", index=False
    )

    result = {
        "status": "complete",
        "round": 14,
        "stage": "paired_seed_confirmation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "meta_train_internal_fold4",
        "fit_folds": list(FIT_FOLDS),
        "early_stopping_fold": EARLY_FOLD,
        "selection_fold": SELECTION_FOLD,
        "seeds": list(expected_seeds),
        "discovery_seed": discovery_seed,
        "confirmation_seeds": [
            seed for seed in expected_seeds if seed != discovery_seed
        ],
        "reference_backbone": REFERENCE_BACKBONE,
        "candidate_backbone": CANDIDATE_BACKBONE,
        "ensemble_definition": (
            "separately within each backbone, arithmetic mean of its five "
            "seed-specific QGH predictions; CPU diagnostic only, no tuning, "
            "and excluded from the confirmation gate"
        ),
        "n_common_queries_per_seed": int(len(canonical)) if canonical is not None else 0,
        "store_manifest_sha256": next(iter(store_hashes)),
        "crossfit_folds_sha256": next(iter(fold_hashes)),
        "confirmation_source_tree_sha256": next(iter(confirmation_source_hashes)),
        "five_seed_background_mean_gain_vs_reference": {
            "Overall": overall_gain,
            "MIMIC": mimic_gain,
            "VitalDB": vital_gain,
        },
        "five_seed_background_overall_positive_seed_count": positive_overall,
        "four_new_seed_mean_gain_vs_reference": confirmation_mean_gains,
        "four_new_seed_overall_positive_seed_count": confirmation_positive_overall,
        "gate_basis": "four prespecified new confirmation seeds only",
        "discovery_seed_used_by_gate": False,
        "gate": gate,
        "passes_confirmation_gate": passes,
        "meta_validation_accessed": False,
        "locked_test_accessed": False,
        "input_evidence": evidence,
    }
    save_json(output / "confirmation.json", result)
    lines = [
        "# Round-14 paired-seed confirmation",
        "",
        "The Round-13 discovery seed and four prespecified new seeds are paired on the exact same fold-4 queries. Meta-validation and the locked meta-test are not accessed.",
        "",
        "Participant-macro MAE is primary. Pooled AAMI/BHS-style entries are retrospective numerical diagnostics only.",
        "",
        "## Per-seed participant-macro results",
        "",
        _markdown_table(per_seed),
        "",
        "## Across-seed mean and SD",
        "",
        _markdown_table(summary),
        "",
        "## Same-seed paired gains",
        "",
        "Positive gain means inception_time_wide QGH is better than resnet_small QGH.",
        "",
        _markdown_table(gains),
        "",
        "## Paired-gain summary",
        "",
        "This five-seed table includes the discovery seed and is background context only; it cannot make the confirmation pass.",
        "",
        _markdown_table(gain_summary),
        "",
        "## Four new confirmation seeds only",
        "",
        "The final confirmation gate is determined exclusively by these four prespecified new seeds.",
        "",
        _markdown_table(confirmation_gain_summary),
        "",
        "## Five-seed equal-weight ensemble participant-macro diagnostics",
        "",
        "Each backbone is ensembled across its own five seed-specific predictions. This CPU-only diagnostic is not used by the gate.",
        "",
        _markdown_table(ensemble_participant),
        "",
        "## Five-seed equal-weight ensemble pooled diagnostics",
        "",
        _markdown_table(ensemble_diagnostics),
        "",
        f"Confirmation gate passed: **{passes}**.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", action="append", required=True)
    parser.add_argument("--candidate-run", action="append", required=True)
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--discovery-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_confirmation_report(
        reference_runs=_parse_seed_runs(args.reference_run, label="--reference-run"),
        candidate_runs=_parse_seed_runs(args.candidate_run, label="--candidate-run"),
        expected_seeds=_parse_expected_seeds(args.expected_seeds),
        discovery_seed=args.discovery_seed,
        output=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
