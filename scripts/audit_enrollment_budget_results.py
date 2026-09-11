"""Read-only verification of completed enrollment-budget result artifacts.

Requires numpy and pandas with a Parquet engine. Prints aggregate evidence to
stdout; never modifies predictions, accesses raw waveforms, or fits a model.
Run where the private completed result files are available. Do not publish
individual prediction or personal-profile files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def audit(root, archive=None):
    score = root / "test_score"
    receipt = json.loads((score / "evaluation_receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["cohort"] == "test"
    assert receipt["all_sixteen_predictions_frozen_before_targets"] is True
    assert receipt["training_feedback"] is False
    assert receipt["test_based_selection"] is False
    budgets = receipt["percentages"]
    methods = receipt["methods"]
    assert budgets == list(range(20, 91, 10)) and len(methods) == 2
    macro = pd.read_csv(score / "all_budgets_participant_macro.csv")
    diagnostic = pd.read_csv(score / "all_budgets_diagnostics.csv")
    assert len(macro) == 48 and len(diagnostic) == 96
    assert not macro.duplicated(["budget_percent", "Setting", "Scope"]).any()
    assert not diagnostic.duplicated(["budget_percent", "Setting", "Scope", "BP"]).any()
    people = {}
    counts = {}
    canonical_keys = canonical_targets = None
    max_difference = 0.0

    def check(observed, expected):
        nonlocal max_difference
        delta = abs(float(observed) - float(expected))
        max_difference = max(max_difference, delta)
        assert np.isclose(observed, expected, rtol=1e-10, atol=1e-10), (observed, expected)

    for budget in budgets:
        for method in methods:
            frozen_path = score / f"p{budget}_{method}_frozen_predictions.parquet"
            assert digest(frozen_path) == receipt["prediction_files"][frozen_path.name]
            frozen = pd.read_parquet(frozen_path).rename(columns={"segment_uid": "event_id"})
            scored = pd.read_parquet(score / f"p{budget}" / f"{method}_scored_predictions.parquet")
            keys = ["subject_uid", "event_id", "source"]
            assert not scored.duplicated(keys).any()
            assert len(scored) == receipt["query_windows"]
            scored = scored.sort_values(keys).reset_index(drop=True)
            frozen = frozen.sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(scored[keys], frozen[keys])
            for bp in ("sbp", "dbp"):
                assert np.array_equal(scored[f"pred_{bp}"], frozen[f"pred_{bp}"])
            target_columns = ["target_sbp", "target_dbp"]
            if canonical_keys is None:
                canonical_keys = scored[keys].copy()
                canonical_targets = scored[target_columns].copy()
            else:
                pd.testing.assert_frame_equal(scored[keys], canonical_keys)
                pd.testing.assert_frame_equal(scored[target_columns], canonical_targets)
            assert np.isfinite(scored[target_columns + ["pred_sbp", "pred_dbp"]].to_numpy()).all()
            scored["SBP"] = abs(scored.pred_sbp - scored.target_sbp)
            scored["DBP"] = abs(scored.pred_dbp - scored.target_dbp)
            per_person = scored.groupby(["subject_uid", "source"])[["SBP", "DBP"]].mean().sort_index()
            per_person["Mean"] = per_person[["SBP", "DBP"]].mean(axis=1)
            people[budget, method] = per_person
            for scope in ("Overall", "MIMIC", "VitalDB"):
                frame = scored if scope == "Overall" else scored.loc[scored.source.eq(scope)]
                persons = per_person if scope == "Overall" else per_person.loc[per_person.index.get_level_values("source") == scope]
                counts[scope] = {"participants": len(persons), "query_windows": len(frame)}
                row = macro.loc[(macro.budget_percent == budget) & (macro.Setting == method) & (macro.Scope == scope)].iloc[0]
                assert row.n_participants == len(persons) and row.n_events == len(frame)
                check(persons.SBP.mean(), row.sbp_mae)
                check(persons.DBP.mean(), row.dbp_mae)
                check(persons.Mean.mean(), row.mean_mae)
                for bp in ("SBP", "DBP"):
                    lower = bp.lower()
                    error = (frame[f"pred_{lower}"] - frame[f"target_{lower}"]).to_numpy(float)
                    target = frame[f"target_{lower}"].to_numpy(float)
                    values = {
                        "MAE": np.abs(error).mean(), "ME": error.mean(),
                        "STD": error.std(ddof=1),
                        "R²": 1 - np.sum(error ** 2) / np.sum((target - target.mean()) ** 2),
                        **{f"≤{limit} mmHg": (np.abs(error) <= limit).mean() * 100 for limit in (5, 10, 15)},
                    }
                    dr = diagnostic.loc[(diagnostic.budget_percent == budget) & (diagnostic.Setting == method) & (diagnostic.Scope == scope) & (diagnostic.BP == bp)].iloc[0]
                    for name, value in values.items():
                        check(value, dr[name])
                    aami = "PASS*" if abs(values["ME"]) <= 5 and values["STD"] <= 8 else "FAIL*"
                    grade = "D"
                    for letter, limits in [("A", (60, 85, 95)), ("B", (50, 75, 90)), ("C", (40, 65, 85))]:
                        if all(values[f"≤{threshold} mmHg"] >= level for threshold, level in zip((5, 10, 15), limits)):
                            grade = letter
                            break
                    assert dr.AAMI == aami
                    assert dr.BHS == f"{'PASS' if grade in ('A', 'B') else 'FAIL'} (Grade {grade})*"

    intervals = pd.read_csv(score / "budget_paired_intervals.csv")
    assert len(intervals) == 126
    # Reproduce the prespecified seven-contrast simultaneous band independently.
    baseline = people[90, "new_person_lora_memory"]
    difference = np.stack([people[p, "new_person_lora_memory"].Mean.to_numpy() - baseline.Mean.to_numpy() for p in budgets[:-1]], axis=1)
    observed = difference.mean(axis=0)
    source = baseline.index.get_level_values("source").to_numpy()
    groups = [np.flatnonzero(source == s) for s in sorted(set(source))]
    rng = np.random.default_rng(20260911)
    draws = np.stack([difference[np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])].mean(axis=0) for _ in range(2000)])
    low, high = np.quantile(draws, [.025, .975], axis=0)
    radius = np.quantile(np.max(abs(draws - observed), axis=1), .95)
    for i, budget in enumerate(budgets[:-1]):
        ir = intervals.loc[intervals.primary_family & intervals.budget_percent.eq(budget)].iloc[0]
        for field, value in {"MAE_increase_vs_90": observed[i], "pointwise_ci95_low": low[i], "pointwise_ci95_high": high[i], "simultaneous_ci95_low": observed[i] - radius, "simultaneous_ci95_high": observed[i] + radius}.items():
            check(value, ir[field])

    participant_changes = []
    for budget in budgets:
        gain = people[budget, "new_person_lora"].Mean - people[budget, "new_person_lora_memory"].Mean
        paired = pd.read_csv(score / f"p{budget}" / "paired_participant_intervals.csv")
        saved = paired.loc[paired.Scope.eq("Overall") & paired.BP.eq("Mean")].iloc[0]
        check(gain.mean(), saved.gain_mmHg)
        assert int((gain > 0).sum()) == saved.improved_participants
        participant_changes.append({"budget_percent": budget, "memory_better": int((gain > 0).sum()), "tied": int((gain == 0).sum()), "memory_worse": int((gain < 0).sum())})
    growth = people[20, "new_person_lora_memory"].Mean - baseline.Mean
    # Export only checksums and aggregate counts, not participant identifiers.
    public_paths = sorted([*score.glob("*.csv"), *score.glob("*.json"), *score.glob("p*/paired_participant_intervals.csv"), *score.glob("p*/uncertainty_contract.json"), root / "prepare" / "budget_summary.csv"])
    hashes = {p.relative_to(root).as_posix(): digest(p) for p in public_paths}
    archive_ok = None
    if archive:
        archive_ok = all((archive / relative).is_file() and digest(archive / relative) == value for relative, value in hashes.items())
        assert archive_ok, "work and archive aggregate files differ or are missing"
    return {
        "status": "pass", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": receipt["protocol_id"], "run_id": root.name,
        "scope": "read-only recomputation from saved predictions, not a training-seed replication",
        "prediction_sets_checked": 16, "participant_macro_rows_checked": 48,
        "pooled_diagnostic_rows_checked": 96, "prespecified_primary_budget_intervals_reproduced": 7,
        "identical_query_keys_and_targets_all_arms": True,
        "frozen_prediction_hashes_and_values_match": True,
        "all_participants_and_queries_retained": True,
        "maximum_absolute_numeric_difference": max_difference,
        "counts": counts, "participant_memory_changes": participant_changes,
        "twenty_to_ninety_memory_mean_mae_changes": {"better": int((growth > 0).sum()), "tied": int((growth == 0).sum()), "worse": int((growth < 0).sum())},
        "work_nas_aggregate_identity": archive_ok, "artifact_sha256": hashes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root, args.archive_root), indent=2))
