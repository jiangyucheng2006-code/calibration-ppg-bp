"""Verify completed 200-person outputs and export aggregate-only evidence.

Run on the server with numpy/pandas/pyarrow. This does not import a training
module, fit a model, open original BP targets, or change existing run outputs.
Only the new --output directory receives a report and public aggregate copies.
Individual predictions, identity lists, adapters and memory banks stay private.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = (
    "personal_mean", "shared_with_anchor", "new_person_lora",
    "new_person_lora_memory", "shared_memory_only", "shared_memory_blend",
    "lora_memory_only", "lora_memory_uniform", "lora_memory_fixed_half",
)
SCOPES = ("Overall", "MIMIC", "VitalDB")
KEYS = ["subject_uid", "event_id", "source"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), minimum=float(values.min()), median=float(np.median(values)),
                maximum=float(values.max()), mean=float(values.mean()))


def close(actual, expected):
    assert np.isclose(actual, expected, atol=1e-10, rtol=0), (actual, expected)


def scope_frame(frame, scope):
    return frame if scope == "Overall" else frame.loc[frame.source.eq(scope)]


def verify_metrics(root, methods, expected_people, expected_sources):
    macro = pd.read_csv(root / "participant_macro.csv")
    diagnostic = pd.read_csv(root / "diagnostic_tables.csv")
    assert len(macro) == len(methods) * 3
    assert len(diagnostic) == len(methods) * 6
    people, frames = {}, {}
    for method in methods:
        frame = pd.read_parquet(root / f"{method}_scored_predictions.parquet")
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        assert len(frame) == expected_people * 40
        assert frame.subject_uid.nunique() == expected_people
        assert frame.groupby("source").subject_uid.nunique().to_dict() == expected_sources
        assert frame.groupby("subject_uid").size().eq(40).all()
        assert not frame.event_id.duplicated().any()
        for bp in ("sbp", "dbp"):
            assert np.isfinite(frame[[f"pred_{bp}", f"target_{bp}"]].to_numpy()).all()
            frame[f"{bp}_mae"] = (frame[f"pred_{bp}"] - frame[f"target_{bp}"]).abs()
        person = frame.groupby(["subject_uid", "source"])[["sbp_mae", "dbp_mae"]].mean().sort_index()
        person["mean_mae"] = person[["sbp_mae", "dbp_mae"]].mean(axis=1)
        for scope in SCOPES:
            subset = person if scope == "Overall" else person.xs(scope, level="source")
            stored = macro.loc[macro.Setting.eq(method) & macro.Scope.eq(scope)]
            assert len(stored) == 1
            stored = stored.iloc[0]
            assert len(subset) == stored.n_participants and len(subset) * 40 == stored.n_events
            for col in ("sbp_mae", "dbp_mae", "mean_mae"):
                close(subset[col].mean(), stored[col])
            scoped = scope_frame(frame, scope)
            for bp in ("SBP", "DBP"):
                target = scoped[f"target_{bp.lower()}"].to_numpy(dtype=float)
                error = scoped[f"pred_{bp.lower()}"].to_numpy(dtype=float) - target
                absolute = np.abs(error)
                row = diagnostic.loc[diagnostic.Setting.eq(method) & diagnostic.Scope.eq(scope) & diagnostic.BP.eq(bp)].iloc[0]
                computed = {"MAE": absolute.mean(), "R²": 1 - np.sum(error ** 2) / np.sum((target - target.mean()) ** 2),
                            "ME": error.mean(), "STD": error.std(ddof=1)}
                computed.update({f"≤{n} mmHg": np.mean(absolute <= n) * 100 for n in (5, 10, 15)})
                for col, value in computed.items():
                    close(value, row[col])
                assert row.AAMI == ("PASS*" if abs(computed["ME"]) <= 5 and computed["STD"] <= 8 else "FAIL*")
                percentages = [computed[f"≤{n} mmHg"] for n in (5, 10, 15)]
                grade = "D"
                for label, limits in (("A", (60, 85, 95)), ("B", (50, 75, 90)), ("C", (40, 65, 85))):
                    if all(a >= b for a, b in zip(percentages, limits)):
                        grade = label
                        break
                assert row.BHS == f"{'PASS' if grade in ('A', 'B') else 'FAIL'} (Grade {grade})*"
        people[method], frames[method] = person, frame
    for scope in SCOPES:
        pd.testing.assert_frame_equal(macro.loc[macro.Scope.eq(scope)].reset_index(drop=True),
                                      pd.read_csv(root / f"{scope}_participant_macro.csv"), check_exact=False, atol=1e-10)
        pd.testing.assert_frame_equal(diagnostic.loc[diagnostic.Scope.eq(scope)].drop(columns="Scope").reset_index(drop=True),
                                      pd.read_csv(root / f"{scope}_diagnostics.csv"), check_exact=False, atol=1e-10)
    return people, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    assert not args.output.exists(), "Use a new review output; do not overwrite evidence"
    plan = read_json(root / "prepare/plan.json")
    receipt = read_json(root / "evaluate/evaluation_receipt.json")
    inner = read_json(root / "population_inner/run.json")
    final = read_json(root / "population_final/run.json")
    frozen = read_json(root / "evaluate/frozen_predictions.json")
    assert plan["protocol_id"] == receipt["protocol_id"] == "post-enrollment-200-v1"
    assert tuple(plan["methods"]) == METHODS
    assert receipt["status"] == inner["status"] == final["status"] == "complete"
    assert receipt["plan_sha256"] == digest(root / "prepare/plan.json")
    assert receipt["population_checkpoint_sha256"] == final["checkpoint_sha256"]
    assert receipt["all_methods_frozen_before_targets"] and receipt["method_count"] == 9
    assert receipt["test_based_selection"] is False and frozen["test_targets_accessed"] is False
    assert receipt["prediction_files"] == frozen["prediction_files"]
    for report in (inner, final):
        assert report["old_checkpoint_used"] is False and report["initialization"] == "from_scratch"
        assert report["enrollment_labels_accessed"] is False and report["test_targets_accessed"] is False
    assert final["fit_subjects"] == 2305 and final["fit_rows"] == 829800
    assert final["fit_subject_ids_sha256"] == plan["audit"]["population_train"]["subject_ids_sha256"]
    assert final["fit_ids_sha256"] == plan["audit"]["population_train"]["segment_ids_sha256"]
    population_ids = set(read_json(root / "population_final/subject_index.json"))
    enrollment_ids = set(plan["selected_subjects"])
    assert len(population_ids) == 2305 and len(enrollment_ids) == 200
    assert not population_ids & enrollment_ids
    assert not population_ids & set(plan["quarantined_subjects"])
    assert plan["selection_uses_labels_or_errors"] is False
    assert plan["selection_unchanged_by_quarantine"] is True
    # Read membership only, never the registration BP or original test target store.
    membership = {}
    for filename, expected_sha in plan["files"].items():
        path = root / "prepare" / filename
        assert digest(path) == expected_sha
        membership[filename] = pd.read_parquet(path, columns=["subject_uid", "segment_uid", "source"])
    reg = membership["enrollment_train.parquet"]
    queries = membership["enrollment_test_inputs.parquet"]
    assert set(reg.subject_uid) == set(queries.subject_uid) == enrollment_ids
    assert reg.groupby("subject_uid").size().eq(360).all()
    assert queries.groupby("subject_uid").size().eq(40).all()
    assert not set(reg.segment_uid) & set(queries.segment_uid)

    profiles, shards = [], {method: [] for method in METHODS}
    profile_files_checked = 0
    for i in range(2):
        directory = root / f"personal_{i}"
        run = read_json(directory / "run.json")
        assert digest(directory / "run.json") == receipt["personal_runs"][i]["run_sha256"]
        assert run["status"] == "complete" and run["test_targets_accessed"] is False
        assert run["shared_frozen"] and run["other_person_adapter_copied"] is False
        assert run["population_checkpoint_sha256"] == final["checkpoint_sha256"]
        assert run["subjects"] == plan["selected_subjects"][i::2]
        assert set(run["profiles"]) == set(run["subjects"])
        for profile in run["profiles"].values():
            assert profile["registration_rows"] == 360 and profile["test_rows"] == 40
            assert profile["shared_frozen_verified"] and profile["reload_equivalence"]
            assert profile["all_ablation_reload_equivalence"]
            for filename, key in (("profile.pt", "profile_sha256"), ("memory_bank.npz", "memory_bank_sha256"),
                                  ("registration_metadata.parquet", "registration_metadata_sha256"),
                                  ("ablation_state.json", "ablation_state_sha256")):
                assert digest(directory / profile["directory"] / filename) == profile[key]
                profile_files_checked += 1
            profiles.append(profile)
        for method in METHODS:
            file = directory / f"{method}_predictions.parquet"
            assert digest(file) == run["prediction_files"][file.name]
            shards[method].append(pd.read_parquet(file))
    assert len(profiles) == 200

    people, frames = verify_metrics(root / "evaluate", METHODS, 200, {"MIMIC": 100, "VitalDB": 100})
    for method in METHODS:
        file = root / "evaluate" / f"{method}_frozen_predictions.parquet"
        assert digest(file) == receipt["prediction_files"][file.name]
        prediction = pd.read_parquet(file).rename(columns={"segment_uid": "event_id"}).sort_values(KEYS).reset_index(drop=True)
        shard = pd.concat(shards[method], ignore_index=True).rename(columns={"segment_uid": "event_id"}).sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(prediction, shard, check_exact=True)
        pd.testing.assert_frame_equal(prediction, frames[method][prediction.columns], check_exact=True)
        assert set(frames[method].subject_uid) == enrollment_ids
        assert set(frames[method].event_id) == set(queries.segment_uid)
        assert frames[method][KEYS + ["target_sbp", "target_dbp"]].equals(frames[METHODS[0]][KEYS + ["target_sbp", "target_dbp"]])
    verify_metrics(root / "population_benchmark", ("population_lora",), 2305, {"MIMIC": 1112, "VitalDB": 1193})

    # Independent implementation of the already-specified paired source bootstrap.
    intervals = pd.read_csv(root / "evaluate/paired_participant_intervals.csv")
    assert len(intervals) == 72 and intervals.primary_contrast.sum() == 1
    contract = read_json(root / "evaluate/uncertainty_contract.json")
    assert contract["bootstrap_replicates"] == 2000 and contract["seed"] == 20260909
    assert contract["unit"] == "participant" and contract["source_stratified"] and contract["paired"]
    for row in intervals.itertuples(index=False):
        delta = people[row.Reference] - people[row.Setting]
        if row.Scope != "Overall":
            delta = delta.loc[delta.index.get_level_values("source") == row.Scope]
        values = delta[{"Mean": "mean_mae", "SBP": "sbp_mae", "DBP": "dbp_mae"}[row.BP]].to_numpy()
        sources = delta.index.get_level_values("source").to_numpy()
        rng = np.random.default_rng(contract["seed"])
        bootstrap_sum = np.zeros(contract["bootstrap_replicates"])
        for source in sorted(set(sources)):
            group = values[sources == source]
            bootstrap_sum += rng.choice(group, size=(len(bootstrap_sum), len(group)), replace=True).sum(axis=1)
        low, high = np.quantile(bootstrap_sum / len(values), [.025, .975])
        close(values.mean(), row.gain_mmHg)
        close(low, row.ci95_low_mmHg)
        close(high, row.ci95_high_mmHg)
        assert row.participants == len(values) and row.improved_participants == (values > 0).sum()

    comparisons = []
    for old, new in (("shared_with_anchor", "new_person_lora"), ("new_person_lora", "new_person_lora_memory"),
                     ("shared_memory_only", "new_person_lora_memory"), ("lora_memory_only", "new_person_lora_memory")):
        delta = people[old] - people[new]
        for scope in SCOPES:
            part = delta if scope == "Overall" else delta.xs(scope, level="source")
            for metric in ("sbp_mae", "dbp_mae", "mean_mae"):
                values = part[metric].to_numpy()
                comparisons.append(dict(reference=old, candidate=new, scope=scope, metric=metric,
                    improvement_mmHg=describe(values), improved_people=int((values > 1e-10).sum()),
                    worsened_people=int((values < -1e-10).sum()), tied_people=int((np.abs(values) <= 1e-10).sum())))

    archive_files, exports = {}, []
    for stage, public_dir in (("evaluate", "enrollment"), ("population_benchmark", "population")):
        paths = sorted((root / stage).glob("*.csv"))
        paths += [root / stage / name for name in ("evaluation_receipt.json", "frozen_predictions.json", "RESULT_TABLES.md")]
        if stage == "evaluate":
            paths.append(root / stage / "uncertainty_contract.json")
        for path in paths:
            relative = path.relative_to(root)
            assert digest(path) == digest(args.archive / relative), f"archive mismatch: {relative}"
            archive_files[str(relative)] = digest(path)
            exports.append((path, Path(public_dir) / path.name))
    public_audit = {key: value for key, value in plan["audit"].items() if key != "quarantined_subjects"}
    public_audit["quarantined_subject_count"] = len(plan["quarantined_subjects"])
    report = dict(status="pass", verified_utc=datetime.now(timezone.utc).isoformat(),
        protocol_id=receipt["protocol_id"], code_source_tree_sha256=final["source_tree_sha256"],
        inspector_sha256=digest(Path(__file__)), source_audit=public_audit,
        source_content_audit="Existing frozen preparation audit checked; raw waveforms not rescanned during this review",
        previous_30_overlap_count=plan["previous_30_overlap_count"],
        population_inner_epochs=inner["epochs_completed"], population_selected_epoch=inner["best_epoch"],
        population_final_epochs=final["epochs_completed"], population_fit_subjects=final["fit_subjects"],
        subjects=200, windows=8000, source_subjects={"MIMIC": 100, "VitalDB": 100},
        selected_personal_epochs=describe([p["selected_epoch"] for p in profiles]),
        zero_epoch_profiles=sum(p["selected_epoch"] == 0 for p in profiles),
        refit_optimizer_steps=describe([p["refit_optimizer_steps"] for p in profiles]),
        adapter_parameter_counts=sorted(set(p["adapter_parameters"] for p in profiles)),
        profile_artifact_hashes_verified=profile_files_checked,
        shared_state_unchanged_profiles=sum(p["shared_frozen_verified"] for p in profiles),
        reload_equivalent_profiles=sum(p["reload_equivalence"] for p in profiles),
        ablation_reload_equivalent_profiles=sum(p["all_ablation_reload_equivalence"] for p in profiles),
        full_method_valid_memory_queries=sum(p["valid_memory_queries"] for p in profiles),
        registration_rows_per_person=360, test_rows_per_person=40,
        all_query_keys_and_targets_matched=True, enrollment_primary_metric_rows_recomputed=27,
        enrollment_diagnostic_rows_recomputed=54, population_primary_metric_rows_recomputed=3,
        population_diagnostic_rows_recomputed=6, saved_paired_interval_rows_recomputed=72,
        analysis="Prespecified paired source-stratified bootstrap and descriptive contrasts on already-scored predictions; no original target store read, new fit, exclusion or independent seed replication",
        paired_comparisons=comparisons, archived_aggregate_sha256=archive_files)
    # Export only whitelisted aggregate files after every verification succeeds.
    args.output.mkdir(parents=True, exist_ok=False)
    for original, relative in exports:
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    (args.output / "verification_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(args.output / "verified_aggregates.zip", "x", compression=zipfile.ZIP_DEFLATED) as package:
        for _, relative in exports:
            package.write(args.output / relative, str(relative))
        package.write(args.output / "verification_summary.json", "verification_summary.json")
    print(json.dumps({key: value for key, value in report.items() if key not in ("paired_comparisons", "archived_aggregate_sha256")}, ensure_ascii=False, indent=2))
    print("VERIFIED_AGGREGATE_PACKAGE=" + str(args.output / "verified_aggregates.zip"))


if __name__ == "__main__":
    main()
