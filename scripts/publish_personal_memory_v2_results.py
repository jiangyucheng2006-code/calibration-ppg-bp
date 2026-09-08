"""Validate and publish allowlisted personal-memory-v2 aggregate reports.

Requires NumPy/pandas, not Torch. Reads exactly 33 JSON/CSV artifacts from one
completed batch; never reads participant predictions, waveforms or checkpoints.
Validation cross-checks server aggregates and recorded contracts, not a fresh
prediction-level recomputation or independent audit of the recorded run claims.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pulsedb_fewshot.personal_memory_v2_report import (  # noqa: E402
    CANDIDATES, DIAGNOSTICS, MODES, comparisons, eligibility,
)
from pulsedb_fewshot.personal_memory_report import EXPECTED_COUNTS  # noqa: E402

SCREEN = "personal-memory-v2"
SEED = 20260907
MATCHING_SEED = 20260908  # Frozen diagnostic sampling seed, distinct from fit seed.
SETTINGS = tuple(sorted(set(CANDIDATES) | set(DIAGNOSTICS) | {"lora_continued_control"}))
MACRO = ["candidate", "view", "split_mode", "n_participants", "n_events", "sbp_mae", "dbp_mae", "mean_mae"]
METRICS = ["sbp_mae", "dbp_mae", "mean_mae"]
POOLED = ["Scope", "Setting", "BP", "MAE", "R²", "ME", "STD", "≤5 mmHg", "≤10 mmHg", "≤15 mmHg", "AAMI", "BHS"]
COMP = MACRO + ["reference_mean_mae", "gain_vs_reference_mmhg", "fixed_blend_mean_mae", "gain_vs_fixed_blend_mmhg", "v1_trained_blend_mean_mae", "gain_vs_v1_trained_blend_mmhg", "reference_candidate"]
DIAG_METRICS = ["Setting", "Scope", "n_participants", "n_events", *METRICS, "sbp_rmse", "dbp_rmse", "sbp_bias", "dbp_bias"]
SUBGROUP = ["Setting", "Scope", "scoring_only_group", "n_participants", "n_events", *METRICS,
            "delta_mean_mae_vs_frozen_lora", "delta_sbp_mae_vs_frozen_lora", "delta_dbp_mae_vs_frozen_lora"]
CONTRACT = {"protocol_id": "development-calbased-analogue-v1", "source_parent_split": "meta_train",
            "read_roles": ["train", "internal_validation"], "selection_role": "internal_validation",
            "official_pulsedb_calbased_reproduction": False}
FORBIDDEN = re.compile(r"/home/|/srv/|[A-Za-z]:[/\\]|\b(?:subject_uid|subject_id|event_id|target_sbp|target_dbp|pred_sbp|pred_dbp|password|private_key|access_token)\b|private_[\w.-]+|\.parquet\b|\.pt\b", re.I)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same(actual, expected, tolerance=2e-5):
    try:
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
    except AssertionError as error:
        raise ValueError("aggregate values disagree: " + str(error)) from error


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def valid_hash(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "missing/invalid SHA-256")


def qualified_standards(row):
    """Historical retrospective numerical labels, not device compliance."""
    aami = "PASS*" if abs(float(row["ME"])) <= 5 and float(row["STD"]) <= 8 else "FAIL*"
    percent = np.array([row[c] for c in POOLED[7:10]], dtype=float)
    grade = next((g for g, thresholds in (("A", [60, 85, 95]), ("B", [50, 75, 90]),
                                         ("C", [40, 65, 85])) if np.all(percent >= thresholds)), "D")
    return aami, f"{'PASS' if grade in {'A', 'B'} else 'FAIL'} (Grade {grade})*"


def frame_equal(actual, expected, keys):
    require(list(actual.columns) == list(expected.columns), "unexpected/private table columns")
    require(len(actual) == len(expected) and not actual.duplicated(keys).any(), "row coverage/uniqueness mismatch")
    left = actual.sort_values(keys).reset_index(drop=True)
    right = expected.sort_values(keys).reset_index(drop=True)
    for column in right:
        if pd.api.types.is_numeric_dtype(right[column]) and not pd.api.types.is_bool_dtype(right[column]):
            same(left[column], right[column], 1e-7)
        else:
            require(left[column].tolist() == right[column].tolist(), "table field mismatch: " + column)


def validate_basic(record, mode=None, *, seed=False, contract=False):
    require(record.get("status") == "complete" and record.get("screen_id") == SCREEN, "incomplete/wrong screen")
    require(record.get("heldout_test_accessed") is False, "held-out access must explicitly be false")
    if mode is not None:
        require(record.get("split_mode") == mode, "mode mismatch")
    if seed:
        require(record.get("seed") == SEED, "seed mismatch")
    if contract:
        require(all(record.get(k) == v for k, v in CONTRACT.items()), "data-role contract mismatch")


def validate_metric_map(metrics):
    require(set(metrics) == set(EXPECTED_COUNTS), "metric source coverage mismatch")
    for scope, counts in EXPECTED_COUNTS.items():
        row = metrics[scope]
        require((row.get("n_participants"), row.get("n_events")) == counts, "full cohort changed")
        values = np.array([row[k] for k in METRICS], dtype=float)
        require(np.isfinite(values).all() and (values >= 0).all(), "nonfinite/negative MAE")
        same(row["mean_mae"], (row["sbp_mae"] + row["dbp_mae"]) / 2)
    # Consistency only: public Overall values always use the original Overall row.
    for key in METRICS:
        expected = sum(metrics[s][key] * EXPECTED_COUNTS[s][0] for s in ("MIMIC", "VitalDB")) / 2051
        same(metrics["Overall"][key], expected)


def validate_macro(frame, mode):
    require(list(frame.columns) == MACRO, "unexpected/private macro columns")
    require(len(frame) == 24 and not frame.duplicated(["candidate", "view"]).any(), "macro matrix incomplete")
    require(set(frame.candidate) == set(SETTINGS) and frame.split_mode.eq(mode).all(), "macro settings/mode mismatch")
    for _, rows in frame.groupby("candidate"):
        validate_metric_map(rows.set_index("view").to_dict("index"))


def validate_pooled(frame, macro, *, setting="candidate", scope="view"):
    require(list(frame.columns) == POOLED, "unexpected/private pooled columns")
    require(len(frame) == len(macro) * 2 and not frame.duplicated(POOLED[:3]).any(), "pooled matrix incomplete")
    expected_keys = {(r[setting], r[scope], bp) for _, r in macro.iterrows() for bp in ("SBP", "DBP")}
    require(set(zip(frame.Setting, frame.Scope, frame.BP)) == expected_keys, "pooled keys mismatch")
    lookup = macro.set_index([setting, scope])
    for _, row in frame.iterrows():
        source = lookup.loc[(row.Setting, row.Scope)]
        same(row.MAE, source[f"{row.BP.lower()}_mae"])
        if f"{row.BP.lower()}_bias" in source:
            same(row.ME, source[f"{row.BP.lower()}_bias"])
        vals = row[POOLED[3:10]].to_numpy(float)
        require(np.isfinite(vals).all(), "nonfinite pooled metric")
        require(row.MAE >= 0 and row.STD >= 0 and row["R²"] <= 1 + 1e-10, "invalid pooled range")
        require(abs(row.ME) <= row.MAE + 2e-5, "signed error exceeds absolute error")
        percent = row[POOLED[7:10]].to_numpy(float)
        require(((percent >= 0) & (percent <= 100)).all() and (np.diff(percent) >= 0).all(), "invalid threshold percentages")
        require((row.AAMI, row.BHS) == qualified_standards(row), "unqualified/incorrect numerical-screen label")


class Inputs:
    """Discover one batch, allowlist its artifacts, and retain hash-only provenance."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        matches = list(self.root.glob("*_both_final/selection.json"))
        require(len(matches) == 1, "exactly one completed batch required")
        self.prefix = matches[0].parent.name.removesuffix("_both_final")
        self.files = {}

    def path(self, role, filename):
        path = self.root / f"{self.prefix}_{role}" / filename
        require(path.is_file() and not path.is_symlink(), "missing or symlinked input: " + role + "/" + filename)
        require(path.resolve().is_relative_to(self.root), "input escaped input root")
        self.files[f"{role}/{filename}"] = {"sha256": digest(path), "bytes": path.stat().st_size}
        return path

    def read(self, role, filename):
        path = self.path(role, filename)
        if filename.endswith(".json"):
            return json.loads(path.read_text(encoding="utf-8"))
        return pd.read_csv(path)

    def check(self, role, filename, expected):
        valid_hash(expected)
        require(digest(self.path(role, filename)) == expected, "input hash mismatch: " + role + "/" + filename)


def validate_training(run, history, metrics, mode, route, main):
    validate_basic(run, mode, seed=True, contract=True)
    candidate = "E1_matched_relation" if route == "e1" else "E2_bp_state_metric"
    require(run.get("candidate") == candidate, "run candidate mismatch")
    fixed = {"personal_label_budget_windows": 320, "train_participants": 2051,
             "train_windows_available": 656320, "internal_validation_participants": 2051,
             "internal_validation_windows": 82040, "validation_query_coverage": 1.0,
             "persistent_LoRA_preserved": True, "retrieved_references": 5}
    require(all(run.get(k) == v for k, v in fixed.items()), "training budget/coverage contract changed")
    args = run["arguments"]
    require(all(args.get(k) == v for k, v in {"seed": SEED, "patience": 8, "epochs": 0,
            "batch_size": 256, "examples_per_epoch": 200000,
            "smoke": False, "smoke_check": False, "synthetic_smoke": False}.items()), "training execution contract changed")
    if route == "e1":
        require(run.get("retrieval_training_policy") == "physical-gap-v2" and
                run.get("fixed_v1_alpha_preserved") is True and
                run.get("fixed_v1_validation_references_preserved") is True, "E1 intervention changed")
    else:
        require(run.get("query_BP_used_for_retrieval") is False and run.get("gate_recalibrated") is False and
                run.get("relation_correction") == "zero" and
                run.get("fixed_alpha") == "unchanged_original256d_cache_support_weight", "E2 intervention changed")
    require(run.get("stop_reason") == "early_stopping", "fit did not complete natural early stopping")
    require(len(history) == run["epochs_completed"] and len(history) >= 8, "history length mismatch")
    require([h["epoch"] for h in history] == list(range(1, len(history) + 1)), "nonconsecutive history")
    require(history[-1]["stale_epochs"] == 8, "patience-8 stop not reached")
    require([h["optimizer_steps_total"] for h in history] == [782 * i for i in range(1, len(history) + 1)], "step history mismatch")
    require(all(h["examples"] == 200000 for h in history), "exposure budget changed")
    require(run["optimizer_steps"] == len(history) * 782 and run["examples_processed"] == len(history) * 200000, "training cost counts mismatch")
    require(np.isfinite(run["runtime_seconds"]) and run["runtime_seconds"] > 0, "invalid runtime")
    initial = run["initial_metrics" if route == "e1" else "initial_random_projection_metrics"]
    for metric_map in [initial, metrics, run["metrics"], *(h["internal_validation"] for h in history)]:
        validate_metric_map(metric_map)
    choices = [(0, initial), *((h["epoch"], h["internal_validation"]) for h in history)]
    best_epoch, best = min(choices, key=lambda item: (item[1]["Overall"]["mean_mae"], item[0]))
    require(run["best_epoch"] == best_epoch, "selected epoch is not best Overall checkpoint")
    trained_epoch, trained = min(choices[1:], key=lambda item: (item[1]["Overall"]["mean_mae"], item[0]))
    if route == "e1":
        require(run["best_trained_epoch"] == trained_epoch, "best-trained epoch mismatch")
        valid_hash(run["best_trained_checkpoint_sha256"])
        for s in EXPECTED_COUNTS:
            same([trained[s][m] for m in METRICS], [run["best_trained_metrics"][s][m] for m in METRICS])
    rows = []
    for s in EXPECTED_COUNTS:
        source = main.loc[main.candidate.eq(candidate) & main["view"].eq(s)].iloc[0]
        for metric_map in (best, metrics, run["metrics"]):
            same([metric_map[s][m] for m in METRICS], source[METRICS].to_numpy(float))
        row = {"split_mode": mode, "candidate": candidate, "Scope": s,
               "epoch_zero_kind": "zero_relation_fixed_blend" if route == "e1" else "initial_random_projection",
               "best_epoch": best_epoch, "best_trained_epoch": trained_epoch,
               "best_trained_evidence": "saved_separate_checkpoint_metrics" if route == "e1" else "derived_history_not_separate_checkpoint",
               "epochs_completed": len(history), "optimizer_steps": run["optimizer_steps"],
               "examples_processed": run["examples_processed"], "runtime_seconds": run["runtime_seconds"],
               "summed_training_epoch_seconds": sum(h["epoch_seconds"] for h in history),
               "trainable_parameters": run["trainable_parameters"], "gpu": run["gpu"], "seed": SEED}
        # Production E1/E2 read manifest.split_mode; the CLI split-mode option
        # is used only when constructing a synthetic smoke cache.
        row["synthetic_smoke_only_cli_split_mode"] = args.get("split_mode")
        row["effective_split_mode_source"] = "cache_manifest_recorded_in_run"
        for label, values in (("epoch_zero", initial), ("best", best), ("best_trained", trained)):
            row.update({f"{label}_{metric}": values[s][metric] for metric in METRICS})
        rows.append(row)
    return rows


def diagnostic_tables(inputs, mode, selection, main):
    role = mode + "_diagnostics"
    diag = inputs.read(role, "summary.json")
    inputs.check(role, "summary.json", selection["diagnostics_sha256"])
    validate_basic(diag, mode, contract=True)
    require(diag.get("parameters_updated") is False and diag.get("query_labels_used_for_routing") is False and
            diag.get("reliability_gate_learned") is False, "diagnostic changed weights/routed using labels")
    require(diag.get("personal_training_budget") == 320 and diag.get("retrieval_count") == 5, "diagnostic label budget changed")
    require(diag.get("temporal_matching_seed") == MATCHING_SEED, "temporal matching seed changed")
    require(diag["lineage_audit"] == {"n_train": 656320, "n_validation": 82040, "n_participants": 2051, "cross_role_lineage": "pass"}, "lineage audit failed")
    replay = diag["reproduction"]
    require(replay.get("status") == "pass" and replay.get("exact_query_keys") is True and replay.get("n_queries") == 82040, "v1 replay incomplete")
    require(0 <= replay["max_absolute_error_mmhg"] <= replay["absolute_tolerance_mmhg"] <= 2e-5, "v1 replay tolerance failed")
    for key in ("cache_manifest_sha256", "source_checkpoint_sha256"):
        valid_hash(diag[key])
        require(diag[key] == selection[key], "diagnostic/report source mismatch")
    frames = {}
    for name in ("metrics.csv", "pooled_diagnostics.csv", "subgroup_diagnostics.csv"):
        entry = diag["files"][name]
        require(entry["path"] == name, "diagnostic path mismatch")
        inputs.check(role, name, entry["sha256"])
        frames[name] = inputs.read(role, name)
    raw = frames["metrics.csv"]
    allowed = DIAG_METRICS + ["participant_mean_mae_p90", "participant_mean_mae_p95", "worst_30_n_participants", "worst_30_mean_mae", "retained_70_n_participants", "retained_70_mean_mae", "tail_definition"]
    require(list(raw.columns) == allowed, "unexpected/private diagnostic columns")
    macro = raw[DIAG_METRICS].copy()  # Oracle tail summaries deliberately excluded.
    require(not macro.duplicated(["Setting", "Scope"]).any(), "duplicate diagnostic rows")
    require(set(macro.Setting) == set(diag["metrics"]), "diagnostic setting mismatch")
    for setting, rows in macro.groupby("Setting"):
        validate_metric_map(rows.set_index("Scope").to_dict("index"))
        for _, row in rows.iterrows():
            same(row[METRICS].to_numpy(float), [diag["metrics"][setting][row.Scope][k] for k in METRICS])
    for candidate in DIAGNOSTICS:
        left = main.loc[main.candidate.eq(candidate)].sort_values("view")
        right = macro.loc[macro.Setting.eq(candidate)].sort_values("Scope")
        same(left[METRICS], right[METRICS])
    pooled = frames["pooled_diagnostics.csv"]
    validate_pooled(pooled, macro, setting="Setting", scope="Scope")
    subgroup = frames["subgroup_diagnostics.csv"]
    require(list(subgroup.columns) == SUBGROUP and not subgroup.duplicated(SUBGROUP[:3]).any(), "unexpected/private subgroup schema")
    require(set(subgroup.Setting) == set(macro.Setting) and set(subgroup.Scope) == set(EXPECTED_COUNTS), "subgroup scope/setting mismatch")
    numeric = subgroup[SUBGROUP[3:]].to_numpy(float)
    require(np.isfinite(numeric).all(), "nonfinite subgroup metric")
    require(subgroup.n_participants.ge(1).all() and subgroup.n_events.ge(subgroup.n_participants).all(), "invalid subgroup counts")
    same(subgroup.mean_mae, (subgroup.sbp_mae + subgroup.dbp_mae) / 2)
    temporal = []
    expected_conditions = {"gap0", "gap60", "gap300"} | ({"past_gap0", "past_gap60", "past_gap300"} if mode == MODES[0] else set())
    require(set(diag["conditions"]) == expected_conditions, "temporal conditions missing")
    for name, condition in diag["conditions"].items():
        require(condition["all_queries_retained"] is True and condition["temporal_fit"]["fit_role"] == "train", "temporal query exclusion/fit-role changed")
        gap = int(name.split("gap")[-1])
        require(condition["gap_s"] == gap and condition["past_only"] == (name.startswith("past_") or mode == MODES[1]), "temporal condition mismatch")
        for role_name, counts in (("train", 656320), ("validation", 82040)):
            values = condition["roles"][role_name]
            require(values["n_queries"] == counts, "temporal coverage changed")
            row = {"split_mode": mode, "condition": name, "role": role_name, "gap_s": gap,
                   "past_only": condition["past_only"], "n_time_matched_validation_queries": condition["n_time_matched_queries"]}
            for k in ("n_queries", "n_memory_fallback", "n_alpha_zero", "n_time_fallback"):
                require(0 <= values[k] <= counts, "invalid temporal fallback count")
                row[k] = values[k]
            for statistic in ("nearest_feature_distance", "selected_record_time_gaps_s", "alpha"):
                row.update({f"{statistic}_{k}": values[statistic][k] for k in ("n_finite", "mean", "p05", "p50", "p95")})
            temporal.append(row)
    return diag, macro, pooled, subgroup, temporal


def render_tables(macro, pooled):
    def table(frame):
        def cell(value):
            return (f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)).replace("|", "\\|").replace("*", "\\*")
        return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |",
                          *("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))])
    lines = ["# Personal-memory v2: complete aggregate result tables", "",
             "Registered-user internal development, seed 20260907; 320 labelled 10-second train windows per person. No held-out evaluation.", "",
             "Participant-macro MAE is primary. Overall uses the original full-cohort aggregate, not an equal average of sources. MIMIC and VitalDB are internal source strata, not independent external validation.", "",
             "CSV files retain full precision. All MAE, ME and STD values are mmHg; threshold columns are percentages. Pooled MAE equals participant-macro MAE here because each participant contributes exactly 40 queries.", "",
             "AAMI/BHS labels marked * are retrospective numerical screens only, not clinical certification, formal device-validation passes or evidence of clinical validity."]
    for mode in MODES:
        lines += ["", f"## {mode}"]
        for scope in EXPECTED_COUNTS:
            primary = macro.loc[macro.split_mode.eq(mode) & macro["view"].eq(scope), ["candidate", "n_participants", "n_events", *METRICS]]
            secondary = pooled.loc[pooled.split_mode.eq(mode) & pooled.Scope.eq(scope), POOLED[1:]]
            lines += ["", f"### {scope}", "", "Primary participant-macro results", "", table(primary), "",
                      "Secondary window-pooled numerical diagnostics", "", table(secondary)]
    return "\n".join(lines) + "\n"


def build_publication(input_root):
    inputs = Inputs(input_root)
    macros, pools, comparisons_all, training, diagnostic_macro, diagnostic_pooled, subgroups, temporal = [], [], [], [], [], [], [], []
    references, evidence, code_trees = {}, {}, set()
    for mode in MODES:
        role = mode + "_report"
        selection = inputs.read(role, "selection.json")
        validate_basic(selection, mode, seed=True)
        require(selection.get("complete_query_coverage") is True and selection.get("selection_is_development_only") is True, "filtered/nondevelopment result")
        names = ("participant_macro_summary.csv", "event_pooled_diagnostics_all_scopes.csv", "comparison_vs_reference.csv")
        require(set(selection["files"]) == set(names), "report file matrix mismatch")
        for name in names:
            inputs.check(role, name, selection["files"][name])
        macro, pooled, comparison = (inputs.read(role, name) for name in names)
        validate_macro(macro, mode)
        validate_pooled(pooled, macro)
        recomputed, reference = comparisons(macro)
        require(reference == selection["reference"], "stronger paired reference changed")
        frame_equal(comparison, recomputed, ["candidate", "view", "split_mode"])
        references[mode] = reference
        diag, dmacro, dpool, subgroup, time_rows = diagnostic_tables(inputs, mode, selection, macro)
        records = selection["runs"]
        require(len(records) == 3 and {r["candidate"] for r in records} == {"E1_matched_relation", "E2_bp_state_metric", "lora_continued_control"}, "fit record coverage mismatch")
        for record in records:
            for key in ("run_sha256", "prediction_sha256", "checkpoint_sha256"):
                valid_hash(record[key])
        for route, candidate in (("e1", "E1_matched_relation"), ("e2", "E2_bp_state_metric")):
            run_role = mode + "_" + route
            record = next(r for r in records if r["candidate"] == candidate)
            inputs.check(run_role, "run.json", record["run_sha256"])
            run, history, metrics = (inputs.read(run_role, name) for name in ("run.json", "history.json", "metrics.json"))
            for key in ("cache_manifest_sha256", "source_checkpoint_sha256"):
                require(run[key] == diag[key], "run/diagnostic source mismatch")
            require(run["checkpoint_sha256"] == record["checkpoint_sha256"], "recorded checkpoint hashes disagree")
            valid_hash(run["source_tree_sha256"])
            code_trees.add(run["source_tree_sha256"])
            if route == "e1":
                mismatch = inputs.read(run_role, "reference_mismatch.json")
                require(mismatch == run["reference_mismatch"], "E1 reference diagnostics changed")
                require(mismatch["policy"] == "physical-gap-v2" and mismatch["alpha_recomputed"] is False and
                        mismatch["validation_references_recomputed"] is False, "E1 reference policy changed")
            training += validate_training(run, history, metrics, mode, route, macro)
        valid_hash(diag["source_tree_sha256"])
        code_trees.add(diag["source_tree_sha256"])
        evidence[mode] = {"cache_manifest_sha256": diag["cache_manifest_sha256"],
                          "source_checkpoint_sha256": diag["source_checkpoint_sha256"],
                          "v1_replay_max_absolute_error_mmhg": diag["reproduction"]["max_absolute_error_mmhg"],
                          "diagnostics_runtime_seconds": diag["runtime_seconds"]}
        macros.append(macro)
        pools.append(pooled.assign(split_mode=mode))
        comparisons_all.append(comparison)
        diagnostic_macro.append(dmacro.assign(split_mode=mode))
        diagnostic_pooled.append(dpool.assign(split_mode=mode))
        subgroups.append(subgroup.assign(split_mode=mode))
        temporal += time_rows
    require(len(code_trees) == 1, "training/diagnostic source code changed across jobs")
    combined = pd.concat(comparisons_all, ignore_index=True)
    final = inputs.read("both_final", "selection.json")
    validate_basic(final, seed=True)
    require(final.get("selection_is_development_only") is True and final.get("automatically_promoted") is False and
            final.get("learned_E3_gate_trained") is False and final.get("new_module_contribution_requires_fixed_blend_and_temporal_controls") is True, "final claim/decision contract changed")
    frame_equal(inputs.read("both_final", "cross_split_comparison.csv"), combined, ["candidate", "split_mode", "view"])
    gate = pd.DataFrame(eligibility(combined))
    frame_equal(inputs.read("both_final", "promotion_gate.csv"), gate, ["candidate"])
    eligible = gate.loc[gate.passes_historical_LoRA_upgrade_gate, "candidate"].tolist()
    require(final.get("eligible_by_historical_LoRA_gate") == eligible, "eligibility list disagrees")
    require(len(inputs.files) == 33, "expected exactly 33 validated input artifacts")
    macro = pd.concat(macros, ignore_index=True)
    pooled = pd.concat(pools, ignore_index=True)
    frames = {"participant_macro_summary.csv": macro, "event_pooled_diagnostics_all_scopes.csv": pooled,
              "comparison_vs_reference.csv": combined, "promotion_gate.csv": gate,
              "training_summary.csv": pd.DataFrame(training), "diagnostic_participant_macro.csv": pd.concat(diagnostic_macro, ignore_index=True),
              "diagnostic_pooled.csv": pd.concat(diagnostic_pooled, ignore_index=True),
              "subgroup_diagnostics.csv": pd.concat(subgroups, ignore_index=True), "temporal_diagnostics.csv": pd.DataFrame(temporal)}
    outputs = {name: frame.to_csv(index=False, lineterminator="\n") for name, frame in frames.items()}
    outputs["RESULT_TABLES.md"] = render_tables(macro, pooled)
    code_paths = [Path(__file__), REPO / "src/pulsedb_fewshot/personal_memory_v2_report.py",
                  REPO / "src/pulsedb_fewshot/personal_memory_report.py", REPO / "src/pulsedb_fewshot/personal_memory_control.py",
                  REPO / "src/pulsedb_fewshot/personal_memory_diagnostics.py",
                  REPO / "src/pulsedb_fewshot/personal_memory_matched.py", REPO / "src/pulsedb_fewshot/personal_memory_metric.py",
                  REPO / "docs/PLAN_PERSONAL_MEMORY_V2.md"]
    verification = {"status": "pass", "screen_id": SCREEN, "seed": SEED, "validated_input_files": inputs.files,
                    "code_contract_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in code_paths},
                    "source_training_tree_sha256": next(iter(code_trees)), "recorded_lineage": evidence,
                    "rows": {name: len(frame) for name, frame in frames.items()}, "references": references,
                    "eligible_by_historical_LoRA_gate": eligible, "automatically_promoted": False,
                    "heldout_test_accessed": False, "learned_E3_gate_trained": False,
                    "validation_level": "aggregate-only recorded-contract and hash cross-check",
                    "raw_prediction_recomputation": False, "checkpoint_bytes_read": False,
                    "continued_LoRA_raw_run_rechecked": False,
                    "metadata_notes": ["Chronological E1/E2 arguments.split_mode records the random_disjoint CLI default. Production retrieval uses cache manifest split_mode; that argument is only used for synthetic smoke cache creation.",
                                       "Frozen temporal matching uses seed 20260908, while trainable fits use 20260907."],
                    "limits": "Original server reports recomputed frozen predictions. This publisher does not repeat that computation; continued-LoRA lineage is checked only through report records. Single-seed internal development is not clinical validation.",
                    "oracle_tail_outputs_excluded": True,
                    "output_sha256": {name: sha256(value.encode("utf-8")).hexdigest() for name, value in outputs.items()}}
    outputs["verification.json"] = json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for name, value in outputs.items():
        require(FORBIDDEN.search(value) is None, "private data/path detected in public output: " + name)
    return outputs, verification


def publish(input_root, output_dir):
    outputs, verification = build_publication(input_root)
    output = Path(output_dir).resolve()
    require(not output.is_relative_to(Path(input_root).resolve()), "public output cannot be inside private input tree")
    # Preflight the whole set before writing anything: changed files never
    # overwrite prior publication, while an identical rerun is idempotent.
    for name, value in outputs.items():
        path = output / name
        require(not path.is_symlink(), "refuse output symlink")
        if path.exists() and path.read_bytes() != value.encode("utf-8"):
            raise FileExistsError("preserve changed existing output: " + name)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        path = output / name
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(value.encode("utf-8"))
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.input_root, args.output_dir)
    print(json.dumps({"status": result["status"], "input_files": len(result["validated_input_files"]), "rows": result["rows"],
                      "eligible_by_historical_LoRA_gate": result["eligible_by_historical_LoRA_gate"]}, indent=2))


if __name__ == "__main__":
    main()
