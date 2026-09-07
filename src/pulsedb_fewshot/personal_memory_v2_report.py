"""Recompute v2 comparisons on all frozen registered-user validation queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .personal_memory_control import digest
from .personal_memory_report import validate_predictions, EXPECTED_COUNTS, save

SCREEN = "personal-memory-v2"
MODES = ("random_disjoint", "chronological_blocked")
CANDIDATES = ("E1_matched_relation", "E2_bp_state_metric",
              "E3_zero_reliability", "E3_trained_reliability")
DIAGNOSTICS = ("D0_frozen_lora", "fixed_blend", "trained_blend",
               "E3_zero_reliability", "E3_trained_reliability")


def source_target_domain(frame):
    """Same float32 reference domain as the original audited LoRA export.

    This does not change prediction values or enlarge the 2e-5 comparison
    tolerance; it avoids comparing raw-float64 vs standardized-float32 labels.
    """
    frame = frame.copy()
    for name in ("target_sbp", "target_dbp"):
        frame[name] = frame[name].astype(np.float32)
    return frame


def comparisons(macro: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    required = set(CANDIDATES) | set(DIAGNOSTICS) | {"lora_continued_control"}
    if set(macro.candidate) != required or len(macro) != len(required) * 3:
        raise ValueError("incomplete v2 candidate/control matrix")
    if macro.duplicated(["candidate", "view"]).any():
        raise ValueError("duplicate candidate/source rows")
    if not np.isfinite(macro["mean_mae"].to_numpy(float)).all():
        raise ValueError("nonfinite comparison metric")
    for _, row in macro.iterrows():
        if (int(row["n_participants"]), int(row["n_events"])) != EXPECTED_COUNTS[row["view"]]:
            raise ValueError("cohort changed")
    controls = macro.loc[macro.candidate.isin(("D0_frozen_lora", "lora_continued_control"))
                         & macro["view"].eq("Overall")]
    reference = controls.sort_values(["mean_mae", "candidate"]).iloc[0]["candidate"]
    result = macro.copy()
    for comparator, column in ((reference, "reference"), ("fixed_blend", "fixed_blend"),
                               ("trained_blend", "v1_trained_blend")):
        by_scope = macro.loc[macro.candidate.eq(comparator)].set_index("view").mean_mae
        result[f"{column}_mean_mae"] = result["view"].map(by_scope)
        result[f"gain_vs_{column}_mmhg"] = result[f"{column}_mean_mae"] - result.mean_mae
    result["reference_candidate"] = reference
    return result, reference


def eligibility(combined: pd.DataFrame) -> list[dict]:
    rows = []
    if set(combined.split_mode) != set(MODES):
        raise ValueError("both modes required")
    for candidate in CANDIDATES:
        data = combined.loc[combined.candidate.eq(candidate)]
        if len(data) != 6 or data.duplicated(["split_mode", "view"]).any():
            raise ValueError("six paired source/mode rows required")
        if set(zip(data.split_mode, data["view"])) != {(m, s) for m in MODES for s in EXPECTED_COUNTS}:
            raise ValueError("incomplete source/mode cross product")
        overall = data.loc[data["view"].eq("Overall")]
        source = data.loc[~data["view"].eq("Overall")]
        passed = overall.gain_vs_reference_mmhg.ge(.15).all() and source.gain_vs_reference_mmhg.gt(0).all()
        rows.append({"candidate": candidate, "passes_historical_LoRA_upgrade_gate": bool(passed),
                     "beats_fixed_blend_in_both_modes": bool(overall.gain_vs_fixed_blend_mmhg.gt(0).all()),
                     "beats_trained_v1_blend_in_both_modes": bool(overall.gain_vs_v1_trained_blend_mmhg.gt(0).all()),
                     "mean_across_modes": float(overall.mean_mae.mean())})
    return rows


def mode_report(args):
    from .calbased_metrics import participant_macro_views, pooled_diagnostics

    if args.output.exists():
        raise FileExistsError("preserve existing report")
    diag = json.loads((args.diagnostics / "summary.json").read_text(encoding="utf-8"))
    contract = {"protocol_id": "development-calbased-analogue-v1", "source_parent_split": "meta_train",
                "read_roles": ["train", "internal_validation"], "selection_role": "internal_validation",
                "official_pulsedb_calbased_reproduction": False}
    if (diag.get("status") != "complete" or diag.get("screen_id") != SCREEN or
            diag.get("split_mode") != args.split_mode or diag.get("heldout_test_accessed") is not False or
            any(diag.get(k) != v for k, v in contract.items())):
        raise ValueError("invalid/incomplete v2 diagnostic artifact")
    records = []

    def diagnostic_frame(name):
        filename = diag["prediction_files"][name]
        if Path(filename).name != filename:
            raise ValueError("prediction path not top-level")
        path = args.diagnostics / filename
        entry = diag["files"][filename]
        if entry["path"] != filename or digest(path) != entry["sha256"]:
            raise ValueError("changed diagnostic predictions")
        return source_target_domain(pd.read_parquet(path))

    expected = diagnostic_frame("D0_frozen_lora")
    expected = validate_predictions(expected, expected)
    frames = {name: validate_predictions(diagnostic_frame(name), expected) for name in DIAGNOSTICS}
    for path, name, screen in ((args.e1, CANDIDATES[0], SCREEN), (args.e2, CANDIDATES[1], SCREEN),
                               (args.control, "lora_continued_control", "personal-memory-v1")):
        run = json.loads((path / "run.json").read_text(encoding="utf-8"))
        if (run.get("status") != "complete" or run.get("candidate") != name or
                run.get("screen_id") != screen or run.get("split_mode") != args.split_mode or
                run.get("heldout_test_accessed") is not False or run.get("seed") != 20260907 or
                any(run.get(k) != v for k, v in contract.items())):
            raise ValueError("incomplete/wrong experimental run: " + name)
        if run.get("cache_manifest_sha256") != diag["cache_manifest_sha256"]:
            raise ValueError("source cache mismatch")
        source_hash = (run.get("initialization", {}).get("source_checkpoint_sha256")
                       if name == "lora_continued_control" else run.get("source_checkpoint_sha256"))
        if source_hash != diag["source_checkpoint_sha256"]:
            raise ValueError("source personal model mismatch")
        if digest(path / "best.pt") != run.get("checkpoint_sha256"):
            raise ValueError("checkpoint changed")
        prediction_path = path / "best_internal_validation_predictions.parquet"
        frames[name] = validate_predictions(source_target_domain(pd.read_parquet(prediction_path)), expected)
        actual = participant_macro_views(frames[name])
        for scope in EXPECTED_COUNTS:
            for metric in ("sbp_mae", "dbp_mae", "mean_mae"):
                np.testing.assert_allclose(actual[scope][metric], run["metrics"][scope][metric], atol=1e-5, rtol=0)
        records.append({"candidate": name, "run_id": path.name, "run_sha256": digest(path / "run.json"),
                        "prediction_sha256": digest(prediction_path), "checkpoint_sha256": run["checkpoint_sha256"]})
    macro, pooled = [], []
    for name, frame in frames.items():
        for scope, metrics in participant_macro_views(frame).items():
            macro.append({"candidate": name, "view": scope, "split_mode": args.split_mode,
                          **{key: metrics[key] for key in ("n_participants", "n_events", "sbp_mae", "dbp_mae", "mean_mae")}})
        pooled.append(pooled_diagnostics(frame, name))
    macro = pd.DataFrame(macro)
    comparison, reference = comparisons(macro)
    args.output.mkdir(parents=True, exist_ok=False)
    macro.to_csv(args.output / "participant_macro_summary.csv", index=False)
    comparison.to_csv(args.output / "comparison_vs_reference.csv", index=False)
    pd.concat(pooled, ignore_index=True).to_csv(args.output / "event_pooled_diagnostics_all_scopes.csv", index=False)
    selection = {"status": "complete", "screen_id": SCREEN, "split_mode": args.split_mode,
                 "heldout_test_accessed": False, "reference": reference, "seed": 20260907,
                 "cache_manifest_sha256": diag["cache_manifest_sha256"], "runs": records,
                 "diagnostics_sha256": digest(args.diagnostics / "summary.json"),
                 "selection_is_development_only": True, "complete_query_coverage": True,
                 "source_checkpoint_sha256": diag["source_checkpoint_sha256"],
                 "standard_claim": "retrospective numerical screening, not clinical certification"}
    selection["files"] = {p.name: digest(p) for p in args.output.glob("*.csv")}
    save(args.output / "selection.json", selection)
    return selection


def joint_report(paths, output):
    if output.exists() or len(paths) != 2:
        raise ValueError("two reports and a fresh output required")
    seen, frames = set(), []
    for path in paths:
        selection = json.loads((path / "selection.json").read_text(encoding="utf-8"))
        mode = selection.get("split_mode")
        if (mode not in MODES or mode in seen or selection.get("status") != "complete" or
                selection.get("screen_id") != SCREEN or selection.get("heldout_test_accessed") is not False):
            raise ValueError("invalid mode report")
        seen.add(mode)
        filename = "comparison_vs_reference.csv"
        if digest(path / filename) != selection["files"][filename]:
            raise ValueError("changed comparison table")
        frame = pd.read_csv(path / filename)
        if not frame.split_mode.eq(mode).all():
            raise ValueError("mixed split rows")
        recomputed, reference = comparisons(frame)
        if reference != selection["reference"]:
            raise ValueError("unpaired control selection")
        for key in ("gain_vs_reference_mmhg", "gain_vs_fixed_blend_mmhg", "gain_vs_v1_trained_blend_mmhg"):
            np.testing.assert_allclose(frame[key], recomputed[key], rtol=0, atol=1e-7)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    decisions = pd.DataFrame(eligibility(combined))
    output.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output / "cross_split_comparison.csv", index=False)
    decisions.to_csv(output / "promotion_gate.csv", index=False)
    result = {"status": "complete", "screen_id": SCREEN, "heldout_test_accessed": False,
              "selection_is_development_only": True, "seed": 20260907,
              "eligible_by_historical_LoRA_gate": decisions.loc[decisions.passes_historical_LoRA_upgrade_gate, "candidate"].tolist(),
              "new_module_contribution_requires_fixed_blend_and_temporal_controls": True,
              "learned_E3_gate_trained": False, "automatically_promoted": False}
    save(output / "selection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--e1", type=Path)
    parser.add_argument("--e2", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--split-mode", choices=MODES)
    parser.add_argument("--mode-reports", type=Path, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode_reports:
        if any((args.diagnostics, args.e1, args.e2, args.control, args.split_mode)):
            parser.error("joint and per-mode inputs cannot mix")
        result = joint_report(args.mode_reports, args.output)
    else:
        if not all((args.diagnostics, args.e1, args.e2, args.control, args.split_mode)):
            parser.error("per-mode report inputs incomplete")
        result = mode_report(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
