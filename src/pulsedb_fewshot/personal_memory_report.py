"""Complete/skip-safe personal-memory reports from immutable prediction artifacts.

Mode reports recompute Overall/MIMIC/VitalDB on the identical full query cohort.
The final comparison selects the stronger frozen/continued LoRA control using
Overall mean MAE in each mode, then uses that SAME control for its source rows.
No per-source cherry-picking, tail removal or held-out predictions are allowed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .personal_memory_control import CANDIDATE as CONTROL, MODES, SCREEN_ID, digest


NEURAL = ("pair_single", "pair_uniform", "pair_retrieved", "pair_distance_blend")
FROZEN = "D0_frozen_lora"
EXPECTED_COUNTS = {"Overall": (2051, 82040), "MIMIC": (1011, 40440), "VitalDB": (1040, 41600)}
KEYS = ["subject_uid", "event_id", "source"]
BP_COLUMNS = ["target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checked_cache_file(cache: Path, manifest, filename: str) -> Path:
    entry = manifest["files"][filename]
    if entry.get("path") != filename:
        raise ValueError("cache file path is not the expected relative filename")
    path = cache / filename
    if digest(path) != entry["sha256"]:
        raise ValueError("cache file changed: " + filename)
    return path


def validate_predictions(prediction, expected):
    if not set(KEYS + BP_COLUMNS).issubset(prediction):
        raise ValueError("prediction columns incomplete")
    if len(prediction) != 82040 or prediction.event_id.duplicated().any():
        raise ValueError("query coverage or uniqueness violation")
    if prediction[KEYS + BP_COLUMNS].isna().any().any():
        raise ValueError("missing key or prediction")
    if not np.isfinite(prediction[BP_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("non-finite prediction or target")
    joined = expected[KEYS + ["target_sbp", "target_dbp"]].merge(
        prediction[KEYS + BP_COLUMNS], on=KEYS, how="left", validate="one_to_one", suffixes=("_expected", ""))
    if joined.isna().any().any():
        raise ValueError("candidate query keys differ from frozen reference")
    for bp in ("sbp", "dbp"):
        np.testing.assert_allclose(joined[f"target_{bp}"], joined[f"target_{bp}_expected"], rtol=0, atol=2e-5)
    for scope, (people, windows) in EXPECTED_COUNTS.items():
        frame = joined if scope == "Overall" else joined.loc[joined.source.eq(scope)]
        if len(frame) != windows or frame.subject_uid.nunique() != people:
            raise ValueError("candidate does not cover the full fixed source cohort")
        if not frame.groupby("subject_uid").size().eq(40).all():
            raise ValueError("each participant must retain all 40 queries")
    if joined.groupby("subject_uid").source.nunique().max() != 1:
        raise ValueError("participant source mapping is inconsistent")
    return joined[KEYS + BP_COLUMNS]


def mode_report(cache: Path, runs: list[Path], output: Path, split_mode: str):
    from .calbased_metrics import participant_macro_views, pooled_diagnostics

    if len(runs) != 5 or len({p.resolve() for p in runs}) != 5:
        raise ValueError("mode report requires one control and four distinct neural runs")
    manifest_path = cache / "manifest.json"
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (manifest.get("status") == "complete" and manifest.get("screen_id") == SCREEN_ID and
            manifest.get("split_mode") == split_mode and manifest.get("heldout_test_accessed") is False and
            manifest.get("source_parent_split") == "meta_train" and
            manifest.get("read_roles") == ["train", "internal_validation"]):
        raise ValueError("invalid completed development cache")
    expected = pd.read_parquet(checked_cache_file(cache, manifest, "private_main_D0_predictions.parquet"))
    canonical = pd.read_parquet(checked_cache_file(cache, manifest, "validation_metadata.parquet"))
    if not canonical.role.eq("internal_validation").all():
        raise ValueError("cache query role is forbidden")
    base = np.load(checked_cache_file(cache, manifest, "validation_base.npy"), mmap_mode="r", allow_pickle=False)
    targets = np.load(checked_cache_file(cache, manifest, "validation_bp.npy"), mmap_mode="r", allow_pickle=False)
    canonical = canonical[KEYS].copy()
    canonical[["target_sbp", "target_dbp"]] = targets
    expected = validate_predictions(expected, canonical)
    np.testing.assert_allclose(expected[["pred_sbp", "pred_dbp"]], base, atol=1e-6, rtol=0)
    summary = json.loads(checked_cache_file(cache, manifest, "probe_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("heldout_test_accessed") is not False:
        raise ValueError("incomplete or invalid original probe result")
    results = [(FROZEN, expected, None)]
    records, candidates, gate_hashes, decisions = [], set(), set(), set()
    for directory in runs:
        run_path = directory / "run.json"
        record = json.loads(run_path.read_text(encoding="utf-8"))
        candidate = record.get("candidate")
        if candidate not in {CONTROL, *NEURAL} or candidate in candidates:
            raise ValueError("unexpected or duplicate candidate")
        candidates.add(candidate)
        if record.get("screen_id") != SCREEN_ID or record.get("split_mode") != split_mode or record.get("heldout_test_accessed") is not False:
            raise ValueError("candidate screen/split/held-out mismatch")
        if record.get("cache_manifest_sha256") != manifest_hash:
            raise ValueError("candidate trained against a different cache")
        gate_hash = record.get("gate_manifest_sha256", record.get("gate", {}).get("manifest_sha256"))
        if not isinstance(gate_hash, str) or len(gate_hash) != 64:
            raise ValueError("candidate must identify its immutable global gate")
        gate_hashes.add(gate_hash)
        status = record.get("status")
        decisions.add(status)
        records.append({"candidate": candidate, "status": status, "run_id": directory.name,
                        "run_manifest_sha256": digest(run_path), "checkpoint_sha256": record.get("checkpoint_sha256")})
        if status == "skipped_no_complementarity":
            continue
        if status != "complete" or record.get("seed") != 20260907:
            raise ValueError("candidate is failed, partial, synthetic, or wrong seed")
        if not (record.get("protocol_id") == "development-calbased-analogue-v1" and
                record.get("source_parent_split") == "meta_train" and
                record.get("read_roles") == ["train", "internal_validation"] and
                record.get("selection_role") == "internal_validation" and
                record.get("official_pulsedb_calbased_reproduction") is False):
            raise ValueError("candidate data access contract differs")
        if digest(directory / "best.pt") != record.get("checkpoint_sha256"):
            raise ValueError("completed candidate checkpoint hash mismatch")
        prediction = pd.read_parquet(directory / "best_internal_validation_predictions.parquet")
        prediction = validate_predictions(prediction, expected)
        actual = participant_macro_views(prediction)
        for scope in EXPECTED_COUNTS:
            for metric in ("sbp_mae", "dbp_mae", "mean_mae"):
                np.testing.assert_allclose(actual[scope][metric], record["metrics"][scope][metric], atol=1e-5, rtol=0)
        results.append((candidate, prediction, record))
    if candidates != {CONTROL, *NEURAL} or len(gate_hashes) != 1:
        raise ValueError("candidate matrix or shared gate mismatch")
    if len(decisions) != 1 or not decisions.issubset({"complete", "skipped_no_complementarity"}):
        raise ValueError("mixed completion/skip states contradict one global scientific gate")
    macro_rows, pooled = [], []
    for candidate, prediction, record in results:
        metrics = participant_macro_views(prediction)
        if candidate == FROZEN:
            for scope in EXPECTED_COUNTS:
                for metric in ("sbp_mae", "dbp_mae", "mean_mae"):
                    np.testing.assert_allclose(metrics[scope][metric], summary["overall"]["D0"][scope][metric], atol=1e-5, rtol=0)
        for scope, values in metrics.items():
            macro_rows.append({"candidate": candidate, "view": scope, "split_mode": split_mode, "seed": 20260907,
                              **{name: values[name] for name in ("n_participants", "n_events", "sbp_mae", "dbp_mae", "mean_mae", "sbp_bias", "dbp_bias")}})
        pooled.append(pooled_diagnostics(prediction, candidate))
    macro = pd.DataFrame(macro_rows)
    diagnostics = pd.concat(pooled, ignore_index=True)
    reference_rows = macro.loc[macro.candidate.isin((FROZEN, CONTROL)) & macro.view.eq("Overall")]
    selected_reference = reference_rows.sort_values(["mean_mae", "candidate"], kind="mergesort").iloc[0].candidate
    reference = macro.loc[macro.candidate.eq(selected_reference), ["view", "mean_mae"]].rename(columns={"mean_mae": "reference_mean_mae"})
    comparison = macro.merge(reference, on="view", validate="many_to_one")
    comparison["reference_candidate"] = selected_reference
    comparison["gain_mmhg"] = comparison.reference_mean_mae - comparison.mean_mae
    skipped = decisions == {"skipped_no_complementarity"}
    selection = {"status": "skipped_no_complementarity" if skipped else "complete", "screen_id": SCREEN_ID,
                 "split_mode": split_mode, "seed": 20260907, "heldout_test_accessed": False,
                 "official_pulsedb_calbased_reproduction": False, "reference": selected_reference,
                 "gate_manifest_sha256": next(iter(gate_hashes)), "cache_manifest_sha256": manifest_hash,
                 "source_checkpoint_sha256": manifest["source_checkpoint_sha256"], "runs": records,
                 "complete_query_coverage": True, "selection_role": "internal_validation",
                 "decision": "retain frozen reference; neural screen deliberately skipped" if skipped else "await joint two-mode gate",
                 "standard_claim": "retrospective numerical screens only, not clinical certification"}
    output.mkdir(parents=True, exist_ok=False)
    macro.to_csv(output / "participant_macro_summary.csv", index=False)
    diagnostics.to_csv(output / "event_pooled_diagnostics_all_scopes.csv", index=False)
    comparison.to_csv(output / "comparison_vs_reference.csv", index=False)
    for scope in EXPECTED_COUNTS:
        diagnostics.loc[diagnostics.Scope.eq(scope)].to_csv(output / f"event_pooled_{scope.lower()}.csv", index=False)
    save(output / "selection.json", selection)
    (output / "README.md").write_text(
        "# Personal-memory development result\n\n" +
        ("The global complementarity gate stopped neural training; all five training jobs are explicit successful skips. " if skipped else "All five training jobs completed, with complete identical internal-validation queries. ") +
        "The reference is the lower-Overall-MAE frozen/continued LoRA in this mode; its source rows remain paired. "
        "Do not promote a candidate before the joint two-mode gate. AAMI/BHS entries are retrospective numerical screens, not clinical certification.\n", encoding="utf-8")
    return selection


def combined_report(reports: list[Path], output: Path):
    if len(reports) != 2 or len({p.resolve() for p in reports}) != 2:
        raise ValueError("exactly two distinct mode reports are required")
    frames, selections = [], {}
    for directory in reports:
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        mode = selection.get("split_mode")
        if mode not in MODES or mode in selections or selection.get("screen_id") != SCREEN_ID or selection.get("heldout_test_accessed") is not False:
            raise ValueError("invalid or duplicate split report")
        if selection.get("status") not in {"complete", "skipped_no_complementarity"}:
            raise ValueError("report is not complete or a valid scientific skip")
        if selection.get("official_pulsedb_calbased_reproduction") is not False:
            raise ValueError("incorrect official benchmark claim")
        selections[mode] = selection
        frame = pd.read_csv(directory / "comparison_vs_reference.csv")
        if not frame.split_mode.eq(mode).all() or frame.duplicated(["candidate", "view"]).any():
            raise ValueError("comparison scope mismatch")
        expected = {FROZEN} if selection["status"] == "skipped_no_complementarity" else {FROZEN, CONTROL, *NEURAL}
        if set(frame.candidate) != expected or len(frame) != 3 * len(expected) or set(frame.view) != set(EXPECTED_COUNTS):
            raise ValueError("incomplete candidate matrix in mode report")
        for _, row in frame.iterrows():
            if (int(row.n_participants), int(row.n_events)) != EXPECTED_COUNTS[row.view]:
                raise ValueError("source cohort changed")
        references = frame.loc[frame.candidate.eq(selection["reference"])].set_index("view")
        for _, row in frame.iterrows():
            np.testing.assert_allclose(row.gain_mmhg, references.loc[row.view, "mean_mae"] - row.mean_mae, rtol=0, atol=1e-6)
        frames.append(frame)
    if len({value["gate_manifest_sha256"] for value in selections.values()}) != 1:
        raise ValueError("mode reports have different global scientific gates")
    statuses = {value["status"] for value in selections.values()}
    if len(statuses) != 1:
        raise ValueError("global scientific gate produced inconsistent mode statuses")
    combined = pd.concat(frames, ignore_index=True)
    skipped = statuses == {"skipped_no_complementarity"}
    rows = []
    if not skipped:
        for candidate in NEURAL:
            data = combined.loc[combined.candidate.eq(candidate)]
            overall = data.loc[data.view.eq("Overall")]
            sources = data.loc[~data.view.eq("Overall")]
            passed = len(overall) == 2 and len(sources) == 4 and overall.gain_mmhg.ge(.15).all() and sources.gain_mmhg.gt(0).all()
            rows.append({"candidate": candidate, "passes_accuracy_gate": bool(passed),
                         "mean_across_modes": float(overall.mean_mae.mean())})
    gate = pd.DataFrame(rows, columns=["candidate", "passes_accuracy_gate", "mean_across_modes"])
    eligible = [] if gate.empty else gate.loc[gate.passes_accuracy_gate].sort_values("mean_across_modes").candidate.tolist()
    result = {"status": "skipped_no_complementarity" if skipped else "complete", "screen_id": SCREEN_ID,
              "seed": 20260907, "heldout_test_accessed": False, "official_pulsedb_calbased_reproduction": False,
              "eligible_candidates": eligible, "references": {mode: value["reference"] for mode, value in selections.items()},
              "gate_manifest_sha256": next(iter(selections.values()))["gate_manifest_sha256"],
              "recommendation": "retain LoRA; no empirical complementarity" if skipped else
                  "confirm eligible candidates with new seeds" if eligible else "retain stronger paired LoRA reference",
              "selection_is_development_only": True, "hypothesis_validation_not_publication_guarantee": True}
    output.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output / "cross_split_comparison.csv", index=False)
    gate.to_csv(output / "promotion_gate.csv", index=False)
    save(output / "selection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--runs", type=Path, nargs=5)
    parser.add_argument("--split-mode", choices=MODES)
    parser.add_argument("--mode-reports", type=Path, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode_reports:
        if any((args.cache_dir, args.runs, args.split_mode)):
            parser.error("mode-reports cannot be combined with mode-level inputs")
        result = combined_report(args.mode_reports, args.output)
    else:
        if not all((args.cache_dir, args.runs, args.split_mode)):
            parser.error("mode report requires cache-dir, five runs, and split-mode")
        result = mode_report(args.cache_dir, args.runs, args.output, args.split_mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
