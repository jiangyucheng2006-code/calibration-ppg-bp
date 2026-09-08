"""One-way scoring of frozen predictions on exact official CalBased test IDs.

All model files, prediction coverage and frozen plan hashes are verified before
the target table is opened. Only exact official test IDs and five label/identity
columns are requested from the existing index. This module neither trains nor
chooses a checkpoint, threshold, routing rule, exclusion or model winner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .official_calbased_train import PROTOCOL_ID, event_ids_sha256, save_json, sha256


KEYS = ["subject_uid", "segment_uid", "source"]
PREDICTION_COLUMNS = [*KEYS, "pred_sbp", "pred_dbp"]
TARGET_COLUMNS = [*KEYS, "sbp", "dbp"]


def _digest(value, name):
    if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
        raise ValueError(f"missing SHA-256: {name}")
    return value


def _path(base: Path, value) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def validate_predictions(frame, test_inputs):
    result = frame.copy()
    if "event_id" in result and "segment_uid" not in result:
        result = result.rename(columns={"event_id": "segment_uid"})
    elif "event_id" in result and not result.event_id.astype(str).equals(result.segment_uid.astype(str)):
        raise ValueError("ambiguous event/segment identities")
    if not set(PREDICTION_COLUMNS).issubset(result):
        raise ValueError("prediction table lacks required identities/predictions")
    forbidden = {"sbp", "dbp", "target_sbp", "target_dbp", "SegSBP", "SegDBP"}
    if forbidden & set(result):
        raise ValueError("frozen prediction file must not contain test targets")
    for column in KEYS:
        if result[column].isna().any():
            raise ValueError("null prediction identity")
        result[column] = result[column].astype(str)
    if result.segment_uid.duplicated().any() or len(result) != len(test_inputs):
        raise ValueError("prediction coverage is duplicated or incomplete")
    if not np.isfinite(result[["pred_sbp", "pred_dbp"]].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite frozen predictions")
    expected = test_inputs[KEYS].astype(str)
    joined = expected.merge(result[PREDICTION_COLUMNS], on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("predictions are not the exact official test cohort/windows")
    return joined.drop(columns="_merge")


def verify_frozen_bundle(args):
    """Return verified predictions without requesting any reference BP values."""
    plan_path = args.frozen_plan.resolve()
    if sha256(plan_path) != _digest(args.expected_plan_sha256, "frozen plan"):
        raise ValueError("frozen prediction plan checksum mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("protocol_id") != PROTOCOL_ID or plan.get("status") != "frozen_predictions"
            or plan.get("methods_frozen_before_test_labels") is not True
            or plan.get("allow_test_based_selection") is not False):
        raise ValueError("plan is not a pre-target frozen official evaluation")
    if bool(plan.get("synthetic_smoke", False)) != args.smoke:
        raise ValueError("synthetic and formal evaluation contracts cannot be mixed")
    store = args.store_root.resolve()
    store_hash = sha256(store / "manifest.json")
    if plan.get("official_contract_sha256") != store_hash:
        raise ValueError("official data contract changed")
    store_manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    if (store_manifest.get("protocol_id") != PROTOCOL_ID or store_manifest.get("status") != "ready"
            or bool(store_manifest.get("synthetic", False)) != args.smoke):
        raise ValueError("store is not ready under this official contract")
    research_plan = _path(plan_path.parent, plan["research_plan_path"])
    if sha256(research_plan) != _digest(plan.get("research_plan_sha256"), "research plan"):
        raise ValueError("frozen research plan changed")
    test_path = store / "test_inputs.parquet"
    if sha256(test_path) != plan.get("test_inputs_sha256"):
        raise ValueError("official test input membership changed")
    test = pd.read_parquet(test_path)
    if not set(KEYS).issubset(test) or set(TARGET_COLUMNS[3:]) & set(test):
        raise ValueError("official test inputs missing IDs or containing targets")
    if any(str(c).lower().startswith(("target_", "abp", "segsbp", "segdbp")) for c in test):
        raise ValueError("official input manifest contains reference-derived fields")
    if test[KEYS].isna().any().any():
        raise ValueError("official test input identity is missing")
    test = test[KEYS].astype(str)
    if test.segment_uid.duplicated().any() or set(test.source) != {"MIMIC", "VitalDB"}:
        raise ValueError("invalid official test identities/source strata")
    counts = test.groupby("subject_uid").size()
    if not args.smoke and (len(counts) != 2506 or not counts.eq(40).all()):
        raise ValueError("exact official CalBased test requires 2506 x 40 windows")
    if event_ids_sha256(test.segment_uid) != plan.get("query_ids_sha256"):
        raise ValueError("frozen query ID set changed")
    if plan.get("official_train_windows_per_subject") != 360:
        raise ValueError("final models must use the declared official 360-window training budget")
    # File hashing does not deserialize BP values; actual target reading is later.
    if sha256(args.full_index) != plan.get("full_index_sha256"):
        raise ValueError("source label index checksum changed")
    candidates = plan.get("candidates", [])
    names = [item.get("name") for item in candidates]
    if not names or len(names) != len(set(names)) or any(not re.fullmatch(r"[a-z][a-z0-9_]*", str(n)) for n in names):
        raise ValueError("invalid/duplicate frozen candidate names")
    loaded, receipts = {}, {}
    for candidate in candidates:
        name = candidate["name"]
        predictions_path = _path(plan_path.parent, candidate["predictions_path"])
        if sha256(predictions_path) != candidate.get("predictions_sha256"):
            raise ValueError(f"frozen predictions changed: {name}")
        final_run_path = _path(plan_path.parent, candidate["final_base_run_path"])
        if sha256(final_run_path) != candidate.get("final_base_run_sha256"):
            raise ValueError(f"final training receipt changed: {name}")
        final_run = json.loads(final_run_path.read_text(encoding="utf-8"))
        if not (final_run.get("protocol_id") == PROTOCOL_ID and final_run.get("stage") == "final"
                and final_run.get("status") == "complete" and final_run.get("official_test_targets_accessed") is False
                and final_run.get("store_manifest_sha256") == store_hash
                and final_run.get("old_checkpoint_used") is False):
            raise ValueError(f"invalid fresh official final base: {name}")
        if not args.smoke and (final_run.get("fit_subjects") != 2506 or final_run.get("fit_rows") != 2506 * 360):
            raise ValueError("final base was not refitted on all official training windows")
        checkpoint_path = final_run_path.parent / "best.pt"
        if sha256(checkpoint_path) != _digest(final_run.get("checkpoint_sha256"), "final base checkpoint"):
            raise ValueError("final base checkpoint changed")
        method_evidence = {}
        if name != "lora":
            if "method_run_path" not in candidate or "method_run_sha256" not in candidate:
                raise ValueError(f"frozen final method receipt required: {name}")
            method_path = _path(plan_path.parent, candidate["method_run_path"])
            if sha256(method_path) != candidate["method_run_sha256"]:
                raise ValueError("final method receipt changed")
            method_run = json.loads(method_path.read_text(encoding="utf-8"))
            expected_status = "synthetic_smoke_complete" if args.smoke else "complete"
            if not (method_run.get("protocol_id") == PROTOCOL_ID and method_run.get("status") == expected_status
                    and method_run.get("stage") == "official_final_frozen_prediction"
                    and method_run.get("method") == name
                    and method_run.get("official_test_targets_accessed") is False
                    and method_run.get("query_bp_is_prediction_input") is False
                    and method_run.get("source_checkpoint_sha256") == final_run["checkpoint_sha256"]
                    and method_run.get("official_contract_sha256") == store_hash
                    and method_run.get("predictions_sha256") == candidate["predictions_sha256"]
                    and method_run.get("frozen_configuration", {}).get("frozen_before_optimizer") is True):
                raise ValueError(f"method was not frozen against the final LoRA before test scoring: {name}")
            method_checkpoint = method_path.parent / "best.pt"
            if sha256(method_checkpoint) != _digest(method_run.get("checkpoint_sha256"), "method checkpoint"):
                raise ValueError("frozen final method checkpoint changed")
            method_evidence = {"method_run_sha256": sha256(method_path),
                               "method_checkpoint_sha256": method_run["checkpoint_sha256"]}
        loaded[name] = validate_predictions(pd.read_parquet(predictions_path), test)
        receipts[name] = {"predictions_sha256": sha256(predictions_path),
                          "final_base_run_sha256": sha256(final_run_path),
                          "final_base_checkpoint_sha256": final_run["checkpoint_sha256"], **method_evidence}
    return plan, test, loaded, receipts


def read_exact_targets(full_index: Path, test: pd.DataFrame):
    """Request only exact test rows/columns; reject extra, absent or changed IDs."""
    targets = pd.read_parquet(full_index, columns=TARGET_COLUMNS,
                              filters=[("segment_uid", "in", test.segment_uid.tolist())])
    if not set(TARGET_COLUMNS).issubset(targets):
        raise ValueError("label index schema mismatch")
    targets = targets[TARGET_COLUMNS].copy()
    for name in KEYS:
        targets[name] = targets[name].astype(str)
    if targets.segment_uid.duplicated().any() or len(targets) != len(test):
        raise ValueError("target predicate did not return exact official test coverage")
    if not np.isfinite(targets[["sbp", "dbp"]].to_numpy(dtype=float)).all():
        raise ValueError("official test targets are nonfinite")
    joined = test.merge(targets, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("test target identity/source mismatch")
    return joined.drop(columns="_merge")


def run(args):
    plan, test, predictions, receipts = verify_frozen_bundle(args)
    # Refuse overwrite/re-scoring into an existing result directory, before labels.
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"protocol_id": PROTOCOL_ID, "status": "verified_predictions_targets_sealed",
        "frozen_plan_sha256": args.expected_plan_sha256,
        "research_plan_sha256": plan["research_plan_sha256"], "official_contract_sha256": plan["official_contract_sha256"],
        "all_predictions_verified_before_target_read": True, "official_test_targets_accessed": False,
        "test_based_model_selection": False, "training_feedback_generated": False,
        "synthetic_smoke": args.smoke, "candidates": receipts,
        "started_utc": datetime.now(timezone.utc).isoformat()}
    contract = json.loads((args.store_root / "manifest.json").read_text(encoding="utf-8"))
    report.update(content_policy=contract.get("content_policy", "strict_content_disjoint"),
                  content_claim_limit=contract.get("content_claim_limit", "strict content checks"),
                  cross_role_exact_ppg_hashes=contract.get("cross_role_exact_ppg_hashes", 0),
                  no_test_rows_removed=True)
    save_json(args.output / "evaluation_receipt.json", report)
    try:
        # Optional torch dependency enters through existing shared metric helpers.
        from .calbased_metrics import participant_macro_views, pooled_diagnostics
        report["official_test_targets_accessed"] = True
        report["status"] = "reading_exact_official_test_targets"
        save_json(args.output / "evaluation_receipt.json", report)
        targets = read_exact_targets(args.full_index, test)
        macro_rows, diagnostic_frames = [], []
        for name, frame in predictions.items():
            joined = frame.merge(targets, on=KEYS, how="left", validate="one_to_one")
            joined = joined.rename(columns={"segment_uid": "event_id", "sbp": "target_sbp", "dbp": "target_dbp"})
            joined.to_parquet(args.output / f"{name}_scored_predictions.parquet", index=False)
            for scope, metrics in participant_macro_views(joined).items():
                macro_rows.append({"Setting": name, "Scope": scope, **metrics})
            diagnostic_frames.append(pooled_diagnostics(joined, name))
        macro = pd.DataFrame(macro_rows)
        diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
        macro.to_csv(args.output / "participant_macro.csv", index=False)
        diagnostics.to_csv(args.output / "diagnostic_tables.csv", index=False)
        for scope in ("Overall", "MIMIC", "VitalDB"):
            macro.loc[macro.Scope.eq(scope)].to_csv(args.output / f"{scope}_participant_macro.csv", index=False)
            diagnostics.loc[diagnostics.Scope.eq(scope)].to_csv(args.output / f"{scope}_diagnostics.csv", index=False)
        lines = ["# Official CalBased frozen test results", "",
            "Exact official same-subject randomized membership; 360 labelled training windows and 40 test windows per person.",
            "Participant-macro MAE is primary. Pooled AAMI/BHS fields are retrospective numerical screens, not clinical device certification.",
            "All methods and predictions were frozen before the test labels were joined. No candidate is promoted or selected by this scorer.", ""]
        if contract.get("content_policy"):
            lines += ["## Source-duplicate disclosure", "",
                      contract["content_claim_limit"],
                      "35 official test windows repeat selected training PPG content. These and all other official members are retained; no deduplicated subset is substituted.",
                      "The predeclared inner splits are unchanged; their known content duplicates are retained and internal/OOF scores must not be described as strictly content-independent.", ""]
        for scope in ("Overall", "MIMIC", "VitalDB"):
            lines += [f"## {scope}", "", "### Participant-macro MAE (mmHg)", "",
                      "| Setting | SBP MAE | DBP MAE | Mean MAE | Subjects | Windows |",
                      "|---|---:|---:|---:|---:|---:|"]
            for row in macro.loc[macro.Scope.eq(scope)].itertuples(index=False):
                lines.append(f"| {row.Setting} | {row.sbp_mae:.4f} | {row.dbp_mae:.4f} | {row.mean_mae:.4f} | {row.n_participants} | {row.n_events} |")
            table = diagnostics.loc[diagnostics.Scope.eq(scope)].drop(columns="Scope")
            lines += ["", "### Window-pooled numerical diagnostics", "", "| " + " | ".join(table.columns) + " |",
                      "| " + " | ".join(["---"] * len(table.columns)) + " |"]
            for row in table.itertuples(index=False, name=None):
                lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
            lines.append("")
        (args.output / "RESULT_TABLES.md").write_text("\n".join(lines), encoding="utf-8")
        report.update(status="complete", test_subjects=int(test.subject_uid.nunique()), test_windows=len(test),
            completed_utc=datetime.now(timezone.utc).isoformat(),
            files={p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "evaluation_receipt.json"})
        save_json(args.output / "evaluation_receipt.json", report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_json(args.output / "evaluation_receipt.json", report)
        raise
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store-root", required=True, type=Path)
    p.add_argument("--frozen-plan", required=True, type=Path)
    p.add_argument("--expected-plan-sha256", required=True)
    p.add_argument("--full-index", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    return p


def main():
    print(json.dumps(run(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
