"""One-way scoring after cohort-specific predictions are frozen and verified."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .official_calbased_train import _loader, _predict, save_json, sha256, event_ids_sha256
from .official_calbased_evaluate import validate_predictions, read_exact_targets
from .post_enrollment_protocol import PROTOCOL, METHODS, partition
from .post_enrollment_population import check_device, load_population


def write_tables(output, targets, predictions, audit, *, title):
    from .calbased_metrics import participant_macro_views, pooled_diagnostics, POOLED_COLUMNS
    macro_rows, diagnostics = [], []
    for name, frame in predictions.items():
        joined = frame.merge(targets, on=["subject_uid", "segment_uid", "source"], validate="one_to_one")
        joined = joined.rename(columns={"segment_uid": "event_id", "sbp": "target_sbp", "dbp": "target_dbp"})
        joined.to_parquet(output / f"{name}_scored_predictions.parquet", index=False)
        views = participant_macro_views(joined)
        for scope, result in views.items():
            macro_rows.append({"Setting": name, "Scope": scope, **result})
        diagnostics.append(pooled_diagnostics(joined, name))
    macro = pd.DataFrame(macro_rows)
    diagnostic = pd.concat(diagnostics, ignore_index=True)
    macro.to_csv(output / "participant_macro.csv", index=False)
    diagnostic.to_csv(output / "diagnostic_tables.csv", index=False)
    lines = [f"# {title}", "", "Protocol: post-enrollment-30-v1. Participant-macro MAE is primary.",
        "The threshold fields below are retrospective numerical AAMI/BHS screens, not device certification.",
        "All windows and participants are retained. Each profile uses 360 labelled registration windows, not 360 independent cuff events.",
        "Random-window evaluation does not establish chronological or long-term performance.", "",
        f"Source-duplicate audit: `{json.dumps(audit, sort_keys=True)}`", ""]
    for scope in ("Overall", "MIMIC", "VitalDB"):
        subset = diagnostic.loc[diagnostic.Scope.eq(scope), list(POOLED_COLUMNS)]
        macro.loc[macro.Scope.eq(scope)].to_csv(output / f"{scope}_participant_macro.csv", index=False)
        subset.to_csv(output / f"{scope}_diagnostics.csv", index=False)
        lines += [f"## {scope}", "", "| " + " | ".join(POOLED_COLUMNS) + " |",
            "| " + " | ".join(["---"] * len(POOLED_COLUMNS)) + " |"]
        for row in subset.to_dict("records"):
            rendered = [f"{row[c]:.4f}" if isinstance(row[c], (int, float)) else str(row[c]) for c in POOLED_COLUMNS]
            lines.append("| " + " | ".join(rendered) + " |")
        lines.append("")
    (output / "RESULT_TABLES.md").write_text("\n".join(lines), encoding="utf-8")


def population_benchmark(args):
    check_device(args.device, args.synthetic)
    plan, source, checkpoint = load_population(args.population_run, args.plan, device=args.device, synthetic=args.synthetic)
    _, inputs = partition(args.plan, "population_test_inputs", synthetic=args.synthetic)
    args.store_root = Path(plan["source_root"])
    args.output.mkdir(parents=True, exist_ok=False)
    from .lora_prs_models import LoraPRSRegressor, PRS_MODELS
    model = LoraPRSRegressor(PRS_MODELS["lora_continue"], subject_count=len(checkpoint["subject_to_index"])).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    anchors = pd.DataFrame.from_dict(checkpoint["subject_anchors"], orient="index")
    loader = _loader(inputs, args, checkpoint["target_scaler"], anchors, checkpoint["subject_to_index"], targets=False, shuffle=False)
    prediction = inputs[["subject_uid", "segment_uid", "source"]].copy()
    prediction[["pred_sbp", "pred_dbp"]] = _predict(model, loader, args.device, checkpoint["target_scaler"])
    file = args.output / "population_lora_predictions.parquet"
    prediction.to_parquet(file, index=False)
    prediction = validate_predictions(pd.read_parquet(file), inputs)
    frozen = {"protocol_id": PROTOCOL, "status": "frozen_predictions", "plan_sha256": sha256(args.plan),
        "checkpoint_sha256": source["checkpoint_sha256"], "predictions_sha256": sha256(file),
        "query_ids_sha256": event_ids_sha256(inputs.segment_uid), "test_targets_accessed": False,
        "training_feedback_permitted": False, "full_index_sha256": sha256(args.full_index)}
    save_json(args.output / "frozen_predictions.json", frozen)
    # No optimization follows this point in this process. A separate queued
    # personal stage consumes only the frozen checkpoint, never these scores.
    targets = read_exact_targets(args.full_index, inputs[["subject_uid", "segment_uid", "source"]])
    write_tables(args.output, targets, {"population_lora": prediction},
        {"test_rows_with_registration_content": plan["audit"]["population_test_rows_with_training_content"]},
        title="Remaining-population frozen benchmark")
    save_json(args.output / "evaluation_receipt.json", dict(frozen, status="complete", test_targets_accessed=True,
        subjects=inputs.subject_uid.nunique(), windows=len(inputs)))


def enrollment_score(args):
    plan, source, _ = load_population(args.population_run, args.plan, synthetic=args.synthetic)
    _, inputs = partition(args.plan, "enrollment_test_inputs", synthetic=args.synthetic)
    if len(args.personal_runs) != 2:
        raise ValueError("both prespecified personal shards required")
    loaded = {name: [] for name in METHODS}
    seen, evidence = set(), []
    for index, root in enumerate(args.personal_runs):
        report = json.loads((root / "run.json").read_text())
        expected_subjects = plan["selected_subjects"][index::2]
        if report.get("protocol_id") != PROTOCOL or report.get("status") != "complete" or report.get("stage") != "personal_registration" or report.get("plan_sha256") != sha256(args.plan) or report.get("population_checkpoint_sha256") != source["checkpoint_sha256"]:
            raise ValueError("invalid personal completion/checkpoint provenance")
        if report.get("test_targets_accessed") is not False or report.get("frozen_predictions") is not True or report.get("shared_frozen") is not True or report.get("other_person_adapter_copied") is not False or report.get("subjects") != expected_subjects:
            raise ValueError("invalid personal information access or shard membership")
        if set(report["profiles"]) != set(expected_subjects) or set(report["profiles"]) & seen:
            raise ValueError("missing or repeated personal profiles")
        seen.update(report["profiles"])
        for info in report["profiles"].values():
            if info.get("shared_frozen_verified") is not True or info.get("reload_equivalence") is not True:
                raise ValueError("profile did not pass frozen/reload checks")
            for name, key in (("profile.pt", "profile_sha256"), ("memory_bank.npz", "memory_bank_sha256"),
                              ("registration_metadata.parquet", "registration_metadata_sha256")):
                if sha256(root / info["directory"] / name) != info[key]:
                    raise ValueError("frozen personal profile changed")
        for name in METHODS:
            filename = f"{name}_predictions.parquet"
            if sha256(root / filename) != report["prediction_files"][filename]:
                raise ValueError("personal predictions changed")
            loaded[name].append(pd.read_parquet(root / filename))
        evidence.append({"run_sha256": sha256(root / "run.json"), "subjects": len(expected_subjects)})
    if seen != set(plan["selected_subjects"]):
        raise ValueError("not all enrollment subjects were processed")
    predictions = {name: validate_predictions(pd.concat(parts, ignore_index=True), inputs) for name, parts in loaded.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, frame in predictions.items():
        file = args.output / f"{name}_frozen_predictions.parquet"
        frame.to_parquet(file, index=False)
        files[file.name] = sha256(file)
    frozen = {"protocol_id": PROTOCOL, "status": "frozen_predictions", "plan_sha256": sha256(args.plan),
        "population_checkpoint_sha256": source["checkpoint_sha256"], "personal_runs": evidence,
        "prediction_files": files, "all_four_methods_frozen_before_targets": True,
        "query_ids_sha256": event_ids_sha256(inputs.segment_uid), "test_targets_accessed": False,
        "test_based_selection": False, "full_index_sha256": sha256(args.full_index)}
    save_json(args.output / "frozen_predictions.json", frozen)
    targets = read_exact_targets(args.full_index, inputs[["subject_uid", "segment_uid", "source"]])
    write_tables(args.output, targets, predictions,
        {"test_rows_with_registration_content": plan["audit"]["enrollment_test_rows_with_training_content"],
         "cross_population_subject_overlap": 0}, title="Thirty-subject post-training enrollment")
    save_json(args.output / "evaluation_receipt.json", dict(frozen, status="complete", test_targets_accessed=True,
        subjects=len(seen), windows=len(inputs), scope="exploratory_not_confirmatory"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("population", "enrollment"), required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--population-run", type=Path, required=True)
    p.add_argument("--personal-runs", type=Path, nargs="*", default=[])
    p.add_argument("--full-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.stage == "population":
        population_benchmark(args)
    else:
        enrollment_score(args)
    print("POST_ENROLLMENT_EVALUATION_COMPLETE=yes", flush=True)
