"""Freeze both candidates before any final-query BP is joined for scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .legacy_enrollment_protocol import PROTOCOL, METHODS, partition, validation_targets
from .legacy_enrollment_train import load_population
from .official_calbased_train import sha256, save_json, event_ids_sha256
from .official_calbased_evaluate import validate_predictions, read_exact_targets
from .post_enrollment_evaluate import write_tables, write_paired_intervals
from .post_enrollment_protocol import UNCERTAINTY


def run(args):
    plan, population, _ = load_population(args.population_run, args.plan, synthetic=args.synthetic)
    _, inputs = partition(args.plan, f"{args.cohort}_inputs", access="evaluation", synthetic=args.synthetic)
    subjects = plan[f"{args.cohort}_subjects"]
    if len(args.personal_runs) != 2:
        raise ValueError("both personal shards are required")
    collected = {name: [] for name in METHODS}
    seen, receipts = set(), []
    for shard, root in enumerate(args.personal_runs):
        report = json.loads((root / "run.json").read_text())
        expected = subjects[shard::2]
        if (report.get("protocol_id") != plan["protocol_id"] or report.get("status") != "complete"
                or report.get("plan_sha256") != sha256(args.plan)
                or report.get("population_checkpoint_sha256") != population["checkpoint_sha256"]
                or report.get("evaluation_cohort") != args.cohort
                or report.get("test_targets_accessed") is not False
                or report.get("shared_frozen") is not True or report.get("frozen_predictions") is not True
                or report.get("subjects") != expected or set(report["profiles"]) != set(expected)
                or set(expected) & seen):
            raise ValueError("personal completion/provenance mismatch")
        seen.update(expected)
        for info in report["profiles"].values():
            if info.get("reload_equivalence") is not True or info.get("shared_frozen_verified") is not True:
                raise ValueError("personal state was not verified")
            for file, key in (("profile.pt", "profile_sha256"), ("memory_bank.npz", "memory_bank_sha256"),
                              ("registration_metadata.parquet", "registration_metadata_sha256")):
                if sha256(root / info["directory"] / file) != info[key]:
                    raise ValueError("personal state changed after predictions")
        for name in METHODS:
            file = root / f"{name}_predictions.parquet"
            if sha256(file) != report["prediction_files"][file.name]:
                raise ValueError("frozen predictions changed")
            collected[name].append(pd.read_parquet(file))
        receipts.append({"run_sha256": sha256(root / "run.json"), "subjects": len(expected)})
    if seen != set(subjects):
        raise ValueError("incomplete target-cohort coverage")
    predictions = {name: validate_predictions(pd.concat(rows, ignore_index=True), inputs) for name, rows in collected.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, frame in predictions.items():
        path = args.output / f"{name}_frozen_predictions.parquet"
        frame.to_parquet(path, index=False)
        files[path.name] = sha256(path)
    frozen = {"protocol_id": plan["protocol_id"], "cohort": args.cohort, "status": "frozen",
              "plan_sha256": sha256(args.plan), "checkpoint_sha256": population["checkpoint_sha256"],
              "methods": METHODS, "prediction_files": files, "personal_receipts": receipts,
              "query_ids_sha256": event_ids_sha256(inputs.segment_uid),
              "all_predictions_frozen_before_targets": True, "test_based_selection": False,
              "subjects": len(seen), "windows": len(inputs)}
    save_json(args.output / "frozen_predictions.json", frozen)
    if args.cohort == "validation":
        targets = validation_targets(args.plan)
    else:
        index = Path(plan["full_index"])
        if sha256(index) != plan["full_index_sha256"]:
            raise ValueError("reference label store changed")
        if plan["protocol_id"] == "full-cohort-enrollment-v1":
            from .full_enrollment_data import read_grouped_targets
            targets = read_grouped_targets(index, inputs)
        else:
            targets = read_exact_targets(index, inputs[["subject_uid", "segment_uid", "source"]])
    write_tables(args.output, targets, predictions,
                 {"cross_outer_subject_overlap": 0, "cross_role_content_overlap": 0},
                 title=f"{plan['protocol_id']}: {args.cohort}", protocol_id=plan["protocol_id"])
    # Existing formatter's old fixed-360 sentence is inappropriate when a
    # content group slightly changes the nominal 90/10 allocation.
    report = args.output / "RESULT_TABLES.md"
    text = report.read_text(encoding="utf-8").replace(
        "All windows and participants are retained. Each profile uses 360 labelled registration windows, not 360 independent cuff events.",
        "All frozen eligible queries and participants are retained. Registration uses approximately 90% of each person's eligible source windows; indivisible overlap groups and minimum one-window roles can change the exact fraction. Consult the cohort and budget audit for exclusions and actual counts. These are not independent cuff events.")
    report.write_text(text, encoding="utf-8")
    write_paired_intervals(args.output, targets, predictions, {"uncertainty": UNCERTAINTY})
    frozen.update(status="complete", scoring_targets_accessed=True, training_feedback=False)
    save_json(args.output / "evaluation_receipt.json", frozen)
    if args.cohort == "validation":
        save_json(args.output / "frozen_methods.json", dict(frozen, status="frozen", config=plan["config"],
                  selection="both prespecified methods retained regardless of validation ordering"))
    return frozen


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--population-run", type=Path, required=True)
    p.add_argument("--personal-runs", nargs=2, type=Path, required=True)
    p.add_argument("--cohort", choices=("validation", "test"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synthetic", action="store_true")
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
