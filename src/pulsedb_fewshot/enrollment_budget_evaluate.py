"""Freeze all eight paired budget arms before opening any scoring targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import enrollment_budget_protocol as protocol
from .legacy_enrollment_protocol import validation_targets
from .legacy_enrollment_train import load_population
from .official_calbased_train import sha256, save_json, event_ids_sha256
from .official_calbased_evaluate import validate_predictions
from .post_enrollment_evaluate import write_tables, write_paired_intervals
from .post_enrollment_protocol import UNCERTAINTY


def budget_intervals(output, targets, all_predictions):
    """Paired, source-stratified participant bootstrap, conditional on one base.

    The primary family is the seven Overall mean-MAE differences versus 90%
    for LoRA+memory. Its max-deviation band covers the family jointly. The
    other intervals are explicitly exploratory/pointwise, not equivalence tests.
    """
    keys = protocol.KEYS
    values = {}
    for (p, method), pred in all_predictions.items():
        joined = pred.merge(targets, on=keys, validate="one_to_one")
        joined["SBP"] = (joined.pred_sbp - joined.sbp).abs()
        joined["DBP"] = (joined.pred_dbp - joined.dbp).abs()
        frame = joined.groupby(["subject_uid", "source"])[["SBP", "DBP"]].mean().sort_index()
        frame["Mean"] = frame[["SBP", "DBP"]].mean(axis=1)
        values[p, method] = frame
    rows = []
    for method in protocol.METHODS:
        for scope in ("Overall", "MIMIC", "VitalDB"):
            base = values[90, method]
            if scope != "Overall":
                base = base.loc[base.index.get_level_values("source") == scope]
            if len(base) < 2:
                raise ValueError("participant uncertainty needs at least two people per scope")
            source = base.index.get_level_values("source").to_numpy()
            indices = [np.flatnonzero(source == s) for s in sorted(set(source))]
            for bp in ("Mean", "SBP", "DBP"):
                difference = np.stack([
                    values[p, method].loc[base.index, bp].to_numpy() - base[bp].to_numpy()
                    for p in protocol.PERCENTAGES[:-1]], axis=1)
                observed = difference.mean(axis=0)
                rng = np.random.default_rng(protocol.CONFIG["bootstrap_seed"])
                sampled = []
                for _ in range(protocol.CONFIG["bootstrap_replicates"]):
                    draw = np.concatenate([rng.choice(i, size=len(i), replace=True) for i in indices])
                    sampled.append(difference[draw].mean(axis=0))
                sampled = np.asarray(sampled)
                low, high = np.quantile(sampled, [0.025, 0.975], axis=0)
                primary = method == "new_person_lora_memory" and scope == "Overall" and bp == "Mean"
                radius = float(np.quantile(np.max(np.abs(sampled - observed), axis=1), .95)) if primary else None
                for j, p in enumerate(protocol.PERCENTAGES[:-1]):
                    rows.append(dict(Setting=method, Scope=scope, BP=bp, budget_percent=p, reference_percent=90,
                        participants=len(base), MAE_increase_vs_90=float(observed[j]),
                        pointwise_ci95_low=float(low[j]), pointwise_ci95_high=float(high[j]),
                        primary_family=primary, simultaneous_ci95_low=None if radius is None else float(observed[j]-radius),
                        simultaneous_ci95_high=None if radius is None else float(observed[j]+radius)))
    pd.DataFrame(rows).to_csv(output / "budget_paired_intervals.csv", index=False)
    save_json(output / "budget_uncertainty_contract.json", dict(
        independent_unit="participant", stratification="PulseDB source", replicates=protocol.CONFIG["bootstrap_replicates"],
        seed=protocol.CONFIG["bootstrap_seed"], positive_difference="lower-budget MAE is worse than 90-percent MAE",
        primary_family="seven Overall mean-MAE differences vs 90% for LoRA+memory",
        simultaneous_method="95% bootstrap maximum absolute centered deviation across seven contrasts",
        conditional_on="this fitted shared model and one nested enrollment ordering",
        noninferiority_margin=None, no_automatic_equivalence_claim=True,
        no_test_based_budget_selection=True, no_temporal_reliability_claim=True))


def run(args):
    plan, inputs = protocol.partition(args.plan, args.cohort, synthetic=args.synthetic)
    _, population, _ = load_population(Path(plan["population_run"]), Path(plan["parent_plan"]), synthetic=args.synthetic)
    if population["checkpoint_sha256"] != plan["population_checkpoint_sha256"]:
        raise ValueError("shared checkpoint changed")
    subjects = plan[f"{args.cohort}_subjects"]
    # Collect and validate every arm before reading the target table.
    predictions, evidence = {}, []
    for p in protocol.PERCENTAGES:
        collected = {method: [] for method in protocol.METHODS}
        seen = set()
        for shard in (0, 1):
            root = args.run_root / f"{args.cohort}_p{p}_{shard}"
            report = json.loads((root / "run.json").read_text())
            expected = subjects[shard::2]
            if (report.get("status") != "complete" or report.get("protocol_id") != protocol.PROTOCOL
                    or report.get("plan_sha256") != sha256(args.plan) or report.get("budget_percent") != p
                    or report.get("population_checkpoint_sha256") != population["checkpoint_sha256"]
                    or report.get("evaluation_cohort") != args.cohort or report.get("subjects") != expected
                    or report.get("test_targets_accessed") is not False or report.get("shared_frozen") is not True
                    or report.get("frozen_predictions") is not True or report.get("other_person_adapter_copied") is not False
                    or report.get("unused_budget_labels_accessed") is not False or report.get("larger_budget_adapter_used") is not False
                    or set(report["profiles"]) != set(expected) or set(expected) & seen):
                raise ValueError("budget profile provenance/coverage failed")
            seen.update(expected)
            for info in report["profiles"].values():
                if (info.get("shared_frozen_verified") is not True or info.get("reload_equivalence") is not True
                        or info.get("budget_percent") != p):
                    raise ValueError("profile frozen/reload/budget verification failed")
                for filename, key in (("profile.pt", "profile_sha256"), ("memory_bank.npz", "memory_bank_sha256"),
                                      ("registration_metadata.parquet", "registration_metadata_sha256")):
                    if sha256(root / info["directory"] / filename) != info[key]:
                        raise ValueError("frozen profile changed")
            for method in protocol.METHODS:
                file = root / f"{method}_predictions.parquet"
                if sha256(file) != report["prediction_files"][file.name]:
                    raise ValueError("predictions changed after freezing")
                collected[method].append(pd.read_parquet(file))
            evidence.append(dict(stage=root.name, run_sha256=sha256(root / "run.json")))
        if seen != set(subjects):
            raise ValueError("all target people must be retained at every budget")
        for method, parts in collected.items():
            predictions[p, method] = validate_predictions(pd.concat(parts, ignore_index=True), inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for (p, method), frame in predictions.items():
        path = args.output / f"p{p}_{method}_frozen_predictions.parquet"
        frame.to_parquet(path, index=False)
        files[path.name] = sha256(path)
    frozen = dict(protocol_id=protocol.PROTOCOL, cohort=args.cohort, status="frozen",
        plan_sha256=sha256(args.plan), checkpoint_sha256=population["checkpoint_sha256"],
        methods=protocol.METHODS, percentages=protocol.PERCENTAGES, config=protocol.CONFIG,
        subjects=len(subjects), query_windows=len(inputs), query_ids_sha256=event_ids_sha256(inputs.segment_uid),
        all_sixteen_predictions_frozen_before_targets=True, prediction_files=files,
        personal_receipts=evidence, test_based_selection=False, training_feedback=False)
    save_json(args.output / "frozen_predictions.json", frozen)
    # This is the sole transition to query-BP access. Training never imports it.
    if args.cohort == "validation":
        targets = validation_targets(Path(plan["parent_plan"]))
    else:
        from .full_enrollment_data import read_grouped_targets
        parent = protocol.load_parent(Path(plan["parent_plan"]), synthetic=args.synthetic)
        if sha256(Path(parent["full_index"])) != parent["full_index_sha256"]:
            raise ValueError("query reference store changed")
        targets = read_grouped_targets(Path(parent["full_index"]), inputs)
    macros, diagnostics = [], []
    for p in protocol.PERCENTAGES:
        directory = args.output / f"p{p}"
        directory.mkdir()
        selected = {name: predictions[p, name] for name in protocol.METHODS}
        write_tables(directory, targets, selected,
            {"outer_subject_overlap": 0, "registration_query_content_overlap": 0},
            title=f"Personal history budget {p}%: {args.cohort}", protocol_id=protocol.PROTOCOL)
        path = directory / "RESULT_TABLES.md"
        text = path.read_text(encoding="utf-8").replace(
            "All windows and participants are retained. Each profile uses 360 labelled registration windows, not 360 independent cuff events.",
            f"All parent eligible participants and identical queries are retained at nominal {p}% enrollment. Actual counts/fractions and one-group fallbacks are in the budget audit. These labeled 10-s windows are not independent cuff measurements.")
        path.write_text(text, encoding="utf-8")
        write_paired_intervals(directory, targets, selected, {"uncertainty": UNCERTAINTY})
        macros.append(pd.read_csv(directory / "participant_macro.csv").assign(budget_percent=p))
        diagnostics.append(pd.read_csv(directory / "diagnostic_tables.csv").assign(budget_percent=p))
    macro, diagnostic = pd.concat(macros, ignore_index=True), pd.concat(diagnostics, ignore_index=True)
    macro.to_csv(args.output / "all_budgets_participant_macro.csv", index=False)
    diagnostic.to_csv(args.output / "all_budgets_diagnostics.csv", index=False)
    for scope in ("Overall", "MIMIC", "VitalDB"):
        macro.loc[macro.Scope.eq(scope)].to_csv(args.output / f"{scope}_all_budgets_participant_macro.csv", index=False)
        diagnostic.loc[diagnostic.Scope.eq(scope)].to_csv(args.output / f"{scope}_all_budgets_diagnostics.csv", index=False)
    budget_intervals(args.output, targets, predictions)
    save_json(args.output / "evaluation_receipt.json", dict(frozen, status="complete", scoring_targets_accessed=True))
    if args.cohort == "validation":
        save_json(args.output / "frozen_methods.json", dict(frozen,
                  selection="all eight budgets and both methods retained without performance-based changes"))
    return frozen


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--cohort", choices=("validation", "test"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synthetic", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
