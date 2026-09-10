"""Fresh personal fitting for one predeclared history budget; no query BP access."""
from __future__ import annotations

import json
from pathlib import Path

from . import enrollment_budget_protocol as protocol
from .legacy_enrollment_train import parser as parent_parser, check_arguments, load_population
from .post_enrollment_personal import run_profiles
from .official_calbased_train import save_json, sha256


def run(args):
    check_arguments(args)
    plan = protocol.load_plan(args.plan, synthetic=args.synthetic)
    if args.population_run.resolve() != Path(plan["population_run"]).resolve():
        raise ValueError("unapproved population model")
    _, population, checkpoint = load_population(args.population_run, Path(plan["parent_plan"]),
                                                device=args.device, synthetic=args.synthetic)
    if population["checkpoint_sha256"] != plan["population_checkpoint_sha256"]:
        raise ValueError("wrong shared checkpoint")
    if args.cohort == "test":
        frozen = json.loads((args.validation_run / "frozen_methods.json").read_text())
        if (frozen.get("plan_sha256") != sha256(args.plan) or frozen.get("status") != "frozen"
                or frozen.get("methods") != protocol.METHODS or frozen.get("percentages") != protocol.PERCENTAGES
                or frozen.get("config") != protocol.CONFIG
                or frozen.get("checkpoint_sha256") != population["checkpoint_sha256"]):
            raise ValueError("all-budget validation freeze must precede final-cohort fitting")
    _, bank = protocol.partition(args.plan, args.cohort, args.percent, synthetic=args.synthetic)
    _, query = protocol.partition(args.plan, args.cohort, synthetic=args.synthetic)
    context = dict(plan, selected_subjects=plan[f"{args.cohort}_subjects"], budget_percent=args.percent)
    report = run_profiles(args, context, population, checkpoint, bank, query)
    report.update(evaluation_cohort=args.cohort, budget_percent=args.percent,
                  unused_budget_labels_accessed=False, larger_budget_adapter_used=False)
    save_json(args.output / "run.json", report)
    return report


def parser():
    p = parent_parser()
    p.add_argument("--percent", type=int, choices=protocol.PERCENTAGES, required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
