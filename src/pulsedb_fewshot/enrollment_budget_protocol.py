"""Nested, label-blind personal budgets with the parent's exact frozen queries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .legacy_enrollment_protocol import (
    FULL_PROTOCOL, FULL_CONFIG, METHODS, KEYS, load_plan as load_parent,
    partition as parent_partition, reject_labels, row_components,
)
from .legacy_enrollment_train import load_population
from .official_calbased_train import save_json, sha256, event_ids_sha256

PROTOCOL = "enrollment-budget-v1"
PERCENTAGES = list(range(20, 100, 10))
SEED = 20260911
CONFIG = dict(FULL_CONFIG, budget_percentages=PERCENTAGES, budget_assignment_seed=SEED,
              registration_fraction="budget_percent_divided_by_100_nominal",
              old_checkpoint_permitted="verified_subject_disjoint_parent_only",
              budget_denominator="eligible_registration_plus_fixed_query_windows",
              query_policy="exact_parent_query_ids_unchanged",
              selection_policy="seeded_label_blind_nested_complete_group_prefix",
              target_count="max_1_floor_percent_times_eligible_count_capped_by_parent_bank",
              upper_budget_policy="90_percent_exact_parent_registration_and_inner_roles",
              single_group_fallback="zero_adapter", personal_warm_start=False,
              shared_checkpoint_policy="reuse_verified_subject_disjoint_parent_only",
              reporting="all_eight_budgets_no_test_selected_winner",
              bootstrap_replicates=2000, bootstrap_seed=20260911)


def person_budgets(metadata, query_count):
    """Return only identities/inner roles; neither BP nor error may select rows.

    Percentages use all eligible personal windows as denominator, not 90% of
    the old bank. Whole groups are indivisible; actual counts are reported.
    """
    reject_labels(metadata)
    if any("error" in str(c).lower() or "residual" in str(c).lower()
           or str(c).lower().startswith(("pred_", "prediction_")) for c in metadata):
        raise ValueError("predictions/errors must not be available to budget assignment")
    m = metadata.sort_values("segment_uid").reset_index(drop=True)
    if m.empty or m.subject_uid.nunique() != 1 or query_count < 1:
        raise ValueError("one nonempty person and fixed queries required")
    subject = str(m.subject_uid.iloc[0])
    groups = row_components(m)
    seed = SEED + int(hashlib.sha256(subject.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    groups = [groups[int(i)] for i in rng.permutation(len(groups))]
    total = len(m) + int(query_count)
    frames, audits = {}, []
    previous = set()
    for percent in PERCENTAGES:
        target = min(len(m), max(1, total * percent // 100))
        if percent == 90:
            selected = m.copy()  # Exact parent roles, including low-count cases.
            selected_groups = groups
        else:
            selected_groups, count = [], 0
            for group in groups:
                selected_groups.append(group)
                count += len(group)
                if count >= target:
                    break
            indices = [i for group in selected_groups for i in group]
            selected = m.loc[indices].copy()
            selected["inner_role"] = "train"
            if len(selected_groups) >= 2:
                # Independent validation lies wholly inside this budget.
                inner_target = max(1, len(selected) // 9)
                inner, count = [], 0
                for group in selected_groups[:-1]:
                    inner.extend(group)
                    count += len(group)
                    if count >= inner_target:
                        break
                selected.loc[inner, "inner_role"] = "internal_validation"
        ids = set(selected.segment_uid)
        if not previous.issubset(ids):
            raise ValueError("budget subsets must be nested")
        previous = ids
        frames[percent] = selected[KEYS + ["inner_role"]].copy()
        audits.append(dict(subject_uid=subject, source=str(m.source.iloc[0]),
            budget_percent=percent, eligible_windows=total, parent_registration_windows=len(m),
            query_windows=int(query_count), requested_windows=target, registration_windows=len(selected),
            actual_fraction=len(selected) / total, unused_windows=len(m)-len(selected),
            registration_groups=len(selected_groups),
            inner_fit_windows=int(selected.inner_role.eq("train").sum()),
            inner_selection_windows=int(selected.inner_role.eq("internal_validation").sum()),
            zero_adapter_fallback=not selected.inner_role.eq("internal_validation").any(),
            registration_ids_sha256=event_ids_sha256(selected.segment_uid)))
    return frames, audits


def frame_audit(frame):
    return dict(rows=len(frame), subjects=int(frame.subject_uid.nunique()),
                segment_ids_sha256=event_ids_sha256(frame.segment_uid),
                subject_ids_sha256=event_ids_sha256(frame.subject_uid.unique()))


def prepare(args):
    parent, population, _ = load_population(args.population_run, args.parent_plan, synthetic=args.synthetic)
    if parent["protocol_id"] != FULL_PROTOCOL:
        raise ValueError("the full-cohort subject-disjoint parent is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit, budgets = {}, []
    for cohort in ("validation", "test"):
        access = f"personal_{cohort}"
        _, bank = parent_partition(args.parent_plan, f"{cohort}_registration", access=access, synthetic=args.synthetic)
        _, query = parent_partition(args.parent_plan, f"{cohort}_inputs", access=access, synthetic=args.synthetic)
        query.to_parquet(output / f"{cohort}_inputs.parquet", index=False)
        audit[f"{cohort}_inputs"] = frame_audit(query)
        if audit[f"{cohort}_inputs"] != parent["audit"][f"{cohort}_inputs"]:
            raise ValueError("parent query membership changed")
        assigned = {p: [] for p in PERCENTAGES}
        query_counts = query.groupby("subject_uid").size()
        # Only metadata enters the actual assignment function. Registration BP
        # is joined afterwards; no query label store is opened by preparation.
        for subject, frame in bank.drop(columns=["sbp", "dbp"]).groupby("subject_uid", sort=True):
            selections, rows = person_budgets(frame, int(query_counts.loc[subject]))
            for p, value in selections.items():
                assigned[p].append(value)
            budgets.extend(dict(row, cohort=cohort) for row in rows)
        for p in PERCENTAGES:
            name = f"{cohort}_p{p}_registration"
            keys = pd.concat(assigned[p], ignore_index=True)
            # Preserve parent row order, including exact 90% parity.
            part = bank.drop(columns="inner_role").merge(keys, on=KEYS, how="inner", sort=False, validate="one_to_one")
            # The parent helper's redundant role tag must describe these new
            # inner roles, not retain the old 90%-budget selection tags.
            if "access_role" in part:
                part["access_role"] = part.split + ":registration:" + part.inner_role
            if p == 90:
                pd.testing.assert_frame_equal(part[bank.columns].reset_index(drop=True), bank.reset_index(drop=True))
            if set(part.subject_uid) != set(query.subject_uid) or set(part.segment_uid) & set(query.segment_uid):
                raise ValueError("people lost or query identities entered enrollment")
            part.to_parquet(output / f"{name}.parquet", index=False)
            audit[name] = frame_audit(part)
    budget_table = pd.DataFrame(budgets)
    budget_table.to_parquet(output / "personal_budgets.parquet", index=False)
    aggregate = budget_table.groupby(["cohort", "source", "budget_percent"]).agg(
        subjects=("subject_uid", "size"), registration_windows=("registration_windows", "sum"),
        query_windows=("query_windows", "sum"), minimum_windows=("registration_windows", "min"),
        median_windows=("registration_windows", "median"), maximum_windows=("registration_windows", "max"),
        mean_actual_fraction=("actual_fraction", "mean"), zero_adapter_fallbacks=("zero_adapter_fallback", "sum"))
    aggregate.to_csv(output / "budget_summary.csv")
    files = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    plan = dict(protocol_id=PROTOCOL, status="ready", synthetic=args.synthetic, config=CONFIG, methods=METHODS,
        parent_plan=str(args.parent_plan.resolve()), parent_plan_sha256=sha256(args.parent_plan),
        population_run=str(args.population_run.resolve()), population_checkpoint_sha256=population["checkpoint_sha256"],
        source_root=parent["source_root"], validation_subjects=parent["validation_subjects"],
        test_subjects=parent["test_subjects"], files=files, audit=audit,
        query_labels_read_during_prepare=False, budget_assignment_used_labels=False,
        original_outer_split_sha256=parent["legacy_split_sha256"], shared_population_retrained=False,
        claims="Exploratory labeled-history dose response; not K-shot, temporal/online reliability or independent confirmation.")
    save_json(output / "plan.json", plan)
    save_json(output / "plan_digest.json", {"sha256": sha256(output / "plan.json")})
    print(aggregate.to_string(), flush=True)
    return plan


def load_plan(path, *, synthetic=False):
    path = Path(path)
    if sha256(path) != json.loads((path.parent / "plan_digest.json").read_text())["sha256"]:
        raise ValueError("budget plan changed")
    plan = json.loads(path.read_text())
    if (plan.get("protocol_id") != PROTOCOL or plan.get("status") != "ready" or plan.get("config") != CONFIG
            or plan.get("methods") != METHODS or bool(plan.get("synthetic")) != synthetic
            or plan.get("budget_assignment_used_labels") is not False
            or plan.get("query_labels_read_during_prepare") is not False):
        raise ValueError("unapproved budget protocol")
    if sha256(Path(plan["parent_plan"])) != plan["parent_plan_sha256"]:
        raise ValueError("parent contract changed")
    parent = load_parent(Path(plan["parent_plan"]), synthetic=synthetic)
    if parent["protocol_id"] != FULL_PROTOCOL or plan["original_outer_split_sha256"] != parent["legacy_split_sha256"]:
        raise ValueError("wrong population split")
    for cohort in ("validation", "test"):
        if (plan[f"{cohort}_subjects"] != parent[f"{cohort}_subjects"]
                or plan["audit"][f"{cohort}_inputs"] != parent["audit"][f"{cohort}_inputs"]):
            raise ValueError("target cohort changed")
    return plan


def partition(path, cohort, percent=None, *, synthetic=False):
    if cohort not in ("validation", "test") or (percent is not None and percent not in PERCENTAGES):
        raise ValueError("invalid budget/cohort")
    plan = load_plan(path, synthetic=synthetic)
    name = f"{cohort}_inputs" if percent is None else f"{cohort}_p{percent}_registration"
    file = Path(path).parent / f"{name}.parquet"
    if sha256(file) != plan["files"][file.name]:
        raise ValueError("budget partition changed")
    frame = pd.read_parquet(file)
    if frame_audit(frame) != plan["audit"][name]:
        raise ValueError("budget membership changed")
    if percent is None:
        reject_labels(frame)
    elif not np.isfinite(frame[["sbp", "dbp"]].to_numpy()).all():
        raise ValueError("invalid enrollment BP")
    return plan, frame


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-plan", type=Path, required=True)
    p.add_argument("--population-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synthetic", action="store_true")
    return p


if __name__ == "__main__":
    prepare(parser().parse_args())
