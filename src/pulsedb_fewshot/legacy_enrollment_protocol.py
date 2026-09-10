"""Old outer participant assignments with large-budget personal enrollment.

The current audited 400-window pool is intersected with the original saved
70/15/15 participant assignments. This is NOT the old event120 dataset, exact
official CalBased, or a K-shot study. Original source files are never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .official_calbased_train import sha256, save_json, event_ids_sha256, REQUIRED
from .official_content_policy import POLICY_COLUMNS, assert_allowed_duplicates
from .official_time_policy import sample_span_audit, TIME_BOUNDARY_POLICY
from .post_enrollment_protocol import FORBIDDEN

PROTOCOL = "legacy-split-enrollment-v1"
FULL_PROTOCOL = "full-cohort-enrollment-v1"
LEGACY_SPLIT_SHA256 = "8705f7cd75d92201bd203c00fb4d8ad8c738c02d7c4b56e5747210ffba504cd7"
SEED = 20260910
METHODS = ["new_person_lora", "new_person_lora_memory"]
KEYS = ["subject_uid", "segment_uid", "source"]
CONFIG = {
    "methods": METHODS, "seed": SEED, "within_person_split": "random_content_grouped",
    "registration_fraction": .9, "personal_inner_fit_fraction": 8 / 9,
    "personal_seed_base": 20260909,
    "minimum_query_windows": 5, "minimum_inner_validation_windows": 5,
    "backbone": "resnet_small", "feature_dim": 256, "adapter_rank": 4,
    "personal_parameters": 2048, "learning_rate": 3e-4, "weight_decay": 1e-4,
    "huber_delta": .5, "gradient_clip": 5., "patience": 8, "batch_size": 64,
    "memory_k": 5, "memory_temperature": .1, "memory_block_size": 40,
    "old_checkpoint_permitted": False, "test_feedback_permitted": False,
    "population_selection": "meta_validation_query_macro_zero_adapter_registration_anchor",
    "personal_selection": "own_registration_inner_validation_then_fresh_full_registration_refit",
    "cross_outer_duplicate_policy": "quarantine_all_members_of_cross_role_identity_components",
    "duplicate_allocation": "same_content_and_overlapping_intervals_stay_in_one_personal_role",
}
FULL_CONFIG = dict(CONFIG, cohort_scope="all_original_subjects_all_valid_windows",
                   minimum_query_windows=1, minimum_inner_validation_windows=1,
                   minimum_inner_fit_windows=1, memory_query_chunk_size=128,
                   exact_within_outer_duplicates="retain_one_canonical_window")
PARTITIONS = ("population_train", "validation_registration", "validation_inputs",
              "test_registration", "test_inputs")
ACCESS = {
    "population": {"population_train", "validation_registration", "validation_inputs"},
    "personal_validation": {"validation_registration", "validation_inputs"},
    "personal_test": {"test_registration", "test_inputs"},
    "evaluation": set(PARTITIONS) - {"population_train"},
}


def reject_labels(frame):
    if FORBIDDEN & set(frame) or any(str(c).lower().startswith(("target_", "abp", "segsbp", "segdbp")) for c in frame):
        raise ValueError("label-free metadata contains forbidden reference columns")


def identity_components(frame):
    """Union identities sharing exact PPG or positive same-record overlap."""
    ids = sorted(frame.subject_uid.unique())
    parent = {s: s for s in ids}
    def root(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    pairs = frame[["ppg_content_sha256", "subject_uid"]].drop_duplicates()
    dup = pairs.loc[pairs.ppg_content_sha256.duplicated(keep=False)]
    for _, g in dup.groupby("ppg_content_sha256", sort=False):
        people = g.subject_uid.tolist()
        for s in people[1:]:
            union(people[0], s)
    for _, g in frame.groupby(["source", "record_id"], sort=False):
        if g.subject_uid.nunique() < 2:
            continue
        active = []
        for start, end, person in g.sort_values("start_time_s")[["start_time_s", "end_time_s", "subject_uid"]].itertuples(index=False, name=None):
            active = [(e, s) for e, s in active if e > start + 1e-7]
            for _, other in active:
                union(person, other)
            active.append((end, person))
    return {s: root(s) for s in ids}


def row_components(person):
    """Indivisible content/interval groups, without BP or original role inputs."""
    person = person.reset_index(drop=True)
    parent = list(range(len(person)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)
    duplicates = person.loc[person.ppg_content_sha256.duplicated(keep=False)]
    for _, g in duplicates.groupby("ppg_content_sha256", sort=False):
        indices = g.index.tolist()
        for i in indices[1:]:
            union(indices[0], i)
    for _, g in person.groupby(["source", "record_id"], sort=False):
        active = []
        for i, start, end in g.sort_values("start_time_s")[["start_time_s", "end_time_s"]].itertuples():
            active = [(j, e) for j, e in active if e > start + 1e-7]
            for j, _ in active:
                union(i, j)
            active.append((i, end))
    groups = {}
    for i in range(len(person)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def assign_person(person, *, full_cohort=False):
    person = person.sort_values("segment_uid").reset_index(drop=True).copy()
    groups = row_components(person)
    seed = SEED + int(hashlib.sha256(person.subject_uid.iloc[0].encode()).hexdigest()[:6], 16)
    rng = np.random.default_rng(seed)
    groups = [groups[i] for i in rng.permutation(len(groups))]
    minimum = 1 if full_cohort else 5
    if full_cohort and len(groups) < 3:
        raise ValueError("insufficient independent groups for registration and query")
    query_target = max(minimum, int(round(len(person) * .1)))
    query_groups, remaining, n = [], [], 0
    for number, group in enumerate(groups):
        if n < query_target and (not full_cohort or number < len(groups) - 2):
            query_groups.append(group)
            n += len(group)
        else:
            remaining.append(group)
    inner_target = max(minimum, int(round((len(person) - n) / 9)))
    inner_groups, fit_groups, m = [], [], 0
    for number, group in enumerate(remaining):
        if m < inner_target and (not full_cohort or number < len(remaining) - 1):
            inner_groups.append(group)
            m += len(group)
        else:
            fit_groups.append(group)
    if sum(map(len, fit_groups)) < minimum or not query_groups or not inner_groups:
        raise ValueError("insufficient independent groups for registration and query")
    person["personal_role"] = "registration"
    person["inner_role"] = "train"
    person.loc[[i for g in inner_groups for i in g], "inner_role"] = "internal_validation"
    q = [i for g in query_groups for i in g]
    person.loc[q, "personal_role"] = "query"
    person.loc[q, "inner_role"] = "query"
    return person


def build_assignment(metadata, legacy, *, full_cohort=False):
    reject_labels(metadata)
    if metadata.segment_uid.duplicated().any() or legacy.subject_uid.duplicated().any():
        raise ValueError("duplicate source row or outer participant assignment")
    if set(legacy.split) != {"meta_train", "meta_validation", "meta_test"}:
        raise ValueError("incorrect original outer roles")
    f = metadata.merge(legacy[["subject_uid", "source", "split"]], on=["subject_uid", "source"],
                       how="left", validate="many_to_one")
    if f.split.isna().any():
        raise ValueError("current participant absent from saved original split")
    component = identity_components(f)
    people = f[["subject_uid", "source", "split"]].drop_duplicates().copy()
    people["component"] = people.subject_uid.map(component)
    cross = people.groupby("component").split.nunique()
    quarantine = sorted(people.loc[people.component.isin(cross[cross.gt(1)].index), "subject_uid"])
    active = f.loc[~f.subject_uid.isin(quarantine)].copy()
    assignments, excluded = [], []
    collapsed = 0
    if full_cohort:
        before = set(active.subject_uid)
        n_before = len(active)
        active = active.sort_values(["subject_uid", "segment_uid"]).drop_duplicates("ppg_content_sha256")
        collapsed = n_before - len(active)
        excluded.extend({"subject_uid": s, "reason": "no_unique_content_after_same_outer_duplicate_collapse"}
                        for s in sorted(before - set(active.subject_uid)))
    for subject, g in active.groupby("subject_uid", sort=True):
        if g.split.iloc[0] == "meta_train":
            assignments.append(g.assign(personal_role="population_fit", inner_role="train"))
        else:
            try:
                assignments.append(assign_person(g, full_cohort=full_cohort))
            except ValueError as exc:
                if "insufficient independent groups" not in str(exc):
                    raise
                excluded.append({"subject_uid": subject, "reason": str(exc)})
    f = pd.concat(assignments, ignore_index=True)
    # Global roles, not subject-local checks: content relabelling must not hide reuse.
    f["access_role"] = f.split + ":" + f.personal_role + ":" + f.inner_role
    if f.groupby("ppg_content_sha256").access_role.nunique().gt(1).any():
        raise ValueError("exact PPG content crosses a fitting/query boundary")
    interval = sample_span_audit(f, "access_role")
    if interval["cross_role_overlap_pairs"]:
        raise ValueError("positive physiological overlap crosses roles")
    audit = {"quarantined_subjects": quarantine, "quarantine_rows": int(metadata.subject_uid.isin(quarantine).sum()),
             "same_outer_duplicate_rows_collapsed": collapsed,
             "insufficient_history_exclusions": excluded, "interval_audit": interval,
             "cross_outer_subject_overlap": 0, "cross_role_content_overlap": 0,
             "original_assignment_changed": False, "assignment_uses_bp_or_errors": False,
             "source_pool_subjects": len(people), "source_pool_rows": len(metadata),
             "eligible_source_subjects": f[["subject_uid", "source", "split"]].drop_duplicates().groupby(["split", "source"]).size().unstack(fill_value=0).to_dict("index")}
    return f, audit


def prepare(args):
    store, output = args.store_root.resolve(), args.output.resolve()
    if not args.synthetic and sha256(args.legacy_split) != LEGACY_SPLIT_SHA256:
        raise ValueError("original subject split checksum changed")
    contract = json.loads((store / "manifest.json").read_text())
    if contract.get("protocol_id") != "pulsedb-official-calbased-v1" or contract.get("status") != "ready" or bool(contract.get("synthetic", False)) != args.synthetic:
        raise ValueError("unapproved source waveform pool")
    train_path, test_path = store / "train_manifest.parquet", store / "test_inputs.parquet"
    if not args.synthetic and sha256(train_path) != contract.get("train_manifest_sha256"):
        raise ValueError("source manifest changed")
    import pyarrow.parquet as pq
    input_columns = sorted((REQUIRED - {"sbp", "dbp", "inner_role"}) | set(POLICY_COLUMNS))
    frames = []
    for path in (train_path, test_path):
        columns = set(pq.ParquetFile(path).schema.names)
        wanted = [c for c in input_columns if c in columns]
        if not (REQUIRED - {"sbp", "dbp", "inner_role"}) <= set(wanted):
            raise ValueError("source input lineage missing")
        frames.append(pd.read_parquet(path, columns=wanted))
    metadata = pd.concat(frames, ignore_index=True)
    if not args.synthetic:
        if metadata.subject_uid.nunique() != 2506 or not metadata.groupby("subject_uid").size().eq(400).all():
            raise ValueError("source pool is not the audited 2506 x 400 windows")
        assert_allowed_duplicates(metadata)
        # Verify source waveforms once, before either fitting job. Individual reads
        # also validate the actual float32 waveform against its row checksum.
        for name in sorted(metadata.waveform_file.unique()):
            file = (store / name).resolve()
            if not file.is_relative_to(store) or not file.is_file():
                raise ValueError("waveform shard missing or outside source pool")
            expected = {x["file"]: x["sha256"] for x in contract["arrays"]}
            if name not in expected or sha256(file) != expected[name]:
                raise ValueError("source waveform shard changed")
    legacy = pd.read_csv(args.legacy_split)
    f, audit = build_assignment(metadata, legacy)
    output.mkdir(parents=True, exist_ok=False)
    f.to_parquet(output / "assignment_inputs.parquet", index=False)
    people = f[["subject_uid", "source", "split"]].drop_duplicates()
    people.to_csv(output / "subject_assignments.csv", index=False)
    parts = {"population_train": f.loc[f.split.eq("meta_train")].copy()}
    for cohort, role in (("validation", "meta_validation"), ("test", "meta_test")):
        parts[f"{cohort}_registration"] = f.loc[f.split.eq(role) & f.personal_role.eq("registration")].copy()
        parts[f"{cohort}_inputs"] = f.loc[f.split.eq(role) & f.personal_role.eq("query")].copy()
    # The final-query labels are deliberately not read/materialized by preparation.
    from .official_calbased_evaluate import read_exact_targets
    for name in PARTITIONS:
        part = parts[name].reset_index(drop=True)
        if not name.endswith("inputs"):
            labels = read_exact_targets(args.full_index, part[KEYS])
            part = part.merge(labels, on=KEYS, validate="one_to_one")
        else:
            reject_labels(part)
        part.to_parquet(output / f"{name}.parquet", index=False)
        audit[name] = {"rows": len(part), "subjects": part.subject_uid.nunique(),
                       "segment_ids_sha256": event_ids_sha256(part.segment_uid),
                       "subject_ids_sha256": event_ids_sha256(part.subject_uid.unique())}
    # Validation labels may select epochs; test query targets remain source-only.
    read_exact_targets(args.full_index, parts["validation_inputs"][KEYS]).to_parquet(output / "validation_targets.parquet", index=False)
    files = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    plan = {"protocol_id": PROTOCOL, "status": "ready", "synthetic": args.synthetic,
            "config": CONFIG, "source_root": str(store), "full_index": str(args.full_index.resolve()),
            "full_index_sha256": sha256(args.full_index), "source_manifest_sha256": sha256(store / "manifest.json"),
            "source_train_sha256": sha256(train_path), "source_test_inputs_sha256": sha256(test_path),
            "legacy_split_path": str(args.legacy_split.resolve()), "legacy_split_sha256": sha256(args.legacy_split),
            "files": files, "audit": audit, "methods": METHODS,
            "validation_subjects": sorted(parts["validation_inputs"].subject_uid.unique()),
            "test_subjects": sorted(parts["test_inputs"].subject_uid.unique()),
            "test_query_labels_read_during_prepare": False,
            "claims": "Exploratory new-user enrollment using the original outer assignment on the current official window pool; not exact CalBased/CalFree, event120, K-shot, future-date, or pristine independent confirmation."}
    save_json(output / "plan.json", plan)
    save_json(output / "plan_digest.json", {"sha256": sha256(output / "plan.json")})
    return plan


def load_plan(path, *, synthetic=False):
    path = Path(path).resolve()
    if sha256(path) != json.loads((path.parent / "plan_digest.json").read_text())["sha256"]:
        raise ValueError("frozen plan checksum changed")
    plan = json.loads(path.read_text())
    expected_config = {PROTOCOL: CONFIG, FULL_PROTOCOL: FULL_CONFIG}.get(plan.get("protocol_id"))
    if expected_config is None or plan.get("status") != "ready" or bool(plan.get("synthetic")) != synthetic or plan.get("config") != expected_config or plan.get("methods") != METHODS:
        raise ValueError("unapproved legacy enrollment contract")
    if not synthetic and plan["legacy_split_sha256"] != LEGACY_SPLIT_SHA256:
        raise ValueError("wrong original participant split")
    if sha256(Path(plan["legacy_split_path"])) != plan["legacy_split_sha256"] or sha256(Path(plan["source_root"]) / "manifest.json") != plan["source_manifest_sha256"]:
        raise ValueError("source provenance changed")
    if set(plan["validation_subjects"]) & set(plan["test_subjects"]):
        raise ValueError("validation/test subject overlap")
    if plan["protocol_id"] == FULL_PROTOCOL:
        if plan.get("all_original_subjects_accounted_for") is not True:
            raise ValueError("full cohort accounting missing")
        if not synthetic and plan.get("original_subjects") != 5361:
            raise ValueError("full cohort was restricted")
    return plan


def partition(path, name, *, access, synthetic=False):
    if name not in ACCESS.get(access, set()):
        raise ValueError("partition access forbidden for this stage")
    path = Path(path).resolve()
    plan = load_plan(path, synthetic=synthetic)
    file = path.parent / f"{name}.parquet"
    if sha256(file) != plan["files"][file.name]:
        raise ValueError("frozen partition checksum changed")
    frame = pd.read_parquet(file)
    expected = plan["audit"][name]
    if len(frame) != expected["rows"] or event_ids_sha256(frame.segment_uid) != expected["segment_ids_sha256"] or event_ids_sha256(frame.subject_uid.unique()) != expected["subject_ids_sha256"]:
        raise ValueError("partition membership changed")
    role = "meta_train" if name == "population_train" else "meta_validation" if name.startswith("validation") else "meta_test"
    if not frame.split.eq(role).all():
        raise ValueError("old outer participant role changed")
    if name.endswith("inputs"):
        reject_labels(frame)
    else:
        if not np.isfinite(frame[["sbp", "dbp"]].to_numpy()).all():
            raise ValueError("invalid permitted registration labels")
    return plan, frame


def validation_targets(path):
    path = Path(path).resolve()
    plan = load_plan(path, synthetic=json.loads(path.read_text()).get("synthetic", False))
    file = path.parent / "validation_targets.parquet"
    if sha256(file) != plan["files"][file.name]:
        raise ValueError("validation target checksum changed")
    return pd.read_parquet(file)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store-root", type=Path, required=True)
    p.add_argument("--legacy-split", type=Path, required=True)
    p.add_argument("--full-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synthetic", action="store_true")
    return p


if __name__ == "__main__":
    print(json.dumps(prepare(parser().parse_args())["audit"], indent=2))
