"""Subject-excluded post-training enrollment; not an exact official benchmark.

The original official 360/40 membership is retained within each person. Whole
subjects are sampled using identity/source metadata only, never BP/errors.
The original store remains unchanged. Fitting reads explicit partition files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .official_calbased_train import (PROTOCOL_ID as SOURCE_PROTOCOL,
    REQUIRED, event_ids_sha256, save_json, sha256, validate_training_frame,
    load_test_inputs, assert_cross_role_intervals)
from .official_time_policy import TIME_BOUNDARY_POLICY
from .official_content_policy import POLICY, AUDIT_SHA256, approved_groups

PROTOCOL = "post-enrollment-30-v1"
SEED = 20260909
PARTITIONS = ("population_train", "population_test_inputs",
              "enrollment_train", "enrollment_test_inputs")
METHODS = ("personal_mean", "shared_with_anchor", "new_person_lora",
           "new_person_lora_memory")
EXPANDED_PROTOCOL = "post-enrollment-200-v1"
EXPANDED_METHODS = METHODS + (
    "shared_memory_only", "shared_memory_blend", "lora_memory_only",
    "lora_memory_uniform", "lora_memory_fixed_half")
PROTOCOLS = {
    PROTOCOL: {"per_source": 15, "synthetic_per_source": 1, "methods": METHODS},
    EXPANDED_PROTOCOL: {"per_source": 100, "synthetic_per_source": 2, "methods": EXPANDED_METHODS},
}
ABLATION_CONTRACT = {
    "shared_memory_only": "zero_adapter_features; donor_BP_only; personal_mean_if_no_legal_donors",
    "shared_memory_blend": "zero_adapter_features; same_distance_gate_and_shared_head",
    "lora_memory_only": "adapted_features; donor_BP_only; personal_mean_if_no_legal_donors",
    "lora_memory_uniform": "same_adapted_top5_donors_and_distance_gate; equal_donor_weights",
    "lora_memory_fixed_half": "same_adapted_top5_and_similarity_weights; fixed_alpha_0.5_if_valid",
}
UNCERTAINTY = {"unit": "participant", "paired": True,
    "bootstrap_replicates": 2000, "seed": SEED, "confidence": .95,
    "source_stratified": True, "primary": "new_person_lora_memory_vs_new_person_lora",
    "scope": "conditional_on_frozen_model; exploratory_pointwise_intervals_not_multiple_testing_claims"}
KEYS = ["subject_uid", "segment_uid", "source"]
FORBIDDEN = {"sbp", "dbp", "target_sbp", "target_dbp", "SegSBP", "SegDBP"}


def select_people(metadata, *, seed=SEED, per_source=15):
    """Selection is deliberately independent of any label or outcome column."""
    people = metadata[["subject_uid", "source"]].drop_duplicates().sort_values(KEYS[:1])
    if people.subject_uid.duplicated().any() or set(people.source) != {"MIMIC", "VitalDB"}:
        raise ValueError("invalid subject/source identities")
    rng = np.random.default_rng(seed)
    selected = []
    for source in ("MIMIC", "VitalDB"):
        ids = sorted(people.loc[people.source.eq(source), "subject_uid"].astype(str))
        if len(ids) <= per_source:
            raise ValueError("not enough remaining population subjects")
        selected.extend(rng.choice(ids, per_source, replace=False).tolist())
    return sorted(selected)


def audit_split(train, test, selected, *, synthetic=False, protocol_id=PROTOCOL):
    """Audit all identities before splitting; no official source rows are deleted."""
    if protocol_id not in PROTOCOLS:
        raise ValueError("unknown enrollment protocol")
    per_source = PROTOCOLS[protocol_id]["per_source"]
    if FORBIDDEN & set(test) or any(str(c).lower().startswith(("target_", "abp", "segsbp", "segdbp")) for c in test):
        raise ValueError("test inputs contain reference labels")
    if train.segment_uid.duplicated().any() or test.segment_uid.duplicated().any():
        raise ValueError("duplicate segment identity")
    if set(train.segment_uid) & set(test.segment_uid):
        raise ValueError("train/test row identities overlap")
    if set(train.subject_uid) != set(test.subject_uid):
        raise ValueError("source train/test subjects differ")
    if not set(selected) <= set(train.subject_uid) or len(selected) != len(set(selected)):
        raise ValueError("invalid selected people")
    selected = set(selected)
    combined = pd.concat([train.assign(source_role="train"), test.assign(source_role="test")])
    combined["cohort"] = np.where(combined.subject_uid.isin(selected), "enrollment", "population")
    if combined.groupby("ppg_content_sha256").cohort.nunique().gt(1).any():
        raise ValueError("PPG content crosses population/new-user boundary")
    assert_cross_role_intervals(combined, "cohort")
    parts = {"population_train": train.loc[~train.subject_uid.isin(selected)].copy(),
             "population_test_inputs": test.loc[~test.subject_uid.isin(selected)].copy(),
             "enrollment_train": train.loc[train.subject_uid.isin(selected)].copy(),
             "enrollment_test_inputs": test.loc[test.subject_uid.isin(selected)].copy()}
    if not synthetic:
        if len(selected) != 2 * per_source or train.subject_uid.nunique() != 2506:
            raise ValueError(f"formal screen requires {2 * per_source} of 2506 people")
        src = train[["subject_uid", "source"]].drop_duplicates().groupby("source").size().to_dict()
        if src != {"MIMIC": 1213, "VitalDB": 1293}:
            raise ValueError("source cohort composition changed")
        if not train.groupby("subject_uid").size().eq(360).all() or not test.groupby("subject_uid").size().eq(40).all():
            raise ValueError("source must preserve exact 360/40 membership")
        selected_src = parts["enrollment_train"][["subject_uid", "source"]].drop_duplicates().groupby("source").size().to_dict()
        if selected_src != {"MIMIC": per_source, "VitalDB": per_source}:
            raise ValueError(f"enrollment stratum counts must be {per_source}/{per_source}")
    evidence = {}
    for name, part in parts.items():
        evidence[name] = {"rows": len(part), "subjects": part.subject_uid.nunique(),
            "subject_ids_sha256": event_ids_sha256(part.subject_uid.unique()),
            "segment_ids_sha256": event_ids_sha256(part.segment_uid),
            "source_subjects": part[["subject_uid", "source"]].drop_duplicates().groupby("source").size().to_dict()}
    evidence["cross_cohort_content_overlap"] = 0
    evidence["cross_cohort_subject_overlap"] = 0
    evidence["enrollment_test_rows_with_training_content"] = int(parts["enrollment_test_inputs"].ppg_content_sha256.isin(parts["enrollment_train"].ppg_content_sha256).sum())
    evidence["population_test_rows_with_training_content"] = int(parts["population_test_inputs"].ppg_content_sha256.isin(parts["population_train"].ppg_content_sha256).sum())
    return parts, evidence


def prepare(store, output, *, synthetic=False, protocol_id=PROTOCOL):
    if protocol_id not in PROTOCOLS:
        raise ValueError("unknown enrollment protocol")
    spec = PROTOCOLS[protocol_id]
    store, output = Path(store).resolve(), Path(output).resolve()
    contract = json.loads((store / "manifest.json").read_text())
    if contract.get("protocol_id") != SOURCE_PROTOCOL or contract.get("status") != "ready":
        raise ValueError("source is not an approved official store")
    if bool(contract.get("synthetic", False)) != synthetic:
        raise ValueError("synthetic/formal mismatch")
    if not synthetic:
        if contract.get("time_boundary_policy") != TIME_BOUNDARY_POLICY or contract.get("content_policy") != POLICY or contract.get("duplicate_audit_sha256") != AUDIT_SHA256:
            raise ValueError("source policies changed")
        approved_groups(str(store / "official_duplicate_audit.json"))
        if sha256(store / "train_manifest.parquet") != contract.get("train_manifest_sha256"):
            raise ValueError("source training manifest changed")
    # Label-free metadata exclusively determines sampling. The next read is
    # preprocessing after selection, not access by a fitted population model.
    identities = pd.read_parquet(store / "train_manifest.parquet", columns=["subject_uid", "source"])
    selected = select_people(identities, per_source=spec["synthetic_per_source"] if synthetic else spec["per_source"])
    train = pd.read_parquet(store / "train_manifest.parquet")
    train = validate_training_frame(train, smoke=synthetic)
    test = (pd.read_parquet(store / "test_inputs.parquet") if synthetic else load_test_inputs(store, train))
    parts, audit = audit_split(train, test, selected, synthetic=synthetic, protocol_id=protocol_id)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, frame in parts.items():
        path = output / f"{name}.parquet"
        frame.reset_index(drop=True).to_parquet(path, index=False)
        files[path.name] = sha256(path)
    plan = {"protocol_id": protocol_id, "status": "ready", "synthetic": synthetic,
        "source_root": str(store), "source_protocol": SOURCE_PROTOCOL,
        "source_manifest_sha256": sha256(store / "manifest.json"),
        "source_train_sha256": sha256(store / "train_manifest.parquet"),
        "source_test_inputs_sha256": sha256(store / "test_inputs.parquet"),
        "selection_seed": SEED, "selected_subjects": selected,
        "selection_uses_labels_or_errors": False, "files": files, "audit": audit,
        "old_checkpoint_permitted": False, "base_initialization": "fresh",
        "personal_shared_parameters": "frozen_encoder_head_and_BN",
        "personal_adapter": "fresh_rank4_A_normal_B_zero",
        "personal_parameter_count": 2048, "personal_selection": "inner_320_40_patience8_including_epoch0",
        "personal_refit": "fresh_adapter_all_360_selected_epochs",
        "methods": list(spec["methods"]), "memory_k": 5, "memory_temperature": 0.1,
        "memory_q95_fit": "each_person_registration_features_leave40block",
        "test_feedback_permitted": False, "official_rows_removed": False,
        "claims": "Exploratory post-training enrollment, randomized-window reconstruction; not few-cuff or future-date validation. Previously studied dataset, not a new confirmatory cohort."}
    if protocol_id == EXPANDED_PROTOCOL:
        previous = select_people(identities, per_source=1 if synthetic else 15)
        plan["previous_30_overlap_count"] = len(set(previous) & set(selected))
        plan["ablation_contract"] = ABLATION_CONTRACT
        plan["uncertainty"] = UNCERTAINTY
    save_json(output / "plan.json", plan)
    return plan


def load_plan(path, *, synthetic=False):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    if plan.get("protocol_id") not in PROTOCOLS or plan.get("status") != "ready" or bool(plan.get("synthetic")) != synthetic:
        raise ValueError("invalid enrollment plan")
    spec = PROTOCOLS[plan["protocol_id"]]
    if plan.get("selection_seed") != SEED or plan.get("selection_uses_labels_or_errors") is not False or plan.get("old_checkpoint_permitted") is not False or plan.get("test_feedback_permitted") is not False:
        raise ValueError("unapproved enrollment access contract")
    expected_people = 2 * (spec["synthetic_per_source"] if synthetic else spec["per_source"])
    if len(plan.get("selected_subjects", [])) != expected_people or len(set(plan["selected_subjects"])) != len(plan["selected_subjects"]):
        raise ValueError("selected people changed")
    if plan.get("methods") != list(spec["methods"]) or plan.get("personal_parameter_count") != 2048:
        raise ValueError("candidate configuration changed")
    if plan.get("memory_k") != 5 or plan.get("memory_temperature") != .1 or plan.get("memory_q95_fit") != "each_person_registration_features_leave40block":
        raise ValueError("memory configuration changed")
    if plan["protocol_id"] == EXPANDED_PROTOCOL and (plan.get("ablation_contract") != ABLATION_CONTRACT or plan.get("uncertainty") != UNCERTAINTY):
        raise ValueError("prespecified ablation or uncertainty contract changed")
    store = Path(plan["source_root"])
    if sha256(store / "manifest.json") != plan["source_manifest_sha256"]:
        raise ValueError("original store contract changed")
    return plan


def partition(path, name, *, synthetic=False):
    if name not in PARTITIONS:
        raise ValueError("unknown partition")
    path = Path(path).resolve()
    plan = load_plan(path, synthetic=synthetic)
    file = path.parent / f"{name}.parquet"
    if sha256(file) != plan["files"][file.name]:
        raise ValueError("partition hash changed")
    frame = pd.read_parquet(file)
    expected = plan["audit"][name]
    if len(frame) != expected["rows"] or event_ids_sha256(frame.segment_uid) != expected["segment_ids_sha256"] or event_ids_sha256(frame.subject_uid.unique()) != expected["subject_ids_sha256"]:
        raise ValueError("partition identities changed")
    selected = set(plan["selected_subjects"])
    if name.startswith("population") and set(frame.subject_uid) & selected:
        raise ValueError("new subject entered population fitting")
    if name.startswith("enrollment") and set(frame.subject_uid) != selected:
        raise ValueError("enrollment subject membership changed")
    if name.endswith("test_inputs") and FORBIDDEN & set(frame):
        raise ValueError("query target access forbidden")
    if name.endswith("train"):
        frame = validate_training_frame(frame, smoke=synthetic, expected_subjects=expected["subjects"])
    store = Path(plan["source_root"]).resolve()
    for name in frame.waveform_file.unique():
        file = (store / name).resolve()
        if not file.is_relative_to(store) or not file.is_file():
            raise ValueError("invalid waveform shard path")
    return plan, frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--protocol", choices=tuple(PROTOCOLS), default=PROTOCOL)
    args = p.parse_args()
    result = prepare(args.store_root, args.output, synthetic=args.synthetic, protocol_id=args.protocol)
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
