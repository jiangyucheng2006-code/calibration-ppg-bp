"""Freeze the 120-second PulseDB event protocol and build leakage-safe manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .events import eventize_segments, summarize_eligibility


PROTOCOL_ID = "event120-v1"
EVENT_WIDTH_SECONDS = 120
KS = (1, 2, 3, 5)
MAX_K = max(KS)
MIN_QUERY_EVENTS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_role_columns(events: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic K-specific roles while preserving one row per event."""

    result = events.copy()
    result["support_candidate"] = result["event_index"].le(MAX_K)
    result["common_query"] = result["event_index"].gt(MAX_K)
    for k in KS:
        result[f"role_k{k}"] = np.select(
            [result["event_index"].le(k), result["event_index"].le(MAX_K)],
            ["support", "unused_calibration_pool"],
            default="query",
        )
    return result


def audit_frozen_manifests(
    development: pd.DataFrame,
    locked_inputs: pd.DataFrame,
    locked_targets: pd.DataFrame,
    subject_splits: pd.DataFrame,
) -> dict[str, object]:
    """Assert subject, time, role, and target-label isolation invariants."""

    failures: list[str] = []
    split_sets = {
        split: set(group["subject_uid"])
        for split, group in subject_splits.groupby("split", sort=False)
    }
    split_names = sorted(split_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if split_sets[left] & split_sets[right]:
                failures.append(f"subject_overlap:{left}:{right}")

    if set(development["split"]) - {"meta_train", "meta_validation"}:
        failures.append("development_contains_nondevelopment_split")
    if set(locked_inputs["split"]) != {"meta_test"}:
        failures.append("locked_inputs_split_invalid")
    if set(locked_targets["split"]) != {"meta_test"}:
        failures.append("locked_targets_split_invalid")

    forbidden_target_columns = {"sbp", "dbp", "target_sbp", "target_dbp"}
    if forbidden_target_columns & set(locked_inputs.columns):
        failures.append("locked_inputs_expose_query_target_columns")
    expected_target_columns = {"event_id", "subject_uid", "sbp", "dbp", "split"}
    if not expected_target_columns.issubset(locked_targets.columns):
        failures.append("locked_targets_missing_required_columns")
    if not locked_targets["event_id"].is_unique:
        failures.append("locked_targets_event_id_not_unique")

    for frame_name, frame in (("development", development), ("locked_inputs", locked_inputs)):
        if not frame["event_id"].is_unique:
            failures.append(f"{frame_name}_event_id_not_unique")
        ordered = frame.sort_values(
            ["subject_uid", "event_index"], kind="mergesort"
        )
        expected = ordered.groupby("subject_uid", sort=False).cumcount().add(1)
        if not expected.reset_index(drop=True).equals(
            ordered["event_index"].reset_index(drop=True)
        ):
            failures.append(f"{frame_name}_event_index_not_contiguous")
        for k in KS:
            role = frame[f"role_k{k}"]
            support = frame.loc[role.eq("support")]
            query = frame.loc[role.eq("query")]
            support_max = support.groupby("subject_uid")["event_index"].max()
            query_min = query.groupby("subject_uid")["event_index"].min()
            aligned = support_max.to_frame("support_max").join(query_min.rename("query_min"))
            if not aligned["support_max"].lt(aligned["query_min"]).all():
                failures.append(f"{frame_name}_support_not_before_query_k{k}")
        query_sets = {
            k: set(frame.loc[frame[f"role_k{k}"].eq("query"), "event_id"])
            for k in KS
        }
        if len({frozenset(values) for values in query_sets.values()}) != 1:
            failures.append(f"{frame_name}_common_query_mismatch")

    for k in KS:
        for target in ("sbp", "dbp"):
            column = f"support_{target}_k{k}"
            if column not in locked_inputs:
                failures.append(f"locked_inputs_missing_{column}")
                continue
            expected_available = locked_inputs[f"role_k{k}"].eq("support")
            if not locked_inputs[column].notna().equals(expected_available):
                failures.append(f"locked_inputs_{column}_availability_invalid")

    input_query_ids = set(locked_inputs.loc[locked_inputs["common_query"], "event_id"])
    target_query_ids = set(locked_targets["event_id"])
    if input_query_ids != target_query_ids:
        failures.append("locked_input_target_query_id_mismatch")

    return {
        "status": "pass" if not failures else "fail",
        "protocol_id": PROTOCOL_ID,
        "failures": failures,
        "subject_split_counts": {
            key: len(value) for key, value in split_sets.items()
        },
        "development_rows": int(len(development)),
        "development_subjects": int(development["subject_uid"].nunique()),
        "locked_input_rows": int(len(locked_inputs)),
        "locked_eligible_subjects": int(locked_inputs["subject_uid"].nunique()),
        "locked_query_targets": int(len(locked_targets)),
        "query_labels_in_model_input": False,
    }


def freeze_event120_protocol(
    full_segments_path: Path,
    subject_splits_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Apply the frozen event rule to every split and write isolated artifacts."""

    full = pd.read_parquet(full_segments_path)
    splits = pd.read_csv(subject_splits_path)
    required_split_columns = {"subject_uid", "split"}
    if not required_split_columns.issubset(splits.columns):
        raise ValueError("subject split file is missing subject_uid or split")
    if splits["subject_uid"].duplicated().any():
        raise AssertionError("subject split assignments are not unique")
    split_map = splits.set_index("subject_uid")["split"]
    full["split"] = full["subject_uid"].map(split_map)
    if full["split"].isna().any():
        raise AssertionError("full segment index contains unassigned subjects")
    if not full["segment_schema_valid"].all():
        raise AssertionError("frozen eventization requires schema-valid segments")

    quality = full.loc[full["include_flag"].eq(1)].copy()
    event_input = quality[
        [
            "subject_uid",
            "record_id",
            "record_order",
            "source",
            "segment_uid",
            "start_time_s",
            "sbp",
            "dbp",
        ]
    ].rename(columns={"subject_uid": "subject_id", "segment_uid": "segment_id"})
    events = eventize_segments(
        event_input, bin_width_sec=float(EVENT_WIDTH_SECONDS)
    ).rename(columns={"subject_id": "subject_uid", "segment_id": "segment_uid"})
    events["split"] = events["subject_uid"].map(split_map)

    eligibility_input = events.rename(columns={"subject_uid": "subject_id"})
    eligibility = summarize_eligibility(
        eligibility_input,
        max_k=MAX_K,
        min_query_events=MIN_QUERY_EVENTS,
    ).rename(columns={"subject_id": "subject_uid"})
    eligibility["split"] = eligibility["subject_uid"].map(split_map)
    eligible_ids = set(eligibility.loc[eligibility["eligible"], "subject_uid"])
    events = events.loc[events["subject_uid"].isin(eligible_ids)].copy()

    # Retention is required after applying the frozen rule, but locked-test BP
    # distributions are not needed for that gate. Persist count-only eligibility
    # for every split so this artifact cannot become a source of locked-outcome
    # inspection during model development.
    eligibility = eligibility[
        [
            "subject_uid",
            "source",
            "split",
            "n_events",
            "n_support_candidates",
            "n_common_query_events",
            "required_events",
            "eligible",
        ]
    ]

    metadata_columns = [
        "segment_uid",
        "raw_file",
        "raw_file_relative_path",
        "raw_file_sha256",
        "ppg_field",
        "ppg_storage_mode",
        "ppg_reference_index",
        "sampling_rate_hz",
        "n_samples",
        "duration_s",
        "ppg_f_mean",
        "ppg_f_std",
    ]
    events = events.merge(
        full[metadata_columns], on="segment_uid", how="left", validate="one_to_one"
    )
    if events["raw_file"].isna().any():
        raise AssertionError("an event representative could not be traced to raw data")
    events.insert(0, "protocol_id", PROTOCOL_ID)
    events = add_role_columns(events)

    development = events.loc[events["split"].isin(["meta_train", "meta_validation"])].copy()
    locked = events.loc[events["split"].eq("meta_test")].copy()
    locked_targets = locked.loc[
        locked["common_query"], ["event_id", "subject_uid", "sbp", "dbp", "split"]
    ].copy()
    locked_inputs = locked.drop(columns=["sbp", "dbp"])
    for k in KS:
        support_mask = locked[f"role_k{k}"].eq("support")
        locked_inputs[f"support_sbp_k{k}"] = locked["sbp"].where(support_mask)
        locked_inputs[f"support_dbp_k{k}"] = locked["dbp"].where(support_mask)

    audit = audit_frozen_manifests(
        development, locked_inputs, locked_targets, splits
    )
    if audit["status"] != "pass":
        raise AssertionError(f"frozen protocol leakage audit failed: {audit['failures']}")

    output_root.mkdir(parents=True, exist_ok=True)
    locked_root = output_root / "locked_evaluator_only"
    locked_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "development_episodes": output_root / "development_episodes.parquet",
        "locked_model_inputs": output_root / "locked_model_inputs.parquet",
        "locked_evaluator_targets": locked_root / "locked_query_targets.parquet",
        "eligibility": output_root / "eligibility_all_splits.parquet",
        "leakage_audit": output_root / "leakage_audit.json",
        "protocol": output_root / "protocol.json",
    }
    development.to_parquet(paths["development_episodes"], index=False)
    locked_inputs.to_parquet(paths["locked_model_inputs"], index=False)
    locked_targets.to_parquet(paths["locked_evaluator_targets"], index=False)
    eligibility.to_parquet(paths["eligibility"], index=False)

    paths["leakage_audit"].write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = {
        "status": "pass",
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_width_seconds": EVENT_WIDTH_SECONDS,
        "ks": list(KS),
        "max_support_candidate_events": MAX_K,
        "min_common_query_events": MIN_QUERY_EVENTS,
        "representative_rule": "closest segment to each record-anchored 120-second bin centre",
        "quality_rule": "segment_schema_valid and IncludeFlag == 1",
        "locked_test_rule": "support BP is available only through evaluator protocol; query BP is stored in evaluator-only target file",
        "selection_basis": "development-only feasibility plus stronger temporal separation than 60 seconds; selected before locked-test eventization/model errors",
        "input_hashes": {
            "full_segments": sha256_file(full_segments_path),
            "subject_splits": sha256_file(subject_splits_path),
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
            if name not in {"protocol"}
        },
        "audit": audit,
    }
    paths["protocol"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-segments", required=True, type=Path)
    parser.add_argument("--subject-splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = freeze_event120_protocol(
        args.full_segments, args.subject_splits, args.output
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
