import pandas as pd

from pulsedb_fewshot.frozen_protocol import KS, add_role_columns, audit_frozen_manifests


def _frame(split: str, subject: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "event_id": [f"{subject}::{i:04d}" for i in range(1, 11)],
            "subject_uid": subject,
            "event_index": range(1, 11),
            "split": split,
        }
    )
    frame = add_role_columns(frame)
    if split == "meta_test":
        for k in KS:
            support = frame[f"role_k{k}"].eq("support")
            frame[f"support_sbp_k{k}"] = pd.Series(120.0, index=frame.index).where(support)
            frame[f"support_dbp_k{k}"] = pd.Series(70.0, index=frame.index).where(support)
    return frame


def test_roles_use_one_common_query_set() -> None:
    frame = _frame("meta_train", "MIMIC:s1")
    query_sets = {
        k: tuple(frame.loc[frame[f"role_k{k}"].eq("query"), "event_id"])
        for k in KS
    }
    assert len(set(query_sets.values())) == 1
    assert len(query_sets[1]) == 5


def test_locked_model_input_has_no_query_labels() -> None:
    development = _frame("meta_train", "MIMIC:s1")
    locked_inputs = _frame("meta_test", "VitalDB:s2")
    locked_targets = pd.DataFrame(
        {
            "event_id": locked_inputs.loc[locked_inputs["common_query"], "event_id"],
            "subject_uid": "VitalDB:s2",
            "sbp": 120.0,
            "dbp": 70.0,
            "split": "meta_test",
        }
    )
    splits = pd.DataFrame(
        {
            "subject_uid": ["MIMIC:s1", "VitalDB:s2"],
            "split": ["meta_train", "meta_test"],
        }
    )
    audit = audit_frozen_manifests(development, locked_inputs, locked_targets, splits)
    assert audit["status"] == "pass"
    assert not audit["query_labels_in_model_input"]


def test_audit_rejects_target_column_in_model_input() -> None:
    development = _frame("meta_validation", "MIMIC:s1")
    locked_inputs = _frame("meta_test", "VitalDB:s2")
    locked_inputs["sbp"] = 120.0
    locked_targets = pd.DataFrame(
        {
            "event_id": locked_inputs.loc[locked_inputs["common_query"], "event_id"],
            "subject_uid": "VitalDB:s2",
            "sbp": 120.0,
            "dbp": 70.0,
            "split": "meta_test",
        }
    )
    splits = pd.DataFrame(
        {
            "subject_uid": ["MIMIC:s1", "VitalDB:s2"],
            "split": ["meta_validation", "meta_test"],
        }
    )
    audit = audit_frozen_manifests(development, locked_inputs, locked_targets, splits)
    assert audit["status"] == "fail"
    assert "locked_inputs_expose_query_target_columns" in audit["failures"]
