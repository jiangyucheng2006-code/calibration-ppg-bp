import json
import math
from pathlib import Path

import pandas as pd
import pytest

from pulsedb_fewshot.analyze_worst_tail_oracle import analyze


def _write_run(
    root: Path,
    name: str,
    errors: dict[str, float],
    *,
    locked_test_accessed: bool = False,
    drop_subject: str | None = None,
) -> Path:
    run = root / name
    run.mkdir()
    rows = []
    for index, (subject_uid, error) in enumerate(errors.items(), start=1):
        if subject_uid == drop_subject:
            continue
        rows.append(
            {
                "subject_uid": subject_uid,
                "event_id": f"{subject_uid}::{index:04d}",
                "k": 5,
                "target_sbp": 100.0,
                "target_dbp": 60.0,
                "pred_sbp": 100.0 + error,
                "pred_dbp": 60.0 + error,
            }
        )
    pd.DataFrame(rows).to_parquet(
        run / "best_validation_predictions.parquet", index=False
    )
    (run / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "split": "meta_validation",
                "locked_test_accessed": locked_test_accessed,
                "slurm_job_id": name,
                "checkpoint_sha256": f"hash-{name}",
            }
        ),
        encoding="utf-8",
    )
    return run


def test_exact_worst_30pct_uses_deterministic_tie_break_and_dynamic_names(
    tmp_path: Path,
) -> None:
    reference_errors = {
        "MIMIC:b": 10.0,
        "MIMIC:a": 10.0,
        "VitalDB:a": 10.0,
        "VitalDB:b": 4.0,
        "MIMIC:c": 1.0,
    }
    candidate_errors = {subject: 1.0 for subject in reference_errors}
    reference = _write_run(tmp_path, "reference", reference_errors)
    candidate = _write_run(tmp_path, "candidate", candidate_errors)
    output = tmp_path / "analysis"

    summary = analyze(
        reference,
        {"specialist": candidate},
        output,
        k=5,
        tail_fraction=0.30,
    )

    assert summary["tail_participants"] == math.ceil(0.30 * 5)
    assert summary["retained_participants"] == 3
    assert summary["tail_group"] == "worst_30pct_subjects"
    assert summary["retained_group"] == "oracle_retained_70pct_subjects"
    membership = pd.read_csv(
        output / "reference_worst_30pct_membership_k5.csv"
    )
    selected = membership.loc[
        membership["is_worst_30pct_reference"], "subject_uid"
    ].tolist()
    assert selected == ["MIMIC:a", "MIMIC:b"]
    assert membership["reference_error_rank"].tolist() == [1, 2, 3, 4, 5]
    assert summary["locked_meta_test_accessed"] is False
    assert "non-deployable" in summary["oracle_warning"]


def test_fixed_reference_tail_and_oracle_routing_are_participant_macro(
    tmp_path: Path,
) -> None:
    reference_errors = {
        "MIMIC:a": 10.0,
        "VitalDB:a": 9.0,
        "MIMIC:b": 5.0,
        "VitalDB:b": 2.0,
        "MIMIC:c": 1.0,
    }
    candidate_errors = {
        "MIMIC:a": 1.0,
        "VitalDB:a": 1.0,
        "MIMIC:b": 50.0,
        "VitalDB:b": 50.0,
        "MIMIC:c": 50.0,
    }
    reference = _write_run(tmp_path, "reference", reference_errors)
    candidate = _write_run(tmp_path, "candidate", candidate_errors)
    output = tmp_path / "analysis"
    analyze(reference, {"tail_specialist": candidate}, output)

    fixed = pd.read_csv(output / "fixed_reference_tail_metrics_k5.csv")
    assert set(fixed["source_scope"]) == {"Overall", "MIMIC", "VitalDB"}
    assert set(fixed["cohort"]) == {
        "overall",
        "worst_30pct_subjects",
        "oracle_retained_70pct_subjects",
    }
    tail = fixed.loc[
        fixed["setting"].eq("tail_specialist")
        & fixed["source_scope"].eq("Overall")
        & fixed["cohort"].eq("worst_30pct_subjects")
    ].iloc[0]
    assert tail["participants"] == 2
    assert tail["participant_macro_mean_mae"] == pytest.approx(1.0)
    assert bool(tail["oracle_only"])
    assert not bool(tail["deployable"])

    routing = pd.read_csv(output / "oracle_routing_mixture_k5.csv")
    overall = routing.loc[routing["source_scope"].eq("Overall")].iloc[0]
    expected_reference = sum(reference_errors.values()) / 5
    expected_routed = (1.0 + 1.0 + 5.0 + 2.0 + 1.0) / 5
    assert overall["reference_mean_mae"] == pytest.approx(expected_reference)
    assert overall["participant_macro_mean_mae"] == pytest.approx(expected_routed)
    assert overall["improvement_vs_reference_mean_mae"] == pytest.approx(
        expected_reference - expected_routed
    )
    assert overall["tail_participants_routed_to_candidate"] == 2
    assert overall["retained_participants_routed_to_reference"] == 3
    assert bool(overall["oracle_only"])
    assert not bool(overall["deployable"])


def test_analysis_rejects_locked_or_mismatched_candidate_runs(tmp_path: Path) -> None:
    errors = {
        "MIMIC:a": 3.0,
        "MIMIC:b": 2.0,
        "VitalDB:a": 3.0,
        "VitalDB:b": 2.0,
    }
    reference = _write_run(tmp_path, "reference", errors)
    locked = _write_run(
        tmp_path, "locked", errors, locked_test_accessed=True
    )
    with pytest.raises(ValueError, match="locked-test non-access"):
        analyze(reference, {"locked": locked}, tmp_path / "locked-output")

    mismatch = _write_run(
        tmp_path, "mismatch", errors, drop_subject="VitalDB:b"
    )
    with pytest.raises(AssertionError, match="query keys differ"):
        analyze(reference, {"mismatch": mismatch}, tmp_path / "mismatch-output")
