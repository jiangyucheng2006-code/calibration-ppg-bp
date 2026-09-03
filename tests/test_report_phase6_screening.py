import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulsedb_fewshot.report_phase6_screening import (
    EVENT_TABLE_COLUMNS,
    _exact_worst_fraction,
    generate_report,
)


SUBJECTS = ["MIMIC:m1", "MIMIC:m2", "VitalDB:v1", "VitalDB:v2"]
SOURCES = ["MIMIC", "MIMIC", "VitalDB", "VitalDB"]
SBP = [100.0, 110.0, 120.0, 130.0]
DBP = [60.0, 65.0, 70.0, 75.0]


def _write_store(root: Path, *, vital_source: str = "VitalDB") -> None:
    root.mkdir(parents=True, exist_ok=True)
    source = ["MIMIC", "MIMIC", vital_source, vital_source]
    metadata = pd.DataFrame(
        {
            "subject_uid": SUBJECTS,
            "event_id": [f"{subject}:event006" for subject in SUBJECTS],
            "split": "meta_validation",
            "common_query": True,
            "source": source,
            "sbp": SBP,
            "dbp": DBP,
        }
    )
    metadata.to_parquet(root / "development_metadata_000.parquet", index=False)


def _predictions(
    *,
    sbp_errors: list[float],
    dbp_errors: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for k in (1, 2, 3, 5):
        for index, subject in enumerate(SUBJECTS):
            rows.append(
                {
                    "subject_uid": subject,
                    "event_id": f"{subject}:event006",
                    "k": k,
                    "target_sbp": SBP[index],
                    "target_dbp": DBP[index],
                    "pred_sbp": SBP[index] + sbp_errors[index] / k,
                    "pred_dbp": DBP[index] + dbp_errors[index] / k,
                }
            )
    return pd.DataFrame(rows)


def _write_run(
    root: Path,
    predictions: pd.DataFrame,
    *,
    seed: int = 20260813,
    locked_test_accessed: bool = False,
    support_policy: str = "fixed_first",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(root / "best_validation_predictions.parquet", index=False)
    (root / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "method": "m0",
                "split": "meta_validation",
                "locked_test_accessed": locked_test_accessed,
                "seed": seed,
                "arguments": {"train_support_policy": support_policy},
            }
        ),
        encoding="utf-8",
    )


def _valid_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    store = tmp_path / "store"
    _write_store(store)
    reference = tmp_path / "event120-v1_phase6_fixed_first_seed20260813_job818"
    candidate = tmp_path / "event120-v1_phase6_candidate_seed20260813_job826"
    _write_run(
        reference,
        _predictions(sbp_errors=[12.0, 8.0, 6.0, 4.0], dbp_errors=[6.0, 4.0, 3.0, 2.0]),
    )
    _write_run(
        candidate,
        _predictions(sbp_errors=[8.0, 6.0, 4.0, 2.0], dbp_errors=[4.0, 3.0, 2.0, 1.0]),
    )
    return store, {"Fixed-first M0": reference, "Candidate": candidate}


def test_phase6_report_writes_overall_and_both_source_tables(tmp_path: Path) -> None:
    store, runs = _valid_fixture(tmp_path)
    output = tmp_path / "report"
    report = generate_report(
        runs=runs,
        reference_setting="Fixed-first M0",
        store_root=store,
        output_dir=output,
        expected_seed=20260813,
    )

    assert report["status"] == "pass"
    assert report["locked_test_accessed"] is False
    assert report["coverage"]["Overall"] == {
        "participants": 4,
        "common_query_events_per_setting_k": 4,
    }
    assert report["coverage"]["MIMIC"]["participants"] == 2
    assert report["coverage"]["VitalDB"]["participants"] == 2

    for name in ("overall", "mimic", "vitaldb"):
        table = pd.read_csv(output / f"phase6_{name}_metrics.csv")
        assert list(table.columns) == EVENT_TABLE_COLUMNS
        assert len(table) == 2 * 4 * 2
        assert set(table["BP"]) == {"SBP", "DBP"}
        assert table["Setting"].str.contains(r"K=(?:1|2|3|5)", regex=True).all()

    participant = pd.read_csv(output / "phase6_participant_macro_by_scope.csv")
    assert set(participant["Scope"]) == {"Overall", "MIMIC", "VitalDB"}
    assert set(participant.loc[participant["Scope"].eq("Overall"), "N participants"]) == {4}
    candidate_delta = participant.loc[
        participant["Setting"].eq("Candidate")
        & participant["Scope"].eq("Overall")
        & participant["K"].eq(1),
        "Paired delta vs reference",
    ].iloc[0]
    assert candidate_delta < 0.0

    tail = pd.read_csv(output / "phase6_oracle_tail_by_scope.csv")
    assert set(tail.loc[tail["Scope"].eq("Overall"), "Worst 30% participants"]) == {2}
    assert set(tail.loc[tail["Scope"].ne("Overall"), "Worst 30% participants"]) == {1}
    assert tail["Selection"].str.contains("oracle", case=False).all()
    reference_tail = pd.read_csv(
        output / "phase6_reference_oracle_tail_comparison.csv"
    )
    assert set(reference_tail["Reference tail setting"]) == {"Fixed-first M0"}

    markdown = (output / "PHASE6_SCREENING_RESULTS.md").read_text(encoding="utf-8")
    assert "MIMIC and VitalDB are PulseDB source strata" in markdown
    assert "oracle diagnostics" in markdown
    assert "formal compliance is not established" in markdown
    serialized = json.loads(
        (output / "phase6_screening_report.json").read_text(encoding="utf-8")
    )
    assert serialized["sources"] == ["MIMIC", "VitalDB"]
    assert all("\\" not in item["run_directory_name"] for item in serialized["inputs"])


def test_exact_worst_fraction_has_exact_size_and_deterministic_tie_break() -> None:
    participants = pd.DataFrame(
        {
            "subject_uid": ["d", "c", "b", "a", "e", "f", "g"],
            "mean_mae": [9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0],
        }
    )
    tail, remaining, threshold = _exact_worst_fraction(participants, fraction=0.30)
    assert len(tail) == math.ceil(0.30 * len(participants)) == 3
    assert tail["subject_uid"].tolist() == ["a", "b", "c"]
    assert len(remaining) == 4
    assert threshold == 9.0


@pytest.mark.parametrize(
    ("locked", "policy", "message"),
    [
        (True, "fixed_first", "locked-test"),
        (False, "rolling_recent", "fixed_first"),
    ],
)
def test_phase6_report_rejects_leakage_or_wrong_support_policy(
    tmp_path: Path,
    locked: bool,
    policy: str,
    message: str,
) -> None:
    store, runs = _valid_fixture(tmp_path)
    bad = runs["Candidate"]
    _write_run(
        bad,
        _predictions(sbp_errors=[8.0, 6.0, 4.0, 2.0], dbp_errors=[4.0, 3.0, 2.0, 1.0]),
        locked_test_accessed=locked,
        support_policy=policy,
    )
    with pytest.raises(AssertionError, match=message):
        generate_report(
            runs=runs,
            reference_setting="Fixed-first M0",
            store_root=store,
            output_dir=tmp_path / "report",
        )


def test_phase6_report_rejects_target_or_query_mismatch(tmp_path: Path) -> None:
    store, runs = _valid_fixture(tmp_path)
    path = runs["Candidate"] / "best_validation_predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions.loc[0, "target_sbp"] += 1.0
    predictions.to_parquet(path, index=False)
    with pytest.raises(AssertionError, match="(?:targets differ|differs from reference targets)"):
        generate_report(
            runs=runs,
            reference_setting="Fixed-first M0",
            store_root=store,
            output_dir=tmp_path / "report",
        )


def test_phase6_report_requires_both_known_sources(tmp_path: Path) -> None:
    store, runs = _valid_fixture(tmp_path)
    _write_store(store, vital_source="MIMIC")
    with pytest.raises(AssertionError, match="expected meta-validation sources"):
        generate_report(
            runs=runs,
            reference_setting="Fixed-first M0",
            store_root=store,
            output_dir=tmp_path / "report",
        )
