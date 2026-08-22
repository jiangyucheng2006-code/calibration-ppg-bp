from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pulsedb_fewshot.round11_backbones import build_report


def _write_backbone_run(root: Path, backbone: str, offset: float) -> None:
    root.mkdir()
    rows: list[dict[str, object]] = []
    for source, prefix in (("MIMIC", "m"), ("VitalDB", "v")):
        for participant_index in range(2):
            participant = f"{source}:{prefix}{participant_index}"
            for event_index in (6, 7):
                target_sbp = 118.0 + participant_index * 8 + event_index
                target_dbp = 70.0 + participant_index * 4 + event_index / 2
                rows.append(
                    {
                        "subject_uid": participant,
                        "event_id": f"{participant}:{event_index}",
                        "k": 5,
                        "source": source,
                        "split": "meta_train",
                        "fold": 4,
                        "target_sbp": target_sbp,
                        "target_dbp": target_dbp,
                        "pred_sbp": target_sbp + offset,
                        "pred_dbp": target_dbp + offset,
                        "population_pred_sbp": target_sbp + offset + 1.0,
                        "population_pred_dbp": target_dbp + offset + 1.0,
                    }
                )
    pd.DataFrame(rows).to_parquet(root / "queries.parquet", index=False)
    (root / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "round": 11,
                "stage": "backbone_evaluation",
                "backbone": backbone,
                "seed": 20260825,
                "meta_validation_accessed": False,
                "meta_validation_used_for_training": False,
                "meta_validation_used_for_early_stopping": False,
                "meta_validation_used_for_candidate_ranking": False,
                "meta_validation_predictions_generated": False,
                "locked_test_accessed": False,
                "query_bp_model_input": False,
                "future_query_model_input": False,
                "source_model_input": False,
                "population_parameter_counts": {"total": 100, "trainable": 100},
                "qgh_parameter_counts": {"total": 200, "trainable": 50},
            }
        ),
        encoding="utf-8",
    )


def test_round11_report_uses_common_fold4_queries_and_three_scopes(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_backbone_run(reference, "resnet_small", 1.0)
    _write_backbone_run(candidate, "conformer", 0.5)
    output = tmp_path / "report"
    result = build_report(
        runs={"resnet_small": reference, "conformer": candidate},
        reference_backbone="resnet_small",
        output=output,
        expected_seed=20260825,
    )
    table = pd.read_csv(output / "participant_macro_internal.csv")
    assert set(table["Scope"]) == {"Overall", "MIMIC", "VitalDB"}
    assert set(table["Model"]) == {"Population", "QGH"}
    assert result["winner_backbone"] == "conformer"
    assert result["passes_internal_gate"] is True
    assert (output / "pooled_diagnostics_internal.csv").is_file()
    assert (output / "model_complexity.csv").is_file()


def test_round11_report_rejects_locked_test_access(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    _write_backbone_run(reference, "resnet_small", 1.0)
    payload = json.loads((reference / "run.json").read_text(encoding="utf-8"))
    payload["locked_test_accessed"] = True
    (reference / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="unsafe flag"):
        build_report(
            runs={"resnet_small": reference},
            reference_backbone="resnet_small",
            output=tmp_path / "report",
            expected_seed=20260825,
        )
