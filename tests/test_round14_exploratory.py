import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from pulsedb_fewshot.round14_exploratory import (  # noqa: E402
    METHODS,
    STANDARDS_HUBER_WEIGHT,
    STANDARDS_THRESHOLD_WEIGHT,
    STANDARDS_THRESHOLD_WEIGHTS,
    STANDARDS_THRESHOLDS_MMHG,
    _split_cache_queries,
    build_internal_report,
    candidate_loss,
    smooth_worst_group_sbp_risk,
    standards_surrogate,
    validate_base_run_pair,
)


def _safe_record(method: str) -> dict[str, object]:
    return {
        "status": "complete",
        "method": method,
        "backbone": "inception_time_wide",
        "seed": 20260828,
        "crossfit_fit_folds": [0, 1, 2],
        "crossfit_validation_fold": 3,
        "crossfit_excluded_folds": [4],
        "locked_test_accessed": False,
        "store_manifest_sha256": "store-sha",
        "crossfit_folds_sha256": "fold-sha",
        "source_tree_sha256": "source-sha",
    }


def test_base_pair_validation_requires_the_matched_wide_qgh_contract() -> None:
    population = _safe_record("population")
    qgh = _safe_record("m0")
    qgh["arguments"] = {
        "loss": "huber",
        "huber_delta": 0.5,
        "use_quality_gate": True,
        "train_support_policy": "fixed_first",
        "ks": [5],
    }
    validate_base_run_pair(
        population,
        qgh,
        expected_seed=20260828,
        store_manifest_sha256="store-sha",
        folds_sha256="fold-sha",
    )
    unsafe = copy.deepcopy(qgh)
    unsafe["crossfit_excluded_folds"] = []
    with pytest.raises(AssertionError, match="exclude fold 4"):
        validate_base_run_pair(
            population,
            unsafe,
            expected_seed=20260828,
            store_manifest_sha256="store-sha",
            folds_sha256="fold-sha",
        )


def test_standards_surrogate_uses_physical_mmhg_and_is_differentiable() -> None:
    target = torch.zeros(2, 2, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)
    target_std = torch.tensor([20.0, 10.0])
    small = torch.zeros_like(target, requires_grad=True)
    large = torch.full_like(target, 1.0, requires_grad=True)
    small_loss, _ = standards_surrogate(small, target, mask, target_std)
    large_loss, diagnostics = standards_surrogate(large, target, mask, target_std)
    assert large_loss > small_loss
    assert diagnostics["physical_huber"] > 0
    assert diagnostics["threshold_surrogate"] > 0
    large_loss.backward()
    assert large.grad is not None
    assert torch.isfinite(large.grad).all()


def test_standards_surrogate_matches_the_frozen_numeric_formula() -> None:
    target = torch.zeros(1, 1, 2)
    prediction = torch.ones(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)
    target_std = torch.tensor([20.0, 10.0])

    loss, diagnostics = standards_surrogate(
        prediction, target, mask, target_std
    )

    # For physical errors [20, 10] mmHg and Huber delta 5 mmHg:
    # mean([5*(20-2.5), 5*(10-2.5)]) / 5^2 = 2.5.
    expected_huber = 2.5
    physical_errors = torch.tensor([20.0, 10.0])
    expected_threshold = sum(
        weight
        * torch.sigmoid(physical_errors - threshold).mean().item()
        for threshold, weight in zip(
            STANDARDS_THRESHOLDS_MMHG,
            STANDARDS_THRESHOLD_WEIGHTS,
            strict=True,
        )
    )
    expected_total = (
        STANDARDS_HUBER_WEIGHT * expected_huber
        + STANDARDS_THRESHOLD_WEIGHT * expected_threshold
    )
    assert diagnostics["physical_huber"] == pytest.approx(expected_huber)
    assert diagnostics["threshold_surrogate"] == pytest.approx(
        expected_threshold
    )
    assert loss.item() == pytest.approx(expected_total)


def test_groupdro_penalizes_unequal_training_group_risk() -> None:
    target = torch.zeros(2, 2, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)
    groups = torch.tensor([[0, 0], [4, 4]])
    equal = torch.ones_like(target, requires_grad=True)
    unequal = torch.zeros_like(target, requires_grad=True)
    unequal.data[1, :, 0] = 4.0
    equal_penalty, _ = smooth_worst_group_sbp_risk(
        equal, target, mask, groups
    )
    unequal_penalty, diagnostics = smooth_worst_group_sbp_risk(
        unequal, target, mask, groups
    )
    assert unequal_penalty > equal_penalty
    assert diagnostics["groupdro_smooth_worst_sbp_risk"] > diagnostics[
        "groupdro_mean_sbp_risk"
    ]
    unequal_penalty.backward()
    assert unequal.grad is not None
    assert torch.isfinite(unequal.grad).all()


def _loss_output() -> dict[str, torch.Tensor]:
    batch, length, support = 3, 4, 5
    prediction = torch.randn(batch, length, 2, requires_grad=True)
    return {
        "prediction": prediction,
        "target": torch.randn(batch, length, 2),
        "mask": torch.ones(batch, length, dtype=torch.bool),
        "pair_delta": torch.randn(
            batch, length, support, 2, requires_grad=True
        ),
        "support_bp": torch.randn(batch, length, support, 2),
        "range_logits": torch.randn(
            batch, length, 6, requires_grad=True
        ),
        "range": torch.randint(0, 3, (batch, length, 2)),
        "group": torch.tensor(
            [[0, 1, 2, 0], [3, 4, 5, 3], [0, 4, 2, 5]]
        ),
    }


@pytest.mark.parametrize("method", METHODS)
def test_all_round14_losses_retain_the_complete_c1_objective(method: str) -> None:
    output = _loss_output()
    loss, diagnostics = candidate_loss(
        output,
        method=method,
        target_std=torch.tensor([20.0, 10.0]),
    )
    assert torch.isfinite(loss)
    assert "calibration_relative_loss" in diagnostics
    loss.backward()
    assert output["prediction"].grad is not None


def test_internal_split_is_participant_disjoint_and_meta_train_only() -> None:
    rows = []
    for fold in range(5):
        for event in (6, 7):
            rows.append(
                {
                    "subject_uid": f"subject-{fold}",
                    "event_id": f"event-{fold}-{event}",
                    "event_index": event,
                    "split": "meta_train",
                    "fold": fold,
                }
            )
    prepared = type("Prepared", (), {"queries": pd.DataFrame(rows)})()
    fit, early, selection = _split_cache_queries(prepared)
    assert set(fit["fold"]) == {0, 1, 2}
    assert set(early["fold"]) == {3}
    assert set(selection["fold"]) == {4}
    prepared.queries.loc[0, "split"] = "meta_validation"
    with pytest.raises(AssertionError, match="non-meta-train"):
        _split_cache_queries(prepared)


def _write_candidate_run(
    root: Path, method: str, offset: float | dict[str, float]
) -> None:
    root.mkdir()
    rows = []
    for source, participant, target_sbp, target_dbp in (
        ("MIMIC", "private-a", 120.0, 75.0),
        ("VitalDB", "private-b", 130.0, 80.0),
    ):
        source_offset = offset[source] if isinstance(offset, dict) else offset
        for event in (6, 7):
            rows.append(
                {
                    "subject_uid": participant,
                    "event_id": f"{participant}:{event}",
                    "k": 5,
                    "source": source,
                    "target_sbp": target_sbp,
                    "target_dbp": target_dbp,
                    "pred_sbp": target_sbp + source_offset,
                    "pred_dbp": target_dbp + source_offset,
                }
            )
    pd.DataFrame(rows).to_parquet(
        root / "selection_predictions.parquet", index=False
    )
    (root / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "round": 14,
                "stage": "exploratory_method",
                "method": method,
                "seed": 20260828,
                "backbone": "inception_time_wide",
                "split": "meta_train_internal_fold4",
                "fit_folds": [0, 1, 2],
                "early_stopping_fold": 3,
                "selection_fold": 4,
                "aami_bhs_compliance_claim": False,
                "source_model_input": False,
                "meta_validation_used_for_training": False,
                "meta_validation_used_for_early_stopping": False,
                "meta_validation_used_for_candidate_ranking": False,
                "meta_validation_predictions_generated": False,
                "locked_test_accessed": False,
                "query_bp_model_input": False,
                "future_query_model_input": False,
                "cache_run_sha256": "common-cache",
                "cache_bindings": {
                    "population_checkpoint_sha256": "population-checkpoint",
                    "qgh_checkpoint_sha256": "qgh-checkpoint",
                    "folds_sha256": "folds",
                    "store_manifest_sha256": "store",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_anchor_run(root: Path, offset: float) -> None:
    _write_candidate_run(root, "calibration_relative", offset)
    predictions = pd.read_parquet(root / "selection_predictions.parquet")
    predictions["fold"] = 4
    predictions["split"] = "meta_train"
    predictions.to_parquet(root / "queries.parquet", index=False)
    (root / "selection_predictions.parquet").unlink()
    (root / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "round": 14,
                "stage": "backbone_evaluation",
                "backbone": "inception_time_wide",
                "seed": 20260828,
                "fit_folds": [0, 1, 2],
                "early_stopping_fold": 3,
                "selection_fold": 4,
                "support_policy": "fixed_first",
                "k": 5,
                "meta_validation_accessed": False,
                "locked_test_accessed": False,
                "query_bp_model_input": False,
                "future_query_model_input": False,
                "source_model_input": False,
                "meta_validation_used_for_training": False,
                "meta_validation_used_for_early_stopping": False,
                "meta_validation_used_for_candidate_ranking": False,
                "meta_validation_predictions_generated": False,
                "population_checkpoint_sha256": "population-checkpoint",
                "qgh_checkpoint_sha256": "qgh-checkpoint",
                "folds_sha256": "folds",
                "store_manifest_sha256": "store",
            }
        ),
        encoding="utf-8",
    )


def test_report_recomputes_all_scopes_from_common_fold4_queries(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_anchor_run(reference, 2.0)
    _write_candidate_run(
        candidate, "calibration_relative_standards", 1.0
    )
    output = tmp_path / "report"
    result = build_internal_report(
        runs={
            "calibration_relative_standards": candidate,
        },
        anchor_run=reference,
        output=output,
        expected_seed=20260828,
    )
    table = pd.read_csv(output / "participant_macro_internal.csv")
    diagnostics = pd.read_csv(output / "pooled_diagnostics_internal.csv")
    gates = pd.read_csv(output / "candidate_gate_internal.csv")
    assert set(table["Scope"]) == {"Overall", "MIMIC", "VitalDB"}
    assert set(diagnostics["Scope"]) == {"Overall", "MIMIC", "VitalDB"}
    assert len(diagnostics) == 2 * 3 * 2
    assert gates.loc[0, "Primary gate"]
    assert result["winner"] == "calibration_relative_standards"
    assert result["passes_internal_gate"] is True


def test_each_candidate_gate_is_independent_of_the_numerical_winner(
    tmp_path: Path,
) -> None:
    anchor = tmp_path / "anchor"
    passing = tmp_path / "passing"
    source_opposed = tmp_path / "source-opposed"
    _write_anchor_run(anchor, 2.0)
    _write_candidate_run(
        passing, "calibration_relative_standards", 1.5
    )
    _write_candidate_run(
        source_opposed,
        "calibration_relative_groupdro",
        {"MIMIC": 0.0, "VitalDB": 2.5},
    )
    output = tmp_path / "report"
    result = build_internal_report(
        runs={
            "calibration_relative_standards": passing,
            "calibration_relative_groupdro": source_opposed,
        },
        anchor_run=anchor,
        output=output,
        expected_seed=20260828,
    )
    gates = pd.read_csv(output / "candidate_gate_internal.csv").set_index(
        "Setting"
    )
    assert result["winner"] == "calibration_relative_groupdro"
    assert not gates.loc["calibration_relative_groupdro", "Primary gate"]
    assert gates.loc["calibration_relative_standards", "Primary gate"]
    assert result["promoted_candidate"] == "calibration_relative_standards"
    assert result["passes_internal_gate"] is True


def test_exploratory_report_rejects_an_unregistered_seed(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="requires seed 20260828"):
        build_internal_report(
            runs={},
            anchor_run=tmp_path / "not-read",
            output=tmp_path / "report",
            expected_seed=20260829,
        )
