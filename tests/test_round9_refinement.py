import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from pulsedb_fewshot.round9_refinement import (  # noqa: E402
    METHODS,
    CausalWindowAttention,
    Round9Model,
    build_internal_report,
)


def _inputs(batch: int = 3) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(batch, 256),
        torch.randn(batch, 5, 256),
        torch.randn(batch, 5, 2),
        torch.randn(batch, 5, 2),
        torch.randn(batch, 2),
        torch.rand(batch, 5),
        torch.randn(batch, 8),
        torch.randn(batch, 5, 8),
    )


def test_round9_reference_starts_as_exact_base_prediction() -> None:
    model = Round9Model(256, METHODS["r8_reference"], physiology_dim=8)
    values = _inputs()
    result = model.static_forward(*values)
    assert result["prediction"].shape == (3, 2)
    assert result["pair_delta"].shape == (3, 5, 2)
    assert result["range_logits"].shape == (3, 6)
    assert torch.equal(result["prediction"], values[4])


@pytest.mark.parametrize("method", sorted(METHODS))
def test_all_round9_candidates_have_finite_forward_pass(method: str) -> None:
    model = Round9Model(256, METHODS[method], physiology_dim=8).eval()
    with torch.no_grad():
        result = model.static_forward(*_inputs())
    assert torch.isfinite(result["prediction"]).all()
    assert torch.isfinite(result["range_logits"]).all()


def test_causal_attention_cannot_see_future_inputs() -> None:
    torch.manual_seed(17)
    model = CausalWindowAttention(12, window=4).eval()
    original = torch.randn(2, 7, 12)
    changed = original.clone()
    changed[:, 4:] += 100.0
    gaps = torch.rand(2, 7)
    mask = torch.ones(2, 7, dtype=torch.bool)
    first = model(original, gaps, mask)
    second = model(changed, gaps, mask)
    assert torch.allclose(first[:, :4], second[:, :4], atol=0.0, rtol=0.0)


def _write_run(root: Path, method: str, offset: float) -> None:
    root.mkdir()
    rows = []
    for source, participant, target_sbp, target_dbp in (
        ("MIMIC", "MIMIC:p1", 120.0, 75.0),
        ("VitalDB", "VitalDB:p2", 130.0, 80.0),
    ):
        rows.append(
            {
                "subject_uid": participant,
                "event_id": f"{participant}:6",
                "k": 5,
                "source": source,
                "target_sbp": target_sbp,
                "target_dbp": target_dbp,
                "pred_sbp": target_sbp + offset,
                "pred_dbp": target_dbp + offset,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "selection_predictions.parquet", index=False)
    (root / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "split": "meta_train_internal_fold4",
                "seed": 20260823,
                "meta_validation_used_for_training": False,
                "meta_validation_used_for_early_stopping": False,
                "meta_validation_used_for_candidate_ranking": False,
                "meta_validation_predictions_generated": False,
                "locked_test_accessed": False,
                "method": method,
            }
        ),
        encoding="utf-8",
    )


def test_internal_report_uses_fold4_and_all_three_scopes(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_run(reference, "r8_reference", offset=2.0)
    _write_run(candidate, "adaptive_fusion", offset=1.0)
    output = tmp_path / "report"
    result = build_internal_report(
        runs={"R9-0 Reference": reference, "R9-1 Adaptive fusion": candidate},
        reference="R9-0 Reference",
        output=output,
        expected_seed=20260823,
    )
    table = pd.read_csv(output / "participant_macro_internal.csv")
    assert set(table["Scope"]) == {"Overall", "MIMIC", "VitalDB"}
    assert result["winner"] == "R9-1 Adaptive fusion"
    assert result["passes_internal_gate"] is True


def test_internal_report_rejects_meta_validation_use(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, "r8_reference", offset=1.0)
    payload = json.loads((run / "run.json").read_text())
    payload["meta_validation_used_for_candidate_ranking"] = True
    (run / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="unsafe meta-validation"):
        build_internal_report(
            runs={"Reference": run},
            reference="Reference",
            output=tmp_path / "report",
            expected_seed=20260823,
        )
