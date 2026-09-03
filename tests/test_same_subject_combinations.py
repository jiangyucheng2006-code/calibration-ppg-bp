import pytest
import json
from pathlib import Path

import pandas as pd
import torch

from pulsedb_fewshot.calbased_screen import PROTOCOL_ID
from pulsedb_fewshot.same_subject_combination_report import build_final_report
from pulsedb_fewshot.models import model_parameter_counts
from pulsedb_fewshot.same_subject_combinations import (
    CALIBRATION_RELATIVE,
    COMBINATIONS,
    FILM,
    MULTI_EVENT_WEIGHTING,
    SUBJECT_LORA,
    SUPPORT_ATTENTION,
    SUPPORT_RELIABILITY,
    SameSubjectCombinationRegressor,
)
from pulsedb_fewshot.same_subject_components import (
    COMPONENTS,
    DESCRIPTOR_DIM,
    SUPPORT_COUNT,
    SameSubjectComponentRegressor,
    waveform_descriptor,
)


def test_combination_matrix_is_bounded_and_covers_the_six_selected_modules() -> None:
    assert len(COMBINATIONS) == 15
    assert set(COMBINATIONS["lora_all_six"].modules) == {
        SUBJECT_LORA,
        FILM,
        SUPPORT_ATTENTION,
        MULTI_EVENT_WEIGHTING,
        SUPPORT_RELIABILITY,
        CALIBRATION_RELATIVE,
    }
    assert all(spec.modules[0] == SUBJECT_LORA for spec in COMBINATIONS.values())
    assert any(len(spec.modules) == 2 for spec in COMBINATIONS.values())
    assert any(len(spec.modules) == 3 for spec in COMBINATIONS.values())
    assert any(len(spec.modules) == 4 for spec in COMBINATIONS.values())
    assert any(len(spec.modules) == 6 for spec in COMBINATIONS.values())


def test_lora_reference_matches_the_isolated_model_parameterization() -> None:
    isolated = SameSubjectComponentRegressor(
        COMPONENTS["residual_subject_lora_rank4"], subject_count=3
    )
    combination = SameSubjectCombinationRegressor(COMBINATIONS["lora"], subject_count=3)
    assert isolated.state_dict().keys() == combination.state_dict().keys()
    assert model_parameter_counts(isolated) == model_parameter_counts(combination)


@pytest.mark.parametrize("candidate", list(COMBINATIONS))
def test_combination_forward_backward_is_finite_and_starts_at_anchor(
    candidate: str,
) -> None:
    torch.manual_seed(17)
    spec = COMBINATIONS[candidate]
    model = SameSubjectCombinationRegressor(spec, subject_count=3)
    batch = 2
    ppg = torch.randn(batch, 1, 256)
    anchor = torch.randn(batch, 2)
    kwargs: dict[str, torch.Tensor] = {
        "query_descriptor": waveform_descriptor(ppg),
        "subject_index": torch.tensor([0, 1]),
    }
    if spec.uses_support:
        kwargs["support_descriptors"] = torch.randn(
            batch, SUPPORT_COUNT, DESCRIPTOR_DIM
        )
        kwargs["support_bp"] = torch.randn(batch, SUPPORT_COUNT, 2)
    output = model(ppg, anchor, **kwargs)
    assert output.shape == (batch, 2)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, anchor, atol=1e-6)
    loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


def test_combination_requires_seen_subject_index() -> None:
    model = SameSubjectCombinationRegressor(COMBINATIONS["lora"], subject_count=3)
    with pytest.raises(ValueError, match="seen-subject index"):
        model(torch.randn(2, 1, 256), torch.randn(2, 2))


def _write_split_report(
    root: Path,
    mode: str,
    *,
    promoted_random_only: bool = False,
) -> None:
    root.mkdir()
    (root / "selection.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "split_mode": mode,
                "seed": 20260902,
                "heldout_test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for candidate in COMBINATIONS:
        for view in ("Overall", "MIMIC", "VitalDB"):
            score = 3.5 if mode == "chronological_blocked" else 3.1
            if candidate == "lora_film":
                score -= 0.20
                if promoted_random_only and mode == "chronological_blocked":
                    score += 0.25
            elif candidate != "lora":
                score += 0.10
            rows.append(
                {
                    "candidate": candidate,
                    "runner": "combination_residual",
                    "view": view,
                    "mean_mae": score,
                    "sbp_mae": score + 1.0,
                    "dbp_mae": score - 1.0,
                }
            )
    pd.DataFrame(rows).to_csv(root / "participant_macro_summary.csv", index=False)


def test_final_report_promotes_only_a_both_mode_both_source_gain(tmp_path: Path) -> None:
    random_report = tmp_path / "random"
    chronological_report = tmp_path / "chronological"
    _write_split_report(random_report, "random_disjoint")
    _write_split_report(chronological_report, "chronological_blocked")
    result = build_final_report(random_report, chronological_report, tmp_path / "out")
    assert result["winner"]["candidate"] == "lora_film"
    assert result["winner"]["passes_robust_gate"] is True
    assert result["heldout_test_accessed"] is False


def test_final_report_retains_lora_when_temporal_gain_fails(tmp_path: Path) -> None:
    random_report = tmp_path / "random"
    chronological_report = tmp_path / "chronological"
    _write_split_report(random_report, "random_disjoint", promoted_random_only=True)
    _write_split_report(
        chronological_report,
        "chronological_blocked",
        promoted_random_only=True,
    )
    result = build_final_report(random_report, chronological_report, tmp_path / "out")
    assert result["winner"]["candidate"] == "lora"
