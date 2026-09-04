import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from pulsedb_fewshot.calbased_screen import PROTOCOL_ID
from pulsedb_fewshot.same_subject_components import DESCRIPTOR_DIM, SUPPORT_COUNT, waveform_descriptor
from pulsedb_fewshot.same_subject_personal_profile_report import build_final_report
from pulsedb_fewshot.same_subject_personal_profiles import (
    PERSONAL_PROFILES,
    PRIMARY_CANDIDATE,
    REFERENCE_CANDIDATE,
    SameSubjectPersonalProfileRegressor,
)


def test_profile_matrix_contains_primary_references_and_ablation_controls() -> None:
    assert len(PERSONAL_PROFILES) == 8
    assert PRIMARY_CANDIDATE in PERSONAL_PROFILES
    assert REFERENCE_CANDIDATE in PERSONAL_PROFILES
    assert "residual_reference" in PERSONAL_PROFILES
    primary = PERSONAL_PROFILES[PRIMARY_CANDIDATE]
    assert primary.code_dim == 32
    assert primary.uses_support
    assert primary.uses_subject_index
    assert primary.use_reliability
    assert primary.participant_trainable_parameters == 34
    assert PERSONAL_PROFILES[REFERENCE_CANDIDATE].participant_trainable_parameters == 2048


@pytest.mark.parametrize("candidate", list(PERSONAL_PROFILES))
def test_profile_forward_backward_is_finite_and_starts_at_anchor(candidate: str) -> None:
    torch.manual_seed(23)
    spec = PERSONAL_PROFILES[candidate]
    model = SameSubjectPersonalProfileRegressor(spec, subject_count=3)
    ppg = torch.randn(2, 1, 256)
    anchor = torch.randn(2, 2)
    kwargs: dict[str, torch.Tensor] = {
        "query_descriptor": waveform_descriptor(ppg),
        "subject_index": torch.tensor([0, 1]),
    }
    if spec.uses_support:
        kwargs["support_descriptors"] = torch.randn(2, SUPPORT_COUNT, DESCRIPTOR_DIM)
        kwargs["support_bp"] = torch.randn(2, SUPPORT_COUNT, 2)
    output = model(ppg, anchor, **kwargs)
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, anchor, atol=1e-6)
    torch.nn.functional.mse_loss(output, torch.randn_like(output)).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() for g in gradients)


def test_profile_requires_subject_index_and_support() -> None:
    spec = PERSONAL_PROFILES[PRIMARY_CANDIDATE]
    model = SameSubjectPersonalProfileRegressor(spec, subject_count=3)
    ppg = torch.randn(2, 1, 256)
    anchor = torch.randn(2, 2)
    with pytest.raises(ValueError, match="support information"):
        model(ppg, anchor, subject_index=torch.tensor([0, 1]))
    support_descriptors = torch.randn(2, SUPPORT_COUNT, DESCRIPTOR_DIM)
    support_bp = torch.randn(2, SUPPORT_COUNT, 2)
    with pytest.raises(ValueError, match="seen-subject index"):
        model(
            ppg,
            anchor,
            support_descriptors=support_descriptors,
            support_bp=support_bp,
        )


def _write_report(root: Path, mode: str, primary_gain: float) -> None:
    root.mkdir()
    (root / "selection.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "split_mode": mode,
                "seed": 20260904,
                "heldout_test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for candidate in PERSONAL_PROFILES:
        for view in ("Overall", "MIMIC", "VitalDB"):
            value = 3.5
            if candidate == PRIMARY_CANDIDATE:
                value -= primary_gain
            elif candidate not in {REFERENCE_CANDIDATE, "residual_reference"}:
                value += 0.1
            if candidate == "residual_reference":
                value += 2.0
            rows.append(
                {
                    "candidate": candidate,
                    "runner": "personal_profile_residual",
                    "view": view,
                    "mean_mae": value,
                    "sbp_mae": value + 1.0,
                    "dbp_mae": value - 1.0,
                }
            )
    pd.DataFrame(rows).to_csv(root / "participant_macro_summary.csv", index=False)


def test_profile_final_report_uses_both_modes_and_sources(tmp_path: Path) -> None:
    random_report = tmp_path / "random"
    chronological_report = tmp_path / "chronological"
    _write_report(random_report, "random_disjoint", 0.2)
    _write_report(chronological_report, "chronological_blocked", 0.2)
    result = build_final_report(random_report, chronological_report, tmp_path / "out")
    assert result["primary_candidate"]["passes_robust_gate"] is True
    assert result["primary_candidate"]["candidate"] == PRIMARY_CANDIDATE
    assert result["heldout_test_accessed"] is False
