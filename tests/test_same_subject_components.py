from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pulsedb_fewshot.same_subject_component_train import (
    _even_support_rows,
    apply_training_rule,
    fit_quality_proxy,
)
from pulsedb_fewshot.same_subject_components import (
    COMPONENTS,
    DESCRIPTOR_DIM,
    SUPPORT_COUNT,
    SameSubjectComponentRegressor,
    waveform_descriptor,
)


def _quality_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_uid": ["MIMIC:a"] * 10 + ["VitalDB:b"] * 10,
            "segment_uid": [f"s{i}" for i in range(20)],
            "source": ["MIMIC"] * 10 + ["VitalDB"] * 10,
            "ppg_f_std": np.exp(np.linspace(-1.0, 1.0, 20)),
        }
    )


def test_component_registry_is_single_factor_and_has_requested_families() -> None:
    assert len(COMPONENTS) == 19
    assert set(COMPONENTS) >= {
        "residual_reference",
        "residual_quality_gate",
        "residual_calibration_relative",
        "residual_support_attention",
        "residual_support_reliability",
        "residual_film",
        "residual_subject_lora_rank4",
        "residual_patch_transformer",
        "residual_conformer",
        "residual_cnn_bilstm",
        "residual_cnn_gru",
        "residual_soft_moe",
        "residual_prototype_moe",
        "residual_demographics_direct",
        "residual_beat_similarity_filter",
    }


def test_waveform_descriptor_is_finite_and_input_only_shape() -> None:
    ppg = torch.randn(3, 1, 1250)
    descriptor = waveform_descriptor(ppg)
    assert descriptor.shape == (3, DESCRIPTOR_DIM)
    assert torch.isfinite(descriptor).all()


@pytest.mark.parametrize(
    "candidate",
    [
        "residual_reference",
        "residual_quality_gate",
        "residual_calibration_relative",
        "residual_support_attention",
        "residual_support_reliability",
        "residual_film",
        "residual_multi_event_weighting",
        "residual_subject_lora_rank4",
        "residual_soft_moe",
        "residual_prototype_moe",
        "residual_demographics_direct",
    ],
)
def test_component_forward_is_finite_and_starts_at_anchor(candidate: str) -> None:
    spec = COMPONENTS[candidate]
    model = SameSubjectComponentRegressor(spec, subject_count=3).eval()
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
    if spec.uses_demographics:
        kwargs["demographics"] = torch.randn(batch, 5)
    output = model(ppg, anchor, **kwargs)
    assert output.shape == (batch, 2)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, anchor, atol=1e-6)


def test_support_rows_are_input_order_selected_without_bp() -> None:
    frame = pd.DataFrame(
        {
            "selection_rank": np.arange(20)[::-1],
            "segment_uid": [f"event-{i:02d}" for i in range(20)],
            "sbp": np.linspace(80, 200, 20),
        }
    )
    selected = _even_support_rows(frame)
    assert len(selected) == SUPPORT_COUNT + 1
    assert selected["selection_rank"].is_monotonic_increasing
    assert selected["segment_uid"].is_unique


def test_quality_rules_fit_only_input_proxy_and_preserve_full_validation_contract() -> None:
    frame, parameters = fit_quality_proxy(_quality_frame())
    assert set(parameters) == {"MIMIC", "VitalDB"}
    assert frame["quality_weight"].between(0.05, 1.0).all()
    filtered, details = apply_training_rule(
        frame, "quality_filter", beat_similarity_path=None
    )
    assert 0 < len(filtered) < len(frame)
    assert details["validation_coverage_changed"] is False


def test_similarity_filter_requires_complete_train_input_table(tmp_path: Path) -> None:
    frame, _ = fit_quality_proxy(_quality_frame())
    similarity = frame[["subject_uid", "segment_uid"]].copy()
    similarity["valid"] = True
    similarity["median_pairwise_similarity"] = np.linspace(0.80, 0.99, len(frame))
    path = tmp_path / "similarity.parquet"
    similarity.to_parquet(path, index=False)
    filtered, details = apply_training_rule(
        frame, "beat_similarity_filter", beat_similarity_path=path
    )
    assert filtered["median_pairwise_similarity"].ge(0.90).all()
    assert details["validation_coverage_changed"] is False

