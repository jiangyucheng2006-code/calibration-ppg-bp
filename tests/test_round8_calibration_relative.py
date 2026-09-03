import numpy as np
import pytest


torch = pytest.importorskip("torch")

from pulsedb_fewshot.round8_calibration_relative import (  # noqa: E402
    CalibrationRelativeModel,
    TimeDecayGRU,
    _physiology_features,
    _range_labels,
)


def test_physio_features_are_finite_and_target_free_shape() -> None:
    time = torch.linspace(0, 20 * torch.pi, 1250)
    waveforms = torch.stack([torch.sin(time), torch.cos(time)])[:, None, :]
    features = _physiology_features(waveforms)
    assert features.shape == (2, 8)
    assert torch.isfinite(features).all()


def test_range_labels_use_only_support_minimum_and_maximum() -> None:
    support = np.array(
        [
            [[110.0, 70.0], [120.0, 80.0], [130.0, 90.0]],
            [[100.0, 60.0], [110.0, 70.0], [120.0, 80.0]],
        ],
        dtype=np.float32,
    )
    target = np.array([[105.0, 85.0], [125.0, 55.0]], dtype=np.float32)
    assert _range_labels(target, support).tolist() == [[0, 1], [2, 0]]


def test_static_model_starts_as_exact_base_prediction() -> None:
    model = CalibrationRelativeModel(
        256, temporal=False, range_auxiliary=True, physiology_dim=8
    )
    batch = 3
    base = torch.randn(batch, 2)
    prediction, pair_delta, hidden, logits = model.static_forward(
        torch.randn(batch, 256),
        torch.randn(batch, 5, 256),
        torch.randn(batch, 5, 2),
        torch.randn(batch, 5, 2),
        base,
        torch.rand(batch, 5),
        torch.randn(batch, 8),
        torch.randn(batch, 5, 8),
    )
    assert prediction.shape == (batch, 2)
    assert pair_delta.shape == (batch, 5, 2)
    assert hidden.shape == (batch, 64)
    assert logits is not None and logits.shape == (batch, 6)
    assert torch.equal(prediction, base)


def test_time_decay_gru_is_causal() -> None:
    torch.manual_seed(7)
    model = TimeDecayGRU(6).eval()
    original = torch.randn(2, 5, 6)
    changed = original.clone()
    changed[:, 3:] += 100.0
    gaps = torch.ones(2, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    first = model(original, gaps, mask)
    second = model(changed, gaps, mask)
    assert torch.allclose(first[:, :3], second[:, :3], atol=0.0, rtol=0.0)
