import numpy as np

from pulsedb_fewshot.baselines import (
    last_cuff_prediction,
    residual_offset_prediction,
    support_mean_prediction,
)


def test_simple_calibration_baselines() -> None:
    support = np.array([[120.0, 70.0], [130.0, 80.0]])
    assert np.allclose(last_cuff_prediction(support, 2), [[130, 80], [130, 80]])
    assert np.allclose(support_mean_prediction(support, 1), [[125, 75]])


def test_residual_offset_uses_only_support_residual() -> None:
    support = np.array([[120.0, 70.0], [130.0, 80.0]])
    support_population = np.array([[115.0, 68.0], [127.0, 77.0]])
    query_population = np.array([[125.0, 72.0]])
    prediction = residual_offset_prediction(support, support_population, query_population)
    assert np.allclose(prediction, [[129.0, 74.5]])
