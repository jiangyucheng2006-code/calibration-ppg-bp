"""Analytical calibration baselines that never access query BP during prediction."""

from __future__ import annotations

import numpy as np


def population_mean_prediction(train_bp: np.ndarray, n_queries: int) -> np.ndarray:
    train_bp = np.asarray(train_bp, dtype=float)
    if train_bp.ndim != 2 or train_bp.shape[1] != 2:
        raise ValueError("train BP must have shape [n, 2]")
    return np.repeat(train_bp.mean(axis=0, keepdims=True), n_queries, axis=0)


def last_cuff_prediction(support_bp: np.ndarray, n_queries: int) -> np.ndarray:
    support_bp = np.asarray(support_bp, dtype=float)
    if support_bp.ndim != 2 or support_bp.shape[1] != 2 or len(support_bp) < 1:
        raise ValueError("support BP must have shape [K, 2] with K >= 1")
    return np.repeat(support_bp[-1:, :], n_queries, axis=0)


def support_mean_prediction(support_bp: np.ndarray, n_queries: int) -> np.ndarray:
    support_bp = np.asarray(support_bp, dtype=float)
    if support_bp.ndim != 2 or support_bp.shape[1] != 2 or len(support_bp) < 1:
        raise ValueError("support BP must have shape [K, 2] with K >= 1")
    return np.repeat(support_bp.mean(axis=0, keepdims=True), n_queries, axis=0)


def residual_offset_prediction(
    support_bp: np.ndarray,
    support_population_prediction: np.ndarray,
    query_population_prediction: np.ndarray,
) -> np.ndarray:
    support_bp = np.asarray(support_bp, dtype=float)
    support_population_prediction = np.asarray(support_population_prediction, dtype=float)
    query_population_prediction = np.asarray(query_population_prediction, dtype=float)
    if support_bp.shape != support_population_prediction.shape:
        raise ValueError("support targets and population predictions must align")
    if support_bp.ndim != 2 or support_bp.shape[1] != 2 or len(support_bp) < 1:
        raise ValueError("support arrays must have shape [K, 2]")
    if query_population_prediction.ndim != 2 or query_population_prediction.shape[1] != 2:
        raise ValueError("query population predictions must have shape [Q, 2]")
    offset = (support_bp - support_population_prediction).mean(axis=0)
    return query_population_prediction + offset
