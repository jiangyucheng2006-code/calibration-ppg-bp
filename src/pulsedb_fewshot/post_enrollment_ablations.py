"""Fixed prediction-time controls and participant-paired uncertainty estimates.

Memory controls accept registration BP and label-free neighbor state only.
Bootstrap intervals are conditional on a fitted model, not training-seed CIs.
"""
from __future__ import annotations

import numpy as np


def memory_variant(bank_bp, query_base, state, *, mode="distance_softmax"):
    bank = np.asarray(bank_bp)
    base = np.asarray(query_base)
    ids = np.asarray(state["knn_indices"])
    weights = np.asarray(state["knn_weights"])
    valid = np.asarray(state["valid"], dtype=bool)
    gate = np.asarray(state["support_weight"])
    if (bank.ndim != 2 or bank.shape[1] != 2 or base.shape != (len(ids), 2)
            or ids.shape != weights.shape or ids.ndim != 2 or ids.shape[1] != 5
            or valid.shape != (len(ids),) or gate.shape != valid.shape):
        raise ValueError("invalid memory input shapes")
    if not all(np.isfinite(x).all() for x in (bank, base, weights, gate)):
        raise ValueError("nonfinite memory inputs")
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids[valid] < 0) or np.any(ids >= len(bank)):
        raise ValueError("invalid legal donor indices")
    if np.any(weights < 0) or np.any(gate < 0) or np.any(gate > 1):
        raise ValueError("invalid memory weights or gate")
    if not np.allclose(weights[valid].sum(1), 1., atol=1e-6):
        raise ValueError("valid donor weights must sum to one")
    if mode == "distance_uniform":
        weights = np.where(valid[:, None], np.full_like(weights, .2), 0.)
        alpha = gate
    elif mode == "fixed_half":
        alpha = valid.astype(float) * .5
    elif mode == "memory_only":
        alpha = valid.astype(float)
    elif mode == "distance_softmax":
        alpha = gate
    else:
        raise ValueError("unknown prespecified memory control")
    memory = (bank[np.maximum(ids, 0)] * weights[..., None]).sum(1)
    memory[~valid] = base[~valid]
    alpha = np.where(valid, alpha, 0.)[:, None]
    return base + alpha * (memory - base)


def paired_stratified_interval(gains, sources, *, replicates=2000, seed=20260909):
    """Positive gain means reference participant MAE minus candidate MAE.

    Resample participants with replacement independently inside each source,
    preserving source sizes. Never resample individual 10-second windows.
    """
    gains, sources = np.asarray(gains, float), np.asarray(sources)
    if gains.ndim != 1 or sources.shape != gains.shape or len(gains) < 2 or not np.isfinite(gains).all():
        raise ValueError("paired bootstrap needs finite gains for at least two participants")
    if replicates != 2000 or seed != 20260909:
        raise ValueError("uncertainty settings are fixed before evaluation")
    rng = np.random.default_rng(seed)
    values = np.zeros(replicates)
    for source in sorted(set(sources)):
        stratum = gains[sources == source]
        draws = rng.integers(0, len(stratum), size=(replicates, len(stratum)))
        values += stratum[draws].sum(1) / len(gains)
    low, high = np.quantile(values, [.025, .975])
    return {"gain_mmHg": float(gains.mean()), "ci95_low_mmHg": float(low),
            "ci95_high_mmHg": float(high), "participants": len(gains),
            "improved_participants": int((gains > 0).sum()),
            "bootstrap_replicates": replicates}
