"""Antisymmetric local BP relations on frozen, personal-LoRA features.

These are experimental reference-conditioned networks, not a replacement or
renaming of LoRA. Neither this model nor its prediction API accepts query BP.
The original personal model and the training-only measurement bank are frozen.
"""

from __future__ import annotations

import torch
from torch import nn


SCREEN_ID = "personal-memory-v1"
CANDIDATES = ("pair_single", "pair_uniform", "pair_retrieved", "pair_distance_blend")


class PersonalMemoryRelation(nn.Module):
    """Predict (query BP - reference BP) / train BP standard deviation.

    One shared G is evaluated in both orders. Antisymmetry is a structural
    constraint, not a claim of physiological causality or cycle consistency.
    """

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("positive feature and hidden dimensions required")
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.Linear(4 * feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)
        )
        # Epoch zero is reference-BP interpolation, not an uncontrolled random
        # mmHg extrapolation. All candidates use this same initialization.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _ordered(self, query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat(
            (query, reference, query - reference, query * reference), dim=-1
        ))

    def forward(self, query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if query.shape != reference.shape or query.shape[-1] != self.feature_dim:
            raise ValueError("query and reference require identical (..., feature_dim) shape")
        return (self._ordered(query, reference) - self._ordered(reference, query)) * 0.5


def reference_prediction(
    relation: PersonalMemoryRelation,
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    reference_bp: torch.Tensor,
    base_bp: torch.Tensor,
    weights: torch.Tensor,
    legal_reference: torch.Tensor,
    support_weight: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw-mmHg prediction and normalized per-reference BP deltas.

    Invalid references have zero weight. An all-invalid query returns its base
    prediction exactly, preserving coverage. There is deliberately no query BP
    parameter: targets belong exclusively in the separate supervised loss.
    """
    if query_features.ndim != 2 or reference_features.ndim != 3:
        raise ValueError("expected query [B,D] and references [B,K,D]")
    batch, count, dimension = reference_features.shape
    if query_features.shape != (batch, dimension):
        raise ValueError("query/reference dimensions disagree")
    if reference_bp.shape != (batch, count, 2) or base_bp.shape != (batch, 2):
        raise ValueError("BP shape mismatch")
    if weights.shape != (batch, count) or legal_reference.shape != weights.shape:
        raise ValueError("reference weights/mask shape mismatch")
    if support_weight.shape != (batch,) or target_std.shape != (2,):
        raise ValueError("support weight/scaler shape mismatch")
    query = query_features[:, None, :].expand_as(reference_features)
    delta = relation(query, reference_features)
    effective = weights * legal_reference.to(weights.dtype)
    total = effective.sum(dim=1, keepdim=True)
    normalized = effective / total.clamp_min(torch.finfo(effective.dtype).eps)
    local_bp = reference_bp + delta * target_std
    memory_bp = (normalized[:, :, None] * local_bp).sum(dim=1)
    alpha = support_weight * (total[:, 0] > 0).to(support_weight.dtype)
    # The difference form preserves the source BP exactly when alpha is zero.
    prediction = base_bp + alpha[:, None] * (memory_bp - base_bp)
    return prediction, delta


def supervised_relation_loss(
    prediction_bp: torch.Tensor,
    query_bp: torch.Tensor,
    delta: torch.Tensor,
    reference_bp: torch.Tensor,
    legal_reference: torch.Tensor,
    target_std: torch.Tensor,
    *,
    huber_delta: float = 0.5,
    pair_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training-only absolute error + normalized pair-delta Huber objective."""
    if huber_delta <= 0 or pair_weight < 0:
        raise ValueError("invalid Huber/pair loss configuration")
    absolute = nn.functional.huber_loss(
        prediction_bp / target_std, query_bp / target_std, delta=huber_delta
    )
    target_delta = (query_bp[:, None, :] - reference_bp) / target_std
    elementwise = nn.functional.huber_loss(
        delta, target_delta, delta=huber_delta, reduction="none"
    ).mean(dim=-1)
    mask = legal_reference.to(elementwise.dtype)
    pair = (elementwise * mask).sum() / mask.sum().clamp_min(1)
    return absolute + pair_weight * pair, absolute, pair
