"""Personal-history fusion under the separate official CalBased protocol.

The relation network is a reused experimental component, not a new name for
LoRA. TrustFusion is a prospective candidate whose novelty is unverified. Its
inputs contain no query target; supervision belongs only in the trainer.
"""
from __future__ import annotations

import torch
from torch import nn

from .personal_memory_models import PersonalMemoryRelation, reference_prediction


class TrustFusion(nn.Module):
    """Learn a bounded adjustment to the fixed memory trust, in mmHg space.

    A zero final layer exactly reproduces the existing fixed blend, including
    alpha=0 or alpha=1. The learned value moves by at most 0.5 and is clipped
    to the unit interval. Insufficient-reference queries return LoRA unchanged.
    A scalar head shares trust for SBP/DBP; a vector head may distinguish them.
    """

    def __init__(self, input_dim: int, *, per_bp: bool = False, hidden_dim: int = 32):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("positive model dimensions required")
        self.per_bp = per_bp
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2 if per_bp else 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, base_bp, memory_bp, fixed_alpha, legal):
        if features.ndim != 2 or base_bp.shape != (len(features), 2):
            raise ValueError("features [N,D] and base BP [N,2] required")
        if memory_bp.shape != base_bp.shape or fixed_alpha.shape != (len(features),):
            raise ValueError("memory/alpha shape mismatch")
        if legal.shape != fixed_alpha.shape:
            raise ValueError("one legal-reference flag per query required")
        original = fixed_alpha[:, None]
        change = 0.5 * torch.tanh(self.network(features))
        # Clamp retains a useful initial gradient at an endpoint; a multiplier
        # of (1-alpha) would make a zero-initialized alpha=1 query untrainable.
        alpha = (original + change).clamp(0, 1)
        alpha = alpha * legal[:, None].to(alpha.dtype)
        return base_bp + alpha * (memory_bp - base_bp), alpha


__all__ = ["PersonalMemoryRelation", "reference_prediction", "TrustFusion"]
