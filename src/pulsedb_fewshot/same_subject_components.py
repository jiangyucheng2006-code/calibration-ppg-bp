"""Isolated component models for the seen-subject PulseDB analogue.

Every candidate keeps the train-label-only participant mean anchor used by
``subject_mean_residual_ppg``.  A candidate changes one mechanism at a time;
the held-out role is intentionally outside this module's data contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .models import build_ppg_encoder


SCREEN_ID = "same-subject-single-component-v1"
REFERENCE_CANDIDATE = "residual_reference"
SUPPORT_COUNT = 5
SUPPORT_RESERVE = 1
DESCRIPTOR_DIM = 10
PROMOTION_MARGIN_MMHG = 0.15


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    backbone: str
    adapter: str
    uses_support: bool = False
    uses_demographics: bool = False
    uses_subject_index: bool = False
    training_rule: str = "standard"
    description: str = ""


COMPONENT_SPECS = (
    ComponentSpec(
        REFERENCE_CANDIDATE,
        "resnet_small",
        "none",
        description="Paired rerun of compact ResNet plus train-role subject mean.",
    ),
    ComponentSpec(
        "residual_quality_gate",
        "resnet_small",
        "quality_gate",
        description="PPG-only soft shrinkage of the learned residual.",
    ),
    ComponentSpec(
        "residual_quality_weighted_loss",
        "resnet_small",
        "none",
        training_rule="quality_weighted_loss",
        description="Input-only robust PPG-amplitude quality weights in Huber loss.",
    ),
    ComponentSpec(
        "residual_ppg_quality_filter",
        "resnet_small",
        "none",
        training_rule="quality_filter",
        description="Remove only the lowest train-role PPG quality decile.",
    ),
    ComponentSpec(
        "residual_calibration_relative",
        "resnet_small",
        "calibration_relative",
        uses_support=True,
        description="Add train-support morphology/BP-relative correction.",
    ),
    ComponentSpec(
        "residual_support_attention",
        "resnet_small",
        "support_attention",
        uses_support=True,
        description="Query-conditioned attention over five train-role supports.",
    ),
    ComponentSpec(
        "residual_support_reliability",
        "resnet_small",
        "support_reliability",
        uses_support=True,
        description="Query-independent reliability weights over five supports.",
    ),
    ComponentSpec(
        "residual_film",
        "resnet_small",
        "film",
        uses_support=True,
        description="Train-support-conditioned FiLM modulation of query features.",
    ),
    ComponentSpec(
        "residual_multi_event_weighting",
        "resnet_small",
        "multi_event_weighting",
        uses_support=True,
        description="Adaptive weighting of five train-role reference-BP events.",
    ),
    ComponentSpec(
        "residual_subject_lora_rank4",
        "resnet_small",
        "subject_lora",
        uses_subject_index=True,
        description="Seen-subject rank-4 feature adapter; not an unseen-subject claim.",
    ),
    ComponentSpec(
        "residual_inception_time_wide",
        "inception_time_wide",
        "none",
        description="Wide InceptionTime encoder under the winning residual formulation.",
    ),
    ComponentSpec(
        "residual_patch_transformer",
        "patch_transformer",
        "none",
        description="Patch Transformer encoder under the same residual formulation.",
    ),
    ComponentSpec(
        "residual_conformer",
        "conformer",
        "none",
        description="Conformer encoder under the same residual formulation.",
    ),
    ComponentSpec(
        "residual_cnn_bilstm",
        "cnn_bilstm_adaptation",
        "none",
        description="PPG-only CNN-BiLSTM encoder under the same residual formulation.",
    ),
    ComponentSpec(
        "residual_cnn_gru",
        "bp_crnn",
        "none",
        description="Compact CNN-GRU encoder under the same residual formulation.",
    ),
    ComponentSpec(
        "residual_soft_moe",
        "resnet_small",
        "soft_moe",
        description="Four learned residual experts with a soft PPG gate.",
    ),
    ComponentSpec(
        "residual_prototype_moe",
        "resnet_small",
        "prototype_moe",
        description="Four waveform-feature prototypes route four residual experts.",
    ),
    ComponentSpec(
        "residual_demographics_direct",
        "resnet_small",
        "demographics",
        uses_demographics=True,
        description="Directly concatenate cleaned age/sex features to the residual head.",
    ),
    ComponentSpec(
        "residual_beat_similarity_filter",
        "resnet_small",
        "none",
        training_rule="beat_similarity_filter",
        description="Train only on similarity >=0.90; validation remains full coverage.",
    ),
)
COMPONENTS = {spec.name: spec for spec in COMPONENT_SPECS}


def waveform_descriptor(ppg: torch.Tensor) -> torch.Tensor:
    """Return input-only morphology/quality descriptors for z-scored PPG.

    ``ppg`` may have shape ``[..., 1, time]``.  The descriptor deliberately
    avoids reference ABP, BP targets, source, participant identity, and future
    windows.
    """

    if ppg.shape[-2] != 1 or ppg.shape[-1] < 4:
        raise ValueError("PPG descriptor expects [..., 1, time] with time >= 4")
    values = ppg.squeeze(-2).float()
    differences = values[..., 1:] - values[..., :-1]
    centered = values - values.mean(dim=-1, keepdim=True)
    scale = values.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    standardized = centered / scale
    skew = standardized.pow(3).mean(dim=-1)
    kurtosis = standardized.pow(4).mean(dim=-1) - 3.0

    def autocorrelation(lag: int) -> torch.Tensor:
        if values.shape[-1] <= lag:
            return torch.zeros_like(skew)
        left = standardized[..., :-lag]
        right = standardized[..., lag:]
        return (left * right).mean(dim=-1)

    descriptor = torch.stack(
        [
            values.amin(dim=-1),
            values.amax(dim=-1),
            differences.abs().mean(dim=-1),
            differences.std(dim=-1, unbiased=False),
            differences.abs().amax(dim=-1),
            skew,
            kurtosis,
            autocorrelation(25),
            autocorrelation(50),
            autocorrelation(100),
        ],
        dim=-1,
    )
    return torch.nan_to_num(descriptor, nan=0.0, posinf=10.0, neginf=-10.0).clamp(
        -10.0, 10.0
    )


def robust_quality_score(ppg_std: torch.Tensor, center: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Train-fitted input-only amplitude proxy in ``(0, 1]``."""

    log_std = torch.log(ppg_std.float().clamp_min(1e-8))
    robust_z = (log_std - center.float()) / scale.float().clamp_min(1e-6)
    return torch.exp(-robust_z.abs() / 3.0).clamp(0.05, 1.0)


class _ZeroResidualHead(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        final = self[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)


class SameSubjectComponentRegressor(nn.Module):
    """Participant-mean residual model with exactly one optional component."""

    def __init__(
        self,
        spec: ComponentSpec,
        *,
        subject_count: int,
        demographic_dim: int = 5,
        expert_count: int = 4,
        lora_rank: int = 4,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = build_ppg_encoder(spec.backbone, feature_dim=256)
        dimension = int(self.encoder.feature_dim)
        head_input = dimension + (demographic_dim if spec.uses_demographics else 0)
        self.residual_head = _ZeroResidualHead(head_input, dimension // 2)

        if spec.adapter == "quality_gate":
            self.quality_gate = nn.Sequential(
                nn.Linear(dimension + DESCRIPTOR_DIM, dimension // 2),
                nn.SiLU(),
                nn.Linear(dimension // 2, 2),
            )
            final = self.quality_gate[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, 4.0)
        elif spec.adapter == "film":
            self.film = nn.Linear(DESCRIPTOR_DIM + 2, dimension * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        elif spec.adapter in {
            "calibration_relative",
            "support_attention",
            "support_reliability",
            "multi_event_weighting",
        }:
            if spec.adapter == "support_attention":
                self.support_scorer = nn.Sequential(
                    nn.Linear(DESCRIPTOR_DIM * 3, 32), nn.SiLU(), nn.Linear(32, 1)
                )
            elif spec.adapter in {"support_reliability", "multi_event_weighting"}:
                score_dim = DESCRIPTOR_DIM + (2 if spec.adapter == "multi_event_weighting" else 0)
                self.support_scorer = nn.Sequential(
                    nn.Linear(score_dim, 32), nn.SiLU(), nn.Linear(32, 1)
                )
            self.support_correction = _ZeroResidualHead(
                dimension + DESCRIPTOR_DIM + 2,
                dimension // 2,
            )
            if spec.adapter == "multi_event_weighting":
                self.anchor_scale = nn.Parameter(torch.zeros(2))
        elif spec.adapter == "subject_lora":
            if subject_count < 1:
                raise ValueError("subject LoRA requires at least one seen participant")
            self.lora_rank = int(lora_rank)
            self.lora_a = nn.Embedding(subject_count, dimension * self.lora_rank)
            self.lora_b = nn.Embedding(subject_count, dimension * self.lora_rank)
            nn.init.normal_(self.lora_a.weight, std=0.01)
            nn.init.zeros_(self.lora_b.weight)
        elif spec.adapter in {"soft_moe", "prototype_moe"}:
            self.residual_head = nn.Identity()
            self.experts = nn.ModuleList(
                [_ZeroResidualHead(dimension, dimension // 2) for _ in range(expert_count)]
            )
            if spec.adapter == "soft_moe":
                self.expert_gate = nn.Sequential(
                    nn.Linear(dimension, dimension // 2),
                    nn.SiLU(),
                    nn.Linear(dimension // 2, expert_count),
                )
            else:
                self.prototypes = nn.Parameter(torch.empty(expert_count, dimension))
                nn.init.normal_(self.prototypes, std=1.0 / math.sqrt(dimension))

    @staticmethod
    def _validate_support(
        support_descriptors: torch.Tensor | None,
        support_bp: torch.Tensor | None,
        batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if support_descriptors is None or support_bp is None:
            raise ValueError("this component requires train-role support descriptors and BP")
        if support_descriptors.shape != (batch, SUPPORT_COUNT, DESCRIPTOR_DIM):
            raise ValueError("support descriptor shape is invalid")
        if support_bp.shape != (batch, SUPPORT_COUNT, 2):
            raise ValueError("support BP shape is invalid")
        return support_descriptors, support_bp

    def forward(
        self,
        ppg: torch.Tensor,
        subject_train_mean: torch.Tensor,
        *,
        query_descriptor: torch.Tensor | None = None,
        support_descriptors: torch.Tensor | None = None,
        support_bp: torch.Tensor | None = None,
        subject_index: torch.Tensor | None = None,
        demographics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if subject_train_mean.ndim != 2 or subject_train_mean.shape[1] != 2:
            raise ValueError("subject_train_mean must have shape [batch, 2]")
        features = self.encoder(ppg)
        batch, dimension = features.shape
        descriptor = query_descriptor if query_descriptor is not None else waveform_descriptor(ppg)
        if descriptor.shape != (batch, DESCRIPTOR_DIM):
            raise ValueError("query descriptor shape is invalid")

        if self.spec.adapter == "subject_lora":
            if subject_index is None or subject_index.shape != (batch,):
                raise ValueError("subject LoRA requires one seen-subject index per query")
            a = self.lora_a(subject_index).reshape(batch, self.lora_rank, dimension)
            b = self.lora_b(subject_index).reshape(batch, dimension, self.lora_rank)
            low_rank = torch.bmm(a, features.unsqueeze(-1))
            features = features + torch.bmm(b, low_rank).squeeze(-1) / self.lora_rank

        if self.spec.adapter in {"soft_moe", "prototype_moe"}:
            expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
            if self.spec.adapter == "soft_moe":
                logits = self.expert_gate(features)
            else:
                normalized = nn.functional.normalize(features, dim=-1)
                prototypes = nn.functional.normalize(self.prototypes, dim=-1)
                logits = 4.0 * normalized @ prototypes.transpose(0, 1)
            residual = torch.sum(torch.softmax(logits, dim=1)[..., None] * expert_outputs, dim=1)
            return subject_train_mean + residual

        support_mean_descriptor = None
        support_mean_bp = None
        if self.spec.uses_support:
            support_descriptors, support_bp = self._validate_support(
                support_descriptors, support_bp, batch
            )
            support_mean_descriptor = support_descriptors.mean(dim=1)
            support_mean_bp = support_bp.mean(dim=1)

        if self.spec.adapter == "film":
            assert support_mean_descriptor is not None and support_mean_bp is not None
            context = torch.cat([support_mean_descriptor, support_mean_bp], dim=-1)
            gamma, beta = self.film(context).chunk(2, dim=-1)
            features = (1.0 + gamma) * features + beta

        head_inputs = [features]
        if self.spec.uses_demographics:
            if demographics is None or demographics.shape != (batch, 5):
                raise ValueError("demographic component requires five cleaned features")
            head_inputs.append(demographics)
        residual = self.residual_head(torch.cat(head_inputs, dim=-1))

        if self.spec.adapter == "quality_gate":
            gate = torch.sigmoid(self.quality_gate(torch.cat([features, descriptor], dim=-1)))
            residual = gate * residual
        elif self.spec.adapter in {
            "calibration_relative",
            "support_attention",
            "support_reliability",
            "multi_event_weighting",
        }:
            assert support_descriptors is not None and support_bp is not None
            expanded_query = descriptor[:, None, :].expand_as(support_descriptors)
            if self.spec.adapter == "support_attention":
                logits = self.support_scorer(
                    torch.cat(
                        [expanded_query, support_descriptors, (expanded_query - support_descriptors).abs()],
                        dim=-1,
                    )
                ).squeeze(-1)
            elif self.spec.adapter in {"support_reliability", "multi_event_weighting"}:
                score_inputs = [support_descriptors]
                if self.spec.adapter == "multi_event_weighting":
                    score_inputs.append(support_bp - subject_train_mean[:, None, :])
                logits = self.support_scorer(torch.cat(score_inputs, dim=-1)).squeeze(-1)
            else:
                logits = torch.zeros(
                    batch, SUPPORT_COUNT, dtype=features.dtype, device=features.device
                )
            weights = torch.softmax(logits, dim=1)
            support_context = torch.sum(weights[..., None] * support_descriptors, dim=1)
            weighted_bp = torch.sum(weights[..., None] * support_bp, dim=1)
            relative_inputs = torch.cat(
                [features, descriptor - support_context, weighted_bp - subject_train_mean],
                dim=-1,
            )
            residual = residual + self.support_correction(relative_inputs)
            if self.spec.adapter == "multi_event_weighting":
                residual = residual + self.anchor_scale * (weighted_bp - subject_train_mean)
        return subject_train_mean + residual

