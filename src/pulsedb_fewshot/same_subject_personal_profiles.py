"""Compact participant profiles for persistent same-subject personalization.

This module is deliberately limited to the development-only seen-participant
track.  Every personal item is indexed by a participant who is present in the
train role, while validation windows are disjoint from all train windows.
No held-out target is part of the model contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .models import build_ppg_encoder
from .same_subject_components import (
    COMPONENTS,
    DESCRIPTOR_DIM,
    SUPPORT_COUNT,
    SameSubjectComponentRegressor,
    _ZeroResidualHead,
    waveform_descriptor,
)


SCREEN_ID = "same-subject-personal-profile-v1"
REFERENCE_CANDIDATE = "subject_lora_rank4"
PRIMARY_CANDIDATE = "personal_profile_code32_reliability"
PROMOTION_MARGIN_MMHG = 0.15
PROFILE_SPLIT_MODES = ("random_disjoint", "chronological_blocked")


@dataclass(frozen=True)
class PersonalProfileSpec:
    name: str
    adapter: str
    description: str
    code_dim: int = 0
    uses_support: bool = False
    uses_subject_index: bool = False
    uses_demographics: bool = False
    training_rule: str = "standard"
    backbone: str = "resnet_small"
    use_dynamic: bool = True
    use_reliability: bool = False

    @property
    def participant_trainable_parameters(self) -> int:
        if self.adapter == "subject_lora_rank4":
            return 2 * 256 * 4
        if self.adapter.startswith("personal_profile") and self.uses_subject_index:
            return self.code_dim + 2
        return 0


PROFILE_SPECS = (
    PersonalProfileSpec(
        "residual_reference",
        "residual_reference",
        "Compact ResNet plus train-role participant BP mean; no personal state.",
    ),
    PersonalProfileSpec(
        REFERENCE_CANDIDATE,
        "subject_lora_rank4",
        "Paired rank-4 participant-indexed LoRA reference.",
        uses_subject_index=True,
    ),
    PersonalProfileSpec(
        "personal_profile_support_only",
        "personal_profile_support_only",
        "Data-derived train-role PPG/BP profile without free participant parameters.",
        code_dim=32,
        uses_support=True,
        use_reliability=True,
    ),
    PersonalProfileSpec(
        "personal_profile_code32_no_support",
        "personal_profile_code32_no_support",
        "Compact participant state without historical PPG support context.",
        code_dim=32,
        uses_subject_index=True,
    ),
    PersonalProfileSpec(
        "personal_profile_code32_no_reliability",
        "personal_profile_code32_no_reliability",
        "Full compact participant profile without reliability shrinkage.",
        code_dim=32,
        uses_support=True,
        uses_subject_index=True,
    ),
    PersonalProfileSpec(
        PRIMARY_CANDIDATE,
        "personal_profile_code32_reliability",
        "Primary compact profile: stable bias plus reliability-gated dynamic change.",
        code_dim=32,
        uses_support=True,
        uses_subject_index=True,
        use_reliability=True,
    ),
    PersonalProfileSpec(
        "personal_profile_code64_reliability",
        "personal_profile_code64_reliability",
        "Capacity control with a 64-dimensional participant state.",
        code_dim=64,
        uses_support=True,
        uses_subject_index=True,
        use_reliability=True,
    ),
    PersonalProfileSpec(
        "personal_profile_code32_stable_only",
        "personal_profile_code32_stable_only",
        "Stable participant correction without the query-dependent personal branch.",
        code_dim=32,
        uses_support=True,
        uses_subject_index=True,
        use_dynamic=False,
    ),
)
PERSONAL_PROFILES = {spec.name: spec for spec in PROFILE_SPECS}


class SameSubjectPersonalProfileRegressor(nn.Module):
    """Decompose BP into anchor, shared change, stable bias, and dynamic change."""

    def __init__(
        self,
        spec: PersonalProfileSpec,
        *,
        subject_count: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        if spec.adapter == "residual_reference":
            self.reference = SameSubjectComponentRegressor(
                COMPONENTS["residual_reference"], subject_count=subject_count
            )
            return
        if spec.adapter == "subject_lora_rank4":
            self.reference = SameSubjectComponentRegressor(
                COMPONENTS["residual_subject_lora_rank4"], subject_count=subject_count
            )
            return
        if spec.code_dim < 1:
            raise ValueError("personal-profile candidates require a positive code_dim")
        if spec.uses_subject_index and subject_count < 1:
            raise ValueError("personal profile requires at least one seen participant")

        self.encoder = build_ppg_encoder(spec.backbone, feature_dim=256)
        dimension = int(self.encoder.feature_dim)
        self.shared_residual = _ZeroResidualHead(dimension, dimension // 2)

        # Five train-role supports summarize morphology, its variability, the
        # support BP offset from the train-role mean, and BP variability.
        support_profile_dim = DESCRIPTOR_DIM * 2 + 4
        self.profile_encoder = nn.Sequential(
            nn.Linear(support_profile_dim, spec.code_dim),
            nn.SiLU(),
            nn.LayerNorm(spec.code_dim),
        )
        if spec.uses_subject_index:
            self.subject_code = nn.Embedding(subject_count, spec.code_dim)
            self.subject_bias = nn.Embedding(subject_count, 2)
            nn.init.normal_(self.subject_code.weight, std=0.01)
            nn.init.zeros_(self.subject_bias.weight)

        self.stable_profile_bias = _ZeroResidualHead(
            spec.code_dim + support_profile_dim,
            max(spec.code_dim, 32),
            dropout=0.1,
        )
        dynamic_input_dim = dimension + spec.code_dim + DESCRIPTOR_DIM + 2
        if spec.use_dynamic:
            self.dynamic_residual = _ZeroResidualHead(
                dynamic_input_dim,
                dimension // 2,
            )
        if spec.use_reliability:
            self.reliability_gate = nn.Sequential(
                nn.Linear(spec.code_dim + DESCRIPTOR_DIM * 2 + 2, 64),
                nn.SiLU(),
                nn.Linear(64, 2),
            )
            final = self.reliability_gate[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, 2.0)

    @staticmethod
    def _validate_support(
        support_descriptors: torch.Tensor | None,
        support_bp: torch.Tensor | None,
        batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if support_descriptors is None or support_bp is None:
            raise ValueError("personal profile requires train-role support information")
        if support_descriptors.shape != (batch, SUPPORT_COUNT, DESCRIPTOR_DIM):
            raise ValueError("support descriptor shape is invalid")
        if support_bp.shape != (batch, SUPPORT_COUNT, 2):
            raise ValueError("support BP shape is invalid")
        return support_descriptors, support_bp

    def _empty_support(
        self,
        descriptor: torch.Tensor,
        subject_train_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros_descriptor = torch.zeros_like(descriptor)
        zeros_bp = torch.zeros_like(subject_train_mean)
        support_profile = torch.cat(
            [zeros_descriptor, zeros_descriptor, zeros_bp, zeros_bp], dim=-1
        )
        return (
            zeros_descriptor,
            zeros_descriptor,
            subject_train_mean,
            zeros_bp,
            support_profile,
        )

    def _summarize_support(
        self,
        support_descriptors: torch.Tensor,
        support_bp: torch.Tensor,
        subject_train_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        descriptor_mean = support_descriptors.mean(dim=1)
        descriptor_std = support_descriptors.std(dim=1, unbiased=False)
        bp_mean = support_bp.mean(dim=1)
        bp_std = support_bp.std(dim=1, unbiased=False)
        support_profile = torch.cat(
            [
                descriptor_mean,
                descriptor_std,
                bp_mean - subject_train_mean,
                bp_std,
            ],
            dim=-1,
        )
        return descriptor_mean, descriptor_std, bp_mean, bp_std, support_profile

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
        del demographics
        if hasattr(self, "reference"):
            return self.reference(
                ppg,
                subject_train_mean,
                query_descriptor=query_descriptor,
                support_descriptors=support_descriptors,
                support_bp=support_bp,
                subject_index=subject_index,
            )
        if subject_train_mean.ndim != 2 or subject_train_mean.shape[1] != 2:
            raise ValueError("subject_train_mean must have shape [batch, 2]")
        features = self.encoder(ppg)
        batch = features.shape[0]
        descriptor = (
            query_descriptor if query_descriptor is not None else waveform_descriptor(ppg)
        )
        if descriptor.shape != (batch, DESCRIPTOR_DIM):
            raise ValueError("query descriptor shape is invalid")

        if self.spec.uses_support:
            supports, support_targets = self._validate_support(
                support_descriptors, support_bp, batch
            )
            summary = self._summarize_support(
                supports, support_targets, subject_train_mean
            )
        else:
            summary = self._empty_support(descriptor, subject_train_mean)
        descriptor_mean, descriptor_std, bp_mean, bp_std, support_profile = summary

        profile = self.profile_encoder(support_profile)
        direct_bias = torch.zeros_like(subject_train_mean)
        if self.spec.uses_subject_index:
            if subject_index is None or subject_index.shape != (batch,):
                raise ValueError("personal profile requires one seen-subject index per query")
            profile = profile + self.subject_code(subject_index)
            direct_bias = self.subject_bias(subject_index)

        shared = self.shared_residual(features)
        stable = direct_bias + self.stable_profile_bias(
            torch.cat([profile, support_profile], dim=-1)
        )
        dynamic = torch.zeros_like(shared)
        if self.spec.use_dynamic:
            dynamic = self.dynamic_residual(
                torch.cat(
                    [
                        features,
                        profile,
                        descriptor - descriptor_mean,
                        bp_mean - subject_train_mean,
                    ],
                    dim=-1,
                )
            )
            if self.spec.use_reliability:
                gate = torch.sigmoid(
                    self.reliability_gate(
                        torch.cat(
                            [
                                profile,
                                (descriptor - descriptor_mean).abs(),
                                descriptor_std,
                                bp_std,
                            ],
                            dim=-1,
                        )
                    )
                )
                dynamic = gate * dynamic
        return subject_train_mean + shared + stable + dynamic
