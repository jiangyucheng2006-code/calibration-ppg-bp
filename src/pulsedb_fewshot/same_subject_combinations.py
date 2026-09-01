"""Controlled combinations for the seen-subject residual personalization model.

The screen keeps the previously selected participant-indexed rank-4 LoRA
adapter in every candidate and adds only modules that improved the paired
single-component reference in both PulseDB source strata.  Held-out targets
are outside this module's data contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .models import build_ppg_encoder
from .same_subject_components import (
    DESCRIPTOR_DIM,
    SUPPORT_COUNT,
    _ZeroResidualHead,
    waveform_descriptor,
)


SCREEN_ID = "same-subject-combination-v1"
REFERENCE_CANDIDATE = "lora"
PROMOTION_MARGIN_MMHG = 0.15
COMBINATION_SPLIT_MODES = ("random_disjoint", "chronological_blocked")

SUBJECT_LORA = "subject_lora"
FILM = "film"
SUPPORT_ATTENTION = "support_attention"
MULTI_EVENT_WEIGHTING = "multi_event_weighting"
SUPPORT_RELIABILITY = "support_reliability"
CALIBRATION_RELATIVE = "calibration_relative"

SUPPORT_MODULES = frozenset(
    {
        SUPPORT_ATTENTION,
        MULTI_EVENT_WEIGHTING,
        SUPPORT_RELIABILITY,
        CALIBRATION_RELATIVE,
    }
)
KNOWN_MODULES = frozenset({SUBJECT_LORA, FILM, *SUPPORT_MODULES})


@dataclass(frozen=True)
class CombinationSpec:
    name: str
    modules: tuple[str, ...]
    description: str
    backbone: str = "resnet_small"
    uses_demographics: bool = False
    training_rule: str = "standard"

    def __post_init__(self) -> None:
        if not self.modules or self.modules[0] != SUBJECT_LORA:
            raise ValueError("every combination must retain subject LoRA")
        if len(self.modules) != len(set(self.modules)):
            raise ValueError("combination modules must be unique")
        unknown = set(self.modules) - KNOWN_MODULES
        if unknown:
            raise ValueError(f"unknown combination modules: {sorted(unknown)}")

    @property
    def adapter(self) -> str:
        return "+".join(self.modules)

    @property
    def uses_support(self) -> bool:
        return bool(set(self.modules) & (SUPPORT_MODULES | {FILM}))

    @property
    def uses_subject_index(self) -> bool:
        return SUBJECT_LORA in self.modules


def _spec(name: str, *modules: str) -> CombinationSpec:
    return CombinationSpec(
        name=name,
        modules=(SUBJECT_LORA, *modules),
        description=" + ".join((SUBJECT_LORA, *modules)),
    )


# A deliberately bounded hierarchy rather than an exhaustive 2^6 search.
# It estimates every LoRA-plus-one effect, selected three-way interactions,
# two mechanistically coherent four-module combinations, and the all-six model.
COMBINATION_SPECS = (
    _spec("lora"),
    _spec("lora_film", FILM),
    _spec("lora_attention", SUPPORT_ATTENTION),
    _spec("lora_multi_event", MULTI_EVENT_WEIGHTING),
    _spec("lora_reliability", SUPPORT_RELIABILITY),
    _spec("lora_calibration_relative", CALIBRATION_RELATIVE),
    _spec("lora_film_attention", FILM, SUPPORT_ATTENTION),
    _spec("lora_film_multi_event", FILM, MULTI_EVENT_WEIGHTING),
    _spec("lora_film_reliability", FILM, SUPPORT_RELIABILITY),
    _spec("lora_film_calibration_relative", FILM, CALIBRATION_RELATIVE),
    _spec(
        "lora_attention_calibration_relative",
        SUPPORT_ATTENTION,
        CALIBRATION_RELATIVE,
    ),
    _spec(
        "lora_multi_event_reliability",
        MULTI_EVENT_WEIGHTING,
        SUPPORT_RELIABILITY,
    ),
    _spec(
        "lora_film_attention_calibration_relative",
        FILM,
        SUPPORT_ATTENTION,
        CALIBRATION_RELATIVE,
    ),
    _spec(
        "lora_film_multi_event_reliability",
        FILM,
        MULTI_EVENT_WEIGHTING,
        SUPPORT_RELIABILITY,
    ),
    _spec(
        "lora_all_six",
        FILM,
        SUPPORT_ATTENTION,
        MULTI_EVENT_WEIGHTING,
        SUPPORT_RELIABILITY,
        CALIBRATION_RELATIVE,
    ),
)
COMBINATIONS = {spec.name: spec for spec in COMBINATION_SPECS}


class SameSubjectCombinationRegressor(nn.Module):
    """Residual model with a fixed subject LoRA core and optional add-ons."""

    def __init__(
        self,
        spec: CombinationSpec,
        *,
        subject_count: int,
        lora_rank: int = 4,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = build_ppg_encoder(spec.backbone, feature_dim=256)
        dimension = int(self.encoder.feature_dim)
        self.residual_head = _ZeroResidualHead(dimension, dimension // 2)

        if subject_count < 1:
            raise ValueError("subject LoRA requires at least one seen participant")
        self.lora_rank = int(lora_rank)
        self.lora_a = nn.Embedding(subject_count, dimension * self.lora_rank)
        self.lora_b = nn.Embedding(subject_count, dimension * self.lora_rank)
        nn.init.normal_(self.lora_a.weight, std=0.01)
        nn.init.zeros_(self.lora_b.weight)

        if FILM in spec.modules:
            self.film = nn.Linear(DESCRIPTOR_DIM + 2, dimension * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        self.support_scorers = nn.ModuleDict()
        self.support_corrections = nn.ModuleDict()
        for module in spec.modules:
            if module not in SUPPORT_MODULES:
                continue
            if module == SUPPORT_ATTENTION:
                self.support_scorers[module] = nn.Sequential(
                    nn.Linear(DESCRIPTOR_DIM * 3, 32),
                    nn.SiLU(),
                    nn.Linear(32, 1),
                )
            elif module in {SUPPORT_RELIABILITY, MULTI_EVENT_WEIGHTING}:
                score_dim = DESCRIPTOR_DIM + (
                    2 if module == MULTI_EVENT_WEIGHTING else 0
                )
                self.support_scorers[module] = nn.Sequential(
                    nn.Linear(score_dim, 32),
                    nn.SiLU(),
                    nn.Linear(32, 1),
                )
            self.support_corrections[module] = _ZeroResidualHead(
                dimension + DESCRIPTOR_DIM + 2,
                dimension // 2,
            )
        if MULTI_EVENT_WEIGHTING in spec.modules:
            self.multi_event_anchor_scale = nn.Parameter(torch.zeros(2))

    @staticmethod
    def _validate_support(
        support_descriptors: torch.Tensor | None,
        support_bp: torch.Tensor | None,
        batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if support_descriptors is None or support_bp is None:
            raise ValueError("combination requires train-role support descriptors and BP")
        if support_descriptors.shape != (batch, SUPPORT_COUNT, DESCRIPTOR_DIM):
            raise ValueError("support descriptor shape is invalid")
        if support_bp.shape != (batch, SUPPORT_COUNT, 2):
            raise ValueError("support BP shape is invalid")
        return support_descriptors, support_bp

    def _support_weights(
        self,
        module: str,
        descriptor: torch.Tensor,
        support_descriptors: torch.Tensor,
        support_bp: torch.Tensor,
        subject_train_mean: torch.Tensor,
    ) -> torch.Tensor:
        batch = descriptor.shape[0]
        expanded_query = descriptor[:, None, :].expand_as(support_descriptors)
        if module == SUPPORT_ATTENTION:
            logits = self.support_scorers[module](
                torch.cat(
                    [
                        expanded_query,
                        support_descriptors,
                        (expanded_query - support_descriptors).abs(),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
        elif module in {SUPPORT_RELIABILITY, MULTI_EVENT_WEIGHTING}:
            score_inputs = [support_descriptors]
            if module == MULTI_EVENT_WEIGHTING:
                score_inputs.append(support_bp - subject_train_mean[:, None, :])
            logits = self.support_scorers[module](
                torch.cat(score_inputs, dim=-1)
            ).squeeze(-1)
        elif module == CALIBRATION_RELATIVE:
            logits = torch.zeros(
                batch,
                SUPPORT_COUNT,
                dtype=support_descriptors.dtype,
                device=support_descriptors.device,
            )
        else:
            raise ValueError(f"unsupported support module {module!r}")
        return torch.softmax(logits, dim=1)

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
        if subject_train_mean.ndim != 2 or subject_train_mean.shape[1] != 2:
            raise ValueError("subject_train_mean must have shape [batch, 2]")
        features = self.encoder(ppg)
        batch, dimension = features.shape
        descriptor = (
            query_descriptor if query_descriptor is not None else waveform_descriptor(ppg)
        )
        if descriptor.shape != (batch, DESCRIPTOR_DIM):
            raise ValueError("query descriptor shape is invalid")
        if subject_index is None or subject_index.shape != (batch,):
            raise ValueError("subject LoRA requires one seen-subject index per query")

        a = self.lora_a(subject_index).reshape(batch, self.lora_rank, dimension)
        b = self.lora_b(subject_index).reshape(batch, dimension, self.lora_rank)
        low_rank = torch.bmm(a, features.unsqueeze(-1))
        features = features + torch.bmm(b, low_rank).squeeze(-1) / self.lora_rank

        if self.spec.uses_support:
            support_descriptors, support_bp = self._validate_support(
                support_descriptors, support_bp, batch
            )
        if FILM in self.spec.modules:
            assert support_descriptors is not None and support_bp is not None
            context = torch.cat(
                [support_descriptors.mean(dim=1), support_bp.mean(dim=1)], dim=-1
            )
            gamma, beta = self.film(context).chunk(2, dim=-1)
            features = (1.0 + gamma) * features + beta

        residual = self.residual_head(features)
        for module in self.spec.modules:
            if module not in SUPPORT_MODULES:
                continue
            assert support_descriptors is not None and support_bp is not None
            weights = self._support_weights(
                module,
                descriptor,
                support_descriptors,
                support_bp,
                subject_train_mean,
            )
            support_context = torch.sum(
                weights[..., None] * support_descriptors, dim=1
            )
            weighted_bp = torch.sum(weights[..., None] * support_bp, dim=1)
            correction_inputs = torch.cat(
                [
                    features,
                    descriptor - support_context,
                    weighted_bp - subject_train_mean,
                ],
                dim=-1,
            )
            residual = residual + self.support_corrections[module](correction_inputs)
            if module == MULTI_EVENT_WEIGHTING:
                residual = residual + self.multi_event_anchor_scale * (
                    weighted_bp - subject_train_mean
                )
        return subject_train_mean + residual
