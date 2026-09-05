"""Mechanism-led feature personalization for registered participants.

These are explicit LoRA/FiLM/bilinear-adaptation comparisons, not a claim of
an entirely new primitive. Personal state is learned from the train role.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .models import build_ppg_encoder
from .same_subject_components import COMPONENTS, SameSubjectComponentRegressor, _ZeroResidualHead
from .same_subject_personal_profiles import PERSONAL_PROFILES, SameSubjectPersonalProfileRegressor

SCREEN_ID = "personal-feature-mechanisms-v1"
REFERENCE = "subject_lora_rank4"
PRIMARY = "subject_nonlinear_rank4"
SPLIT_MODES = ("random_disjoint", "chronological_blocked")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    adapter: str
    code_dim: int = 0
    rank: int = 4
    uses_subject_index: bool = True
    uses_support: bool = False
    uses_demographics: bool = False
    training_rule: str = "standard"
    backbone: str = "resnet_small"

    @property
    def participant_trainable_parameters(self):
        if not self.uses_subject_index:
            return 0
        if self.adapter in {"lora", "nonlinear"}:
            return 2 * 256 * self.rank
        return self.code_dim + 2


SPECS = (
    FeatureSpec(REFERENCE, "lora"),
    FeatureSpec("shared_lora_rank4", "lora", uses_subject_index=False),
    FeatureSpec("subject_lora_rank1", "lora", rank=1),
    FeatureSpec("output_profile32", "output", code_dim=32),
    FeatureSpec("feature_affine32", "affine", code_dim=32),
    FeatureSpec("shared_bilinear32", "bilinear", code_dim=32),
    FeatureSpec("shared_bilinear64", "bilinear", code_dim=64),
    FeatureSpec(PRIMARY, "nonlinear"),
)
FEATURE_MODELS = {s.name: s for s in SPECS}


class PersonalFeatureRegressor(nn.Module):
    def __init__(self, spec: FeatureSpec, *, subject_count: int):
        super().__init__()
        if subject_count < 1:
            raise ValueError("registered participants required")
        self.spec = spec
        if spec.adapter in {"lora", "nonlinear"}:
            self.reference = SameSubjectComponentRegressor(
                COMPONENTS["residual_subject_lora_rank4"],
                subject_count=subject_count if spec.uses_subject_index else 1,
                lora_rank=spec.rank,
            )
        elif spec.adapter == "output":
            self.reference = SameSubjectPersonalProfileRegressor(
                PERSONAL_PROFILES["personal_profile_code32_no_support"], subject_count=subject_count)
        else:
            self.encoder = build_ppg_encoder(spec.backbone, feature_dim=256)
            self.residual_head = _ZeroResidualHead(256, 128)
            self.subject_code = nn.Embedding(subject_count, spec.code_dim)
            self.subject_bias = nn.Embedding(subject_count, 2)
            nn.init.normal_(self.subject_code.weight, std=0.01)
            nn.init.zeros_(self.subject_bias.weight)
            if spec.adapter == "affine":
                self.affine = nn.Linear(spec.code_dim, 512, bias=False)
                nn.init.normal_(self.affine.weight, std=0.01)
            elif spec.adapter == "bilinear":
                self.analysis = nn.Linear(256, spec.code_dim, bias=False)
                self.synthesis = nn.Linear(spec.code_dim, 256, bias=False)
            else:
                raise ValueError("unknown feature adaptation")

    def personal_parameters(self):
        if not self.spec.uses_subject_index:
            return ()
        if self.spec.adapter in {"lora", "nonlinear"}:
            return (self.reference.lora_a.weight, self.reference.lora_b.weight)
        if self.spec.adapter == "output":
            return (self.reference.subject_code.weight, self.reference.subject_bias.weight)
        return (self.subject_code.weight, self.subject_bias.weight)

    def forward(self, ppg, subject_train_mean, *, subject_index=None, **kwargs):
        batch = ppg.shape[0]
        if subject_train_mean.shape != (batch, 2):
            raise ValueError("anchor must be [batch,2]")
        if self.spec.uses_subject_index:
            if subject_index is None or subject_index.shape != (batch,):
                raise ValueError("one registered participant index is required per window")
        else:
            subject_index = torch.zeros(batch, dtype=torch.long, device=ppg.device)
        if self.spec.adapter in {"lora", "output"}:
            return self.reference(ppg, subject_train_mean, subject_index=subject_index, **kwargs)
        if self.spec.adapter == "nonlinear":
            ref = self.reference
            z = ref.encoder(ppg)
            a = ref.lora_a(subject_index).reshape(batch, self.spec.rank, 256)
            b = ref.lora_b(subject_index).reshape(batch, 256, self.spec.rank)
            # 2*SiLU has derivative one at zero: matches the linear control locally.
            low = 2 * nn.functional.silu(torch.bmm(a, z.unsqueeze(-1)))
            adapted = z + torch.bmm(b, low).squeeze(-1) / self.spec.rank
            return subject_train_mean + ref.residual_head(adapted)
        z = self.encoder(ppg)
        c = self.subject_code(subject_index)
        if self.spec.adapter == "affine":
            gamma, beta = self.affine(c).chunk(2, dim=-1)
            adapted = z * (1 + gamma) + beta
        else:
            # Shared trainable feature directions; personal coefficients are persistent.
            adapted = z + self.synthesis(c * self.analysis(z))
        return subject_train_mean + self.subject_bias(subject_index) + self.residual_head(adapted)
