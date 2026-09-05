"""Warm-start LoRA plus persistent personal residual state (PRS).

PRS is a compact correction of the existing prediction, not a second complete
BP estimator. This combines established adapter/residual concepts; novelty
and accuracy are empirical questions, not assumptions.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .same_subject_components import COMPONENTS, SameSubjectComponentRegressor, _ZeroResidualHead, waveform_descriptor

SCREEN_ID = "lora-prs-continuation-v1"
REFERENCE = "lora_continue"
PRIMARY = "lora_prs_dynamic"
SPLIT_MODES = ("random_disjoint", "chronological_blocked")


@dataclass(frozen=True)
class PRSSpec:
    name: str
    adapter: str
    code_dim: int = 0
    freeze_base: bool = False
    correction_penalty: float = 0.0
    uses_subject_index: bool = True
    uses_support: bool = False
    uses_demographics: bool = False
    backbone: str = "resnet_small"
    training_rule: str = "standard"

    @property
    def added_personal_parameters(self):
        return 0 if self.adapter == "continue" else 2 + self.code_dim

    @property
    def participant_trainable_parameters(self):
        return self.added_personal_parameters + (0 if self.freeze_base else 2048)

    @property
    def stored_personal_parameters(self):
        return 2048 + self.added_personal_parameters


SPECS = (
    PRSSpec(REFERENCE, "continue"),
    PRSSpec("lora_prs_bias", "bias"),
    PRSSpec(PRIMARY, "dynamic", code_dim=32),
    PRSSpec("lora_prs_dynamic_shrink", "dynamic", code_dim=32, correction_penalty=0.01),
    PRSSpec("lora_prs_dynamic_frozen", "dynamic", code_dim=32, freeze_base=True),
)
PRS_MODELS = {spec.name: spec for spec in SPECS}


class LoraPRSRegressor(nn.Module):
    def __init__(self, spec, *, subject_count):
        super().__init__()
        self.spec = spec
        self.base = SameSubjectComponentRegressor(COMPONENTS["residual_subject_lora_rank4"], subject_count=subject_count)
        if spec.adapter != "continue":
            self.subject_bias = nn.Embedding(subject_count, 2)
            nn.init.zeros_(self.subject_bias.weight)
        if spec.adapter == "dynamic":
            self.subject_code = nn.Embedding(subject_count, spec.code_dim)
            nn.init.normal_(self.subject_code.weight, std=0.01)
            self.correction_head = _ZeroResidualHead(256 + spec.code_dim + 10, 128)
        if spec.freeze_base:
            self.base.requires_grad_(False)
            self.base.eval()
        self.latest_correction = None

    def train(self, mode=True):
        super().train(mode)
        if self.spec.freeze_base:
            self.base.eval()  # Frozen BN statistics and dropout, not just frozen gradients.
        return self

    def forward(self, ppg, subject_train_mean, *, subject_index=None, query_descriptor=None, **kwargs):
        if subject_index is None or subject_index.shape != (len(ppg),):
            raise ValueError("registered participant index required")
        if subject_train_mean.shape != (len(ppg), 2):
            raise ValueError("BP anchor must be [batch,2]")
        z = self.base.encoder(ppg)
        a = self.base.lora_a(subject_index).reshape(len(ppg), 4, 256)
        b = self.base.lora_b(subject_index).reshape(len(ppg), 256, 4)
        z = z + torch.bmm(b, torch.bmm(a, z.unsqueeze(-1))).squeeze(-1) / 4
        original = subject_train_mean + self.base.residual_head(z)
        correction = torch.zeros_like(original)
        if self.spec.adapter != "continue":
            correction = correction + self.subject_bias(subject_index)
        if self.spec.adapter == "dynamic":
            descriptor = waveform_descriptor(ppg) if query_descriptor is None else query_descriptor
            correction = correction + self.correction_head(torch.cat(
                [z, self.subject_code(subject_index), descriptor], dim=-1))
        self.latest_correction = correction
        return original + correction

    def penalty(self):
        return self.spec.correction_penalty * self.latest_correction.float().square().mean()

    def optimizer(self, args):
        groups = []
        base = [p for p in self.base.parameters() if p.requires_grad]
        extra = [p for n, p in self.named_parameters() if not n.startswith("base.") and p.requires_grad]
        if base:
            groups.append({"params": base, "lr": args.base_learning_rate})
        if extra:
            groups.append({"params": extra, "lr": args.learning_rate})
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay)
