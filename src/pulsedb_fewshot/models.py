"""Neural baselines and variable-K personalized PPG-BP models."""

from __future__ import annotations

from collections.abc import Sequence
import copy

import torch
from torch import nn


BACKBONE_NAMES = (
    "resnet_small",
    "resnet_deep",
    "inception_time",
    "patch_transformer",
    "conformer",
)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        padding = (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU())
        super().__init__(*layers)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ConvNormAct(in_channels, out_channels, 5, stride=stride),
            ConvNormAct(out_channels, out_channels, 5, activation=False),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvNormAct(
                in_channels, out_channels, 1, stride=stride, activation=False
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.shortcut(inputs))


class MultiScaleStem(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        branch_channels: int = 16,
        output_channels: int = 48,
        kernels: Sequence[int] = (3, 7, 15),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvNormAct(input_channels, branch_channels, kernel, stride=2)
                for kernel in kernels
            ]
        )
        self.fusion = ConvNormAct(branch_channels * len(kernels), output_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([branch(inputs) for branch in self.branches], dim=1))


class MultiScaleResNetEncoder(nn.Module):
    """A compact PPG-only population encoder derived from the audited old backbone."""

    def __init__(self, input_channels: int = 1, feature_dim: int = 256) -> None:
        super().__init__()
        self.backbone_name = "resnet_small"
        self.stem = MultiScaleStem(input_channels=input_channels)
        self.blocks = nn.Sequential(
            ResidualBlock1D(48, 64),
            ResidualBlock1D(64, 96, stride=2),
            ResidualBlock1D(96, 128, stride=2),
            ResidualBlock1D(128, 192, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(192, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU()
        )
        self.feature_dim = feature_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("PPG input must have shape [batch, 1, time]")
        return self.projection(self.pool(self.blocks(self.stem(inputs))))


class DeepMultiScaleResNetEncoder(nn.Module):
    """Higher-capacity ResNet control with the same input/output contract.

    This is deliberately a same-family capacity control rather than a claim of
    reproducing a particular published XResNet implementation.  It separates
    the effect of a larger network from the effect of changing architecture.
    """

    def __init__(self, input_channels: int = 1, feature_dim: int = 256) -> None:
        super().__init__()
        self.backbone_name = "resnet_deep"
        self.stem = MultiScaleStem(
            input_channels=input_channels,
            branch_channels=24,
            output_channels=72,
            kernels=(3, 9, 21),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(72, 96),
            ResidualBlock1D(96, 128, stride=2),
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 192, stride=2),
            ResidualBlock1D(192, 192),
            ResidualBlock1D(192, 256, stride=2),
            ResidualBlock1D(256, 256),
            ResidualBlock1D(256, 384, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(384, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU()
        )
        self.feature_dim = feature_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("PPG input must have shape [batch, 1, time]")
        return self.projection(self.pool(self.blocks(self.stem(inputs))))


class InceptionModule1D(nn.Module):
    """Single InceptionTime-style multiscale temporal module."""

    def __init__(self, in_channels: int, branch_channels: int = 32) -> None:
        super().__init__()
        bottleneck_channels = 32
        self.bottleneck = (
            ConvNormAct(in_channels, bottleneck_channels, 1)
            if in_channels > 1
            else nn.Identity()
        )
        effective_channels = bottleneck_channels if in_channels > 1 else in_channels
        self.convolutions = nn.ModuleList(
            [
                ConvNormAct(effective_channels, branch_channels, kernel)
                for kernel in (39, 19, 9)
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            ConvNormAct(in_channels, branch_channels, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        reduced = self.bottleneck(inputs)
        branches = [layer(reduced) for layer in self.convolutions]
        branches.append(self.pool_branch(inputs))
        return torch.cat(branches, dim=1)


class InceptionTimeEncoder(nn.Module):
    """Single-network InceptionTime-style encoder for 10-second PPG."""

    def __init__(self, input_channels: int = 1, feature_dim: int = 256) -> None:
        super().__init__()
        self.backbone_name = "inception_time"
        self.stem = ConvNormAct(input_channels, 32, 15, stride=2)
        modules: list[nn.Module] = []
        residuals: list[nn.Module] = []
        channels = 32
        for index in range(6):
            modules.append(InceptionModule1D(channels, branch_channels=32))
            residual_input_channels = 32 if index == 2 else 128
            residuals.append(
                ConvNormAct(residual_input_channels, 128, 1, activation=False)
                if index in {2, 5}
                else nn.Identity()
            )
            channels = 128
        self.modules_ = nn.ModuleList(modules)
        self.residuals = nn.ModuleList(residuals)
        self.activation = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(128, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU()
        )
        self.feature_dim = feature_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("PPG input must have shape [batch, 1, time]")
        value = self.stem(inputs)
        block_input = value
        for index, module in enumerate(self.modules_):
            value = module(value)
            if index in {2, 5}:
                value = self.activation(value + self.residuals[index](block_input))
                block_input = value
        return self.projection(self.pool(value))


class PatchTransformerEncoder(nn.Module):
    """PatchTST-inspired encoder adapted to fixed-length univariate PPG.

    PatchTST was proposed for forecasting and self-supervised representation
    learning.  This implementation borrows only its patch-token principle; it
    is trained from scratch for BP regression under this project's protocol.
    """

    def __init__(self, input_channels: int = 1, feature_dim: int = 256) -> None:
        super().__init__()
        self.backbone_name = "patch_transformer"
        model_dim = 128
        self.patch = nn.Conv1d(input_channels, model_dim, kernel_size=50, stride=25)
        self.position = nn.Parameter(torch.zeros(1, 64, model_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=384,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=4)
        self.norm = nn.LayerNorm(model_dim)
        self.projection = nn.Sequential(
            nn.Linear(model_dim, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU()
        )
        self.feature_dim = feature_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("PPG input must have shape [batch, 1, time]")
        tokens = self.patch(inputs).transpose(1, 2)
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("PPG creates more patch tokens than positional capacity")
        tokens = tokens + self.position[:, : tokens.shape[1]]
        tokens = self.norm(self.blocks(tokens)).mean(dim=1)
        return self.projection(tokens)


class ConformerBlock1D(nn.Module):
    """Compact Conformer block combining attention and local convolution."""

    def __init__(self, dimension: int, *, heads: int = 4, kernel_size: int = 31) -> None:
        super().__init__()
        self.ffn1_norm = nn.LayerNorm(dimension)
        self.ffn1 = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dimension * 4, dimension),
            nn.Dropout(0.1),
        )
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.1, batch_first=True
        )
        self.conv_norm = nn.LayerNorm(dimension)
        self.conv_pointwise_in = nn.Conv1d(dimension, dimension * 2, 1)
        self.conv_depthwise = nn.Conv1d(
            dimension,
            dimension,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dimension,
        )
        self.conv_batch_norm = nn.BatchNorm1d(dimension)
        self.conv_pointwise_out = nn.Conv1d(dimension, dimension, 1)
        self.ffn2_norm = nn.LayerNorm(dimension)
        self.ffn2 = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dimension * 4, dimension),
            nn.Dropout(0.1),
        )
        self.final_norm = nn.LayerNorm(dimension)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs + 0.5 * self.ffn1(self.ffn1_norm(inputs))
        normalized = self.attention_norm(value)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        value = value + attended
        convolution = self.conv_norm(value).transpose(1, 2)
        convolution = nn.functional.glu(self.conv_pointwise_in(convolution), dim=1)
        convolution = nn.functional.silu(
            self.conv_batch_norm(self.conv_depthwise(convolution))
        )
        convolution = self.conv_pointwise_out(convolution).transpose(1, 2)
        value = value + convolution
        value = value + 0.5 * self.ffn2(self.ffn2_norm(value))
        return self.final_norm(value)


class ConformerEncoder1D(nn.Module):
    """Compact Conformer backbone for local pulse shape and global context."""

    def __init__(self, input_channels: int = 1, feature_dim: int = 256) -> None:
        super().__init__()
        self.backbone_name = "conformer"
        model_dim = 128
        self.patch = nn.Conv1d(input_channels, model_dim, kernel_size=50, stride=25)
        self.position = nn.Parameter(torch.zeros(1, 64, model_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([ConformerBlock1D(model_dim) for _ in range(4)])
        self.projection = nn.Sequential(
            nn.Linear(model_dim, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU()
        )
        self.feature_dim = feature_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("PPG input must have shape [batch, 1, time]")
        tokens = self.patch(inputs).transpose(1, 2)
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("PPG creates more patch tokens than positional capacity")
        tokens = tokens + self.position[:, : tokens.shape[1]]
        for block in self.blocks:
            tokens = block(tokens)
        return self.projection(tokens.mean(dim=1))


def build_ppg_encoder(
    backbone: str = "resnet_small", *, input_channels: int = 1, feature_dim: int = 256
) -> nn.Module:
    """Create a PPG encoder with a shared ``[B, 1, T] -> [B, D]`` contract."""

    constructors = {
        "resnet_small": MultiScaleResNetEncoder,
        "resnet_deep": DeepMultiScaleResNetEncoder,
        "inception_time": InceptionTimeEncoder,
        "patch_transformer": PatchTransformerEncoder,
        "conformer": ConformerEncoder1D,
    }
    if backbone not in constructors:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {BACKBONE_NAMES}")
    return constructors[backbone](
        input_channels=input_channels, feature_dim=feature_dim
    )


def model_parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


class PopulationRegressor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder or MultiScaleResNetEncoder()
        self.head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(self.encoder.feature_dim, 1)
                )
                for _ in range(2)
            ]
        )

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(features) for head in self.head], dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.predict_from_features(self.encoder(inputs))


class SiameseDeltaRegressor(nn.Module):
    """Predict signed SBP/DBP change from a support anchor to a query."""

    def __init__(self, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.encoder = encoder or MultiScaleResNetEncoder()
        dimension = self.encoder.feature_dim
        self.delta_head = nn.Sequential(
            nn.Linear(dimension * 3, dimension),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(dimension, 2),
        )

    def forward(
        self,
        query_ppg: torch.Tensor,
        support_ppg: torch.Tensor,
        support_bp: torch.Tensor,
    ) -> torch.Tensor:
        query_features = self.encoder(query_ppg)
        support_features = self.encoder(support_ppg)
        signed_difference = query_features - support_features
        delta = self.delta_head(
            torch.cat([query_features, support_features, signed_difference], dim=-1)
        )
        return support_bp + delta


class VariableKPersonalizer(nn.Module):
    """Residual-anchor support-set model with optional FiLM and reliability weights."""

    def __init__(
        self,
        population: PopulationRegressor | None = None,
        *,
        use_film: bool = True,
        query_conditioned_weights: bool = False,
        anchor_mode: str = "mean",
        use_quality_gate: bool = False,
        use_demographics: bool = False,
        demographic_dim: int = 5,
        demographic_mode: str = "encoded",
    ) -> None:
        super().__init__()
        self.population = population or PopulationRegressor()
        dimension = self.population.encoder.feature_dim
        token_dimension = dimension + 4
        self.support_token = nn.Sequential(
            nn.Linear(token_dimension, dimension), nn.SiLU(), nn.Linear(dimension, dimension)
        )
        self.use_film = use_film
        self.query_conditioned_weights = query_conditioned_weights
        if anchor_mode not in {"mean", "median"}:
            raise ValueError("anchor_mode must be 'mean' or 'median'")
        self.anchor_mode = anchor_mode
        self.use_quality_gate = use_quality_gate
        self.use_demographics = use_demographics
        if demographic_mode not in {"encoded", "direct"}:
            raise ValueError("demographic_mode must be 'encoded' or 'direct'")
        self.demographic_mode = demographic_mode
        if query_conditioned_weights:
            self.reliability = nn.Sequential(
                nn.Linear(dimension * 2 + 2, dimension),
                nn.SiLU(),
                nn.Linear(dimension, 1),
            )
        if use_film:
            self.film = nn.Linear(dimension, dimension * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        if use_demographics and demographic_mode == "encoded":
            self.demographic_encoder = nn.Sequential(
                nn.Linear(demographic_dim, 32),
                nn.SiLU(),
                nn.Linear(32, dimension),
                nn.LayerNorm(dimension),
            )
        if use_quality_gate:
            self.quality_gate = nn.Sequential(
                nn.Linear(dimension * 3, dimension),
                nn.SiLU(),
                nn.Linear(dimension, 2),
            )
            quality_final = self.quality_gate[-1]
            assert isinstance(quality_final, nn.Linear)
            nn.init.zeros_(quality_final.weight)
            nn.init.constant_(quality_final.bias, 4.0)
        correction_inputs = dimension * 2
        if use_demographics:
            correction_inputs += (
                dimension if demographic_mode == "encoded" else demographic_dim
            )
        self.correction = nn.Sequential(
            nn.Linear(correction_inputs, dimension),
            nn.SiLU(),
            nn.Linear(dimension, 2),
        )
        final = self.correction[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def _masked_weights(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("every episode requires at least one support event")
        masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.softmax(masked, dim=1)

    @staticmethod
    def _masked_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Coordinate-wise median with an exact midpoint for even support counts."""

        if mask.dtype != torch.bool:
            mask = mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("every episode requires at least one support event")
        masked = values.masked_fill(~mask[..., None], torch.inf)
        ordered = masked.sort(dim=1).values
        counts = mask.sum(dim=1)
        lower = ((counts - 1) // 2)[:, None, None].expand(-1, 1, values.shape[-1])
        upper = (counts // 2)[:, None, None].expand(-1, 1, values.shape[-1])
        lower_values = ordered.gather(1, lower).squeeze(1)
        upper_values = ordered.gather(1, upper).squeeze(1)
        return (lower_values + upper_values) / 2

    def forward(
        self,
        query_ppg: torch.Tensor,
        support_ppg: torch.Tensor,
        support_bp: torch.Tensor,
        support_mask: torch.Tensor,
        demographics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if support_ppg.ndim != 4:
            raise ValueError("support PPG must have shape [batch, Kmax, 1, time]")
        batch, maximum_k, channels, length = support_ppg.shape
        query_features = self.population.encoder(query_ppg)
        support_features = self.population.encoder(
            support_ppg.reshape(batch * maximum_k, channels, length)
        ).reshape(batch, maximum_k, -1)
        return self.forward_from_features(
            query_features,
            support_features,
            support_bp,
            support_mask,
            demographics,
        )

    def forward_from_features(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        support_bp: torch.Tensor,
        support_mask: torch.Tensor,
        demographics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict from already encoded PPG features.

        This is mathematically identical to :meth:`forward` after the shared
        encoder.  It supports leakage-safe feature caching without changing
        the population, support-anchor, quality-gate, or correction mapping.
        """

        if query_features.ndim != 2:
            raise ValueError("query features must have shape [batch, feature]")
        if support_features.ndim != 3:
            raise ValueError(
                "support features must have shape [batch, Kmax, feature]"
            )
        batch, maximum_k, dimension = support_features.shape
        if query_features.shape != (batch, dimension):
            raise ValueError("query/support feature shapes are inconsistent")
        if support_bp.shape != (batch, maximum_k, 2):
            raise ValueError("support BP must have shape [batch, Kmax, 2]")
        if support_mask.shape != (batch, maximum_k):
            raise ValueError("support mask must have shape [batch, Kmax]")
        population_query = self.population.predict_from_features(query_features)
        population_support = self.population.predict_from_features(
            support_features.reshape(batch * maximum_k, -1)
        ).reshape(batch, maximum_k, 2)
        support_residual = support_bp - population_support
        tokens = self.support_token(
            torch.cat([support_features, support_bp, support_residual], dim=-1)
        )

        if self.query_conditioned_weights:
            expanded_query = query_features[:, None, :].expand(-1, maximum_k, -1)
            logits = self.reliability(
                torch.cat(
                    [expanded_query, support_features, support_residual], dim=-1
                )
            ).squeeze(-1)
        else:
            logits = torch.zeros(
                batch, maximum_k, dtype=query_features.dtype, device=query_features.device
            )
        weights = self._masked_weights(logits, support_mask)
        if self.anchor_mode == "median":
            anchor_residual = self._masked_median(support_residual, support_mask)
        else:
            anchor_residual = torch.sum(weights[..., None] * support_residual, dim=1)
        context = torch.sum(weights[..., None] * tokens, dim=1)
        support_ppg_context = torch.sum(weights[..., None] * support_features, dim=1)

        personalized_query = query_features
        if self.use_film:
            gamma, beta = self.film(context).chunk(2, dim=-1)
            personalized_query = (1.0 + gamma) * personalized_query + beta
        correction_inputs = [personalized_query, context]
        if self.use_demographics:
            if demographics is None:
                raise ValueError("demographic conditioning requires demographic features")
            if self.demographic_mode == "encoded":
                correction_inputs.append(self.demographic_encoder(demographics))
            else:
                correction_inputs.append(demographics)
        correction = self.correction(torch.cat(correction_inputs, dim=-1))
        personalization = anchor_residual + correction
        if self.use_quality_gate:
            # The gate receives PPG-derived features only. It never observes the
            # query BP, query error, or reference ABP.
            gate = torch.sigmoid(
                self.quality_gate(
                    torch.cat(
                        [
                            query_features,
                            support_ppg_context,
                            (query_features - support_ppg_context).abs(),
                        ],
                        dim=-1,
                    )
                )
            )
            personalization = gate * personalization
        return population_query + personalization


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, *, rank: int = 4, alpha: float = 4.0) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = copy.deepcopy(base)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.a.weight, std=0.02)
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.b(self.a(inputs)) * self.scale


def configure_personal_adaptation(
    population: PopulationRegressor,
    mode: str,
    *,
    lora_rank: int = 4,
) -> PopulationRegressor:
    """Clone a population model and expose only the prespecified trainable subset."""

    model = copy.deepcopy(population)
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    if mode == "head":
        for parameter in model.head.parameters():
            parameter.requires_grad = True
    elif mode == "lora":
        def replace_linears(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRALinear(child, rank=lora_rank))
                else:
                    replace_linears(child)

        replace_linears(model)
    elif mode != "full":
        raise ValueError("mode must be 'head', 'full', or 'lora'")
    return model
