"""Neural baselines and variable-K personalized PPG-BP models."""

from __future__ import annotations

from collections.abc import Sequence
import copy

import torch
from torch import nn


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


class PopulationRegressor(nn.Module):
    def __init__(
        self,
        encoder: MultiScaleResNetEncoder | None = None,
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

    def __init__(self, encoder: MultiScaleResNetEncoder | None = None) -> None:
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
        if use_demographics:
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
        correction_inputs = dimension * (3 if use_demographics else 2)
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
            correction_inputs.append(self.demographic_encoder(demographics))
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
