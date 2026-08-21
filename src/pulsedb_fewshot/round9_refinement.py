"""Round-9 leakage-safe calibration-relative refinement screen.

Candidate selection is confined to meta-train cross-fit participants. Folds
0--2 fit each candidate, fold 3 controls patience-8 early stopping, and fold 4
ranks methods. The prepared cache may contain quarantined meta-validation rows,
but this screen never trains on, predicts, scores, or ranks with them. The
locked meta-test is not accessed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .phase6e_residual import KEYS
from .report_round8 import _diagnostic_rows
from .round8_calibration_relative import (
    PHYSIOLOGY_NAMES,
    SUPPORT_K,
    PreparedRound8,
    _range_labels,
    _sequence_batches,
    _tensor,
)
from .training import participant_macro_metrics, save_json, seed_everything


@dataclass(frozen=True)
class Candidate:
    prediction_mode: str = "correction"
    temporal_mode: str = "gru"
    dbp_physiology: bool = False
    bias_penalty: float = 0.0
    temporal_delta_penalty: float = 0.0
    support_consistency: float = 0.0
    personal_direction: bool = False


METHODS: dict[str, Candidate] = {
    "r8_reference": Candidate(),
    "adaptive_fusion": Candidate(prediction_mode="adaptive_fusion"),
    "range_soft_experts": Candidate(prediction_mode="range_soft_experts"),
    "dbp_specific_physiology": Candidate(dbp_physiology=True),
    "bias_regularized": Candidate(bias_penalty=0.05),
    "causal_attention": Candidate(temporal_mode="attention"),
    "personal_bp_direction": Candidate(personal_direction=True),
    "temporal_delta_consistency": Candidate(temporal_delta_penalty=0.10),
    "support_dropout_consistency": Candidate(support_consistency=0.10),
}


class TimeDecayGRU(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(dimension, dimension)
        self.log_decay = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self, inputs: torch.Tensor, gaps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, length, dimension = inputs.shape
        hidden = torch.zeros(batch, dimension, device=inputs.device, dtype=inputs.dtype)
        outputs: list[torch.Tensor] = []
        rate = torch.nn.functional.softplus(self.log_decay)
        for position in range(length):
            decayed = hidden * torch.exp(-rate * gaps[:, position : position + 1])
            proposed = self.cell(inputs[:, position], decayed)
            active = mask[:, position : position + 1]
            hidden = torch.where(active, proposed, hidden)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


class CausalWindowAttention(nn.Module):
    """Causal attention over a bounded recent history with learned time decay."""

    def __init__(self, dimension: int, window: int = 16) -> None:
        super().__init__()
        self.dimension = dimension
        self.window = window
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.output = nn.Linear(dimension, dimension)
        self.log_decay = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self, inputs: torch.Tensor, gaps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, length, _ = inputs.shape
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        elapsed = torch.cumsum(gaps.clamp_min(0.0), dim=1)
        rate = torch.nn.functional.softplus(self.log_decay)
        outputs: list[torch.Tensor] = []
        for position in range(length):
            start = max(0, position - self.window + 1)
            keys = k[:, start : position + 1]
            values = v[:, start : position + 1]
            valid = mask[:, start : position + 1]
            scores = torch.sum(q[:, position : position + 1] * keys, dim=-1)
            scores = scores / math.sqrt(self.dimension)
            age = elapsed[:, position : position + 1] - elapsed[:, start : position + 1]
            scores = scores - rate * age
            scores = scores.masked_fill(~valid, -torch.inf)
            no_history = ~valid.any(dim=1)
            if no_history.any():
                scores = scores.clone()
                scores[no_history, -1] = 0.0
            weights = torch.softmax(scores, dim=1)
            context = torch.sum(weights[..., None] * values, dim=1)
            active = mask[:, position : position + 1]
            outputs.append(torch.where(active, self.output(context), torch.zeros_like(context)))
        return torch.stack(outputs, dim=1)


class Round9Model(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        candidate: Candidate,
        *,
        physiology_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.hidden_dim = hidden_dim
        self.physiology_dim = physiology_dim
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        pair_dim = hidden_dim * 4 + 5
        self.pair = nn.Sequential(
            nn.Linear(pair_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
        )
        self.pair_delta = nn.Linear(hidden_dim, 2)
        self.attention = nn.Linear(hidden_dim, 1)

        if candidate.dbp_physiology:
            self.dbp_physiology = nn.Sequential(
                nn.Linear(physiology_dim * 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.dbp_physiology[-1].weight)
            nn.init.zeros_(self.dbp_physiology[-1].bias)

        if candidate.personal_direction:
            self.personal_relation = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )

        static_dim = hidden_dim * 2 + 6
        if candidate.personal_direction:
            static_dim += hidden_dim
        self.static = nn.Sequential(
            nn.Linear(static_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.static_correction = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.static_correction.weight)
        nn.init.zeros_(self.static_correction.bias)
        self.range_head = nn.Linear(hidden_dim, 6)

        if candidate.prediction_mode == "adaptive_fusion":
            self.fusion_gate = nn.Linear(hidden_dim, 2)
            self.fusion_residual = nn.Linear(hidden_dim, 2)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, -2.0)
            nn.init.zeros_(self.fusion_residual.weight)
            nn.init.zeros_(self.fusion_residual.bias)
        elif candidate.prediction_mode == "range_soft_experts":
            self.range_experts = nn.Linear(hidden_dim, 6)
            nn.init.zeros_(self.range_experts.weight)
            nn.init.zeros_(self.range_experts.bias)

        if candidate.temporal_mode == "gru":
            self.temporal_core: nn.Module = TimeDecayGRU(hidden_dim)
        elif candidate.temporal_mode == "attention":
            self.temporal_core = CausalWindowAttention(hidden_dim, window=16)
        else:
            raise ValueError(f"unknown temporal mode {candidate.temporal_mode}")
        self.temporal_correction = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.temporal_correction.weight)
        nn.init.zeros_(self.temporal_correction.bias)

    def _personal_direction(
        self, support: torch.Tensor, support_bp: torch.Tensor
    ) -> torch.Tensor:
        relations = []
        for left in range(SUPPORT_K):
            for right in range(left + 1, SUPPORT_K):
                ds = support[:, right] - support[:, left]
                dy = support_bp[:, right] - support_bp[:, left]
                relations.append(torch.cat([ds, ds.abs(), dy, dy.abs()], dim=-1))
        return self.personal_relation(torch.stack(relations, dim=1)).mean(dim=1)

    def static_forward(
        self,
        query_embedding: torch.Tensor,
        support_embedding: torch.Tensor,
        support_bp_norm: torch.Tensor,
        support_population_norm: torch.Tensor,
        base_norm: torch.Tensor,
        event_gap: torch.Tensor,
        query_physiology: torch.Tensor,
        support_physiology: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query = self.project(query_embedding)
        support = self.project(support_embedding)
        expanded = query[:, None, :].expand_as(support)
        pair_inputs = torch.cat(
            [
                expanded,
                support,
                expanded - support,
                (expanded - support).abs(),
                support_bp_norm,
                base_norm[:, None, :] - support_population_norm,
                event_gap[..., None],
            ],
            dim=-1,
        )
        pair_hidden = self.pair(pair_inputs)
        pair_delta = self.pair_delta(pair_hidden)
        logits = self.attention(pair_hidden).squeeze(-1)
        if support_mask is not None:
            if not support_mask.any(dim=1).all():
                raise ValueError("every example must retain at least one support event")
            logits = logits.masked_fill(~support_mask, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        pair_prediction = torch.sum(
            weights[..., None] * (support_bp_norm + pair_delta), dim=1
        )
        context = torch.sum(weights[..., None] * pair_hidden, dim=1)
        static_parts = [
            query,
            context,
            base_norm,
            pair_prediction,
            pair_prediction - base_norm,
        ]
        if self.candidate.personal_direction:
            static_parts.append(self._personal_direction(support, support_bp_norm))
        hidden = self.static(torch.cat(static_parts, dim=-1))
        range_logits = self.range_head(hidden)

        mode = self.candidate.prediction_mode
        if mode == "correction":
            prediction = base_norm + self.static_correction(hidden)
        elif mode == "adaptive_fusion":
            gate = torch.sigmoid(self.fusion_gate(hidden))
            prediction = (
                (1.0 - gate) * base_norm
                + gate * pair_prediction
                + self.fusion_residual(hidden)
            )
        elif mode == "range_soft_experts":
            probabilities = torch.softmax(range_logits.reshape(-1, 2, 3), dim=-1)
            experts = self.range_experts(hidden).reshape(-1, 2, 3)
            prediction = base_norm + torch.sum(probabilities * experts, dim=-1)
        else:
            raise ValueError(f"unknown prediction mode {mode}")

        if self.candidate.dbp_physiology:
            query_phys = query_physiology[:, None, :].expand_as(support_physiology)
            phys = torch.cat(
                [
                    query_phys,
                    support_physiology,
                    query_phys - support_physiology,
                    (query_phys - support_physiology).abs(),
                ],
                dim=-1,
            )
            dbp_support = self.dbp_physiology(phys).squeeze(-1)
            dbp_correction = torch.sum(weights * dbp_support, dim=1)
            prediction = prediction.clone()
            prediction[:, 1] = prediction[:, 1] + dbp_correction
        return {
            "prediction": prediction,
            "pair_delta": pair_delta,
            "hidden": hidden,
            "range_logits": range_logits,
        }


def _make_arrays(
    data: PreparedRound8,
    frame: pd.DataFrame,
    *,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    physiology_mean: np.ndarray,
    physiology_std: np.ndarray,
) -> dict[str, np.ndarray]:
    return data.arrays(
        frame,
        mean=target_mean,
        std=target_std,
        physiology_mean=physiology_mean,
        physiology_std=physiology_std,
    )


def _padded_batch(
    arrays: dict[str, np.ndarray], sequences: list[np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    length = max(map(len, sequences))
    batch = len(sequences)
    width = arrays["query"].shape[1]
    physiology = arrays["query_physiology"].shape[1]
    shapes = {
        "query": (batch, length, width),
        "support": (batch, length, SUPPORT_K, width),
        "support_bp": (batch, length, SUPPORT_K, 2),
        "support_population": (batch, length, SUPPORT_K, 2),
        "base": (batch, length, 2),
        "target": (batch, length, 2),
        "event_gap": (batch, length, SUPPORT_K),
        "sequence_gap": (batch, length),
        "range": (batch, length, 2),
        "query_physiology": (batch, length, physiology),
        "support_physiology": (batch, length, SUPPORT_K, physiology),
    }
    padded = {
        key: np.zeros(shape, dtype=np.int64 if key == "range" else np.float32)
        for key, shape in shapes.items()
    }
    mask = np.zeros((batch, length), dtype=bool)
    for row, indexes in enumerate(sequences):
        for key in padded:
            padded[key][row, : len(indexes)] = arrays[key][indexes]
        mask[row, : len(indexes)] = True
    return padded, mask


def _forward_sequences(
    model: Round9Model,
    arrays: dict[str, np.ndarray],
    sequences: list[np.ndarray],
    device: torch.device,
    *,
    support_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    padded, mask = _padded_batch(arrays, sequences)
    batch, length = mask.shape
    width = padded["query"].shape[-1]
    physiology = padded["query_physiology"].shape[-1]
    flat_support_mask = None
    if support_mask is not None:
        flat_support_mask = support_mask[:, None, :].expand(batch, length, SUPPORT_K)
        flat_support_mask = flat_support_mask.reshape(batch * length, SUPPORT_K)
    static = model.static_forward(
        _tensor(padded["query"].reshape(batch * length, width), device),
        _tensor(padded["support"].reshape(batch * length, SUPPORT_K, width), device),
        _tensor(padded["support_bp"].reshape(batch * length, SUPPORT_K, 2), device),
        _tensor(padded["support_population"].reshape(batch * length, SUPPORT_K, 2), device),
        _tensor(padded["base"].reshape(batch * length, 2), device),
        _tensor(padded["event_gap"].reshape(batch * length, SUPPORT_K), device),
        _tensor(padded["query_physiology"].reshape(batch * length, physiology), device),
        _tensor(
            padded["support_physiology"].reshape(
                batch * length, SUPPORT_K, physiology
            ),
            device,
        ),
        flat_support_mask,
    )
    valid = _tensor(mask, device, dtype=torch.bool)
    hidden = static["hidden"].reshape(batch, length, -1)
    temporal = model.temporal_core(
        hidden, _tensor(padded["sequence_gap"], device), valid
    )
    prediction = static["prediction"].reshape(batch, length, 2)
    prediction = prediction + model.temporal_correction(temporal)
    return {
        "prediction": prediction,
        "pair_delta": static["pair_delta"].reshape(batch, length, SUPPORT_K, 2),
        "range_logits": static["range_logits"].reshape(batch, length, 6),
        "target": _tensor(padded["target"], device),
        "support_bp": _tensor(padded["support_bp"], device),
        "range": _tensor(padded["range"], device, dtype=torch.long),
        "mask": valid,
    }


def _loss(
    output: dict[str, torch.Tensor],
    candidate: Candidate,
    *,
    consistency_prediction: torch.Tensor | None = None,
) -> torch.Tensor:
    prediction = output["prediction"]
    target = output["target"]
    mask = output["mask"]
    main = torch.nn.functional.huber_loss(
        prediction, target, reduction="none", delta=0.5
    ).mean(-1)
    pair_target = target[..., None, :] - output["support_bp"]
    pair = torch.nn.functional.huber_loss(
        output["pair_delta"], pair_target, reduction="none", delta=0.5
    ).mean((-1, -2))
    logits = output["range_logits"].reshape(*prediction.shape[:-1], 2, 3)
    ranges = output["range"]
    range_loss = 0.5 * (
        torch.nn.functional.cross_entropy(
            logits[..., 0, :].reshape(-1, 3), ranges[..., 0].reshape(-1), reduction="none"
        )
        + torch.nn.functional.cross_entropy(
            logits[..., 1, :].reshape(-1, 3), ranges[..., 1].reshape(-1), reduction="none"
        )
    ).reshape(main.shape)
    loss = main + 0.25 * pair + 0.10 * range_loss
    weights = mask.float()
    participant_loss = (loss * weights).sum(1) / weights.sum(1).clamp_min(1.0)
    total = participant_loss.mean()

    if candidate.bias_penalty:
        residual = (prediction - target) * weights[..., None]
        bias = residual.sum((0, 1)) / weights.sum().clamp_min(1.0)
        total = total + candidate.bias_penalty * torch.sum(bias**2)
    if candidate.temporal_delta_penalty:
        pair_mask = mask[:, 1:] & mask[:, :-1]
        if pair_mask.any():
            predicted_delta = prediction[:, 1:] - prediction[:, :-1]
            target_delta = target[:, 1:] - target[:, :-1]
            delta = torch.nn.functional.huber_loss(
                predicted_delta, target_delta, reduction="none", delta=0.5
            ).mean(-1)
            total = total + candidate.temporal_delta_penalty * delta[pair_mask].mean()
    if candidate.support_consistency and consistency_prediction is not None:
        consistency = torch.nn.functional.smooth_l1_loss(
            consistency_prediction, prediction.detach(), reduction="none"
        ).mean(-1)
        total = total + candidate.support_consistency * consistency[mask].mean()
    return total


def _predict(
    model: Round9Model,
    arrays: dict[str, np.ndarray],
    frame: pd.DataFrame,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions = np.zeros((len(frame), 2), dtype=np.float32)
    with torch.no_grad():
        for sequences in _sequence_batches(frame, batch_size=12):
            output = _forward_sequences(model, arrays, sequences, device)
            values = output["prediction"].cpu().numpy()
            for row, indexes in enumerate(sequences):
                predictions[indexes] = values[row, : len(indexes)]
    return predictions


def _scored_predictions(
    frame: pd.DataFrame,
    prediction_norm: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> pd.DataFrame:
    prediction = prediction_norm * target_std + target_mean
    result = frame[KEYS + ["source", "target_sbp", "target_dbp"]].copy()
    result["pred_sbp"] = prediction[:, 0]
    result["pred_dbp"] = prediction[:, 1]
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact deterministic Markdown table without optional packages."""
    columns = list(frame.columns)

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def train_screen(
    *, prepared: Path, output: Path, method: str, seed: int
) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unknown Round-9 method {method}")
    if output.exists():
        raise FileExistsError(output)
    candidate = METHODS[method]
    data = PreparedRound8(prepared)
    train = data.queries.loc[data.queries["round8_role"].eq("train")].copy()
    fit = train.loc[train["fold"].isin([0, 1, 2])].copy()
    early = train.loc[train["fold"].eq(3)].copy()
    selection = train.loc[train["fold"].eq(4)].copy()
    for frame in (fit, early, selection):
        frame.sort_values(["subject_uid", "event_index", "event_id"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    participant_sets = [set(x["subject_uid"].astype(str)) for x in (fit, early, selection)]
    if any(participant_sets[a] & participant_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("Round-9 internal participant roles overlap")

    target_mean = fit[["target_sbp", "target_dbp"]].to_numpy(np.float32).mean(0)
    target_std = fit[["target_sbp", "target_dbp"]].to_numpy(np.float32).std(0)
    target_std[target_std <= 1e-6] = 1.0
    fit_qrows = fit["query_embedding_row"].to_numpy(np.int64)
    fit_srows = fit["support_row"].to_numpy(np.int64)
    physiology_pool = np.concatenate(
        [
            np.asarray(data.query_physiology[fit_qrows]),
            np.asarray(data.support_physiology[np.unique(fit_srows)]).reshape(
                -1, data.support_physiology.shape[-1]
            ),
        ]
    ).astype(np.float32)
    physiology_mean = physiology_pool.mean(0)
    physiology_std = physiology_pool.std(0)
    physiology_std[physiology_std <= 1e-6] = 1.0
    arrays_fit = _make_arrays(
        data, fit, target_mean=target_mean, target_std=target_std,
        physiology_mean=physiology_mean, physiology_std=physiology_std
    )
    arrays_early = _make_arrays(
        data, early, target_mean=target_mean, target_std=target_std,
        physiology_mean=physiology_mean, physiology_std=physiology_std
    )
    arrays_selection = _make_arrays(
        data, selection, target_mean=target_mean, target_std=target_std,
        physiology_mean=physiology_mean, physiology_std=physiology_std
    )

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Round-9 screening requires CUDA")
    model = Round9Model(
        data.query_embeddings.shape[1], candidate,
        physiology_dim=data.query_physiology.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_score = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    batches = _sequence_batches(fit, batch_size=12)
    rng = np.random.default_rng(seed)
    for epoch in itertools.count(1):
        model.train()
        rng.shuffle(batches)
        total = 0.0
        for sequences in batches:
            optimizer.zero_grad(set_to_none=True)
            full = _forward_sequences(model, arrays_fit, sequences, device)
            consistency = None
            if candidate.support_consistency:
                support_mask = torch.ones(
                    len(sequences), SUPPORT_K, dtype=torch.bool, device=device
                )
                dropped = torch.as_tensor(
                    rng.integers(0, SUPPORT_K, size=len(sequences)), device=device
                )
                support_mask[torch.arange(len(sequences), device=device), dropped] = False
                consistency = _forward_sequences(
                    model, arrays_fit, sequences, device, support_mask=support_mask
                )["prediction"]
            loss = _loss(full, candidate, consistency_prediction=consistency)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
        early_prediction = _predict(model, arrays_early, early, device)
        early_scored = _scored_predictions(
            early, early_prediction, target_mean, target_std
        )
        early_score = float(participant_macro_metrics(early_scored)["mean_mae"])
        history.append({
            "epoch": epoch,
            "train_batch_loss_sum": total,
            "fold3_participant_macro_mean_mae": early_score,
        })
        if early_score < best_score:
            best_score = early_score
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= 8:
            break
    if best_state is None:
        raise RuntimeError("Round-9 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    selection_prediction = _predict(model, arrays_selection, selection, device)
    predictions = _scored_predictions(
        selection, selection_prediction, target_mean, target_std
    )
    metrics = {
        scope: participant_macro_metrics(group)
        for scope, group in [("Overall", predictions)]
        + [
            (source, predictions.loc[predictions["source"].eq(source)])
            for source in sorted(predictions["source"].unique())
        ]
    }
    output.mkdir(parents=True)
    predictions.to_parquet(output / "selection_predictions.parquet", index=False)
    torch.save({
        "model_state": best_state,
        "method": method,
        "candidate": candidate.__dict__,
        "target_mean": target_mean,
        "target_std": target_std,
        "physiology_mean": physiology_mean,
        "physiology_std": physiology_std,
    }, output / "best.pt")
    save_json(output / "history.json", history)
    payload = {
        "status": "complete",
        "round": 9,
        "method": method,
        "seed": seed,
        "k": SUPPORT_K,
        "split": "meta_train_internal_fold4",
        "fit_folds": [0, 1, 2],
        "early_stopping_fold": 3,
        "selection_fold": 4,
        "meta_validation_used_for_training": False,
        "meta_validation_used_for_early_stopping": False,
        "meta_validation_used_for_candidate_ranking": False,
        "meta_validation_predictions_generated": False,
        "locked_test_accessed": False,
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "source_model_input": False,
        "participants": int(predictions["subject_uid"].nunique()),
        "queries": int(len(predictions)),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stop_reason": "early_stopping_patience_8",
        "candidate": candidate.__dict__,
        "metrics": metrics,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output / "run.json", payload)
    return payload


def build_internal_report(
    *, runs: dict[str, Path], reference: str, output: Path, expected_seed: int
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    canonical: pd.DataFrame | None = None
    for setting, root in runs.items():
        payload = json.loads((root / "run.json").read_text())
        if payload.get("status") != "complete":
            raise AssertionError(f"{setting} is incomplete")
        if payload.get("split") != "meta_train_internal_fold4":
            raise AssertionError(f"{setting} has wrong selection split")
        if payload.get("seed") != expected_seed:
            raise AssertionError(f"{setting} has wrong seed")
        for key in (
            "meta_validation_used_for_training",
            "meta_validation_used_for_early_stopping",
            "meta_validation_used_for_candidate_ranking",
            "meta_validation_predictions_generated",
        ):
            if payload.get(key) is not False:
                raise AssertionError(f"{setting} has unsafe meta-validation flag {key}")
        if payload.get("locked_test_accessed") is not False:
            raise AssertionError(f"{setting} accessed locked test")
        predictions = pd.read_parquet(root / "selection_predictions.parquet")
        prediction_frames.append(predictions.assign(Setting=setting))
        check = predictions[KEYS + ["target_sbp", "target_dbp"]].sort_values(KEYS)
        check.reset_index(drop=True, inplace=True)
        if canonical is None:
            canonical = check
        elif not canonical.equals(check):
            raise AssertionError(f"{setting} has a different selection query set")
        for scope, group in [("Overall", predictions)] + [
            (source, predictions.loc[predictions["source"].eq(source)])
            for source in sorted(predictions["source"].unique())
        ]:
            metric = participant_macro_metrics(group)
            records.append({
                "Setting": setting,
                "Scope": scope,
                "N participants": int(group["subject_uid"].nunique()),
                "N queries": int(len(group)),
                "SBP participant-macro MAE": metric["sbp_mae"],
                "DBP participant-macro MAE": metric["dbp_mae"],
                "Mean participant-macro MAE": metric["mean_mae"],
            })
    table = pd.DataFrame(records)
    settings = list(runs)
    diagnostics = _diagnostic_rows(
        pd.concat(prediction_frames, ignore_index=True, sort=False), settings
    )
    overall = table.loc[table["Scope"].eq("Overall")].sort_values(
        ["Mean participant-macro MAE", "Setting"], kind="mergesort"
    )
    winner = str(overall.iloc[0]["Setting"])
    if reference not in runs:
        raise KeyError(f"reference setting {reference} is absent")
    ref = table.loc[table["Setting"].eq(reference)].set_index("Scope")
    win = table.loc[table["Setting"].eq(winner)].set_index("Scope")
    gains = {
        scope: float(
            ref.loc[scope, "Mean participant-macro MAE"]
            - win.loc[scope, "Mean participant-macro MAE"]
        )
        for scope in ("Overall", "MIMIC", "VitalDB")
    }
    passes = gains["Overall"] >= 0.15 and gains["MIMIC"] > 0 and gains["VitalDB"] > 0
    comparison_rows: list[dict[str, object]] = []
    for setting in settings:
        candidate = table.loc[table["Setting"].eq(setting)].set_index("Scope")
        for scope in ("Overall", "MIMIC", "VitalDB"):
            comparison_rows.append(
                {
                    "Setting": setting,
                    "Scope": scope,
                    "Reference mean participant-macro MAE": float(
                        ref.loc[scope, "Mean participant-macro MAE"]
                    ),
                    "Candidate mean participant-macro MAE": float(
                        candidate.loc[scope, "Mean participant-macro MAE"]
                    ),
                    "Candidate minus reference": float(
                        candidate.loc[scope, "Mean participant-macro MAE"]
                        - ref.loc[scope, "Mean participant-macro MAE"]
                    ),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "participant_macro_internal.csv", index=False)
    diagnostics.to_csv(output / "pooled_diagnostics_internal.csv", index=False)
    comparison.to_csv(output / "comparison_vs_reference_internal.csv", index=False)
    summary = {
        "status": "complete",
        "round": 9,
        "seed": expected_seed,
        "split": "meta_train_internal_fold4",
        "meta_validation_used_for_training": False,
        "meta_validation_used_for_early_stopping": False,
        "meta_validation_used_for_candidate_ranking": False,
        "meta_validation_predictions_generated": False,
        "locked_test_accessed": False,
        "reference": reference,
        "winner": winner,
        "gain_vs_reference": gains,
        "passes_internal_gate": passes,
        "candidate_count": len(runs),
    }
    save_json(output / "selection.json", summary)
    lines = [
        "# Round-9 internal method screen",
        "",
        "Candidate ranking uses meta-train fold 4 only. Meta-validation was not used for training, early stopping, prediction, scoring, or candidate ranking; the locked meta-test was not accessed.",
        "",
        _markdown_table(
            table.sort_values(["Scope", "Mean participant-macro MAE"])
        ),
        "",
        "## Change versus the internal reference",
        "",
        "Negative candidate-minus-reference values are better.",
        "",
        _markdown_table(comparison),
        "",
        "## Event-pooled diagnostic metrics",
        "",
        "Participant-macro MAE above is primary. AAMI/BHS entries below are retrospective numerical screens only and do not establish device compliance.",
        "",
        _markdown_table(diagnostics),
        "",
        f"Internal winner: **{winner}**.",
        f"Internal promotion gate passed: **{passes}**.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _parse_runs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--run must use SETTING=PATH")
        setting, path = value.split("=", 1)
        if not setting or setting in result:
            raise ValueError(f"invalid or duplicate setting {setting}")
        result[setting] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-screen")
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--method", choices=sorted(METHODS), required=True)
    train.add_argument("--seed", type=int, default=20260823)
    report = commands.add_parser("report-internal")
    report.add_argument("--run", action="append", required=True)
    report.add_argument("--reference", required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--expected-seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "train-screen":
        result = train_screen(
            prepared=args.prepared, output=args.output,
            method=args.method, seed=args.seed
        )
    else:
        result = build_internal_report(
            runs=_parse_runs(args.run), reference=args.reference,
            output=args.output, expected_seed=args.expected_seed
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
