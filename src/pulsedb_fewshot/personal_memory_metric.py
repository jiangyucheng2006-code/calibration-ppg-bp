"""E2: train-only BP-state metric for frozen registered-user memory retrieval.

This is an application of supervised metric learning, not a novel invention of
metric learning. A bias-free 256->32 projection regresses same-person cosine
distance to g/(1+g), where g is mean squared BP difference divided by the saved
train-only SBP/DBP standard deviations. Half the sampled pairs are original-PPG
near neighbours; half are balanced across train-only temporal-distance strata.
Near-waveform, large-BP-gap TRAIN pairs get weight 3 (otherwise 1).

The complete v1 40-window train-block protection remains in force. Only the
retrieval representation changes: bank labels, k=5, softmax temperature=0.1,
and every query's original 256D distance-alpha are frozen. There is no relation
network, learned gate, E1 sampling modification or query-label retrieval input.
Validation BP can select a checkpoint, never choose neighbours or pair strata.
The source encoder saw train labels: this is NOT encoder-level OOF training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .calbased_metrics import participant_macro_views, pooled_diagnostics
from .personal_memory_prepare import FIELDS, audit_metadata, validate_export_manifest
from .personal_memory_train import PROTOCOL_ID, READ_ROLES, load_cache
from .training import file_sha256, seed_everything, source_tree_sha256


SCREEN_ID = "personal-memory-v2"
CANDIDATE = "E2_bp_state_metric"
MODES = ("random_disjoint", "chronological_blocked")


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


class BPStateProjection(nn.Module):
    def __init__(self, input_dim=256, output_dim=32):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def forward(self, features):
        if features.ndim < 2 or features.shape[-1] != self.projection.in_features:
            raise ValueError("projection feature dimension mismatch")
        return F.normalize(self.projection(F.normalize(features, dim=-1)), dim=-1, eps=1e-12)


def metric_objective(model, query_features, reference_features, query_bp, reference_bp,
                     target_std, close_pair):
    """All BP arguments are train-role labels; no validation-label interface."""
    if target_std.shape != (2,) or not bool(torch.isfinite(target_std).all()) or bool((target_std <= 0).any()):
        raise ValueError("positive train-only BP scales required")
    zq, zr = model(query_features), model(reference_features)
    distance = ((1 - (zq * zr).sum(-1).clamp(-1, 1)) * .5).clamp(0, 1)
    gap_squared = (((query_bp - reference_bp) / target_std) ** 2).mean(-1)
    target = gap_squared / (1 + gap_squared)
    hard = close_pair & (gap_squared >= 1)
    weight = 1 + 2 * hard.to(distance.dtype)
    losses = F.smooth_l1_loss(distance, target, reduction="none", beta=.1)
    loss = (losses * weight).sum() / weight.sum()
    return loss, {"hard_pairs": hard.sum().detach(), "mean_target": target.mean().detach()}


def _blocks(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row["recording_uid"], row["time_axis_uid"])].append(i)
    block = np.full(len(rows), -1, dtype=np.int64)
    offset = 0
    for key in sorted(groups):
        order = sorted(groups[key], key=lambda i: (rows[i]["start_s"], rows[i]["window_uid"]))
        for rank, index in enumerate(order):
            block[index] = offset + rank // 40
        offset += (len(order) + 39) // 40
    return block


def legal_mask(bank_rows, query_rows, mode, *, training=False):
    """Label-free same-person donor mask, exactly retaining v1 block policy.

    Comparable past clocks in chronological mode follow v1 (the clock encodes
    comparability); physiological overlap is checked within the same record.
    """
    if mode not in MODES or not bank_rows or not query_rows:
        raise ValueError("valid mode and nonempty role metadata required")
    source = {row["source"] for row in bank_rows}
    person = {row["subject_uid"] for row in bank_rows}
    if len(source) != 1 or len(person) != 1:
        raise ValueError("one bank group must contain one source/person")
    if any(row["subject_uid"] not in person or row["source"] not in source for row in query_rows):
        raise ValueError("cross-person/source references are forbidden")
    if any(row["role"] != "train" for row in bank_rows):
        raise ValueError("memory bank must be train role")
    expected = "train" if training else "internal_validation"
    if any(row["role"] != expected for row in query_rows):
        raise ValueError("query role mismatch")
    bstart = np.array([row["start_s"] for row in bank_rows])
    bend = np.array([row["end_s"] for row in bank_rows])
    qstart = np.array([row["start_s"] for row in query_rows])
    qend = np.array([row["end_s"] for row in query_rows])
    axis = np.array([row["time_axis_uid"] for row in query_rows])[:, None] == np.array([row["time_axis_uid"] for row in bank_rows])[None]
    record = np.array([row["recording_uid"] for row in query_rows])[:, None] == np.array([row["recording_uid"] for row in bank_rows])[None]
    same_window = np.array([row["window_uid"] for row in query_rows])[:, None] == np.array([row["window_uid"] for row in bank_rows])[None]
    same_hash = np.array([row["waveform_sha256"] for row in query_rows])[:, None] == np.array([row["waveform_sha256"] for row in bank_rows])[None]
    overlap = axis & record & (bstart[None] < qend[:, None] - 1e-7) & (bend[None] > qstart[:, None] + 1e-7)
    legal = ~(same_window | same_hash | overlap)
    if training:
        block = _blocks(bank_rows)
        lookup = {row["window_uid"]: i for i, row in enumerate(bank_rows)}
        if any(row["window_uid"] not in lookup for row in query_rows):
            raise ValueError("train query is absent from its own bank")
        qblock = np.array([block[lookup[row["window_uid"]]] for row in query_rows])
        legal &= qblock[:, None] != block[None]
    if mode == "chronological_blocked":
        legal &= axis & (bend[None] <= qstart[:, None] + 1e-7)
    return legal


def build_layout(train_rows, validation_rows, mode, cached_train_neighbours, cached_validation_neighbours=None):
    """Build label-free legal layout and temporal-stratified train pair pools.

    The four comparable-clock distance quartiles are computed per train query,
    using only legal donors. A fifth explicit unknown-clock stratum is allowed
    in random mode; unknown times are never assigned fabricated seconds.
    Eight spread donors per stratum bound storage; repeated slots only fill a
    small stratum for sampling and never appear as repeated inference donors.
    """
    audited = audit_metadata(train_rows, validation_rows)
    groups, queries = defaultdict(list), defaultdict(list)
    for i, row in enumerate(train_rows):
        groups[row["subject_uid"]].append(i)
    for i, row in enumerate(validation_rows):
        queries[row["subject_uid"]].append(i)
    if set(groups) != set(queries):
        raise ValueError("both roles must retain the same registered people")
    bank_orders, query_orders, val_masks = [], [], []
    temporal = np.full((len(train_rows), 5, 8), -1, dtype=np.int64)
    for subject in sorted(groups):
        # Stable tie order matches the original window_uid ascending rule.
        bi = np.array(sorted(groups[subject], key=lambda i: train_rows[i]["window_uid"]), dtype=np.int64)
        qi = np.array(queries[subject], dtype=np.int64)
        bank = [train_rows[i] for i in bi]
        query = [validation_rows[i] for i in qi]
        train_legal = legal_mask(bank, bank, mode, training=True)
        valid_legal = legal_mask(bank, query, mode)
        local_lookup = {global_i: i for i, global_i in enumerate(bi)}
        if cached_validation_neighbours is not None:
            for local, global_i in enumerate(qi):
                for donor in cached_validation_neighbours[global_i]:
                    if donor >= 0 and (int(donor) not in local_lookup or not valid_legal[local, local_lookup[int(donor)]]):
                        raise ValueError("original validation reference violates source/time legality")
        for local, global_i in enumerate(bi):
            cached = cached_train_neighbours[global_i]
            for donor in cached[cached >= 0]:
                if int(donor) not in local_lookup or not train_legal[local, local_lookup[int(donor)]]:
                    raise ValueError("cached close pair violates current block/time legality")
            donors = np.flatnonzero(train_legal[local])
            if not len(donors):
                continue
            row = bank[local]
            comparable = np.array([bank[j]["time_axis_uid"] == row["time_axis_uid"] for j in donors])
            timed = donors[comparable]
            if len(timed):
                gap = np.array([max(bank[j]["start_s"] - row["end_s"], row["start_s"] - bank[j]["end_s"]) for j in timed])
                timed = timed[np.argsort(gap, kind="stable")]
                strata = list(np.array_split(timed, 4))
            else:
                strata = [np.array([], dtype=np.int64) for _ in range(4)]
            strata.append(donors[~comparable])
            for stratum, donor_group in enumerate(strata):
                if len(donor_group):
                    selected = donor_group[np.linspace(0, len(donor_group) - 1, 8, dtype=int)]
                    temporal[global_i, stratum] = bi[selected]
        bank_orders.append(bi)
        query_orders.append(qi)
        val_masks.append(valid_legal)
    if len({len(x) for x in bank_orders}) != 1 or len({len(x) for x in query_orders}) != 1:
        raise ValueError("batched production retrieval requires equal per-person role sizes")
    valid_query = (cached_train_neighbours >= 0).all(axis=1) & (temporal >= 0).any(axis=(1, 2))
    if not valid_query.any():
        raise ValueError("no train query has sufficient legal history after full 40-window exclusion")
    return {"bank_order": np.stack(bank_orders), "query_order": np.stack(query_orders),
            "validation_legal": np.stack(val_masks), "temporal_pool": temporal,
            "eligible_train": np.flatnonzero(valid_query), "metadata_audit": audited,
            "temporal_strata": ["distance_quartile_1", "distance_quartile_2", "distance_quartile_3", "distance_quartile_4", "unknown_clock"]}


def sample_training_pairs(tensors, size, generator):
    eligible = tensors["eligible_train"]
    query = eligible[torch.randint(len(eligible), (size,), device=eligible.device, generator=generator)]
    close = torch.rand(size, device=eligible.device, generator=generator) < .5
    near_pool = tensors["train_indices"][query]
    near_slot = torch.randint(5, (size,), device=eligible.device, generator=generator)
    near = near_pool.gather(1, near_slot[:, None])[:, 0]
    temporal = tensors["temporal_pool"][query]
    # Uniform choice among available strata, then among its stored donors.
    stratum_valid = temporal[:, :, 0] >= 0
    random_scores = torch.rand(stratum_valid.shape, device=eligible.device, generator=generator).masked_fill(~stratum_valid, -1)
    stratum = random_scores.argmax(1)
    within = torch.randint(8, (size,), device=eligible.device, generator=generator)
    distant = temporal[torch.arange(size, device=eligible.device), stratum, within]
    donor = torch.where(close, near, distant)
    if bool((donor < 0).any()) or bool((donor == query).any()):
        raise ValueError("sampled train pair is invalid/self")
    return query, donor, close


def fixed_alpha_fusion(reference_bp, indices, weights, base_prediction, original_alpha):
    """Inference accepts reference BP only; query targets are not arguments."""
    valid = (indices >= 0).all(-1)
    safe = indices.clamp_min(0)
    memory = (reference_bp[safe] * weights[..., None]).sum(-2)
    memory = torch.where(valid[..., None], memory, base_prediction)
    alpha = original_alpha * valid.to(original_alpha.dtype)
    prediction = base_prediction + alpha[..., None] * (memory - base_prediction)
    return prediction, memory


@torch.no_grad()
def retrieve(model, tensors, *, subjects_per_batch=32):
    """Vectorized same-person top5; only frozen features and lineage masks."""
    model.eval()
    train_z = model(tensors["train_features"])
    validation_z = model(tensors["validation_features"])
    count = len(validation_z)
    indices = torch.full((count, 5), -1, dtype=torch.long, device=train_z.device)
    weights = torch.zeros((count, 5), dtype=train_z.dtype, device=train_z.device)
    for start in range(0, len(tensors["bank_order"]), subjects_per_batch):
        bank = tensors["bank_order"][start:start + subjects_per_batch]
        query = tensors["query_order"][start:start + subjects_per_batch]
        mask = tensors["validation_legal"][start:start + subjects_per_batch]
        scores = torch.bmm(validation_z[query], train_z[bank].transpose(1, 2)).clamp(-1, 1)
        scores = scores.masked_fill(~mask, -torch.inf)
        # Stable sort preserves the canonical window_uid tie break.
        chosen = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :5]
        selected = bank[:, None, :].expand(-1, query.shape[1], -1).gather(2, chosen)
        covered = mask.sum(-1) >= 5
        selected = torch.where(covered[..., None], selected, -1)
        chosen_scores = scores.gather(2, chosen)
        chosen_scores = torch.where(covered[..., None], chosen_scores, torch.zeros_like(chosen_scores))
        selected_weight = torch.softmax(chosen_scores / .1, dim=-1) * covered[..., None]
        indices[query.reshape(-1)] = selected.reshape(-1, 5)
        weights[query.reshape(-1)] = selected_weight.reshape(-1, 5)
    return indices, weights


@torch.no_grad()
def prediction_values(model, tensors, *, subjects_per_batch=32):
    indices, weights = retrieve(model, tensors, subjects_per_batch=subjects_per_batch)
    prediction, memory = fixed_alpha_fusion(tensors["train_bp"], indices, weights,
                                          tensors["validation_base"], tensors["validation_support"])
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(memory).all()):
        raise FloatingPointError("non-finite retrieval/fusion prediction")
    return prediction, memory, indices, weights


def prediction_frame(metadata, values):
    result = metadata[["subject_uid", "event_id", "source", "target_sbp", "target_dbp"]].copy()
    array = values.detach().cpu().numpy()
    result["pred_sbp"], result["pred_dbp"] = array[:, 0], array[:, 1]
    return result


def _tensors(arrays, layout, device):
    data = {key: torch.as_tensor(np.array(value, copy=True), device=device,
                                dtype=torch.long if key.endswith("indices") else torch.float32)
            for key, value in arrays.items() if key not in {"validation_bp", "train_weights", "train_support"}}
    for key in ("bank_order", "query_order", "validation_legal", "temporal_pool", "eligible_train"):
        # CPU as_tensor shares NumPy storage by default. Each execution owns
        # its masks/pools: a fallback diagnostic must not mutate the audited
        # layout or contaminate another inference/test invocation.
        data[key] = torch.as_tensor(np.array(layout[key], copy=True), device=device,
                                    dtype=torch.bool if key == "validation_legal" else torch.long)
    return data


def train(args):
    started = time.monotonic()
    smoke_check = bool(getattr(args, "smoke_check", False))
    if args.smoke and smoke_check:
        raise ValueError("synthetic smoke and real-cache smoke-check are distinct")
    if smoke_check:
        args.epochs, args.examples_per_epoch, args.batch_size = 1, 1024, 256
    if not (args.smoke or smoke_check) and (args.epochs != 0 or args.patience != 8 or args.seed != 20260907 or
                           args.batch_size != 256 or args.examples_per_epoch != 200000 or args.learning_rate != 3e-4):
        raise ValueError("production requires seed20260907, no epoch cap, patience8, batch256, 200000 exposures")
    if min(args.batch_size, args.examples_per_epoch, args.patience, args.subjects_per_batch) < 1 or args.epochs < 0:
        raise ValueError("invalid training controls")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing to overwrite experiment output")
    manifest_hash = file_sha256(args.cache_dir / "manifest.json")
    manifest, arrays, metadata = load_cache(args.cache_dir, "pair_distance_blend", args.smoke)
    if not args.smoke:
        validate_export_manifest({**manifest, "status": "exported"})
        if manifest.get("screen_id") != "personal-memory-v1" or manifest.get("preparation", {}).get("train_excluded_block_size") != 40:
            raise ValueError("E2 requires the original complete 40-window-block v1 cache")
    # Narrow the metadata passed to retrieval: query targets are omitted.
    layout = build_layout(metadata["train"][list(FIELDS)].to_dict("records"),
                          metadata["validation"][list(FIELDS)].to_dict("records"),
                          manifest["split_mode"], arrays["train_indices"], arrays["validation_indices"])
    if file_sha256(args.cache_dir / "manifest.json") != manifest_hash:
        raise ValueError("cache manifest changed while loading")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable")
    seed_everything(args.seed)
    tensors = _tensors(arrays, layout, device)
    del arrays
    std = torch.tensor(manifest["target_scaler"]["std"], dtype=torch.float32, device=device)
    model = BPStateProjection().to(device)
    if sum(p.numel() for p in model.parameters()) != 8192:
        raise AssertionError("fixed 256x32 bias-free projection must have 8192 parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    original_value, original_memory = fixed_alpha_fusion(tensors["train_bp"], tensors["validation_indices"],
        tensors["validation_weights"], tensors["validation_base"], tensors["validation_support"])
    original_frame = prediction_frame(metadata["validation"], original_value)
    original_metrics = participant_macro_views(original_frame)
    original_frame.to_parquet(args.output / "original_fixed_blend_predictions.parquet", index=False)
    pooled_diagnostics(original_frame, "original_256d_fixed_blend").to_csv(args.output / "original_fixed_blend_pooled.csv", index=False)
    best_value, best_memory, best_indices, best_weights = prediction_values(model, tensors, subjects_per_batch=args.subjects_per_batch)
    best_frame = prediction_frame(metadata["validation"], best_value)
    best_metrics = participant_macro_views(best_frame)
    initial_metrics = best_metrics
    best_frame.to_parquet(args.output / "initial_internal_validation_predictions.parquet", index=False)
    best_score, best_epoch, epoch, stale, steps = best_metrics["Overall"]["mean_mae"], 0, 0, 0, 0
    history = []
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def checkpoint():
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "epoch": best_epoch, "candidate": CANDIDATE, "seed": args.seed,
                    "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
                    "cache_manifest_sha256": manifest_hash, "target_scaler": manifest["target_scaler"],
                    "alpha_source": "unchanged_v1_original256d_query_support_weight", "metrics": best_metrics,
                    "frozen_personal_state_is_in_source_checkpoint": True}, args.output / "best.pt")

    checkpoint()
    print(json.dumps({"initial_epoch": 0, "initial_projection_metrics": initial_metrics,
                      "original_fixed_blend_metrics": original_metrics}), flush=True)
    while args.epochs == 0 or epoch < args.epochs:
        epoch += 1
        model.train()
        epoch_start, loss_total, hard_total, examples = time.monotonic(), 0., 0, 0
        for start in range(0, args.examples_per_epoch, args.batch_size):
            size = min(args.batch_size, args.examples_per_epoch - start)
            query, donor, close = sample_training_pairs(tensors, size, generator)
            optimizer.zero_grad(set_to_none=True)
            loss, details = metric_objective(model, tensors["train_features"][query], tensors["train_features"][donor],
                tensors["train_bp"][query], tensors["train_bp"][donor], std, close)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite metric loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            examples += size
            loss_total += float(loss.detach()) * size
            hard_total += int(details["hard_pairs"])
        value, memory, indices, weights = prediction_values(model, tensors, subjects_per_batch=args.subjects_per_batch)
        frame = prediction_frame(metadata["validation"], value)
        metrics = participant_macro_views(frame)
        score = float(metrics["Overall"]["mean_mae"])
        improved = score < best_score
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            best_frame, best_metrics, best_memory, best_indices, best_weights = frame, metrics, memory, indices, weights
            checkpoint()
        else:
            stale += 1
        item = {"epoch": epoch, "train_metric_loss": loss_total / examples, "hard_training_pairs": hard_total,
                "examples": examples, "optimizer_steps_total": steps, "epoch_seconds": time.monotonic() - epoch_start,
                "internal_validation": metrics, "improved": improved, "stale_epochs": stale}
        history.append(item)
        save_json(args.output / "history.json", history)
        print(json.dumps(item), flush=True)
        if stale >= args.patience:
            break
    best_frame.to_parquet(args.output / "best_internal_validation_predictions.parquet", index=False)
    memory_frame = prediction_frame(metadata["validation"], best_memory)
    memory_frame.to_parquet(args.output / "best_memory_only_predictions.parquet", index=False)
    pooled_diagnostics(best_frame, CANDIDATE).to_csv(args.output / "internal_validation_pooled_diagnostics.csv", index=False)
    pooled_diagnostics(memory_frame, CANDIDATE + "_memory_only_diagnostic").to_csv(args.output / "memory_only_pooled_diagnostics.csv", index=False)
    np.save(args.output / "private_best_validation_indices.npy", best_indices.cpu().numpy(), allow_pickle=False)
    np.save(args.output / "private_best_validation_weights.npy", best_weights.cpu().numpy(), allow_pickle=False)
    save_json(args.output / "metrics.json", best_metrics)
    run = {"status": "synthetic_smoke_complete" if args.smoke else "real_smoke" if smoke_check else "complete", "screen_id": SCREEN_ID,
           "candidate": CANDIDATE, "protocol_id": PROTOCOL_ID, "seed": args.seed,
           "split_mode": manifest["split_mode"], "source_parent_split": "meta_train", "read_roles": READ_ROLES,
           "selection_role": "internal_validation", "heldout_test_accessed": False,
           "official_pulsedb_calbased_reproduction": False, "trainable_parameters": 8192,
           "new_participant_trainable_parameters": 0, "persistent_LoRA_preserved": True,
           "feature_provenance": "frozen_supervised_train_encoder_not_OOF",
           "training_objective": "cosine_distance_regression_to_bounded_train_scaled_same_person_BP_gap",
           "pair_sampling": "half_cached_PPG_near_half_uniform_available_temporal_quartile_or_unknown_clock",
           "hard_pair_weight": "3_if_PPGnear_and_train_mean_squared_scaled_BPgap_ge1_else1",
           "training_guard": "original40window_block_plus_self_content_overlap_and_comparable_past_for_chrono",
           "query_BP_used_for_retrieval": False, "fixed_alpha": "unchanged_original256d_cache_support_weight",
           "gate_recalibrated": False, "relation_correction": "zero", "retrieved_references": 5,
           "softmax_temperature": .1, "validation_query_coverage": 1.,
           "train_participants": int(metadata["train"].subject_uid.nunique()), "train_windows_available": len(metadata["train"]),
           "internal_validation_participants": int(metadata["validation"].subject_uid.nunique()),
           "internal_validation_windows": len(metadata["validation"]),
           "personal_label_budget_windows": manifest.get("personal_training_budget", manifest.get("personal_label_budget_windows", 320)),
           "eligible_train_queries_for_pair_loss": len(layout["eligible_train"]),
           "fallback_validation_windows": int((best_indices < 0).all(-1).sum()),
           "metrics": best_metrics, "memory_only_metrics": participant_macro_views(memory_frame),
           "initial_random_projection_metrics": initial_metrics, "original_fixed_blend_metrics": original_metrics,
           "epochs_completed": epoch, "best_epoch": best_epoch, "optimizer_steps": steps,
           "examples_processed": epoch * args.examples_per_epoch,
           "stop_reason": "early_stopping" if stale >= args.patience else "real_smoke_four_steps" if smoke_check else "synthetic_smoke_epoch_cap",
           "runtime_seconds": time.monotonic() - started, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
           "cache_manifest_sha256": manifest_hash, "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
           "checkpoint_sha256": file_sha256(args.output / "best.pt"),
           "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
           "metadata_audit": layout["metadata_audit"], "device": str(device),
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           "standard_claim": "retrospective pooled numerical screen_not_clinical_certification",
           "novelty_claim": "supervised_metric_learning_application_not_invention_of_metric_learning",
           "memory_storage_claim": "original256d_and_projected32d_both_kept_no_total_storage_reduction_claim"}
    save_json(args.output / "run.json", run)
    return run


def make_synthetic_cache(root: Path, split_mode="random_disjoint"):
    """Small explicit synthetic cache, not derived from any patient signals."""
    from .personal_memory_prepare import prepare_neighbors

    if root.exists() and any(root.iterdir()):
        raise FileExistsError("synthetic cache directory already populated")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    rows, data = {}, {}
    for role, count in (("train", 80), ("validation", 4)):
        records, features, targets = [], [], []
        for person in range(4):
            source = "MIMIC" if person < 2 else "VitalDB"
            subject = f"{source}:synthetic-{person}"
            record = f"{source}:synthetic-record-{person}"
            for within in range(count):
                uid = f"{subject}-{role}-{within:03}"
                start = (within if role == "train" else 85 + within) * 20.
                sbp = 120 + person * 5 + 12 * np.sin(within * .13)
                dbp = 75 + person * 2 + 7 * np.sin(within * .11)
                feature = rng.normal(size=256).astype(np.float32)
                feature[:2] = [sbp / 20, dbp / 10]
                records.append({"subject_uid": subject, "event_id": uid, "source": source,
                    "window_uid": uid, "waveform_sha256": hashlib.sha256(uid.encode()).hexdigest(),
                    "recording_uid": record, "time_axis_uid": record, "start_s": start, "end_s": start + 10,
                    "role": "train" if role == "train" else "internal_validation", "target_sbp": sbp, "target_dbp": dbp})
                features.append(feature)
                targets.append([sbp, dbp])
        rows[role] = records
        data[f"{role}_features"] = np.array(features, dtype=np.float32)
        data[f"{role}_bp"] = np.array(targets, dtype=np.float32)
        data[f"{role}_base"] = data[f"{role}_bp"] + np.array([3., -2.], dtype=np.float32)
    neighbours = prepare_neighbors(data["train_features"], data["validation_features"], rows["train"], rows["validation"], mode=split_mode)
    manifest = {"status": "complete", "protocol_id": PROTOCOL_ID, "screen_id": "personal-memory-v1",
                "split_mode": split_mode, "source_parent_split": "meta_train", "read_roles": READ_ROLES,
                "heldout_test_accessed": False, "synthetic_smoke": True, "personal_label_budget_windows": 80,
                "target_scaler": {"mean": [125., 75.], "std": [20., 10.]},
                "source_checkpoint_sha256": "0" * 64, "files": {}}
    for role in ("train", "validation"):
        pd.DataFrame(rows[role]).to_parquet(root / f"{role}_metadata.parquet", index=False)
        for suffix in ("features", "bp", "base"):
            np.save(root / f"{role}_{suffix}.npy", data[f"{role}_{suffix}"], allow_pickle=False)
        for suffix in ("knn_indices", "uniform_indices", "knn_weights", "support_weight"):
            np.save(root / f"{role}_{suffix}.npy", neighbours[role][suffix], allow_pickle=False)
    for path in root.iterdir():
        manifest["files"][path.name] = {"path": path.name, "sha256": file_sha256(path)}
    save_json(root / "manifest.json", manifest)
    return root


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--examples-per-epoch", type=int, default=200000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--subjects-per-batch", type=int, default=32)
    p.add_argument("--smoke", action="store_true", help="Only caches explicitly marked synthetic_smoke")
    p.add_argument("--smoke-check", action="store_true", help="Full audited real cache, exactly four train steps; never a completed scientific fit")
    p.add_argument("--synthetic-smoke", action="store_true", help="Create a small synthetic-only cache adjacent to output and run four steps")
    p.add_argument("--split-mode", choices=MODES, default="random_disjoint", help="Only used to create the synthetic smoke cache")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.synthetic_smoke:
        if args.cache_dir is not None or args.smoke_check:
            p.error("synthetic-smoke creates its own cache; cannot mix real-cache arguments")
        args.cache_dir = make_synthetic_cache(args.output.parent / (args.output.name + "_synthetic_cache"), args.split_mode)
        args.smoke, args.epochs, args.examples_per_epoch, args.batch_size = True, 1, 1024, 256
    elif args.cache_dir is None:
        p.error("--cache-dir is required unless --synthetic-smoke")
    print(json.dumps(train(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
