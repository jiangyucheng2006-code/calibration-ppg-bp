"""Train the population, Siamese, or variable-K personalized model."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Sampler, WeightedRandomSampler

from .models import PopulationRegressor, SiameseDeltaRegressor, VariableKPersonalizer
from .training import (
    EpisodicDataset,
    PopulationDataset,
    file_sha256,
    fit_target_scaler,
    load_store_metadata,
    participant_macro_metrics,
    predict_episodic,
    predict_population,
    save_json,
    seed_everything,
    source_tree_sha256,
)


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )


def _participant_balanced_sampler(
    metadata: pd.DataFrame, seed: int, num_samples: int
) -> WeightedRandomSampler:
    counts = metadata["subject_uid"].value_counts()
    weights = metadata["subject_uid"].map(lambda value: 1.0 / counts[value]).to_numpy()
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=min(num_samples, len(metadata)),
        replacement=True,
        generator=generator,
    )


def _episodic_balanced_sampler(
    dataset: EpisodicDataset,
    seed: int,
    num_samples: int,
    *,
    mode: str = "participant_balanced",
    bp_change_alpha: float = 2.0,
    participant_multipliers: dict[str, float] | None = None,
) -> WeightedRandomSampler:
    identities = pd.Series(dataset.participant_ids)
    counts = identities.value_counts()
    weights = identities.map(lambda value: 1.0 / counts[value]).to_numpy(dtype=float)
    if mode == "bp_change_aware":
        if bp_change_alpha <= 0:
            raise ValueError("bp_change_alpha must be positive")
        changes = pd.Series(dataset.bp_change_scores(), dtype=float).to_numpy()
        scale = float(pd.Series(changes).quantile(0.90))
        if not scale > 0:
            raise ValueError("BP-change sampler requires a positive training change scale")
        weights *= 1.0 + bp_change_alpha * (changes / scale).clip(0.0, 1.0)
    elif mode != "participant_balanced":
        raise ValueError(f"unsupported episodic sampling mode: {mode}")
    if participant_multipliers is not None:
        missing = set(identities.astype(str)) - set(participant_multipliers)
        if missing:
            raise ValueError(
                "participant multiplier table does not cover the episodic dataset"
            )
        multipliers = identities.astype(str).map(participant_multipliers).to_numpy(
            dtype=float
        )
        if not np.isfinite(multipliers).all() or not (multipliers > 0).all():
            raise ValueError("participant sampling multipliers must be finite and positive")
        weights *= multipliers
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=min(num_samples, len(dataset)),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


class ParticipantEpisodeBatchSampler(Sampler[list[int]]):
    """Create batches with repeated episodes from distinct participants.

    Ordinary episode sampling often places only one episode per participant in
    a batch, which would make a so-called participant CVaR effectively an event
    CVaR.  This sampler draws a fixed number of episodes for each of several
    distinct, uniformly sampled participants, so participant aggregation is
    real rather than nominal.
    """

    def __init__(
        self,
        dataset: EpisodicDataset,
        *,
        seed: int,
        num_samples: int,
        batch_size: int,
        episodes_per_participant: int,
    ) -> None:
        if episodes_per_participant < 1:
            raise ValueError("episodes_per_participant must be positive")
        if batch_size < episodes_per_participant:
            raise ValueError("batch_size must cover at least one participant")
        if batch_size % episodes_per_participant:
            raise ValueError(
                "batch_size must be divisible by episodes_per_participant"
            )
        groups: dict[str, list[int]] = {}
        for index, subject_uid in enumerate(dataset.participant_ids):
            groups.setdefault(str(subject_uid), []).append(index)
        if not groups:
            raise ValueError("participant batch sampler received an empty dataset")
        self.subjects = sorted(groups)
        self.indexes = groups
        self.participants_per_batch = batch_size // episodes_per_participant
        if self.participants_per_batch > len(self.subjects):
            raise ValueError("batch requests more distinct participants than available")
        available_samples = min(int(num_samples), len(dataset))
        self.n_batches = max(1, available_samples // batch_size)
        self.episodes_per_participant = episodes_per_participant
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            participant_positions = torch.randperm(
                len(self.subjects), generator=self.generator
            )[: self.participants_per_batch]
            batch: list[int] = []
            for position in participant_positions.tolist():
                subject_uid = self.subjects[position]
                candidates = self.indexes[subject_uid]
                draws = torch.randint(
                    len(candidates),
                    (self.episodes_per_participant,),
                    generator=self.generator,
                )
                batch.extend(candidates[index] for index in draws.tolist())
            order = torch.randperm(len(batch), generator=self.generator).tolist()
            yield [batch[index] for index in order]

    def __len__(self) -> int:
        return self.n_batches


def _load_population_checkpoint(
    path: Path, device: torch.device
) -> tuple[PopulationRegressor, dict[str, list[float]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PopulationRegressor()
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint["target_scaler"]


def _epoch_numbers(max_epochs: int) -> Iterable[int]:
    """Yield one-based epoch numbers; zero means early-stopping-only training."""

    if max_epochs < 0:
        raise ValueError("epochs must be nonnegative; use 0 for no epoch-count cap")
    if max_epochs == 0:
        return itertools.count(1)
    return range(1, max_epochs + 1)


def _coordinate_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    huber_delta: float,
) -> torch.Tensor:
    """Return an unreduced loss for each BP coordinate.

    Keeping this loss unreduced is required for the participant-level tail
    objective below.  The target values are standardized SBP/DBP values, so
    ``huber_delta`` is expressed in standardized BP units.
    """

    if loss_name == "mse":
        return (prediction - target).square()
    if loss_name == "huber":
        return nn.functional.huber_loss(
            prediction,
            target,
            reduction="none",
            delta=huber_delta,
        )
    raise ValueError(f"unsupported loss: {loss_name}")


def _participant_tail_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    subject_uids: list[str] | tuple[str, ...],
    *,
    loss_name: str,
    huber_delta: float,
    tail_fraction: float,
    tail_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine ordinary risk with empirical participant-level CVaR.

    The batch is sampled with participant balancing.  Coordinate losses are
    first averaged within each episode, then within each participant appearing
    in the batch.  The empirical CVaR term is the mean loss of the highest-loss
    ``ceil(tail_fraction * n_participants)`` participants.  This uses only
    meta-train labels during supervised training and never creates an
    inference-time tail flag.
    """

    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if not 0.0 <= tail_weight <= 1.0:
        raise ValueError("tail_weight must be in [0, 1]")
    if len(subject_uids) != len(prediction):
        raise ValueError("subject_uids length must match the batch size")
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have equal [batch, BP] shape")

    episode_loss = _coordinate_loss(
        prediction.float(),
        target.float(),
        loss_name=loss_name,
        huber_delta=huber_delta,
    ).mean(dim=1)
    participant_losses: list[torch.Tensor] = []
    # dict insertion order is deterministic and follows the collated batch.
    positions: dict[str, list[int]] = {}
    for index, subject_uid in enumerate(subject_uids):
        positions.setdefault(str(subject_uid), []).append(index)
    for indexes in positions.values():
        participant_losses.append(episode_loss[indexes].mean())
    participant_loss = torch.stack(participant_losses)
    mean_risk = participant_loss.mean()
    tail_n = max(
        1,
        int(torch.ceil(torch.tensor(tail_fraction * len(participant_losses))).item()),
    )
    tail_risk = participant_loss.topk(tail_n, largest=True, sorted=False).values.mean()
    objective = (1.0 - tail_weight) * mean_risk + tail_weight * tail_risk
    diagnostics = {
        "mean_participant_batch_risk": float(mean_risk.detach()),
        "tail_participant_batch_risk": float(tail_risk.detach()),
        "batch_participants": float(len(participant_losses)),
        "tail_participants": float(tail_n),
    }
    return objective, diagnostics


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.patience < 1:
        raise ValueError("patience must be positive")
    if args.loss == "huber" and args.huber_delta <= 0:
        raise ValueError("--huber-delta must be positive")
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must be in (0, 1]")
    if not 0.0 <= args.tail_weight <= 1.0:
        raise ValueError("--tail-weight must be in [0, 1]")
    if args.tail_objective != "mean" and args.method != "m0":
        raise ValueError("participant-tail training is currently defined for method m0 only")
    if (args.crossfit_folds is None) != (args.crossfit_heldout_fold is None):
        raise ValueError(
            "--crossfit-folds and --crossfit-heldout-fold must be provided together"
        )
    if args.crossfit_heldout_fold is not None and args.crossfit_heldout_fold < 0:
        raise ValueError("--crossfit-heldout-fold must be nonnegative")
    if args.participant_risk_labels is not None and args.method != "m0":
        raise ValueError("cross-fitted participant weighting is defined for method m0 only")
    if args.participant_risk_labels is not None and args.tail_objective != "mean":
        raise ValueError(
            "cross-fitted participant weighting cannot be combined with another "
            "tail objective in the same single-factor run"
        )
    if args.hard_participant_only and args.participant_risk_labels is None:
        raise ValueError("--hard-participant-only requires --participant-risk-labels")
    if args.hard_participant_weight < 1.0:
        raise ValueError("--hard-participant-weight must be at least one")
    if args.use_demographics and args.demographics_path is None:
        raise ValueError("--use-demographics requires --demographics-path")
    if (
        args.anchor_mode != "mean"
        or args.use_quality_gate
        or args.use_demographics
        or args.episode_sampling != "participant_balanced"
    ) and args.method != "m0":
        raise ValueError("Phase-6 isolated candidate options are defined for method m0 only")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required but is unavailable")
    metadata = load_store_metadata(args.store_root, "development")
    original_train = metadata.loc[metadata["split"].eq("meta_train")].copy()
    if args.crossfit_folds is None:
        train_metadata = original_train
        validation_metadata = metadata.loc[metadata["split"].eq("meta_validation")].copy()
        run_split = "meta_validation"
        scaler_metadata = metadata
    else:
        folds = pd.read_parquet(args.crossfit_folds)
        required_folds = {"subject_uid", "source", "fold"}
        missing_folds = required_folds - set(folds.columns)
        if missing_folds:
            raise ValueError(f"cross-fit fold table missing {sorted(missing_folds)}")
        if folds["subject_uid"].duplicated().any():
            raise AssertionError("cross-fit fold table contains duplicate participants")
        expected_subjects = set(original_train["subject_uid"].astype(str))
        observed_subjects = set(folds["subject_uid"].astype(str))
        if observed_subjects != expected_subjects:
            raise AssertionError("cross-fit folds do not exactly cover meta-train participants")
        subject_source = (
            original_train[["subject_uid", "source"]]
            .drop_duplicates()
            .assign(subject_uid=lambda frame: frame["subject_uid"].astype(str))
        )
        checked = folds.assign(subject_uid=folds["subject_uid"].astype(str)).merge(
            subject_source,
            on="subject_uid",
            how="left",
            suffixes=("_fold", "_metadata"),
            validate="one_to_one",
        )
        if not checked["source_fold"].astype(str).equals(
            checked["source_metadata"].astype(str)
        ):
            raise AssertionError("cross-fit fold sources differ from metadata")
        heldout = int(args.crossfit_heldout_fold)
        if heldout not in set(pd.to_numeric(folds["fold"]).astype(int)):
            raise ValueError(f"held-out cross-fit fold {heldout} is absent")
        fold_lookup = folds.assign(
            subject_uid=folds["subject_uid"].astype(str),
            fold=pd.to_numeric(folds["fold"], errors="raise").astype(int),
        ).set_index("subject_uid")["fold"]
        event_folds = original_train["subject_uid"].astype(str).map(fold_lookup)
        if event_folds.isna().any():
            raise AssertionError("at least one meta-train event has no cross-fit fold")
        train_metadata = original_train.loc[event_folds.ne(heldout)].copy()
        validation_metadata = original_train.loc[event_folds.eq(heldout)].copy()
        if train_metadata.empty or validation_metadata.empty:
            raise ValueError("cross-fit train or held-out fold is empty")
        run_split = f"meta_train_crossfit_fold_{heldout}"
        scaler_metadata = train_metadata
    if set(train_metadata["subject_uid"]) & set(validation_metadata["subject_uid"]):
        raise AssertionError("participant leakage between training and validation")
    if args.max_train_examples is not None:
        train_metadata = train_metadata.head(args.max_train_examples).copy()
    if args.max_validation_examples is not None:
        validation_metadata = validation_metadata.head(args.max_validation_examples).copy()
    # fit_target_scaler explicitly selects meta_train rows. Passing the full
    # development metadata preserves a single audited entry point while
    # keeping meta-validation targets out of preprocessing parameters.
    scaler = fit_target_scaler(scaler_metadata)
    validation_metadata = validation_metadata.loc[
        validation_metadata["common_query"]
        | validation_metadata["support_candidate"]
    ].copy()
    demographics = None
    if args.use_demographics:
        demographics = pd.read_parquet(args.demographics_path)
        required_demo = {
            "subject_uid",
            "split",
            "age_z",
            "age_valid",
            "sex_female",
            "sex_male",
            "sex_unknown",
        }
        missing_demo = required_demo - set(demographics.columns)
        if missing_demo:
            raise ValueError(f"demographic table missing columns: {sorted(missing_demo)}")
        if demographics["split"].eq("meta_test").any():
            raise AssertionError("development demographic table contains meta-test rows")
        expected_subjects = set(train_metadata["subject_uid"]) | set(
            validation_metadata["subject_uid"]
        )
        if not expected_subjects.issubset(set(demographics["subject_uid"])):
            raise AssertionError("demographic table does not cover the development subjects")

    participant_multipliers: dict[str, float] | None = None
    if args.participant_risk_labels is not None:
        risk_labels = pd.read_parquet(args.participant_risk_labels)
        required_labels = {"subject_uid", "source", "hard_oof", "label_split"}
        missing_labels = required_labels - set(risk_labels.columns)
        if missing_labels:
            raise ValueError(
                f"participant risk label table missing {sorted(missing_labels)}"
            )
        if risk_labels["subject_uid"].duplicated().any():
            raise AssertionError("participant risk labels contain duplicate subjects")
        if set(risk_labels["label_split"].astype(str)) != {
            "meta_train_crossfit_oof"
        }:
            raise AssertionError("participant risk labels are not cross-fitted meta-train labels")
        labelled_subjects = set(risk_labels["subject_uid"].astype(str))
        expected_labelled_subjects = set(original_train["subject_uid"].astype(str))
        if labelled_subjects != expected_labelled_subjects:
            raise AssertionError(
                "participant risk labels do not exactly cover meta-train participants"
            )
        label_source = risk_labels[["subject_uid", "source"]].assign(
            subject_uid=lambda frame: frame["subject_uid"].astype(str)
        )
        metadata_source = original_train[["subject_uid", "source"]].drop_duplicates().assign(
            subject_uid=lambda frame: frame["subject_uid"].astype(str)
        )
        checked_sources = label_source.merge(
            metadata_source,
            on="subject_uid",
            how="left",
            suffixes=("_label", "_metadata"),
            validate="one_to_one",
        )
        if not checked_sources["source_label"].astype(str).equals(
            checked_sources["source_metadata"].astype(str)
        ):
            raise AssertionError("participant risk label sources differ from metadata")
        labels = risk_labels.assign(
            subject_uid=risk_labels["subject_uid"].astype(str),
            hard_oof=risk_labels["hard_oof"].astype(bool),
        ).set_index("subject_uid")["hard_oof"]
        training_subjects = set(train_metadata["subject_uid"].astype(str))
        if not training_subjects.issubset(set(labels.index)):
            raise AssertionError("risk labels do not cover all training participants")
        if args.hard_participant_only:
            hard_subjects = set(labels.index[labels.to_numpy(dtype=bool)])
            train_metadata = train_metadata.loc[
                train_metadata["subject_uid"].astype(str).isin(hard_subjects)
            ].copy()
            if train_metadata.empty:
                raise AssertionError("hard-participant-only training set is empty")
        else:
            participant_multipliers = {
                subject: args.hard_participant_weight if bool(labels.loc[subject]) else 1.0
                for subject in training_subjects
            }

    siamese = args.method == "siamese"
    if args.method == "population":
        model: nn.Module = PopulationRegressor()
        train_dataset = PopulationDataset(train_metadata, args.store_root, scaler)
        validation_dataset = PopulationDataset(
            validation_metadata.loc[validation_metadata["common_query"]].copy(),
            args.store_root,
            scaler,
        )
        sampler = _participant_balanced_sampler(
            train_metadata, args.seed, args.episodes_per_epoch
        )
    else:
        if args.population_checkpoint is None:
            raise ValueError("personalized methods require --population-checkpoint")
        population, checkpoint_scaler = _load_population_checkpoint(
            args.population_checkpoint, device
        )
        if checkpoint_scaler != scaler:
            raise AssertionError("target scaler differs from population checkpoint")
        if siamese:
            model = SiameseDeltaRegressor(population.encoder)
            episodic_ks = (1,)
        else:
            configuration = {
                "m0": (False, False),
                "m1": (True, False),
                "m2": (True, True),
            }
            use_film, attention = configuration[args.method]
            model = VariableKPersonalizer(
                population,
                use_film=use_film,
                query_conditioned_weights=attention,
                anchor_mode=args.anchor_mode,
                use_quality_gate=args.use_quality_gate,
                use_demographics=args.use_demographics,
            )
            for parameter in model.population.parameters():
                parameter.requires_grad = False
            episodic_ks = tuple(args.ks)
        train_dataset = EpisodicDataset(
            train_metadata,
            args.store_root,
            scaler,
            ks=episodic_ks,
            rolling_support=(
                not siamese and args.train_support_policy == "rolling_recent"
            ),
            demographics=demographics if args.use_demographics else None,
        )
        validation_dataset = EpisodicDataset(
            validation_metadata,
            args.store_root,
            scaler,
            ks=episodic_ks,
            rolling_support=False,
            demographics=demographics if args.use_demographics else None,
        )
        sampler = (
            None
            if args.tail_objective == "mean_cvar"
            else _episodic_balanced_sampler(
                train_dataset,
                args.seed,
                args.episodes_per_epoch,
                mode=args.episode_sampling,
                bp_change_alpha=args.bp_change_alpha,
                participant_multipliers=participant_multipliers,
            )
        )

    model = model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.loss == "mse":
        loss_function: nn.Module = nn.MSELoss()
    elif args.loss == "huber":
        loss_function = nn.HuberLoss(delta=args.huber_delta)
    else:  # pragma: no cover - argparse guards this branch.
        raise ValueError(f"unsupported loss: {args.loss}")
    loader_common = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    if args.tail_objective == "mean_cvar":
        if not isinstance(train_dataset, EpisodicDataset):
            raise AssertionError("tail training requires an episodic dataset")
        participant_batch_sampler = ParticipantEpisodeBatchSampler(
            train_dataset,
            seed=args.seed,
            num_samples=args.episodes_per_epoch,
            batch_size=args.batch_size,
            episodes_per_participant=args.tail_episodes_per_participant,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=participant_batch_sampler,
            **loader_common,
        )
    else:
        if sampler is None:
            raise AssertionError("ordinary training requires a sampler")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            drop_last=True,
            **loader_common,
        )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, object]] = []
    best_score = float("inf")
    best_epoch: int | None = None
    best_metrics: dict[str, object] | None = None
    epochs_without_improvement = 0
    stop_reason = "epoch_cap"
    checkpoint_path = args.output / "best.pt"
    for epoch in _epoch_numbers(args.epochs):
        model.train()
        if isinstance(model, VariableKPersonalizer):
            # The population mapping is a frozen reference for residual
            # anchoring. Keep BatchNorm statistics and dropout frozen too.
            model.population.eval()
        total_loss = 0.0
        total_examples = 0
        total_mean_batch_risk = 0.0
        total_tail_batch_risk = 0.0
        tail_batches = 0
        for batch in train_loader:
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                if args.method == "population":
                    prediction = model(batch["ppg"].to(device, non_blocking=True))
                elif siamese:
                    prediction = model(
                        batch["query_ppg"].to(device, non_blocking=True),
                        batch["support_ppg"][:, 0].to(device, non_blocking=True),
                        batch["support_bp"][:, 0].to(device, non_blocking=True),
                    )
                else:
                    demographics_batch = batch.get("demographics")
                    prediction = model(
                        batch["query_ppg"].to(device, non_blocking=True),
                        batch["support_ppg"].to(device, non_blocking=True),
                        batch["support_bp"].to(device, non_blocking=True),
                        batch["support_mask"].to(device, non_blocking=True),
                        demographics_batch.to(device, non_blocking=True)
                        if demographics_batch is not None
                        else None,
                    )
                if args.tail_objective == "mean_cvar":
                    loss, tail_diagnostics = _participant_tail_objective(
                        prediction,
                        target,
                        batch["subject_uid"],
                        loss_name=args.loss,
                        huber_delta=args.huber_delta,
                        tail_fraction=args.tail_fraction,
                        tail_weight=args.tail_weight,
                    )
                    total_mean_batch_risk += tail_diagnostics[
                        "mean_participant_batch_risk"
                    ]
                    total_tail_batch_risk += tail_diagnostics[
                        "tail_participant_batch_risk"
                    ]
                    tail_batches += 1
                else:
                    loss = loss_function(prediction.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total_examples += len(target)

        if args.method == "population":
            predictions = predict_population(
                model, validation_loader, device, scaler
            )
            metrics = participant_macro_metrics(predictions)
        else:
            predictions = predict_episodic(
                model, validation_loader, device, scaler, siamese=siamese
            )
            by_k = {
                str(k): participant_macro_metrics(group)
                for k, group in predictions.groupby("k")
            }
            metrics = {
                "mean_mae": float(
                    sum(item["mean_mae"] for item in by_k.values()) / len(by_k)
                ),
                "worst_30_mean_mae": float(
                    sum(item["worst_30_mean_mae"] for item in by_k.values())
                    / len(by_k)
                ),
                "retained_70_mean_mae": float(
                    sum(item["retained_70_mean_mae"] for item in by_k.values())
                    / len(by_k)
                ),
                "by_k": by_k,
            }
        record = {
            "epoch": epoch,
            # ``train_mse`` is retained for backward-compatible readers.  For
            # Huber or CVaR runs it stores the actual optimized objective, not
            # literal MSE; ``train_objective`` is the unambiguous field.
            "train_mse": total_loss / max(total_examples, 1),
            "train_objective": total_loss / max(total_examples, 1),
            "train_objective_name": args.tail_objective,
            "validation": metrics,
        }
        if tail_batches:
            record["train_mean_participant_batch_risk"] = (
                total_mean_batch_risk / tail_batches
            )
            record["train_tail_participant_batch_risk"] = (
                total_tail_batch_risk / tail_batches
            )
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        score = float(metrics["mean_mae"])
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            epochs_without_improvement = 0
            torch.save(
                {
                    "method": args.method,
                    "model_state": model.state_dict(),
                    "target_scaler": scaler,
                    "epoch": epoch,
                    "validation": metrics,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
            predictions.to_parquet(args.output / "best_validation_predictions.parquet", index=False)
        else:
            epochs_without_improvement += 1
        save_json(args.output / "history.json", history)
        if epochs_without_improvement >= args.patience:
            stop_reason = "early_stopping"
            break

    if best_epoch is None or best_metrics is None:
        raise RuntimeError("training completed without a valid checkpoint")

    result = {
        "status": "complete",
        "method": args.method,
        "split": run_split,
        "locked_test_accessed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "train_participants": int(train_metadata["subject_uid"].nunique()),
        "validation_participants": int(validation_metadata["subject_uid"].nunique()),
        "train_events": int(len(train_metadata)),
        "validation_events": int(len(validation_metadata)),
        "crossfit_heldout_fold": args.crossfit_heldout_fold,
        "participant_risk_weighting": args.participant_risk_labels is not None,
        "hard_participant_only": args.hard_participant_only,
        "crossfit_folds_sha256": (
            file_sha256(args.crossfit_folds) if args.crossfit_folds else None
        ),
        "participant_risk_labels_sha256": (
            file_sha256(args.participant_risk_labels)
            if args.participant_risk_labels
            else None
        ),
        "target_scaler": scaler,
        "best_validation_mean_mae": float(best_metrics["mean_mae"]),
        "best_validation_worst_30_mean_mae": (
            float(best_metrics["worst_30_mean_mae"])
            if "worst_30_mean_mae" in best_metrics
            else None
        ),
        "best_validation_retained_70_mean_mae": (
            float(best_metrics["retained_70_mean_mae"])
            if "retained_70_mean_mae" in best_metrics
            else None
        ),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "store_manifest_sha256": file_sha256(args.store_root / "materialization.json"),
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "arguments": vars(args) | {
            "store_root": str(args.store_root),
            "output": str(args.output),
            "population_checkpoint": str(args.population_checkpoint)
            if args.population_checkpoint
            else None,
            "demographics_path": str(args.demographics_path)
            if args.demographics_path
            else None,
            "crossfit_folds": str(args.crossfit_folds)
            if args.crossfit_folds
            else None,
            "participant_risk_labels": str(args.participant_risk_labels)
            if args.participant_risk_labels
            else None,
        },
    }
    save_json(args.output / "run.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["population", "siamese", "m0", "m1", "m2"], required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="maximum epochs; use 0 for no epoch-count cap and stop by patience",
    )
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--train-support-policy",
        choices=["rolling_recent", "fixed_first"],
        default="rolling_recent",
        help=(
            "rolling_recent uses the latest K prior events during meta-training; "
            "fixed_first always uses events 1..K and matches evaluation support."
        ),
    )
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=0.5,
        help="Huber transition in standardized BP units; used only with --loss huber.",
    )
    parser.add_argument(
        "--tail-objective",
        choices=["mean", "mean_cvar"],
        default="mean",
        help=(
            "mean uses the ordinary batch loss; mean_cvar combines ordinary "
            "participant-balanced risk with the highest-loss participant tail."
        ),
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.30,
        help="Fraction of highest-loss batch participants used by mean_cvar.",
    )
    parser.add_argument(
        "--tail-weight",
        type=float,
        default=0.50,
        help="Convex weight assigned to empirical participant CVaR.",
    )
    parser.add_argument(
        "--tail-episodes-per-participant",
        type=int,
        default=4,
        help=(
            "Episodes sampled per distinct participant in each mean_cvar batch; "
            "the batch size must be divisible by this value."
        ),
    )
    parser.add_argument("--episodes-per-epoch", type=int, default=200000)
    parser.add_argument(
        "--episode-sampling",
        choices=["participant_balanced", "bp_change_aware"],
        default="participant_balanced",
    )
    parser.add_argument("--bp-change-alpha", type=float, default=2.0)
    parser.add_argument("--anchor-mode", choices=["mean", "median"], default="mean")
    parser.add_argument("--use-quality-gate", action="store_true")
    parser.add_argument("--use-demographics", action="store_true")
    parser.add_argument("--demographics-path", type=Path)
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 5],
        default=[1, 2, 3, 5],
        help="Episodic calibration budgets to train and validate.",
    )
    parser.add_argument("--crossfit-folds", type=Path)
    parser.add_argument("--crossfit-heldout-fold", type=int)
    parser.add_argument("--participant-risk-labels", type=Path)
    parser.add_argument("--hard-participant-only", action="store_true")
    parser.add_argument(
        "--hard-participant-weight",
        type=float,
        default=4.0,
        help="Sampling multiplier for cross-fitted hard participants.",
    )
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
