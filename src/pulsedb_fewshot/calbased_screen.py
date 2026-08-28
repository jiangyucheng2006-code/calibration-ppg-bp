"""Leakage-safe candidate matrix for the development CalBased analogue.

This module plans a *seen-subject* comparison track. It is deliberately
separate from the participant-disjoint event120-v1 primary protocol and from
the official PulseDB CalBased benchmark. The first screening stage reads only
the ``train`` and ``internal_validation`` roles. ``heldout_test`` is reserved
for a later winner-only run and is never a model-selection role.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL_ID = "development-calbased-analogue-v1"
SOURCE_PARENT_SPLIT = "meta_train"
FORBIDDEN_PARENT_SPLITS = ("meta_validation", "meta_test")
SUBJECT_COUNT = 2058
ROLE_WINDOWS_PER_SUBJECT = {
    "train": 320,
    "internal_validation": 40,
    "heldout_test": 40,
}
SELECTION_ROLE = "internal_validation"
SCREENING_READ_ROLES = ("train", "internal_validation")
HELDOUT_ROLE = "heldout_test"
SPLIT_MODES = ("random_disjoint", "chronological_blocked")
DEFAULT_SEED = 20260828
EARLY_STOPPING_PATIENCE = 8


@dataclass(frozen=True)
class CandidateSpec:
    """One controlled first-round model candidate."""

    name: str
    runner: str
    backbone: str | None
    feature_dim: int | None
    loss: str | None
    support_k: int | None
    use_quality_gate: bool
    requires_population_checkpoint: bool
    literature_adaptation: bool
    seen_subject_only: bool
    executable_first_round: bool
    deferred_reason: str | None
    description: str


CANDIDATE_SPECS = (
    CandidateSpec(
        name="subject_train_mean",
        runner="analysis_baseline",
        backbone=None,
        feature_dim=None,
        loss=None,
        support_k=None,
        use_quality_gate=False,
        requires_population_checkpoint=False,
        literature_adaptation=False,
        seen_subject_only=True,
        executable_first_round=True,
        deferred_reason=None,
        description=(
            "Predict each validation window with that subject's mean SBP/DBP "
            "computed from train labels only."
        ),
    ),
    CandidateSpec(
        name="subject_mean_residual_ppg",
        runner="subject_mean_residual",
        backbone="resnet_small",
        feature_dim=256,
        loss="huber",
        support_k=None,
        use_quality_gate=False,
        requires_population_checkpoint=False,
        literature_adaptation=False,
        seen_subject_only=True,
        executable_first_round=True,
        deferred_reason=None,
        description=(
            "Predict subject train-label mean plus a compact PPG network's "
            "window-level residual; validation labels never enter the mean."
        ),
    ),
    CandidateSpec(
        name="compact_resnet",
        runner="population_regression",
        backbone="resnet_small",
        feature_dim=256,
        loss="huber",
        support_k=None,
        use_quality_gate=False,
        requires_population_checkpoint=False,
        literature_adaptation=False,
        seen_subject_only=True,
        executable_first_round=True,
        deferred_reason=None,
        description="Compact repository ResNet trained on the same-subject train role.",
    ),
    CandidateSpec(
        name="inception_time_wide",
        runner="population_regression",
        backbone="inception_time_wide",
        feature_dim=256,
        loss="huber",
        support_k=None,
        use_quality_gate=False,
        requires_population_checkpoint=False,
        literature_adaptation=False,
        seen_subject_only=True,
        executable_first_round=True,
        deferred_reason=None,
        description=(
            "Existing wide InceptionTime repository backbone retrained under the "
            "same-subject window protocol."
        ),
    ),
    CandidateSpec(
        name="patch_transformer",
        runner="population_regression",
        backbone="patch_transformer",
        feature_dim=256,
        loss="huber",
        support_k=None,
        use_quality_gate=False,
        requires_population_checkpoint=False,
        literature_adaptation=False,
        seen_subject_only=True,
        executable_first_round=True,
        deferred_reason=None,
        description=(
            "Existing repository patch Transformer retrained under the "
            "same-subject window protocol."
        ),
    ),
    CandidateSpec(
        name="compact_resnet_qgh",
        runner="qgh",
        backbone="resnet_small",
        feature_dim=256,
        loss="huber",
        support_k=5,
        use_quality_gate=True,
        requires_population_checkpoint=True,
        literature_adaptation=False,
        seen_subject_only=True,
        executable_first_round=False,
        deferred_reason=(
            "Requires a separate same-subject support-role adapter and a train-only "
            "population checkpoint; do not submit the participant-disjoint runner."
        ),
        description=(
            "Existing fixed-first K=5 residual anchor with quality gate and "
            "Huber loss, fitted without validation targets."
        ),
    ),
    CandidateSpec(
        name="compact_resnet_calibration_relative",
        runner="calibration_relative",
        backbone="resnet_small",
        feature_dim=256,
        loss="huber",
        support_k=5,
        use_quality_gate=True,
        requires_population_checkpoint=True,
        literature_adaptation=False,
        seen_subject_only=True,
        executable_first_round=False,
        deferred_reason=(
            "Existing causal correction consumes participant-disjoint cross-fitted "
            "artifacts and must not be pointed at this store without adaptation."
        ),
        description=(
            "Existing calibration-relative correction family using train-role "
            "supports and causal/query PPG features only."
        ),
    ),
    CandidateSpec(
        name="self_attention_resunet_adaptation",
        runner="population_regression",
        backbone="self_attention_resunet_adaptation",
        feature_dim=256,
        loss="huber",
        support_k=None,
        use_quality_gate=False,
        requires_population_checkpoint=False,
        literature_adaptation=True,
        seen_subject_only=True,
        executable_first_round=True,
        deferred_reason=None,
        description=(
            "PPG-only residual U-Net with bottleneck self-attention; architecture "
            "adaptation, not an exact multimodal paper reproduction."
        ),
    ),
    CandidateSpec(
        name="runet_resunet_encoder_adaptation",
        runner="population_regression",
        backbone="runet_resunet_encoder_adaptation",
        feature_dim=256,
        loss="huber",
        support_k=None,
        use_quality_gate=False,
        requires_population_checkpoint=False,
        literature_adaptation=True,
        seen_subject_only=True,
        executable_first_round=True,
        deferred_reason=None,
        description=(
            "PPG-only 1-D rU-Net/ResUNet encoder adaptation; no ECG, demographics, "
            "or ABP-waveform auxiliary target, and not an exact paper reproduction."
        ),
    ),
    CandidateSpec(
        name="cnn_bilstm_adaptation",
        runner="population_regression",
        backbone="cnn_bilstm_adaptation",
        feature_dim=256,
        loss="huber",
        support_k=None,
        use_quality_gate=False,
        requires_population_checkpoint=False,
        literature_adaptation=True,
        seen_subject_only=True,
        executable_first_round=True,
        deferred_reason=None,
        description=(
            "PPG-only CNN-BiLSTM architecture adaptation; no ECG stream and not "
            "an exact reproduction of the dual-stream publication."
        ),
    ),
    CandidateSpec(
        name="cnn_transformer_aff_adaptation",
        runner="population_regression",
        backbone="cnn_transformer_aff_adaptation",
        feature_dim=256,
        loss="huber",
        support_k=None,
        use_quality_gate=False,
        requires_population_checkpoint=False,
        literature_adaptation=True,
        seen_subject_only=True,
        executable_first_round=True,
        deferred_reason=None,
        description=(
            "PPG-only multi-kernel CNN, adaptive branch fusion, and Transformer "
            "architecture adaptation; not an exact paper reproduction."
        ),
    ),
)
CANDIDATES = {candidate.name: candidate for candidate in CANDIDATE_SPECS}


def fit_subject_train_means(train_metadata: pd.DataFrame) -> pd.DataFrame:
    """Fit per-subject SBP/DBP means from the train role only."""

    required = {"subject_uid", "sbp", "dbp"}
    missing = required - set(train_metadata.columns)
    if missing:
        raise ValueError(f"train metadata missing columns: {sorted(missing)}")
    if "role" in train_metadata and not train_metadata["role"].eq("train").all():
        raise ValueError("subject means may be fitted from the train role only")
    targets = train_metadata[["sbp", "dbp"]].to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("train labels must be finite")
    means = (
        train_metadata.assign(subject_uid=train_metadata["subject_uid"].astype(str))
        .groupby("subject_uid", sort=True, as_index=False)[["sbp", "dbp"]]
        .mean()
        .rename(columns={"sbp": "subject_train_sbp", "dbp": "subject_train_dbp"})
    )
    if means.empty:
        raise ValueError("cannot fit subject means from an empty train role")
    return means


def predict_subject_train_mean(
    query_metadata: pd.DataFrame, subject_means: pd.DataFrame
) -> pd.DataFrame:
    """Apply train-only subject means without reading query BP labels."""

    if "subject_uid" not in query_metadata:
        raise ValueError("query metadata is missing subject_uid")
    required = {"subject_uid", "subject_train_sbp", "subject_train_dbp"}
    missing = required - set(subject_means.columns)
    if missing:
        raise ValueError(f"subject mean table missing columns: {sorted(missing)}")
    queries = query_metadata[["subject_uid"]].copy()
    queries["subject_uid"] = queries["subject_uid"].astype(str)
    means = subject_means[list(required)].copy()
    means["subject_uid"] = means["subject_uid"].astype(str)
    if means["subject_uid"].duplicated().any():
        raise ValueError("subject mean table contains duplicate participants")
    predictions = queries.merge(means, on="subject_uid", how="left", validate="many_to_one")
    if predictions[["subject_train_sbp", "subject_train_dbp"]].isna().any().any():
        raise ValueError("every query participant must occur in the train role")
    return predictions.rename(
        columns={
            "subject_train_sbp": "sbp_pred",
            "subject_train_dbp": "dbp_pred",
        }
    )


def _validate_store_manifest(store_root: Path) -> dict[str, Any]:
    """Validate provenance without opening any role table or held-out labels."""

    manifest_path = store_root / "materialization.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing CalBased store manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(
            f"expected protocol_id={PROTOCOL_ID!r}, got {manifest.get('protocol_id')!r}"
        )
    if manifest.get("source_parent_split") != SOURCE_PARENT_SPLIT:
        raise ValueError("CalBased analogue must be derived only from meta_train")
    declared_sources = set(manifest.get("source_parent_splits", [SOURCE_PARENT_SPLIT]))
    if declared_sources != {SOURCE_PARENT_SPLIT}:
        raise ValueError("store declares a forbidden parent split")
    if int(manifest.get("subject_count", -1)) != SUBJECT_COUNT:
        raise ValueError(f"store must contain exactly {SUBJECT_COUNT} participants")
    observed_counts = manifest.get("windows_per_subject")
    if observed_counts != ROLE_WINDOWS_PER_SUBJECT:
        raise ValueError(
            "store windows_per_subject must be exactly "
            f"{ROLE_WINDOWS_PER_SUBJECT}, got {observed_counts!r}"
        )
    return manifest


def build_screen_plan(
    *,
    store_root: Path,
    output_root: Path,
    seed: int = DEFAULT_SEED,
    candidate_names: list[str] | None = None,
    split_modes: list[str] | None = None,
    validate_store: bool = True,
) -> dict[str, Any]:
    """Build a single-seed, validation-only first-round screen plan."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be one integer; multi-seed screening is not allowed")
    selected = candidate_names or list(CANDIDATES)
    unknown = set(selected) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    if len(selected) != len(set(selected)):
        raise ValueError("candidate list must be unique")
    modes = split_modes or ["random_disjoint"]
    unknown_modes = set(modes) - set(SPLIT_MODES)
    if unknown_modes:
        raise ValueError(f"unknown split modes: {sorted(unknown_modes)}")
    if len(modes) != len(set(modes)):
        raise ValueError("split modes must be unique")
    if validate_store:
        _validate_store_manifest(store_root)

    jobs: list[dict[str, Any]] = []
    deferred_candidates: list[dict[str, Any]] = []
    for mode in modes:
        for name in selected:
            candidate = CANDIDATES[name]
            if not candidate.executable_first_round:
                deferred_candidates.append(
                    {
                        "candidate": asdict(candidate),
                        "split_mode": mode,
                        "scheduled": False,
                    }
                )
                continue
            jobs.append(
                {
                    "job_name": f"{PROTOCOL_ID}__{mode}__{name}",
                    "candidate": asdict(candidate),
                    "split_mode": mode,
                    "seed": seed,
                    "store_root": str(store_root),
                    "output": str(output_root / mode / name / f"seed-{seed}"),
                    "read_roles": list(SCREENING_READ_ROLES),
                    "selection_role": SELECTION_ROLE,
                    "heldout_test_accessed": False,
                    "max_epochs": None,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "selection_metric": "participant_macro_mean_mae",
                }
            )
    plan = {
        "protocol_id": PROTOCOL_ID,
        "track": "development_only_same_subject_analogue",
        "official_pulsedb_calbased_reproduction": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_parent_split": SOURCE_PARENT_SPLIT,
        "forbidden_parent_splits": list(FORBIDDEN_PARENT_SPLITS),
        "subject_count": SUBJECT_COUNT,
        "windows_per_subject": ROLE_WINDOWS_PER_SUBJECT,
        "selection_role": SELECTION_ROLE,
        "screening_read_roles": list(SCREENING_READ_ROLES),
        "heldout_role": HELDOUT_ROLE,
        "heldout_test_accessed": False,
        "winner_only_test_policy": (
            "Select on internal_validation, then refit the winner on train plus "
            "internal_validation and evaluate heldout_test exactly once."
        ),
        "single_seed_development_screen": True,
        "seed": seed,
        "max_epochs": None,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "jobs": jobs,
        "deferred_candidates": deferred_candidates,
    }
    validate_screen_plan(plan)
    return plan


def validate_screen_plan(plan: dict[str, Any]) -> None:
    """Fail closed when model selection could see a held-out or locked role."""

    if plan.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("wrong CalBased analogue protocol_id")
    if plan.get("source_parent_split") != SOURCE_PARENT_SPLIT:
        raise ValueError("screen must derive from meta_train only")
    if set(plan.get("forbidden_parent_splits", [])) != set(FORBIDDEN_PARENT_SPLITS):
        raise ValueError("locked parent split guard is incomplete")
    if plan.get("selection_role") != SELECTION_ROLE:
        raise ValueError("model selection must use internal_validation only")
    if tuple(plan.get("screening_read_roles", [])) != SCREENING_READ_ROLES:
        raise ValueError("screen may read train and internal_validation roles only")
    if plan.get("heldout_test_accessed") is not False:
        raise ValueError("heldout_test must not be accessed during screening")
    if plan.get("single_seed_development_screen") is not True:
        raise ValueError("first-round screen must remain single-seed")
    if plan.get("max_epochs") is not None:
        raise ValueError("screen must be early-stopping-only with no epoch cap")
    if plan.get("early_stopping_patience") != EARLY_STOPPING_PATIENCE:
        raise ValueError("screen requires patience=8")
    for job in plan.get("jobs", []):
        candidate = job.get("candidate", {})
        name = candidate.get("name")
        if name not in CANDIDATES:
            raise ValueError(f"plan contains unknown candidate {name!r}")
        if tuple(job.get("read_roles", [])) != SCREENING_READ_ROLES:
            raise ValueError(f"job {name!r} attempts to read a forbidden role")
        if job.get("selection_role") != SELECTION_ROLE:
            raise ValueError(f"job {name!r} has an invalid selection role")
        if job.get("heldout_test_accessed") is not False:
            raise ValueError(f"job {name!r} accesses heldout_test")
        if candidate.get("literature_adaptation") and "adaptation" not in str(name):
            raise ValueError("literature-family candidates must be named adaptation")
        if candidate.get("seen_subject_only") is not True:
            raise ValueError("all CalBased analogue candidates must be marked seen-subject")
        if candidate.get("executable_first_round") is not True:
            raise ValueError(f"deferred candidate {name!r} must not appear in jobs")
    for deferred in plan.get("deferred_candidates", []):
        candidate = deferred.get("candidate", {})
        if candidate.get("executable_first_round") is not False:
            raise ValueError("deferred_candidates contains an executable candidate")
        if not candidate.get("deferred_reason"):
            raise ValueError("deferred candidate requires a reason")


def write_screen_plan(plan: dict[str, Any], output: Path) -> tuple[Path, Path]:
    """Write a human-auditable JSON plan and compact TSV job matrix."""

    validate_screen_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "screen_plan.json"
    tsv_path = output / "screen_jobs.tsv"
    json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = []
    for job in plan["jobs"]:
        candidate = job["candidate"]
        rows.append(
            {
                "job_name": job["job_name"],
                "candidate": candidate["name"],
                "runner": candidate["runner"],
                "backbone": candidate["backbone"] or "",
                "split_mode": job["split_mode"],
                "seed": job["seed"],
                "selection_role": job["selection_role"],
                "heldout_test_accessed": job["heldout_test_accessed"],
                "output": job["output"],
            }
        )
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    return json_path, tsv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--candidate", action="append", choices=list(CANDIDATES), dest="candidates"
    )
    parser.add_argument(
        "--split-mode", action="append", choices=list(SPLIT_MODES), dest="split_modes"
    )
    parser.add_argument(
        "--plan-without-store",
        action="store_true",
        help=(
            "Generate an inert plan before materialization. This never starts "
            "training and does not weaken role guards."
        ),
    )
    args = parser.parse_args()
    plan = build_screen_plan(
        store_root=args.store_root,
        output_root=args.output / "runs",
        seed=args.seed,
        candidate_names=args.candidates,
        split_modes=args.split_modes,
        validate_store=not args.plan_without_store,
    )
    json_path, tsv_path = write_screen_plan(plan, args.output)
    print(
        json.dumps(
            {
                "status": "planned_only",
                "training_started": False,
                "screen_plan": str(json_path),
                "screen_jobs": str(tsv_path),
                "heldout_test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
