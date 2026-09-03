"""Evaluate a leakage-safe risk-gated general/expert PPG-BP pipeline.

The deployable routes in this module use only a risk model fitted from
cross-fitted meta-train predictions.  Meta-validation query BP is read only
after routing to audit identification and prediction performance.  Oracle
tail routing is emitted separately and is never labelled deployable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .tail_risk import _score_features, build_risk_features
from .training import load_store_metadata, participant_macro_metrics, save_json


KEYS = ["subject_uid", "event_id", "k"]


def _load_validation_run(path: Path, *, k: int = 5) -> tuple[pd.DataFrame, dict[str, object]]:
    run_path = path / "run.json"
    predictions_path = path / "best_validation_predictions.parquet"
    if not run_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"incomplete validation run: {path}")
    record = json.loads(run_path.read_text(encoding="utf-8"))
    if record.get("status") != "complete" or record.get("split") != "meta_validation":
        raise AssertionError(f"run is not a complete meta-validation result: {path}")
    if record.get("locked_test_accessed") is not False:
        raise AssertionError("run does not explicitly deny locked-test access")
    frame = pd.read_parquet(predictions_path)
    frame = frame.loc[pd.to_numeric(frame["k"]).astype(int).eq(k)].copy()
    if frame.empty:
        raise AssertionError(f"run has no K={k} predictions: {path}")
    if frame.duplicated(KEYS).any():
        raise AssertionError(f"run contains duplicate query keys: {path}")
    return frame.sort_values(KEYS, kind="mergesort").reset_index(drop=True), record


def _assert_aligned(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    if not reference[KEYS].equals(candidate[KEYS]):
        raise AssertionError("general and expert query keys differ")
    for target in ("target_sbp", "target_dbp"):
        if not np.allclose(
            reference[target].to_numpy(float),
            candidate[target].to_numpy(float),
            rtol=0.0,
            atol=1e-4,
        ):
            raise AssertionError("general and expert query targets differ")


def _assert_qgate_huber_run(record: Mapping[str, object], *, label: str) -> None:
    """Reject an accidentally mixed Phase-6D general or specialist run."""
    arguments = record.get("arguments")
    if not isinstance(arguments, dict):
        raise AssertionError(f"{label} run has no argument provenance")
    expected = {
        "train_support_policy": "fixed_first",
        "loss": "huber",
        "use_quality_gate": True,
    }
    for key, value in expected.items():
        if arguments.get(key) != value:
            raise AssertionError(
                f"{label} run has {key}={arguments.get(key)!r}; expected {value!r}"
            )
    if record.get("method") != "m0":
        raise AssertionError(f"{label} run is not an M0-family result")


def _participant_errors(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"source", "subject_uid", "pred_sbp", "pred_dbp", "target_sbp", "target_dbp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction table missing {sorted(missing)}")
    work = frame.copy()
    work["ae_sbp"] = (work["pred_sbp"] - work["target_sbp"]).abs()
    work["ae_dbp"] = (work["pred_dbp"] - work["target_dbp"]).abs()
    participant = work.groupby(["source", "subject_uid"], as_index=False).agg(
        sbp_mae=("ae_sbp", "mean"),
        dbp_mae=("ae_dbp", "mean"),
        queries=("event_id", "size"),
    )
    participant["mean_mae"] = (participant["sbp_mae"] + participant["dbp_mae"]) / 2.0
    return participant


def assign_true_tail_by_source(
    participant: pd.DataFrame, *, fraction: float = 0.30
) -> pd.DataFrame:
    """Assign the evaluation-only exact error tail independently by source."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("tail fraction must be between zero and one")
    rows: list[pd.DataFrame] = []
    for source, group in participant.groupby("source", sort=True):
        ordered = group.sort_values(
            ["mean_mae", "subject_uid"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        n_tail = int(math.ceil(fraction * len(ordered)))
        ordered["true_tail_rank"] = np.arange(1, len(ordered) + 1)
        ordered["true_hard"] = ordered["true_tail_rank"].le(n_tail)
        ordered["tail_fraction"] = fraction
        rows.append(ordered)
    return pd.concat(rows, ignore_index=True)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    if not labels.any():
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    rank_sum = float(ranks[labels].sum())
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def binary_identification_metrics(
    labels: np.ndarray, scores: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    labels = labels.astype(bool)
    predicted = predicted.astype(bool)
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(~labels & predicted))
    tn = int(np.sum(~labels & ~predicted))
    fn = int(np.sum(labels & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "average_precision": _average_precision(labels, scores),
        "roc_auc": _roc_auc(labels, scores),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0,
        "balanced_accuracy": float((recall + specificity) / 2),
        "predicted_high_risk_fraction": float(predicted.mean()),
        "true_tail_fraction": float(labels.mean()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def _top_fraction_flag(frame: pd.DataFrame, *, fraction: float) -> pd.Series:
    result = pd.Series(False, index=frame.index)
    for _, group in frame.groupby("source", sort=True):
        ordered = group.sort_values(
            ["risk_score", "subject_uid"], ascending=[False, True], kind="mergesort"
        )
        result.loc[ordered.index[: int(math.ceil(fraction * len(ordered)))]] = True
    return result


def _scope_frames(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "Overall", frame
    for source in sorted(frame["source"].dropna().unique()):
        yield str(source), frame.loc[frame["source"].eq(source)]


def _prediction_metric_rows(setting: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope, selected in _scope_frames(frame):
        metric = participant_macro_metrics(selected)
        rows.append(
            {
                "setting": setting,
                "scope": scope,
                "participants": metric["n_participants"],
                "queries": metric["n_events"],
                "sbp_mae": metric["sbp_mae"],
                "dbp_mae": metric["dbp_mae"],
                "mean_mae": metric["mean_mae"],
                "worst_30_mean_mae": metric["worst_30_mean_mae"],
                "retained_70_mean_mae": metric["retained_70_mean_mae"],
            }
        )
    return rows


def _pooled_rows(setting: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope, selected in _scope_frames(frame):
        for bp in ("sbp", "dbp"):
            target = selected[f"target_{bp}"].to_numpy(float)
            prediction = selected[f"pred_{bp}"].to_numpy(float)
            error = prediction - target
            absolute = np.abs(error)
            denominator = float(np.sum((target - target.mean()) ** 2))
            r_squared = float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan")
            std = float(np.std(error, ddof=1))
            within_5 = float(np.mean(absolute <= 5.0) * 100)
            within_10 = float(np.mean(absolute <= 10.0) * 100)
            within_15 = float(np.mean(absolute <= 15.0) * 100)
            aami = "PASS*" if abs(float(error.mean())) <= 5.0 and std <= 8.0 else "FAIL*"
            if within_5 >= 60 and within_10 >= 85 and within_15 >= 95:
                grade = "A"
            elif within_5 >= 50 and within_10 >= 75 and within_15 >= 90:
                grade = "B"
            elif within_5 >= 40 and within_10 >= 65 and within_15 >= 85:
                grade = "C"
            else:
                grade = "D"
            rows.append(
                {
                    "Setting": setting,
                    "Scope": scope,
                    "BP": bp.upper(),
                    "MAE": float(absolute.mean()),
                    "R2": r_squared,
                    "ME": float(error.mean()),
                    "STD": std,
                    "within_5_mmHg": within_5,
                    "within_10_mmHg": within_10,
                    "within_15_mmHg": within_15,
                    "AAMI": aami,
                    "BHS": f"Grade {grade}*",
                }
            )
    return rows


def _parse_experts(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("expert must be supplied as Setting=/path/to/run")
        setting, path = value.split("=", 1)
        setting = setting.strip()
        if not setting or setting in result:
            raise ValueError(f"invalid or duplicate expert setting: {setting!r}")
        result[setting] = Path(path)
    if not result:
        raise ValueError("at least one expert is required")
    return result


def evaluate_phase6d_pipeline(
    store_root: Path,
    general_run: Path,
    experts: Mapping[str, Path],
    risk_checkpoint: Path,
    output_dir: Path,
    *,
    tail_fraction: float = 0.30,
) -> dict[str, object]:
    general, general_record = _load_validation_run(general_run, k=5)
    _assert_qgate_huber_run(general_record, label="general")
    metadata = load_store_metadata(store_root, "development")
    source_lookup = metadata.loc[
        metadata["split"].eq("meta_validation") & metadata["common_query"],
        ["subject_uid", "event_id", "source"],
    ]
    general = general.merge(
        source_lookup, on=["subject_uid", "event_id"], how="left", validate="one_to_one"
    )
    if general["source"].isna().any():
        raise AssertionError("general predictions could not be assigned a source")

    checkpoint = torch.load(risk_checkpoint, map_location="cpu", weights_only=False)
    risk_record_path = risk_checkpoint.with_name("risk_run.json")
    if not risk_record_path.is_file():
        raise FileNotFoundError("risk checkpoint is missing its risk_run.json provenance")
    risk_record = json.loads(risk_record_path.read_text(encoding="utf-8"))
    if (
        risk_record.get("status") != "complete"
        or risk_record.get("split") != "meta_train_crossfit_risk_validation"
        or risk_record.get("locked_test_accessed") is not False
    ):
        raise AssertionError("risk model is not a complete cross-fitted meta-train result")
    event_threshold = float(checkpoint["event_threshold"])
    participant_threshold = float(checkpoint["participant_threshold"])
    features = build_risk_features(store_root, general)
    risk_scores = _score_features(features, risk_checkpoint)
    if len(risk_scores) != len(general):
        raise AssertionError("risk score count does not match general predictions")
    general["risk_score"] = risk_scores
    general["event_predicted_high"] = general["risk_score"].ge(event_threshold)

    participant = assign_true_tail_by_source(
        _participant_errors(general), fraction=tail_fraction
    )
    source_participants = participant.groupby("source")["subject_uid"].nunique()
    if int(source_participants.sum()) != int(participant["subject_uid"].nunique()):
        raise AssertionError("source participant counts do not sum to Overall")
    participant_scores = general.groupby(["source", "subject_uid"], as_index=False).agg(
        risk_score=("risk_score", "mean")
    )
    participant = participant.merge(
        participant_scores,
        on=["source", "subject_uid"],
        how="left",
        validate="one_to_one",
    )
    participant["threshold_predicted_high"] = participant["risk_score"].ge(
        participant_threshold
    )
    participant["top30_predicted_high"] = _top_fraction_flag(
        participant, fraction=tail_fraction
    )
    participant_flags = participant[
        [
            "source",
            "subject_uid",
            "true_hard",
            "threshold_predicted_high",
            "top30_predicted_high",
            "risk_score",
            "mean_mae",
        ]
    ]
    general = general.merge(
        participant_flags,
        on=["source", "subject_uid"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_participant"),
    )

    identification_rows: list[dict[str, object]] = []
    separation_rows: list[dict[str, object]] = []
    for scope, group in _scope_frames(participant):
        for policy, predicted_column in (
            ("frozen_meta_train_threshold", "threshold_predicted_high"),
            ("fixed_top30_risk_diagnostic", "top30_predicted_high"),
        ):
            metrics = binary_identification_metrics(
                group["true_hard"].to_numpy(bool),
                group["risk_score"].to_numpy(float),
                group[predicted_column].to_numpy(bool),
            )
            identification_rows.append(
                {
                    "scope": scope,
                    "policy": policy,
                    "participants": int(len(group)),
                    **metrics,
                }
            )
        for label, selected in group.groupby("threshold_predicted_high", sort=True):
            separation_rows.append(
                {
                    "scope": scope,
                    "risk_group": "predicted_high" if bool(label) else "predicted_low",
                    "participants": int(len(selected)),
                    "sbp_mae": float(selected["sbp_mae"].mean()),
                    "dbp_mae": float(selected["dbp_mae"].mean()),
                    "mean_mae": float(selected["mean_mae"].mean()),
                    "true_tail_fraction": float(selected["true_hard"].mean()),
                }
            )

    prediction_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    saved_predictions: list[pd.DataFrame] = []

    prediction_rows.extend(_prediction_metric_rows("General QGate + Huber", general))
    pooled_rows.extend(_pooled_rows("General QGate + Huber", general))

    for expert_name, expert_path in experts.items():
        expert, expert_record = _load_validation_run(expert_path, k=5)
        _assert_qgate_huber_run(expert_record, label=expert_name)
        _assert_aligned(general, expert)
        expert = expert.merge(
            source_lookup,
            on=["subject_uid", "event_id"],
            how="left",
            validate="one_to_one",
        )
        expert["risk_score"] = general["risk_score"].to_numpy(float)
        expert["event_predicted_high"] = general["event_predicted_high"].to_numpy(bool)
        expert["threshold_predicted_high"] = general["threshold_predicted_high"].to_numpy(bool)
        expert["top30_predicted_high"] = general["top30_predicted_high"].to_numpy(bool)
        expert["true_hard"] = general["true_hard"].to_numpy(bool)

        routes: list[tuple[str, pd.DataFrame, bool, str]] = []
        routes.append((f"{expert_name} standalone", expert.copy(), True, "standalone"))
        for route_name, flag, deployable, boundary in (
            ("event hard route", general["event_predicted_high"].to_numpy(bool), True, "current-event input-visible"),
            ("participant threshold route", general["threshold_predicted_high"].to_numpy(bool), False, "retrospective participant score aggregation"),
            ("participant top30 route", general["top30_predicted_high"].to_numpy(bool), False, "cohort-level fixed-coverage diagnostic"),
            ("oracle true-tail route", general["true_hard"].to_numpy(bool), False, "query-error oracle"),
        ):
            routed = general.copy()
            for bp in ("sbp", "dbp"):
                routed[f"pred_{bp}"] = np.where(
                    flag,
                    expert[f"pred_{bp}"].to_numpy(float),
                    general[f"pred_{bp}"].to_numpy(float),
                )
            routes.append((f"{expert_name} | {route_name}", routed, deployable, boundary))
        soft = general.copy()
        for bp in ("sbp", "dbp"):
            soft[f"pred_{bp}"] = (
                (1.0 - risk_scores) * general[f"pred_{bp}"].to_numpy(float)
                + risk_scores * expert[f"pred_{bp}"].to_numpy(float)
            )
        routes.append((f"{expert_name} | event soft fusion", soft, True, "current-event input-visible"))

        for setting, routed, deployable, boundary in routes:
            prediction_rows.extend(_prediction_metric_rows(setting, routed))
            pooled_rows.extend(_pooled_rows(setting, routed))
            compact = routed[
                [
                    "subject_uid",
                    "event_id",
                    "k",
                    "source",
                    "target_sbp",
                    "target_dbp",
                    "pred_sbp",
                    "pred_dbp",
                    "risk_score",
                    "event_predicted_high",
                    "threshold_predicted_high",
                    "top30_predicted_high",
                    "true_hard",
                ]
            ].copy()
            compact["setting"] = setting
            compact["deployable"] = deployable
            compact["routing_boundary"] = boundary
            compact["expert_job"] = str(expert_record.get("slurm_job_id"))
            saved_predictions.append(compact)

        for scope, group in _scope_frames(general):
            true_ids = set(
                participant.loc[
                    participant["true_hard"]
                    & (participant["source"].eq(scope) if scope != "Overall" else True),
                    "subject_uid",
                ]
            )
            general_tail = group.loc[group["subject_uid"].isin(true_ids)]
            expert_tail = expert.loc[expert["subject_uid"].isin(true_ids)]
            general_metric = participant_macro_metrics(general_tail)
            expert_metric = participant_macro_metrics(expert_tail)
            oracle_rows.append(
                {
                    "expert": expert_name,
                    "scope": scope,
                    "true_tail_participants": general_metric["n_participants"],
                    "general_tail_sbp_mae": general_metric["sbp_mae"],
                    "general_tail_dbp_mae": general_metric["dbp_mae"],
                    "general_tail_mean_mae": general_metric["mean_mae"],
                    "expert_tail_sbp_mae": expert_metric["sbp_mae"],
                    "expert_tail_dbp_mae": expert_metric["dbp_mae"],
                    "expert_tail_mean_mae": expert_metric["mean_mae"],
                    "expert_minus_general_tail_mean_mae": expert_metric["mean_mae"]
                    - general_metric["mean_mae"],
                    "oracle_only": True,
                }
            )

    identification = pd.DataFrame(identification_rows)
    separation = pd.DataFrame(separation_rows)
    prediction_metrics = pd.DataFrame(prediction_rows)
    pooled = pd.DataFrame(pooled_rows)
    oracle = pd.DataFrame(oracle_rows)
    all_predictions = pd.concat(saved_predictions, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    identification.to_csv(output_dir / "risk_identification_metrics.csv", index=False)
    separation.to_csv(output_dir / "predicted_risk_group_errors.csv", index=False)
    prediction_metrics.to_csv(output_dir / "pipeline_participant_macro.csv", index=False)
    pooled.to_csv(output_dir / "pipeline_pooled_metrics.csv", index=False)
    oracle.to_csv(output_dir / "oracle_expert_tail_comparison.csv", index=False)
    participant.to_parquet(output_dir / "private_participant_risk_audit.parquet", index=False)
    all_predictions.to_parquet(output_dir / "pipeline_predictions.parquet", index=False)
    payload = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "meta_validation",
        "locked_test_accessed": False,
        "k": 5,
        "general_job": general_record.get("slurm_job_id"),
        "risk_threshold_source": "meta_train_crossfit_risk_validation",
        "event_threshold": event_threshold,
        "participant_threshold": participant_threshold,
        "tail_fraction": tail_fraction,
        "true_tail_definition": (
            "evaluation-only within-source general-model participant mean MAE; "
            "never used for routing"
        ),
        "experts": list(experts),
        "participants": int(participant["subject_uid"].nunique()),
        "queries": int(len(general)),
        "claim_limit": (
            "single-seed development evaluation; participant aggregation, top-30%, "
            "and true-tail routes are diagnostic; locked test and external validity "
            "are not established"
        ),
    }
    save_json(output_dir / "pipeline_run.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--general-run", type=Path, required=True)
    parser.add_argument("--expert", action="append", default=[], required=True)
    parser.add_argument("--risk-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-fraction", type=float, default=0.30)
    args = parser.parse_args()
    payload = evaluate_phase6d_pipeline(
        args.store_root,
        args.general_run,
        _parse_experts(args.expert),
        args.risk_checkpoint,
        args.output_dir,
        tail_fraction=args.tail_fraction,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
