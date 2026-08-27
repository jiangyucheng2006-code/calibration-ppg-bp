from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pulsedb_fewshot.round14_confirmation import build_confirmation_report


SEEDS = (20260827, 20260828, 20260829, 20260830, 20260831)


def _write_evaluation(
    root: Path,
    *,
    seed: int,
    backbone: str,
    offset: float,
    source_hash: str,
) -> None:
    root.mkdir()
    rows: list[dict[str, object]] = []
    for source, prefix in (("MIMIC", "m"), ("VitalDB", "v")):
        for participant_index in range(2):
            participant = f"{source}:{prefix}{participant_index}"
            for event_index in (6, 7):
                target_sbp = 118.0 + participant_index * 8 + event_index
                target_dbp = 70.0 + participant_index * 4 + event_index / 2
                rows.append(
                    {
                        "subject_uid": participant,
                        "event_id": f"{participant}:{event_index}",
                        "k": 5,
                        "source": source,
                        "split": "meta_train",
                        "fold": 4,
                        "target_sbp": target_sbp,
                        "target_dbp": target_dbp,
                        "pred_sbp": target_sbp + offset,
                        "pred_dbp": target_dbp + offset,
                    }
                )
    pd.DataFrame(rows).to_parquet(root / "queries.parquet", index=False)
    safety = {
        "meta_validation_accessed": False,
        "meta_validation_used_for_training": False,
        "meta_validation_used_for_early_stopping": False,
        "meta_validation_used_for_candidate_ranking": False,
        "meta_validation_predictions_generated": False,
        "locked_test_accessed": False,
        "query_bp_model_input": False,
        "future_query_model_input": False,
        "source_model_input": False,
    }
    payload = {
        "status": "complete",
        "round": 13 if seed == 20260827 else 14,
        "stage": "backbone_evaluation",
        "backbone": backbone,
        "seed": seed,
        "split": "meta_train_internal_only",
        "fit_folds": [0, 1, 2],
        "early_stopping_fold": 3,
        "selection_fold": 4,
        "support_policy": "fixed_first",
        "k": 5,
        "evaluation_source_tree_sha256": source_hash,
        "qgh_parameter_counts": {
            "total": 200 if backbone == "resnet_small" else 300,
            "trainable": 50,
        },
        "training_audit": {
            "status": "pass",
            "source_tree_sha256": source_hash,
            "store_manifest_sha256": "store-a",
            "crossfit_folds_sha256": "folds-a",
            "population_microbatch": 32,
            "population_accumulation": 4,
            "population_effective_batch": 128,
            "qgh_microbatch": 16,
            "qgh_accumulation": 4,
            "qgh_effective_batch": 64,
            "episodes_per_epoch": 99968,
        },
        **safety,
    }
    (root / "run.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_runs(
    tmp_path: Path,
    *,
    bad_seeds: set[int] | None = None,
) -> tuple[dict[int, Path], dict[int, Path]]:
    bad_seeds = bad_seeds or set()
    references: dict[int, Path] = {}
    candidates: dict[int, Path] = {}
    for seed in SEEDS:
        source_hash = "source-r13" if seed == 20260827 else "source-r14"
        reference = tmp_path / f"reference-{seed}"
        candidate = tmp_path / f"candidate-{seed}"
        _write_evaluation(
            reference,
            seed=seed,
            backbone="resnet_small",
            offset=1.0,
            source_hash=source_hash,
        )
        _write_evaluation(
            candidate,
            seed=seed,
            backbone="inception_time_wide",
            offset=1.2 if seed in bad_seeds else 0.5,
            source_hash=source_hash,
        )
        references[seed] = reference
        candidates[seed] = candidate
    return references, candidates


def test_round14_confirmation_builds_paired_three_scope_report(
    tmp_path: Path,
) -> None:
    references, candidates = _make_runs(tmp_path)
    output = tmp_path / "report"
    result = build_confirmation_report(
        reference_runs=references,
        candidate_runs=candidates,
        expected_seeds=SEEDS,
        discovery_seed=20260827,
        output=output,
    )

    per_seed = pd.read_csv(output / "per_seed_participant_macro_internal.csv")
    gains = pd.read_csv(output / "paired_gains_internal.csv")
    diagnostics = pd.read_csv(
        output / "cross_seed_ensemble_diagnostics_internal.csv"
    )
    assert set(per_seed["Seed"]) == set(SEEDS)
    assert set(per_seed["Scope"]) == {"Overall", "MIMIC", "VitalDB"}
    assert set(per_seed["Setting"]) == {
        "resnet_small | QGH",
        "inception_time_wide | QGH",
    }
    assert set(gains["Scope"]) == {"Overall", "MIMIC", "VitalDB"}
    assert gains["Mean MAE gain"].eq(0.5).all()
    assert set(diagnostics["Setting"]) == {
        "resnet_small | 5-seed equal-weight QGH ensemble",
        "inception_time_wide | 5-seed equal-weight QGH ensemble",
    }
    assert result["passes_confirmation_gate"] is True
    assert result["five_seed_background_overall_positive_seed_count"] == 5
    assert result["four_new_seed_overall_positive_seed_count"] == 4
    assert result["four_new_seed_mean_gain_vs_reference"]["Overall"] == pytest.approx(
        0.5
    )
    assert result["meta_validation_accessed"] is False
    assert result["locked_test_accessed"] is False


def test_round14_confirmation_rejects_locked_test_access(tmp_path: Path) -> None:
    references, candidates = _make_runs(tmp_path)
    path = candidates[20260829] / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["locked_test_accessed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssertionError, match="unsafe flag locked_test_accessed"):
        build_confirmation_report(
            reference_runs=references,
            candidate_runs=candidates,
            expected_seeds=SEEDS,
            discovery_seed=20260827,
            output=tmp_path / "report",
        )


def test_round14_confirmation_rejects_noncommon_targets(tmp_path: Path) -> None:
    references, candidates = _make_runs(tmp_path)
    path = candidates[20260830] / "queries.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "target_sbp"] += 1.0
    frame.to_parquet(path, index=False)

    with pytest.raises(AssertionError, match="different query targets"):
        build_confirmation_report(
            reference_runs=references,
            candidate_runs=candidates,
            expected_seeds=SEEDS,
            discovery_seed=20260827,
            output=tmp_path / "report",
        )


def test_round14_confirmation_rejects_new_seed_source_drift(tmp_path: Path) -> None:
    references, candidates = _make_runs(tmp_path)
    for root in (references[20260831], candidates[20260831]):
        path = root / "run.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["evaluation_source_tree_sha256"] = "source-r14-b"
        payload["training_audit"]["source_tree_sha256"] = "source-r14-b"
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssertionError, match="one source tree"):
        build_confirmation_report(
            reference_runs=references,
            candidate_runs=candidates,
            expected_seeds=SEEDS,
            discovery_seed=20260827,
            output=tmp_path / "report",
        )


def test_round14_confirmation_requires_four_positive_overall_seeds(
    tmp_path: Path,
) -> None:
    references, candidates = _make_runs(
        tmp_path, bad_seeds={20260830, 20260831}
    )
    result = build_confirmation_report(
        reference_runs=references,
        candidate_runs=candidates,
        expected_seeds=SEEDS,
        discovery_seed=20260827,
        output=tmp_path / "report",
    )
    assert (
        result["five_seed_background_mean_gain_vs_reference"]["Overall"] >= 0.15
    )
    assert result["five_seed_background_overall_positive_seed_count"] == 3
    assert result["four_new_seed_overall_positive_seed_count"] == 2
    assert (
        result["gate"][
            "four_new_seeds_overall_positive_count_at_least_3_of_4"
        ]
        is False
    )
    assert result["passes_confirmation_gate"] is False


def _set_prediction_offset(root: Path, offset: float) -> None:
    path = root / "queries.parquet"
    frame = pd.read_parquet(path)
    frame["pred_sbp"] = frame["target_sbp"] + offset
    frame["pred_dbp"] = frame["target_dbp"] + offset
    frame.to_parquet(path, index=False)


def test_round14_confirmation_rejects_negative_new_seed_mean_gain(
    tmp_path: Path,
) -> None:
    references, candidates = _make_runs(tmp_path)
    _set_prediction_offset(references[20260827], 3.0)
    _set_prediction_offset(candidates[20260827], 0.5)
    for seed in (20260828, 20260829, 20260830):
        _set_prediction_offset(candidates[seed], 0.9)
    _set_prediction_offset(candidates[20260831], 1.5)

    result = build_confirmation_report(
        reference_runs=references,
        candidate_runs=candidates,
        expected_seeds=SEEDS,
        discovery_seed=20260827,
        output=tmp_path / "report",
    )
    assert (
        result["five_seed_background_mean_gain_vs_reference"]["Overall"] >= 0.15
    )
    assert result["five_seed_background_overall_positive_seed_count"] == 4
    assert result["four_new_seed_overall_positive_seed_count"] == 3
    assert result["four_new_seed_mean_gain_vs_reference"]["Overall"] < 0.0
    assert (
        result["gate"][
            "four_new_seeds_overall_mean_gain_at_least_0_15"
        ]
        is False
    )
    assert result["passes_confirmation_gate"] is False


def test_round14_cross_seed_ensembles_average_each_backbone_separately(
    tmp_path: Path,
) -> None:
    references, candidates = _make_runs(tmp_path)
    reference_offsets = dict(zip(SEEDS, (1.0, 1.1, 1.2, 1.3, 1.4), strict=True))
    candidate_offsets = dict(zip(SEEDS, (0.1, 0.2, 0.3, 0.4, 0.5), strict=True))
    for seed in SEEDS:
        _set_prediction_offset(references[seed], reference_offsets[seed])
        _set_prediction_offset(candidates[seed], candidate_offsets[seed])

    output = tmp_path / "report"
    build_confirmation_report(
        reference_runs=references,
        candidate_runs=candidates,
        expected_seeds=SEEDS,
        discovery_seed=20260827,
        output=output,
    )
    participant = pd.read_csv(
        output / "cross_seed_ensemble_participant_macro_internal.csv"
    ).set_index(["Setting", "Scope"])
    pooled = pd.read_csv(
        output / "cross_seed_ensemble_diagnostics_internal.csv"
    ).set_index(["Setting", "Scope", "BP"])

    reference_key = (
        "resnet_small | 5-seed equal-weight QGH ensemble",
        "Overall",
    )
    candidate_key = (
        "inception_time_wide | 5-seed equal-weight QGH ensemble",
        "Overall",
    )
    assert participant.loc[reference_key, "Mean participant-macro MAE"] == pytest.approx(
        1.2
    )
    assert participant.loc[candidate_key, "Mean participant-macro MAE"] == pytest.approx(
        0.3
    )
    assert pooled.loc[(*reference_key, "SBP"), "MAE"] == pytest.approx(1.2)
    assert pooled.loc[(*candidate_key, "DBP"), "MAE"] == pytest.approx(0.3)
