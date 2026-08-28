import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from pulsedb_fewshot.calbased_screen import (
    CANDIDATES,
    PROTOCOL_ID,
    ROLE_WINDOWS_PER_SUBJECT,
    SUBJECT_COUNT,
    build_screen_plan,
    fit_subject_train_means,
    predict_subject_train_mean,
    validate_screen_plan,
    write_screen_plan,
)


def _write_store_manifest(store_root: Path) -> None:
    store_root.mkdir()
    (store_root / "materialization.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "source_parent_split": "meta_train",
                "source_parent_splits": ["meta_train"],
                "subject_count": SUBJECT_COUNT,
                "windows_per_subject": ROLE_WINDOWS_PER_SUBJECT,
            }
        ),
        encoding="utf-8",
    )


def test_candidate_matrix_names_every_literature_family_as_adaptation() -> None:
    assert "subject_train_mean" in CANDIDATES
    assert "subject_mean_residual_ppg" in CANDIDATES
    assert "inception_time_wide" in CANDIDATES
    assert "patch_transformer" in CANDIDATES
    assert "compact_resnet_qgh" in CANDIDATES
    assert "compact_resnet_calibration_relative" in CANDIDATES
    adaptations = [
        candidate for candidate in CANDIDATES.values() if candidate.literature_adaptation
    ]
    assert len(adaptations) == 4
    assert all("adaptation" in candidate.name for candidate in adaptations)
    assert all(
        "exact" in candidate.description and "reproduction" in candidate.description
        for candidate in adaptations
    )


def test_screen_plan_is_single_seed_validation_only_and_never_reads_test(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    _write_store_manifest(store_root)
    plan = build_screen_plan(
        store_root=store_root,
        output_root=tmp_path / "runs",
        seed=11,
        split_modes=["random_disjoint", "chronological_blocked"],
    )
    assert plan["source_parent_split"] == "meta_train"
    assert plan["selection_role"] == "internal_validation"
    assert plan["heldout_test_accessed"] is False
    assert plan["single_seed_development_screen"] is True
    assert plan["max_epochs"] is None
    assert plan["early_stopping_patience"] == 8
    executable = sum(
        candidate.executable_first_round for candidate in CANDIDATES.values()
    )
    assert len(plan["jobs"]) == 2 * executable
    assert len(plan["deferred_candidates"]) == 2 * (len(CANDIDATES) - executable)
    assert {tuple(job["read_roles"]) for job in plan["jobs"]} == {
        ("train", "internal_validation")
    }
    assert not any("heldout_test" in job["read_roles"] for job in plan["jobs"])


def test_screen_plan_fails_closed_if_a_job_reads_heldout_test(tmp_path: Path) -> None:
    plan = build_screen_plan(
        store_root=tmp_path / "not_materialized",
        output_root=tmp_path / "runs",
        validate_store=False,
    )
    leaked = copy.deepcopy(plan)
    leaked["jobs"][0]["read_roles"].append("heldout_test")
    with pytest.raises(ValueError, match="forbidden role"):
        validate_screen_plan(leaked)


def test_store_manifest_rejects_meta_validation_or_locked_meta_test(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    _write_store_manifest(store_root)
    manifest_path = store_root / "materialization.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_parent_splits"] = ["meta_train", "meta_validation"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden parent split"):
        build_screen_plan(
            store_root=store_root,
            output_root=tmp_path / "runs",
        )


def test_subject_train_mean_uses_train_labels_only() -> None:
    train = pd.DataFrame(
        {
            "role": ["train", "train", "train", "train"],
            "subject_uid": ["a", "a", "b", "b"],
            "sbp": [100.0, 120.0, 130.0, 150.0],
            "dbp": [60.0, 70.0, 80.0, 90.0],
        }
    )
    means = fit_subject_train_means(train)
    validation = pd.DataFrame(
        {
            "subject_uid": ["a", "b"],
            "sbp": [1.0, 999.0],
            "dbp": [2.0, 888.0],
        }
    )
    first = predict_subject_train_mean(validation, means)
    changed_targets = validation.assign(sbp=[555.0, 444.0], dbp=[333.0, 222.0])
    second = predict_subject_train_mean(changed_targets, means)
    pd.testing.assert_frame_equal(first, second)
    assert first[["sbp_pred", "dbp_pred"]].to_numpy().tolist() == [
        [110.0, 65.0],
        [140.0, 85.0],
    ]
    contaminated = train.copy()
    contaminated.loc[0, "role"] = "internal_validation"
    with pytest.raises(ValueError, match="train role only"):
        fit_subject_train_means(contaminated)


def test_plan_writer_creates_auditable_json_and_tsv(tmp_path: Path) -> None:
    plan = build_screen_plan(
        store_root=tmp_path / "future_store",
        output_root=tmp_path / "runs",
        candidate_names=["subject_train_mean", "subject_mean_residual_ppg"],
        validate_store=False,
    )
    json_path, tsv_path = write_screen_plan(plan, tmp_path / "plan")
    assert json.loads(json_path.read_text(encoding="utf-8"))["heldout_test_accessed"] is False
    jobs = pd.read_csv(tsv_path, sep="\t")
    assert jobs["candidate"].tolist() == [
        "subject_train_mean",
        "subject_mean_residual_ppg",
    ]
    assert not jobs["heldout_test_accessed"].any()


def test_subject_mean_residual_model_starts_at_subject_mean() -> None:
    torch = pytest.importorskip("torch")
    from pulsedb_fewshot.models import SubjectMeanResidualRegressor

    model = SubjectMeanResidualRegressor().eval()
    ppg = torch.randn(2, 1, 250)
    subject_mean = torch.tensor([[120.0, 70.0], [135.0, 82.0]])
    with torch.no_grad():
        prediction = model(ppg, subject_mean)
    assert torch.equal(prediction, subject_mean)
