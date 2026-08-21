import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from pulsedb_fewshot.models import (  # noqa: E402
    MultiScaleResNetEncoder,
    PopulationRegressor,
    VariableKPersonalizer,
)
from pulsedb_fewshot.round10_end_to_end import (  # noqa: E402
    METHODS,
    Round10Model,
    _loss,
    _time_decay_gru_forward,
    build_internal_report,
    configure_encoder_trainability,
)
from pulsedb_fewshot.round9_refinement import TimeDecayGRU  # noqa: E402
from pulsedb_fewshot.train import _validate_crossfit_arguments  # noqa: E402


def test_variable_k_feature_path_matches_waveform_path() -> None:
    torch.manual_seed(10)
    model = VariableKPersonalizer(
        PopulationRegressor(), use_film=False, use_quality_gate=True
    ).eval()
    query = torch.randn(2, 1, 1250)
    support = torch.randn(2, 5, 1, 1250)
    support_bp = torch.randn(2, 5, 2)
    mask = torch.ones(2, 5, dtype=torch.bool)
    with torch.no_grad():
        direct = model(query, support, support_bp, mask)
        query_features = model.population.encoder(query)
        support_features = model.population.encoder(support.reshape(10, 1, 1250))
        support_features = support_features.reshape(2, 5, -1)
        cached = model.forward_from_features(
            query_features, support_features, support_bp, mask
        )
    assert torch.equal(direct, cached)


@pytest.mark.parametrize(
    ("mode", "must_contain", "must_exclude"),
    [
        ("frozen", None, "projection"),
        ("projection", "projection", "blocks.3"),
        ("last_block", "blocks.3", "blocks.2"),
        ("last_two_blocks", "blocks.2", "blocks.1"),
        ("full", "stem", None),
    ],
)
def test_encoder_trainability_is_limited_to_requested_suffix(
    mode: str, must_contain: str | None, must_exclude: str | None
) -> None:
    encoder = MultiScaleResNetEncoder()
    names = configure_encoder_trainability(encoder, mode)
    if mode == "frozen":
        assert names == []
    if must_contain is not None:
        assert any(must_contain in name for name in names)
    if must_exclude is not None:
        assert not any(must_exclude in name for name in names)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_round10_candidate_initialization_is_finite(method: str) -> None:
    model = Round10Model(MultiScaleResNetEncoder(), METHODS[method]).eval()
    query = model.encoder(torch.randn(2, 1, 1250))
    support = model.encoder(torch.randn(10, 1, 1250)).reshape(2, 5, -1)
    base = torch.randn(2, 2)
    result = model.head.static_forward(
        query,
        support,
        torch.randn(2, 5, 2),
        torch.randn(2, 5, 2),
        base,
        torch.rand(2, 5),
        torch.empty(2, 0),
        torch.empty(2, 5, 0),
    )
    assert torch.isfinite(result["prediction"]).all()
    assert result["pair_hidden"].shape == (2, 5, 64)
    if method == "frozen_reference":
        assert torch.equal(result["prediction"], base)


def test_chunked_temporal_inference_matches_full_sequence() -> None:
    torch.manual_seed(11)
    core = TimeDecayGRU(8).eval()
    inputs = torch.randn(1, 19, 8)
    gaps = torch.rand(1, 19)
    mask = torch.ones(1, 19, dtype=torch.bool)
    full, full_final = _time_decay_gru_forward(core, inputs, gaps, mask)
    first, state = _time_decay_gru_forward(
        core, inputs[:, :7], gaps[:, :7], mask[:, :7]
    )
    second, state = _time_decay_gru_forward(
        core,
        inputs[:, 7:13],
        gaps[:, 7:13],
        mask[:, 7:13],
        initial_hidden=state,
    )
    third, chunked_final = _time_decay_gru_forward(
        core,
        inputs[:, 13:],
        gaps[:, 13:],
        mask[:, 13:],
        initial_hidden=state,
    )
    assert torch.equal(full, torch.cat([first, second, third], dim=1))
    assert torch.equal(full_final, chunked_final)


@pytest.mark.parametrize("method", ["last_block", "last_block_direction"])
def test_round10_loss_backpropagates_from_bfloat16_outputs(method: str) -> None:
    """Mirror the CUDA-autocast output dtype with float32 prepared targets."""

    prediction = torch.randn(2, 3, 2, dtype=torch.bfloat16, requires_grad=True)
    pair_delta = torch.randn(
        2, 3, 5, 2, dtype=torch.bfloat16, requires_grad=True
    )
    range_logits = torch.randn(
        2, 3, 6, dtype=torch.bfloat16, requires_grad=True
    )
    direction_logits = torch.randn(
        2, 3, 5, 2, dtype=torch.bfloat16, requires_grad=True
    )
    output = {
        "prediction": prediction,
        "pair_delta": pair_delta,
        "range_logits": range_logits,
        "direction_logits": direction_logits,
        "target": torch.randn(2, 3, 2, dtype=torch.float32),
        "support_bp": torch.randn(2, 3, 5, 2, dtype=torch.float32),
        "range": torch.randint(0, 3, (2, 3, 2)),
        "mask": torch.ones(2, 3, dtype=torch.bool),
    }
    loss = _loss(output, METHODS[method])
    assert loss.dtype == torch.float32
    loss.backward()
    assert prediction.grad is not None
    assert pair_delta.grad is not None
    assert range_logits.grad is not None
    if method == "last_block_direction":
        assert direction_logits.grad is not None


def test_explicit_internal_split_does_not_require_legacy_heldout_fold(
    tmp_path: Path,
) -> None:
    assert _validate_crossfit_arguments(
        crossfit_folds=tmp_path / "folds.parquet",
        heldout_fold=None,
        fit_folds=[0, 1, 2],
        validation_fold=3,
    ) is True


def test_explicit_internal_split_rejects_selection_fold_in_fit(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        _validate_crossfit_arguments(
            crossfit_folds=tmp_path / "folds.parquet",
            heldout_fold=None,
            fit_folds=[0, 1, 2, 3],
            validation_fold=3,
        )


def _write_run(root: Path, method: str, offset: float) -> None:
    root.mkdir()
    rows = []
    for source, participant, target_sbp, target_dbp in (
        ("MIMIC", "MIMIC:p1", 120.0, 75.0),
        ("VitalDB", "VitalDB:p2", 130.0, 80.0),
    ):
        rows.append(
            {
                "subject_uid": participant,
                "event_id": f"{participant}:6",
                "k": 5,
                "source": source,
                "target_sbp": target_sbp,
                "target_dbp": target_dbp,
                "pred_sbp": target_sbp + offset,
                "pred_dbp": target_dbp + offset,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "selection_predictions.parquet", index=False)
    (root / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "round": 10,
                "split": "meta_train_internal_fold4",
                "seed": 20260824,
                "meta_validation_used_for_training": False,
                "meta_validation_used_for_early_stopping": False,
                "meta_validation_used_for_candidate_ranking": False,
                "meta_validation_predictions_generated": False,
                "locked_test_accessed": False,
                "method": method,
            }
        ),
        encoding="utf-8",
    )


def test_round10_report_recomputes_all_scopes_and_gate(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_run(reference, "frozen_reference", 2.0)
    _write_run(candidate, "last_block", 1.0)
    output = tmp_path / "report"
    result = build_internal_report(
        runs={"T10-0 Frozen": reference, "T10-2 Last block": candidate},
        reference="T10-0 Frozen",
        output=output,
        expected_seed=20260824,
    )
    table = pd.read_csv(output / "participant_macro_internal.csv")
    assert set(table["Scope"]) == {"Overall", "MIMIC", "VitalDB"}
    assert result["winner"] == "T10-2 Last block"
    assert result["passes_internal_gate"] is True


def test_round10_report_rejects_meta_validation_use(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, "frozen_reference", 1.0)
    payload = json.loads((run / "run.json").read_text())
    payload["meta_validation_predictions_generated"] = True
    (run / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="unsafe flag"):
        build_internal_report(
            runs={"Reference": run},
            reference="Reference",
            output=tmp_path / "report",
            expected_seed=20260824,
        )
