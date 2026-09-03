import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from pulsedb_fewshot.train import (  # noqa: E402
    ParticipantEpisodeBatchSampler,
    _episodic_balanced_sampler,
    _participant_tail_objective,
)
from pulsedb_fewshot.training import participant_macro_metrics  # noqa: E402


def test_participant_tail_objective_aggregates_duplicate_participants_first() -> None:
    prediction = torch.tensor(
        [[0.0, 0.0], [2.0, 2.0], [1.0, 1.0], [3.0, 3.0]],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)

    objective, diagnostics = _participant_tail_objective(
        prediction,
        target,
        ["a", "a", "b", "c"],
        loss_name="mse",
        huber_delta=0.5,
        tail_fraction=0.30,
        tail_weight=0.50,
    )

    # Per-participant risks are a=2, b=1 and c=9.  The highest ceil(.3*3)=1
    # participant defines CVaR, so the combined risk is .5*4 + .5*9 = 6.5.
    assert objective.item() == pytest.approx(6.5)
    assert diagnostics["batch_participants"] == 3
    assert diagnostics["tail_participants"] == 1
    objective.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_participant_tail_objective_rejects_invalid_configuration() -> None:
    prediction = torch.zeros(2, 2)
    target = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="tail_fraction"):
        _participant_tail_objective(
            prediction,
            target,
            ["a", "b"],
            loss_name="mse",
            huber_delta=0.5,
            tail_fraction=0.0,
            tail_weight=0.5,
        )


def test_participant_episode_batch_sampler_repeats_distinct_participants() -> None:
    class DatasetStub:
        participant_ids = [
            "a",
            "a",
            "a",
            "b",
            "b",
            "b",
            "c",
            "c",
            "c",
            "d",
            "d",
            "d",
            "e",
            "e",
            "e",
        ]

        def __len__(self) -> int:
            return len(self.participant_ids)

    dataset = DatasetStub()
    sampler = ParticipantEpisodeBatchSampler(
        dataset,  # type: ignore[arg-type]
        seed=7,
        num_samples=8,
        batch_size=8,
        episodes_per_participant=2,
    )
    batch = next(iter(sampler))
    identities = [dataset.participant_ids[index] for index in batch]
    counts = pd.Series(identities).value_counts()
    assert len(batch) == 8
    assert len(counts) == 4
    assert set(counts) == {2}


def test_participant_macro_tail_is_exact_ceil_30_percent_with_stable_ties() -> None:
    predictions = pd.DataFrame(
        {
            "subject_uid": ["s3", "s1", "s2", "s4"],
            "event_id": ["e3", "e1", "e2", "e4"],
            "target_sbp": [100.0] * 4,
            "target_dbp": [60.0] * 4,
            "pred_sbp": [110.0, 110.0, 104.0, 100.0],
            "pred_dbp": [70.0, 70.0, 64.0, 60.0],
        }
    )

    metrics = participant_macro_metrics(predictions)

    assert metrics["worst_30_n_participants"] == 2
    assert metrics["retained_70_n_participants"] == 2
    assert metrics["worst_30_mean_mae"] == pytest.approx(10.0)
    assert metrics["retained_70_mean_mae"] == pytest.approx(2.0)
    assert str(metrics["tail_definition"]).startswith("oracle diagnostic")


def test_cross_fitted_hard_participant_multiplier_changes_sampling_weight() -> None:
    class DatasetStub:
        participant_ids = ["easy", "easy", "hard", "hard"]

        def __len__(self) -> int:
            return len(self.participant_ids)

    sampler = _episodic_balanced_sampler(
        DatasetStub(),  # type: ignore[arg-type]
        seed=7,
        num_samples=100,
        participant_multipliers={"easy": 1.0, "hard": 4.0},
    )

    assert sampler.weights.tolist() == pytest.approx([0.5, 0.5, 2.0, 2.0])


def test_cross_fitted_multiplier_requires_complete_subject_coverage() -> None:
    class DatasetStub:
        participant_ids = ["easy", "hard"]

        def __len__(self) -> int:
            return len(self.participant_ids)

    with pytest.raises(ValueError, match="does not cover"):
        _episodic_balanced_sampler(
            DatasetStub(),  # type: ignore[arg-type]
            seed=7,
            num_samples=10,
            participant_multipliers={"easy": 1.0},
        )
