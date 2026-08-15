import itertools

import pandas as pd
import pytest

from pulsedb_fewshot.train import _epoch_numbers
from pulsedb_fewshot.training import fit_target_scaler, participant_macro_metrics


def test_participant_macro_metrics_averages_participants_not_events() -> None:
    predictions = pd.DataFrame(
        {
            "subject_uid": ["s1", "s1", "s2"],
            "event_id": ["e1", "e2", "e3"],
            "target_sbp": [100.0, 100.0, 100.0],
            "target_dbp": [60.0, 60.0, 60.0],
            "pred_sbp": [100.0, 100.0, 110.0],
            "pred_dbp": [60.0, 60.0, 64.0],
        }
    )
    metrics = participant_macro_metrics(predictions)
    assert metrics["n_participants"] == 2
    assert metrics["n_events"] == 3
    assert metrics["sbp_mae"] == 5.0
    assert metrics["dbp_mae"] == 2.0


def test_target_scaler_uses_meta_train_only() -> None:
    metadata = pd.DataFrame(
        {
            "split": ["meta_train", "meta_train", "meta_validation"],
            "sbp": [100.0, 120.0, 1000.0],
            "dbp": [60.0, 80.0, 1000.0],
        }
    )

    scaler = fit_target_scaler(metadata)

    assert scaler["mean"] == [110.0, 70.0]
    assert scaler["std"] == [10.0, 10.0]


def test_epoch_numbers_supports_early_stopping_only_mode() -> None:
    assert list(_epoch_numbers(3)) == [1, 2, 3]
    assert list(itertools.islice(_epoch_numbers(0), 5)) == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="nonnegative"):
        list(_epoch_numbers(-1))
