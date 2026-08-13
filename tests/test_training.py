import pandas as pd

from pulsedb_fewshot.training import participant_macro_metrics


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
