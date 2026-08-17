from pathlib import Path

import pandas as pd

from pulsedb_fewshot.analyze_tail_bp_ranges import analyze


def test_bp_range_analysis_is_participant_macro(tmp_path: Path) -> None:
    events_path = tmp_path / "events.parquet"
    participants_path = tmp_path / "participants.csv"
    output = tmp_path / "output"
    pd.DataFrame(
        {
            "subject_uid": ["a", "a", "b", "b"],
            "event_id": ["a1", "a2", "b1", "b2"],
            "target_sbp": [100.0, 120.0, 140.0, 160.0],
            "target_dbp": [55.0, 65.0, 75.0, 85.0],
            "abs_error_sbp": [1.0, 2.0, 3.0, 4.0],
            "abs_error_dbp": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_parquet(events_path, index=False)
    pd.DataFrame(
        {
            "subject_uid": ["a", "b"],
            "mean_mae": [1.5, 3.5],
            "high_error_q80": [False, True],
            "target_sbp_mean": [110.0, 150.0],
            "target_dbp_mean": [60.0, 80.0],
            "target_sbp_std": [10.0, 20.0],
            "target_dbp_std": [5.0, 10.0],
            "abs_delta_support_sbp_mean": [5.0, 15.0],
            "abs_delta_support_dbp_mean": [3.0, 9.0],
            "event_index_mean": [10.0, 20.0],
        }
    ).to_csv(participants_path, index=False)
    summary = analyze(events_path, participants_path, output)
    ranges = pd.read_csv(output / "bp_range_participant_macro.csv")
    assert summary["locked_meta_test_accessed"] is False
    assert set(ranges["tail_group"]) == {
        "remaining_80pct_subjects",
        "worst_20pct_subjects",
    }
    worst_high = ranges.loc[
        ranges["tail_group"].eq("worst_20pct_subjects")
        & ranges["bp"].eq("SBP")
        & ranges["bp_range"].eq(">=150"),
        "participant_macro_event_fraction",
    ].iloc[0]
    assert worst_high == 0.5
