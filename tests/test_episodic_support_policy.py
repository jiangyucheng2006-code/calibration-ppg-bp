from pathlib import Path

import numpy as np
import pandas as pd

from pulsedb_fewshot.training import EpisodicDataset


def _metadata(store_root: Path) -> pd.DataFrame:
    waveforms = np.stack(
        [np.linspace(0.0, 1.0, 32, dtype=np.float32) + index for index in range(8)]
    )
    np.save(store_root / "waveforms.npy", waveforms)
    return pd.DataFrame(
        {
            "subject_uid": ["s1"] * 8,
            "event_id": [f"s1::{index:04d}" for index in range(1, 9)],
            "event_index": list(range(1, 9)),
            "common_query": [False] * 5 + [True] * 3,
            "sbp": [101.0 + index for index in range(8)],
            "dbp": [61.0 + index for index in range(8)],
            "n_samples": [32] * 8,
            "waveform_file": ["waveforms.npy"] * 8,
            "waveform_row": list(range(8)),
        }
    )


def test_fixed_first_support_never_uses_later_prior_events(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    scaler = {"mean": [100.0, 60.0], "std": [10.0, 5.0]}
    fixed = EpisodicDataset(
        metadata, tmp_path, scaler, ks=(2,), rolling_support=False
    )
    rolling = EpisodicDataset(
        metadata, tmp_path, scaler, ks=(2,), rolling_support=True
    )

    # Third query is event 8. Fixed support is events 1-2; rolling support is 6-7.
    fixed_episode = fixed[2]
    rolling_episode = rolling[2]
    np.testing.assert_allclose(
        fixed_episode["support_bp"][:2, 0].numpy(), [0.1, 0.2], atol=1e-6
    )
    np.testing.assert_allclose(
        rolling_episode["support_bp"][:2, 0].numpy(), [0.6, 0.7], atol=1e-6
    )
    assert fixed_episode["event_id"] == rolling_episode["event_id"] == "s1::0008"


def test_bp_change_scores_increase_when_query_moves_away_from_support(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    scaler = {"mean": [100.0, 60.0], "std": [10.0, 5.0]}
    dataset = EpisodicDataset(
        metadata, tmp_path, scaler, ks=(1,), rolling_support=False
    )
    scores = dataset.bp_change_scores()
    assert len(scores) == 3
    assert scores[0] < scores[1] < scores[2]


def test_demographics_are_joined_by_subject_not_event_order(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    scaler = {"mean": [100.0, 60.0], "std": [10.0, 5.0]}
    demographics = pd.DataFrame(
        {
            "subject_uid": ["s1"],
            "age_z": [1.5],
            "age_valid": [1.0],
            "sex_female": [0.0],
            "sex_male": [1.0],
            "sex_unknown": [0.0],
        }
    )
    dataset = EpisodicDataset(
        metadata,
        tmp_path,
        scaler,
        ks=(1,),
        rolling_support=False,
        demographics=demographics,
    )
    np.testing.assert_allclose(
        dataset[0]["demographics"].numpy(), [1.5, 1.0, 0.0, 1.0, 0.0]
    )
