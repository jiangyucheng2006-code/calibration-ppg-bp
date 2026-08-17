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
