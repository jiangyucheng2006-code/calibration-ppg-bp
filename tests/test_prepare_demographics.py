from pathlib import Path

import pandas as pd

from pulsedb_fewshot.prepare_demographics import prepare


def test_demographic_preparation_uses_train_only_age_scaler_and_masks_invalid(
    tmp_path: Path,
) -> None:
    source = tmp_path / "segments.parquet"
    output = tmp_path / "demographics.parquet"
    pd.DataFrame(
        {
            "subject_uid": ["train_a", "train_a", "train_b", "train_c", "val_a"],
            "split": [
                "meta_train",
                "meta_train",
                "meta_train",
                "meta_train",
                "meta_validation",
            ],
            "source": ["A", "A", "A", "A", "B"],
            "age": [40.0, 42.0, 0.0, 61.0, 99.0],
            "gender": ["M", "M", "F", "F", "F"],
        }
    ).to_parquet(source, index=False)
    summary = prepare(source, output)
    result = pd.read_parquet(output).set_index("subject_uid")
    assert result.loc["train_a", "age_clean"] == 41.0
    assert result.loc["train_b", "age_valid"] == 0.0
    assert result.loc["train_b", "age_z"] == 0.0
    assert result.loc["val_a", "age_z"] != 0.0
    assert summary["age_mean_meta_train"] == 51.0
    assert summary["age_invalid_participants"] == 1
