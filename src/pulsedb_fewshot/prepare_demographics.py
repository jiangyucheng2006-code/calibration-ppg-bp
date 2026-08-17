"""Prepare leakage-safe participant age/sex features for development experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_demographics import _normalize_gender


def _mode_or_unknown(values: pd.Series) -> str:
    usable = values.loc[values.isin(["F", "M"])]
    if usable.empty:
        return "UNKNOWN"
    counts = usable.value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "UNKNOWN"
    return str(counts.index[0])


def prepare(input_path: Path, output_path: Path) -> dict[str, object]:
    segments = pd.read_parquet(
        input_path, columns=["subject_uid", "split", "source", "age", "gender"]
    )
    if segments["split"].eq("meta_test").any():
        raise AssertionError("development demographic preparation must not contain meta-test rows")
    segments["age"] = pd.to_numeric(segments["age"], errors="coerce")
    segments["age_usable"] = segments["age"].where(segments["age"].between(18, 100))
    segments["gender_normalized"] = segments["gender"].map(_normalize_gender)

    participant = segments.groupby("subject_uid", as_index=False).agg(
        split=("split", "first"),
        source=("source", "first"),
        age_clean=("age_usable", "median"),
        age_nunique_valid=("age_usable", lambda values: values.dropna().nunique()),
        sex=("gender_normalized", _mode_or_unknown),
    )
    participant["age_valid"] = participant["age_clean"].notna().astype(np.float32)

    training_ages = participant.loc[
        participant["split"].eq("meta_train") & participant["age_valid"].eq(1),
        "age_clean",
    ]
    if training_ages.empty:
        raise ValueError("meta-train contains no valid adult ages")
    age_mean = float(training_ages.mean())
    age_std = float(training_ages.std(ddof=0))
    if not np.isfinite(age_std) or age_std <= 1e-8:
        raise ValueError("invalid meta-train age standard deviation")
    participant["age_z"] = (
        (participant["age_clean"] - age_mean) / age_std
    ).where(participant["age_valid"].eq(1), 0.0)
    participant["sex_female"] = participant["sex"].eq("F").astype(np.float32)
    participant["sex_male"] = participant["sex"].eq("M").astype(np.float32)
    participant["sex_unknown"] = participant["sex"].eq("UNKNOWN").astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    participant.to_parquet(output_path, index=False)
    summary = {
        "analysis": "development-only participant demographic preparation",
        "locked_meta_test_accessed": False,
        "participants": int(participant["subject_uid"].nunique()),
        "age_rule": "participant median of finite ages in [18, 100] years",
        "invalid_age_encoding": "age_z=0 with age_valid=0",
        "sex_rule": "participant mode of F/M; ties or unavailable values are UNKNOWN",
        "age_scaler_fit_split": "meta_train only",
        "age_mean_meta_train": age_mean,
        "age_std_meta_train": age_std,
        "age_invalid_participants": int(participant["age_valid"].eq(0).sum()),
        "age_inconsistent_valid_participants": int(
            participant["age_nunique_valid"].gt(1).sum()
        ),
        "sex_unknown_participants": int(participant["sex"].eq("UNKNOWN").sum()),
        "output": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
