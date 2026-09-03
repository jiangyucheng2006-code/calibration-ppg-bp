"""Audit development-only age/sex fields before demographic conditioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_gender(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"M", "MALE"}:
        return "M"
    if text in {"F", "FEMALE"}:
        return "F"
    if text in {"", "NAN", "NONE", "UNKNOWN", "U"}:
        return "UNKNOWN"
    return "OTHER"


def audit(input_path: Path, output_dir: Path) -> dict[str, object]:
    segments = pd.read_parquet(input_path)
    required = {"subject_uid", "split", "source", "age", "gender"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"development segment table missing columns: {sorted(missing)}")
    if segments["split"].eq("meta_test").any():
        raise AssertionError("development demographic audit must not contain meta-test rows")

    frame = segments[list(required)].copy()
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["gender_normalized"] = frame["gender"].map(_normalize_gender)
    consistency = frame.groupby("subject_uid", as_index=False).agg(
        split=("split", "first"),
        source=("source", "first"),
        age=("age", "first"),
        age_nunique=("age", lambda values: values.dropna().nunique()),
        gender=("gender_normalized", "first"),
        gender_nunique=("gender_normalized", "nunique"),
        segment_rows=("subject_uid", "size"),
    )
    consistency["age_missing"] = consistency["age"].isna()
    consistency["age_implausible"] = consistency["age"].notna() & (
        consistency["age"].lt(0) | consistency["age"].gt(120)
    )
    consistency["age_under_18"] = consistency["age"].notna() & consistency["age"].lt(18)
    consistency["age_inconsistent"] = consistency["age_nunique"].gt(1)
    consistency["gender_missing"] = consistency["gender"].eq("UNKNOWN")
    consistency["gender_inconsistent"] = consistency["gender_nunique"].gt(1)

    by_split_source = (
        consistency.groupby(["split", "source"], as_index=False)
        .agg(
            participants=("subject_uid", "nunique"),
            age_missing_rate=("age_missing", "mean"),
            age_mean=("age", "mean"),
            age_std=("age", "std"),
            gender_missing_rate=("gender_missing", "mean"),
            female_rate=("gender", lambda values: float(values.eq("F").mean())),
            male_rate=("gender", lambda values: float(values.eq("M").mean())),
        )
        .sort_values(["split", "source"])
    )
    gender_counts = (
        consistency.groupby(["split", "source", "gender"], as_index=False)
        .agg(participants=("subject_uid", "nunique"))
        .sort_values(["split", "source", "gender"])
    )

    summary = {
        "analysis": "development-only demographic audit",
        "input": str(input_path),
        "locked_meta_test_accessed": False,
        "participants": int(consistency["subject_uid"].nunique()),
        "splits": {
            str(key): int(value)
            for key, value in consistency.groupby("split")["subject_uid"].nunique().items()
        },
        "age_missing_rate": float(consistency["age_missing"].mean()),
        "age_implausible_count": int(consistency["age_implausible"].sum()),
        "age_under_18_count": int(consistency["age_under_18"].sum()),
        "age_inconsistent_count": int(consistency["age_inconsistent"].sum()),
        "age_min": float(consistency["age"].min()) if consistency["age"].notna().any() else None,
        "age_max": float(consistency["age"].max()) if consistency["age"].notna().any() else None,
        "gender_missing_rate": float(consistency["gender_missing"].mean()),
        "gender_inconsistent_count": int(consistency["gender_inconsistent"].sum()),
        "gender_counts": {
            str(key): int(value) for key, value in consistency["gender"].value_counts().items()
        },
        "conditioning_ready": bool(
            consistency["age_inconsistent"].sum() == 0
            and consistency["gender_inconsistent"].sum() == 0
            and consistency["age_implausible"].sum() == 0
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    consistency.to_parquet(output_dir / "participant_demographics.parquet", index=False)
    by_split_source.to_csv(output_dir / "demographics_by_split_source.csv", index=False)
    gender_counts.to_csv(output_dir / "gender_counts.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
