"""Deterministic subject-level partition assignment and overlap checks."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _allocation_counts(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = np.asarray(fractions, dtype=float) * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for idx in order[:remainder]:
        counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def assign_subject_splits(
    subjects: pd.DataFrame,
    *,
    seed: int = 20260809,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> pd.DataFrame:
    """Assign subjects to meta-train, meta-validation, or locked meta-test.

    Assignment is stratified by ``source``. Additional BP-distribution
    stratification can be added after the real PulseDB cohort audit.
    """

    required = {"subject_id", "source"}
    missing = required.difference(subjects.columns)
    if missing:
        raise ValueError(f"Missing required subject columns: {sorted(missing)}")
    unique = subjects[["subject_id", "source"]].drop_duplicates()
    if unique["subject_id"].duplicated().any():
        raise ValueError("Each subject_id must map to exactly one source")
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("fractions must be three positive values summing to one")

    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    split_names = np.asarray(["meta_train", "meta_validation", "meta_test"])
    for _, group in unique.groupby("source", sort=True):
        group = group.sort_values("subject_id", kind="mergesort").reset_index(drop=True)
        permutation = rng.permutation(len(group))
        group = group.iloc[permutation].reset_index(drop=True)
        counts = _allocation_counts(len(group), fractions)
        labels = np.repeat(split_names, counts)
        group["split"] = labels
        pieces.append(group)

    result = pd.concat(pieces, ignore_index=True)
    result = result.sort_values("subject_id", kind="mergesort").reset_index(drop=True)
    result["split_seed"] = seed
    digest_input = "\n".join(
        f"{row.subject_id}\t{row.source}\t{row.split}" for row in result.itertuples()
    )
    result.attrs["sha256"] = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    assert_disjoint_subject_splits(result)
    return result


def assert_disjoint_subject_splits(splits: pd.DataFrame) -> None:
    """Raise when a participant appears in more than one partition."""

    required = {"subject_id", "split"}
    missing = required.difference(splits.columns)
    if missing:
        raise ValueError(f"Missing required split columns: {sorted(missing)}")
    counts = splits.groupby("subject_id")["split"].nunique()
    overlapping = counts[counts > 1]
    if not overlapping.empty:
        raise AssertionError(
            "Subject overlap detected across partitions: "
            + ", ".join(map(str, overlapping.index[:10]))
        )
