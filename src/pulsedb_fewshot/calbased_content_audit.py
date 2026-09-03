"""Input-only exact-content audit for the same-subject development benchmark.

The protocol intentionally permits the same participant to appear in fitting,
internal-validation, and held-out roles, but it must never treat byte-identical
PPG windows as independent observations.  This module reads only ``PPG_F`` and
the storage locators needed to find it.  It never accepts a held-out BP target
path and never inspects BP values.

The conservative policy is participant-level exclusion: if any selected PPG
content is duplicated within one split mode, every participant represented in
that duplicate group is excluded from *both* split modes.  Applying one common
participant set keeps random and chronological comparisons paired and avoids
using role-specific replacement rules that could indirectly tune the cohort.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

from .calbased_materialize import _content_sha256
from .calbased_protocol import ROLES, SPLIT_MODES, TARGET_COLUMNS
from .materialize import _read_ppg


INPUT_COLUMNS = (
    "split_mode",
    "role",
    "subject_uid",
    "source",
    "segment_uid",
    "raw_file",
    "ppg_storage_mode",
    "ppg_reference_index",
    "n_samples",
)


def _hash_raw_file(
    raw_file: str,
    locators: list[tuple[int, str, int, int]],
) -> list[tuple[int, str]]:
    """Hash unique PPG locators from one raw file in one file-open operation."""

    results: list[tuple[int, str]] = []
    with h5py.File(raw_file, "r") as h5file:
        for locator_id, storage_mode, reference_index, n_samples in locators:
            waveform = _read_ppg(h5file, storage_mode, reference_index)
            if waveform.size != n_samples:
                raise AssertionError(
                    "waveform length mismatch during input-only content audit: "
                    f"{raw_file}#{reference_index}: {waveform.size} != {n_samples}"
                )
            if not np.isfinite(waveform).all() or float(np.std(waveform)) <= 0:
                raise AssertionError(
                    "invalid waveform during input-only content audit: "
                    f"{raw_file}#{reference_index}"
                )
            results.append((locator_id, _content_sha256(waveform)))
    return results


def _read_input_projection(mode_inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if not mode_inputs or set(mode_inputs).difference(SPLIT_MODES):
        raise ValueError("mode_inputs must contain only supported split modes")

    projected: list[pd.DataFrame] = []
    for split_mode, frame in mode_inputs.items():
        missing = set(INPUT_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(
                f"{split_mode} input manifest is missing columns: {sorted(missing)}"
            )
        if TARGET_COLUMNS & set(INPUT_COLUMNS):
            raise AssertionError("content-audit projection unexpectedly contains targets")
        work = frame.loc[:, INPUT_COLUMNS].copy()
        if set(work["split_mode"].astype(str)) != {split_mode}:
            raise AssertionError(f"mixed split modes in {split_mode} content input")
        if set(work["role"].astype(str)) != set(ROLES):
            raise AssertionError(f"incomplete role coverage in {split_mode} content input")
        if work["segment_uid"].duplicated().any():
            raise AssertionError(f"duplicate segment_uid in {split_mode} content input")
        projected.append(work)

    content = pd.concat(projected, ignore_index=True)
    content["subject_uid"] = content["subject_uid"].astype(str)
    content["source"] = content["source"].astype(str)
    content["raw_file"] = content["raw_file"].astype(str)
    content["ppg_storage_mode"] = content["ppg_storage_mode"].astype(str)
    content["ppg_reference_index"] = content["ppg_reference_index"].astype(int)
    content["n_samples"] = content["n_samples"].astype(int)
    return content


def _hash_unique_locators(content: pd.DataFrame, *, workers: int) -> pd.DataFrame:
    if workers <= 0:
        raise ValueError("workers must be positive")
    locator_columns = [
        "raw_file",
        "ppg_storage_mode",
        "ppg_reference_index",
        "n_samples",
    ]
    locators = content.loc[:, locator_columns].drop_duplicates().reset_index(drop=True)
    locators["content_locator_id"] = np.arange(len(locators), dtype=np.int64)

    tasks: list[tuple[str, list[tuple[int, str, int, int]]]] = []
    for raw_file, group in locators.groupby("raw_file", sort=False):
        entries = [
            (
                int(row.content_locator_id),
                str(row.ppg_storage_mode),
                int(row.ppg_reference_index),
                int(row.n_samples),
            )
            for row in group.itertuples(index=False)
        ]
        tasks.append((str(raw_file), entries))

    hashes: dict[int, str] = {}
    if workers == 1:
        for raw_file, entries in tasks:
            hashes.update(_hash_raw_file(raw_file, entries))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_hash_raw_file, raw_file, entries): raw_file
                for raw_file, entries in tasks
            }
            for future in as_completed(futures):
                hashes.update(future.result())
    if len(hashes) != len(locators):
        raise AssertionError(
            f"content hash count mismatch: {len(hashes)} != {len(locators)}"
        )
    locators["ppg_content_sha256"] = locators["content_locator_id"].map(hashes)
    if locators["ppg_content_sha256"].isna().any():
        raise AssertionError("one or more PPG locators were not hashed")
    return locators


def _mode_duplicate_summary(frame: pd.DataFrame) -> tuple[dict[str, Any], set[str]]:
    roles = tuple(ROLES)
    role_hashes = {
        role: set(frame.loc[frame["role"].eq(role), "ppg_content_sha256"])
        for role in roles
    }
    cross_role: dict[str, int] = {}
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            cross_role[f"{left}__{right}"] = len(role_hashes[left] & role_hashes[right])
    within_role = {
        role: int(
            frame.loc[frame["role"].eq(role), "ppg_content_sha256"]
            .duplicated()
            .sum()
        )
        for role in roles
    }

    duplicate_mask = frame["ppg_content_sha256"].duplicated(keep=False)
    duplicates = frame.loc[duplicate_mask].copy()
    affected_subjects = set(duplicates["subject_uid"].astype(str))
    group_sizes = duplicates.groupby("ppg_content_sha256").size()
    source_counts = (
        duplicates.loc[:, ["source", "subject_uid"]]
        .drop_duplicates()
        .groupby("source")
        .size()
        .astype(int)
        .to_dict()
    )
    summary = {
        "selected_rows": int(len(frame)),
        "unique_content_hashes": int(frame["ppg_content_sha256"].nunique()),
        "duplicate_hash_groups": int(len(group_sizes)),
        "duplicate_rows": int(len(duplicates)),
        "duplicate_rows_beyond_first": int((group_sizes - 1).sum()),
        "affected_subjects": int(len(affected_subjects)),
        "affected_subjects_by_source": source_counts,
        "cross_role_overlap_counts": cross_role,
        "within_role_duplicate_counts": within_role,
        "status": "pass" if duplicates.empty else "requires_subject_exclusion",
    }
    return summary, affected_subjects


def audit_calbased_candidate_content(
    mode_inputs: Mapping[str, pd.DataFrame],
    *,
    workers: int = 4,
) -> tuple[dict[str, Any], set[str]]:
    """Return an input-only audit and common participant exclusion set.

    Duplicate detection is performed independently inside each split mode.
    Identical windows selected by both experimental modes are expected and are
    not treated as leakage between otherwise independent experiments.
    """

    content = _read_input_projection(mode_inputs)
    locators = _hash_unique_locators(content, workers=workers)
    locator_columns = [
        "raw_file",
        "ppg_storage_mode",
        "ppg_reference_index",
        "n_samples",
    ]
    content = content.merge(
        locators.loc[:, locator_columns + ["ppg_content_sha256"]],
        on=locator_columns,
        how="left",
        validate="many_to_one",
    )
    if content["ppg_content_sha256"].isna().any():
        raise AssertionError("content hashes did not map back to all selected rows")

    excluded: set[str] = set()
    modes: dict[str, Any] = {}
    for split_mode, frame in content.groupby("split_mode", sort=True):
        summary, affected = _mode_duplicate_summary(frame)
        modes[str(split_mode)] = summary
        excluded.update(affected)

    subject_set_sha256 = hashlib.sha256(
        "\n".join(sorted(excluded)).encode("utf-8")
    ).hexdigest()
    audit = {
        "status": "pass" if not excluded else "exclude_then_rebuild",
        "method": "SHA-256 of canonical little-endian float32 PPG_F sample bytes",
        "policy": "exclude every participant represented in an exact-content duplicate group from all split modes",
        "split_modes_compared_independently": True,
        "input_projection": list(INPUT_COLUMNS),
        "bp_target_columns_loaded": False,
        "heldout_test_targets_accessed": False,
        "selected_rows": int(len(content)),
        "unique_storage_locators_read": int(len(locators)),
        "excluded_subject_count": int(len(excluded)),
        "excluded_subject_set_sha256": subject_set_sha256,
        "modes": modes,
    }
    return audit, excluded

