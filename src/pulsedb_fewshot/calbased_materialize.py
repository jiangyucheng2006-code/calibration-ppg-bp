"""Materialize PPG for the development-only CalBased analogue protocol.

The store layout is::

    <store>/<split_mode>/train_metadata_*.parquet
    <store>/<split_mode>/train_signals_*.npy
    <store>/<split_mode>/internal_validation_metadata_*.parquet
    <store>/<split_mode>/internal_validation_signals_*.npy
    <store>/<split_mode>/heldout_test_metadata_*.parquet
    <store>/<split_mode>/heldout_test_signals_*.npy

Training/screening code should load only the ``train`` and
``internal_validation`` prefixes.  The held-out metadata has no BP target
columns, and this entry point has no argument for the separately quarantined
held-out target file.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .calbased_protocol import PROTOCOL_ID, SPLIT_MODES, TARGET_COLUMNS
from .materialize import _read_ppg, _sha256, _stable_shard


REQUIRED_COLUMNS = {
    "protocol_id",
    "split_mode",
    "role",
    "subject_uid",
    "source",
    "segment_uid",
    "raw_file",
    "ppg_field",
    "ppg_storage_mode",
    "ppg_reference_index",
    "n_samples",
}
DEVELOPMENT_ROLES = {"train", "internal_validation"}


def _content_sha256(waveform: np.ndarray) -> str:
    canonical = np.asarray(waveform, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _validate_inputs(
    development: pd.DataFrame,
    heldout: pd.DataFrame,
) -> str:
    for name, frame in (("development", development), ("heldout", heldout)):
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} manifest is missing columns: {sorted(missing)}")
        if frame["segment_uid"].duplicated().any():
            raise AssertionError(f"{name} segment_uid values are not unique")
        if set(frame["protocol_id"].astype(str)) != {PROTOCOL_ID}:
            raise AssertionError(f"{name} manifest protocol_id is not {PROTOCOL_ID}")
        if set(frame["ppg_field"].astype(str)) != {"PPG_F"}:
            raise AssertionError(f"{name} manifest is not PPG_F-only")
        if "original_split" in frame and set(frame["original_split"].astype(str)) != {
            "meta_train"
        }:
            raise AssertionError(f"{name} manifest contains a protected subject split")

    if set(development["role"].astype(str)) != DEVELOPMENT_ROLES:
        raise AssertionError("development manifest must contain train and internal_validation")
    if set(heldout["role"].astype(str)) != {"heldout_test"}:
        raise AssertionError("heldout manifest role must be heldout_test")
    if TARGET_COLUMNS & set(heldout.columns):
        raise AssertionError("heldout model-input metadata exposes BP targets")
    if not {"sbp", "dbp"}.issubset(development.columns):
        raise ValueError("development manifest is missing supervised BP targets")

    split_modes = set(development["split_mode"].astype(str)) | set(
        heldout["split_mode"].astype(str)
    )
    if len(split_modes) != 1 or not split_modes.issubset(SPLIT_MODES):
        raise AssertionError(f"manifests contain invalid/mixed split modes: {split_modes}")
    split_mode = split_modes.pop()

    development_ids = set(development["segment_uid"].astype(str))
    heldout_ids = set(heldout["segment_uid"].astype(str))
    if development_ids & heldout_ids:
        raise AssertionError("development and heldout manifests share segment_uid values")
    subject_sets = {
        "train": set(
            development.loc[
                development["role"].eq("train"), "subject_uid"
            ].astype(str)
        ),
        "internal_validation": set(
            development.loc[
                development["role"].eq("internal_validation"), "subject_uid"
            ].astype(str)
        ),
        "heldout_test": set(heldout["subject_uid"].astype(str)),
    }
    if len({frozenset(subjects) for subjects in subject_sets.values()}) != 1:
        raise AssertionError("same-subject protocol does not have identical subject coverage")
    return split_mode


def _write_role_plans(
    frame: pd.DataFrame,
    store_dir: Path,
    *,
    role: str,
    n_shards: int,
) -> list[tuple[str, str]]:
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    work = frame.copy()
    work["waveform_shard"] = work["subject_uid"].map(
        lambda subject_uid: _stable_shard(str(subject_uid), n_shards)
    )
    tasks: list[tuple[str, str]] = []
    for shard, group in work.groupby("waveform_shard", sort=True):
        sort_columns = [
            column
            for column in ("subject_uid", "selection_rank", "segment_uid")
            if column in group.columns
        ]
        group = group.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
        group["waveform_row"] = np.arange(len(group), dtype=np.int64)
        signal_name = f"{role}_signals_{int(shard):03d}.npy"
        # WaveformAccessor receives output_root, so persist a path relative to
        # that root while keeping the required split-mode subdirectory.
        group["waveform_file"] = f"{store_dir.name}/{signal_name}"
        plan_path = store_dir / f"{role}_metadata_{int(shard):03d}.parquet"
        signal_path = store_dir / signal_name
        group.to_parquet(plan_path, index=False)
        tasks.append((str(plan_path), str(signal_path)))
    return tasks


def _materialize_one_role_shard(plan_path: str, signal_path: str) -> dict[str, Any]:
    plan = Path(plan_path)
    signal = Path(signal_path)
    frame = pd.read_parquet(plan)
    if frame.empty:
        raise ValueError(f"empty materialization plan: {plan}")
    sample_counts = set(frame["n_samples"].astype(int))
    if len(sample_counts) != 1:
        raise ValueError(f"mixed waveform lengths in {plan}: {sample_counts}")
    n_samples = sample_counts.pop()
    signals = np.lib.format.open_memmap(
        signal, mode="w+", dtype=np.float32, shape=(len(frame), n_samples)
    )
    content_hashes = [""] * len(frame)
    for raw_file, group in frame.groupby("raw_file", sort=False):
        with h5py.File(raw_file, "r") as h5file:
            for row in group.itertuples(index=False):
                waveform = _read_ppg(
                    h5file,
                    str(row.ppg_storage_mode),
                    int(row.ppg_reference_index),
                )
                if waveform.size != n_samples:
                    raise AssertionError(
                        f"waveform length mismatch for {row.segment_uid}: {waveform.size}"
                    )
                if not np.isfinite(waveform).all() or float(np.std(waveform)) <= 0:
                    raise AssertionError(f"invalid waveform for {row.segment_uid}")
                waveform_row = int(row.waveform_row)
                signals[waveform_row] = waveform
                content_hashes[waveform_row] = _content_sha256(waveform)
    signals.flush()
    del signals
    frame["ppg_content_sha256"] = content_hashes
    temporary = plan.with_suffix(plan.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(plan)
    return {
        "role": str(frame["role"].iloc[0]),
        "metadata": str(plan),
        "signals": str(signal),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_uid"].nunique()),
        "n_samples": int(n_samples),
        "metadata_sha256": _sha256(plan),
        "signals_sha256": _sha256(signal),
        "signals_size": int(signal.stat().st_size),
    }


def _exact_content_overlap_audit(metadata_paths: list[Path]) -> dict[str, Any]:
    frames = [
        pd.read_parquet(path, columns=["role", "segment_uid", "ppg_content_sha256"])
        for path in metadata_paths
    ]
    content = pd.concat(frames, ignore_index=True)
    sets = {
        role: set(content.loc[content["role"].eq(role), "ppg_content_sha256"])
        for role in ("train", "internal_validation", "heldout_test")
    }
    pairs: dict[str, int] = {}
    roles = ("train", "internal_validation", "heldout_test")
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            pairs[f"{left}__{right}"] = len(sets[left] & sets[right])
    within_role_duplicates = {
        role: int(
            content.loc[content["role"].eq(role), "ppg_content_sha256"].duplicated().sum()
        )
        for role in roles
    }
    globally_unique = not any(pairs.values()) and not any(
        within_role_duplicates.values()
    )
    return {
        "method": "SHA-256 of canonical little-endian float32 PPG_F sample bytes",
        "cross_role_overlap_counts": pairs,
        "within_role_duplicate_counts": within_role_duplicates,
        "global_exact_content_unique": globally_unique,
        "status": "pass" if globally_unique else "fail",
    }


def materialize_calbased_ppg(
    development_fit_path: Path,
    heldout_inputs_path: Path,
    output_root: Path,
    *,
    train_shards: int = 32,
    validation_shards: int = 8,
    heldout_shards: int = 8,
    workers: int = 4,
) -> dict[str, Any]:
    """Materialize all roles without loading held-out BP targets."""

    development = pd.read_parquet(development_fit_path)
    heldout = pd.read_parquet(heldout_inputs_path)
    split_mode = _validate_inputs(development, heldout)
    report_path = output_root / "materialization.json"
    existing_report: dict[str, Any] | None = None
    if output_root.exists() and any(output_root.iterdir()):
        if not report_path.is_file():
            raise FileExistsError(
                f"non-empty store has no compatible materialization report: {output_root}"
            )
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        if existing_report.get("protocol_id") != PROTOCOL_ID:
            raise FileExistsError("existing store belongs to a different protocol")
        if split_mode in existing_report.get("split_modes_materialized", []):
            raise FileExistsError(f"split mode is already materialized: {split_mode}")
    store_dir = output_root / split_mode
    if store_dir.exists() and any(store_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite split-mode store: {store_dir}")
    store_dir.mkdir(parents=True, exist_ok=True)

    role_frames = {
        "train": development.loc[development["role"].eq("train")].copy(),
        "internal_validation": development.loc[
            development["role"].eq("internal_validation")
        ].copy(),
        "heldout_test": heldout.copy(),
    }
    shard_counts = {
        "train": train_shards,
        "internal_validation": validation_shards,
        "heldout_test": heldout_shards,
    }
    tasks: list[tuple[str, str]] = []
    for role, frame in role_frames.items():
        tasks.extend(
            _write_role_plans(
                frame,
                store_dir,
                role=role,
                n_shards=shard_counts[role],
            )
        )

    results: list[dict[str, Any]] = []
    if workers == 1:
        for plan, signal in tasks:
            results.append(_materialize_one_role_shard(plan, signal))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_materialize_one_role_shard, plan, signal): (plan, signal)
                for plan, signal in tasks
            }
            for future in as_completed(futures):
                results.append(future.result())

    expected_rows = len(development) + len(heldout)
    materialized_rows = sum(int(result["rows"]) for result in results)
    if materialized_rows != expected_rows:
        raise AssertionError(
            f"materialized row total mismatch: {materialized_rows} != {expected_rows}"
        )
    metadata_paths = [Path(result["metadata"]) for result in results]
    content_audit = _exact_content_overlap_audit(metadata_paths)
    if content_audit["status"] != "pass":
        raise AssertionError(
            "exact PPG content is duplicated within the accepted protocol: "
            + json.dumps(
                {
                    "cross_role": content_audit["cross_role_overlap_counts"],
                    "within_role": content_audit["within_role_duplicate_counts"],
                },
                sort_keys=True,
            )
        )

    per_subject_windows: dict[str, int] = {}
    for role, frame in role_frames.items():
        counts = frame.groupby("subject_uid").size()
        if counts.nunique() != 1:
            raise AssertionError(f"{role} window counts differ across participants")
        per_subject_windows[role] = int(counts.iloc[0])
    subject_count = int(development["subject_uid"].nunique())
    subject_ids = sorted(development["subject_uid"].astype(str).unique())
    subject_set_sha256 = hashlib.sha256(
        "\n".join(subject_ids).encode("utf-8")
    ).hexdigest()
    mode_report = {
        "split_mode": split_mode,
        "store_dir": str(store_dir),
        "development_fit_input": str(development_fit_path.resolve()),
        "heldout_test_input": str(heldout_inputs_path.resolve()),
        "rows_by_role": {
            role: int(len(frame)) for role, frame in role_frames.items()
        },
        "subjects_by_role": {
            role: int(frame["subject_uid"].nunique())
            for role, frame in role_frames.items()
        },
        "materialized_rows": int(materialized_rows),
        "exact_ppg_content_overlap_audit": content_audit,
        "shards": sorted(results, key=lambda result: str(result["metadata"])),
    }
    if existing_report is None:
        materializations: dict[str, Any] = {}
    else:
        if int(existing_report.get("subject_count", -1)) != subject_count:
            raise AssertionError("split modes do not contain the same participant count")
        if existing_report.get("subject_set_sha256") != subject_set_sha256:
            raise AssertionError("split modes do not contain the same participant set")
        if existing_report.get("windows_per_subject") != per_subject_windows:
            raise AssertionError("split modes have different per-subject role counts")
        materializations = dict(existing_report.get("materializations", {}))
    materializations[split_mode] = mode_report

    report = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "split_modes_materialized": sorted(materializations),
        "materializations": materializations,
        "source_subject_split": "frozen meta_train only",
        "source_parent_split": "meta_train",
        "source_parent_splits": ["meta_train"],
        "subject_count": subject_count,
        "subject_set_sha256": subject_set_sha256,
        "windows_per_subject": per_subject_windows,
        "meta_validation_windows_accessed": False,
        "locked_meta_test_windows_accessed": False,
        "heldout_test_targets_accessed": False,
        "heldout_test_targets_path_accepted_by_entrypoint": False,
        "screen_loader_roles": ["train", "internal_validation"],
        "screen_loader_includes_heldout_test": False,
        "model_selection_role": "internal_validation",
        "dtype": "float32",
        "normalization": "none in store; training loader may apply per-window z-score",
        "materialized_rows": int(
            sum(item["materialized_rows"] for item in materializations.values())
        ),
        "exact_ppg_content_overlap_audits": {
            mode: item["exact_ppg_content_overlap_audit"]
            for mode, item in materializations.items()
        },
        "shards": [
            shard
            for mode in sorted(materializations)
            for shard in materializations[mode]["shards"]
        ],
    }
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_report.replace(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-fit", required=True, type=Path)
    parser.add_argument("--heldout-inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-shards", type=int, default=32)
    parser.add_argument("--validation-shards", type=int, default=8)
    parser.add_argument("--heldout-shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = materialize_calbased_ppg(
        args.development_fit,
        args.heldout_inputs,
        args.output,
        train_shards=args.train_shards,
        validation_shards=args.validation_shards,
        heldout_shards=args.heldout_shards,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
