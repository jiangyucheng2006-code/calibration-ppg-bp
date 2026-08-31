"""Prepare both development CalBased analogue protocols and PPG stores.

This is the fail-closed, end-to-end data entry point.  It predicate-loads only
the frozen ``meta_train`` rows, builds both window-assignment controls, writes
their development/held-out manifests separately, and materializes ``PPG_F``
without ever accepting a meta-validation, locked-meta-test, or held-out-target
input for model screening.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .calbased_content_audit import audit_calbased_candidate_content
from .calbased_materialize import materialize_calbased_ppg
from .calbased_protocol import (
    DEFAULT_SEED,
    EXPECTED_META_TRAIN_SUBJECTS,
    PROTOCOL_ID,
    SPLIT_MODES,
    build_calbased_analogue,
    load_frozen_meta_train_segments,
    write_calbased_analogue,
)


def prepare_calbased_data(
    segment_index_path: Path,
    subject_splits_path: Path,
    protocol_root: Path,
    store_root: Path,
    *,
    split_modes: tuple[str, ...] = SPLIT_MODES,
    seed: int = DEFAULT_SEED,
    expected_subjects: int = EXPECTED_META_TRAIN_SUBJECTS,
    require_include_flag: bool = False,
    train_shards: int = 32,
    validation_shards: int = 8,
    heldout_shards: int = 8,
    workers: int = 4,
) -> dict[str, Any]:
    """Build, input-audit, and materialize both development tracks."""

    if not split_modes or len(set(split_modes)) != len(split_modes):
        raise ValueError("split_modes must be nonempty and unique")
    invalid_modes = set(split_modes).difference(SPLIT_MODES)
    if invalid_modes:
        raise ValueError(f"unsupported split modes: {sorted(invalid_modes)}")
    if protocol_root.exists() and any(protocol_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite protocol root: {protocol_root}")
    if store_root.exists() and any(store_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite PPG store root: {store_root}")

    segments, subject_splits, loading_filter = load_frozen_meta_train_segments(
        segment_index_path, subject_splits_path
    )
    initial_artifacts: dict[str, Any] = {}
    for split_mode in split_modes:
        artifacts = build_calbased_analogue(
            segments,
            subject_splits,
            split_mode=split_mode,
            seed=seed,
            require_include_flag=require_include_flag,
        )
        observed_subjects = int(artifacts.audit["eligible_subjects"])
        artifacts.audit["expected_meta_train_subjects"] = expected_subjects
        artifacts.audit["input_loading_filter"] = loading_filter
        artifacts.audit["segment_index_path"] = str(segment_index_path.resolve())
        artifacts.audit["subject_splits_path"] = str(subject_splits_path.resolve())
        if observed_subjects != expected_subjects:
            raise AssertionError(
                "eligible same-subject cohort does not match the frozen expectation: "
                f"{observed_subjects} != {expected_subjects}"
            )

        initial_artifacts[split_mode] = artifacts

    content_inputs = {
        split_mode: pd.concat(
            [
                artifacts.development_fit_manifest.drop(
                    columns=["sbp", "dbp"], errors="ignore"
                ),
                artifacts.heldout_test_inputs,
            ],
            ignore_index=True,
        )
        for split_mode, artifacts in initial_artifacts.items()
    }
    content_audit, excluded_subjects = audit_calbased_candidate_content(
        content_inputs,
        workers=workers,
    )
    retained_segments = segments.loc[
        ~segments["subject_uid"].astype(str).isin(excluded_subjects)
    ].copy()
    retained_subjects = expected_subjects - len(excluded_subjects)
    if retained_subjects <= 0:
        raise AssertionError("exact-content exclusion removed the entire cohort")

    mode_records: dict[str, Any] = {}
    final_materialization: dict[str, Any] | None = None
    for split_mode in split_modes:
        artifacts = build_calbased_analogue(
            retained_segments,
            subject_splits,
            split_mode=split_mode,
            seed=seed,
            require_include_flag=require_include_flag,
        )
        observed_subjects = int(artifacts.audit["eligible_subjects"])
        if observed_subjects != retained_subjects:
            raise AssertionError(
                "retained same-subject cohort does not match the content-audit expectation: "
                f"{observed_subjects} != {retained_subjects}"
            )
        artifacts.audit["expected_meta_train_subjects_before_content_audit"] = (
            expected_subjects
        )
        artifacts.audit["input_only_exact_content_excluded_subjects"] = len(
            excluded_subjects
        )
        artifacts.audit["retained_subjects_after_content_audit"] = retained_subjects
        artifacts.audit["content_exclusion_subject_set_sha256"] = content_audit[
            "excluded_subject_set_sha256"
        ]
        artifacts.audit["input_loading_filter"] = loading_filter
        artifacts.audit["segment_index_path"] = str(segment_index_path.resolve())
        artifacts.audit["subject_splits_path"] = str(subject_splits_path.resolve())

        mode_protocol_root = protocol_root / split_mode
        manifest_paths = write_calbased_analogue(artifacts, mode_protocol_root)
        final_materialization = materialize_calbased_ppg(
            Path(manifest_paths["development_fit_manifest"]),
            Path(manifest_paths["heldout_test_inputs"]),
            store_root,
            train_shards=train_shards,
            validation_shards=validation_shards,
            heldout_shards=heldout_shards,
            workers=workers,
        )
        mode_records[split_mode] = {
            "protocol_audit": artifacts.audit,
            "protocol_artifacts": manifest_paths,
            "store_dir": str((store_root / split_mode).resolve()),
        }

    if final_materialization is None:
        raise AssertionError("no split mode was prepared")
    report = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "split_modes": list(split_modes),
        "seed": seed,
        "expected_subjects_before_content_audit": expected_subjects,
        "retained_subjects": retained_subjects,
        "input_only_exact_content_audit": content_audit,
        "input_only_exact_content_excluded_subjects": len(excluded_subjects),
        "source_parent_split": "meta_train",
        "source_parent_splits": ["meta_train"],
        "input_loading_filter": loading_filter,
        "meta_validation_windows_accessed": False,
        "locked_meta_test_windows_accessed": False,
        "heldout_test_targets_accessed_by_materializer": False,
        "screen_loader_roles": ["train", "internal_validation"],
        "protocol_root": str(protocol_root.resolve()),
        "store_root": str(store_root.resolve()),
        "modes": mode_records,
        "materialization": final_materialization,
    }
    protocol_root.mkdir(parents=True, exist_ok=True)
    private_root = protocol_root / "private"
    private_root.mkdir(parents=True, exist_ok=True)
    excluded_frame = pd.DataFrame(
        {
            "subject_uid": sorted(excluded_subjects),
            "reason": [
                "selected exact PPG_F content duplicated within at least one split mode"
            ]
            * len(excluded_subjects),
        }
    )
    excluded_frame.to_parquet(
        private_root / "input_only_content_excluded_subjects.parquet", index=False
    )
    report_path = protocol_root / "data_preparation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-index", required=True, type=Path)
    parser.add_argument("--subject-splits", required=True, type=Path)
    parser.add_argument("--protocol-output", required=True, type=Path)
    parser.add_argument("--store-output", required=True, type=Path)
    parser.add_argument(
        "--split-mode", action="append", choices=SPLIT_MODES, dest="split_modes"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-subjects", type=int, default=EXPECTED_META_TRAIN_SUBJECTS
    )
    parser.add_argument("--require-include-flag", action="store_true")
    parser.add_argument("--train-shards", type=int, default=32)
    parser.add_argument("--validation-shards", type=int, default=8)
    parser.add_argument("--heldout-shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare_calbased_data(
        args.segment_index,
        args.subject_splits,
        args.protocol_output,
        args.store_output,
        split_modes=tuple(args.split_modes or SPLIT_MODES),
        seed=args.seed,
        expected_subjects=args.expected_subjects,
        require_include_flag=args.require_include_flag,
        train_shards=args.train_shards,
        validation_shards=args.validation_shards,
        heldout_shards=args.heldout_shards,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
