"""Batch audit for a source-balanced PulseDB v2 schema pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .schema_audit import audit_pulsedb_file


SOURCE_DIRECTORIES = ("PulseDB_MIMIC", "PulseDB_Vital")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_download_manifest(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    table = pd.read_csv(path, dtype=str).fillna("")
    required = {"source", "file_name", "sha256"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Download manifest is missing columns: {missing}")

    records: dict[tuple[str, str], dict[str, str]] = {}
    for row in table.to_dict(orient="records"):
        key = (str(row["source"]), str(row["file_name"]))
        if key in records:
            raise ValueError(f"Duplicate download-manifest key: {key}")
        records[key] = {str(key): str(value) for key, value in row.items()}
    return records


def audit_pulsedb_pilot(
    data_root: Path,
    output_dir: Path,
    *,
    download_manifest: Path,
    expected_per_source: int = 5,
) -> dict[str, Any]:
    """Audit all selected pilot files and build one traceable combined index."""

    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = _load_download_manifest(download_manifest.resolve())

    selected_files: list[tuple[str, Path]] = []
    for source_directory in SOURCE_DIRECTORIES:
        source_root = data_root / source_directory
        if source_root.exists():
            selected_files.extend(
                (source_directory, path)
                for path in sorted(source_root.glob("*.mat"), key=lambda item: item.name)
            )

    source_counts = Counter(source for source, _ in selected_files)
    required_failures: list[str] = []
    warnings: list[str] = []

    if expected_per_source > 0:
        for source_directory in SOURCE_DIRECTORIES:
            if source_counts[source_directory] != expected_per_source:
                required_failures.append(
                    f"expected_{expected_per_source}_{source_directory}_files"
                )

    selected_keys = {(source, path.name) for source, path in selected_files}
    manifest_keys = set(manifest_records)
    if selected_keys != manifest_keys:
        required_failures.append("download_manifest_exactly_matches_selected_files")

    summaries: list[dict[str, Any]] = []
    combined_indexes: list[pd.DataFrame] = []
    file_reports: list[dict[str, Any]] = []

    for source_directory, input_path in selected_files:
        key = (source_directory, input_path.name)
        manifest_row = manifest_records.get(key, {})
        expected_sha256 = manifest_row.get("sha256") or None
        per_file_output = output_dir / "per_file" / source_directory / input_path.stem
        report = audit_pulsedb_file(
            input_path,
            per_file_output,
            expected_sha256=expected_sha256,
            source="auto",
            project_fields_only=True,
        )
        file_reports.append(report)

        if report["status"] == "fail":
            required_failures.append(f"single_file_audit_failed:{source_directory}/{input_path.name}")
        if len(report["identities"]["subject_ids"]) != 1:
            required_failures.append(
                f"single_subject_mapping_failed:{source_directory}/{input_path.name}"
            )

        segment_index = pd.read_parquet(report["artifacts"]["segment_index_parquet"])
        segment_index.insert(0, "dataset_source_directory", source_directory)
        segment_index.insert(1, "dataset_source", report["input"]["source"])
        segment_index.insert(2, "raw_file_name", input_path.name)
        segment_index.insert(3, "raw_file_relative_path", str(input_path.relative_to(data_root)))
        segment_index.insert(4, "raw_file_sha256", report["input"]["sha256"])
        segment_index.insert(
            5,
            "subject_uid",
            report["input"]["source"] + ":" + segment_index["subject_id"].astype(str),
        )
        combined_indexes.append(segment_index)

        summary: dict[str, Any] = {
            "source_directory": source_directory,
            "source": report["input"]["source"],
            "file_name": input_path.name,
            "size_bytes": report["input"]["size_bytes"],
            "sha256": report["input"]["sha256"],
            "status": report["status"],
            "n_windows": report["n_windows"],
            "subject_ids": ",".join(report["identities"]["subject_ids"]),
            "case_ids": ",".join(report["identities"]["case_ids"]),
            "sampling_rate_hz": report["time"]["sampling_rate_hz"].get("median"),
            "window_duration_s": report["time"]["duration_s"].get("median"),
            "overlap_count": report["time"]["overlap_count"],
            "duplicate_interval_count": report["time"]["duplicate_interval_count"],
            "warning_count": len(report["warnings"]),
            "required_failure_count": len(report["required_failures"]),
        }
        for width in (60, 120, 300):
            preview = report["event_preview"][str(width)]
            summary[f"events_{width}s"] = preview["n_events"]
            summary[f"common_queries_{width}s"] = preview[
                "n_common_query_events_after_event_5"
            ]
            summary[f"initially_eligible_{width}s"] = preview[
                "initial_eligibility_if_used"
            ]
        summaries.append(summary)

    summary_table = pd.DataFrame(summaries)
    combined_index = (
        pd.concat(combined_indexes, ignore_index=True)
        if combined_indexes
        else pd.DataFrame()
    )

    if summaries:
        sha_values = [row["sha256"] for row in summaries]
        if len(set(sha_values)) != len(sha_values):
            required_failures.append("raw_file_sha256_values_are_unique")
    if not combined_index.empty:
        file_subject_pairs = combined_index[["raw_file_relative_path", "subject_uid"]].drop_duplicates()
        subject_file_counts = file_subject_pairs.groupby("subject_uid")["raw_file_relative_path"].nunique()
        if bool((subject_file_counts > 1).any()):
            required_failures.append("subject_uid_appears_in_multiple_raw_files")

    summary_csv = output_dir / "pilot_file_summary.csv"
    summary_parquet = output_dir / "pilot_file_summary.parquet"
    combined_csv = output_dir / "pilot_segment_index.csv"
    combined_parquet = output_dir / "pilot_segment_index.parquet"
    report_json = output_dir / "pilot_audit.json"
    report_markdown = output_dir / "pilot_audit.md"

    summary_table.to_csv(summary_csv, index=False)
    summary_table.to_parquet(summary_parquet, index=False)
    combined_index.to_csv(combined_csv, index=False)
    combined_index.to_parquet(combined_parquet, index=False)

    all_warnings = [
        {
            "source": report["input"]["source"],
            "file_name": Path(report["input"]["path"]).name,
            "warnings": report["warnings"],
        }
        for report in file_reports
        if report["warnings"]
    ]
    if all_warnings:
        warnings.append("One or more files have diagnostic warnings; inspect per-file reports.")

    status = "fail" if required_failures else ("pass_with_warnings" if warnings else "pass")
    pilot_report: dict[str, Any] = {
        "report_type": "PulseDB_v2_source_balanced_schema_pilot_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "selection_rule": "lexicographically_first_5_mat_filenames_per_source",
        "data_root": str(data_root),
        "download_manifest": str(download_manifest.resolve()),
        "expected_per_source": expected_per_source,
        "n_files": len(selected_files),
        "source_counts": dict(source_counts),
        "n_segments": int(len(combined_index)),
        "n_subject_uids": (
            int(combined_index["subject_uid"].nunique()) if not combined_index.empty else 0
        ),
        "required_failures": sorted(set(required_failures)),
        "warnings": warnings,
        "per_file_warnings": all_warnings,
        "artifacts": {
            "file_summary_csv": str(summary_csv),
            "file_summary_parquet": str(summary_parquet),
            "segment_index_csv": str(combined_csv),
            "segment_index_parquet": str(combined_parquet),
            "json": str(report_json),
            "markdown": str(report_markdown),
        },
        "interpretation_limits": [
            "This pilot validates parser and data-lineage feasibility only.",
            "Pilot event counts must not select the final event spacing.",
            "The participant split must be frozen on the full schema-valid cohort before development-only spacing selection.",
        ],
    }

    report_json.write_text(
        json.dumps(_jsonable(pilot_report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown_lines = [
        "# PulseDB v2 source-balanced pilot audit",
        "",
        f"- Status: `{status}`",
        f"- Files: `{pilot_report['n_files']}`",
        f"- Source counts: `{pilot_report['source_counts']}`",
        f"- Combined segments: `{pilot_report['n_segments']}`",
        f"- Unique source-qualified subjects: `{pilot_report['n_subject_uids']}`",
        f"- Required failures: `{pilot_report['required_failures'] or 'none'}`",
        "",
        "## Per-file feasibility",
        "",
    ]
    for row in summaries:
        markdown_lines.append(
            f"- `{row['source_directory']}/{row['file_name']}`: status={row['status']}, "
            f"windows={row['n_windows']}, events(60/120/300s)="
            f"{row['events_60s']}/{row['events_120s']}/{row['events_300s']}"
        )
    markdown_lines.extend(["", "## Limits", ""])
    markdown_lines.extend(f"- {item}" for item in pilot_report["interpretation_limits"])
    report_markdown.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(f"PILOT_AUDIT_STATUS={status}")
    print(f"PILOT_FILE_COUNT={pilot_report['n_files']}")
    print(f"PILOT_SOURCE_COUNTS={pilot_report['source_counts']}")
    print(f"PILOT_SEGMENT_COUNT={pilot_report['n_segments']}")
    print(f"PILOT_SUBJECT_UID_COUNT={pilot_report['n_subject_uids']}")
    print(
        "PILOT_REQUIRED_FAILURES="
        + (",".join(pilot_report["required_failures"]) if pilot_report["required_failures"] else "none")
    )
    print(f"PILOT_AUDIT_JSON={report_json}")
    print(f"PILOT_SEGMENT_INDEX={combined_parquet}")
    return pilot_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--download-manifest", required=True, type=Path)
    parser.add_argument("--expected-per-source", default=5, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_pulsedb_pilot(
        args.data_root,
        args.output,
        download_manifest=args.download_manifest,
        expected_per_source=args.expected_per_source,
    )
    if report["status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
