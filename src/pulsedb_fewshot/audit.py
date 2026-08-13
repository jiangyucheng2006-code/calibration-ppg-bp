"""Command-line audit for PulseDB segment manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .events import build_episode_assignments, eventize_segments, summarize_eligibility
from .splits import assign_subject_splits


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("--segments must be a .csv or .parquet file")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_audit(
    segments_path: Path,
    output_dir: Path,
    *,
    bin_width_sec: float,
    min_query_events: int,
    seed: int,
) -> dict[str, object]:
    segments = _read_table(segments_path)
    events = eventize_segments(segments, bin_width_sec=bin_width_sec)
    eligibility = summarize_eligibility(events, max_k=5, min_query_events=min_query_events)
    eligible = eligibility[eligibility["eligible"]].copy()
    splits = assign_subject_splits(eligible[["subject_id", "source"]], seed=seed)
    assignments = build_episode_assignments(
        events[events["subject_id"].isin(eligible["subject_id"])],
        ks=(1, 2, 3, 5),
        min_query_events=min_query_events,
    ).merge(splits[["subject_id", "split"]], on="subject_id", how="left", validate="many_to_one")

    if assignments.loc[assignments["role"] == "query", "split"].isna().any():
        raise AssertionError("Every eligible query event must have a subject split")

    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "events.csv", index=False)
    eligibility.to_csv(output_dir / "eligibility.csv", index=False)
    splits.to_csv(output_dir / "subject_splits.csv", index=False)
    assignments.to_csv(output_dir / "episode_assignments.csv", index=False)

    summary: dict[str, object] = {
        "segments_path": str(segments_path.resolve()),
        "segments_sha256": _sha256(segments_path),
        "bin_width_sec": bin_width_sec,
        "min_query_events": min_query_events,
        "ks": [1, 2, 3, 5],
        "seed": seed,
        "n_segments": int(len(segments)),
        "n_events": int(len(events)),
        "n_subjects_total": int(eligibility["subject_id"].nunique()),
        "n_subjects_eligible": int(eligible["subject_id"].nunique()),
        "split_counts": {k: int(v) for k, v in splits["split"].value_counts().items()},
        "split_sha256": splits.attrs.get("sha256"),
    }
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bin-width-sec", required=True, type=float)
    parser.add_argument("--min-query-events", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_audit(
        args.segments,
        args.output,
        bin_width_sec=args.bin_width_sec,
        min_query_events=args.min_query_events,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
