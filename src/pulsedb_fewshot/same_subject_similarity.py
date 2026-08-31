"""Compute train-role beat similarity for the isolated same-subject screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .analyze_beat_similarity import beat_similarity
from .calbased_screen import PROTOCOL_ID
from .calbased_train import _load_role_metadata
from .training import WaveformAccessor, save_json


def build_train_similarity(
    store_root: Path,
    split_mode: str,
    output: Path,
    *,
    progress_every: int = 10000,
) -> dict[str, object]:
    metadata = _load_role_metadata(store_root, split_mode, "train")
    accessor = WaveformAccessor(store_root)
    rows: list[dict[str, object]] = []
    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        ppg = accessor.get(str(row.waveform_file), int(row.waveform_row))
        result = beat_similarity(ppg.numpy().reshape(-1), float(row.sampling_rate_hz))
        rows.append(
            {
                "subject_uid": str(row.subject_uid),
                "segment_uid": str(row.segment_uid),
                "source": str(row.source),
                "valid": bool(result.get("valid", False)),
                "median_pairwise_similarity": result.get("pairwise_corr_median"),
                "n_beats": int(result.get("n_beats", 0)),
                "reason": str(result.get("reason", "unknown")),
            }
        )
        if index % progress_every == 0:
            print(json.dumps({"processed": index, "total": len(metadata)}), flush=True)
    table = pd.DataFrame(rows)
    if table[["subject_uid", "segment_uid"]].duplicated().any():
        raise ValueError("similarity output contains duplicate train windows")
    output.mkdir(parents=True, exist_ok=False)
    table.to_parquet(output / "train_beat_similarity.parquet", index=False)
    valid = table.loc[table["valid"]]
    payload = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "split_mode": split_mode,
        "role": "train",
        "input_signal_only": True,
        "bp_targets_used": False,
        "internal_validation_accessed": False,
        "heldout_test_accessed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "windows": int(len(table)),
        "valid_windows": int(len(valid)),
        "similarity_ge_0_90_windows": int(
            pd.to_numeric(valid["median_pairwise_similarity"], errors="coerce").ge(0.90).sum()
        ),
    }
    save_json(output / "run.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--split-mode", default="random_disjoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_train_similarity(args.store_root, args.split_mode, args.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

