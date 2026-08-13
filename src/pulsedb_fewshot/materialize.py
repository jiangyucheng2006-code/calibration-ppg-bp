"""Materialize frozen-event PPG waveforms into sharded work-area NumPy arrays."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _stable_shard(subject_uid: str, n_shards: int) -> int:
    digest = hashlib.sha256(subject_uid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_shards


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_ppg(
    h5file: h5py.File, storage_mode: str, reference_index: int
) -> np.ndarray:
    dataset = h5file["Subj_Wins"]["PPG_F"]
    if storage_mode == "references":
        references = np.asarray(dataset[()]).reshape(-1, order="F")
        value = np.asarray(h5file[references[int(reference_index)]][()])
    elif storage_mode == "direct_single_window":
        if int(reference_index) != 0:
            raise ValueError("direct single-window storage requires reference index zero")
        value = np.asarray(dataset[()])
    else:
        raise ValueError(f"unsupported PPG storage mode: {storage_mode}")
    return np.asarray(value, dtype=np.float32).reshape(-1, order="F")


def _materialize_one(plan_path: str, signal_path: str) -> dict[str, object]:
    plan_file = Path(plan_path)
    signal_file = Path(signal_path)
    frame = pd.read_parquet(plan_file)
    if frame.empty:
        raise ValueError(f"empty shard plan: {plan_file}")
    sample_counts = set(frame["n_samples"].astype(int))
    if len(sample_counts) != 1:
        raise ValueError(f"mixed waveform lengths in {plan_file}: {sample_counts}")
    n_samples = sample_counts.pop()
    signals = np.lib.format.open_memmap(
        signal_file, mode="w+", dtype=np.float32, shape=(len(frame), n_samples)
    )
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
                        f"waveform length mismatch for {row.event_id}: {waveform.size}"
                    )
                if not np.isfinite(waveform).all() or float(np.std(waveform)) <= 0:
                    raise AssertionError(f"invalid waveform for {row.event_id}")
                signals[int(row.waveform_row)] = waveform
    signals.flush()
    del signals
    return {
        "plan": str(plan_file),
        "signals": str(signal_file),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_uid"].nunique()),
        "n_samples": int(n_samples),
        "signals_size": signal_file.stat().st_size,
        "signals_sha256": _sha256(signal_file),
        "plan_sha256": _sha256(plan_file),
    }


def _write_plans(
    frame: pd.DataFrame, root: Path, *, prefix: str, n_shards: int
) -> list[tuple[str, str]]:
    frame = frame.copy()
    frame["waveform_shard"] = frame["subject_uid"].map(
        lambda value: _stable_shard(str(value), n_shards)
    )
    tasks: list[tuple[str, str]] = []
    for shard, group in frame.groupby("waveform_shard", sort=True):
        group = group.sort_values(
            ["subject_uid", "event_index"], kind="mergesort"
        ).reset_index(drop=True)
        group["waveform_row"] = np.arange(len(group), dtype=np.int64)
        signal_name = f"{prefix}_signals_{int(shard):03d}.npy"
        group["waveform_file"] = signal_name
        plan = root / f"{prefix}_metadata_{int(shard):03d}.parquet"
        signal = root / signal_name
        group.to_parquet(plan, index=False)
        tasks.append((str(plan), str(signal)))
    return tasks


def materialize_event_waveforms(
    development_path: Path,
    locked_inputs_path: Path,
    output_root: Path,
    *,
    development_shards: int = 32,
    locked_shards: int = 8,
    workers: int = 4,
) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty materialization root: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    development = pd.read_parquet(development_path)
    locked_inputs = pd.read_parquet(locked_inputs_path)
    required = {
        "event_id",
        "subject_uid",
        "split",
        "event_index",
        "raw_file",
        "ppg_storage_mode",
        "ppg_reference_index",
        "n_samples",
    }
    for name, frame in (("development", development), ("locked", locked_inputs)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} manifest missing columns: {sorted(missing)}")
        if frame["event_id"].duplicated().any():
            raise AssertionError(f"{name} event IDs are not unique")
    if {"sbp", "dbp"} & set(locked_inputs.columns):
        raise AssertionError("locked model input exposes query BP columns")
    if set(development["split"]) - {"meta_train", "meta_validation"}:
        raise AssertionError("development manifest contains locked-test rows")
    if set(locked_inputs["split"]) != {"meta_test"}:
        raise AssertionError("locked input contains non-test rows")

    tasks = _write_plans(
        development,
        output_root,
        prefix="development",
        n_shards=development_shards,
    ) + _write_plans(
        locked_inputs,
        output_root,
        prefix="locked_inputs",
        n_shards=locked_shards,
    )
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_materialize_one, plan, signals): (plan, signals)
            for plan, signals in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"MATERIALIZED={Path(str(result['signals'])).name} "
                f"ROWS={result['rows']} SHA256={result['signals_sha256']}",
                flush=True,
            )

    result_rows = sum(int(item["rows"]) for item in results)
    expected_rows = len(development) + len(locked_inputs)
    if result_rows != expected_rows:
        raise AssertionError(
            f"materialized row total mismatch: {result_rows} != {expected_rows}"
        )
    report = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": "event120-v1",
        "dtype": "float32",
        "normalization": "none in store; per-window z-score is applied by the training loader",
        "development_input": str(development_path),
        "locked_input": str(locked_inputs_path),
        "development_rows": int(len(development)),
        "locked_input_rows": int(len(locked_inputs)),
        "materialized_rows": int(result_rows),
        "locked_query_bp_present": False,
        "shards": sorted(results, key=lambda item: str(item["signals"])),
    }
    report_path = output_root / "materialization.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--locked-inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--development-shards", type=int, default=32)
    parser.add_argument("--locked-shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = materialize_event_waveforms(
        args.development,
        args.locked_inputs,
        args.output,
        development_shards=args.development_shards,
        locked_shards=args.locked_shards,
        workers=args.workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
