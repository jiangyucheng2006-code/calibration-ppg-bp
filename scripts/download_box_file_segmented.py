"""Resumable parallel-range downloader for one checksum-pinned Box file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def chunk_ranges(size: int, chunk_size: int) -> list[tuple[int, int, int]]:
    if size <= 0 or chunk_size <= 0:
        raise ValueError("size and chunk_size must be positive")
    result = []
    for index, start in enumerate(range(0, size, chunk_size)):
        result.append((index, start, min(size - 1, start + chunk_size - 1)))
    return result


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--chunk-directory", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    name = metadata["file_name"]
    url = metadata["signed_url"]
    expected_size = int(metadata["size_bytes"])
    expected_sha1 = metadata["sha1"]
    if name != args.destination.name:
        raise RuntimeError(f"metadata/destination mismatch: {name}")
    if urlparse(url).hostname != "public.boxcloud.com":
        raise RuntimeError("unexpected signed URL host")
    if re.fullmatch(r"[0-9a-f]{40}", expected_sha1) is None:
        raise RuntimeError("invalid expected SHA-1")
    if args.workers < 1 or args.chunk_size < 1:
        raise RuntimeError("workers and chunk-size must be positive")

    ranges = chunk_ranges(expected_size, args.chunk_size)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.chunk_directory.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.log_directory.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    started_at = utc_now()
    states = {index: "pending" for index, _, _ in ranges}
    errors: dict[int, str] = {}

    def chunk_path(index: int) -> Path:
        return args.chunk_directory / f"chunk-{index:04d}.bin"

    def write_status(phase: str) -> None:
        with lock:
            completed_bytes = 0
            for index, start, end in ranges:
                path = chunk_path(index)
                completed_bytes += min(
                    path.stat().st_size if path.exists() else 0, end - start + 1
                )
            atomic_json(
                args.status,
                {
                    "phase": phase,
                    "file_name": name,
                    "started_at_utc": started_at,
                    "updated_at_utc": utc_now(),
                    "expected_bytes": expected_size,
                    "completed_chunk_bytes": completed_bytes,
                    "percent": round(100.0 * completed_bytes / expected_size, 4),
                    "workers": args.workers,
                    "chunk_size": args.chunk_size,
                    "chunk_count": len(ranges),
                    "state_counts": {
                        state: sum(value == state for value in states.values())
                        for state in sorted(set(states.values()))
                    },
                    "errors": errors,
                    "destination": str(args.destination),
                    "chunk_directory": str(args.chunk_directory),
                },
            )

    def fetch(item: tuple[int, int, int]) -> int:
        index, start, end = item
        expected = end - start + 1
        path = chunk_path(index)
        if path.exists() and path.stat().st_size == expected:
            with lock:
                states[index] = "complete"
            return expected
        temporary = path.with_suffix(".tmp")
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        log_path = args.log_directory / f"chunk-{index:04d}.log"
        with lock:
            states[index] = "downloading"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"START {utc_now()} range={start}-{end}\n")
            log.flush()
            completed = subprocess.run(
                [
                    "curl",
                    "--location",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "12",
                    "--retry-all-errors",
                    "--retry-delay",
                    "5",
                    "--connect-timeout",
                    "20",
                    "--speed-time",
                    "120",
                    "--speed-limit",
                    "1024",
                    "--range",
                    f"{start}-{end}",
                    "--output",
                    str(temporary),
                    "--write-out",
                    "%{http_code}",
                    url,
                ],
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                check=False,
            )
            http_code = completed.stdout[-3:] if completed.stdout else ""
            log.write(
                f"END {utc_now()} curl={completed.returncode} http={http_code}\n"
            )
        if completed.returncode != 0:
            raise RuntimeError(f"chunk {index}: curl code {completed.returncode}")
        if http_code != "206":
            raise RuntimeError(f"chunk {index}: expected HTTP 206, got {http_code}")
        actual = temporary.stat().st_size if temporary.exists() else -1
        if actual != expected:
            raise RuntimeError(f"chunk {index}: size {actual} != {expected}")
        os.replace(temporary, path)
        with lock:
            states[index] = "complete"
        return expected

    write_status("downloading")
    stop_monitor = threading.Event()

    def monitor() -> None:
        while not stop_monitor.wait(10):
            write_status("downloading")

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    failures: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch, item): item for item in ranges}
            for future in as_completed(futures):
                index, _, _ = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    message = str(exc)
                    with lock:
                        states[index] = "failed"
                        errors[index] = message
                    failures.append(message)
                    print(f"CHUNK_FAILED index={index} error={message}", flush=True)
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=30)
        write_status("download_failed" if failures else "download_complete")

    if failures:
        print(f"SEGMENTED_DOWNLOAD_FAILED failures={len(failures)}", flush=True)
        return 1

    assembled = args.destination.with_name(args.destination.name + ".assembled")
    digest = hashlib.sha1()
    assembled_size = 0
    write_status("assembling")
    with assembled.open("wb") as output:
        for index, start, end in ranges:
            path = chunk_path(index)
            expected = end - start + 1
            if path.stat().st_size != expected:
                raise RuntimeError(f"chunk {index} changed before assembly")
            with path.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    assembled_size += len(block)
        output.flush()
        os.fsync(output.fileno())
    actual_sha1 = digest.hexdigest()
    if assembled_size != expected_size:
        raise RuntimeError(f"assembled size {assembled_size} != {expected_size}")
    if actual_sha1 != expected_sha1:
        raise RuntimeError(f"assembled SHA-1 {actual_sha1} != {expected_sha1}")

    os.replace(assembled, args.destination)
    if args.destination.stat().st_size != expected_size:
        raise RuntimeError("destination size changed after atomic replacement")
    if sha1_file(args.destination) != expected_sha1:
        raise RuntimeError("destination failed post-replacement SHA-1")
    write_status("verified")
    # Keep the final status at 100% before removing the now-redundant chunks.
    # The destination has already passed the official SHA-1 twice at this point.
    shutil.rmtree(args.chunk_directory)
    print(f"SEGMENTED_DOWNLOAD_COMPLETE file={name}", flush=True)
    print(f"SIZE={expected_size}", flush=True)
    print(f"OFFICIAL_SHA1={expected_sha1}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
