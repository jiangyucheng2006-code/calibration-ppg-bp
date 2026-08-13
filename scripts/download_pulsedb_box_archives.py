"""Resumable, checksum-gated PulseDB v2 Box archive acquisition."""

from __future__ import annotations

import argparse
import csv
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


EXPECTED_COMMIT = "db0824f18d9a462458e46fe94c31283a93a5c0d5"
FILE_RE = re.compile(r"PulseDB_(?:MIMIC|Vital)\.zip\.\d{3}$")
SAFETY_MARGIN_BYTES = 20 * 1024**3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


class Progress:
    def __init__(
        self,
        rows: list[dict[str, str]],
        destination: Path,
        status_path: Path,
    ) -> None:
        self.rows = rows
        self.destination = destination
        self.status_path = status_path
        self.lock = threading.Lock()
        self.started_at = utc_now()
        self.states: dict[str, dict[str, object]] = {
            row["file_name"]: {"state": "pending", "message": ""} for row in rows
        }

    def update(self, name: str, state: str, message: str = "") -> None:
        with self.lock:
            self.states[name] = {"state": state, "message": message}
            self.write_locked()

    def write(self) -> None:
        with self.lock:
            self.write_locked()

    def write_locked(self) -> None:
        expected_total = sum(int(row["size_bytes"]) for row in self.rows)
        present_total = 0
        file_rows: list[dict[str, object]] = []
        for row in self.rows:
            target = self.destination / row["file_name"]
            present = target.stat().st_size if target.exists() else 0
            present_total += min(present, int(row["size_bytes"]))
            file_rows.append(
                {
                    "file_name": row["file_name"],
                    "source": row["source"],
                    "expected_bytes": int(row["size_bytes"]),
                    "present_bytes": present,
                    **self.states[row["file_name"]],
                }
            )
        payload: dict[str, object] = {
            "phase": "PulseDB_v2_Box_archive_acquisition",
            "started_at_utc": self.started_at,
            "updated_at_utc": utc_now(),
            "expected_files": len(self.rows),
            "expected_bytes": expected_total,
            "present_bytes": present_total,
            "percent": round(100.0 * present_total / expected_total, 4),
            "verified_files": sum(
                state["state"] == "verified" for state in self.states.values()
            ),
            "failed_files": sum(
                state["state"] == "failed" for state in self.states.values()
            ),
            "destination": str(self.destination),
            "files": file_rows,
        }
        atomic_json(self.status_path, payload)


def validate_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 26:
        raise RuntimeError(f"expected 26 archive parts, found {len(rows)}")
    names = [row["file_name"] for row in rows]
    if len(set(names)) != len(names):
        raise RuntimeError("manifest contains duplicate filenames")
    for row in rows:
        if FILE_RE.fullmatch(row["file_name"]) is None:
            raise RuntimeError(f"unsafe or unexpected filename: {row['file_name']!r}")
        if row["source"] not in {"PulseDB_MIMIC", "PulseDB_Vital"}:
            raise RuntimeError(f"unexpected source: {row['source']!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["sha1"]):
            raise RuntimeError(f"invalid SHA-1 for {row['file_name']}")
        if int(row["size_bytes"]) <= 0:
            raise RuntimeError(f"invalid size for {row['file_name']}")
        if row["source_commit"] != EXPECTED_COMMIT:
            raise RuntimeError(f"unexpected source commit for {row['file_name']}")
        if urlparse(row["signed_url"]).hostname != "public.boxcloud.com":
            raise RuntimeError(f"unexpected signed URL host for {row['file_name']}")
    if sum(row["source"] == "PulseDB_MIMIC" for row in rows) != 16:
        raise RuntimeError("manifest does not contain 16 MIMIC parts")
    if sum(row["source"] == "PulseDB_Vital" for row in rows) != 10:
        raise RuntimeError("manifest does not contain 10 Vital parts")
    return rows


def download_one(
    row: dict[str, str],
    destination: Path,
    log_directory: Path,
    progress: Progress,
) -> str:
    name = row["file_name"]
    target = destination / name
    expected_size = int(row["size_bytes"])
    expected_sha1 = row["sha1"]
    log_path = log_directory / f"{name}.log"

    progress.update(name, "checking")
    if target.exists():
        current_size = target.stat().st_size
        if current_size > expected_size:
            raise RuntimeError(
                f"{name}: existing file exceeds expected size "
                f"({current_size} > {expected_size})"
            )
        if current_size == expected_size:
            progress.update(name, "hashing")
            actual_sha1 = sha1_file(target)
            if actual_sha1 != expected_sha1:
                raise RuntimeError(
                    f"{name}: complete-size file has wrong SHA-1 {actual_sha1}"
                )
            progress.update(name, "verified", "existing file passed SHA-1")
            return name

    progress.update(name, "downloading")
    command = [
        "curl",
        "--location",
        "--fail",
        "--show-error",
        "--continue-at",
        "-",
        "--retry",
        "20",
        "--retry-all-errors",
        "--retry-delay",
        "15",
        "--connect-timeout",
        "30",
        "--speed-time",
        "300",
        "--speed-limit",
        "1024",
        "--output",
        str(target),
        row["signed_url"],
    ]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"START {utc_now()} existing_bytes={target.stat().st_size if target.exists() else 0}\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=log, check=False)
        log.write(f"CURL_EXIT {utc_now()} code={completed.returncode}\n")
        log.flush()
    if completed.returncode != 0:
        raise RuntimeError(f"{name}: curl exited with code {completed.returncode}")
    if not target.exists() or target.stat().st_size != expected_size:
        actual_size = target.stat().st_size if target.exists() else -1
        raise RuntimeError(
            f"{name}: size mismatch after download {actual_size} != {expected_size}"
        )

    progress.update(name, "hashing")
    actual_sha1 = sha1_file(target)
    if actual_sha1 != expected_sha1:
        raise RuntimeError(f"{name}: SHA-1 mismatch {actual_sha1} != {expected_sha1}")
    progress.update(name, "verified", "size and official SHA-1 passed")
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    rows = validate_manifest(args.manifest)
    args.destination.mkdir(parents=True, exist_ok=True)
    args.log_directory.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(int(row["size_bytes"]) for row in rows)
    existing_bytes = sum(
        min(
            (args.destination / row["file_name"]).stat().st_size
            if (args.destination / row["file_name"]).exists()
            else 0,
            int(row["size_bytes"]),
        )
        for row in rows
    )
    required_bytes = total_bytes - existing_bytes + SAFETY_MARGIN_BYTES
    free_bytes = shutil.disk_usage(args.destination).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"insufficient free space: need {required_bytes}, have {free_bytes}"
        )

    progress = Progress(rows, args.destination, args.status)
    progress.write()
    stop_monitor = threading.Event()

    def monitor() -> None:
        while not stop_monitor.wait(15):
            progress.write()

    monitor_thread = threading.Thread(target=monitor, name="progress-monitor", daemon=True)
    monitor_thread.start()

    failures: list[str] = []
    print(
        f"ACQUISITION_START files={len(rows)} total_bytes={total_bytes} "
        f"free_bytes={free_bytes} workers={args.workers}",
        flush=True,
    )
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_one, row, args.destination, args.log_directory, progress
                ): row
                for row in rows
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    name = future.result()
                    print(f"ARCHIVE_VERIFIED file={name}", flush=True)
                except Exception as exc:  # retain other resumable downloads
                    message = str(exc)
                    failures.append(message)
                    progress.update(row["file_name"], "failed", message)
                    print(f"ARCHIVE_FAILED file={row['file_name']} error={message}", flush=True)
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=30)
        progress.write()

    if failures:
        print(f"ACQUISITION_FAILED failures={len(failures)}", flush=True)
        return 1
    print("ACQUISITION_COMPLETE files=26 all_official_sha1_passed=yes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
