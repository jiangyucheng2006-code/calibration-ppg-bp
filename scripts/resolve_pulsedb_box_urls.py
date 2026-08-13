"""Resolve official PulseDB Box links to temporary Box CDN URLs.

The Rutgers Box entry host is blocked by the HPC network, while the final
public.boxcloud.com CDN host is reachable.  This helper performs one-byte range
requests through a caller-supplied local proxy and writes a temporary runtime
manifest for a server-side downloader.  It never downloads archive contents.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


SOURCE_COMMIT = "db0824f18d9a462458e46fe94c31283a93a5c0d5"
CONTENT_RANGE_RE = re.compile(r"bytes\s+0-0/(\d+)$")


def resolve(row: dict[str, str], proxy: str, timeout_seconds: float) -> dict[str, str]:
    response = requests.get(
        row["url"],
        headers={"Range": "bytes=0-0"},
        proxies={"http": proxy, "https": proxy},
        timeout=(10, timeout_seconds),
        allow_redirects=True,
        stream=True,
    )
    try:
        if response.status_code != 206:
            raise RuntimeError(
                f"{row['file_name']}: expected HTTP 206, got {response.status_code}"
            )
        content_range = response.headers.get("Content-Range", "")
        match = CONTENT_RANGE_RE.fullmatch(content_range.strip())
        if match is None:
            raise RuntimeError(
                f"{row['file_name']}: invalid Content-Range {content_range!r}"
            )
        if not response.url.startswith("https://public.boxcloud.com/"):
            raise RuntimeError(
                f"{row['file_name']}: unexpected final host in {response.url!r}"
            )
        return {
            **row,
            "size_bytes": match.group(1),
            "signed_url": response.url,
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": SOURCE_COMMIT,
        }
    finally:
        response.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 26:
        raise RuntimeError(f"expected 26 official archive parts, found {len(rows)}")

    resolved: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(resolve, row, args.proxy, args.timeout_seconds): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            result = future.result()
            resolved[row["file_name"]] = result
            print(
                f"RESOLVED file={row['file_name']} size_bytes={result['size_bytes']}",
                flush=True,
            )

    ordered = [resolved[row["file_name"]] for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_name",
        "source",
        "url",
        "sha1",
        "size_bytes",
        "signed_url",
        "resolved_at_utc",
        "source_commit",
    ]
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=args.output.name + ".", suffix=".tmp", dir=args.output.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, args.output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    total_bytes = sum(int(row["size_bytes"]) for row in ordered)
    print(f"RESOLVE_COMPLETE files={len(ordered)} total_bytes={total_bytes}")
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
