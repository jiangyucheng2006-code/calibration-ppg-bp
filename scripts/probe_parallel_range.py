"""Measure parallel HTTP range throughput without touching dataset files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-json", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--bytes-per-connection", type=int, default=1_048_576)
    parser.add_argument("--base-offset", type=int, default=2_147_483_648)
    args = parser.parse_args()

    if args.connections < 1 or args.bytes_per_connection < 1:
        raise ValueError("connections and bytes-per-connection must be positive")

    metadata = json.loads(args.url_json.read_text(encoding="utf-8"))
    url = metadata["signed_url"]
    if urlparse(url).hostname != "public.boxcloud.com":
        raise RuntimeError("unexpected signed URL host")

    if args.scratch.exists():
        raise RuntimeError(f"probe scratch path already exists: {args.scratch}")
    args.scratch.mkdir(parents=True)

    def fetch(index: int) -> int:
        start = args.base_offset + index * args.bytes_per_connection
        end = start + args.bytes_per_connection - 1
        output = args.scratch / f"part-{index:03d}.bin"
        completed = subprocess.run(
            [
                "curl",
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "20",
                "--max-time",
                "180",
                "--range",
                f"{start}-{end}",
                "--output",
                str(output),
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"range {index} failed with curl code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        actual = output.stat().st_size
        if actual != args.bytes_per_connection:
            raise RuntimeError(
                f"range {index} size mismatch: {actual} != "
                f"{args.bytes_per_connection}"
            )
        return actual

    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.connections) as executor:
            sizes = list(executor.map(fetch, range(args.connections)))
        elapsed = time.monotonic() - started
        total = sum(sizes)
        print("PARALLEL_RANGE_PROBE=pass")
        print(f"CONNECTIONS={args.connections}")
        print(f"TOTAL_BYTES={total}")
        print(f"ELAPSED_SECONDS={elapsed:.6f}")
        print(f"THROUGHPUT_BYTES_PER_SECOND={total / elapsed:.3f}")
        print(f"THROUGHPUT_MIB_PER_SECOND={total / elapsed / 1024**2:.6f}")
    finally:
        shutil.rmtree(args.scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
