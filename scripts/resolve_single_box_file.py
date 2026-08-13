"""Resolve one checksum-pinned PulseDB Box part to a temporary CDN URL."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


CONTENT_RANGE = re.compile(r"bytes\s+0-0/(\d+)$")
SOURCE_COMMIT = "db0824f18d9a462458e46fe94c31283a93a5c0d5"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 26:
        raise RuntimeError(f"expected 26 official parts, found {len(rows)}")
    matches = [row for row in rows if row["file_name"] == args.file_name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one official row for {args.file_name}")
    row = matches[0]
    if re.fullmatch(r"[0-9a-f]{40}", row["sha1"]) is None:
        raise RuntimeError("invalid official SHA-1")

    opener = build_opener(ProxyHandler({"http": args.proxy, "https": args.proxy}))
    request = Request(row["url"], headers={"Range": "bytes=0-0"})
    response = opener.open(request, timeout=args.timeout_seconds)
    try:
        if response.status != 206:
            raise RuntimeError(f"expected HTTP 206, got {response.status}")
        match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", "").strip())
        if match is None:
            raise RuntimeError("invalid Content-Range")
        if urlparse(response.geturl()).hostname != "public.boxcloud.com":
            raise RuntimeError("unexpected final host")
        payload = {
            "file_name": row["file_name"],
            "source": row["source"],
            "url": row["url"],
            "sha1": row["sha1"],
            "size_bytes": int(match.group(1)),
            "signed_url": response.geturl(),
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": SOURCE_COMMIT,
        }
    finally:
        response.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=args.output.name + ".", suffix=".tmp", dir=args.output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    print(f"RESOLVED file={payload['file_name']}")
    print(f"SIZE_BYTES={payload['size_bytes']}")
    print(f"RESOLVED_AT_UTC={payload['resolved_at_utc']}")
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
