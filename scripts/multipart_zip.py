#!/usr/bin/env python3
"""Inspect or extract a ZIP that was split into ``.zip.001`` style parts.

The parts are exposed to :mod:`zipfile` as one seekable virtual byte stream, so
the 362 GiB PulseDB archives do not need to be concatenated into a second copy.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import time
import zipfile


class MultipartReader(io.RawIOBase):
    """Read consecutive files as one seekable binary stream."""

    def __init__(self, parts: list[Path]):
        if not parts:
            raise ValueError("at least one part is required")
        self.parts = parts
        self.sizes = [path.stat().st_size for path in parts]
        self.ends: list[int] = []
        running = 0
        for size in self.sizes:
            running += size
            self.ends.append(running)
        self.length = running
        self.position = 0
        self._handle = None
        self._part_index = -1
        super().__init__()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            new_position = offset
        elif whence == os.SEEK_CUR:
            new_position = self.position + offset
        elif whence == os.SEEK_END:
            new_position = self.length + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if new_position < 0:
            raise OSError("negative seek position")
        self.position = min(new_position, self.length)
        return self.position

    def _open_part(self, index: int):
        if self._part_index != index:
            if self._handle is not None:
                self._handle.close()
            self._handle = self.parts[index].open("rb")
            self._part_index = index
        start = 0 if index == 0 else self.ends[index - 1]
        self._handle.seek(self.position - start)

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
        remaining = min(size, self.length - self.position)
        chunks: list[bytes] = []
        while remaining:
            index = bisect.bisect_right(self.ends, self.position)
            self._open_part(index)
            available = self.ends[index] - self.position
            chunk = self._handle.read(min(remaining, available))
            if not chunk:
                raise OSError(f"unexpected EOF in {self.parts[index]}")
            chunks.append(chunk)
            self.position += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def discover_parts(first_part: Path) -> list[Path]:
    name = first_part.name
    if not name.endswith(".001"):
        raise ValueError("--first-part must end in .001")
    prefix = name[:-3]
    candidates = sorted(first_part.parent.glob(prefix + "[0-9][0-9][0-9]"))
    expected = [first_part.parent / f"{prefix}{index:03d}" for index in range(1, len(candidates) + 1)]
    if candidates != expected:
        raise ValueError(f"multipart sequence is not contiguous: {candidates}")
    return candidates


def safe_member_path(name: str) -> Path:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    if pure.parts[0].endswith(":"):
        raise ValueError(f"unsafe drive-qualified member path: {name!r}")
    return Path(*pure.parts)


def inspect_archive(first_part: Path, expected_mat_count: int | None) -> tuple[dict, list[zipfile.ZipInfo]]:
    parts = discover_parts(first_part)
    with MultipartReader(parts) as reader, zipfile.ZipFile(reader) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate member names found")
        for info in infos:
            safe_member_path(info.filename)
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted member is not allowed: {info.filename}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"symbolic link member is not allowed: {info.filename}")
        mat_infos = [info for info in infos if not info.is_dir() and info.filename.lower().endswith(".mat")]
        if expected_mat_count is not None and len(mat_infos) != expected_mat_count:
            raise ValueError(
                f"expected {expected_mat_count} MAT members, found {len(mat_infos)}"
            )
        summary = {
            "first_part": str(first_part),
            "part_count": len(parts),
            "archive_bytes": sum(path.stat().st_size for path in parts),
            "member_count": len(infos),
            "mat_member_count": len(mat_infos),
            "uncompressed_bytes": sum(info.file_size for info in infos),
            "compressed_member_bytes": sum(info.compress_size for info in infos),
            "compression_methods": sorted({info.compress_type for info in infos}),
            "first_members": names[:10],
            "last_members": names[-10:],
        }
        return summary, infos


def extract_archive(
    first_part: Path,
    output_root: Path,
    expected_mat_count: int | None,
    progress_path: Path | None,
) -> dict:
    summary, _ = inspect_archive(first_part, expected_mat_count)
    parts = discover_parts(first_part)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    extracted_files = 0
    extracted_bytes = 0
    with MultipartReader(parts) as reader, zipfile.ZipFile(reader) as archive:
        infos = archive.infolist()
        for index, info in enumerate(infos, start=1):
            relative = safe_member_path(info.filename)
            target = output_root / relative
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"refusing to overwrite existing file: {target}")
            temporary = target.with_name(target.name + ".partial")
            if temporary.exists():
                raise FileExistsError(f"stale partial extraction exists: {temporary}")
            digest = hashlib.sha256()
            with archive.open(info, "r") as source, temporary.open("xb") as destination:
                while True:
                    block = source.read(8 * 1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    digest.update(block)
            if temporary.stat().st_size != info.file_size:
                raise OSError(f"size mismatch after extracting {info.filename}")
            temporary.replace(target)
            extracted_files += 1
            extracted_bytes += info.file_size
            if progress_path is not None and (extracted_files == 1 or extracted_files % 25 == 0):
                progress = {
                    **summary,
                    "phase": "extracting",
                    "members_processed": index,
                    "members_total": len(infos),
                    "files_extracted": extracted_files,
                    "bytes_extracted": extracted_bytes,
                    "last_member": info.filename,
                    "last_member_sha256": digest.hexdigest(),
                    "elapsed_seconds": time.time() - started,
                }
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_progress = progress_path.with_suffix(progress_path.suffix + ".tmp")
                temporary_progress.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
                temporary_progress.replace(progress_path)
    final = {
        **summary,
        "phase": "extracted",
        "files_extracted": extracted_files,
        "bytes_extracted": extracted_bytes,
        "elapsed_seconds": time.time() - started,
        "output_root": str(output_root),
    }
    if progress_path is not None:
        progress_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inspect", "extract"))
    parser.add_argument("--first-part", type=Path, required=True)
    parser.add_argument("--expected-mat-count", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "inspect":
        summary, _ = inspect_archive(args.first_part, args.expected_mat_count)
    else:
        if args.output_root is None:
            raise SystemExit("--output-root is required for extract")
        summary = extract_archive(
            args.first_part,
            args.output_root,
            args.expected_mat_count,
            args.progress,
        )
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MULTIPART_ZIP_ERROR={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
