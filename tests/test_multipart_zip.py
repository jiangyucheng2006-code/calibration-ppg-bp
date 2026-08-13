from pathlib import Path
import zipfile

import pytest

from scripts.multipart_zip import MultipartReader, discover_parts, inspect_archive


def make_split_zip(tmp_path: Path) -> Path:
    combined = tmp_path / "sample.zip"
    with zipfile.ZipFile(combined, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr("PulseDB_MIMIC/p000001.mat", b"A" * 1500)
        archive.writestr("PulseDB_MIMIC/p000002.mat", b"B" * 1800)
    payload = combined.read_bytes()
    combined.unlink()
    first = tmp_path / "sample.zip.001"
    for index, start in enumerate(range(0, len(payload), 97), start=1):
        (tmp_path / f"sample.zip.{index:03d}").write_bytes(payload[start : start + 97])
    return first


def test_virtual_reader_opens_split_zip(tmp_path: Path):
    first = make_split_zip(tmp_path)
    parts = discover_parts(first)
    with MultipartReader(parts) as reader, zipfile.ZipFile(reader) as archive:
        assert archive.read("PulseDB_MIMIC/p000001.mat") == b"A" * 1500
        assert archive.read("PulseDB_MIMIC/p000002.mat") == b"B" * 1800


def test_inspect_counts_mat_members(tmp_path: Path):
    first = make_split_zip(tmp_path)
    summary, _ = inspect_archive(first, expected_mat_count=2)
    assert summary["mat_member_count"] == 2
    assert summary["part_count"] > 1


def test_inspect_rejects_wrong_expected_count(tmp_path: Path):
    first = make_split_zip(tmp_path)
    with pytest.raises(ValueError, match="expected 3 MAT members"):
        inspect_archive(first, expected_mat_count=3)
