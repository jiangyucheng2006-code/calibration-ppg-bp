"""Acquire the two public, publisher-linked PulseDB CalBased assignment files.

Resolve on a connected desktop using HEAD only; download data only on the
server. Private temporary CDN URLs belong in ignored local_archive, not Git.
No BP targets or waveforms are parsed by this acquisition utility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

COMMIT = "db0824f18d9a462458e46fe94c31283a93a5c0d5"
GUIDE = f"https://github.com/pulselabteam/PulseDB/blob/{COMMIT}/Info_Files/File_Preparation_Guide.md"
SHARE = "b0q4v7kiqwbw3o9mczey9o615sjo2qsp"
# File IDs, sizes and SHA1 values observed in the official Box file pages.
FILES = (
    ("CalBased_Test_Info.mat", "1558406702219", 396689373, "0887f79e4618b19ff0be9ab0190f2a389432a108"),
    ("Train_Info.mat", "1558406704619", 3583749903, "c785192a7a860a71e4006e97ac0fd9e7c4b83697"),
)


def hashes(path: Path) -> dict[str, str]:
    digests = {k: hashlib.new(k) for k in ("sha1", "sha256")}
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            for digest in digests.values():
                digest.update(chunk)
    return {k: d.hexdigest() for k, d in digests.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("resolve", "download"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nas", type=Path)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    if args.mode == "resolve":
        rows = []
        for name, file_id, size, sha1 in FILES:
            url = f"https://rutgers.box.com/index.php?rm=box_download_shared_file&shared_name={SHARE}&file_id=f_{file_id}"
            proc = subprocess.run(["curl", "-I", "-L", "-sS", "--max-time", "45", "-w", "\nFINAL_URL=%{url_effective}\n", url], check=True, capture_output=True, text=True)
            final_url = proc.stdout.split("FINAL_URL=")[-1].strip()
            if urlparse(final_url).hostname != "public.boxcloud.com":
                raise RuntimeError(f"Unexpected CDN host for {name}")
            lengths = [line.split(":", 1)[1].strip() for line in proc.stdout.splitlines() if line.lower().startswith("content-length:")]
            if not lengths or int(lengths[-1]) != size:
                raise RuntimeError(f"Official size mismatch for {name}")
            rows.append(dict(name=name, file_id=file_id, size_bytes=size, official_box_sha1=sha1, public_url=url, signed_url=final_url))
            print(f"RESOLVED={name} BYTES={size}", flush=True)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(dict(source_commit=COMMIT, guide=GUIDE, resolved_at=datetime.now(timezone.utc).isoformat(), files=rows), indent=2) + "\n", encoding="utf-8")
        return

    if args.nas is None or args.work is None:
        parser.error("download requires --nas and --work")
    if str(args.nas).startswith("C:") or str(args.work).startswith("C:"):
        raise RuntimeError("Raw assignment files must remain server-side")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["source_commit"] != COMMIT:
        raise RuntimeError("Source commit mismatch")
    args.nas.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    receipt = dict(source_commit=COMMIT, guide=GUIDE, test_bp_or_waveform_parsed=False, files=[])
    for row, expected in zip(manifest["files"], FILES, strict=True):
        if (row["name"], row["file_id"], row["size_bytes"], row["official_box_sha1"]) != expected:
            raise RuntimeError("Manifest identity mismatch")
        if urlparse(row["signed_url"]).hostname != "public.boxcloud.com":
            raise RuntimeError("Unexpected download host")
        target = args.nas / row["name"]
        if not target.exists():
            partial = target.with_suffix(".mat.part")
            subprocess.run(["curl", "-L", "--fail", "--show-error", "--silent", "--connect-timeout", "20", "--max-time", "10800", "--retry", "2", "-C", "-", "-o", str(partial), row["signed_url"]], check=True)
            if partial.stat().st_size != row["size_bytes"]:
                raise RuntimeError("Downloaded size mismatch")
            hd = hashes(partial)
            if hd["sha1"] != row["official_box_sha1"]:
                raise RuntimeError("Official Box SHA1 mismatch")
            partial.rename(target)
        else:
            hd = hashes(target)
            if target.stat().st_size != row["size_bytes"] or hd["sha1"] != row["official_box_sha1"]:
                raise RuntimeError("Existing file failed identity check; preserved")
        work_target = args.work / target.name
        if work_target.exists():
            if hashes(work_target) != hd:
                raise RuntimeError("Existing work file differs; preserved")
        else:
            staging = work_target.with_suffix(".mat.copying")
            if staging.exists():
                raise RuntimeError("Existing work staging copy preserved; inspect before retry")
            shutil.copy2(target, staging)
            if hashes(staging) != hd:
                raise RuntimeError("Work staging copy checksum mismatch")
            staging.rename(work_target)
        if hashes(work_target) != hd:
            raise RuntimeError("NAS/work identity failure")
        receipt["files"].append(dict(name=row["name"], size_bytes=target.stat().st_size, nas=str(target), work=str(work_target), public_url=row["public_url"], **hd))
        receipt["updated_at"] = datetime.now(timezone.utc).isoformat()
        for root in (args.nas, args.work):
            (root / "acquisition_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"VERIFIED={target.name} SIZE={target.stat().st_size} SHA256={hd['sha256']}", flush=True)
    print("OFFICIAL_INFO_ACQUISITION_COMPLETE=yes", flush=True)


if __name__ == "__main__":
    main()
