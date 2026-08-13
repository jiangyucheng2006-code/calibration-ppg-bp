#!/bin/bash
#SBATCH --job-name=ppg_extract_v2
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
nas_root="/home/$USER/nas/ppg_bp"
project="$work_root/code/pulsedb_fewshot"
python="$work_root/envs/train/bin/python"
archive_root="$nas_root/data/raw/PulseDB_v2/downloads/box_parts"
staging_root="$nas_root/data/raw/PulseDB_v2/extraction_staging_20260811/Segment_Files"
pilot_root="$nas_root/data/raw/PulseDB_v2/Segment_Files"
work_report="$work_root/data/manifests/full_extraction"
nas_report="$nas_root/data/manifests/full_extraction"

expected_mimic=2423
expected_vital=2938
expected_total=5361
expected_uncompressed_bytes=636177283705
minimum_headroom_bytes=53687091200

mkdir -p "$work_report" "$nas_report" "$work_root/logs"

echo "===== PULSEDB V2 CONTROLLED MULTIPART EXTRACTION ====="
echo "TIME=$(date -Is)"
echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ARCHIVE_ROOT=$archive_root"
echo "STAGING_ROOT=$staging_root"
echo "EXPECTED_TOTAL=$expected_total"

if find "$staging_root" -type f -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: staging root already contains files; refusing a non-resume extraction" >&2
  exit 1
fi

available_bytes="$(df -B1 --output=avail "$nas_root" | tail -n 1 | tr -d ' ')"
required_bytes="$((expected_uncompressed_bytes + minimum_headroom_bytes))"
echo "AVAILABLE_BYTES=$available_bytes"
echo "REQUIRED_BYTES=$required_bytes"
if [ "$available_bytes" -lt "$required_bytes" ]; then
  echo "ERROR: insufficient NAS capacity for controlled extraction" >&2
  exit 1
fi

cd "$project"
"$python" -m pytest tests/test_multipart_zip.py -q

"$python" scripts/multipart_zip.py inspect \
  --first-part "$archive_root/PulseDB_MIMIC.zip.001" \
  --expected-mat-count "$expected_mimic" \
  --report "$work_report/PulseDB_MIMIC_archive_inspect.json"

"$python" scripts/multipart_zip.py inspect \
  --first-part "$archive_root/PulseDB_Vital.zip.001" \
  --expected-mat-count "$expected_vital" \
  --report "$work_report/PulseDB_Vital_archive_inspect.json"

mkdir -p "$staging_root"

"$python" scripts/multipart_zip.py extract \
  --first-part "$archive_root/PulseDB_MIMIC.zip.001" \
  --expected-mat-count "$expected_mimic" \
  --output-root "$staging_root" \
  --progress "$work_report/PulseDB_MIMIC_extract_progress.json" \
  --report "$work_report/PulseDB_MIMIC_extract_report.json" \
  > "$work_root/logs/PulseDB_MIMIC_extract-${SLURM_JOB_ID}.out" \
  2> "$work_root/logs/PulseDB_MIMIC_extract-${SLURM_JOB_ID}.err" &
mimic_pid=$!

"$python" scripts/multipart_zip.py extract \
  --first-part "$archive_root/PulseDB_Vital.zip.001" \
  --expected-mat-count "$expected_vital" \
  --output-root "$staging_root" \
  --progress "$work_report/PulseDB_Vital_extract_progress.json" \
  --report "$work_report/PulseDB_Vital_extract_report.json" \
  > "$work_root/logs/PulseDB_Vital_extract-${SLURM_JOB_ID}.out" \
  2> "$work_root/logs/PulseDB_Vital_extract-${SLURM_JOB_ID}.err" &
vital_pid=$!

cleanup_children() {
  kill "$mimic_pid" "$vital_pid" 2>/dev/null || true
}
trap cleanup_children INT TERM

set +e
wait "$mimic_pid"
mimic_rc=$?
wait "$vital_pid"
vital_rc=$?
set -e
trap - INT TERM

echo "MIMIC_EXTRACT_RC=$mimic_rc"
echo "VITAL_EXTRACT_RC=$vital_rc"
if [ "$mimic_rc" -ne 0 ] || [ "$vital_rc" -ne 0 ]; then
  echo "ERROR: one or both extraction processes failed" >&2
  exit 1
fi

"$python" - "$staging_root" "$pilot_root" "$work_report/pulsedb_v2_extraction_validation.json" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

staging = Path(sys.argv[1])
pilot = Path(sys.argv[2])
report_path = Path(sys.argv[3])
expected = {"PulseDB_MIMIC": 2423, "PulseDB_Vital": 2938}
signature = b"\x89HDF\r\n\x1a\n"
name_re = re.compile(r"p\d{6}\.mat")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


source_counts = {}
bad_names = []
bad_signatures = []
unexpected_files = []
for source, expected_count in expected.items():
    source_root = staging / source
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    mat_files = [path for path in files if path.suffix.lower() == ".mat"]
    source_counts[source] = len(mat_files)
    unexpected_files.extend(str(path) for path in files if path.suffix.lower() != ".mat")
    for path in mat_files:
        if not name_re.fullmatch(path.name):
            bad_names.append(str(path))
        with path.open("rb") as handle:
            handle.seek(512)
            if handle.read(8) != signature:
                bad_signatures.append(str(path))
    if len(mat_files) != expected_count:
        raise RuntimeError(f"{source}: expected {expected_count} MAT files, found {len(mat_files)}")

partials = sorted(str(path) for path in staging.rglob("*.partial"))
if bad_names or bad_signatures or unexpected_files or partials:
    raise RuntimeError(
        f"validation failures: bad_names={len(bad_names)}, "
        f"bad_signatures={len(bad_signatures)}, unexpected={len(unexpected_files)}, "
        f"partials={len(partials)}"
    )

pilot_matches = []
for pilot_file in sorted(pilot.glob("PulseDB_*/*.mat")):
    staged_file = staging / pilot_file.relative_to(pilot)
    if not staged_file.is_file():
        raise RuntimeError(f"pilot file is missing from extracted archive: {staged_file}")
    pilot_hash = sha256(pilot_file)
    staged_hash = sha256(staged_file)
    if pilot_hash != staged_hash:
        raise RuntimeError(f"pilot/extracted content mismatch: {pilot_file}")
    pilot_matches.append(
        {"relative_path": str(pilot_file.relative_to(pilot)), "sha256": pilot_hash}
    )

report = {
    "status": "pass",
    "staging_root": str(staging),
    "source_counts": source_counts,
    "total_mat_files": sum(source_counts.values()),
    "bad_name_count": len(bad_names),
    "bad_hdf5_signature_count": len(bad_signatures),
    "unexpected_file_count": len(unexpected_files),
    "partial_file_count": len(partials),
    "pilot_files_compared": len(pilot_matches),
    "pilot_files_identical": True,
    "pilot_matches": pilot_matches,
}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

rsync -a "$work_report/" "$nas_report/"
cmp "$work_report/pulsedb_v2_extraction_validation.json" \
    "$nas_report/pulsedb_v2_extraction_validation.json"

echo "PULSEDB_V2_EXTRACTION_STAGING_COMPLETE=yes"
echo "EXTRACTED_MAT_FILES=$expected_total"
echo "STAGING_ROOT=$staging_root"
echo "WORK_REPORT=$work_report"
echo "NAS_REPORT=$nas_report"
