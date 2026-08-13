#!/usr/bin/env bash
set -uo pipefail

project_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
code_root="$project_root/code/pulsedb_fewshot"
environment_root="$project_root/envs/train"
input_file="$project_root/data/raw/PulseDB_v2/Segment_Files/PulseDB_MIMIC/p000160.mat"
output_dir="$project_root/data/manifests/p000160_full_audit"
archive_dir="$archive_root/data/manifests/p000160_full_audit"
work_log_dir="$project_root/logs"
archive_log_dir="$archive_root/logs"
timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="$work_log_dir/p000160_full_audit_${timestamp}.log"
expected_sha256="2b9fa8fcc6708041ed866de8d1ecfd4ca6f0c6982802f95dd8fc0d4d2f832783"

mkdir -p "$output_dir" "$archive_dir" "$work_log_dir" "$archive_log_dir"

set +e
{
  echo "===== P000160 COMPREHENSIVE PHASE-2 AUDIT ====="
  echo "TIME=$(date -Is)"
  echo "HOST=$(hostname)"
  echo "INPUT=$input_file"

  source /opt/conda/etc/profile.d/conda.sh
  conda activate "$environment_root"
  cd "$code_root"

  echo "PYTHON=$(command -v python)"
  python --version
  python -m pip check

  echo "===== BUNDLE TESTS ====="
  python -m pytest -q tests/test_schema_audit.py

  echo "===== REAL FILE AUDIT ====="
  python -m pulsedb_fewshot.schema_audit \
    --input "$input_file" \
    --output "$output_dir" \
    --expected-sha256 "$expected_sha256" \
    --source MIMIC
} 2>&1 | tee "$log_file"
audit_status=${PIPESTATUS[0]}
set -e

cp -p "$log_file" "$archive_log_dir/"

if compgen -G "$output_dir/*" >/dev/null; then
  cp -pr "$output_dir/." "$archive_dir/"
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate "$environment_root"
conda env export -p "$environment_root" > "$archive_root/environment.yml"

if [ "$audit_status" -ne 0 ]; then
  echo "P000160_FULL_AUDIT_COMPLETE=no"
  echo "AUDIT_EXIT_CODE=$audit_status"
  echo "WORK_LOG=$log_file"
  echo "NAS_LOG=$archive_log_dir/$(basename "$log_file")"
  exit "$audit_status"
fi

work_json="$output_dir/p000160_full_audit.json"
work_markdown="$output_dir/p000160_full_audit.md"
work_csv="$output_dir/p000160_segment_index.csv"
work_parquet="$output_dir/p000160_segment_index.parquet"

for artifact in "$work_json" "$work_markdown" "$work_csv" "$work_parquet"; do
  test -s "$artifact"
  cmp -s "$artifact" "$archive_dir/$(basename "$artifact")"
done

python -m json.tool "$work_json" >/dev/null

echo "ARCHIVE_IDENTICAL=yes"
echo "ENVIRONMENT_REEXPORTED=yes"
echo "P000160_FULL_AUDIT_COMPLETE=yes"
echo "WORK_OUTPUT=$output_dir"
echo "NAS_OUTPUT=$archive_dir"
echo "WORK_LOG=$log_file"
echo "NAS_LOG=$archive_log_dir/$(basename "$log_file")"
