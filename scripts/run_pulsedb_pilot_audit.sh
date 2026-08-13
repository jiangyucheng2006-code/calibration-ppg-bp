#!/usr/bin/env bash

set -euo pipefail

ppg_user="${USER:?USER is required}"
ppg_work_root="/home/${ppg_user}/work/ppg_bp"
ppg_nas_root="/home/${ppg_user}/nas/ppg_bp"
ppg_project_root="${ppg_work_root}/code/pulsedb_fewshot"
ppg_env="${ppg_work_root}/envs/train"
ppg_data_root="${ppg_work_root}/data/raw/PulseDB_v2/Segment_Files"
ppg_manifest="${ppg_work_root}/data/manifests/pilot_download_manifest.csv"
ppg_output="${ppg_work_root}/data/manifests/pulsedb_v2_pilot"
ppg_nas_output="${ppg_nas_root}/data/manifests/pulsedb_v2_pilot"
ppg_timestamp="$(date +%Y%m%d-%H%M%S)"
ppg_log="${ppg_work_root}/logs/pulsedb_v2_pilot_audit_${ppg_timestamp}.log"
ppg_nas_log_dir="${ppg_nas_root}/logs"

mkdir -p \
  "${ppg_work_root}/logs" \
  "${ppg_nas_log_dir}" \
  "${ppg_output}" \
  "${ppg_nas_output}"

ppg_archive_log() {
  if [[ -f "${ppg_log}" ]]; then
    # The Synology-backed NAS does not support preserving every POSIX metadata
    # field exposed by the hot NVMe workspace.  Archive log content without
    # requesting metadata preservation, then verify it byte-for-byte.
    cp "${ppg_log}" "${ppg_nas_log_dir}/"
    cmp -s "${ppg_log}" "${ppg_nas_log_dir}/$(basename "${ppg_log}")"
  fi
}
trap ppg_archive_log EXIT

{
  echo "===== PULSEDB V2 SOURCE-BALANCED PILOT AUDIT ====="
  echo "TIME=$(date -Is)"
  echo "HOST=$(hostname)"
  echo "USER=${ppg_user}"
  echo "PROJECT_ROOT=${ppg_project_root}"
  echo "DATA_ROOT=${ppg_data_root}"
  echo "MANIFEST=${ppg_manifest}"
  echo

  test -r /opt/conda/etc/profile.d/conda.sh
  test -d "${ppg_env}"
  test -d "${ppg_project_root}"
  test -f "${ppg_manifest}"

  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${ppg_env}"
  cd "${ppg_project_root}"

  echo "PYTHON=$(command -v python)"
  python --version
  python -m pip check

  ppg_mimic_count="$(find "${ppg_data_root}/PulseDB_MIMIC" -maxdepth 1 -type f -name '*.mat' | wc -l)"
  ppg_vital_count="$(find "${ppg_data_root}/PulseDB_Vital" -maxdepth 1 -type f -name '*.mat' | wc -l)"
  ppg_total_count="$((ppg_mimic_count + ppg_vital_count))"
  echo "MIMIC_FILE_COUNT=${ppg_mimic_count}"
  echo "VITAL_FILE_COUNT=${ppg_vital_count}"
  echo "TOTAL_FILE_COUNT=${ppg_total_count}"
  test "${ppg_mimic_count}" -eq 5
  test "${ppg_vital_count}" -eq 5
  test "${ppg_total_count}" -eq 10

  echo "===== PROJECT TESTS ====="
  python -m pytest -q

  echo "===== REAL PILOT AUDIT ====="
  python -m pulsedb_fewshot.pilot_audit \
    --data-root "${ppg_data_root}" \
    --output "${ppg_output}" \
    --download-manifest "${ppg_manifest}" \
    --expected-per-source 5

  echo "===== REPORT GATE ====="
  python - "${ppg_output}/pilot_audit.json" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
assert report["status"] in {"pass", "pass_with_warnings"}, report["status"]
assert report["n_files"] == 10, report["n_files"]
assert report["source_counts"] == {"PulseDB_MIMIC": 5, "PulseDB_Vital": 5}
assert report["n_subject_uids"] == 10, report["n_subject_uids"]
assert not report["required_failures"], report["required_failures"]
print(f"REPORT_STATUS={report['status']}")
print(f"REPORT_SEGMENTS={report['n_segments']}")
print(f"REPORT_SUBJECT_UIDS={report['n_subject_uids']}")
print("REPORT_GATE=pass")
PY

  echo "===== ARCHIVE OUTPUTS ====="
  rsync -a "${ppg_output}/" "${ppg_nas_output}/"
  diff -qr "${ppg_output}" "${ppg_nas_output}"
  echo "ARCHIVE_IDENTICAL=yes"

  echo "===== ARTIFACT HASHES ====="
  find "${ppg_output}" -maxdepth 1 -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum

  conda env export -p "${ppg_env}" > "${ppg_nas_root}/environment.yml"
  echo "ENVIRONMENT_REEXPORTED=yes"
  echo "PULSEDB_SERVER_PILOT_AUDIT_COMPLETE=yes"
  echo "WORK_OUTPUT=${ppg_output}"
  echo "NAS_OUTPUT=${ppg_nas_output}"
  echo "WORK_LOG=${ppg_log}"
  echo "NAS_LOG=${ppg_nas_log_dir}/$(basename "${ppg_log}")"
} 2>&1 | tee "${ppg_log}"
