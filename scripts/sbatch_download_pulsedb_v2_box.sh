#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --job-name=ppg_full_download
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/pulsedb_v2_full_download_%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/pulsedb_v2_full_download_%j.err

set -euo pipefail
umask 077

project_root="/home/${USER}/work/ppg_bp"
archive_root="/home/${USER}/nas/ppg_bp"
code_root="${project_root}/code/pulsedb_fewshot"
environment_root="${project_root}/envs/train"
manifest="${project_root}/data/manifests/full_acquisition/box_signed_manifest.csv"
destination="${archive_root}/data/raw/PulseDB_v2/downloads/box_parts"
status_path="${project_root}/data/manifests/full_acquisition/pulsedb_v2_download_status.json"
part_log_directory="${project_root}/logs/pulsedb_v2_download_parts"
archive_status_directory="${archive_root}/data/manifests/full_acquisition"

mkdir -p \
    "${destination}" \
    "$(dirname "${status_path}")" \
    "${part_log_directory}" \
    "${archive_status_directory}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "${environment_root}"

echo "PHASE3_FULL_DOWNLOAD_START=yes"
echo "TIME=$(date -Is)"
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "MANIFEST=${manifest}"
echo "DESTINATION=${destination}"
echo "STATUS=${status_path}"

# Box CDN URLs are short lived. Start every archive task immediately so queued
# parts do not inherit an already-expired signed URL. Completed parts are still
# SHA-1 checked and skipped by the downloader.
python "${code_root}/scripts/download_pulsedb_box_archives.py" \
    --manifest "${manifest}" \
    --destination "${destination}" \
    --status "${status_path}" \
    --log-directory "${part_log_directory}" \
    --workers 26

cp "${status_path}" "${archive_status_directory}/pulsedb_v2_download_status.json"
cmp -s \
    "${status_path}" \
    "${archive_status_directory}/pulsedb_v2_download_status.json"

echo "PHASE3_ARCHIVE_DOWNLOAD_COMPLETE=yes"
echo "ALL_OFFICIAL_SHA1_PASSED=yes"
