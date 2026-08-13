#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --job-name=ppg_finish_005
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/pulsedb_finish_005_%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/pulsedb_finish_005_%j.err

set -euo pipefail

project="/home/${USER}/work/ppg_bp"
archive="/home/${USER}/nas/ppg_bp"
code="$project/code/pulsedb_fewshot"
environment="$project/envs/train"
manifest="$project/data/manifests/full_acquisition/box_signed_manifest.csv"
metadata="$project/data/manifests/full_acquisition/pulsedb_mimic_005_signed.json"
destination="$archive/data/raw/PulseDB_v2/downloads/box_parts"
target="$destination/PulseDB_MIMIC.zip.005"
chunk_directory="$archive/data/raw/PulseDB_v2/downloads/.PulseDB_MIMIC.zip.005.chunks"
segment_status="$project/data/manifests/full_acquisition/pulsedb_mimic_005_segmented_status.json"
segment_logs="$project/logs/pulsedb_mimic_005_chunks"
global_status="$project/data/manifests/full_acquisition/pulsedb_v2_download_status.json"
global_logs="$project/logs/pulsedb_v2_download_parts"
nas_status="$archive/data/manifests/full_acquisition"

mkdir -p "$destination" "$segment_logs" "$nas_status"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$environment"

if [[ -n "${PPG_PRIOR_JOB_ID:-}" && -f "$segment_status" ]]; then
  prior_status="$project/data/manifests/full_acquisition/pulsedb_mimic_005_segmented_status_job${PPG_PRIOR_JOB_ID}_final.json"
  cp "$segment_status" "$prior_status"
  cp "$prior_status" "$nas_status/$(basename "$prior_status")"
fi

echo "FINAL_PART_SEGMENTED_START=yes"
echo "TIME=$(date -Is)"
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

python "$code/scripts/download_box_file_segmented.py" \
  --metadata "$metadata" \
  --destination "$target" \
  --chunk-directory "$chunk_directory" \
  --status "$segment_status" \
  --log-directory "$segment_logs" \
  --workers 128 \
  --chunk-size 67108864

echo "FINAL_PART_SEGMENTED_VERIFIED=yes"

python "$code/scripts/download_pulsedb_box_archives.py" \
  --manifest "$manifest" \
  --destination "$destination" \
  --status "$global_status" \
  --log-directory "$global_logs" \
  --workers 26

cp "$segment_status" "$nas_status/pulsedb_mimic_005_segmented_status.json"
cp "$global_status" "$nas_status/pulsedb_v2_download_status.json"
cmp -s "$segment_status" "$nas_status/pulsedb_mimic_005_segmented_status.json"
cmp -s "$global_status" "$nas_status/pulsedb_v2_download_status.json"

echo "PHASE3_ARCHIVE_DOWNLOAD_COMPLETE=yes"
echo "ALL_OFFICIAL_SHA1_PASSED=yes"
