#!/bin/bash
# Data movement only: no GPU is requested. Originals remain on NAS.
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jiangyu.cheng/work/ppg_bp/logs/official_calbased_stage_%j.log
set -euo pipefail
umask 077
ppg_project="${PPG_PROJECT_ROOT:?immutable code snapshot required}"
test ! -w "$ppg_project"
ppg_work=/home/jiangyu.cheng/work/ppg_bp
ppg_nas=/home/jiangyu.cheng/nas/ppg_bp
ppg_receipt="${1:?staging receipt path required}"
[[ "$ppg_receipt" == "$ppg_work/data/manifests/official_calbased_stage_"*.json ]]
export PYTHONPATH="$ppg_project/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
archive() {
  local code=$?
  trap - EXIT
  mkdir -p "$ppg_nas/data/manifests" "$ppg_nas/logs"
  if [[ -f "$ppg_receipt" ]]; then cp "$ppg_receipt" "$ppg_nas/data/manifests/"; fi
  cp "$ppg_work/logs/official_calbased_stage_${SLURM_JOB_ID}.log" "$ppg_nas/logs/" || true
  exit "$code"
}
trap archive EXIT
cd "$ppg_project"
"$ppg_work/envs/train/bin/python" -m unittest discover -s tests -p test_official_calbased_stage.py -v
"$ppg_work/envs/train/bin/python" -m pulsedb_fewshot.official_calbased_stage \
  --membership "$ppg_work/data/manifests/official_info_20260908/Train_Info_membership.parquet" \
  --segment-index "$ppg_work/data/manifests/pulsedb_v2_full_cohort/pulsedb_v2_full_segment_index.parquet" \
  --source-root "$ppg_nas/data/raw/PulseDB_v2/extraction_staging_20260811/Segment_Files" \
  --target-root "$ppg_work/data/raw/PulseDB_v2/Segment_Files" \
  --receipt "$ppg_receipt" --reserve-gib 64
