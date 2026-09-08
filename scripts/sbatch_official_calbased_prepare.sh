#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jiangyu.cheng/work/ppg_bp/logs/official_calbased_prepare_%j.log
set -euo pipefail
umask 077
ppg_project="${PPG_PROJECT_ROOT:?immutable code snapshot required}"
test ! -w "$ppg_project"
ppg_work=/home/jiangyu.cheng/work/ppg_bp
ppg_nas=/home/jiangyu.cheng/nas/ppg_bp
ppg_python="$ppg_work/envs/train/bin/python"
ppg_store="${1:?new output store required}"
[[ "$ppg_store" == "$ppg_work/data/processed/pulsedb-official-calbased-v1_"* ]]
[[ ! -e "$ppg_store" ]]
export PYTHONPATH="$ppg_project/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
archive() {
  local code=$?
  trap - EXIT
  mkdir -p "$ppg_nas/data/manifests/$(basename "$ppg_store")" "$ppg_nas/logs"
  if [[ -f "$ppg_store/manifest.json" ]]; then
    cp "$ppg_store/manifest.json" "$ppg_nas/data/manifests/$(basename "$ppg_store")/"
  fi
  cp "$ppg_work/logs/official_calbased_prepare_${SLURM_JOB_ID}.log" "$ppg_nas/logs/" || true
  exit "$code"
}
trap archive EXIT
cd "$ppg_project"
"$ppg_python" -m unittest discover -s tests -p test_official_calbased_data.py -v
"$ppg_python" -m pulsedb_fewshot.official_calbased_data \
  --info-root "$ppg_work/data/manifests/official_info_20260908" \
  --segment-index "$ppg_work/data/manifests/pulsedb_v2_full_cohort/pulsedb_v2_full_segment_index.parquet" \
  --legacy-subjects "$ppg_work/data/manifests/pulsedb_v2_full_cohort/subject_splits.csv" \
  --mimic-root "$ppg_work/data/raw/PulseDB_v2/Segment_Files/PulseDB_MIMIC" \
  --vital-root "$ppg_work/data/raw/PulseDB_v2/Segment_Files/PulseDB_Vital" \
  --output "$ppg_store" --workers 2 --shards 32
