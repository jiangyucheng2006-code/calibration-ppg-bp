#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/jiangyu.cheng/work/ppg_bp/logs/official_content_recovery_%j.log
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
  for name in manifest.json postprepare_audit.json official_duplicate_audit.json; do
    if [[ -f "$ppg_store/$name" ]]; then cp "$ppg_store/$name" "$ppg_nas/data/manifests/$(basename "$ppg_store")/"; fi
  done
  cp "$ppg_work/logs/official_content_recovery_${SLURM_JOB_ID}.log" "$ppg_nas/logs/" || true
  exit "$code"
}
trap archive EXIT
cd "$ppg_project"
"$ppg_python" -m pulsedb_fewshot.official_calbased_recover \
  --failed-store "$ppg_work/data/processed/pulsedb-official-calbased-v1_20260908-152000" \
  --output "$ppg_store" --info-root "$ppg_work/data/manifests/official_info_20260908" \
  --duplicate-audit "$ppg_work/data/manifests/official_content_duplicates_20260908.json" \
  --content-policy official-source-duplicates-retained-20260908
"$ppg_python" "$ppg_project/scripts/audit_official_prepared_store.py" \
  --store-root "$ppg_store" --output "$ppg_store/postprepare_audit.json"
