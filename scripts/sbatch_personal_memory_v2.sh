#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=72:00:00
#SBATCH --output=/home/jiangyu.cheng/work/ppg_bp/logs/personal_memory_v2_%j.log
set -euo pipefail
umask 077
ppg_project="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_project"
ppg_work="/home/jiangyu.cheng/work/ppg_bp"
ppg_nas="/home/jiangyu.cheng/nas/ppg_bp"
ppg_python="$ppg_work/envs/train/bin/python"
export PYTHONPATH="$ppg_project/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
ppg_stage="$1"
ppg_mode="$2"
ppg_output="$3"
shift 3
[[ "$ppg_output" == "$ppg_work/outputs/personal-memory-v2_"* ]]
[[ ! -e "$ppg_output" ]]
[[ "$ppg_mode" == random_disjoint || "$ppg_mode" == chronological_blocked || "$ppg_mode" == both ]]
ppg_archive() {
  local code=$?
  trap - EXIT
  if [[ -d "$ppg_output" ]]; then
    mkdir -p "$ppg_nas/outputs/$(basename "$ppg_output")"
    rsync -a "$ppg_output/" "$ppg_nas/outputs/$(basename "$ppg_output")/" || code=1
  fi
  cp "$ppg_work/logs/personal_memory_v2_${SLURM_JOB_ID}.log" "$ppg_nas/logs/" || true
  exit "$code"
}
trap ppg_archive EXIT
cd "$ppg_project"
ppg_v1="$ppg_work/outputs/personal-memory-v1_20260907-084550"
ppg_cache="${ppg_v1}_${ppg_mode}_cache"
ppg_relation="${ppg_v1}_${ppg_mode}_pair_distance_blend"
ppg_gate="${ppg_v1}_both_gate/gate.json"
case "$ppg_stage" in
 smoke)
  mkdir -p "$ppg_output"
  "$ppg_python" -m pytest -q -p no:cacheprovider tests/test_personal_memory*.py
  "$ppg_python" -m pulsedb_fewshot.personal_memory_matched --synthetic-smoke --split-mode "$ppg_mode" --device cuda --output "$ppg_output/e1"
  "$ppg_python" -m pulsedb_fewshot.personal_memory_metric --synthetic-smoke --split-mode "$ppg_mode" --device cuda --output "$ppg_output/e2"
  "$ppg_python" -m pulsedb_fewshot.personal_memory_matched --cache-dir "$ppg_cache" --gate-manifest "$ppg_gate" --smoke-check --device cuda --output "$ppg_output/e1_real"
  "$ppg_python" -m pulsedb_fewshot.personal_memory_metric --cache-dir "$ppg_cache" --smoke-check --device cuda --output "$ppg_output/e2_real"
  ppg_source_job=1500
  if [[ "$ppg_mode" == chronological_blocked ]]; then ppg_source_job=1507; fi
  ppg_source="$ppg_work/outputs/lora-prs-continuation-v1_${ppg_mode}_lora_continue_seed20260907_job${ppg_source_job}"
  "$ppg_python" -m pulsedb_fewshot.personal_memory_smoke --run "$ppg_source" --store-root "$ppg_work/data/processed/development-calbased-analogue-v1" --split-mode "$ppg_mode" --output "$ppg_output/real_data_smoke.json"
  ;;
 diagnostics)
  "$ppg_python" -m pulsedb_fewshot.personal_memory_diagnostics --cache-dir "$ppg_cache" --source-relation-run "$ppg_relation" --output "$ppg_output" --device cuda
  ;;
 e1)
  "$ppg_python" -m pulsedb_fewshot.personal_memory_matched --cache-dir "$ppg_cache" --gate-manifest "$ppg_gate" --output "$ppg_output" --device cuda --seed 20260907
  ;;
 e2)
  "$ppg_python" -m pulsedb_fewshot.personal_memory_metric --cache-dir "$ppg_cache" --output "$ppg_output" --device cuda --seed 20260907
  ;;
 report)
  "$ppg_python" -m pulsedb_fewshot.personal_memory_v2_report --diagnostics "$1" --e1 "$2" --e2 "$3" --control "${ppg_v1}_${ppg_mode}_lora_continued_control" --split-mode "$ppg_mode" --output "$ppg_output"
  ;;
 final)
  "$ppg_python" -m pulsedb_fewshot.personal_memory_v2_report --mode-reports "$1" "$2" --output "$ppg_output"
  ;;
 *) echo 'Unknown v2 stage' >&2; exit 2 ;;
esac
