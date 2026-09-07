#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=72:00:00
#SBATCH --output=/home/jiangyu.cheng/work/ppg_bp/logs/personal_memory_%j.log
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
[[ "$ppg_output" == "$ppg_work/outputs/personal-memory-v1_"* ]]
[[ ! -e "$ppg_output" ]]
ppg_archive() {
  local code=$?
  trap - EXIT
  if [[ -d "$ppg_output" ]]; then
    mkdir -p "$ppg_nas/outputs/$(basename "$ppg_output")"
    rsync -a "$ppg_output/" "$ppg_nas/outputs/$(basename "$ppg_output")/" || code=1
  fi
  cp "$ppg_work/logs/personal_memory_${SLURM_JOB_ID}.log" "$ppg_nas/logs/" || true
  exit "$code"
}
trap ppg_archive EXIT
cd "$ppg_project"
ppg_store="$ppg_work/data/processed/development-calbased-analogue-v1"
ppg_source_job=1500
if [[ "$ppg_mode" == chronological_blocked ]]; then ppg_source_job=1507; fi
ppg_source="$ppg_work/outputs/lora-prs-continuation-v1_${ppg_mode}_lora_continue_seed20260907_job${ppg_source_job}"
case "$ppg_stage" in
 smoke)
  mkdir -p "$ppg_output"
  "$ppg_python" -m pytest -q -p no:cacheprovider tests/test_personal_memory*.py
  "$ppg_python" -m pulsedb_fewshot.personal_memory_smoke --run "$ppg_source" --store-root "$ppg_store" --split-mode "$ppg_mode" --output "$ppg_output/smoke.json"
  ;;
 cache)
  "$ppg_python" -m pulsedb_fewshot.personal_memory_export --run "$ppg_source" --store-root "$ppg_store" --split-mode "$ppg_mode" --output "$ppg_output"
  "$ppg_python" -m pulsedb_fewshot.personal_memory_prepare --cache-dir "$ppg_output"
  ;;
 gate)
  mkdir -p "$ppg_output"
  "$ppg_python" -m pulsedb_fewshot.personal_memory_stage_gate --cache-dirs "$1" "$2" --output "$ppg_output/gate.json"
  ;;
 train)
  ppg_candidate="$1"; ppg_cache="$2"; ppg_gate="$3"
  if [[ "$ppg_candidate" == lora_continued_control ]]; then
    "$ppg_python" -m pulsedb_fewshot.personal_memory_control --run "$ppg_source" --store-root "$ppg_store" --split-mode "$ppg_mode" --output "$ppg_output" --cache-dir "$ppg_cache" --gate-manifest "$ppg_gate" --require-cuda
  else
    "$ppg_python" -m pulsedb_fewshot.personal_memory_train --candidate "$ppg_candidate" --output "$ppg_output" --cache-dir "$ppg_cache" --gate-manifest "$ppg_gate" --device cuda
  fi
  ;;
 report)
  ppg_cache="$1"; shift
  "$ppg_python" -m pulsedb_fewshot.personal_memory_report --cache-dir "$ppg_cache" --runs "$@" --output "$ppg_output" --split-mode "$ppg_mode"
  ;;
 final)
  "$ppg_python" -m pulsedb_fewshot.personal_memory_report --mode-reports "$@" --output "$ppg_output"
  ;;
 *) echo "Unsupported stage" >&2; exit 2 ;;
esac
