#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 POPULATION_RUN SEED [GPU_A] [GPU_B]" >&2
  exit 2
fi

population_run="$1"
seed="$2"
gpu_a="${3:-rtx_5080}"
gpu_b="${4:-rtx_5070_ti}"
test "$seed" -eq 20260904
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must identify an immutable source snapshot}"
project_root="$(readlink -f "$project_root")"
scripts="$project_root/scripts"
manifest_root="$work_root/outputs/submission_manifests"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest="$manifest_root/fewshot_adapter_bank_seed${seed}_${timestamp}.tsv"

test -d "$project_root"
test ! -w "$project_root"
test -f "$population_run/run.json"
test -f "$population_run/best.pt"
mkdir -p "$manifest_root" "$archive_root/outputs/submission_manifests"

cache_job="$(sbatch --parsable \
  --gres="gpu:${gpu_a}:1" \
  "$scripts/sbatch_fewshot_adapter_bank_cache.sh" \
  "$population_run" "$seed")"
cache_job="${cache_job%%;*}"
cache_dir="$work_root/outputs/event120-v1_fewshot_adapter_bank_cache_seed${seed}_job${cache_job}"

settings=(m0_reference)
bases=(0)
modes=(none)
for basis in 5 10 15 20 25 30; do
  for mode in top5 dense; do
    settings+=("bank$(printf '%02d' "$basis")_${mode}")
    bases+=("$basis")
    modes+=("$mode")
  done
done

jobs=()
runs=()
choose_gpu() {
  local basis="$1"
  local mode="$2"
  if [[ "$basis" == "0" ]]; then
    printf '%s\n' "$gpu_b"
  elif [[ "$basis" == "5" ]]; then
    # Keep the mathematically equivalent M=5 jobs on one device so that their
    # prediction difference is a clean implementation-consistency check.
    printf '%s\n' "$gpu_a"
  elif (( (basis / 5) % 2 == 0 )); then
    [[ "$mode" == "top5" ]] && printf '%s\n' "$gpu_a" || printf '%s\n' "$gpu_b"
  else
    [[ "$mode" == "top5" ]] && printf '%s\n' "$gpu_b" || printf '%s\n' "$gpu_a"
  fi
}

for index in "${!settings[@]}"; do
  setting="${settings[$index]}"
  basis="${bases[$index]}"
  mode="${modes[$index]}"
  gpu="$(choose_gpu "$basis" "$mode")"
  job="$(sbatch --parsable \
    --dependency="afterok:${cache_job}" \
    --gres="gpu:${gpu}:1" \
    "$scripts/sbatch_fewshot_adapter_bank_candidate.sh" \
    "$setting" "$basis" "$mode" "$cache_dir" "$population_run" "$seed")"
  job="${job%%;*}"
  jobs+=("$job")
  runs+=("${setting}=$work_root/outputs/event120-v1_${setting}_seed${seed}_job${job}")
done

dependency="$(IFS=:; echo "${jobs[*]}")"
report_job="$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  "$scripts/sbatch_fewshot_adapter_bank_report.sh" \
  "$seed" "${runs[@]}")"
report_job="${report_job%%;*}"

{
  printf 'role\tsetting\tbasis_count\trouting_mode\tjob_id\tdependency\tgpu\tartifact\n'
  printf 'cache\tfeature_cache\tNA\tNA\t%s\tnone\t%s\t%s\n' \
    "$cache_job" "$gpu_a" "$cache_dir"
  for index in "${!settings[@]}"; do
    setting="${settings[$index]}"
    basis="${bases[$index]}"
    mode="${modes[$index]}"
    gpu="$(choose_gpu "$basis" "$mode")"
    printf 'train\t%s\t%s\t%s\t%s\tafterok:%s\t%s\t%s\n' \
      "$setting" "$basis" "$mode" "${jobs[$index]}" "$cache_job" "$gpu" "${runs[$index]#*=}"
  done
  printf 'report\tall\tNA\tNA\t%s\tafterok:%s\tCPU\t%s\n' \
    "$report_job" "$dependency" \
    "$work_root/outputs/event120-v1_fewshot_adapter_bank_report_seed${seed}_job${report_job}"
} > "$manifest"
rsync -a --no-perms --no-owner --no-group "$manifest" \
  "$archive_root/outputs/submission_manifests/"

echo "FEWSHOT_ADAPTER_BANK_SUBMITTED=yes"
echo "MANIFEST=$manifest"
echo "CACHE_JOB=$cache_job"
echo "TRAIN_JOBS=${jobs[*]}"
echo "REPORT_JOB=$report_job"
