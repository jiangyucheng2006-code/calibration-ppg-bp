#!/bin/bash

set -euo pipefail

seed="20260822"
work_root="/home/$USER/work/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
cd "$project_root"

prepare_job="$(sbatch --parsable scripts/sbatch_round8_prepare.sh)"
prepared="$work_root/outputs/event120-v1_round8_prepared_job${prepare_job}"

declare -a methods=(
  pair_delta
  pair_delta_temporal
  pair_delta_range
  pair_delta_temporal_range
  pair_delta_temporal_range_physiology
)
declare -a labels=(
  "R8-1 Pairwise delta"
  "R8-2 Pairwise delta + causal time"
  "R8-3 Pairwise delta + range auxiliary"
  "R8-4 Pairwise delta + causal time + range"
  "R8-5 R8-4 + generic PPG change features"
)
declare -a jobs=()
declare -a specifications=()
for index in "${!methods[@]}"; do
  gres="gpu:rtx_5080:1"
  if (( index % 2 == 1 )); then
    gres="gpu:rtx_5070_ti:1"
  fi
  job="$(sbatch --parsable \
    --dependency="afterok:${prepare_job}" \
    --gres="$gres" \
    scripts/sbatch_round8_calibration_relative.sh \
    "${methods[$index]}" "$prepared" "$seed")"
  jobs+=("$job")
  run="$work_root/outputs/event120-v1_round8_${methods[$index]}_seed${seed}_job${job}"
  specifications+=("${labels[$index]}=$run")
done

population128_job="$(sbatch --parsable \
  --gres=gpu:rtx_5070_ti:1 \
  scripts/sbatch_round8_population128.sh "$seed")"
population128="$work_root/outputs/event120-v1_round8_population128_seed${seed}_job${population128_job}/best.pt"

direct_demo_job="$(sbatch --parsable \
  --gres=gpu:rtx_5080:1 \
  scripts/sbatch_round8_qgate_huber.sh \
  demographics_direct256 \
  "$work_root/outputs/event120-v1_repeat5-v1_population_seed20260813_job782/best.pt" \
  256 direct "$seed")"
direct_demo_run="$work_root/outputs/event120-v1_round8_demographics_direct256_seed${seed}_job${direct_demo_job}"
specifications+=("R8-7 Direct demographics + 256-D=$direct_demo_run")

encoder128_job="$(sbatch --parsable \
  --dependency="afterok:${population128_job}" \
  --gres=gpu:rtx_5070_ti:1 \
  scripts/sbatch_round8_qgate_huber.sh \
  qgate_huber128 "$population128" 128 none "$seed")"
encoder128_run="$work_root/outputs/event120-v1_round8_qgate_huber128_seed${seed}_job${encoder128_job}"
specifications+=("R8-8 Quality Gate + Huber 128-D=$encoder128_run")

dependency_jobs=("${jobs[@]}" "$direct_demo_job" "$encoder128_job")
dependency="$(IFS=:; echo "${dependency_jobs[*]}")"
report_job="$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  scripts/sbatch_round8_report.sh "$seed" "${specifications[@]}")"

printf 'ROUND8_PREPARE_JOB=%s\n' "$prepare_job"
printf 'ROUND8_RELATIVE_JOBS=%s\n' "${jobs[*]}"
printf 'ROUND8_POPULATION128_JOB=%s\n' "$population128_job"
printf 'ROUND8_DIRECT_DEMOGRAPHICS_JOB=%s\n' "$direct_demo_job"
printf 'ROUND8_QGATE_HUBER128_JOB=%s\n' "$encoder128_job"
printf 'ROUND8_REPORT_JOB=%s\n' "$report_job"
