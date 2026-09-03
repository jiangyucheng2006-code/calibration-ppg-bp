#!/bin/bash

set -euo pipefail

seed="${1:-20260824}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest_work="$work_root/outputs/submission_manifests/round10_${timestamp}.tsv"
manifest_nas="$archive_root/outputs/submission_manifests/round10_${timestamp}.tsv"

cd "$project_root"
mkdir -p "$(dirname "$manifest_work")" "$(dirname "$manifest_nas")"
printf 'stage\tmethod\tlabel\tjob_id\trun\tseed\n' > "$manifest_work"

population_job="$(sbatch --parsable scripts/sbatch_round10_population.sh "$seed")"
population_run="$work_root/outputs/event120-v1_round10_internal_population_seed${seed}_job${population_job}"
printf 'base\tpopulation\tInternal population base\t%s\t%s\t%s\n' \
  "$population_job" "$population_run" "$seed" >> "$manifest_work"

qgh_job="$(sbatch --parsable --dependency="afterok:${population_job}" \
  scripts/sbatch_round10_qgh.sh "$population_run" "$seed")"
qgh_run="$work_root/outputs/event120-v1_round10_internal_qgh_seed${seed}_job${qgh_job}"
printf 'base\tqgh\tInternal Quality Gate + Huber base\t%s\t%s\t%s\n' \
  "$qgh_job" "$qgh_run" "$seed" >> "$manifest_work"

prepare_job="$(sbatch --parsable --dependency="afterok:${qgh_job}" \
  scripts/sbatch_round10_prepare.sh "$population_run" "$qgh_run")"
prepared="$work_root/outputs/event120-v1_round10_prepared_job${prepare_job}"
printf 'prepare\tprepared\tRound-10 raw-waveform index and frozen base predictions\t%s\t%s\t%s\n' \
  "$prepare_job" "$prepared" "$seed" >> "$manifest_work"

declare -a methods=(
  frozen_reference
  projection_only
  last_block
  last_two_blocks
  full_encoder
  last_block_direction
  last_block_temporal
  last_block_adaptive
  last_block_joint
)
declare -a labels=(
  "T10-0 Frozen-encoder reference"
  "T10-1 Projection-only adaptation"
  "T10-2 Last-block adaptation"
  "T10-3 Last-two-block adaptation"
  "T10-4 Full-encoder adaptation"
  "T10-5 Last block + pair direction"
  "T10-6 Last block + temporal consistency"
  "T10-7 Last block + adaptive fusion"
  "T10-8 Last block + direction + temporal"
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
    scripts/sbatch_round10_candidate.sh \
    "${methods[$index]}" "$prepared" "$population_run" "$seed")"
  run="$work_root/outputs/event120-v1_round10_internal_${methods[$index]}_seed${seed}_job${job}"
  jobs+=("$job")
  specifications+=("${labels[$index]}=$run")
  printf 'candidate\t%s\t%s\t%s\t%s\t%s\n' \
    "${methods[$index]}" "${labels[$index]}" "$job" "$run" "$seed" \
    >> "$manifest_work"
done

dependency="$(IFS=:; echo "${jobs[*]}")"
report_job="$(sbatch --parsable --dependency="afterok:${dependency}" \
  scripts/sbatch_round10_report.sh "$seed" "${specifications[@]}")"
report_run="$work_root/outputs/event120-v1_round10_internal_report_seed${seed}_job${report_job}"
printf 'report\treport\tRound-10 internal report\t%s\t%s\t%s\n' \
  "$report_job" "$report_run" "$seed" >> "$manifest_work"

cp "$manifest_work" "$manifest_nas"
cmp -s "$manifest_work" "$manifest_nas"
printf 'ROUND10_POPULATION_JOB=%s\n' "$population_job"
printf 'ROUND10_QGH_JOB=%s\n' "$qgh_job"
printf 'ROUND10_PREPARE_JOB=%s\n' "$prepare_job"
printf 'ROUND10_CANDIDATE_JOBS=%s\n' "${jobs[*]}"
printf 'ROUND10_REPORT_JOB=%s\n' "$report_job"
printf 'ROUND10_SUBMISSION_MANIFEST=%s\n' "$manifest_work"
