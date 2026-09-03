#!/bin/bash

set -euo pipefail

seed="${1:-20260827}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:-$work_root/code/pulsedb_fewshot}"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest_work="$work_root/outputs/submission_manifests/round13_capacity_backbones_${timestamp}.tsv"
manifest_nas="$archive_root/outputs/submission_manifests/round13_capacity_backbones_${timestamp}.tsv"

cd "$project_root"
mkdir -p "$(dirname "$manifest_work")" "$(dirname "$manifest_nas")"
printf 'stage\tbackbone\tjob_id\trun\tseed\tgres\tmicrobatch\taccumulation\teffective_batch\n' > "$manifest_work"

backbones=(
  resnet_small
  resnet_depth2
  resnet_wide1p5
  inception_time
  inception_time_wide
  patch_transformer
  patch_transformer_deep
  patch_transformer_wide
  patch_transformer_highres
  patch_transformer_longpatch
  conformer
  conformer_large
  convnext_1d
)
population_batches=(32 32 32 32 32 32 32 32 32 32 32 32 32)
qgh_batches=(16 16 16 16 16 16 16 16 16 16 16 16 16)
evaluation_batches=(512 256 256 512 256 128 96 64 64 128 64 16 64)
population_accumulations=(4 4 4 4 4 4 4 4 4 4 4 4 4)
qgh_accumulations=(4 4 4 4 4 4 4 4 4 4 4 4 4)

evaluation_jobs=()
run_specs=()
all_jobs=()
for index in "${!backbones[@]}"; do
  backbone="${backbones[$index]}"
  gres="gpu:rtx_5080:1"
  if (( index % 2 == 1 )); then
    gres="gpu:rtx_5070_ti:1"
  fi
  population_batch="${population_batches[$index]}"
  qgh_batch="${qgh_batches[$index]}"
  evaluation_batch="${evaluation_batches[$index]}"
  population_accumulation="${population_accumulations[$index]}"
  qgh_accumulation="${qgh_accumulations[$index]}"

  population_job="$(sbatch --parsable --gres="$gres" \
    --export=ALL,PPG_PROJECT_ROOT="$project_root" \
    scripts/sbatch_round13_capacity_population.sh \
    "$backbone" "$seed" "$population_batch" "$population_accumulation")"
  population_run="$work_root/outputs/event120-v1_round13_${backbone}_population_seed${seed}_job${population_job}"
  printf 'population\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$backbone" "$population_job" "$population_run" "$seed" "$gres" \
    "$population_batch" "$population_accumulation" 128 >> "$manifest_work"
  all_jobs+=("$population_job")

  qgh_job="$(sbatch --parsable --dependency="afterok:${population_job}" --gres="$gres" \
    --export=ALL,PPG_PROJECT_ROOT="$project_root" \
    scripts/sbatch_round13_capacity_qgh.sh \
    "$backbone" "$population_run" "$seed" "$qgh_batch" "$qgh_accumulation")"
  qgh_run="$work_root/outputs/event120-v1_round13_${backbone}_qgh_seed${seed}_job${qgh_job}"
  printf 'qgh\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$backbone" "$qgh_job" "$qgh_run" "$seed" "$gres" \
    "$qgh_batch" "$qgh_accumulation" 64 >> "$manifest_work"
  all_jobs+=("$qgh_job")

  evaluation_job="$(sbatch --parsable --dependency="afterok:${qgh_job}" --gres="$gres" \
    --export=ALL,PPG_PROJECT_ROOT="$project_root" \
    scripts/sbatch_round13_capacity_evaluate.sh \
    "$backbone" "$population_run" "$qgh_run" "$seed" "$evaluation_batch")"
  evaluation_run="$work_root/outputs/event120-v1_round13_${backbone}_evaluation_seed${seed}_job${evaluation_job}"
  printf 'evaluation\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$backbone" "$evaluation_job" "$evaluation_run" "$seed" "$gres" \
    "$evaluation_batch" 1 "$evaluation_batch" >> "$manifest_work"
  evaluation_jobs+=("$evaluation_job")
  all_jobs+=("$evaluation_job")
  run_specs+=("${backbone}=${evaluation_run}")
done

dependency="$(IFS=:; echo "${evaluation_jobs[*]}")"
report_job="$(sbatch --parsable --dependency="afterok:${dependency}" \
  --export=ALL,PPG_PROJECT_ROOT="$project_root" \
  scripts/sbatch_round13_capacity_report.sh "$seed" "${run_specs[@]}")"
report_run="$work_root/outputs/event120-v1_round13_capacity_report_seed${seed}_job${report_job}"
printf 'report\tall\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$report_job" "$report_run" "$seed" "none" "n/a" "n/a" "n/a" >> "$manifest_work"
all_jobs+=("$report_job")

archive_job="$(sbatch --parsable --dependency="afterok:${report_job}" \
  --export=ALL,PPG_PROJECT_ROOT="$project_root" \
  scripts/sbatch_round13_capacity_archive.sh "$manifest_work")"
printf 'archive\tall\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$archive_job" "$archive_root/logs/round13_capacity_job${archive_job}" \
  "$seed" "none" "n/a" "n/a" "n/a" >> "$manifest_work"
all_jobs+=("$archive_job")

cp "$manifest_work" "$manifest_nas"
cmp -s "$manifest_work" "$manifest_nas"
printf 'ROUND13_JOB_IDS=%s\n' "$(IFS=,; echo "${all_jobs[*]}")"
printf 'ROUND13_REPORT_JOB=%s\n' "$report_job"
printf 'ROUND13_ARCHIVE_JOB=%s\n' "$archive_job"
printf 'ROUND13_REPORT_RUN=%s\n' "$report_run"
printf 'ROUND13_SUBMISSION_MANIFEST=%s\n' "$manifest_work"
