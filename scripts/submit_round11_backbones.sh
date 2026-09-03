#!/bin/bash

set -euo pipefail

seed="${1:-20260825}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest_work="$work_root/outputs/submission_manifests/round11a_backbones_${timestamp}.tsv"
manifest_nas="$archive_root/outputs/submission_manifests/round11a_backbones_${timestamp}.tsv"

cd "$project_root"
mkdir -p "$(dirname "$manifest_work")" "$(dirname "$manifest_nas")"
printf 'stage\tbackbone\tjob_id\trun\tseed\n' > "$manifest_work"

backbones=(resnet_small resnet_deep inception_time patch_transformer conformer)
evaluation_jobs=()
run_specs=()
for index in "${!backbones[@]}"; do
  backbone="${backbones[$index]}"
  gres="gpu:rtx_5080:1"
  if (( index % 2 == 1 )); then
    gres="gpu:rtx_5070_ti:1"
  fi
  population_job="$(sbatch --parsable --gres="$gres" \
    scripts/sbatch_round11_backbone_population.sh "$backbone" "$seed")"
  population_run="$work_root/outputs/event120-v1_round11a_${backbone}_population_seed${seed}_job${population_job}"
  printf 'population\t%s\t%s\t%s\t%s\n' \
    "$backbone" "$population_job" "$population_run" "$seed" >> "$manifest_work"

  qgh_job="$(sbatch --parsable --dependency="afterok:${population_job}" --gres="$gres" \
    scripts/sbatch_round11_backbone_qgh.sh "$backbone" "$population_run" "$seed")"
  qgh_run="$work_root/outputs/event120-v1_round11a_${backbone}_qgh_seed${seed}_job${qgh_job}"
  printf 'qgh\t%s\t%s\t%s\t%s\n' \
    "$backbone" "$qgh_job" "$qgh_run" "$seed" >> "$manifest_work"

  evaluation_job="$(sbatch --parsable --dependency="afterok:${qgh_job}" --gres="$gres" \
    scripts/sbatch_round11_backbone_evaluate.sh \
    "$backbone" "$population_run" "$qgh_run" "$seed")"
  evaluation_run="$work_root/outputs/event120-v1_round11a_${backbone}_evaluation_seed${seed}_job${evaluation_job}"
  printf 'evaluation\t%s\t%s\t%s\t%s\n' \
    "$backbone" "$evaluation_job" "$evaluation_run" "$seed" >> "$manifest_work"
  evaluation_jobs+=("$evaluation_job")
  run_specs+=("${backbone}=${evaluation_run}")
done

dependency="$(IFS=:; echo "${evaluation_jobs[*]}")"
report_job="$(sbatch --parsable --dependency="afterok:${dependency}" \
  scripts/sbatch_round11_backbone_report.sh "$seed" "${run_specs[@]}")"
report_run="$work_root/outputs/event120-v1_round11a_backbone_report_seed${seed}_job${report_job}"
printf 'report\tall\t%s\t%s\t%s\n' \
  "$report_job" "$report_run" "$seed" >> "$manifest_work"

cp "$manifest_work" "$manifest_nas"
cmp -s "$manifest_work" "$manifest_nas"
printf 'ROUND11A_REPORT_JOB=%s\n' "$report_job"
printf 'ROUND11A_REPORT_RUN=%s\n' "$report_run"
printf 'ROUND11A_SUBMISSION_MANIFEST=%s\n' "$manifest_work"
