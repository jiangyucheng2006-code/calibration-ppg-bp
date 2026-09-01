#!/bin/bash

set -euo pipefail

seed="${1:-20260902}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable snapshot}"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest_work="$work_root/outputs/submission_manifests/same_subject_combinations_${timestamp}.tsv"
manifest_nas="$archive_root/outputs/submission_manifests/same_subject_combinations_${timestamp}.tsv"

test -d "$project_root"
test ! -w "$project_root"
test -f "$work_root/data/processed/development-calbased-analogue-v1/materialization.json"
mkdir -p "$(dirname "$manifest_work")" "$(dirname "$manifest_nas")"
printf 'kind\tsplit_mode\tcandidate\tjob_id\trun\tseed\n' > "$manifest_work"

random_candidates=(
  lora
  lora_film
  lora_attention
  lora_multi_event
  lora_reliability
  lora_calibration_relative
  lora_film_attention
  lora_film_multi_event
  lora_film_reliability
  lora_film_calibration_relative
  lora_attention_calibration_relative
  lora_multi_event_reliability
  lora_film_attention_calibration_relative
  lora_film_multi_event_reliability
  lora_all_six
)

# The same full matrix receives the stricter temporal control so that this
# terminal screen can select or reject a combination without another search.
chronological_candidates=("${random_candidates[@]}")

submit_split() {
  local split_mode="$1"
  shift
  local candidates=("$@")
  local job_ids=()
  local run_dirs=()
  local index candidate gres job_id run_dir
  for index in "${!candidates[@]}"; do
    candidate="${candidates[$index]}"
    gres="gpu:rtx_5080:1"
    if (( index % 2 == 1 )); then gres="gpu:rtx_5070_ti:1"; fi
    job_id="$(sbatch --parsable \
      --job-name="ppg_ssc_${split_mode:0:1}_${index}" \
      --gres="$gres" \
      --export=ALL,PPG_PROJECT_ROOT="$project_root" \
      "$project_root/scripts/sbatch_same_subject_combination.sh" \
      "$candidate" "$split_mode" "$seed" 64)"
    job_id="${job_id%%;*}"
    run_dir="$work_root/outputs/same-subject-combination-v1_${split_mode}_${candidate}_seed${seed}_job${job_id}"
    printf 'training\t%s\t%s\t%s\t%s\t%s\n' \
      "$split_mode" "$candidate" "$job_id" "$run_dir" "$seed" >> "$manifest_work"
    job_ids+=("$job_id")
    run_dirs+=("$run_dir")
  done
  local dependency report_job report_run
  dependency="$(IFS=:; echo "${job_ids[*]}")"
  report_job="$(sbatch --parsable \
    --dependency="afterok:${dependency}" \
    --export=ALL,PPG_PROJECT_ROOT="$project_root" \
    "$project_root/scripts/sbatch_same_subject_combination_report.sh" \
    "$split_mode" "$seed" "${run_dirs[@]}")"
  report_job="${report_job%%;*}"
  report_run="$work_root/outputs/same-subject-combination-v1_${split_mode}_report_seed${seed}_job${report_job}"
  printf 'report\t%s\tall_combinations\t%s\t%s\t%s\n' \
    "$split_mode" "$report_job" "$report_run" "$seed" >> "$manifest_work"
  if [[ "$split_mode" == "random_disjoint" ]]; then
    random_report_job="$report_job"
    random_report_run="$report_run"
  else
    chronological_report_job="$report_job"
    chronological_report_run="$report_run"
  fi
  echo "${split_mode^^}_TRAINING_JOBS=${job_ids[*]}"
  echo "${split_mode^^}_REPORT_JOB=$report_job"
}

submit_split random_disjoint "${random_candidates[@]}"
submit_split chronological_blocked "${chronological_candidates[@]}"

final_report_job="$(sbatch --parsable \
  --dependency="afterok:${random_report_job}:${chronological_report_job}" \
  --export=ALL,PPG_PROJECT_ROOT="$project_root" \
  "$project_root/scripts/sbatch_same_subject_combination_final_report.sh" \
  "$seed" "$random_report_run" "$chronological_report_run")"
final_report_job="${final_report_job%%;*}"
final_report_run="$work_root/outputs/same-subject-combination-v1_final_report_seed${seed}_job${final_report_job}"
printf 'final_report\tboth\trobust_selection\t%s\t%s\t%s\n' \
  "$final_report_job" "$final_report_run" "$seed" >> "$manifest_work"

cp "$manifest_work" "$manifest_nas"
cmp -s "$manifest_work" "$manifest_nas"
echo "FINAL_REPORT_JOB=$final_report_job"
echo "SAME_SUBJECT_COMBINATION_MANIFEST=$manifest_work"
