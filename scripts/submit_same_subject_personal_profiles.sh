#!/bin/bash

set -euo pipefail
seed="${1:-20260904}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable snapshot}"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest_work="$work_root/outputs/submission_manifests/same_subject_personal_profiles_${timestamp}.tsv"
manifest_nas="$archive_root/outputs/submission_manifests/same_subject_personal_profiles_${timestamp}.tsv"

test -d "$project_root"
test ! -w "$project_root"
test -f "$work_root/data/processed/development-calbased-analogue-v1/materialization.json"
mkdir -p "$(dirname "$manifest_work")" "$(dirname "$manifest_nas")"
printf 'kind\tsplit_mode\tcandidate\tjob_id\trun\tseed\n' > "$manifest_work"

candidates=(
  residual_reference
  subject_lora_rank4
  personal_profile_support_only
  personal_profile_code32_no_support
  personal_profile_code32_no_reliability
  personal_profile_code32_reliability
  personal_profile_code64_reliability
  personal_profile_code32_stable_only
)

submit_split() {
  local split_mode="$1"
  local job_ids=()
  local run_dirs=()
  local index candidate gres job_id run_dir
  for index in "${!candidates[@]}"; do
    candidate="${candidates[$index]}"
    gres="gpu:rtx_5080:1"
    if (( index % 2 == 1 )); then gres="gpu:rtx_5070_ti:1"; fi
    job_id="$(sbatch --parsable \
      --job-name="ppg_ssp_${split_mode:0:1}_${index}" \
      --gres="$gres" \
      --export=ALL,PPG_PROJECT_ROOT="$project_root" \
      "$project_root/scripts/sbatch_same_subject_personal_profile.sh" \
      "$candidate" "$split_mode" "$seed" 64)"
    job_id="${job_id%%;*}"
    run_dir="$work_root/outputs/same-subject-personal-profile-v1_${split_mode}_${candidate}_seed${seed}_job${job_id}"
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
    "$project_root/scripts/sbatch_same_subject_personal_profile_report.sh" \
    "$split_mode" "$seed" "${run_dirs[@]}")"
  report_job="${report_job%%;*}"
  report_run="$work_root/outputs/same-subject-personal-profile-v1_${split_mode}_report_seed${seed}_job${report_job}"
  printf 'report\t%s\tall_profiles\t%s\t%s\t%s\n' \
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

submit_split random_disjoint
submit_split chronological_blocked
final_report_job="$(sbatch --parsable \
  --dependency="afterok:${random_report_job}:${chronological_report_job}" \
  --export=ALL,PPG_PROJECT_ROOT="$project_root" \
  "$project_root/scripts/sbatch_same_subject_personal_profile_final_report.sh" \
  "$seed" "$random_report_run" "$chronological_report_run")"
final_report_job="${final_report_job%%;*}"
final_report_run="$work_root/outputs/same-subject-personal-profile-v1_final_report_seed${seed}_job${final_report_job}"
printf 'final_report\tboth\tprimary_vs_controls\t%s\t%s\t%s\n' \
  "$final_report_job" "$final_report_run" "$seed" >> "$manifest_work"
cp "$manifest_work" "$manifest_nas"
cmp -s "$manifest_work" "$manifest_nas"
echo "FINAL_REPORT_JOB=$final_report_job"
echo "SAME_SUBJECT_PERSONAL_PROFILE_MANIFEST=$manifest_work"
