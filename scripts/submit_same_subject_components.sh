#!/bin/bash

set -euo pipefail

seed="${1:-20260831}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable snapshot}"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest_work="$work_root/outputs/submission_manifests/same_subject_components_${timestamp}.tsv"
manifest_nas="$archive_root/outputs/submission_manifests/same_subject_components_${timestamp}.tsv"

test -d "$project_root"
test ! -w "$project_root"
test -f "$work_root/data/processed/development-calbased-analogue-v1/materialization.json"
mkdir -p "$(dirname "$manifest_work")" "$(dirname "$manifest_nas")"
printf 'kind\tcandidate\tjob_id\trun\tseed\n' > "$manifest_work"

similarity_job="$(sbatch --parsable \
  --export=ALL,PPG_PROJECT_ROOT="$project_root" \
  "$project_root/scripts/sbatch_same_subject_similarity.sh")"
similarity_job="${similarity_job%%;*}"
similarity_run="$work_root/outputs/same-subject-single-component-v1_train_similarity_job${similarity_job}"
similarity_path="$similarity_run/train_beat_similarity.parquet"
printf 'preparation\tbeat_similarity\t%s\t%s\t%s\n' \
  "$similarity_job" "$similarity_run" "$seed" >> "$manifest_work"

candidates=(
  residual_reference
  residual_quality_gate
  residual_quality_weighted_loss
  residual_ppg_quality_filter
  residual_calibration_relative
  residual_support_attention
  residual_support_reliability
  residual_film
  residual_multi_event_weighting
  residual_subject_lora_rank4
  residual_inception_time_wide
  residual_patch_transformer
  residual_conformer
  residual_cnn_bilstm
  residual_cnn_gru
  residual_soft_moe
  residual_prototype_moe
  residual_demographics_direct
  residual_beat_similarity_filter
)

job_ids=()
run_dirs=()
for index in "${!candidates[@]}"; do
  candidate="${candidates[$index]}"
  gres="gpu:rtx_5080:1"
  if (( index % 2 == 1 )); then gres="gpu:rtx_5070_ti:1"; fi
  batch_size=128
  case "$candidate" in
    residual_patch_transformer|residual_conformer) batch_size=48 ;;
    residual_cnn_bilstm|residual_soft_moe|residual_prototype_moe|residual_subject_lora_rank4) batch_size=64 ;;
  esac
  dependency=()
  if [[ "$candidate" == "residual_beat_similarity_filter" ]]; then
    dependency=(--dependency="afterok:${similarity_job}")
  fi
  job_id="$(sbatch --parsable \
    --job-name="ppg_ss_${index}" \
    --gres="$gres" \
    "${dependency[@]}" \
    --export=ALL,PPG_PROJECT_ROOT="$project_root",PPG_BEAT_SIMILARITY_PATH="$similarity_path" \
    "$project_root/scripts/sbatch_same_subject_component.sh" \
    "$candidate" "$seed" "$batch_size")"
  job_id="${job_id%%;*}"
  run_dir="$work_root/outputs/same-subject-single-component-v1_random_disjoint_${candidate}_seed${seed}_job${job_id}"
  printf 'training\t%s\t%s\t%s\t%s\n' \
    "$candidate" "$job_id" "$run_dir" "$seed" >> "$manifest_work"
  job_ids+=("$job_id")
  run_dirs+=("$run_dir")
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
report_job="$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  --export=ALL,PPG_PROJECT_ROOT="$project_root" \
  "$project_root/scripts/sbatch_same_subject_component_report.sh" \
  "$seed" "${run_dirs[@]}")"
report_job="${report_job%%;*}"
report_run="$work_root/outputs/same-subject-single-component-v1_report_seed${seed}_job${report_job}"
printf 'report\tall_components\t%s\t%s\t%s\n' \
  "$report_job" "$report_run" "$seed" >> "$manifest_work"

cp "$manifest_work" "$manifest_nas"
cmp -s "$manifest_work" "$manifest_nas"
echo "SAME_SUBJECT_COMPONENT_SIMILARITY_JOB=$similarity_job"
echo "SAME_SUBJECT_COMPONENT_TRAINING_JOBS=${job_ids[*]}"
echo "SAME_SUBJECT_COMPONENT_REPORT_JOB=$report_job"
echo "SAME_SUBJECT_COMPONENT_MANIFEST=$manifest_work"

