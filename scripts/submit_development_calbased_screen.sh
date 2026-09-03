#!/bin/bash

set -euo pipefail

mode_request="${1:-both}"
seed="${2:-20260828}"
case "$mode_request" in
  both) split_modes=(random_disjoint chronological_blocked) ;;
  random_disjoint|chronological_blocked) split_modes=("$mode_request") ;;
  *) echo "ERROR: mode must be both, random_disjoint, or chronological_blocked" >&2; exit 2 ;;
esac

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="${PPG_PROJECT_ROOT:?PPG_PROJECT_ROOT must name an immutable snapshot}"
store_root="$work_root/data/processed/development-calbased-analogue-v1"
prep_dependency="${PPG_PREP_DEPENDENCY:-}"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest_work="$work_root/outputs/submission_manifests/development_calbased_${mode_request}_${timestamp}.tsv"
manifest_nas="$archive_root/outputs/submission_manifests/development_calbased_${mode_request}_${timestamp}.tsv"

test -d "$project_root"
test ! -w "$project_root"
cd "$project_root"
if [[ ! -f "$store_root/materialization.json" && -z "$prep_dependency" ]]; then
  echo "ERROR: store is absent and no PPG_PREP_DEPENDENCY was supplied: $store_root" >&2
  exit 1
fi
upstream_dependency=()
if [[ -n "$prep_dependency" ]]; then
  [[ "$prep_dependency" =~ ^[0-9]+$ ]] || {
    echo "ERROR: PPG_PREP_DEPENDENCY must be one numeric Slurm job ID" >&2
    exit 2
  }
  upstream_dependency=(--dependency="afterok:${prep_dependency}")
fi
mkdir -p "$(dirname "$manifest_work")" "$(dirname "$manifest_nas")"
printf 'status\tcandidate\tjob_id\trun\tsplit_mode\tseed\n' > "$manifest_work"

candidates=(
  subject_mean_residual_ppg
  compact_resnet
  inception_time_wide
  patch_transformer
  self_attention_resunet_adaptation
  runet_resunet_encoder_adaptation
  cnn_bilstm_adaptation
  cnn_transformer_aff_adaptation
)
report_jobs=()
for split_mode in "${split_modes[@]}"; do
  job_ids=()
  run_dirs=()
  baseline_job="$(sbatch --parsable \
    "${upstream_dependency[@]}" \
    --export=ALL,PPG_PROJECT_ROOT="$project_root" \
    scripts/sbatch_development_calbased_baseline.sh "$split_mode" "$seed")"
  baseline_job="${baseline_job%%;*}"
  baseline_run="$work_root/outputs/development-calbased-analogue-v1_${split_mode}_subject_train_mean_seed${seed}_job${baseline_job}"
  printf 'submitted\tsubject_train_mean\t%s\t%s\t%s\t%s\n' \
    "$baseline_job" "$baseline_run" "$split_mode" "$seed" >> "$manifest_work"
  job_ids+=("$baseline_job")
  run_dirs+=("$baseline_run")

  for index in "${!candidates[@]}"; do
    candidate="${candidates[$index]}"
    gres="gpu:rtx_5080:1"
    if (( index % 2 == 1 )); then
      gres="gpu:rtx_5070_ti:1"
    fi
    batch_size=128
    if [[ "$candidate" == "self_attention_resunet_adaptation" || "$candidate" == "runet_resunet_encoder_adaptation" ]]; then
      batch_size=24
    elif [[ "$candidate" == "cnn_transformer_aff_adaptation" || "$candidate" == "patch_transformer" ]]; then
      batch_size=64
    fi
    job_id="$(sbatch --parsable --gres="$gres" \
      "${upstream_dependency[@]}" \
      --export=ALL,PPG_PROJECT_ROOT="$project_root" \
      scripts/sbatch_development_calbased_candidate.sh \
      "$candidate" "$split_mode" "$seed" "$batch_size")"
    job_id="${job_id%%;*}"
    run_dir="$work_root/outputs/development-calbased-analogue-v1_${split_mode}_${candidate}_seed${seed}_job${job_id}"
    printf 'submitted\t%s\t%s\t%s\t%s\t%s\n' \
      "$candidate" "$job_id" "$run_dir" "$split_mode" "$seed" >> "$manifest_work"
    job_ids+=("$job_id")
    run_dirs+=("$run_dir")
  done

  printf 'deferred\tcompact_resnet_qgh\t\t\t%s\t%s\n' \
    "$split_mode" "$seed" >> "$manifest_work"
  printf 'deferred\tcompact_resnet_calibration_relative\t\t\t%s\t%s\n' \
    "$split_mode" "$seed" >> "$manifest_work"

  dependency="$(IFS=:; echo "${job_ids[*]}")"
  report_job="$(sbatch --parsable --dependency="afterok:${dependency}" \
    --export=ALL,PPG_PROJECT_ROOT="$project_root" \
    scripts/sbatch_development_calbased_report.sh \
    "$split_mode" "$seed" "${run_dirs[@]}")"
  report_job="${report_job%%;*}"
  report_run="$work_root/outputs/development-calbased-analogue-v1_${split_mode}_screen_report_seed${seed}_job${report_job}"
  printf 'report\tinternal_validation_selection\t%s\t%s\t%s\t%s\n' \
    "$report_job" "$report_run" "$split_mode" "$seed" >> "$manifest_work"
  report_jobs+=("$report_job")
done

cp "$manifest_work" "$manifest_nas"
cmp -s "$manifest_work" "$manifest_nas"
printf 'DEVELOPMENT_CALBASED_REPORT_JOBS=%s\n' "${report_jobs[*]}"
printf 'DEVELOPMENT_CALBASED_SUBMISSION_MANIFEST=%s\n' "$manifest_work"
