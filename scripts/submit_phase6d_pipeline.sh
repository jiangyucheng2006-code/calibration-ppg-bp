#!/bin/bash

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
pipeline_root="$work_root/outputs/event120-v1_phase6d_qgh_routing"
folds="$work_root/outputs/event120-v1_tailrisk-v1/folds/meta_train_crossfit_folds.parquet"
manifest_root="$work_root/outputs/submission_manifests"
reference_population="$work_root/outputs/event120-v1_repeat5-v1_population_seed20260813_job782/best.pt"

cd "$project_root"
mkdir -p "$manifest_root"
test -f "$folds"
test -f "$reference_population"
[[ ! -e "$pipeline_root" ]] || { echo "ERROR: pipeline root already exists: $pipeline_root" >&2; exit 1; }
# Reserve the unique experiment root before the first submission so an
# accidental second invocation cannot enqueue a duplicate dependency chain.
mkdir "$pipeline_root"

population_jobs=(852 854 856 858 860)
fold_jobs=()
fold_runs=()
for fold in 0 1 2 3 4; do
  fold_seed=$((20260818 + fold))
  population_run="$work_root/outputs/event120-v1_tailrisk_xfit_population_fold${fold}_seed${fold_seed}_job${population_jobs[$fold]}"
  population_checkpoint="$population_run/best.pt"
  test -f "$population_checkpoint"
  if (( fold % 2 == 0 )); then gpu_type=rtx_5070_ti; else gpu_type=rtx_5080; fi
  job=$(sbatch --parsable --gres="gpu:${gpu_type}:1" --job-name="ppg_p6d_xq${fold}" \
    scripts/sbatch_phase6d_crossfit_qgh.sh "$fold" "$folds" "$fold_seed" "$population_checkpoint")
  fold_jobs+=("$job")
  fold_runs+=("$work_root/outputs/event120-v1_phase6d_xfit_qgate_huber_fold${fold}_seed${fold_seed}_job${job}")
done

fold_dependency=$(IFS=:; echo "${fold_jobs[*]}")
aggregate_job=$(sbatch --parsable --dependency="afterok:${fold_dependency}" \
  scripts/sbatch_phase6d_aggregate.sh "$folds" "${fold_runs[@]}")
classifier_job=$(sbatch --parsable --dependency="afterok:${aggregate_job}" \
  scripts/sbatch_phase6d_classifier.sh 20260819)

weight2_job=$(sbatch --parsable --dependency="afterok:${aggregate_job}" --gres=gpu:rtx_5070_ti:1 \
  --job-name=ppg_p6d_qghw2 scripts/sbatch_phase6d_specialist.sh \
  qghuber_weight2 weighted 2.0 20260813 "$reference_population")
weight2_run="$work_root/outputs/event120-v1_phase6d_qghuber_weight2_seed20260813_job${weight2_job}"

weight4_job=$(sbatch --parsable --dependency="afterok:${aggregate_job}" --gres=gpu:rtx_5080:1 \
  --job-name=ppg_p6d_qghw4 scripts/sbatch_phase6d_specialist.sh \
  qghuber_weight4 weighted 4.0 20260813 "$reference_population")
weight4_run="$work_root/outputs/event120-v1_phase6d_qghuber_weight4_seed20260813_job${weight4_job}"

hard_job=$(sbatch --parsable --dependency="afterok:${aggregate_job}" --gres=gpu:rtx_5070_ti:1 \
  --job-name=ppg_p6d_qghard scripts/sbatch_phase6d_specialist.sh \
  qghuber_hard_only hard_only 1.0 20260813 "$reference_population")
hard_run="$work_root/outputs/event120-v1_phase6d_qghuber_hard_only_seed20260813_job${hard_job}"

report_dependency="${classifier_job}:${weight2_job}:${weight4_job}:${hard_job}"
report_job=$(sbatch --parsable --dependency="afterok:${report_dependency}" \
  scripts/sbatch_phase6d_report.sh \
  "QGHuber-weight2=${weight2_run}" \
  "QGHuber-weight4=${weight4_run}" \
  "QGHuber-hard-only=${hard_run}")

manifest="$manifest_root/event120-v1_phase6d_qgh_routing_$(date +%Y%m%d-%H%M%S).txt"
{
  echo "PHASE=Phase 6d risk identification specialist verification and complete routing"
  echo "PROTOCOL=event120-v1_fixed_first_k5_event6plus"
  echo "GENERAL_MODEL=M0_quality_gate_huber_job841"
  echo "TAIL_LABEL_SOURCE=quality_gate_huber_meta_train_crossfit_oof_k5"
  echo "TAIL_FRACTION=0.30_within_source"
  echo "PRIMARY_ROUTING=input_visible_event_risk_threshold_frozen_on_meta_train"
  echo "LOCKED_META_TEST_ACCESSED=no"
  for fold in 0 1 2 3 4; do echo "CROSSFIT_QGH_FOLD_${fold}_JOB=${fold_jobs[$fold]}"; done
  echo "OOF_AGGREGATION_JOB=$aggregate_job"
  echo "RISK_CLASSIFIER_JOB=$classifier_job"
  echo "QGHUBER_WEIGHT2_EXPERT_JOB=$weight2_job"
  echo "QGHUBER_WEIGHT4_EXPERT_JOB=$weight4_job"
  echo "QGHUBER_HARD_ONLY_EXPERT_JOB=$hard_job"
  echo "COMPLETE_PIPELINE_REPORT_JOB=$report_job"
  echo "MULTI_SEED_CONFIRMATION_SUBMITTED=no"
} | tee "$manifest"

mkdir -p "$archive_root/outputs/submission_manifests"
cp -p "$manifest" "$archive_root/outputs/submission_manifests/"
echo "PHASE6D_PIPELINE_SUBMITTED=yes"
