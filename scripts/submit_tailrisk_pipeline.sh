#!/bin/bash

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
manifest_root="$work_root/outputs/submission_manifests"
pipeline_root="$work_root/outputs/event120-v1_tailrisk-v1"
folds_path="$pipeline_root/folds/meta_train_crossfit_folds.parquet"
seed_base=20260818
reference_population="$work_root/outputs/event120-v1_repeat5-v1_population_seed20260813_job782/best.pt"

cd "$project_root"
mkdir -p "$manifest_root"
test -f "$reference_population"
if [[ -e "$pipeline_root" ]]; then
  echo "ERROR: tail-risk pipeline root already exists: $pipeline_root" >&2
  exit 1
fi

prepare_job=$(sbatch --parsable \
  scripts/sbatch_tailrisk_prepare.sh "$seed_base")

population_jobs=()
m0_jobs=()
m0_runs=()
for fold in 0 1 2 3 4; do
  fold_seed=$((seed_base + fold))
  if (( fold % 2 == 0 )); then
    gpu_type="rtx_5070_ti"
  else
    gpu_type="rtx_5080"
  fi
  population_job=$(sbatch --parsable \
    --dependency="afterok:${prepare_job}" \
    --gres="gpu:${gpu_type}:1" \
    --job-name="ppg_xpop_f${fold}" \
    scripts/sbatch_tailrisk_crossfit_model.sh \
    population "$fold" "$folds_path" "$fold_seed")
  population_run="$work_root/outputs/event120-v1_tailrisk_xfit_population_fold${fold}_seed${fold_seed}_job${population_job}"
  population_checkpoint="$population_run/best.pt"
  m0_job=$(sbatch --parsable \
    --dependency="afterok:${population_job}" \
    --gres="gpu:${gpu_type}:1" \
    --job-name="ppg_xm0_f${fold}" \
    scripts/sbatch_tailrisk_crossfit_model.sh \
    m0 "$fold" "$folds_path" "$fold_seed" "$population_checkpoint")
  m0_run="$work_root/outputs/event120-v1_tailrisk_xfit_m0_fold${fold}_seed${fold_seed}_job${m0_job}"
  population_jobs+=("$population_job")
  m0_jobs+=("$m0_job")
  m0_runs+=("$m0_run")
done

m0_dependency=$(IFS=:; echo "${m0_jobs[*]}")
aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${m0_dependency}" \
  scripts/sbatch_tailrisk_aggregate.sh "$folds_path" "${m0_runs[@]}")

classifier_job=$(sbatch --parsable \
  --dependency="afterok:${aggregate_job}" \
  scripts/sbatch_tailrisk_classifier.sh "$seed_base")

expert_plain_job=$(sbatch --parsable \
  --dependency="afterok:${aggregate_job}" \
  --gres=gpu:rtx_5070_ti:1 \
  --job-name=ppg_tail_exp4 \
  scripts/sbatch_tailrisk_specialist.sh \
  expert_weight4 no weighted 4.0 20260813 "$reference_population")
expert_plain_run="$work_root/outputs/event120-v1_tailrisk_expert_weight4_seed20260813_job${expert_plain_job}"

expert_hard_job=$(sbatch --parsable \
  --dependency="afterok:${aggregate_job}" \
  --gres=gpu:rtx_5080:1 \
  --job-name=ppg_tail_hard \
  scripts/sbatch_tailrisk_specialist.sh \
  expert_hard_only no hard_only 1.0 20260813 "$reference_population")
expert_hard_run="$work_root/outputs/event120-v1_tailrisk_expert_hard_only_seed20260813_job${expert_hard_job}"

expert_qgate_hard_job=$(sbatch --parsable \
  --dependency="afterok:${aggregate_job}" \
  --gres=gpu:rtx_5070_ti:1 \
  --job-name=ppg_tail_qhard \
  scripts/sbatch_tailrisk_specialist.sh \
  qgate_expert_hard_only yes hard_only 1.0 20260813 "$reference_population")
expert_qgate_hard_run="$work_root/outputs/event120-v1_tailrisk_qgate_expert_hard_only_seed20260813_job${expert_qgate_hard_job}"

route_plain_job=$(sbatch --parsable \
  --dependency="afterok:${classifier_job}:${expert_plain_job}" \
  scripts/sbatch_tailrisk_route.sh expert_weight4 "$expert_plain_run")
route_hard_job=$(sbatch --parsable \
  --dependency="afterok:${classifier_job}:${expert_hard_job}" \
  scripts/sbatch_tailrisk_route.sh expert_hard_only "$expert_hard_run")
route_qgate_hard_job=$(sbatch --parsable \
  --dependency="afterok:${classifier_job}:${expert_qgate_hard_job}" \
  scripts/sbatch_tailrisk_route.sh qgate_expert_hard_only "$expert_qgate_hard_run")

manifest="$manifest_root/event120-v1_tailrisk-v1_$(date +%Y%m%d-%H%M%S).txt"
{
  echo "PHASE=Phase 6c cross-fitted hard-participant identification and specialist routing"
  echo "PROTOCOL=event120-v1_fixed_first_event6plus"
  echo "TAIL_LABEL_SOURCE=meta_train_crossfit_oof_k5"
  echo "TAIL_FRACTION=0.30_within_source"
  echo "RISK_FEATURES=input_visible_ppg_prediction_and_support_bp_only"
  echo "LOCKED_META_TEST_ACCESSED=no"
  echo "PREPARE_JOB=$prepare_job"
  for fold in 0 1 2 3 4; do
    echo "FOLD_${fold}_POPULATION_JOB=${population_jobs[$fold]}"
    echo "FOLD_${fold}_M0_JOB=${m0_jobs[$fold]}"
  done
  echo "OOF_AGGREGATION_JOB=$aggregate_job"
  echo "RISK_CLASSIFIER_JOB=$classifier_job"
  echo "EXPERT_WEIGHT4_JOB=$expert_plain_job"
  echo "EXPERT_HARD_ONLY_JOB=$expert_hard_job"
  echo "QGATE_EXPERT_HARD_ONLY_JOB=$expert_qgate_hard_job"
  echo "ROUTE_EXPERT_WEIGHT4_JOB=$route_plain_job"
  echo "ROUTE_EXPERT_HARD_ONLY_JOB=$route_hard_job"
  echo "ROUTE_QGATE_EXPERT_HARD_ONLY_JOB=$route_qgate_hard_job"
  echo "MULTI_SEED_CONFIRMATION_SUBMITTED=no"
} | tee "$manifest"

mkdir -p "$archive_root/outputs/submission_manifests"
cp -p "$manifest" "$archive_root/outputs/submission_manifests/"
echo "TAILRISK_PIPELINE_SUBMITTED=yes"
