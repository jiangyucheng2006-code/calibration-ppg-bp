#!/bin/bash

set -euo pipefail

seed="${1:-20260823}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
prepared="$work_root/outputs/event120-v1_round8_prepared_job960"
timestamp="$(date +%Y%m%d-%H%M%S)"
manifest_work="$work_root/outputs/submission_manifests/round9_${timestamp}.tsv"
manifest_nas="$archive_root/outputs/submission_manifests/round9_${timestamp}.tsv"

cd "$project_root"
test -f "$prepared/queries.parquet"
mkdir -p "$(dirname "$manifest_work")" "$(dirname "$manifest_nas")"

declare -a methods=(
  r8_reference
  adaptive_fusion
  range_soft_experts
  dbp_specific_physiology
  bias_regularized
  causal_attention
  personal_bp_direction
  temporal_delta_consistency
  support_dropout_consistency
)
declare -a labels=(
  "R9-0 R8 architecture reference"
  "R9-1 Adaptive base-personal fusion"
  "R9-2 Soft BP-range experts"
  "R9-3 DBP-specific physiology"
  "R9-4 Bias regularization"
  "R9-5 Short causal attention"
  "R9-6 Personal BP direction"
  "R9-7 Temporal delta consistency"
  "R9-8 Support dropout consistency"
)
declare -a jobs=()
declare -a specifications=()

printf 'method\tlabel\tjob_id\trun\tseed\n' > "$manifest_work"
for index in "${!methods[@]}"; do
  gres="gpu:rtx_5080:1"
  if (( index % 2 == 1 )); then
    gres="gpu:rtx_5070_ti:1"
  fi
  job="$(sbatch --parsable \
    --gres="$gres" \
    scripts/sbatch_round9_screen.sh \
    "${methods[$index]}" "$prepared" "$seed")"
  run="$work_root/outputs/event120-v1_round9_internal_${methods[$index]}_seed${seed}_job${job}"
  jobs+=("$job")
  specifications+=("${labels[$index]}=$run")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${methods[$index]}" "${labels[$index]}" "$job" "$run" "$seed" \
    >> "$manifest_work"
done

dependency="$(IFS=:; echo "${jobs[*]}")"
report_job="$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  scripts/sbatch_round9_report.sh "$seed" "${specifications[@]}")"
printf 'report\tRound-9 internal report\t%s\t%s\t%s\n' \
  "$report_job" \
  "$work_root/outputs/event120-v1_round9_internal_report_seed${seed}_job${report_job}" \
  "$seed" >> "$manifest_work"

cp -p "$manifest_work" "$manifest_nas"
cmp -s "$manifest_work" "$manifest_nas"
printf 'ROUND9_SCREEN_JOBS=%s\n' "${jobs[*]}"
printf 'ROUND9_REPORT_JOB=%s\n' "$report_job"
printf 'ROUND9_SUBMISSION_MANIFEST=%s\n' "$manifest_work"
