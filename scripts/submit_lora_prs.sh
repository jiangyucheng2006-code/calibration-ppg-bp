#!/bin/bash
set -euo pipefail
ppg_project="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_project"
ppg_work="/home/$USER/work/ppg_bp"
ppg_archive="/home/$USER/nas/ppg_bp/outputs/submission_manifests"
ppg_seed=20260907
ppg_manifest="$ppg_work/outputs/submission_manifests/lora_prs_$(date +%Y%m%d-%H%M%S).tsv"
mkdir -p "$(dirname "$ppg_manifest")" "$ppg_archive"
printf 'kind\tsplit_mode\tcandidate\tjob_id\trun\tseed\n' > "$ppg_manifest"
trap 'cp "$ppg_manifest" "$ppg_archive/"' EXIT
candidates=(lora_continue lora_prs_bias lora_prs_dynamic lora_prs_dynamic_shrink lora_prs_dynamic_frozen)
report_jobs=()
reports=()
for mode in random_disjoint chronological_blocked; do
  gres="gpu:rtx_5080:1"
  if [[ "$mode" == chronological_blocked ]]; then gres="gpu:rtx_5070_ti:1"; fi
  smoke="$(sbatch --parsable --gres="$gres" --export=ALL,PPG_PROJECT_ROOT="$ppg_project" \
    "$ppg_project/scripts/sbatch_lora_prs_smoke.sh" "$mode")"
  smoke="${smoke%%;*}"
  printf 'smoke\t%s\tall\t%s\t%s\t%s\n' "$mode" "$smoke" "$ppg_work/outputs/lora-prs-continuation-v1_${mode}_smoke_job${smoke}" "$ppg_seed" >> "$ppg_manifest"
  jobs=()
  runs=()
  for i in "${!candidates[@]}"; do
    candidate="${candidates[$i]}"
    job="$(sbatch --parsable --gres="$gres" --dependency="afterok:$smoke" --job-name="ppg_prs_${mode:0:1}_${i}" \
      --export=ALL,PPG_PROJECT_ROOT="$ppg_project" "$ppg_project/scripts/sbatch_lora_prs.sh" "$candidate" "$mode" "$ppg_seed")"
    job="${job%%;*}"
    run="$ppg_work/outputs/lora-prs-continuation-v1_${mode}_${candidate}_seed${ppg_seed}_job${job}"
    jobs+=("$job")
    runs+=("$run")
    printf 'training\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$candidate" "$job" "$run" "$ppg_seed" >> "$ppg_manifest"
  done
  dependency="$(IFS=:; echo "${jobs[*]}")"
  job="$(sbatch --parsable --dependency="afterok:$dependency" --export=ALL,PPG_PROJECT_ROOT="$ppg_project" \
    "$ppg_project/scripts/sbatch_lora_prs_report.sh" "$mode" "$ppg_seed" "${runs[@]}")"
  job="${job%%;*}"
  run="$ppg_work/outputs/lora-prs-continuation-v1_${mode}_report_seed${ppg_seed}_job${job}"
  report_jobs+=("$job")
  reports+=("$run")
  printf 'report\t%s\tall\t%s\t%s\t%s\n' "$mode" "$job" "$run" "$ppg_seed" >> "$ppg_manifest"
done
job="$(sbatch --parsable --dependency="afterok:${report_jobs[0]}:${report_jobs[1]}" --export=ALL,PPG_PROJECT_ROOT="$ppg_project" \
  "$ppg_project/scripts/sbatch_lora_prs_report.sh" final "$ppg_seed" "${reports[@]}")"
job="${job%%;*}"
run="$ppg_work/outputs/lora-prs-continuation-v1_final_report_seed${ppg_seed}_job${job}"
printf 'final_report\tboth\tall\t%s\t%s\t%s\n' "$job" "$run" "$ppg_seed" >> "$ppg_manifest"
cat "$ppg_manifest"
echo "MANIFEST=$ppg_manifest"
