#!/bin/bash
set -euo pipefail
ppg_project="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_project"
ppg_work="/home/$USER/work/ppg_bp"
ppg_seed=20260906
ppg_manifest="$ppg_work/outputs/submission_manifests/personal_feature_$(date +%Y%m%d-%H%M%S).tsv"
mkdir -p "$(dirname "$ppg_manifest")"
printf 'kind\tsplit_mode\tcandidate\tjob_id\trun\tseed\n' > "$ppg_manifest"
candidates=(subject_lora_rank4 shared_lora_rank4 subject_lora_rank1 output_profile32 feature_affine32 shared_bilinear32 shared_bilinear64 subject_nonlinear_rank4)
report_jobs=()
reports=()
for mode in random_disjoint chronological_blocked; do
  jobs=()
  runs=()
  for i in "${!candidates[@]}"; do
    candidate="${candidates[$i]}"
    gres="gpu:rtx_5080:1"
    if [[ "$mode" == chronological_blocked ]]; then gres="gpu:rtx_5070_ti:1"; fi
    job="$(sbatch --parsable --gres="$gres" --job-name="ppg_pf_${mode:0:1}_${i}" \
      --export=ALL,PPG_PROJECT_ROOT="$ppg_project" "$ppg_project/scripts/sbatch_personal_feature.sh" "$candidate" "$mode" "$ppg_seed")"
    job="${job%%;*}"
    run="$ppg_work/outputs/personal-feature-mechanisms-v1_${mode}_${candidate}_seed${ppg_seed}_job${job}"
    jobs+=("$job")
    runs+=("$run")
    printf 'training\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$candidate" "$job" "$run" "$ppg_seed" >> "$ppg_manifest"
  done
  dependency="$(IFS=:; echo "${jobs[*]}")"
  job="$(sbatch --parsable --dependency="afterok:$dependency" --export=ALL,PPG_PROJECT_ROOT="$ppg_project" \
    "$ppg_project/scripts/sbatch_personal_feature_report.sh" "$mode" "$ppg_seed" "${runs[@]}")"
  job="${job%%;*}"
  run="$ppg_work/outputs/personal-feature-mechanisms-v1_${mode}_report_seed${ppg_seed}_job${job}"
  report_jobs+=("$job")
  reports+=("$run")
  printf 'report\t%s\tall\t%s\t%s\t%s\n' "$mode" "$job" "$run" "$ppg_seed" >> "$ppg_manifest"
done
job="$(sbatch --parsable --dependency="afterok:${report_jobs[0]}:${report_jobs[1]}" --export=ALL,PPG_PROJECT_ROOT="$ppg_project" \
  "$ppg_project/scripts/sbatch_personal_feature_report.sh" final "$ppg_seed" "${reports[@]}")"
job="${job%%;*}"
run="$ppg_work/outputs/personal-feature-mechanisms-v1_final_report_seed${ppg_seed}_job${job}"
printf 'final_report\tboth\tall\t%s\t%s\t%s\n' "$job" "$run" "$ppg_seed" >> "$ppg_manifest"
cp "$ppg_manifest" "/home/$USER/nas/ppg_bp/outputs/submission_manifests/"
cat "$ppg_manifest"
echo "MANIFEST=$ppg_manifest"
