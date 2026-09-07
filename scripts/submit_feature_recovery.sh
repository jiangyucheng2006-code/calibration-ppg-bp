#!/bin/bash
set -euo pipefail
ppg_project="${PPG_PROJECT_ROOT:?original snapshot required}"
ppg_recovery="${PPG_RECOVERY_ROOT:?recovery snapshot required}"
test ! -w "$ppg_project"
test ! -w "$ppg_recovery"
ppg_work="/home/$USER/work/ppg_bp"
ppg_manifest="$ppg_work/outputs/submission_manifests/feature_recovery_$(date +%Y%m%d-%H%M%S).tsv"
ppg_archive="/home/$USER/nas/ppg_bp/outputs/submission_manifests"
mkdir -p "$(dirname "$ppg_manifest")" "$ppg_archive"
printf 'kind\tjob_id\trun\treplaces_job\n' > "$ppg_manifest"
trap 'cp "$ppg_manifest" "$ppg_archive/"' EXIT
ppg_exports="ALL,PPG_PROJECT_ROOT=$ppg_project,PPG_RECOVERY_ROOT=$ppg_recovery"
smoke="$(sbatch --parsable --time=00:15:00 --job-name=ppg_feature_retry_smoke --export="$ppg_exports" \
  "$ppg_recovery/scripts/sbatch_feature_recovery.sh" smoke)"
smoke="${smoke%%;*}"
printf 'smoke\t%s\t%s\t\n' "$smoke" "$ppg_work/outputs/personal-feature-recovery-smoke_job${smoke}" >> "$ppg_manifest"
retry="$(sbatch --parsable --dependency="afterok:$smoke" --job-name=ppg_feature_retry64 --export="$ppg_exports" \
  "$ppg_recovery/scripts/sbatch_feature_recovery.sh" train)"
retry="${retry%%;*}"
retry_run="$ppg_work/outputs/personal-feature-mechanisms-v1_random_disjoint_shared_bilinear64_seed20260906_job${retry}"
printf 'training\t%s\t%s\t1486\n' "$retry" "$retry_run" >> "$ppg_manifest"
candidates=(subject_lora_rank4 shared_lora_rank4 subject_lora_rank1 output_profile32 feature_affine32 shared_bilinear32 shared_bilinear64 subject_nonlinear_rank4)
old_jobs=(1480 1481 1482 1483 1484 1485 1486 1487)
runs=()
for i in "${!candidates[@]}"; do
  if [[ "${old_jobs[$i]}" == 1486 ]]; then
    runs+=("$retry_run")
  else
    run="$ppg_work/outputs/personal-feature-mechanisms-v1_random_disjoint_${candidates[$i]}_seed20260906_job${old_jobs[$i]}"
    test -s "$run/run.json"
    runs+=("$run")
  fi
done
report="$(sbatch --parsable --dependency="afterok:$retry" --export="$ppg_exports" \
  "$ppg_project/scripts/sbatch_personal_feature_report.sh" random_disjoint 20260906 "${runs[@]}")"
report="${report%%;*}"
random_report="$ppg_work/outputs/personal-feature-mechanisms-v1_random_disjoint_report_seed20260906_job${report}"
printf 'split_report\t%s\t%s\t1488\n' "$report" "$random_report" >> "$ppg_manifest"
chrono_report="$ppg_work/outputs/personal-feature-mechanisms-v1_chronological_blocked_report_seed20260906_job1497"
test -s "$chrono_report/selection.json"
final="$(sbatch --parsable --dependency="afterok:$report" --export="$ppg_exports" \
  "$ppg_project/scripts/sbatch_personal_feature_report.sh" final 20260906 "$random_report" "$chrono_report")"
final="${final%%;*}"
printf 'final_report\t%s\t%s\t1498\n' "$final" "$ppg_work/outputs/personal-feature-mechanisms-v1_final_report_seed20260906_job${final}" >> "$ppg_manifest"
# Old pending reports are canceled separately only after replacement verification.
cat "$ppg_manifest"
echo "MANIFEST=$ppg_manifest"
