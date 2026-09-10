#!/bin/bash
# Eight prespecified budgets; two parallel GPU lanes; no automatic new studies.
set -euo pipefail
umask 077
[[ "$(hostname)" == slurm ]]
ppg_code="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_code"
ppg_work=/home/jiangyu.cheng/work/ppg_bp
ppg_nas=/home/jiangyu.cheng/nas/ppg_bp
ppg_tag="${PPG_BATCH_TAG:?tag}"
[[ "$ppg_tag" =~ ^[0-9]{8}-[0-9]{6}$ ]]
ppg_root="$ppg_work/outputs/enrollment-budget-v1_$ppg_tag"
ppg_smoke="${PPG_SMOKE_JOB:?completed smoke required}"
[[ "$ppg_smoke" =~ ^[0-9]+$ ]]
[[ "$(sacct -n -X -j "$ppg_smoke" --format=State --parsable2 | tr -d ' ')" == COMPLETED ]]
[[ "$(sacct -n -X -j "$ppg_smoke" --format=ExitCode --parsable2 | tr -d ' ')" == 0:0 ]]
PYTHONPATH="$ppg_code/src" "$ppg_work/envs/train/bin/python" -c 'import json,sys; from pathlib import Path; from pulsedb_fewshot.training import source_tree_sha256; r=json.loads(Path(sys.argv[1]).read_text()); assert r["status"]=="pass" and r["gpu_test_passed"] is True; assert r["protocol_id"]=="enrollment-budget-v1" and r["snapshot"]==sys.argv[2] and r["slurm_job_id"]==sys.argv[3]; assert r["source_tree_sha256"]==source_tree_sha256(Path(sys.argv[2]))' "$ppg_root/smoke/receipt.json" "$ppg_code" "$ppg_smoke"
ppg_gres="$(sinfo -N -h -n hpc-2 -o %G)"
[[ "$ppg_gres" == *gpu:rtx_5080:1* && "$ppg_gres" == *gpu:rtx_5070_ti:1* ]]
ppg_manifest="$ppg_root/submission.tsv"
test ! -e "$ppg_manifest"
printf 'stage\tjob_id\tdependency\tresource\nsmoke\t%s\tnone\trtx_5080\n' "$ppg_smoke" > "$ppg_manifest"
mkdir -p "$ppg_nas/outputs/$(basename "$ppg_root")"
trap 'cp "$ppg_manifest" "$ppg_nas/outputs/$(basename "$ppg_root")/"' EXIT
submit() {
  local stage="$1" dependency="$2" gpu="$3" duration="$4" cpus="$5" mem="$6" id
  local options=(--parsable --export=ALL --job-name="ppg_budget_${stage}" --kill-on-invalid-dep=yes --dependency="afterok:$dependency" --time="$duration" --cpus-per-task="$cpus" --mem="$mem")
  [[ "$gpu" == none ]] || options+=(--gres="gpu:$gpu:1")
  id="$(sbatch "${options[@]}" "$ppg_code/scripts/sbatch_enrollment_budget.sh" "$stage" "$ppg_root")"
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]]
  printf '%s\t%s\t%s\t%s\n' "$stage" "$id" "$dependency" "$gpu" >> "$ppg_manifest"
  printf '%s' "$id"
}
prepare="$(submit prepare "$ppg_smoke" none 03:00:00 2 12G)"
lane0="$prepare"
lane1="$prepare"
for percent in 20 30 40 50 60 70 80 90; do
  lane0="$(submit "validation_p${percent}_0" "$lane0" rtx_5080 24:00:00 2 12G)"
  lane1="$(submit "validation_p${percent}_1" "$lane1" rtx_5070_ti 24:00:00 2 12G)"
done
validation="$(submit validation_score "$lane0:$lane1" none 03:00:00 2 12G)"
lane0="$validation"
lane1="$validation"
for percent in 20 30 40 50 60 70 80 90; do
  lane0="$(submit "test_p${percent}_0" "$lane0" rtx_5080 24:00:00 2 12G)"
  lane1="$(submit "test_p${percent}_1" "$lane1" rtx_5070_ti 24:00:00 2 12G)"
done
final="$(submit test_score "$lane0:$lane1" none 03:00:00 2 12G)"
printf 'FINAL_EVALUATOR=%s\n' "$final"
cat "$ppg_manifest"
