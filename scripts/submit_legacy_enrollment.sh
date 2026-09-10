#!/bin/bash
# One bounded two-method experiment, not recursive optimization from test scores.
set -euo pipefail
umask 077
[[ "$(hostname)" == slurm ]]
ppg_code="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_code"
ppg_work=/home/jiangyu.cheng/work/ppg_bp
ppg_nas=/home/jiangyu.cheng/nas/ppg_bp
ppg_tag="${PPG_BATCH_TAG:?batch tag}"
[[ "$ppg_tag" =~ ^[0-9]{8}-[0-9]{6}$ ]]
ppg_root="$ppg_work/outputs/legacy-split-enrollment-v1_$ppg_tag"
ppg_store="${PPG_OFFICIAL_STORE:?audited source pool}"
ppg_smoke="${PPG_SMOKE_JOB:?completed smoke job}"
[[ "$ppg_smoke" =~ ^[0-9]+$ ]]
[[ "$(sacct -n -X -j "$ppg_smoke" --format=State --parsable2 | tr -d ' ')" == COMPLETED ]]
[[ "$(sacct -n -X -j "$ppg_smoke" --format=ExitCode --parsable2 | tr -d ' ')" == 0:0 ]]
PYTHONPATH="$ppg_code/src" "$ppg_work/envs/train/bin/python" -c 'import json,sys; from pathlib import Path; from pulsedb_fewshot.training import source_tree_sha256; r=json.loads(Path(sys.argv[1]).read_text()); assert r["status"]=="pass" and r["gpu_test_passed"] is True; assert r["protocol_id"]=="legacy-split-enrollment-v1" and r["snapshot"]==sys.argv[2] and r["source_store"]==sys.argv[3] and r["slurm_job_id"]==sys.argv[4]; assert r["source_tree_sha256"]==source_tree_sha256(Path(sys.argv[2]))' "$ppg_root/smoke/receipt.json" "$ppg_code" "$ppg_store" "$ppg_smoke"
test ! -e "$ppg_root/submission.tsv"
ppg_gres="$(sinfo -N -h -n hpc-2 -o %G)"
[[ "$ppg_gres" == *gpu:rtx_5080:1* && "$ppg_gres" == *gpu:rtx_5070_ti:1* ]]
ppg_manifest="$ppg_root/submission.tsv"
printf 'stage\tjob_id\tdependency\tresource\n' > "$ppg_manifest"
printf 'smoke\t%s\tnone\trtx_5080\n' "$ppg_smoke" >> "$ppg_manifest"
mkdir -p "$ppg_nas/outputs/$(basename "$ppg_root")"
trap 'cp "$ppg_manifest" "$ppg_nas/outputs/$(basename "$ppg_root")/"' EXIT
submit() {
  local stage="$1" dependency="$2" gpu="$3" duration="$4" id
  local options=(--parsable --export=ALL --job-name="ppg_legacy_${stage}" --kill-on-invalid-dep=yes --dependency="afterok:$dependency" --time="$duration")
  [[ "$gpu" == none ]] || options+=(--gres="gpu:$gpu:1")
  id="$(sbatch "${options[@]}" "$ppg_code/scripts/sbatch_legacy_enrollment.sh" "$stage" "$ppg_root" "$ppg_store")"
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]]
  printf '%s\t%s\t%s\t%s\n' "$stage" "$id" "$dependency" "$gpu" >> "$ppg_manifest"
  printf '%s' "$id"
}
prepare="$(submit prepare "$ppg_smoke" none 02:00:00)"
population="$(submit population "$prepare" rtx_5080 72:00:00)"
v0="$(submit validation_0 "$population" rtx_5080 24:00:00)"
v1="$(submit validation_1 "$population" rtx_5070_ti 24:00:00)"
validation="$(submit validation_score "$v0:$v1" none 02:00:00)"
t0="$(submit test_0 "$validation" rtx_5080 24:00:00)"
t1="$(submit test_1 "$validation" rtx_5070_ti 24:00:00)"
final="$(submit test_score "$t0:$t1" none 02:00:00)"
printf 'FINAL_EVALUATOR=%s\n' "$final"
cat "$ppg_manifest"
