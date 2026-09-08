#!/bin/bash
# Finite prespecified six-method pipeline; no automatic test-feedback loop.
set -euo pipefail
umask 077
ppg_project="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_project"
[[ "$(hostname)" == slurm ]]
ppg_work=/home/jiangyu.cheng/work/ppg_bp
ppg_nas=/home/jiangyu.cheng/nas/ppg_bp
ppg_tag="${PPG_BATCH_TAG:?explicit batch tag required}"
[[ "$ppg_tag" =~ ^[0-9]{8}-[0-9]{6}$ ]]
ppg_prepare="${PPG_PREPARE_JOB:?verified preparation job ID required}"
[[ "$ppg_prepare" =~ ^[0-9]+$ ]]
ppg_store="${PPG_OFFICIAL_STORE:?exact official store path required}"
ppg_root="$ppg_work/outputs/official-calbased-v1_$ppg_tag"
[[ ! -e "$ppg_root" ]]
ppg_gres="$(sinfo -N -h -n hpc-2 -o %G)"
[[ "$ppg_gres" == *gpu:rtx_5080:1* && "$ppg_gres" == *gpu:rtx_5070_ti:1* ]]
mkdir -p "$ppg_root" "$ppg_nas/outputs/$(basename "$ppg_root")"
ppg_manifest="$ppg_root/submission.tsv"
printf 'stage\tjob_id\tdependency\tresource\toutput\n' > "$ppg_manifest"
trap 'cp "$ppg_manifest" "$ppg_nas/outputs/$(basename "$ppg_root")/"' EXIT
submit() {
  local stage="$1" dependency="$2" gpu="$3"
  local options=(--parsable --export=ALL --job-name="ppg_off_${stage}" --kill-on-invalid-dep=yes)
  [[ "$dependency" == none ]] || options+=(--dependency="afterok:$dependency")
  [[ "$gpu" == none ]] || options+=(--gres="gpu:$gpu:1")
  [[ "$stage" != smoke ]] || options+=(--time=00:30:00 --mem=8G)
  [[ "$stage" != evaluate ]] || options+=(--time=02:00:00 --mem=8G)
  local id
  id="$(sbatch "${options[@]}" "$ppg_project/scripts/sbatch_official_calbased.sh" "$stage" "$ppg_root" "$ppg_store")"
  id="${id%%;*}"
  [[ "$id" =~ ^[0-9]+$ ]]
  printf '%s\t%s\t%s\t%s\t%s\n' "$stage" "$id" "$dependency" "$gpu" "$ppg_root/$stage" >> "$ppg_manifest"
  printf '%s' "$id"
}
if [[ -n "${PPG_SMOKE_JOB:-}" ]]; then
  [[ "$PPG_SMOKE_JOB" =~ ^[0-9]+$ ]]
  smoke="$PPG_SMOKE_JOB"
  printf 'smoke\t%s\tnone\trtx_5080\tpre-submission verified snapshot smoke\n' "$smoke" >> "$ppg_manifest"
else
  smoke="$(submit smoke none rtx_5080)"
fi
inner="$(submit inner_lora "$smoke:$ppg_prepare" rtx_5080)"
oof0="$(submit oof_0 "$smoke:$ppg_prepare" rtx_5070_ti)"
oof1="$(submit oof_1 "$oof0" rtx_5070_ti)"
oof2="$(submit oof_2 "$oof1" rtx_5070_ti)"
fixed="$(submit inner_fixed "$inner" rtx_5080)"
v1="$(submit inner_v1 "$fixed" rtx_5080)"
e1="$(submit inner_e1 "$v1" rtx_5080)"
trust1="$(submit inner_trust_scalar "$e1:$oof2" rtx_5080)"
trust2="$(submit inner_trust_bp "$e1:$oof2" rtx_5070_ti)"
final="$(submit final_lora "$trust1:$trust2" rtx_5080)"
f0="$(submit final_fixed "$final" rtx_5080)"
f1="$(submit final_v1 "$f0" rtx_5080)"
f2="$(submit final_e1 "$final" rtx_5070_ti)"
f3="$(submit final_trust_scalar "$f1" rtx_5080)"
f4="$(submit final_trust_bp "$f2" rtx_5070_ti)"
submit evaluate "$f3:$f4" none
printf '\nSUBMISSION_MANIFEST=%s\n' "$ppg_manifest"
cat "$ppg_manifest"
