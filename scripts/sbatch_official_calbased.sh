#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=72:00:00
#SBATCH --output=/home/jiangyu.cheng/work/ppg_bp/logs/official_calbased_%j.log
set -euo pipefail
umask 077
ppg_project="${PPG_PROJECT_ROOT:?immutable snapshot required}"
test ! -w "$ppg_project"
ppg_work=/home/jiangyu.cheng/work/ppg_bp
ppg_nas=/home/jiangyu.cheng/nas/ppg_bp
ppg_python="$ppg_work/envs/train/bin/python"
ppg_stage="${1:?stage}"; ppg_root="${2:?batch root}"; ppg_store="${3:?store}"; shift 3
[[ "$ppg_root" == "$ppg_work/outputs/official-calbased-v1_"* ]]
[[ "$ppg_store" == "$ppg_work/data/processed/pulsedb-official-calbased-v1_"* ]]
export PYTHONPATH="$ppg_project/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ppg_root"
ppg_output="$ppg_root/$ppg_stage"
archive() {
  local code=$?
  trap - EXIT
  mkdir -p "$ppg_nas/outputs/$(basename "$ppg_root")" "$ppg_nas/logs"
  if [[ -d "$ppg_output" ]]; then
    mkdir -p "$ppg_nas/outputs/$(basename "$ppg_root")/$(basename "$ppg_output")"
    rsync -a --no-perms --no-owner --no-group "$ppg_output/" "$ppg_nas/outputs/$(basename "$ppg_root")/$(basename "$ppg_output")/" || code=1
  fi
  cp "$ppg_work/logs/official_calbased_${SLURM_JOB_ID}.log" "$ppg_nas/logs/" || true
  exit "$code"
}
trap archive EXIT
cd "$ppg_project"
case "$ppg_stage" in
  smoke)
    mkdir "$ppg_output"
    "$ppg_python" -m unittest discover -s tests -p 'test_official*.py' -v
    "$ppg_python" -c 'import torch; from pulsedb_fewshot.lora_prs_models import LoraPRSRegressor, PRS_MODELS; assert torch.cuda.is_available(); m=LoraPRSRegressor(PRS_MODELS["lora_continue"],subject_count=2).cuda(); y=m(torch.randn(4,1,1250,device="cuda"),torch.zeros(4,2,device="cuda"),subject_index=torch.tensor([0,1,0,1],device="cuda")); y.square().mean().backward(); print("GPU_FORWARD_BACKWARD=pass",torch.cuda.get_device_name(0))'
    ;;
  inner_lora)
    "$ppg_python" -m pulsedb_fewshot.official_calbased_train --store-root "$ppg_store" --output "$ppg_output" --stage inner
    ;;
  oof_0|oof_1|oof_2)
    "$ppg_python" -m pulsedb_fewshot.official_calbased_train --store-root "$ppg_store" --output "$ppg_output" --stage oof --fold "${ppg_stage#oof_}" --oof-epochs 25
    ;;
  inner_fixed|inner_v1|inner_e1|inner_trust_scalar|inner_trust_bp)
    ppg_method="${ppg_stage#inner_}"
    ppg_options=()
    if [[ "$ppg_method" == trust_* ]]; then ppg_options+=(--oof-roots "$ppg_root/oof_0/cache" "$ppg_root/oof_1/cache" "$ppg_root/oof_2/cache"); fi
    "$ppg_python" -m pulsedb_fewshot.official_memory_train --stage inner --cache-root "$ppg_root/inner_lora/cache" --method "$ppg_method" --output "$ppg_output" "${ppg_options[@]}"
    ;;
  final_lora)
    "$ppg_python" -m pulsedb_fewshot.official_calbased_train --store-root "$ppg_store" --output "$ppg_output" --stage final --selection-run "$ppg_root/inner_lora" --include-test-inputs
    ;;
  final_fixed|final_v1|final_e1|final_trust_scalar|final_trust_bp)
    ppg_method="${ppg_stage#final_}"
    "$ppg_python" -m pulsedb_fewshot.official_memory_train --stage final --cache-root "$ppg_root/final_lora/cache" --method "$ppg_method" --selection-run "$ppg_root/inner_$ppg_method/run.json" --output "$ppg_output"
    ;;
  evaluate)
    ppg_index="$ppg_work/data/manifests/pulsedb_v2_full_cohort/pulsedb_v2_full_segment_index.parquet"
    "$ppg_python" -m pulsedb_fewshot.official_calbased_freeze --batch-root "$ppg_root" --store-root "$ppg_store" --research-plan "$ppg_project/docs/PLAN_OFFICIAL_CALBASED_20260908.md" --full-index "$ppg_index"
    ppg_plan="$ppg_root/frozen_test_predictions.json"
    ppg_hash="$(sha256sum "$ppg_plan" | cut -d ' ' -f1)"
    "$ppg_python" -m pulsedb_fewshot.official_calbased_evaluate --store-root "$ppg_store" --frozen-plan "$ppg_plan" --expected-plan-sha256 "$ppg_hash" --full-index "$ppg_index" --output "$ppg_output"
    cp "$ppg_plan" "$ppg_nas/outputs/$(basename "$ppg_root")/"
    ;;
  *) printf 'Unknown official stage: %s\n' "$ppg_stage" >&2; exit 2 ;;
esac
