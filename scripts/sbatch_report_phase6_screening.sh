#!/bin/bash
#SBATCH --job-name=ppg_p6_report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
project_root="$work_root/code/pulsedb_fewshot"
run_root="$work_root/outputs"
run_id="event120-v1_phase6_screening_report_job${SLURM_JOB_ID}"
output="$run_root/$run_id"
screening="$output/screening"
features="$output/reference_worst30_features"
oracle="$output/fixed_reference_oracle"

reference="$run_root/event120-v1_phase6_fixed_first_seed20260813_job818"
huber="$run_root/event120-v1_phase6_fixedfirst_huber_seed20260813_job826"
bp_change="$run_root/event120-v1_phase6_fixedfirst_bp_change_seed20260813_job827"
median_anchor="$run_root/event120-v1_phase6_fixedfirst_median_anchor_seed20260813_job828"
quality_gate="$run_root/event120-v1_phase6_fixedfirst_ppg_quality_gate_seed20260813_job829"
demographics="$run_root/event120-v1_phase6_fixedfirst_demographics_seed20260813_job830"
store="$work_root/data/processed/event120-v1"
demographics_table="$work_root/data/manifests/phase6_demographics/participant_demographics_clean.parquet"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

for path in "$reference" "$huber" "$bp_change" "$median_anchor" "$quality_gate" "$demographics"; do
  test -f "$path/run.json"
  test -f "$path/best_validation_predictions.parquet"
done
test -f "$demographics_table"
mkdir -p "$output"

python -m pulsedb_fewshot.report_phase6_screening \
  --run "Fixed-first M0=$reference" \
  --run "Huber=$huber" \
  --run "BP-change sampling=$bp_change" \
  --run "Median anchor=$median_anchor" \
  --run "PPG quality gate=$quality_gate" \
  --run "Age and sex=$demographics" \
  --reference-setting "Fixed-first M0" \
  --expected-seed 20260813 \
  --store-root "$store" \
  --output-dir "$screening"

python -m pulsedb_fewshot.analyze_residual_tail \
  --predictions "$reference/best_validation_predictions.parquet" \
  --store-root "$store" \
  --output-dir "$features" \
  --k 5 \
  --tail-fraction 0.30 \
  --demographics-path "$demographics_table"

python -m pulsedb_fewshot.analyze_worst_tail_oracle \
  --reference-run "$reference" \
  --candidate-run "Huber=$huber" \
  --candidate-run "BP-change sampling=$bp_change" \
  --candidate-run "Median anchor=$median_anchor" \
  --candidate-run "PPG quality gate=$quality_gate" \
  --candidate-run "Age and sex=$demographics" \
  --output-dir "$oracle" \
  --k 5 \
  --tail-fraction 0.30

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
diff -qr "$output" "$archive_root/outputs/$run_id"

echo "PHASE6_SCREENING_REPORT_ARCHIVE_IDENTICAL=yes"
echo "WORK_OUTPUT=$output"
echo "NAS_OUTPUT=$archive_root/outputs/$run_id"
