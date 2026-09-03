#!/bin/bash
#SBATCH --job-name=ppg_p6b_report
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
run_id="event120-v1_phase6b_factorial_report_job${SLURM_JOB_ID}"
output="$run_root/$run_id"
screening="$output/screening"
oracle="$output/fixed_reference_oracle"
bootstrap="$output/bootstrap"
store="$work_root/data/processed/event120-v1"

reference="$run_root/event120-v1_phase6_fixed_first_seed20260813_job818"
huber="$run_root/event120-v1_phase6_fixedfirst_huber_seed20260813_job826"
quality_gate="$run_root/event120-v1_phase6_fixedfirst_ppg_quality_gate_seed20260813_job829"
cvar="$run_root/event120-v1_phase6b_cvar_seed20260813_job840"
quality_gate_huber="$run_root/event120-v1_phase6b_qgate_huber_seed20260813_job841"
quality_gate_cvar="$run_root/event120-v1_phase6b_qgate_cvar_seed20260813_job842"
huber_cvar="$run_root/event120-v1_phase6b_huber_cvar_seed20260813_job843"
quality_gate_huber_cvar="$run_root/event120-v1_phase6b_qgate_huber_cvar_seed20260813_job844"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$work_root/envs/train"
cd "$project_root"

for path in \
  "$reference" \
  "$huber" \
  "$quality_gate" \
  "$cvar" \
  "$quality_gate_huber" \
  "$quality_gate_cvar" \
  "$huber_cvar" \
  "$quality_gate_huber_cvar"; do
  test -f "$path/run.json"
  test -f "$path/best_validation_predictions.parquet"
done

mkdir -p "$output"

python -m pulsedb_fewshot.report_phase6_screening \
  --run "Fixed-first M0=$reference" \
  --run "Huber=$huber" \
  --run "PPG quality gate=$quality_gate" \
  --run "Participant CVaR=$cvar" \
  --run "Quality gate + Huber=$quality_gate_huber" \
  --run "Quality gate + CVaR=$quality_gate_cvar" \
  --run "Huber + CVaR=$huber_cvar" \
  --run "Quality gate + Huber + CVaR=$quality_gate_huber_cvar" \
  --reference-setting "Fixed-first M0" \
  --expected-seed 20260813 \
  --store-root "$store" \
  --output-dir "$screening"

python -m pulsedb_fewshot.analyze_worst_tail_oracle \
  --reference-run "$reference" \
  --candidate-run "Huber=$huber" \
  --candidate-run "PPG quality gate=$quality_gate" \
  --candidate-run "Participant CVaR=$cvar" \
  --candidate-run "Quality gate + Huber=$quality_gate_huber" \
  --candidate-run "Quality gate + CVaR=$quality_gate_cvar" \
  --candidate-run "Huber + CVaR=$huber_cvar" \
  --candidate-run "Quality gate + Huber + CVaR=$quality_gate_huber_cvar" \
  --output-dir "$oracle" \
  --k 5 \
  --tail-fraction 0.30

python -m pulsedb_fewshot.report_phase6b_bootstrap \
  --run "Fixed-first M0=$reference" \
  --run "Huber=$huber" \
  --run "PPG quality gate=$quality_gate" \
  --run "Participant CVaR=$cvar" \
  --run "Quality gate + Huber=$quality_gate_huber" \
  --run "Quality gate + CVaR=$quality_gate_cvar" \
  --run "Huber + CVaR=$huber_cvar" \
  --run "Quality gate + Huber + CVaR=$quality_gate_huber_cvar" \
  --comparison "Huber vs M0|Huber|Fixed-first M0" \
  --comparison "Quality gate vs M0|PPG quality gate|Fixed-first M0" \
  --comparison "CVaR vs M0|Participant CVaR|Fixed-first M0" \
  --comparison "Quality gate + Huber vs M0|Quality gate + Huber|Fixed-first M0" \
  --comparison "Huber added to quality gate|Quality gate + Huber|PPG quality gate" \
  --comparison "CVaR added to quality gate|Quality gate + CVaR|PPG quality gate" \
  --comparison "CVaR added to Huber|Huber + CVaR|Huber" \
  --comparison "CVaR added to quality gate + Huber|Quality gate + Huber + CVaR|Quality gate + Huber" \
  --store-root "$store" \
  --output-dir "$bootstrap" \
  --expected-seed 20260813 \
  --repetitions 20000 \
  --bootstrap-seed 20260819

mkdir -p "$archive_root/outputs/$run_id"
rsync -a "$output/" "$archive_root/outputs/$run_id/"
diff -qr "$output" "$archive_root/outputs/$run_id"

echo "PHASE6B_FACTORIAL_REPORT_ARCHIVE_IDENTICAL=yes"
echo "WORK_OUTPUT=$output"
echo "NAS_OUTPUT=$archive_root/outputs/$run_id"
