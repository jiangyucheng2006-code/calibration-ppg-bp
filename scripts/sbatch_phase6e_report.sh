#!/bin/bash
#SBATCH --job-name=ppg_p6e_report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err
set -euo pipefail
work=/home/$USER/work/ppg_bp; archive=/home/$USER/nas/ppg_bp; root=$work/code/pulsedb_fewshot; run=event120-v1_phase6e_seven_route_report_job${SLURM_JOB_ID}; out=$work/outputs/$run
source /opt/conda/etc/profile.d/conda.sh; conda activate $work/envs/train; cd $root
cmd=(python -m pulsedb_fewshot.report_phase6e --general-run $work/outputs/event120-v1_phase6b_qgate_huber_seed20260813_job841 --output $out)
for spec in "$@"; do cmd+=(--run "$spec"); done
"${cmd[@]}"
mkdir -p $archive/outputs/$run; rsync -a $out/ $archive/outputs/$run/; diff -qr $out $archive/outputs/$run
