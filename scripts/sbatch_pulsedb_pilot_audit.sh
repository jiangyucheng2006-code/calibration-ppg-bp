#!/usr/bin/env bash
#SBATCH --job-name=pulsedb-pilot-audit
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

ppg_runner="/home/${USER}/work/ppg_bp/code/pulsedb_fewshot/scripts/run_pulsedb_pilot_audit.sh"
test -x "${ppg_runner}"
exec "${ppg_runner}"
