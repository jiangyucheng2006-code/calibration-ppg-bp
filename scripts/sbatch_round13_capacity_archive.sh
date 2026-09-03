#!/bin/bash
#SBATCH --job-name=ppg_r13_archive
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/work/ppg_bp/logs/%x-%j.out
#SBATCH --error=/home/%u/work/ppg_bp/logs/%x-%j.err

set -euo pipefail

manifest="${1:?submission manifest is required}"
work_root="/home/$USER/work/ppg_bp"
archive_root="/home/$USER/nas/ppg_bp"
archive_logs="$archive_root/logs/round13_capacity_job${SLURM_JOB_ID}"

test -f "$manifest"
mkdir -p "$archive_logs"
cp -p "$manifest" "$archive_logs/"

tail -n +2 "$manifest" | while IFS=$'\t' read -r stage backbone job_id run _rest; do
  case "$stage" in
    population) job_name="ppg_r13_pop" ;;
    qgh) job_name="ppg_r13_qgh" ;;
    evaluation) job_name="ppg_r13_eval" ;;
    report) job_name="ppg_r13_report" ;;
    archive) continue ;;
    *) echo "ERROR: unknown manifest stage $stage" >&2; exit 1 ;;
  esac
  for extension in out err; do
    log_file="$work_root/logs/${job_name}-${job_id}.${extension}"
    test -f "$log_file"
    cp -p "$log_file" "$archive_logs/"
  done
  test -d "$run"
  archived_run="$archive_root/outputs/$(basename "$run")"
  test -d "$archived_run"
  diff -qr "$run" "$archived_run" >/dev/null
  printf 'ARCHIVE_VERIFIED=%s:%s\n' "$stage" "$backbone"
done

echo "ROUND13_LOG_ARCHIVE=$archive_logs"
echo "ROUND13_ARCHIVE_COMPLETE=yes"
