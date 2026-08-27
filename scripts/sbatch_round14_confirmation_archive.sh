#!/bin/bash
#SBATCH --job-name=ppg_r14_archive
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
archive_logs="$archive_root/logs/round14_confirmation_job${SLURM_JOB_ID}"

test -f "$manifest"
mkdir -p "$archive_logs"
test ! -e "$archive_logs/$(basename "$manifest")"
cp -- "$manifest" "$archive_logs/"
cmp -s "$manifest" "$archive_logs/$(basename "$manifest")"

tail -n +2 "$manifest" | while IFS=$'\t' read -r \
  stage backbone seed job_id dependency run gres microbatch accumulation effective_batch; do
  case "$stage" in
    existing_evaluation)
      test -d "$run"
      archived_run="$archive_root/outputs/$(basename "$run")"
      test -d "$archived_run"
      diff -qr "$run" "$archived_run" >/dev/null
      printf 'ARCHIVE_VERIFIED=%s:%s:%s\n' "$stage" "$backbone" "$seed"
      continue
      ;;
    smoke) job_name="ppg_r14_smoke" ;;
    population) job_name="ppg_r14_pop" ;;
    qgh) job_name="ppg_r14_qgh" ;;
    evaluation) job_name="ppg_r14_eval" ;;
    report) job_name="ppg_r14_report" ;;
    archive) continue ;;
    *) echo "ERROR: unknown manifest stage $stage" >&2; exit 1 ;;
  esac
  for extension in out err; do
    log_file="$work_root/logs/${job_name}-${job_id}.${extension}"
    target="$archive_logs/$(basename "$log_file")"
    test -f "$log_file"
    test ! -e "$target"
    cp -- "$log_file" "$target"
    cmp -s "$log_file" "$target"
  done
  test -d "$run"
  archived_run="$archive_root/outputs/$(basename "$run")"
  test -d "$archived_run"
  diff -qr "$run" "$archived_run" >/dev/null
  printf 'ARCHIVE_VERIFIED=%s:%s:%s\n' "$stage" "$backbone" "$seed"
done

echo "ROUND14_CONFIRMATION_LOG_ARCHIVE=$archive_logs"
echo "ROUND14_CONFIRMATION_ARCHIVE_COMPLETE=yes"
