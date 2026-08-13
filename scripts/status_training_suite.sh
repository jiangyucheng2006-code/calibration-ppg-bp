#!/bin/bash

set -euo pipefail

work_root="/home/$USER/work/ppg_bp"
jobs=(754 755 756 757 764 772 773 774 775 776 777)

echo "===== LIVE QUEUE ====="
squeue -u "$USER" -o '%.12i %.24j %.2t %.10M %.10l %.6D %R'
echo
echo "===== SUITE ACCOUNTING ====="
sacct -j "$(IFS=,; echo "${jobs[*]}")" \
  --format=JobID,JobName,State,ExitCode,Elapsed,NodeList \
  -n -P | grep -v '\.batch\|\.extern' || true
echo
echo "===== LATEST TRAINING EPOCHS ====="
for job in 772 773 774 775 776; do
  log=$(find "$work_root/logs" -maxdepth 1 -type f -name "*-${job}.out" -print -quit)
  if [ -n "$log" ]; then
    printf 'JOB %s | ' "$job"
    grep '^{' "$log" | grep '"epoch"' | tail -n 1 || echo "no epoch record yet"
  fi
done
echo
echo "===== COMPLETED RUN SUMMARIES ====="
find "$work_root/outputs" -mindepth 2 -maxdepth 2 -type f -name run.json -print \
  | sort | while read -r path; do
      python - "$path" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
run = json.loads(path.read_text(encoding="utf-8"))
print(
    f"{path.parent.name}|method={run.get('method')}|"
    f"best_validation_mean_mae={run.get('best_validation_mean_mae')}|"
    f"checkpoint_sha256={run.get('checkpoint_sha256')}"
)
PY
    done
