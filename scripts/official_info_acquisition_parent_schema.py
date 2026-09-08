"""Read only identity and interval metadata from existing project indexes."""
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

base = Path("/home/jiangyu.cheng/work/ppg_bp/data/manifests/pulsedb_v2_full_cohort")
index = pq.ParquetFile(base / "pulsedb_v2_full_segment_index.parquet")
allowed = "source subject_uid segment_uid segment_row record_id raw_file start_time_s end_time_s duration_s sample_interval_s n_samples".split()
selected = [k for k in allowed if k in index.schema_arrow.names]
first = next(index.iter_batches(batch_size=1, columns=selected)).to_pylist()[0]
print(json.dumps(dict(index_columns=index.schema_arrow.names, index_rows=index.metadata.num_rows, whitelisted_first_row=first), default=str), flush=True)
split_path = base / "subject_splits.csv"
columns = pd.read_csv(split_path, nrows=0).columns.tolist()
approved = "source subject_uid subject_id split role meta_split parent_split parent_role partition development_fold fold".split()
usecols = [k for k in columns if k in approved]
frame = pd.read_csv(split_path, usecols=usecols)
counts = {k: {str(key): int(n) for key, n in frame[k].value_counts(dropna=False).items()} for k in usecols if "subject" not in k}
print(json.dumps(dict(split_columns=columns, split_rows=len(frame), approved_read_columns=usecols, split_counts=counts)), flush=True)
