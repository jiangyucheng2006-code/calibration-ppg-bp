"""Inspect official assignment structure and decode identity/index fields only.

All non-whitelisted datasets, including BP values, remain unread. The private
membership parquet belongs beside server metadata, never in the public repo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()
    with h5py.File(args.input, "r") as handle:
        keys = [k for k in handle if not k.startswith("#")]
        if len(keys) != 1 or not isinstance(handle[keys[0]], h5py.Group):
            raise RuntimeError(f"Unexpected MATLAB root schema: {keys}")
        group = handle[keys[0]]
        schema = dict(root=keys[0], fields={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in group.items()}, read_fields=["Subj_Name", "Subj_SegIDX"])
        print(json.dumps(schema), flush=True)
        if args.schema_only:
            return
        names = group["Subj_Name"][()].ravel(order="F")
        indices = group["Subj_SegIDX"][()].ravel(order="F")
        if len(names) != len(indices):
            raise RuntimeError("Unequal whitelist field lengths")
        subjects, raw_indices = [], []
        for nref, iref in zip(names, indices, strict=True):
            if not isinstance(nref, h5py.Reference) or not isinstance(iref, h5py.Reference):
                raise RuntimeError("Expected MATLAB reference cells")
            chars = np.asarray(handle[nref][()]).ravel(order="F")
            name = "".join(chr(int(x)) for x in chars)
            idx = np.asarray(handle[iref][()]).ravel(order="F")
            if idx.size != 1 or not np.isfinite(idx[0]) or int(idx[0]) != idx[0] or idx[0] < 1:
                raise RuntimeError("Invalid raw MATLAB segment index")
            if len(name) < 8 or name[-1] not in "01":
                raise RuntimeError("Unknown official subject/source encoding")
            subjects.append(name)
            raw_indices.append(int(idx[0]))
        frame = pd.DataFrame(dict(official_subject=subjects, matlab_subj_segidx=raw_indices))
        frame["raw_subject_id"] = frame.official_subject.str[:7]
        frame["source"] = frame.official_subject.str[-1].map({"0": "MIMIC", "1": "VitalDB"})
        if frame.duplicated(["official_subject", "matlab_subj_segidx"]).any():
            raise RuntimeError("Duplicate official identity/index membership")
        counts = frame.groupby("official_subject").size()
        summary = dict(file=args.input.name, root=keys[0], n_rows=len(frame), n_subjects=len(counts), windows_per_subject_counts={str(int(k)): int(v) for k, v in counts.value_counts().sort_index().items()}, source_windows={str(k): int(v) for k, v in frame.groupby("source").size().items()}, source_subjects={str(k): int(v) for k, v in frame.groupby("source").official_subject.nunique().items()}, read_fields=["Subj_Name", "Subj_SegIDX"], bp_targets_read=False, waveform_read=False)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output_dir / (args.input.stem + "_membership.parquet"), index=False)
        (args.output_dir / (args.input.stem + "_schema.json")).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / (args.input.stem + "_summary.json")).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
