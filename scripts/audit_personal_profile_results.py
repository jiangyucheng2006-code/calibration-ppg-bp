"""Read-only audit of saved development predictions; emit aggregate evidence only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(manifest: Path) -> dict:
    entries = list(csv.DictReader(manifest.open(), delimiter="\t"))
    bases = {}
    reports = {}
    for entry in entries:
        if entry["kind"] == "report":
            path = Path(entry["run"])
            reports[entry["split_mode"]] = (
                pd.read_csv(path / "participant_macro_summary.csv"),
                pd.read_csv(path / "event_pooled_diagnostics_all_scopes.csv"),
            )
    records = []
    checksums = {}
    for entry in entries:
        path = Path(entry["run"])
        archive = Path(str(path).replace("/work/", "/nas/", 1))
        if entry["kind"] != "training":
            for file in path.iterdir():
                if file.is_file():
                    assert sha256(file) == sha256(archive / file.name)
            checksums[entry["split_mode"]] = {
                file.name: sha256(file) for file in path.iterdir() if file.is_file()
            }
            continue
        run = json.loads((path / "run.json").read_text())
        assert run["heldout_test_accessed"] is False
        assert run["selection_role"] == "internal_validation"
        frame = pd.read_parquet(path / "best_internal_validation_predictions.parquet")
        keys = ["subject_uid", "event_id", "source"]
        assert not frame.duplicated(keys).any()
        frame = frame.sort_values(keys).reset_index(drop=True)
        mode = entry["split_mode"]
        expected = frame[keys + ["target_sbp", "target_dbp"]]
        if mode in bases:
            pd.testing.assert_frame_equal(expected, bases[mode])
        else:
            bases[mode] = expected
        assert len(frame) == 82040 and frame.subject_uid.nunique() == 2051
        assert frame.groupby("subject_uid").size().eq(40).all()
        assert np.isfinite(frame[["target_sbp", "target_dbp", "pred_sbp", "pred_dbp"]]).all().all()
        macro, pooled = reports[mode]
        for scope in ["Overall", "MIMIC", "VitalDB"]:
            part = frame if scope == "Overall" else frame.loc[frame.source.eq(scope)]
            row = macro.loc[macro.candidate.eq(entry["candidate"]) & macro.view.eq(scope)].iloc[0]
            assert len(part) == int(row.n_events)
            assert part.subject_uid.nunique() == int(row.n_participants)
            for bp in ["sbp", "dbp"]:
                y = part[f"target_{bp}"].to_numpy(dtype=float)
                err = part[f"pred_{bp}"].to_numpy(dtype=float) - y
                subject_mae = pd.Series(abs(err), index=part.subject_uid).groupby(level=0).mean().mean()
                assert np.isclose(subject_mae, row[f"{bp}_mae"], atol=2e-5)
                diagnostic = pooled.loc[pooled.Setting.eq(entry["candidate"]) & pooled.Scope.eq(scope) & pooled.BP.eq(bp.upper())].iloc[0]
                computed = {"MAE": abs(err).mean(), "ME": err.mean(), "STD": err.std(ddof=1),
                            "R²": 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()}
                computed.update({f"≤{limit} mmHg": 100 * (abs(err) <= limit).mean() for limit in [5, 10, 15]})
                for key, value in computed.items():
                    assert np.isclose(value, diagnostic[key], atol=2e-5), (entry["job_id"], scope, key)
        verified_files = [f for f in path.iterdir() if f.suffix in (".json", ".parquet", ".csv")]
        assert all(sha256(f) == sha256(archive / f.name) for f in verified_files)
        records.append({key: run[key] for key in ["candidate", "split_mode", "seed", "slurm_job_id", "best_epoch", "epochs_completed", "stop_reason", "support_count", "participant_trainable_parameters", "parameter_counts", "source_tree_sha256", "store_manifest_sha256", "heldout_test_accessed"]})
    assert len(records) == 16
    return {"status": "pass", "training_run_count": len(records), "identical_validation_keys_and_targets_within_each_mode": True,
            "participant_macro_and_pooled_metrics_recomputed": True, "report_work_nas_identical": True,
            "run_metadata_prediction_and_profile_archives_identical": True,
            "checkpoint_binary_hashes_rechecked_in_this_audit": False,
            "heldout_test_accessed": False, "report_checksums": checksums, "runs": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest), indent=2))
