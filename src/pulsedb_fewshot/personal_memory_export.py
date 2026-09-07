"""Export frozen personalized features from the registered-user LoRA reference.

This reads train/internal_validation only. Validation BP is stored for scoring,
never passed to the model. Hashes are checked against materialized raw PPG_F;
the feature hook captures the actual post-LoRA tensor, not a reconstructed path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .calbased_train import load_screen_metadata, _autocast, _fit_scaler
from .calbased_screen import fit_subject_train_means
from .lora_prs_models import LoraPRSRegressor, PRS_MODELS
from .training import file_sha256, save_json, seed_everything, source_tree_sha256

SOURCES = {
    "random_disjoint": {"job": "1500", "hash": "64041f60e654f5ee48979a3898a444c690ceb58a617ed968ac9c435daef045ff", "gpu": "NVIDIA GeForce RTX 5080"},
    "chronological_blocked": {"job": "1507", "hash": "5d0ed8afb19bbdf8f69316de7be24cad4fc904ccd95a16dd3f9c8a88d26cfdd2", "gpu": "NVIDIA GeForce RTX 5070 Ti"},
}
SCREEN_ID = "personal-memory-v1"


def check_reference_targets(saved, raw):
    """Compare in the source loader's float32 domain, not raw MATLAB float64.

    The old prediction writer de-standardizes the float32 training target.
    Its last rounding step can differ from the raw float64 by slightly more
    than2e-5, even though the float32-domain disagreement is one ULP. This
    retains the2e-5 bound after matching the documented source representation.
    It does not alter labels or allow changed query identities.
    """
    a = np.asarray(saved, dtype=np.float32)
    b = np.asarray(raw, dtype=np.float32)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid reference scoring targets")
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if difference.max() > 2e-5:
        raise ValueError("reference scoring targets mismatch in source float32 domain")
    return {"comparison_dtype": "float32_source_loader", "max_abs_mmhg": float(difference.max()),
            "mean_abs_mmhg": float(difference.mean()), "absolute_tolerance_mmhg": 2e-5}


def check_source(run_dir: Path, split_mode: str, store_root: Path):
    metadata = json.loads((run_dir / "run.json").read_text())
    spec = SOURCES[split_mode]
    if not (metadata["status"] == "complete" and metadata["candidate"] == "lora_continue"
            and metadata["screen_id"] == "lora-prs-continuation-v1"
            and metadata["protocol_id"] == "development-calbased-analogue-v1"
            and metadata["split_mode"] == split_mode
            and metadata["source_parent_split"] == "meta_train"
            and metadata["read_roles"] == ["train", "internal_validation"]
            and metadata["selection_role"] == "internal_validation"
            and metadata["heldout_test_accessed"] is False):
        raise ValueError("source checkpoint protocol/provenance mismatch")
    checkpoint_path = run_dir / "best.pt"
    if file_sha256(checkpoint_path) != spec["hash"] or metadata["checkpoint_sha256"] != spec["hash"]:
        raise ValueError("reference checkpoint differs from frozen approved source")
    if file_sha256(store_root / "materialization.json") != metadata["store_manifest_sha256"]:
        raise ValueError("materialized data manifest changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return metadata, checkpoint


def canonical_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Use record-local real times; never invent a cross-record chronological axis."""
    if not set(frame.role).issubset({"train", "internal_validation"}):
        raise ValueError("forbidden memory/export role")
    if frame.record_id.isna().any() or frame.record_id.astype(str).eq("").any():
        raise ValueError("missing record-local clock")
    start = frame.start_time_s.to_numpy(dtype=float)
    end = start + frame.duration_s.to_numpy(dtype=float)
    inclusive_end = frame.end_time_s.to_numpy(dtype=float) + frame.sample_interval_s.to_numpy(dtype=float)
    if not np.isfinite(np.stack([start, end, inclusive_end])).all() or not np.allclose(end, inclusive_end, rtol=0, atol=1e-4):
        raise ValueError("record interval/sample count inconsistent")
    recording = frame.source.astype(str) + ":" + frame.record_id.astype(str)
    result = pd.DataFrame({
        "subject_uid": frame.subject_uid.astype(str), "event_id": frame.segment_uid.astype(str),
        "source": frame.source.astype(str), "window_uid": frame.segment_uid.astype(str),
        "waveform_sha256": frame.ppg_content_sha256.astype(str),
        "recording_uid": recording, "time_axis_uid": recording,
        "start_s": start, "end_s": end, "role": frame.role.astype(str),
        "target_sbp": frame.sbp.to_numpy(dtype=np.float32),
        "target_dbp": frame.dbp.to_numpy(dtype=np.float32),
    })
    if not result.waveform_sha256.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("missing raw PPG content hash")
    return result


class ExportDataset(Dataset):
    """Array-backed sequential loader; verify raw bytes before standardization."""
    def __init__(self, frame, store_root, means, indices, scaler):
        self.files = frame.waveform_file.astype(str).to_numpy()
        self.rows = frame.waveform_row.to_numpy(dtype=int)
        self.hashes = frame.ppg_content_sha256.astype(str).to_numpy()
        self.arrays = {name: np.load(store_root / name, mmap_mode="r") for name in set(self.files)}
        ordered_means = means.set_index("subject_uid").reindex(frame.subject_uid)
        raw_anchor = ordered_means[["subject_train_sbp", "subject_train_dbp"]].to_numpy(dtype=np.float32)
        self.anchors = (raw_anchor - np.asarray(scaler["mean"], np.float32)) / np.asarray(scaler["std"], np.float32)
        self.people = np.array([indices[s] for s in frame.subject_uid.astype(str)], dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        raw = np.asarray(self.arrays[self.files[i]][self.rows[i]], dtype=np.float32).copy()
        if hashlib.sha256(raw.tobytes(order="C")).hexdigest() != self.hashes[i]:
            raise ValueError(f"raw PPG content hash mismatch at allowed row {i}")
        scale = float(raw.std())
        if not np.isfinite(raw).all() or scale <= 1e-8:
            raise ValueError("invalid PPG window")
        x = (raw - float(raw.mean())) / scale
        return torch.from_numpy(x[None]), torch.from_numpy(self.anchors[i]), int(self.people[i])


def run(args):
    started = time.monotonic()
    seed_everything(20260907)
    if not torch.cuda.is_available():
        raise RuntimeError("export requires an allocated GPU")
    device = torch.device("cuda")
    if torch.cuda.get_device_name(0) != SOURCES[args.split_mode]["gpu"]:
        raise ValueError("use the source run's GPU type for strict reference reproduction")
    metadata, checkpoint = check_source(args.run, args.split_mode, args.store_root)
    train, validation = load_screen_metadata(args.store_root, args.split_mode)
    scaler = checkpoint["target_scaler"]
    if scaler != _fit_scaler(train):
        raise ValueError("source scaler differs from train-only fit")
    indices = checkpoint["subject_to_index"]
    if sorted(indices) != sorted(train.subject_uid.unique()) or sorted(indices.values()) != list(range(len(indices))):
        raise ValueError("source personal-state mapping mismatch")
    means = fit_subject_train_means(train)
    model = LoraPRSRegressor(PRS_MODELS["lora_continue"], subject_count=len(indices)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "exporting", "screen_id": SCREEN_ID,
                "protocol_id": metadata["protocol_id"], "split_mode": args.split_mode,
                "source_parent_split": "meta_train", "read_roles": ["train", "internal_validation"],
                "heldout_test_accessed": False, "source_run": str(args.run),
                "source_checkpoint_sha256": metadata["checkpoint_sha256"],
                "store_manifest_sha256": metadata["store_manifest_sha256"],
                "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
                "target_scaler": scaler, "personal_training_budget": 320,
                "retrieval_count": 5, "parameters_updated": False,
                "feature_provenance": "frozen_model_fitted_on_all_train_not_cross_fitted",
                "train_prediction_provenance": "in_sample", "files": {},
                "clock_policy": "record-local; no cross-record chronological inference",
                "gpu": torch.cuda.get_device_name(0), "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    save_json(args.output / "manifest.json", manifest)
    captured = {}
    hook = model.base.residual_head.register_forward_pre_hook(
        lambda _module, inputs: captured.update(features=inputs[0].detach()))
    mean = torch.tensor(scaler["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(scaler["std"], dtype=torch.float32, device=device)
    try:
        for prefix, frame in [("train", train), ("validation", validation)]:
            canonical = canonical_metadata(frame)
            dataset = ExportDataset(frame, args.store_root, means, indices, scaler)
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
            features = np.lib.format.open_memmap(args.output / f"{prefix}_features.npy", mode="w+", dtype=np.float32, shape=(len(frame), 256))
            prediction = np.lib.format.open_memmap(args.output / f"{prefix}_base.npy", mode="w+", dtype=np.float32, shape=(len(frame), 2))
            offset = 0
            with torch.inference_mode():
                for x, anchor, person in loader:
                    with _autocast(device):
                        y = model(x.to(device), anchor.to(device), subject_index=person.to(device))
                    stop = offset + len(x)
                    features[offset:stop] = captured["features"].float().cpu().numpy()
                    prediction[offset:stop] = (y.float() * std + mean).cpu().numpy()
                    offset = stop
                    if offset % (64 * 500) == 0:
                        print(json.dumps({"phase": "export", "role": prefix, "rows": offset, "total": len(frame)}), flush=True)
            features.flush()
            prediction.flush()
            if not np.isfinite(features).all() or not np.isfinite(prediction).all():
                raise ValueError("nonfinite frozen outputs")
            if prefix == "validation":
                saved = pd.read_parquet(args.run / "best_internal_validation_predictions.parquet")
                joined = canonical[["subject_uid", "event_id", "source"]].merge(saved, on=["subject_uid", "event_id", "source"], how="left", validate="one_to_one")
                if len(joined) != 82040 or joined.isna().any().any():
                    raise ValueError("reference validation prediction keys mismatch")
                manifest["reference_target_representation_check"] = check_reference_targets(
                    joined[["target_sbp", "target_dbp"]], frame[["sbp", "dbp"]])
                difference = np.abs(np.asarray(prediction) - joined[["pred_sbp", "pred_dbp"]].to_numpy())
                manifest["reference_reproduction"] = {"max_abs_mmhg": float(difference.max()), "mean_abs_mmhg": float(difference.mean())}
                if difference.max() > 0.05 or difference.mean() > 0.005:
                    raise ValueError(f"frozen reference reproduction failed: {manifest['reference_reproduction']}")
            canonical.to_parquet(args.output / f"{prefix}_metadata.parquet", index=False)
            np.save(args.output / f"{prefix}_bp.npy", frame[["sbp", "dbp"]].to_numpy(dtype=np.float32))
            manifest[f"{prefix}_rows"] = len(frame)
            del features, prediction, dataset, loader
    finally:
        hook.remove()
    for p in sorted(args.output.iterdir()):
        if p.suffix in {".npy", ".parquet"}:
            manifest["files"][p.name] = {"path": p.name, "sha256": file_sha256(p)}
    manifest.update(status="exported", runtime_seconds=time.monotonic() - started,
                    raw_content_hashes_verified=manifest["train_rows"] + manifest["validation_rows"])
    save_json(args.output / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--store-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split-mode", choices=list(SOURCES), required=True)
    print(json.dumps(run(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()
