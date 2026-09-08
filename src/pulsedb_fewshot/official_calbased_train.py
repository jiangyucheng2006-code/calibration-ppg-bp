"""Fresh persistent-LoRA training under the exact official CalBased protocol.

The store is prepared separately from the official membership files. Inner
validation is a subset of official TRAIN, never the official test. OOF models
are initialized from scratch and exclude their complete assigned fold from
encoder, personal adapters, BP anchors and target scaling. No old checkpoint
or official test target can enter this runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd

from .official_time_policy import TIME_BOUNDARY_POLICY, sample_span_audit

try:  # Contract-only tests can run without the optional training dependency.
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


PROTOCOL_ID = "pulsedb-official-calbased-v1"
STAGES = ("inner", "oof", "final")
REQUIRED = {"subject_uid", "segment_uid", "source", "waveform_file",
            "waveform_row", "ppg_content_sha256", "record_id", "start_time_s",
            "end_time_s", "duration_s", "sample_interval_s", "inner_role",
            "sbp", "dbp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_ids_sha256(values) -> str:
    return hashlib.sha256(("\n".join(sorted(map(str, values))) + "\n").encode()).hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_training_frame(frame: pd.DataFrame, *, smoke: bool = False) -> pd.DataFrame:
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"official train manifest missing {sorted(missing)}")
    f = frame.copy().reset_index(drop=True)
    for name in ("subject_uid", "segment_uid", "source", "record_id", "waveform_file"):
        if f[name].isna().any() or f[name].astype(str).eq("").any():
            raise ValueError(f"missing identity/path in {name}")
        f[name] = f[name].astype(str)
    if set(f.source) != {"MIMIC", "VitalDB"}:
        raise ValueError("official CalBased train must include both source strata")
    if f.segment_uid.duplicated().any():
        raise ValueError("duplicate official segment identity")
    if f[["subject_uid", "source"]].drop_duplicates().subject_uid.duplicated().any():
        raise ValueError("participant has inconsistent source")
    if set(f.inner_role) != {"train", "internal_validation"}:
        raise ValueError("official train contains forbidden inner role")
    if not f.ppg_content_sha256.astype(str).str.fullmatch("[0-9a-f]{64}").all():
        raise ValueError("invalid PPG content hash")
    for name in ("waveform_row", "start_time_s", "end_time_s", "duration_s", "sample_interval_s", "sbp", "dbp"):
        f[name] = pd.to_numeric(f[name], errors="raise")
        if not np.isfinite(f[name].to_numpy(dtype=float)).all():
            raise ValueError(f"nonfinite {name}")
    if not ((f.waveform_row >= 0) & (f.waveform_row == np.floor(f.waveform_row))).all():
        raise ValueError("invalid waveform row index")
    f["waveform_row"] = f.waveform_row.astype(np.int64)
    if not np.allclose(f.duration_s, 10, rtol=0, atol=1e-4):
        raise ValueError("official windows must last 10 seconds")
    if not np.allclose(f.sample_interval_s, 1 / 125, rtol=0, atol=1e-6):
        raise ValueError("official PPG must contain 125 Hz samples")
    if not np.allclose(f.end_time_s + f.sample_interval_s,
                       f.start_time_s + f.duration_s, rtol=0, atol=1e-4):
        raise ValueError("inconsistent sample-span duration and sample interval")
    counts = f.groupby(["subject_uid", "inner_role"]).size().unstack(fill_value=0)
    if (counts == 0).any().any():
        raise ValueError("every registered subject needs both inner roles")
    if not smoke and (len(counts) != 2506 or not counts["train"].eq(320).all()
                      or not counts["internal_validation"].eq(40).all()):
        raise ValueError("official cohort requires 2506 people with 320/40 inner train/validation")
    # A repeated participant is intentional; exact data reuse across roles is not.
    if f.groupby("ppg_content_sha256").inner_role.nunique().gt(1).any():
        raise ValueError("PPG content crosses inner roles")
    assert_cross_role_intervals(f, "inner_role")
    return f


def assert_cross_role_intervals(frame: pd.DataFrame, role: str) -> None:
    if sample_span_audit(frame, role)["cross_role_overlap_pairs"]:
        raise ValueError("physiological interval crosses model fitting boundary")


def load_store(root: Path, *, smoke: bool = False):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("status") != "ready":
        raise ValueError("official store is not approved/ready")
    if smoke != bool(manifest.get("synthetic", False)):
        raise ValueError("smoke requires synthetic store; real stores require formal validation")
    if not smoke and manifest.get("time_boundary_policy") != TIME_BOUNDARY_POLICY:
        raise ValueError("official store requires the recorded sample-span time contract")
    # Deliberately do not enumerate/open test files or raw subject MAT files.
    path = root / "train_manifest.parquet"
    frame = validate_training_frame(pd.read_parquet(path), smoke=smoke)
    declared = manifest.get("train_manifest_sha256")
    if declared is not None and declared != sha256(path):
        raise ValueError("official training manifest hash changed")
    for name in frame.waveform_file.unique():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError("waveform file missing or escaping official store")
    return manifest, frame


def select_stage(frame: pd.DataFrame, stage: str, fold: int | None = None):
    if stage not in STAGES:
        raise ValueError("unsupported official stage")
    inner = frame.loc[frame.inner_role.eq("train")].copy()
    validation = frame.loc[frame.inner_role.eq("internal_validation")].copy()
    if stage == "inner":
        return inner, {"train": inner, "validation": validation}
    if stage == "final":
        return frame.copy(), {"train": frame.copy()}
    if fold not in (0, 1, 2) or "inner_fold" not in inner:
        raise ValueError("OOF needs a prespecified inner_fold and --fold 0/1/2")
    if set(inner.inner_fold) != {0, 1, 2}:
        raise ValueError("OOF fold assignment must be exactly 0/1/2")
    assignment = inner.groupby("subject_uid").inner_fold.nunique()
    if not assignment.eq(3).all():
        raise ValueError("every registered subject must contribute to each OOF fold")
    held = inner.loc[inner.inner_fold.eq(fold)].copy()
    fit = inner.loc[~inner.inner_fold.eq(fold)].copy()
    overlap = set(held.ppg_content_sha256) & set(fit.ppg_content_sha256)
    if overlap:
        raise ValueError("OOF held fold reuses fit waveform content")
    combined = pd.concat([fit.assign(fit_role="source_fit"), held.assign(fit_role="excluded_fold")])
    assert_cross_role_intervals(combined, "fit_role")
    return fit, {"source_fit": fit, "excluded_fold": held, "outer_validation": validation}


def fit_personal_state(fit: pd.DataFrame):
    values = fit[["sbp", "dbp"]].to_numpy(dtype=np.float64)
    mean, std = values.mean(0), values.std(0)
    if not np.isfinite(values).all() or (std <= 0).any():
        raise ValueError("invalid fit-only target scaler")
    anchors = fit.groupby("subject_uid", sort=True)[["sbp", "dbp"]].mean()
    mapping = {str(subject): i for i, subject in enumerate(anchors.index)}
    return {"mean": mean.tolist(), "std": std.tolist()}, anchors, mapping


def canonical_metadata(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    recording = frame.source.astype(str) + ":" + frame.record_id.astype(str)
    out = pd.DataFrame({"subject_uid": frame.subject_uid.astype(str),
        "event_id": frame.segment_uid.astype(str), "window_uid": frame.segment_uid.astype(str),
        "source": frame.source.astype(str), "recording_uid": recording,
        "time_axis_uid": recording, "start_s": frame.start_time_s.astype(float),
        "end_s": frame.end_time_s.astype(float),
        "duration_s": frame.duration_s.astype(float),
        "sample_interval_s": frame.sample_interval_s.astype(float),
        "time_boundary_policy": TIME_BOUNDARY_POLICY,
        "waveform_sha256": frame.ppg_content_sha256.astype(str), "role": role})
    if role in {"source_fit", "excluded_fold", "outer_validation"}:
        out["oof_role"] = role
    # Labels live in a separate file, not in prediction inputs/identity metadata.
    return out.reset_index(drop=True)


def resolve_fixed_epochs(args, manifest_hash: str) -> tuple[int, dict]:
    if args.stage == "inner":
        if args.epochs and not args.smoke:
            raise ValueError("formal inner training has patience-8 stopping, no fixed epoch cap")
        return int(args.epochs or 0), {}
    if args.stage == "oof":
        if args.selection_run is not None or args.epochs:
            raise ValueError("OOF epochs are prespecified; selection-run/epochs cannot set them")
        if not args.smoke and args.oof_epochs != 25:
            raise ValueError("formal OOF training budget is prespecified as 25 epochs")
        if args.oof_epochs <= 0:
            raise ValueError("OOF epochs must be positive")
        return int(args.oof_epochs), {"fixed_epochs": int(args.oof_epochs),
            "epoch_selection_role": "none_prespecified", "epoch_policy": "prespecified_smoke" if args.smoke else "prespecified_25_epochs",
            "epoch_source_fit_included_all_inner_training_labels": False}
    if args.selection_run is None:
        if not args.smoke:
            raise ValueError("OOF/final require --selection-run from completed inner training")
        if args.epochs <= 0:
            raise ValueError("synthetic fixed-epoch stage requires positive --epochs")
        return int(args.epochs), {}
    path = args.selection_run / "run.json"
    selected = json.loads(path.read_text(encoding="utf-8"))
    if not (selected.get("status") == "complete" and selected.get("stage") == "inner"
            and selected.get("protocol_id") == PROTOCOL_ID
            and selected.get("store_manifest_sha256") == manifest_hash
            and selected.get("official_test_targets_accessed") is False):
        raise ValueError("invalid inner epoch-selection provenance")
    if selected.get("seed") != args.seed:
        raise ValueError("fixed-epoch source must use the same base seed")
    for key in ("batch_size", "learning_rate", "weight_decay", "huber_delta", "gradient_clip"):
        if key not in selected.get("arguments", {}) or selected["arguments"][key] != getattr(args, key):
            raise ValueError(f"final refit must preserve inner selected training setting {key}")
    epochs = int(selected["best_epoch"])
    if epochs < 1 or (args.epochs and args.epochs != epochs):
        raise ValueError("fixed epochs must equal the selected inner epoch")
    return epochs, {"epoch_selection_run_sha256": sha256(path), "fixed_epochs": epochs,
                    "epoch_selection_role": "official_train_internal_validation",
                    "epoch_policy": "inherited_from_full_inner_fit_outer_validation_selection",
                    "epoch_source_fit_included_all_inner_training_labels": True}


class OfficialDataset(Dataset):
    def __init__(self, frame, root, scaler, anchors, mapping, *, targets: bool):
        if torch is None:
            raise RuntimeError("install optional torch training dependency")
        self.files = frame.waveform_file.astype(str).to_numpy()
        self.rows = frame.waveform_row.to_numpy(dtype=np.int64)
        self.hashes = frame.ppg_content_sha256.astype(str).to_numpy()
        self.arrays = {name: np.load(root / name, mmap_mode="r") for name in set(self.files)}
        for name, array in self.arrays.items():
            if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 1250:
                raise ValueError(f"invalid float32 [N,1250] waveform shard {name}")
        self.mean = np.asarray(scaler["mean"], np.float32)
        self.std = np.asarray(scaler["std"], np.float32)
        raw_anchors = anchors.reindex(frame.subject_uid)[["sbp", "dbp"]].to_numpy(dtype=np.float32)
        if not np.isfinite(raw_anchors).all():
            raise ValueError("prediction person has no permitted fitted personal state")
        self.anchors = (raw_anchors - self.mean) / self.std
        self.people = np.array([mapping[s] for s in frame.subject_uid], dtype=np.int64)
        self.targets = ((frame[["sbp", "dbp"]].to_numpy(dtype=np.float32) - self.mean) / self.std) if targets else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        raw = np.asarray(self.arrays[self.files[i]][self.rows[i]], dtype=np.float32).copy()
        if hashlib.sha256(raw.tobytes(order="C")).hexdigest() != self.hashes[i]:
            raise ValueError("raw PPG content changed after official materialization")
        std = float(raw.std())
        if not np.isfinite(raw).all() or std <= 1e-8:
            raise ValueError("nonfinite or constant PPG")
        item = {"ppg": torch.from_numpy(((raw - float(raw.mean())) / std)[None]),
                "anchor": torch.from_numpy(self.anchors[i]), "person": int(self.people[i])}
        if self.targets is not None:
            item["target"] = torch.from_numpy(self.targets[i])
        return item


def _loader(frame, args, scaler, anchors, mapping, *, targets, shuffle):
    return torch.utils.data.DataLoader(OfficialDataset(frame, args.store_root, scaler, anchors, mapping, targets=targets),
        batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=args.device == "cuda", persistent_workers=args.workers > 0)


def _predict(model, loader, device, scaler):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            y = model(batch["ppg"].to(device), batch["anchor"].to(device), subject_index=batch["person"].to(device))
            predictions.append(y.float().cpu().numpy())
    result = np.concatenate(predictions) * np.asarray(scaler["std"], np.float32) + np.asarray(scaler["mean"], np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite LoRA prediction")
    return result


def participant_score(frame, prediction):
    error = np.abs(np.asarray(prediction, float) - frame[["sbp", "dbp"]].to_numpy(dtype=float))
    table = pd.DataFrame(error, columns=["sbp", "dbp"])
    table["subject_uid"] = frame.subject_uid.to_numpy()
    return float(table.groupby("subject_uid")[["sbp", "dbp"]].mean().to_numpy().mean())


def export_cache(model, frame, role, args, scaler, anchors, mapping, provenance):
    root = args.output / "cache" / role
    root.mkdir(parents=True, exist_ok=False)
    has_labels = role != "test_inputs"
    loader = _loader(frame, args, scaler, anchors, mapping, targets=False, shuffle=False)
    features = np.lib.format.open_memmap(root / "features.npy", mode="w+", dtype=np.float32, shape=(len(frame), 256))
    base = np.lib.format.open_memmap(root / "base_bp.npy", mode="w+", dtype=np.float32, shape=(len(frame), 2))
    capture = {}
    hook = model.base.residual_head.register_forward_pre_hook(lambda _m, inputs: capture.update(z=inputs[0].detach()))
    offset = 0
    model.eval().requires_grad_(False)
    try:
        with torch.inference_mode():
            for batch in loader:
                y = model(batch["ppg"].to(args.device), batch["anchor"].to(args.device), subject_index=batch["person"].to(args.device))
                stop = offset + len(y)
                features[offset:stop] = capture["z"].float().cpu().numpy()
                base[offset:stop] = y.float().cpu().numpy() * np.asarray(scaler["std"], np.float32) + np.asarray(scaler["mean"], np.float32)
                offset = stop
    finally:
        hook.remove()
    features.flush()
    base.flush()
    if offset != len(frame) or not np.isfinite(features).all() or not np.isfinite(base).all():
        raise ValueError("invalid/incomplete feature cache")
    canonical_metadata(frame, role).to_parquet(root / "metadata.parquet", index=False)
    if has_labels:
        np.save(root / "bp.npy", frame[["sbp", "dbp"]].to_numpy(dtype=np.float32))
    payload = dict(provenance, role=role, rows=len(frame), feature_dim=256,
                   time_boundary_policy=TIME_BOUNDARY_POLICY,
                   labels_present=has_labels, target_units="mmHg", status="complete")
    payload["files"] = {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file()}
    save_json(root / "provenance.json", payload)
    return payload


def load_test_inputs(root: Path, train: pd.DataFrame) -> pd.DataFrame:
    """Explicit final-only input access; no test labels are accepted or opened."""
    frame = pd.read_parquet(root / "test_inputs.parquet")
    forbidden = {"sbp", "dbp", "target_sbp", "target_dbp", "SegSBP", "SegDBP", "ABP"}
    if forbidden & set(frame) or any(str(c).lower().startswith(("abp", "target_", "segsbp", "segdbp")) for c in frame):
        raise ValueError("official test input table contains targets")
    missing = (REQUIRED - {"sbp", "dbp", "inner_role"}) - set(frame)
    if missing or frame.segment_uid.duplicated().any():
        raise ValueError("invalid official test input identities")
    if set(frame.subject_uid) != set(train.subject_uid):
        raise ValueError("official CalBased test subjects must match training")
    if not frame.groupby("subject_uid").size().eq(40).all():
        raise ValueError("official CalBased test requires exactly 40 inputs per subject")
    if set(frame.segment_uid) & set(train.segment_uid) or set(frame.ppg_content_sha256) & set(train.ppg_content_sha256):
        raise ValueError("official test input overlaps official training data")
    combined = pd.concat([train.assign(access_role="train"), frame.assign(access_role="test_inputs")])
    assert_cross_role_intervals(combined, "access_role")
    for name in frame.waveform_file.astype(str).unique():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("invalid official test waveform path")
    return frame.reset_index(drop=True)


def run(args):
    if torch is None:
        raise RuntimeError("optional torch dependency required for model training")
    if args.stage != "final" and args.include_test_inputs:
        raise ValueError("only frozen final refit can explicitly export test inputs")
    if args.epochs < 0 or args.learning_rate <= 0 or args.huber_delta <= 0 or args.gradient_clip <= 0 or args.weight_decay < 0:
        raise ValueError("invalid epoch/optimizer configuration")
    if args.stage != "oof" and args.fold is not None:
        raise ValueError("--fold applies only to OOF training")
    if args.device == "cuda" and (not torch.cuda.is_available() or not os.getenv("SLURM_JOB_ID")):
        raise RuntimeError("formal CUDA training requires an allocated Slurm GPU")
    if args.device == "cpu" and not args.smoke:
        raise ValueError("CPU is supported only for synthetic smoke; use allocated GPU for formal fits")
    if args.workers < 0 or args.batch_size < 2 or args.patience != 8:
        raise ValueError("invalid loader configuration or non-prespecified patience")
    if args.smoke and args.stage == "inner" and args.epochs < 1:
        raise ValueError("synthetic smoke must use a short positive --epochs bound")
    if args.smoke:
        torch.set_num_threads(1)
    args.store_root = args.store_root.resolve()
    _, frame = load_store(args.store_root, smoke=args.smoke)
    fit, exports = select_stage(frame, args.stage, args.fold)
    manifest_hash = sha256(args.store_root / "manifest.json")
    fixed_epochs, selection = resolve_fixed_epochs(args, manifest_hash)
    scaler, anchors, mapping = fit_personal_state(fit)
    args.output.mkdir(parents=True, exist_ok=False)
    seed = int(args.seed) + (int(args.fold) + 1 if args.stage == "oof" else 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from .lora_prs_models import LoraPRSRegressor, PRS_MODELS
    from .training import source_tree_sha256
    from .calbased_metrics import participant_macro_views, pooled_diagnostics
    model = LoraPRSRegressor(PRS_MODELS["lora_continue"], subject_count=len(mapping)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    objective = torch.nn.HuberLoss(delta=args.huber_delta)
    train_loader = _loader(fit, args, scaler, anchors, mapping, targets=True, shuffle=True)
    validation = exports.get("validation") if args.stage == "inner" else None
    validation_loader = (_loader(validation, args, scaler, anchors, mapping, targets=False, shuffle=False)
                         if validation is not None else None)
    excluded = exports.get("excluded_fold", pd.DataFrame({"segment_uid": []}))
    started = time.monotonic()
    report = {"protocol_id": PROTOCOL_ID, "stage": args.stage, "status": "running", "candidate": "lora",
        "time_boundary_policy": TIME_BOUNDARY_POLICY,
        "seed": args.seed, "effective_seed": seed, "fold": args.fold, "synthetic": args.smoke,
        "started_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "initialization": "from_scratch_no_previous_checkpoint", "old_checkpoint_used": False,
        "initial_checkpoint_eligible_for_selection": False,
        "official_test_targets_accessed": False, "official_test_inputs_accessed": False,
        "encoder_protocol": "model-level-exclusion" if args.stage == "oof" else "fit-only-supervision",
        "encoder_fit_event_ids_sha256": event_ids_sha256(fit.segment_uid),
        "excluded_event_ids_sha256": event_ids_sha256(excluded.segment_uid),
        "fit_rows": len(fit), "fit_subjects": len(mapping), "target_scaler": scaler,
        "scaler_and_anchor_fit_role": "source_fit" if args.stage == "oof" else "train",
        "store_manifest_sha256": manifest_hash, "train_manifest_sha256": sha256(args.store_root / "train_manifest.parquet"),
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "stored_personal_parameters_per_subject": 2048,
        "selection_role": "official_train_internal_validation" if args.stage == "inner" else "none_fixed_epochs",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, **selection}
    save_json(args.output / "run.json", report)
    save_json(args.output / "subject_index.json", mapping)
    anchors.reset_index().to_parquet(args.output / "subject_train_anchors.parquet", index=False)
    fit[["subject_uid", "segment_uid", "source"]].to_parquet(args.output / "encoder_fit_manifest.parquet", index=False)
    excluded.to_parquet(args.output / "excluded_manifest.parquet", index=False)
    history, best, best_epoch, stale, step = [], float("inf"), 0, 0, 0
    try:
        epoch = 0
        while True:
            epoch += 1
            model.train()
            loss_sum, n = 0.0, 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                y = model(batch["ppg"].to(args.device), batch["anchor"].to(args.device), subject_index=batch["person"].to(args.device))
                loss = objective(y, batch["target"].to(args.device))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(y)
                n += len(y)
                step += 1
            row = {"epoch": epoch, "train_loss": loss_sum / n, "optimizer_steps": step}
            improved = True
            if validation_loader is not None:
                pred = _predict(model, validation_loader, args.device, scaler)
                score = participant_score(validation, pred)
                row["validation_mean_mae"] = score
                improved = score < best - 1e-8
                if improved:
                    best, stale = score, 0
                else:
                    stale += 1
            if improved:
                best_epoch = epoch
                torch.save({"model_state": model.state_dict(), "subject_to_index": mapping,
                    "target_scaler": scaler, "subject_anchors": anchors.to_dict("index"),
                    "epoch": epoch, "protocol_id": PROTOCOL_ID, "stage": args.stage,
                    "encoder_fit_event_ids_sha256": report["encoder_fit_event_ids_sha256"]}, args.output / "best.pt")
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch, "seed": seed, "best_epoch": best_epoch,
                        "protocol_id": PROTOCOL_ID}, args.output / "last.pt")
            history.append(row)
            save_json(args.output / "history.json", history)
            print(json.dumps(row), flush=True)
            if (fixed_epochs and epoch >= fixed_epochs) or (validation is not None and stale >= args.patience):
                break
        checkpoint = torch.load(args.output / "best.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        report.update(best_epoch=best_epoch, epochs_completed=epoch, optimizer_steps=step,
                      stop_reason="fixed_epochs" if fixed_epochs else "early_stopping",
                      checkpoint_sha256=sha256(args.output / "best.pt"))
        if args.include_test_inputs:
            exports["test_inputs"] = load_test_inputs(args.store_root, frame)
            report["official_test_inputs_accessed"] = True
        provenance = {key: report[key] for key in ("protocol_id", "stage", "fold", "seed", "effective_seed",
            "encoder_protocol", "encoder_fit_event_ids_sha256", "excluded_event_ids_sha256", "checkpoint_sha256",
            "store_manifest_sha256", "official_test_targets_accessed", "old_checkpoint_used")}
        provenance.update(target_scaler=scaler, fitted_personal_rows=len(fit),
                          bp_units="mmHg", official_test_accessed=report["official_test_inputs_accessed"],
                          official_contract_sha256=manifest_hash,
                          fit_transforms_on_source_fit_only=True,
                          encoder_initialization="scratch", synthetic_smoke=args.smoke,
                          model_selection_excluded_fold_labels_used=False,
                          epoch_policy=report.get("epoch_policy", "internal_validation_early_stopping"),
                          epoch_source_fit_included_all_inner_training_labels=report.get(
                              "epoch_source_fit_included_all_inner_training_labels", False),
                          subject_index_sha256=sha256(args.output / "subject_index.json"),
                          fit_manifest_sha256=sha256(args.output / "encoder_fit_manifest.parquet"))
        report["caches"] = {}
        for role, rows in exports.items():
            report["caches"][role] = export_cache(model, rows, role, args, scaler, anchors, mapping, provenance)
            if role == "test_inputs":
                inputs_only = canonical_metadata(rows, role)[["subject_uid", "event_id", "source"]]
                inputs_only[["pred_sbp", "pred_dbp"]] = np.load(args.output / "cache" / role / "base_bp.npy")
                inputs_only.to_parquet(args.output / "official_test_input_predictions.parquet", index=False)
                report["predictions_sha256"] = sha256(args.output / "official_test_input_predictions.parquet")
                report["frozen_configuration"] = True
            if role in {"validation", "excluded_fold", "outer_validation"}:
                predictions = canonical_metadata(rows, role)[["subject_uid", "event_id", "source"]]
                targets = rows[["sbp", "dbp"]].to_numpy(dtype=np.float32)
                predictions[["target_sbp", "target_dbp"]] = targets
                predictions[["pred_sbp", "pred_dbp"]] = np.load(args.output / "cache" / role / "base_bp.npy")
                predictions.to_parquet(args.output / f"{role}_predictions.parquet", index=False)
                save_json(args.output / f"{role}_metrics.json", participant_macro_views(predictions))
                pooled_diagnostics(predictions, "official_lora").to_csv(args.output / f"{role}_diagnostics.csv", index=False)
        report.update(status="complete", runtime_seconds=time.monotonic() - started,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        save_json(args.output / "run.json", report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", runtime_seconds=time.monotonic() - started)
        save_json(args.output / "run.json", report)
        raise
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--fold", type=int, choices=(0, 1, 2))
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--oof-epochs", type=int, default=25)
    p.add_argument("--selection-run", type=Path)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=0.5)
    p.add_argument("--gradient-clip", type=float, default=5.0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--include-test-inputs", action="store_true")
    return p


def main():
    print(json.dumps(run(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
