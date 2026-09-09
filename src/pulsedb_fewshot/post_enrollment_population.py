"""Fresh population fit excluding all enrollment subjects; no old checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from .official_calbased_train import (fit_personal_state, _loader, _predict,
    participant_score, save_json, sha256, event_ids_sha256)
from .post_enrollment_protocol import SEED, partition, load_plan


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_device(device, synthetic):
    if device == "cuda" and (not torch.cuda.is_available() or not os.getenv("SLURM_JOB_ID")):
        raise RuntimeError("CUDA requires a Slurm GPU allocation")
    if device == "cpu" and not synthetic:
        raise ValueError("formal training requires Slurm GPU; CPU is synthetic only")
    torch.set_num_threads(1 if synthetic else 2)


def shared_digest(state):
    """Includes frozen buffers as well as parameters; excludes personal adapters."""
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name.startswith(("base.lora_a.", "base.lora_b.")):
            continue
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(args):
    check_device(args.device, args.synthetic)
    plan, frame = partition(args.plan, "population_train", synthetic=args.synthetic)
    protocol_id = plan["protocol_id"]
    args.store_root = Path(plan["source_root"])
    if args.patience != 8 or args.seed != SEED or (args.epochs and not args.synthetic):
        raise ValueError("prespecified seed/patience or formal epoch policy changed")
    if args.stage == "inner":
        fit = frame.loc[frame.inner_role.eq("train")].copy()
        validation = frame.loc[frame.inner_role.eq("internal_validation")].copy()
        fixed_epochs = args.epochs
        selection_sha = None
    else:
        if args.selection_run is None:
            raise ValueError("final requires a completed inner-selection run")
        selection_path = args.selection_run / "run.json"
        source = json.loads(selection_path.read_text())
        if source.get("status") != "complete" or source.get("protocol_id") != protocol_id or source.get("stage") != "inner" or source.get("plan_sha256") != sha256(args.plan) or source.get("test_targets_accessed") is not False:
            raise ValueError("invalid population epoch selection provenance")
        for key in ("seed", "batch_size", "learning_rate", "weight_decay", "huber_delta", "gradient_clip"):
            if source["arguments"][key] != getattr(args, key):
                raise ValueError("final settings differ from inner fit")
        fixed_epochs = int(source["best_epoch"])
        if fixed_epochs < 1:
            raise ValueError("invalid selected population epoch")
        selection_sha = sha256(selection_path)
        fit, validation = frame, None
    if set(fit.subject_uid) & set(plan["selected_subjects"]):
        raise ValueError("enrollment people exposed to population model")
    scaler, anchors, mapping = fit_personal_state(fit)
    args.output.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    from .lora_prs_models import LoraPRSRegressor, PRS_MODELS
    from .training import source_tree_sha256
    model = LoraPRSRegressor(PRS_MODELS["lora_continue"], subject_count=len(mapping)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    objective = torch.nn.HuberLoss(delta=args.huber_delta)
    train_loader = _loader(fit, args, scaler, anchors, mapping, targets=True, shuffle=True)
    val_loader = None if validation is None else _loader(validation, args, scaler, anchors, mapping, targets=False, shuffle=False)
    report = {"protocol_id": protocol_id, "stage": args.stage, "status": "running",
        "plan_sha256": sha256(args.plan), "synthetic": args.synthetic,
        "started_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "old_checkpoint_used": False, "initialization": "from_scratch",
        "test_targets_accessed": False, "enrollment_labels_accessed": False,
        "fit_subjects": len(mapping), "fit_rows": len(fit), "target_scaler": scaler,
        "fit_ids_sha256": event_ids_sha256(fit.segment_uid),
        "fit_subject_ids_sha256": event_ids_sha256(mapping),
        "selection_run_sha256": selection_sha,
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    save_json(args.output / "run.json", report)
    save_json(args.output / "subject_index.json", mapping)
    fit[["subject_uid", "segment_uid", "source"]].to_parquet(args.output / "fit_manifest.parquet", index=False)
    anchors.reset_index().to_parquet(args.output / "anchors.parquet", index=False)
    history, best, best_epoch, stale, steps = [], float("inf"), 0, 0, 0
    started = time.monotonic()
    try:
        epoch = 0
        while True:
            epoch += 1
            model.train()
            loss_sum, n = 0., 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                output = model(batch["ppg"].to(args.device), batch["anchor"].to(args.device), subject_index=batch["person"].to(args.device))
                loss = objective(output, batch["target"].to(args.device))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite population loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                steps += 1
                n += len(output)
                loss_sum += float(loss.detach()) * len(output)
            row = {"epoch": epoch, "train_loss": loss_sum / n, "optimizer_steps": steps}
            improved = True
            if val_loader is not None:
                score = participant_score(validation, _predict(model, val_loader, args.device, scaler))
                row["validation_mean_mae"] = score
                improved = score < best - 1e-8
                if improved:
                    best, stale = score, 0
                else:
                    stale += 1
            if improved:
                best_epoch = epoch
                torch.save({"model_state": model.state_dict(), "protocol_id": protocol_id,
                    "stage": args.stage, "subject_to_index": mapping, "target_scaler": scaler,
                    "subject_anchors": anchors.to_dict("index"), "epoch": epoch,
                    "plan_sha256": sha256(args.plan)}, args.output / "best.pt")
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "protocol_id": protocol_id, "stage": args.stage}, args.output / "last.pt")
            history.append(row)
            save_json(args.output / "history.json", history)
            print(json.dumps(row), flush=True)
            if (fixed_epochs and epoch >= fixed_epochs) or (val_loader is not None and stale >= args.patience):
                break
        checkpoint = torch.load(args.output / "best.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        if validation is not None:
            from .calbased_metrics import participant_macro_views, pooled_diagnostics
            scored = validation[["subject_uid", "segment_uid", "source", "sbp", "dbp"]].rename(columns={"segment_uid": "event_id", "sbp": "target_sbp", "dbp": "target_dbp"})
            scored[["pred_sbp", "pred_dbp"]] = _predict(model, val_loader, args.device, scaler)
            scored.to_parquet(args.output / "validation_predictions.parquet", index=False)
            save_json(args.output / "validation_macro.json", participant_macro_views(scored))
            pooled_diagnostics(scored, "population_inner_lora").to_csv(args.output / "validation_diagnostics.csv", index=False)
        report.update(status="complete", best_epoch=best_epoch, epochs_completed=epoch,
            stop_reason="fixed_selected_epochs" if fixed_epochs else "patience8",
            optimizer_steps=steps, checkpoint_sha256=sha256(args.output / "best.pt"),
            shared_state_sha256=shared_digest(model.state_dict()), runtime_seconds=time.monotonic() - started,
            completed_utc=datetime.now(timezone.utc).isoformat())
        save_json(args.output / "run.json", report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", runtime_seconds=time.monotonic() - started)
        save_json(args.output / "run.json", report)
        raise
    return report


def load_population(run_root, plan_path, *, device="cpu", synthetic=False):
    """Verify exclusion/checkpoint lineage before loading any transferable state."""
    run_root = Path(run_root)
    plan = load_plan(plan_path, synthetic=synthetic)
    report = json.loads((run_root / "run.json").read_text())
    if report.get("protocol_id") != plan["protocol_id"] or report.get("stage") != "final" or report.get("status") != "complete" or report.get("plan_sha256") != sha256(plan_path) or report.get("old_checkpoint_used") is not False or report.get("enrollment_labels_accessed") is not False or report.get("test_targets_accessed") is not False:
        raise ValueError("invalid fresh population checkpoint provenance")
    if sha256(run_root / "best.pt") != report["checkpoint_sha256"]:
        raise ValueError("population checkpoint changed")
    checkpoint = torch.load(run_root / "best.pt", map_location=device, weights_only=False)
    mapping = checkpoint["subject_to_index"]
    expected = plan["audit"]["population_train"]
    if set(mapping) & set(plan["selected_subjects"]) or event_ids_sha256(mapping) != expected["subject_ids_sha256"] or report["fit_ids_sha256"] != expected["segment_ids_sha256"] or report["fit_rows"] != expected["rows"]:
        raise ValueError("population fit did not exclude exactly the selected people")
    if checkpoint.get("plan_sha256") != sha256(plan_path) or shared_digest(checkpoint["model_state"]) != report["shared_state_sha256"]:
        raise ValueError("shared-state provenance mismatch")
    return plan, report, checkpoint


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=("inner", "final"), required=True)
    p.add_argument("--selection-run", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=.5)
    p.add_argument("--gradient-clip", type=float, default=5.)
    p.add_argument("--patience", type=int, default=8)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
