"""Fresh shared fitting and paired personal enrollment on the old outer split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .legacy_enrollment_protocol import PROTOCOL, SEED, CONFIG, load_plan, partition, validation_targets
from .official_calbased_train import fit_personal_state, _loader, _predict, participant_score, sha256, save_json, event_ids_sha256
from .post_enrollment_population import seed_all, check_device, shared_digest
from .post_enrollment_personal import fresh_person_model, PERSONAL_KEYS, run_profiles


def check_arguments(args):
    check_device(args.device, args.synthetic)
    for name in ("learning_rate", "weight_decay", "huber_delta", "gradient_clip", "batch_size", "patience"):
        if getattr(args, name) != CONFIG[name]:
            raise ValueError(f"frozen training setting changed: {name}")
    if args.epochs and not args.synthetic:
        raise ValueError("formal fitting has patience-eight stopping, no epoch cap")


def population(args):
    check_arguments(args)
    plan, fit = partition(args.plan, "population_train", access="population", synthetic=args.synthetic)
    _, bank = partition(args.plan, "validation_registration", access="population", synthetic=args.synthetic)
    _, query = partition(args.plan, "validation_inputs", access="population", synthetic=args.synthetic)
    targets = validation_targets(args.plan)
    validation = query.merge(targets, on=["subject_uid", "segment_uid", "source"], validate="one_to_one")
    if set(fit.subject_uid) & (set(bank.subject_uid) | set(plan["test_subjects"])):
        raise ValueError("new participants entered shared-model fitting")
    args.store_root = Path(plan["source_root"])
    scaler, anchors, mapping = fit_personal_state(fit)
    val_anchors = bank.groupby("subject_uid")[["sbp", "dbp"]].mean()
    args.output.mkdir(parents=True, exist_ok=False)
    seed_all(SEED)
    from .lora_prs_models import LoraPRSRegressor, PRS_MODELS
    from .training import source_tree_sha256
    model = LoraPRSRegressor(PRS_MODELS["lora_continue"], subject_count=len(mapping)).to(args.device)
    # The selector asks whether shared features generalize to validation people.
    # It has ZERO personal correction, only their permitted registration anchor.
    # It is an epoch-selection criterion, not an extra reported candidate.
    selector = fresh_person_model(model.state_dict(), SEED + 1, args.device)
    seed_all(SEED + 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    objective = torch.nn.HuberLoss(delta=args.huber_delta)
    loader = _loader(fit, args, scaler, anchors, mapping, targets=True, shuffle=True)
    val_loader = _loader(query, args, scaler, val_anchors, {s: 0 for s in bank.subject_uid.unique()}, targets=False, shuffle=False)
    report = {"protocol_id": PROTOCOL, "status": "running", "stage": "population",
              "plan_sha256": sha256(args.plan), "old_checkpoint_used": False,
              "test_targets_accessed": False, "test_registration_accessed": False,
              "shared_fit_role": "meta_train_only", "checkpoint_selection_role": "meta_validation",
              "selection_method": CONFIG["population_selection"],
              "fit_rows": len(fit), "fit_subjects": len(mapping),
              "fit_ids_sha256": event_ids_sha256(fit.segment_uid),
              "fit_subject_ids_sha256": event_ids_sha256(mapping),
              "target_scaler": scaler, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
              "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[2]),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    save_json(args.output / "run.json", report)
    save_json(args.output / "subject_index.json", mapping)
    fit[["subject_uid", "segment_uid", "source"]].to_parquet(args.output / "fit_manifest.parquet", index=False)
    best, best_epoch, stale, steps = float("inf"), 0, 0, 0
    history, started = [], time.monotonic()
    try:
        epoch = 0
        while True:
            epoch += 1
            model.train()
            loss_total, n = 0., 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(batch["ppg"].to(args.device), batch["anchor"].to(args.device), subject_index=batch["person"].to(args.device))
                loss = objective(pred, batch["target"].to(args.device))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite population loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                steps += 1
                n += len(pred)
                loss_total += float(loss.detach()) * len(pred)
            state = selector.state_dict()
            state.update({k: v for k, v in model.state_dict().items() if k not in PERSONAL_KEYS})
            selector.load_state_dict(state, strict=True)
            if torch.count_nonzero(selector.base.lora_b.weight).item() != 0:
                raise ValueError("selection borrowed a training person's adapter")
            val_pred = _predict(selector, val_loader, args.device, scaler)
            score = participant_score(validation, val_pred)
            improved = score < best - 1e-8
            if improved:
                best, best_epoch, stale = score, epoch, 0
                torch.save({"model_state": model.state_dict(), "protocol_id": PROTOCOL,
                            "plan_sha256": sha256(args.plan), "subject_to_index": mapping,
                            "target_scaler": scaler, "epoch": epoch}, args.output / "best.pt")
            else:
                stale += 1
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch, "protocol_id": PROTOCOL, "plan_sha256": sha256(args.plan)}, args.output / "last.pt")
            row = {"epoch": epoch, "train_loss": loss_total / n, "validation_mean_mae": score,
                   "best_epoch": best_epoch, "stale_epochs": stale, "optimizer_steps": steps,
                   "elapsed_seconds": time.monotonic() - started}
            history.append(row)
            save_json(args.output / "history.json", history)
            print(json.dumps(row), flush=True)
            if stale >= args.patience or (args.synthetic and epoch >= args.epochs):
                break
        saved = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
        report.update(status="complete", best_epoch=best_epoch, epochs_completed=epoch,
                      checkpoint_sha256=sha256(args.output / "best.pt"),
                      shared_state_sha256=shared_digest(saved["model_state"]),
                      stop_reason="synthetic_budget" if args.synthetic else "patience8",
                      runtime_seconds=time.monotonic() - started,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        save_json(args.output / "run.json", report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_json(args.output / "run.json", report)
        raise
    return report


def load_population(root, plan_path, *, device="cpu", synthetic=False):
    plan = load_plan(plan_path, synthetic=synthetic)
    root = Path(root)
    report = json.loads((root / "run.json").read_text())
    if (report.get("protocol_id") != PROTOCOL or report.get("status") != "complete"
            or report.get("stage") != "population" or report.get("plan_sha256") != sha256(plan_path)
            or report.get("old_checkpoint_used") is not False or report.get("test_targets_accessed") is not False
            or report.get("test_registration_accessed") is not False or report.get("shared_fit_role") != "meta_train_only"
            or report.get("checkpoint_selection_role") != "meta_validation"):
        raise ValueError("invalid subject-disjoint population provenance")
    if sha256(root / "best.pt") != report["checkpoint_sha256"]:
        raise ValueError("population checkpoint changed")
    checkpoint = torch.load(root / "best.pt", map_location=device, weights_only=False)
    expected = plan["audit"]["population_train"]
    mapping = checkpoint["subject_to_index"]
    if (set(mapping) & (set(plan["test_subjects"]) | set(plan["validation_subjects"]))
            or event_ids_sha256(mapping) != expected["subject_ids_sha256"]
            or report["fit_ids_sha256"] != expected["segment_ids_sha256"]
            or report["fit_rows"] != expected["rows"] or checkpoint.get("plan_sha256") != sha256(plan_path)
            or shared_digest(checkpoint["model_state"]) != report["shared_state_sha256"]):
        raise ValueError("population model does not match the excluded-person contract")
    return plan, report, checkpoint


def personal(args):
    check_arguments(args)
    plan, report, checkpoint = load_population(args.population_run, args.plan, device=args.device, synthetic=args.synthetic)
    if args.cohort == "test":
        # Both methods on validation people must finish before the final cohort.
        freeze = json.loads((args.validation_run / "frozen_methods.json").read_text())
        if freeze.get("plan_sha256") != sha256(args.plan) or freeze.get("checkpoint_sha256") != report["checkpoint_sha256"] or freeze.get("methods") != plan["methods"] or freeze.get("status") != "frozen" or freeze.get("config") != CONFIG:
            raise ValueError("validation completion/frozen-method gate missing")
    access = f"personal_{args.cohort}"
    _, bank = partition(args.plan, f"{args.cohort}_registration", access=access, synthetic=args.synthetic)
    _, query = partition(args.plan, f"{args.cohort}_inputs", access=access, synthetic=args.synthetic)
    context = dict(plan, selected_subjects=plan[f"{args.cohort}_subjects"])
    result = run_profiles(args, context, report, checkpoint, bank, query)
    result["evaluation_cohort"] = args.cohort
    save_json(args.output / "run.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("population", "personal"), required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--population-run", type=Path)
    p.add_argument("--validation-run", type=Path)
    p.add_argument("--cohort", choices=("validation", "test"), default="validation")
    p.add_argument("--shard", type=int, choices=(0, 1), default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    for name in ("learning_rate", "weight_decay", "huber_delta", "gradient_clip"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=CONFIG[name])
    for name in ("batch_size", "patience"):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=CONFIG[name])
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(population(args) if args.stage == "population" else personal(args), indent=2))
