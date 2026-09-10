"""Create new personal LoRA/memory profiles using registration labels only."""
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

from .official_calbased_train import _loader, canonical_metadata, sha256, save_json, event_ids_sha256
from .post_enrollment_protocol import EXPANDED_PROTOCOL, SEED, partition
from .post_enrollment_ablations import memory_variant
from .post_enrollment_population import load_population, shared_digest, seed_all, check_device

PERSONAL_KEYS = {"base.lora_a.weight", "base.lora_b.weight"}


def fresh_person_model(shared_state, seed, device):
    from .lora_prs_models import LoraPRSRegressor, PRS_MODELS
    seed_all(seed)
    model = LoraPRSRegressor(PRS_MODELS["lora_continue"], subject_count=1).to(device)
    common = {k: v for k, v in shared_state.items() if k not in PERSONAL_KEYS}
    keys = model.load_state_dict(common, strict=False)
    if set(keys.missing_keys) != PERSONAL_KEYS or keys.unexpected_keys:
        raise ValueError("unexpected transferable state; cannot borrow another person's state")
    model.requires_grad_(False).eval()
    model.base.lora_a.weight.requires_grad_(True)
    model.base.lora_b.weight.requires_grad_(True)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 2048:
        raise ValueError("personal training must update exactly 2048 adapter parameters")
    if not torch.count_nonzero(model.base.lora_b.weight).item() == 0:
        raise ValueError("new person's adapter must start at zero correction")
    if shared_digest(model.state_dict()) != shared_digest(shared_state):
        raise ValueError("shared network changed while constructing new person")
    return model


def raw_features(model, frame, args, scaler, anchors, subject):
    loader = _loader(frame, args, scaler, anchors, {subject: 0}, targets=False, shuffle=False)
    result = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            result.append(model.base.encoder(batch["ppg"].to(args.device)).float().cpu().numpy())
    features = np.concatenate(result).astype(np.float32)
    if features.shape != (len(frame), 256) or not np.isfinite(features).all():
        raise ValueError("invalid frozen raw feature cache")
    return features


def feature_forward(model, features, anchor):
    a = model.base.lora_a.weight.reshape(4, 256)
    b = model.base.lora_b.weight.reshape(256, 4)
    adapted = features + (features @ a.T) @ b.T / 4
    return anchor + model.base.residual_head(adapted), adapted


def predict(model, features, anchor_mm, scaler, device):
    mean, std = np.asarray(scaler["mean"], np.float32), np.asarray(scaler["std"], np.float32)
    anchor = torch.as_tensor((np.asarray(anchor_mm, np.float32) - mean) / std, device=device)
    model.eval()
    with torch.inference_mode():
        prediction, adapted = feature_forward(model, torch.as_tensor(features, device=device), anchor)
    return prediction.cpu().numpy() * std + mean, adapted.cpu().numpy()


def fit_adapter(model, train_z, train_bp, anchor_mm, scaler, args, *, validation=None, fixed_epochs=None):
    """No query/test-target argument exists. Shared encoder, head and BN stay frozen."""
    shared_before = shared_digest(model.state_dict())
    mean, std = np.asarray(scaler["mean"], np.float32), np.asarray(scaler["std"], np.float32)
    z = torch.as_tensor(train_z, device=args.device)
    targets = torch.as_tensor((np.asarray(train_bp, np.float32) - mean) / std, device=args.device)
    anchor = torch.as_tensor((np.asarray(anchor_mm, np.float32) - mean) / std, device=args.device)
    optimizer = torch.optim.AdamW([model.base.lora_a.weight, model.base.lora_b.weight], lr=args.learning_rate, weight_decay=args.weight_decay)
    objective = torch.nn.HuberLoss(delta=args.huber_delta)
    model.eval()  # Gradients stay enabled for A/B; shared dropout/BN are fixed.
    history, best_epoch, stale, steps = [], 0, 0, 0
    if validation is not None:
        val_z, val_bp = validation
        best = float(np.abs(predict(model, val_z, anchor_mm, scaler, args.device)[0] - val_bp).mean())
        history.append({"epoch": 0, "validation_mean_mae": best, "optimizer_steps": 0})
    elif fixed_epochs is None or fixed_epochs < 0:
        raise ValueError("refit needs a fixed nonnegative selected epoch count")
    epoch = 0
    while fixed_epochs is None or epoch < fixed_epochs:
        epoch += 1
        loss_sum = 0.
        indices = torch.randperm(len(z), device=args.device)
        for start in range(0, len(z), args.batch_size):
            index = indices[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            output, _ = feature_forward(model, z[index], anchor)
            loss = objective(output, targets[index])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite personal loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.base.lora_a.weight, model.base.lora_b.weight], args.gradient_clip)
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach()) * len(index)
        row = {"epoch": epoch, "train_loss": loss_sum / len(z), "optimizer_steps": steps}
        if validation is not None:
            score = float(np.abs(predict(model, val_z, anchor_mm, scaler, args.device)[0] - val_bp).mean())
            row["validation_mean_mae"] = score
            if score < best - 1e-8:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append(row)
        if validation is not None and (stale >= 8 or (args.synthetic and epoch >= 2)):
            break
    if shared_digest(model.state_dict()) != shared_before:
        raise ValueError("personal optimization changed shared parameters or buffers")
    return {"selected_epoch": best_epoch, "epochs_completed": epoch, "optimizer_steps": steps,
            "shared_unchanged": True, "history": history}


def memory_predict(bank_z, query_z, bank_bp, query_base, bank_frame, query_frame):
    """Fixed donor weighting and per-person registration-only q95; no query BP."""
    from .personal_memory_prepare import prepare_neighbors
    from .official_memory_train import canonical_rows
    from .official_content_policy import audit_official_metadata
    bmeta, qmeta = canonical_metadata(bank_frame, "train"), canonical_metadata(query_frame, "test_inputs")
    bank_rows, query_rows = canonical_rows(bmeta, "train"), canonical_rows(qmeta, "internal_validation")
    audit = audit_official_metadata(bank_rows, query_rows)
    neighbors = prepare_neighbors(bank_z, query_z, bank_rows, query_rows,
        mode="random_disjoint", k=5, block_size=40, audit=False)
    state = neighbors["validation"]
    output = memory_variant(bank_bp, query_base, state)
    if not np.isfinite(output).all():
        raise ValueError("nonfinite memory prediction")
    return output, {"q95": neighbors["train_distance_cutpoints"]["q95"],
        "reference_count": 5, "temperature": .1, "block_size": 40,
        "valid_queries": int(state["valid"].sum()), "audit": audit}, state


def run(args):
    check_device(args.device, args.synthetic)
    plan, base_report, checkpoint = load_population(args.population_run, args.plan, device=args.device, synthetic=args.synthetic)
    _, calibration = partition(args.plan, "enrollment_train", synthetic=args.synthetic)
    _, queries = partition(args.plan, "enrollment_test_inputs", synthetic=args.synthetic)
    return run_profiles(args, plan, base_report, checkpoint, calibration, queries)


def run_profiles(args, plan, base_report, checkpoint, calibration, queries):
    """Shared personal fitter; each caller must validate its own named protocol.

    This helper does not select participants or open a label store. The original
    30/200-person entry point retains its unchanged, strict partition loader.
    """
    protocol_id, methods = plan["protocol_id"], plan["methods"]
    if {"sbp", "dbp", "target_sbp", "target_dbp"} & set(queries):
        raise ValueError("personal query inputs must not contain BP labels")
    if set(calibration.segment_uid) & set(queries.segment_uid):
        raise ValueError("personal registration/query identities overlap")
    if set(calibration.subject_uid) != set(plan["selected_subjects"]) or set(queries.subject_uid) != set(plan["selected_subjects"]):
        raise ValueError("personal fitter received incorrect subject membership")
    if args.shard not in (0, 1):
        raise ValueError("exactly two balanced source shards are defined")
    # Sorted alternating people, not source-based separation: both jobs contribute
    # independently fitted profiles and all scopes are joined by the final scorer.
    subjects = plan["selected_subjects"][args.shard::2]
    args.store_root = Path(plan["source_root"])
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"protocol_id": protocol_id, "stage": "personal_registration", "status": "running",
        "plan_sha256": sha256(args.plan), "population_checkpoint_sha256": base_report["checkpoint_sha256"],
        "shard": args.shard, "subjects": subjects, "test_targets_accessed": False,
        "shared_frozen": True, "other_person_adapter_copied": False,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "profiles": {}}
    save_json(args.output / "run.json", report)
    started = time.monotonic()
    scaler = checkpoint["target_scaler"]
    predictions = {name: [] for name in methods}
    try:
        for subject in subjects:
            tag = hashlib.sha256(subject.encode()).hexdigest()[:16]
            directory = args.output / tag
            directory.mkdir(exist_ok=False)
            bank = calibration.loc[calibration.subject_uid.eq(subject)].reset_index(drop=True)
            query = queries.loc[queries.subject_uid.eq(subject)].reset_index(drop=True)
            anchor_all = bank[["sbp", "dbp"]].mean().to_numpy(np.float32)
            anchors = bank.groupby("subject_uid")[["sbp", "dbp"]].mean()
            seed = SEED + int(hashlib.sha256(subject.encode()).hexdigest()[:6], 16)
            model = fresh_person_model(checkpoint["model_state"], seed, args.device)
            bank_z = raw_features(model, bank, args, scaler, anchors, subject)
            # Even raw query features are deferred until registration is complete.
            train_mask = bank.inner_role.eq("train").to_numpy()
            val_mask = ~train_mask
            anchor_inner = bank.loc[train_mask, ["sbp", "dbp"]].mean().to_numpy(np.float32)
            selection = fit_adapter(model, bank_z[train_mask], bank.loc[train_mask, ["sbp", "dbp"]].to_numpy(np.float32),
                anchor_inner, scaler, args, validation=(bank_z[val_mask], bank.loc[val_mask, ["sbp", "dbp"]].to_numpy(np.float32)))
            save_json(directory / "selection.json", selection)
            # Fresh fit, not continuation from the selection model; all 360
            # labelled registration windows are now legitimately available.
            model = fresh_person_model(checkpoint["model_state"], seed, args.device)
            shared_before = shared_digest(model.state_dict())
            refit = fit_adapter(model, bank_z, bank[["sbp", "dbp"]].to_numpy(np.float32), anchor_all,
                scaler, args, fixed_epochs=selection["selected_epoch"])
            save_json(directory / "refit.json", refit)
            query_z = raw_features(model, query, args, scaler, anchors, subject)
            lora_bp, query_adapted = predict(model, query_z, anchor_all, scaler, args.device)
            _, bank_adapted = predict(model, bank_z, anchor_all, scaler, args.device)
            memory_bp, memory_state, neighbors = memory_predict(bank_adapted, query_adapted,
                bank[["sbp", "dbp"]].to_numpy(np.float32), lora_bp, bank, query)
            personal_state = {key: model.state_dict()[key].detach().cpu() for key in PERSONAL_KEYS}
            torch.save({"protocol_id": protocol_id, "subject_uid": subject, "adapter": personal_state,
                "anchor_mmHg": anchor_all.tolist(), "target_scaler": scaler,
                "population_checkpoint_sha256": base_report["checkpoint_sha256"],
                "plan_sha256": sha256(args.plan), "registration_ids_sha256": event_ids_sha256(bank.segment_uid),
                "shared_state_sha256": shared_before, "memory_state": memory_state}, directory / "profile.pt")
            np.savez_compressed(directory / "memory_bank.npz", features=bank_adapted,
                shared_features=bank_z,
                reference_bp=bank[["sbp", "dbp"]].to_numpy(np.float32))
            canonical_metadata(bank, "train").to_parquet(directory / "registration_metadata.parquet", index=False)
            # Verify a profile can be reloaded for the right account and version.
            saved = torch.load(directory / "profile.pt", map_location=args.device, weights_only=False)
            if saved["subject_uid"] != subject or saved["population_checkpoint_sha256"] != base_report["checkpoint_sha256"]:
                raise ValueError("personal profile identity/version mismatch")
            restored = fresh_person_model(checkpoint["model_state"], seed + 1, args.device)
            shared_bp = predict(restored, query_z, anchor_all, scaler, args.device)[0]
            if set(saved["adapter"]) != PERSONAL_KEYS:
                raise ValueError("invalid serialized personal state")
            # A partial load reports BatchNorm's num_batches_tracked differently
            # across state-dict versions. Merge only the two permitted tensors
            # into the verified shared state, then require an exact strict load.
            restored_state = restored.state_dict()
            restored_state.update(saved["adapter"])
            restored.load_state_dict(restored_state, strict=True)
            if shared_digest(restored.state_dict()) != shared_before:
                raise ValueError("profile reload changed shared network state")
            restored_bp, restored_z = predict(restored, query_z, saved["anchor_mmHg"], saved["target_scaler"], args.device)
            with np.load(directory / "memory_bank.npz", allow_pickle=False) as stored_bank:
                restored_memory, restored_state, _ = memory_predict(stored_bank["features"], restored_z,
                    stored_bank["reference_bp"], restored_bp, bank, query)
            if not np.allclose(restored_bp, lora_bp, atol=1e-5, rtol=0) or not np.allclose(restored_memory, memory_bp, atol=1e-5, rtol=0) or restored_state["q95"] != memory_state["q95"]:
                raise ValueError("saved profile does not reproduce predictions")
            if shared_digest(model.state_dict()) != base_report["shared_state_sha256"]:
                raise ValueError("shared base changed during personal adaptation")
            values = {"personal_mean": np.tile(anchor_all, (len(query), 1)),
                "shared_with_anchor": shared_bp, "new_person_lora": lora_bp,
                "new_person_lora_memory": memory_bp}
            if protocol_id == EXPANDED_PROTOCOL:
                bank_bp = bank[["sbp", "dbp"]].to_numpy(np.float32)
                shared_memory, shared_memory_state, shared_neighbors = memory_predict(
                    bank_z, query_z, bank_bp, shared_bp, bank, query)
                values.update(
                    shared_memory_only=memory_variant(bank_bp, values["personal_mean"], shared_neighbors, mode="memory_only"),
                    shared_memory_blend=shared_memory,
                    lora_memory_only=memory_variant(bank_bp, values["personal_mean"], neighbors, mode="memory_only"),
                    lora_memory_uniform=memory_variant(bank_bp, lora_bp, neighbors, mode="distance_uniform"),
                    lora_memory_fixed_half=memory_variant(bank_bp, lora_bp, neighbors, mode="fixed_half"))
                with np.load(directory / "memory_bank.npz", allow_pickle=False) as stored_bank:
                    reproduced_shared, reproduced_shared_state, reproduced_shared_neighbors = memory_predict(
                        stored_bank["shared_features"], query_z, stored_bank["reference_bp"], shared_bp, bank, query)
                    _, _, reproduced_neighbors = memory_predict(stored_bank["features"], restored_z,
                        stored_bank["reference_bp"], restored_bp, bank, query)
                    reproduced = {
                        "shared_memory_only": memory_variant(stored_bank["reference_bp"], values["personal_mean"], reproduced_shared_neighbors, mode="memory_only"),
                        "shared_memory_blend": reproduced_shared,
                        "lora_memory_only": memory_variant(stored_bank["reference_bp"], values["personal_mean"], reproduced_neighbors, mode="memory_only"),
                        "lora_memory_uniform": memory_variant(stored_bank["reference_bp"], restored_bp, reproduced_neighbors, mode="distance_uniform"),
                        "lora_memory_fixed_half": memory_variant(stored_bank["reference_bp"], restored_bp, reproduced_neighbors, mode="fixed_half")}
                if any(not np.allclose(values[name], pred, atol=1e-5, rtol=0) for name, pred in reproduced.items()):
                    raise ValueError("saved profile does not reproduce ablation predictions")
                if reproduced_shared_state["q95"] != shared_memory_state["q95"]:
                    raise ValueError("saved shared memory threshold changed")
                save_json(directory / "ablation_state.json", {
                    "methods": methods, "shared_memory": shared_memory_state,
                    "adapted_memory": memory_state, "all_ablation_reload_equivalence": True,
                    "test_targets_accessed": False})
            if tuple(methods) == ("new_person_lora", "new_person_lora_memory"):
                values = {name: values[name] for name in methods}
            if set(values) != set(methods):
                raise ValueError("prediction settings do not match the frozen protocol")
            for name, pred in values.items():
                rows = query[["subject_uid", "segment_uid", "source"]].copy()
                rows[["pred_sbp", "pred_dbp"]] = pred
                predictions[name].append(rows)
            profile_report = {"directory": tag, "registration_rows": len(bank), "test_rows": len(query),
                "personal_seed": seed, "selected_epoch": selection["selected_epoch"],
                "refit_optimizer_steps": refit["optimizer_steps"], "adapter_parameters": 2048,
                "shared_frozen_verified": True, "reload_equivalence": True,
                "profile_sha256": sha256(directory / "profile.pt"),
                "memory_bank_sha256": sha256(directory / "memory_bank.npz"),
                "registration_metadata_sha256": sha256(directory / "registration_metadata.parquet"),
                "memory_q95": memory_state["q95"], "valid_memory_queries": memory_state["valid_queries"]}
            if protocol_id == EXPANDED_PROTOCOL:
                profile_report["ablation_state_sha256"] = sha256(directory / "ablation_state.json")
                profile_report["all_ablation_reload_equivalence"] = True
            report["profiles"][subject] = profile_report
            save_json(args.output / "run.json", report)
            print(json.dumps({"completed_profiles": len(report["profiles"]), "total": len(subjects),
                "profile": tag, "selected_epoch": selection["selected_epoch"], "reload_equivalence": True}), flush=True)
        report["prediction_files"] = {}
        for name in methods:
            path = args.output / f"{name}_predictions.parquet"
            pd.concat(predictions[name], ignore_index=True).to_parquet(path, index=False)
            report["prediction_files"][path.name] = sha256(path)
        report.update(status="complete", frozen_predictions=True, runtime_seconds=time.monotonic() - started)
        save_json(args.output / "run.json", report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_json(args.output / "run.json", report)
        raise
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--population-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shard", type=int, choices=(0, 1), required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=.5)
    p.add_argument("--gradient-clip", type=float, default=5.)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
