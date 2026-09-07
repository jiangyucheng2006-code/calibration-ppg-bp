"""Real train-only source replay and finite local-relation GPU checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .calbased_train import load_screen_metadata, _autocast
from .calbased_screen import fit_subject_train_means
from .lora_prs_models import LoraPRSRegressor, PRS_MODELS
from .personal_memory_export import check_source, ExportDataset, SOURCES, canonical_metadata
from .personal_memory_models import PersonalMemoryRelation, reference_prediction, supervised_relation_loss
from .training import seed_everything


def run(args):
    seed_everything(20260907)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != SOURCES[args.split_mode]["gpu"]:
        raise RuntimeError("matching allocated source GPU is required")
    source, checkpoint = check_source(args.run, args.split_mode, args.store_root)
    train, _ = load_screen_metadata(args.store_root, args.split_mode)
    means = fit_subject_train_means(train)
    # Within a training participant only; these synthetic pairings do not count
    # as the real blocked-donor protocol or reported research results.
    rows = train.loc[train.subject_uid.eq(train.subject_uid.iloc[0])].head(64).copy()
    canonical_metadata(rows)
    ds = ExportDataset(rows, args.store_root, means, checkpoint["subject_to_index"], checkpoint["target_scaler"])
    x, anchor, people = next(iter(DataLoader(ds, batch_size=64, num_workers=0)))
    model = LoraPRSRegressor(PRS_MODELS["lora_continue"], subject_count=len(checkpoint["subject_to_index"])).cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    captured = {}
    handle = model.base.residual_head.register_forward_pre_hook(lambda _m, a: captured.update(z=a[0].detach()))
    scaler = checkpoint["target_scaler"]
    std = torch.tensor(scaler["std"], device="cuda")
    mean = torch.tensor(scaler["mean"], device="cuda")
    with torch.inference_mode(), _autocast(torch.device("cuda")):
        p = model(x.cuda(), anchor.cuda(), subject_index=people.cuda())
    features = captured["z"].float().clone()
    base = (p.float() * std + mean).clone()
    handle.remove()
    target = torch.tensor(rows[["sbp", "dbp"]].to_numpy(dtype=np.float32), device="cuda")
    relation = PersonalMemoryRelation().cuda()
    optimizer = torch.optim.AdamW(relation.parameters(), lr=3e-4)
    donor = (torch.arange(64, device="cuda")[:, None] + torch.arange(1, 6, device="cuda")) % 64
    features, base = features.detach().clone(), base.detach().clone()
    loss_values = []
    for step in range(4):
        optimizer.zero_grad()
        pred, delta = reference_prediction(relation, features, features[donor], target[donor], base,
                                         torch.full((64, 5), .2, device="cuda"),
                                         torch.ones((64, 5), dtype=torch.bool, device="cuda"),
                                         torch.ones(64, device="cuda"), std)
        loss, _, _ = supervised_relation_loss(pred, target, delta, target[donor],
                                             torch.ones((64, 5), dtype=torch.bool, device="cuda"), std)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(relation.parameters(), 5)
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in relation.parameters()):
            raise FloatingPointError("real-feature relation smoke has nonfinite loss/gradient")
        optimizer.step()
        loss_values.append(float(loss.detach()))
    relation.eval()
    with torch.no_grad():
        antisym_error = (relation(features, features.roll(1, 0)) + relation(features.roll(1, 0), features)).abs().max().item()
    if antisym_error != 0 or not np.isfinite(features.cpu().numpy()).all():
        raise ValueError("antisymmetry or feature finiteness failure")
    result = {"status": "pass", "split_mode": args.split_mode,
              "gpu": torch.cuda.get_device_name(0), "train_rows": 64,
              "raw_content_hash_verified": True, "canonical_record_times_checked": True,
              "source_checkpoint_sha256": source["checkpoint_sha256"],
              "relation_training_steps": 4, "training_losses": loss_values,
              "antisymmetry_max_error": antisym_error, "heldout_test_accessed": False,
              "smoke_weights_discarded": True, "not_a_performance_result": True}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--store-root", required=True, type=Path)
    p.add_argument("--split-mode", choices=list(SOURCES), required=True)
    p.add_argument("--output", required=True, type=Path)
    print(json.dumps(run(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()
