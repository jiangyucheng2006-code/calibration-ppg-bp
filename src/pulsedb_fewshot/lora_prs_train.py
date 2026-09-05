"""Initialize all continuation candidates from the same immutable LoRA source."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .same_subject_component_train import train_component
from .training import file_sha256
from .lora_prs_models import PRS_MODELS, SCREEN_ID, SPLIT_MODES, LoraPRSRegressor

# Already-completed registered-user checkpoints, not pending feature-screen runs.
SOURCE_HASHES = {
    "random_disjoint": "04c965346c8721a0bae8e74c8ec84f670dad8cdd145694dbff32e5a186fa92f2",
    "chronological_blocked": "225760ceea3c88ae486b6d5d147978e9c8a16781a9c481eccd9cae831c108f81",
}


def initialize(model, args, scaler, subject_indices):
    run = json.loads((args.source_run / "run.json").read_text())
    if not (run["status"] == "complete" and run["candidate"] == "subject_lora_rank4"
            and run["screen_id"] == "same-subject-personal-profile-v1"
            and run["split_mode"] == args.split_mode and run["heldout_test_accessed"] is False
            and run["selection_role"] == "internal_validation" and run["source_parent_split"] == "meta_train"
            and run["read_roles"] == ["train", "internal_validation"]):
        raise ValueError("ineligible source checkpoint")
    path = args.source_run / "best.pt"
    digest = file_sha256(path)
    if digest != SOURCE_HASHES[args.split_mode] or digest != run["checkpoint_sha256"]:
        raise ValueError("source checkpoint differs from the frozen source")
    if file_sha256(args.store_root / "materialization.json") != run["store_manifest_sha256"]:
        raise ValueError("source and continuation stores differ")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["subject_to_index"] != subject_indices:
        raise ValueError("personal state mapping differs")
    for key in ["mean", "std"]:
        if not np.allclose(ckpt["target_scaler"][key], scaler[key], atol=1e-9, rtol=0):
            raise ValueError("train-only scaler differs")
    state = ckpt["model_state"]
    if not all(key.startswith("reference.") for key in state):
        raise ValueError("unexpected source state layout")
    model.base.load_state_dict({key.removeprefix("reference."): value for key, value in state.items()}, strict=True)
    return {"source_checkpoint_sha256": digest, "source_run": str(args.source_run),
            "source_best_epoch": run["best_epoch"], "source_seed": run["seed"],
            "source_gpu": run["gpu"], "source_selection_role": "internal_validation",
            "source_heldout_test_accessed": False, "optimizer_reinitialized": True,
            "base_learning_rate": args.base_learning_rate, "correction_learning_rate": args.learning_rate,
            "correction_penalty": model.spec.correction_penalty, "base_frozen": model.spec.freeze_base,
            "initial_checkpoint_eligible_for_selection": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", choices=list(PRS_MODELS), required=True)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--store-root", type=Path, required=True)
    p.add_argument("--split-mode", choices=SPLIT_MODES, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--examples-per-epoch", type=int, default=200000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--base-learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=0.5)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--demographics-path", type=Path)
    p.add_argument("--beat-similarity-path", type=Path)
    args = p.parse_args()
    print(json.dumps(train_component(args, components=PRS_MODELS, model_factory=LoraPRSRegressor,
        screen_id=SCREEN_ID, runner="lora_prs_continuation", allowed_split_modes=SPLIT_MODES,
        initializer=initialize, optimizer_factory=lambda model, a: model.optimizer(a),
        training_penalty=lambda model: model.penalty()), indent=2))


if __name__ == "__main__":
    main()
