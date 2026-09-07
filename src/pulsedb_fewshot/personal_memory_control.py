"""Matched continued-LoRA reference, gated before loading data or PyTorch.

Same nominal sampled examples per epoch, seed and early-stopping policy as the
memory screen; cached-head training is NOT compute/epoch-count equivalent to
end-to-end continuation. Both sources remain private registered-user models.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path


SCREEN_ID = "personal-memory-v1"
CANDIDATE = "lora_continued_control"
MODES = ("random_disjoint", "chronological_blocked")


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(part)
    return hasher.hexdigest()


def gate_contract(gate_path: Path, cache_dir: Path, split_mode: str):
    """Small-JSON checks only; a stopped scientific gate never opens data."""
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    status = gate.get("decision", gate.get("status"))
    if status not in {"proceed_to_train", "stop_no_complementarity"}:
        raise ValueError("unknown or incomplete scientific gate")
    if gate.get("status") not in {status, None} or gate.get("screen_id") != SCREEN_ID:
        raise ValueError("gate status/screen mismatch")
    if gate.get("heldout_test_accessed") is not False:
        raise ValueError("scientific gate does not explicitly exclude held-out access")
    hashes = gate.get("cache_manifest_sha256", {})
    if set(hashes) != set(MODES) or any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise ValueError("gate must bind cache manifests from both split modes")
    cache_path = cache_dir / "manifest.json"
    cache_hash = digest(cache_path)
    if hashes[split_mode] != cache_hash:
        raise ValueError("cache manifest changed since the global gate")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("split_mode") != split_mode or cache.get("heldout_test_accessed") is not False:
        raise ValueError("cache split/held-out contract mismatch")
    if cache.get("source_parent_split") != "meta_train" or cache.get("read_roles") != ["train", "internal_validation"]:
        raise ValueError("cache data-role contract mismatch")
    return gate, cache, {"gate_manifest_sha256": digest(gate_path), "cache_manifest_sha256": cache_hash,
                         "gate_decision": status}


def initialize(model, args, scaler, subject_indices):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from .personal_memory_export import check_source, SOURCES, ExportDataset
    from .calbased_train import load_screen_metadata, _autocast
    from .calbased_screen import fit_subject_train_means

    source, checkpoint = check_source(args.run, args.split_mode, args.store_root)
    if args._cache["source_checkpoint_sha256"] != SOURCES[args.split_mode]["hash"]:
        raise ValueError("cache came from a different frozen reference")
    if args._cache["store_manifest_sha256"] != source["store_manifest_sha256"]:
        raise ValueError("cache/source training data differ")
    if checkpoint["subject_to_index"] != subject_indices:
        raise ValueError("source personal-state index differs from continuation")
    for key in ("mean", "std"):
        if not np.allclose(checkpoint["target_scaler"][key], scaler[key], rtol=0, atol=1e-9):
            raise ValueError("source target scaler differs from train-only fit")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = next(model.parameters()).device
    if device.type != "cuda" or torch.cuda.get_device_name(0) != SOURCES[args.split_mode]["gpu"]:
        raise ValueError("continuation must use the frozen source's GPU type")
    for filename in ("validation_base.npy", "validation_metadata.parquet", "validation_bp.npy"):
        entry = args._cache["files"][filename]
        if entry["path"] != filename or digest(args.cache_dir / filename) != entry["sha256"]:
            raise ValueError("validation cache file changed")
    train, validation = load_screen_metadata(args.store_root, args.split_mode)
    expected_metadata = pd.read_parquet(args.cache_dir / "validation_metadata.parquet")
    expected = np.load(args.cache_dir / "validation_base.npy", mmap_mode="r")
    targets = np.load(args.cache_dir / "validation_bp.npy", mmap_mode="r")
    if len(expected_metadata) != len(validation) or len(validation) != 82040:
        raise ValueError("incomplete reference validation cohort")
    for actual, expected_column in (("subject_uid", "subject_uid"), ("segment_uid", "event_id"), ("source", "source")):
        if validation[actual].astype(str).tolist() != expected_metadata[expected_column].astype(str).tolist():
            raise ValueError("cached validation order/keys differ from raw data")
    np.testing.assert_allclose(targets, validation[["sbp", "dbp"]].to_numpy(), rtol=0, atol=2e-5)
    dataset = ExportDataset(validation, args.store_root, fit_subject_train_means(train), subject_indices, scaler)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    mean = torch.tensor(scaler["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(scaler["std"], dtype=torch.float32, device=device)
    maximum = total = 0.0
    elements = offset = 0
    model.eval()
    with torch.inference_mode():
        for x, anchor, person in loader:
            with _autocast(device):
                prediction = model(x.to(device), anchor.to(device), subject_index=person.to(device))
            prediction = (prediction.float() * std + mean).cpu().numpy()
            difference = np.abs(prediction - expected[offset:offset + len(x)])
            maximum = max(maximum, float(difference.max()))
            total += float(difference.sum(dtype=np.float64))
            elements += difference.size
            offset += len(x)
    reproduction = {"max_abs_mmhg": maximum, "mean_abs_mmhg": total / elements}
    if maximum > 0.05 or total / elements > 0.005:
        raise ValueError(f"epoch-zero source/cache reproduction failed: {reproduction}")
    return {**args._gate_evidence, "source_checkpoint_sha256": SOURCES[args.split_mode]["hash"],
            "source_run_id": args.run.name, "source_best_epoch": source["best_epoch"],
            "source_gpu": source["gpu"], "source_seed": source["seed"],
            "optimizer_reinitialized": True, "initial_checkpoint_eligible_for_selection": True,
            "initial_reproduction": reproduction,
            "fairness": "same nominal raw-sample exposure per epoch; not compute or total epoch-count matched"}


def run(args):
    if (args.seed, args.epochs, args.patience, args.batch_size, args.workers, args.examples_per_epoch) != (20260907, 0, 8, 64, 0, 200000):
        raise ValueError("continuation screen sampling/seed/stopping contract is frozen")
    if (args.learning_rate, args.base_learning_rate, args.weight_decay, args.huber_delta) != (1e-5, 1e-5, 1e-4, .5):
        raise ValueError("continuation optimizer contract is frozen")
    gate, cache, evidence = gate_contract(args.gate_manifest, args.cache_dir, args.split_mode)
    if evidence["gate_decision"] == "stop_no_complementarity":
        record = {"status": "skipped_no_complementarity", "screen_id": SCREEN_ID,
                  "protocol_id": "development-calbased-analogue-v1", "candidate": CANDIDATE,
                  "split_mode": args.split_mode, "seed": args.seed, "heldout_test_accessed": False,
                  "source_parent_split": "meta_train", "read_roles": [], "parameters_updated": False,
                  **evidence}
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record
    # Only an explicit proceed decision reaches PyTorch, data and checkpoints.
    from .same_subject_component_train import train_component
    from .lora_prs_models import PRS_MODELS, LoraPRSRegressor
    from .training import save_json
    args.candidate = CANDIDATE
    args.demographics_path = None
    args.beat_similarity_path = None
    args._cache, args._gate_evidence = cache, evidence
    spec = replace(PRS_MODELS["lora_continue"], name=CANDIDATE)
    result = train_component(args, components={CANDIDATE: spec}, model_factory=LoraPRSRegressor,
        screen_id=SCREEN_ID, runner="personal_memory_control", allowed_split_modes=MODES,
        initializer=initialize, optimizer_factory=lambda model, a: model.optimizer(a))
    result.update(evidence)
    result["fairness"] = "same nominal sampled examples per epoch and patience; cached neural heads are not compute-matched"
    # Preserve only stable public-size initialization evidence, not a copy of
    # the full cache manifest inside command-line argument metadata.
    result["arguments"].pop("_cache", None)
    result["arguments"].pop("_gate_evidence", None)
    save_json(args.output / "run.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "store-root", "cache-dir", "gate-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split-mode", choices=MODES, required=True)
    for name, default in (("seed", 20260907), ("epochs", 0), ("patience", 8), ("batch-size", 64), ("workers", 0), ("examples-per-epoch", 200000)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("learning-rate", 1e-5), ("base-learning-rate", 1e-5), ("weight-decay", 1e-4), ("huber-delta", .5)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--require-cuda", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
