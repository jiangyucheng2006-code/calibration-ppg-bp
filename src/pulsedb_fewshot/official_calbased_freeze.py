"""Bind the six predeclared predictions before official test BP is joined."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .official_calbased_train import PROTOCOL_ID, event_ids_sha256, save_json, sha256
from .official_calbased_evaluate import validate_predictions

METHODS = ("fixed", "v1", "e1", "trust_scalar", "trust_bp")


def freeze(args):
    output = args.batch_root / "frozen_test_predictions.json"
    if output.exists():
        raise FileExistsError("preserve existing frozen evaluation")
    store_hash = sha256(args.store_root / "manifest.json")
    test = pd.read_parquet(args.store_root / "test_inputs.parquet")
    if {"sbp", "dbp", "target_sbp", "target_dbp"} & set(test):
        raise ValueError("input manifest contains official targets")
    final_base = args.batch_root / "final_lora"
    base_run = json.loads((final_base / "run.json").read_text())
    if not (base_run.get("status") == "complete" and base_run.get("stage") == "final"
            and base_run.get("fit_rows") == 2506 * 360
            and base_run.get("official_test_targets_accessed") is False
            and base_run.get("store_manifest_sha256") == store_hash):
        raise ValueError("final LoRA is not a completed exact official full-budget fit")
    cache = final_base / "cache" / "test_inputs"
    provenance = json.loads((cache / "provenance.json").read_text())
    if (cache / "bp.npy").exists() or provenance.get("labels_present") is not False:
        raise ValueError("official query labels are present in feature cache")
    for filename in ("metadata.parquet", "base_bp.npy"):
        if sha256(cache / filename) != provenance["files"][filename]:
            raise ValueError("LoRA input prediction cache changed")
    meta = pd.read_parquet(cache / "metadata.parquet")
    pred = np.load(cache / "base_bp.npy", allow_pickle=False)
    frame = meta[["subject_uid", "event_id", "source"]].copy()
    frame[["pred_sbp", "pred_dbp"]] = pred
    validate_predictions(frame, test)
    lora_predictions = final_base / "official_test_input_predictions.parquet"
    if lora_predictions.exists():
        existing = validate_predictions(pd.read_parquet(lora_predictions), test)
        expected = validate_predictions(frame, test)
        pd.testing.assert_frame_equal(existing, expected)
    else:
        frame.to_parquet(lora_predictions, index=False)
    candidates = []
    for method in ("lora", *METHODS):
        run_root = final_base if method == "lora" else args.batch_root / f"final_{method}"
        run_path = run_root / "run.json"
        run = json.loads(run_path.read_text())
        if (run.get("status") != "complete" or run.get("official_test_targets_accessed") is not False
                or run.get("protocol_id") != PROTOCOL_ID):
            raise ValueError(f"method incomplete or read targets: {method}")
        if method != "lora" and (run.get("personal_label_budget_windows") != 360
                or run.get("source_checkpoint_sha256") != base_run["checkpoint_sha256"]
                or run.get("stage") != "official_final_frozen_prediction"
                or run.get("method") != method or run.get("official_contract_sha256") != store_hash
                or run.get("query_bp_is_prediction_input") is not False
                or run.get("frozen_configuration", {}).get("frozen_before_optimizer") is not True):
            raise ValueError("memory method does not use the paired full-budget LoRA")
        predictions = run_root / "official_test_input_predictions.parquet"
        validate_predictions(pd.read_parquet(predictions), test)
        if method != "lora" and sha256(predictions) != run.get("predictions_sha256"):
            raise ValueError("method predictions changed after inference")
        checkpoint = run_root / "best.pt"
        if sha256(checkpoint) != run["checkpoint_sha256"]:
            raise ValueError("method checkpoint changed")
        candidates.append({"name": method, "predictions_path": str(predictions),
            "predictions_sha256": sha256(predictions), "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": run["checkpoint_sha256"], "method_run_path": str(run_path),
            "method_run_sha256": sha256(run_path), "final_base_run_path": str(final_base / "run.json"),
            "final_base_run_sha256": sha256(final_base / "run.json")})
    plan = {"protocol_id": PROTOCOL_ID, "status": "frozen_predictions",
        "methods_frozen_before_test_labels": True, "allow_test_based_selection": False,
        "synthetic_smoke": False, "official_contract_sha256": store_hash,
        "test_inputs_sha256": sha256(args.store_root / "test_inputs.parquet"),
        "query_ids_sha256": event_ids_sha256(test.segment_uid),
        "research_plan_path": str(args.research_plan.resolve()),
        "research_plan_sha256": sha256(args.research_plan),
        "full_index_sha256": sha256(args.full_index),
        "official_train_windows_per_subject": 360, "candidates": candidates,
        "created_utc": datetime.now(timezone.utc).isoformat()}
    save_json(output, plan)
    print(json.dumps({"frozen_plan": str(output), "sha256": sha256(output), "candidates": len(candidates)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("batch-root", "store-root", "research-plan", "full-index"):
        parser.add_argument("--" + name, type=Path, required=True)
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
