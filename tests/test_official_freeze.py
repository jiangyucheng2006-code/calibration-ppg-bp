"""Mock-contract tests for freezing all six methods before target access.

These fixtures are not model results or a synthetic benchmark: Parquet reads
are replaced by small explicit tables and checkpoint files are inert bytes.
They exercise receipt/coverage/label-access behavior without Torch or PyArrow.
"""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pulsedb_fewshot import official_calbased_freeze as freezer
from pulsedb_fewshot.official_calbased_train import PROTOCOL_ID, save_json, sha256


def fixture(root):
    batch, store = root / "batch", root / "store"
    batch.mkdir()
    store.mkdir()
    save_json(store / "manifest.json", {"protocol_id": PROTOCOL_ID, "status": "ready"})
    test = pd.DataFrame({"subject_uid": ["person_m", "person_m", "person_v", "person_v"],
                         "segment_uid": ["m:1", "m:2", "v:1", "v:2"],
                         "source": ["MIMIC", "MIMIC", "VitalDB", "VitalDB"]})
    inputs_path = store / "test_inputs.parquet"
    inputs_path.write_bytes(b"mock official input membership, no target fields")
    index_path = root / "index.parquet"
    index_path.write_bytes(b"sealed reference index may be hashed, never deserialized by freezer")
    research_plan = root / "plan.md"
    research_plan.write_text("Predeclared: lora fixed v1 e1 trust_scalar trust_bp.", encoding="utf-8")
    prediction = test.rename(columns={"segment_uid": "event_id"}).assign(
        pred_sbp=np.array([120., 121., 130., 132.]), pred_dbp=np.array([70., 71., 75., 77.]))
    tables = {inputs_path: test}
    base = batch / "final_lora"
    base.mkdir()
    (base / "best.pt").write_bytes(b"inert fresh base checkpoint fixture")
    base_hash = sha256(base / "best.pt")
    for method in ("lora", *freezer.METHODS):
        method_root = batch / f"final_{method}"
        method_root.mkdir(exist_ok=True)
        if method != "lora":
            (method_root / "best.pt").write_bytes(("inert method checkpoint " + method).encode())
        pred_path = method_root / "official_test_input_predictions.parquet"
        pred_path.write_bytes(("mock frozen predictions " + method).encode())
        tables[pred_path] = prediction.copy()
        report = {"protocol_id": PROTOCOL_ID, "status": "complete", "method": method,
                  "stage": "final" if method == "lora" else "official_final_frozen_prediction",
                  "official_test_targets_accessed": False, "official_test_inputs_accessed": True,
                  "store_manifest_sha256": sha256(store / "manifest.json"),
                  "official_contract_sha256": sha256(store / "manifest.json"),
                  "fit_subjects": 2506, "fit_rows": 2506 * 360,
                  "old_checkpoint_used": False, "personal_label_budget_windows": 360,
                  "source_checkpoint_sha256": base_hash, "query_bp_is_prediction_input": False,
                  "checkpoint_sha256": sha256(method_root / "best.pt"),
                  "predictions_sha256": sha256(pred_path),
                  "frozen_configuration": {"frozen_before_optimizer": True}}
        save_json(method_root / "run.json", report)
    cache = base / "cache" / "test_inputs"
    cache.mkdir(parents=True)
    meta_path = cache / "metadata.parquet"
    meta_path.write_bytes(b"mock label-free feature metadata")
    tables[meta_path] = prediction[["subject_uid", "event_id", "source"]]
    np.save(cache / "base_bp.npy", prediction[["pred_sbp", "pred_dbp"]].to_numpy(), allow_pickle=False)
    save_json(cache / "provenance.json", {"labels_present": False,
              "files": {filename: sha256(cache / filename) for filename in ("metadata.parquet", "base_bp.npy")}})
    args = argparse.Namespace(batch_root=batch, store_root=store, research_plan=research_plan, full_index=index_path)

    def read(path, **kwargs):
        path = Path(path)
        if path == index_path:
            raise AssertionError("freezer attempted to deserialize official BP targets")
        if path not in tables:
            raise AssertionError(f"unexpected Parquet path: {path}")
        return tables[path].copy()
    return args, tables, read


class OfficialFreezerContractTests(unittest.TestCase):
    def test_all_six_methods_bound_without_target_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, tables, reader = fixture(Path(tmp))
            with patch.object(pd, "read_parquet", side_effect=reader) as loader:
                freezer.freeze(args)
            plan_path = args.batch_root / "frozen_test_predictions.json"
            plan = json.loads(plan_path.read_text())
            self.assertEqual([c["name"] for c in plan["candidates"]], ["lora", *freezer.METHODS])
            self.assertTrue(plan["methods_frozen_before_test_labels"])
            self.assertFalse(plan["allow_test_based_selection"])
            self.assertEqual(plan["official_train_windows_per_subject"], 360)
            self.assertEqual(plan["full_index_sha256"], sha256(args.full_index))
            for item in plan["candidates"]:
                self.assertEqual(item["predictions_sha256"], sha256(item["predictions_path"]))
                self.assertEqual(item["method_run_sha256"], sha256(item["method_run_path"]))
            self.assertNotIn(args.full_index, [Path(call.args[0]) for call in loader.call_args_list])

    def test_existing_freeze_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, reader = fixture(Path(tmp))
            path = args.batch_root / "frozen_test_predictions.json"
            path.write_text("existing freeze", encoding="utf-8")
            with patch.object(pd, "read_parquet") as loader:
                with self.assertRaises(FileExistsError):
                    freezer.freeze(args)
                loader.assert_not_called()
            self.assertEqual(path.read_text(), "existing freeze")

    def test_incomplete_last_method_prevents_whole_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, reader = fixture(Path(tmp))
            path = args.batch_root / "final_trust_bp" / "run.json"
            report = json.loads(path.read_text())
            report["status"] = "running"
            save_json(path, report)
            with patch.object(pd, "read_parquet", side_effect=reader):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    freezer.freeze(args)
            self.assertFalse((args.batch_root / "frozen_test_predictions.json").exists())

    def test_partial_prediction_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, tables, reader = fixture(Path(tmp))
            path = args.batch_root / "final_e1" / "official_test_input_predictions.parquet"
            tables[path] = tables[path].iloc[:-1]
            with patch.object(pd, "read_parquet", side_effect=reader):
                with self.assertRaisesRegex(ValueError, "coverage"):
                    freezer.freeze(args)
            self.assertFalse((args.batch_root / "frozen_test_predictions.json").exists())

    def test_reference_label_file_in_input_cache_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, reader = fixture(Path(tmp))
            (args.batch_root / "final_lora/cache/test_inputs/bp.npy").write_bytes(b"must not be opened")
            with patch.object(pd, "read_parquet", side_effect=reader):
                with self.assertRaisesRegex(ValueError, "query labels"):
                    freezer.freeze(args)

    def test_reference_labels_in_input_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, tables, reader = fixture(Path(tmp))
            tables[args.store_root / "test_inputs.parquet"]["target_sbp"] = 123.
            with patch.object(pd, "read_parquet", side_effect=reader):
                with self.assertRaisesRegex(ValueError, "contains official targets"):
                    freezer.freeze(args)

    def test_changed_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, reader = fixture(Path(tmp))
            (args.batch_root / "final_v1/best.pt").write_bytes(b"different weights")
            with patch.object(pd, "read_parquet", side_effect=reader):
                with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                    freezer.freeze(args)

    def test_unpaired_lora_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _, reader = fixture(Path(tmp))
            path = args.batch_root / "final_v1/run.json"
            report = json.loads(path.read_text())
            report["source_checkpoint_sha256"] = "a" * 64
            save_json(path, report)
            with patch.object(pd, "read_parquet", side_effect=reader):
                with self.assertRaisesRegex(ValueError, "paired full-budget"):
                    freezer.freeze(args)


if __name__ == "__main__":
    unittest.main()
