"""Final target-access ordering and fixed-query coverage checks."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pulsedb_fewshot import official_calbased_evaluate as evaluate
from pulsedb_fewshot import official_calbased_train as train


def frames():
    rows = []
    for person in range(6):
        source = "MIMIC" if person < 3 else "VitalDB"
        for i in range(2):
            rows.append({"subject_uid": f"p{person}", "segment_uid": f"p{person}:test{i}", "source": source})
    inputs = pd.DataFrame(rows)
    targets = inputs.assign(sbp=np.arange(len(inputs)) + 110., dbp=np.arange(len(inputs)) * .5 + 65.)
    predictions = inputs.rename(columns={"segment_uid": "event_id"}).assign(
        pred_sbp=targets.sbp.to_numpy() + 2., pred_dbp=targets.dbp.to_numpy() - 1.)
    return inputs, targets, predictions


def bundle(root, *, real_parquet=False):
    inputs, targets, predictions = frames()
    store = root / "store"
    store.mkdir()
    train.save_json(store / "manifest.json", {"protocol_id": train.PROTOCOL_ID, "status": "ready", "synthetic": True})
    full_index = root / "index.parquet"
    pred_path = root / "lora.parquet"
    if real_parquet:
        inputs.to_parquet(store / "test_inputs.parquet", index=False)
        targets.to_parquet(full_index, index=False)
        predictions.to_parquet(pred_path, index=False)
    else:
        (store / "test_inputs.parquet").write_bytes(b"input-only-table")
        full_index.write_bytes(b"reference-label-index-do-not-deserialize-during-verification")
        pred_path.write_bytes(b"frozen-input-only-predictions")
    (root / "research_plan.md").write_text("Fixed methods before official labels; same-subject evaluation.", encoding="utf-8")
    final = root / "final"
    final.mkdir()
    (final / "best.pt").write_bytes(b"fresh-final-model-placeholder-for-evaluation-only")
    train.save_json(final / "run.json", {"protocol_id": train.PROTOCOL_ID, "stage": "final", "status": "complete",
        "official_test_targets_accessed": False, "store_manifest_sha256": train.sha256(store / "manifest.json"),
        "old_checkpoint_used": False, "fit_subjects": 6, "fit_rows": 66,
        "checkpoint_sha256": train.sha256(final / "best.pt")})
    plan = {"protocol_id": train.PROTOCOL_ID, "status": "frozen_predictions", "synthetic_smoke": True,
        "methods_frozen_before_test_labels": True, "allow_test_based_selection": False,
        "official_contract_sha256": train.sha256(store / "manifest.json"),
        "research_plan_path": "research_plan.md", "research_plan_sha256": train.sha256(root / "research_plan.md"),
        "test_inputs_sha256": train.sha256(store / "test_inputs.parquet"),
        "query_ids_sha256": train.event_ids_sha256(inputs.segment_uid), "official_train_windows_per_subject": 360,
        "full_index_sha256": train.sha256(full_index), "candidates": [{"name": "lora", "predictions_path": "lora.parquet",
            "predictions_sha256": train.sha256(pred_path), "final_base_run_path": "final/run.json",
            "final_base_run_sha256": train.sha256(final / "run.json")}]}
    plan_path = root / "frozen_predictions.json"
    train.save_json(plan_path, plan)
    args = argparse.Namespace(frozen_plan=plan_path, expected_plan_sha256=train.sha256(plan_path), store_root=store,
        full_index=full_index, output=root / "evaluation", smoke=True)
    def read(path, **kwargs):
        path = Path(path)
        if path.name == "test_inputs.parquet":
            return inputs.copy()
        if path.name == "lora.parquet":
            return predictions.copy()
        if path.name == "index.parquet":
            raise AssertionError("test targets requested during pre-target verification")
        raise AssertionError(f"unexpected path {path}")
    return args, plan, read


class OfficialEvaluationContractTests(unittest.TestCase):
    def test_all_predictions_verified_without_target_read(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan, read = bundle(Path(directory))
            with patch.object(pd, "read_parquet", side_effect=read) as loader:
                _, test, predictions, receipts = evaluate.verify_frozen_bundle(args)
            self.assertEqual(len(test), 12)
            self.assertEqual(list(predictions), ["lora"])
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(receipts["lora"]["predictions_sha256"], plan["candidates"][0]["predictions_sha256"])

    def test_wrong_frozen_plan_hash_stops_before_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = bundle(Path(directory))
            args.expected_plan_sha256 = "0" * 64
            with patch.object(pd, "read_parquet") as loader:
                with self.assertRaisesRegex(ValueError, "plan checksum"):
                    evaluate.verify_frozen_bundle(args)
            loader.assert_not_called()

    def test_changed_prediction_stops_before_target_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, read = bundle(root)
            (root / "lora.parquet").write_bytes(b"updated prediction after freeze")
            with patch.object(pd, "read_parquet", side_effect=read) as loader:
                with self.assertRaisesRegex(ValueError, "predictions changed"):
                    evaluate.verify_frozen_bundle(args)
            self.assertEqual(loader.call_count, 1)

    def test_changed_research_plan_stops_before_target_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, read = bundle(root)
            (root / "research_plan.md").write_text("changed", encoding="utf-8")
            with patch.object(pd, "read_parquet", side_effect=read) as loader:
                with self.assertRaisesRegex(ValueError, "research plan changed"):
                    evaluate.verify_frozen_bundle(args)
            loader.assert_not_called()

    def test_old_checkpoint_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, plan, read = bundle(root)
            receipt = json.loads((root / "final/run.json").read_text())
            receipt["old_checkpoint_used"] = True
            train.save_json(root / "final/run.json", receipt)
            plan["candidates"][0]["final_base_run_sha256"] = train.sha256(root / "final/run.json")
            train.save_json(args.frozen_plan, plan)
            args.expected_plan_sha256 = train.sha256(args.frozen_plan)
            with patch.object(pd, "read_parquet", side_effect=read):
                with self.assertRaisesRegex(ValueError, "invalid fresh official final base"):
                    evaluate.verify_frozen_bundle(args)

    def test_model_selection_from_test_prohibited(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan, read = bundle(Path(directory))
            plan["allow_test_based_selection"] = True
            train.save_json(args.frozen_plan, plan)
            args.expected_plan_sha256 = train.sha256(args.frozen_plan)
            with patch.object(pd, "read_parquet", side_effect=read) as loader:
                with self.assertRaisesRegex(ValueError, "pre-target frozen"):
                    evaluate.verify_frozen_bundle(args)
            loader.assert_not_called()

    def test_predicted_targets_not_allowed(self):
        inputs, targets, predictions = frames()
        predictions["target_sbp"] = targets.sbp
        with self.assertRaisesRegex(ValueError, "must not contain test targets"):
            evaluate.validate_predictions(predictions, inputs)

    def test_incomplete_coverage_rejected(self):
        inputs, _, predictions = frames()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluate.validate_predictions(predictions.iloc[:-1], inputs)

    def test_subject_or_source_change_rejected(self):
        inputs, _, predictions = frames()
        predictions.loc[0, "subject_uid"] = "wrong"
        with self.assertRaisesRegex(ValueError, "exact official test"):
            evaluate.validate_predictions(predictions, inputs)

    def test_nonfinite_predictions_rejected(self):
        inputs, _, predictions = frames()
        predictions.loc[0, "pred_sbp"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            evaluate.validate_predictions(predictions, inputs)

    def test_target_read_uses_exact_id_predicate_and_columns(self):
        inputs, targets, _ = frames()
        with patch.object(pd, "read_parquet", return_value=targets) as loader:
            result = evaluate.read_exact_targets(Path("index.parquet"), inputs)
        loader.assert_called_once_with(Path("index.parquet"), columns=evaluate.TARGET_COLUMNS,
                                      filters=[("segment_uid", "in", inputs.segment_uid.tolist())])
        pd.testing.assert_frame_equal(result, targets)

    def test_extra_target_row_not_silently_used(self):
        inputs, targets, _ = frames()
        targets = pd.concat([targets, targets.iloc[[0]].assign(segment_uid="extra")])
        with patch.object(pd, "read_parquet", return_value=targets):
            with self.assertRaisesRegex(ValueError, "exact official test coverage"):
                evaluate.read_exact_targets(Path("index.parquet"), inputs)

    def test_target_read_only_on_all_final_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, plan, read = bundle(root)
            receipt = json.loads((root / "final/run.json").read_text())
            receipt["stage"] = "inner"
            train.save_json(root / "final/run.json", receipt)
            plan["candidates"][0]["final_base_run_sha256"] = train.sha256(root / "final/run.json")
            train.save_json(args.frozen_plan, plan)
            args.expected_plan_sha256 = train.sha256(args.frozen_plan)
            with patch.object(pd, "read_parquet", side_effect=read):
                with self.assertRaisesRegex(ValueError, "invalid fresh official final base"):
                    evaluate.verify_frozen_bundle(args)


@unittest.skipUnless(train.torch is not None and importlib.util.find_spec("pyarrow"), "torch/pyarrow runtime required")
class OfficialEvaluationIntegrationTest(unittest.TestCase):
    def test_exact_scoring_three_scopes_and_no_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = bundle(Path(directory), real_parquet=True)
            report = evaluate.run(args)
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["official_test_targets_accessed"])
            self.assertFalse(report["test_based_model_selection"])
            macro = pd.read_csv(args.output / "participant_macro.csv")
            self.assertEqual(set(macro.Scope), {"Overall", "MIMIC", "VitalDB"})
            self.assertTrue(np.allclose(macro.sbp_mae, 2.))
            self.assertTrue(np.allclose(macro.dbp_mae, 1.))
            for source in ("Overall", "MIMIC", "VitalDB"):
                self.assertTrue((args.output / f"{source}_diagnostics.csv").is_file())
            self.assertTrue((args.output / "RESULT_TABLES.md").is_file())
            with self.assertRaises(FileExistsError):
                evaluate.run(args)


if __name__ == "__main__":
    unittest.main()
