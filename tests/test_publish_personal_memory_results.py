"""Public aggregate integrity regressions; no torch, network or private data."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publish_personal_memory_results", ROOT / "scripts" / "publish_personal_memory_results.py")
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)


class PersonalMemoryPublicationTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / "results" / "personal_memory_v1"
        macro, pooled, comparisons, selections = {}, {}, {}, {}
        for mode in PUBLISHER.MODES:
            macro[mode] = pd.read_csv(directory / mode / "participant_macro_summary.csv")
            pooled[mode] = pd.read_csv(directory / mode / "event_pooled_diagnostics_all_scopes.csv")
            comparisons[mode] = pd.read_csv(directory / mode / "comparison_vs_reference.csv")
            selections[mode] = json.loads((directory / mode / "selection.json").read_text(encoding="utf-8"))
        combined = pd.read_csv(directory / "final" / "cross_split_comparison.csv")
        gate = pd.read_csv(directory / "final" / "promotion_gate.csv")
        final = json.loads((directory / "final" / "selection.json").read_text(encoding="utf-8"))
        self.bundle = [macro, pooled, comparisons, selections, combined, gate, final]

    def test_completed_real_aggregate_bundle(self):
        checked = PUBLISHER.validate(*self.bundle)
        self.assertEqual(checked["eligible_candidates"], [])
        self.assertEqual(checked["completed_model_fits"], 10)
        self.assertEqual(checked["pooled_diagnostic_rows"], 72)
        self.assertEqual(checked["references"]["chronological_blocked"], "D0_frozen_lora")

    def test_incomplete_cohort_is_rejected(self):
        self.bundle[0]["random_disjoint"].loc[0, "n_events"] -= 1
        with self.assertRaisesRegex(ValueError, "cohort"):
            PUBLISHER.validate(*self.bundle)

    def test_missing_candidate_is_rejected(self):
        self.bundle[0]["random_disjoint"] = self.bundle[0]["random_disjoint"].iloc[:-3]
        with self.assertRaisesRegex(ValueError, "macro"):
            PUBLISHER.validate(*self.bundle)

    def test_unqualified_standard_is_rejected(self):
        self.bundle[1]["random_disjoint"].loc[0, "AAMI"] = "PASS"
        with self.assertRaisesRegex(ValueError, "AAMI/BHS"):
            PUBLISHER.validate(*self.bundle)

    def test_non_cumulative_percentages_are_rejected(self):
        self.bundle[1]["random_disjoint"].loc[0, "≤15 mmHg"] = 1
        with self.assertRaisesRegex(ValueError, "percentages"):
            PUBLISHER.validate(*self.bundle)

    def test_wrong_paired_reference_is_rejected(self):
        self.bundle[3]["random_disjoint"]["reference"] = "D0_frozen_lora"
        with self.assertRaisesRegex(ValueError, "stronger Overall"):
            PUBLISHER.validate(*self.bundle)

    def test_cross_split_metric_change_is_rejected(self):
        self.bundle[4].loc[0, "mean_mae"] += 1
        with self.assertRaises(AssertionError):
            PUBLISHER.validate(*self.bundle)

    def test_false_promotion_is_rejected(self):
        self.bundle[5].loc[0, "passes_accuracy_gate"] = True
        with self.assertRaisesRegex(ValueError, "promotion gate"):
            PUBLISHER.validate(*self.bundle)

    def test_heldout_access_is_rejected(self):
        self.bundle[6]["heldout_test_accessed"] = True
        with self.assertRaisesRegex(ValueError, "held-out"):
            PUBLISHER.validate(*self.bundle)

    def test_private_paths_and_prediction_fields_are_recognized(self):
        for value in ("/home/example/private", "C:\\Research\\private", "subject_uid", "pred_sbp"):
            self.assertIsNotNone(PUBLISHER.FORBIDDEN.search(value))
        self.assertIsNone(PUBLISHER.FORBIDDEN.search("source_checkpoint_sha256"))

    def test_render_has_six_full_metric_sections_and_scope_limits(self):
        macro, pooled, _, _, combined, _, final = deepcopy(self.bundle)
        text = PUBLISHER.render(macro, pooled, combined, final)
        for mode in PUBLISHER.MODES:
            for scope in PUBLISHER.SCOPES:
                self.assertIn(f"### {mode} · {scope}", text)
        self.assertIn("epoch 0", text)
        self.assertIn("320", text)
        self.assertIn("60 秒实验针对冻结诊断方法", text)
        self.assertIn("不正式替换", text)


if __name__ == "__main__":
    unittest.main()
