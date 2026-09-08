"""Public aggregate validation without Torch or private participant artifacts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publish_memory_v2", REPO / "scripts/publish_personal_memory_v2_results.py")
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def metric_map(score=3.0):
    return {scope: {"n_participants": people, "n_events": events,
                    "sbp_mae": score + 1, "dbp_mae": score - 1, "mean_mae": score}
            for scope, (people, events) in publisher.EXPECTED_COUNTS.items()}


def macro_fixture(mode="random_disjoint"):
    return pd.DataFrame([{"candidate": candidate, "view": scope, "split_mode": mode, **metrics}
                         for candidate in publisher.SETTINGS for scope, metrics in metric_map().items()], columns=publisher.MACRO)


def pooled_fixture(macro):
    return pd.DataFrame([{"Scope": r["view"], "Setting": r.candidate, "BP": bp,
                          "MAE": r[f"{bp.lower()}_mae"], "R²": .8, "ME": .1, "STD": 5,
                          "≤5 mmHg": 80, "≤10 mmHg": 94, "≤15 mmHg": 98,
                          "AAMI": "PASS*", "BHS": "PASS (Grade A)*"}
                         for _, r in macro.iterrows() for bp in ("SBP", "DBP")], columns=publisher.POOLED)


def training_fixture(mode="random_disjoint", route="e1"):
    candidate = "E1_matched_relation" if route == "e1" else "E2_bp_state_metric"
    history = [{"epoch": i, "internal_validation": metric_map(3.1), "stale_epochs": i,
                "optimizer_steps_total": 782 * i, "examples": 200000, "epoch_seconds": 1.0}
               for i in range(1, 9)]
    run = {"status": "complete", "screen_id": publisher.SCREEN, "seed": publisher.SEED,
           "heldout_test_accessed": False, "split_mode": mode, "candidate": candidate, **publisher.CONTRACT,
           "personal_label_budget_windows": 320, "train_participants": 2051, "train_windows_available": 656320,
           "internal_validation_participants": 2051, "internal_validation_windows": 82040,
           "validation_query_coverage": 1.0, "persistent_LoRA_preserved": True, "retrieved_references": 5,
           "arguments": {"seed": publisher.SEED, "patience": 8, "epochs": 0, "batch_size": 256,
                         "examples_per_epoch": 200000, "split_mode": "random_disjoint",
                         "smoke": False, "smoke_check": False, "synthetic_smoke": False},
           "retrieval_training_policy": "physical-gap-v2", "fixed_v1_alpha_preserved": True,
           "fixed_v1_validation_references_preserved": True, "query_BP_used_for_retrieval": False,
           "gate_recalibrated": False, "relation_correction": "zero", "fixed_alpha": "unchanged_original256d_cache_support_weight",
           "stop_reason": "early_stopping", "epochs_completed": 8, "optimizer_steps": 6256,
           "examples_processed": 1600000, "runtime_seconds": 15.0, "initial_metrics": metric_map(),
           "initial_random_projection_metrics": metric_map(), "metrics": metric_map(), "best_epoch": 0,
           "best_trained_epoch": 1, "best_trained_metrics": metric_map(3.1),
           "best_trained_checkpoint_sha256": "a" * 64, "trainable_parameters": 65730 if route == "e1" else 8192,
           "gpu": "test GPU"}
    return run, history, metric_map(), macro_fixture(mode)


class PublicationValidationTests(unittest.TestCase):
    def test_complete_macro_and_pooled(self):
        frame = macro_fixture()
        publisher.validate_macro(frame, publisher.MODES[0])
        publisher.validate_pooled(pooled_fixture(frame), frame)

    def test_private_macro_column_rejected(self):
        frame = macro_fixture().assign(subject_uid="secret")
        with self.assertRaisesRegex(ValueError, "private macro"):
            publisher.validate_macro(frame, publisher.MODES[0])

    def test_full_query_cohort_required(self):
        frame = macro_fixture()
        frame.loc[0, "n_events"] -= 1
        with self.assertRaisesRegex(ValueError, "cohort"):
            publisher.validate_macro(frame, publisher.MODES[0])

    def test_nan_macro_rejected(self):
        frame = macro_fixture()
        frame.loc[0, "sbp_mae"] = float("nan")
        with self.assertRaises(ValueError):
            publisher.validate_macro(frame, publisher.MODES[0])

    def test_overall_is_consistency_checked_not_equal_source_average(self):
        values = metric_map()
        values["MIMIC"].update(sbp_mae=3, dbp_mae=1, mean_mae=2)
        values["VitalDB"].update(sbp_mae=5, dbp_mae=3, mean_mae=4)
        with self.assertRaises(ValueError):
            publisher.validate_metric_map(values)
        for k in publisher.METRICS:
            values["Overall"][k] = (values["MIMIC"][k] * 1011 + values["VitalDB"][k] * 1040) / 2051
        publisher.validate_metric_map(values)

    def test_unqualified_standard_label_rejected(self):
        macro = macro_fixture()
        pooled = pooled_fixture(macro)
        pooled.loc[0, "AAMI"] = "PASS"
        with self.assertRaisesRegex(ValueError, "numerical-screen"):
            publisher.validate_pooled(pooled, macro)

    def test_invalid_percentages_rejected(self):
        macro = macro_fixture()
        pooled = pooled_fixture(macro)
        pooled.loc[0, "≤5 mmHg"] = 99
        with self.assertRaisesRegex(ValueError, "threshold"):
            publisher.validate_pooled(pooled, macro)

    def test_tampered_comparison_rejected(self):
        macro = macro_fixture()
        comparison, _ = publisher.comparisons(macro)
        bad = comparison.copy()
        bad.loc[0, "gain_vs_reference_mmhg"] += .1
        with self.assertRaises(ValueError):
            publisher.frame_equal(bad, comparison, ["candidate", "view", "split_mode"])

    def test_training_best_zero_and_actual_best_trained_distinct(self):
        run, history, metrics, macro = training_fixture()
        rows = publisher.validate_training(run, history, metrics, publisher.MODES[0], "e1", macro)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["best_epoch"], 0)
        self.assertEqual(rows[0]["best_trained_epoch"], 1)
        self.assertEqual(rows[0]["best_trained_evidence"], "saved_separate_checkpoint_metrics")

    def test_e2_best_trained_is_history_only(self):
        run, history, metrics, macro = training_fixture(route="e2")
        rows = publisher.validate_training(run, history, metrics, publisher.MODES[0], "e2", macro)
        self.assertEqual(rows[0]["best_trained_evidence"], "derived_history_not_separate_checkpoint")

    def test_chrono_unused_synthetic_cli_default_is_explicit(self):
        run, history, metrics, macro = training_fixture(mode=publisher.MODES[1])
        rows = publisher.validate_training(run, history, metrics, publisher.MODES[1], "e1", macro)
        self.assertEqual(rows[0]["split_mode"], "chronological_blocked")
        self.assertEqual(rows[0]["synthetic_smoke_only_cli_split_mode"], "random_disjoint")

    def test_incomplete_patience_or_steps_rejected(self):
        for field, bad in (("stale_epochs", 7), ("optimizer_steps_total", 6255)):
            run, history, metrics, macro = training_fixture()
            history[-1][field] = bad
            with self.assertRaises(ValueError):
                publisher.validate_training(run, history, metrics, publisher.MODES[0], "e1", macro)

    def test_wrong_best_epoch_rejected(self):
        run, history, metrics, macro = training_fixture()
        run["best_epoch"] = 1
        with self.assertRaisesRegex(ValueError, "selected epoch"):
            publisher.validate_training(run, history, metrics, publisher.MODES[0], "e1", macro)

    def test_heldout_or_wrong_role_rejected(self):
        for key, value in (("heldout_test_accessed", True), ("source_parent_split", "meta_test")):
            run, history, metrics, macro = training_fixture()
            run[key] = value
            with self.assertRaises(ValueError):
                publisher.validate_training(run, history, metrics, publisher.MODES[0], "e1", macro)

    def test_inputs_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            role = Path(tmp) / "batch_both_final"
            role.mkdir()
            (role / "selection.json").write_text("{}", encoding="utf-8")
            inputs = publisher.Inputs(tmp)
            inputs.check("both_final", "selection.json", publisher.digest(role / "selection.json"))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                inputs.check("both_final", "selection.json", "a" * 64)

    def test_output_preflight_prevents_partial_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "public"
            output.mkdir()
            (output / "old.csv").write_text("existing", encoding="utf-8")
            with patch.object(publisher, "build_publication", return_value=({"new.csv": "new", "old.csv": "changed"}, {})):
                with self.assertRaises(FileExistsError):
                    publisher.publish(Path(tmp) / "inputs", output)
            self.assertFalse((output / "new.csv").exists())
            self.assertEqual((output / "old.csv").read_text(), "existing")

    def test_identical_rerun_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(publisher, "build_publication", return_value=({"result.csv": "ok\n"}, {"status": "pass"})):
                for _ in range(2):
                    self.assertEqual(publisher.publish(Path(tmp) / "inputs", Path(tmp) / "public")["status"], "pass")

    def test_markdown_has_every_main_row_and_qualification(self):
        macros = pd.concat([macro_fixture(m) for m in publisher.MODES], ignore_index=True)
        pools = pd.concat([pooled_fixture(macro_fixture(m)).assign(split_mode=m) for m in publisher.MODES], ignore_index=True)
        text = publisher.render_tables(macros, pools)
        self.assertEqual(text.count("### "), 6)
        self.assertEqual(text.count("| E1_matched_relation |"), 18)
        self.assertIn("retrospective numerical screens only", text)
        self.assertIsNone(publisher.FORBIDDEN.search(text))

    def test_private_content_scanner(self):
        for value in ("/home/research/results", "private_example_predictions.parquet", "subject_uid", "best.pt", "C:\\Research\\private"):
            self.assertIsNotNone(publisher.FORBIDDEN.search(value))

    @unittest.skipUnless((REPO / "local_archive/personal_memory_v2_results_20260908").is_dir(), "private aggregate archive not installed")
    def test_optional_completed_archive_integration(self):
        outputs, report = publisher.build_publication(REPO / "local_archive/personal_memory_v2_results_20260908")
        self.assertEqual(len(report["validated_input_files"]), 33)
        self.assertEqual(report["rows"]["participant_macro_summary.csv"], 48)
        self.assertEqual(report["rows"]["event_pooled_diagnostics_all_scopes.csv"], 96)
        self.assertEqual(report["eligible_by_historical_LoRA_gate"], [])
        self.assertFalse(report["raw_prediction_recomputation"])
        self.assertTrue(all(publisher.FORBIDDEN.search(value) is None for value in outputs.values()))


if __name__ == "__main__":
    unittest.main()
