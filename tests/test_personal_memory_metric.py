"""Synthetic-only tests for E2 retrieval metric and information boundaries."""

from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from pulsedb_fewshot.personal_memory_metric import (
        BPStateProjection, CANDIDATE, _tensors, build_layout, fixed_alpha_fusion,
        legal_mask, make_synthetic_cache, metric_objective, parser,
        prediction_values, retrieve, sample_training_pairs, train,
    )
    from pulsedb_fewshot.personal_memory_prepare import FIELDS
    from pulsedb_fewshot.personal_memory_train import load_cache


@unittest.skipIf(torch is None, "PyTorch absent; E2 tests require server smoke, not a claimed local pass")
class PersonalMemoryMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.cache = make_synthetic_cache(Path(cls.temp.name) / "cache")
        cls.manifest, cls.arrays, cls.metadata = load_cache(cls.cache, "pair_distance_blend", smoke=True)
        cls.rows = {role: cls.metadata[role][list(FIELDS)].to_dict("records") for role in ("train", "validation")}
        cls.layout = build_layout(cls.rows["train"], cls.rows["validation"], "random_disjoint",
                                  cls.arrays["train_indices"], cls.arrays["validation_indices"])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def tensors(self):
        return _tensors(self.arrays, self.layout, torch.device("cpu"))

    def test_projection_capacity_and_finite_gradient(self):
        model = BPStateProjection()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8192)
        loss, _ = metric_objective(model, torch.randn(12, 256), torch.randn(12, 256),
            torch.randn(12, 2) * 20 + 100, torch.randn(12, 2) * 20 + 100,
            torch.tensor([20., 10.]), torch.ones(12, dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.projection.weight.grad).all())
        self.assertTrue((model.projection.weight.grad != 0).any())

    def test_invalid_train_scales_fail(self):
        with self.assertRaisesRegex(ValueError, "scales"):
            metric_objective(BPStateProjection(), torch.randn(3, 256), torch.randn(3, 256),
                torch.zeros(3, 2), torch.zeros(3, 2), torch.tensor([0., 10.]), torch.zeros(3, dtype=torch.bool))

    def test_projection_dimension_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            BPStateProjection()(torch.randn(3, 128))

    def test_complete_training_block_and_self_excluded(self):
        bank = self.rows["train"][:80]
        mask = legal_mask(bank, bank, "random_disjoint", training=True)
        self.assertFalse(mask[:40, :40].any())
        self.assertFalse(mask[40:, 40:].any())
        self.assertTrue(mask[:40, 40:].all())
        self.assertFalse(np.diag(mask).any())

    def test_chronological_only_comparable_past(self):
        bank = self.rows["train"][:80]
        mask = legal_mask(bank, bank, "chronological_blocked", training=True)
        self.assertFalse(mask[:40].any())
        self.assertTrue(mask[40:, :40].all())
        query = deepcopy(self.rows["validation"][:4])
        query[0]["time_axis_uid"] = "another_clock"
        valmask = legal_mask(bank, query, "chronological_blocked")
        self.assertFalse(valmask[0].any())

    def test_same_person_required(self):
        with self.assertRaisesRegex(ValueError, "cross-person"):
            legal_mask(self.rows["train"][:80], self.rows["validation"][4:8], "random_disjoint")

    def test_validation_role_and_hash_guard(self):
        query = deepcopy(self.rows["validation"][:4])
        query[0]["role"] = "heldout_test"
        with self.assertRaisesRegex(ValueError, "role"):
            legal_mask(self.rows["train"][:80], query, "random_disjoint")
        query[0]["role"] = "internal_validation"
        query[0]["waveform_sha256"] = self.rows["train"][0]["waveform_sha256"]
        self.assertFalse(legal_mask(self.rows["train"][:80], query, "random_disjoint")[0, 0])

    def test_validation_targets_cannot_change_retrieval_or_prediction(self):
        model = BPStateProjection()
        tensors = self.tensors()
        first = prediction_values(model, tensors, subjects_per_batch=2)
        # No target tensor is created from query BP, and the inference API has
        # no label argument. An unrelated injected tensor cannot affect output.
        self.assertNotIn("validation_bp", tensors)
        tensors["validation_bp"] = torch.full((16, 2), 99999.)
        second = prediction_values(model, tensors, subjects_per_batch=2)
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        for function in (retrieve, prediction_values, fixed_alpha_fusion):
            self.assertNotIn("query_bp", inspect.signature(function).parameters)

    def test_metadata_target_values_do_not_change_legality_or_pair_pools(self):
        rows = deepcopy(self.rows)
        for row in rows["validation"]:
            row.update(target_sbp=-100000., target_dbp=100000.)
        altered = build_layout(rows["train"], rows["validation"], "random_disjoint", self.arrays["train_indices"])
        for key in ("bank_order", "query_order", "validation_legal", "temporal_pool", "eligible_train"):
            np.testing.assert_array_equal(self.layout[key], altered[key])

    def test_original_alpha_is_not_recalibrated(self):
        tensors = self.tensors()
        alpha = tensors["validation_support"].clone()
        model = BPStateProjection()
        prediction, memory, _, _ = prediction_values(model, tensors)
        torch.testing.assert_close(tensors["validation_support"], alpha, atol=0, rtol=0)
        torch.testing.assert_close(prediction, tensors["validation_base"] + alpha[:, None] * (memory - tensors["validation_base"]))
        tensors["validation_support"] = torch.zeros_like(alpha)
        zero, _, _, _ = prediction_values(model, tensors)
        torch.testing.assert_close(zero, tensors["validation_base"], atol=0, rtol=0)

    def test_all_invalid_falls_back_and_retains_query(self):
        tensors = self.tensors()
        q = int(tensors["query_order"][0, 0])
        tensors["validation_legal"][0, 0] = False
        predicted, memory, indices, weights = prediction_values(BPStateProjection(), tensors)
        torch.testing.assert_close(predicted[q], tensors["validation_base"][q], atol=0, rtol=0)
        torch.testing.assert_close(memory[q], tensors["validation_base"][q], atol=0, rtol=0)
        self.assertEqual(len(predicted), 16)
        self.assertTrue((indices[q] == -1).all())
        self.assertTrue((weights[q] == 0).all())

    def test_tensor_layout_storage_is_independent_per_execution(self):
        first, second = self.tensors(), self.tensors()
        for key in ("bank_order", "query_order", "validation_legal", "temporal_pool", "eligible_train"):
            expected = self.layout[key].copy()
            original = first[key].reshape(-1)[0].item()
            first[key].reshape(-1)[0] = not original if key == "validation_legal" else original + 1
            np.testing.assert_array_equal(self.layout[key], expected)
            np.testing.assert_array_equal(second[key].cpu().numpy(), expected)
            third = self.tensors()
            np.testing.assert_array_equal(third[key].cpu().numpy(), expected)

    def test_batched_retrieval_consistent_and_five_distinct(self):
        tensors = self.tensors()
        model = BPStateProjection()
        i1, w1 = retrieve(model, tensors, subjects_per_batch=1)
        i2, w2 = retrieve(model, tensors, subjects_per_batch=4)
        torch.testing.assert_close(i1, i2, atol=0, rtol=0)
        torch.testing.assert_close(w1, w2, atol=2e-6, rtol=0)
        self.assertTrue((torch.diff(i1.sort(1).values, dim=1) > 0).all())
        torch.testing.assert_close(w1.sum(1), torch.ones(16))

    def test_sampled_training_pairs_preserve_identity_and_block(self):
        tensors = self.tensors()
        generator = torch.Generator().manual_seed(10)
        q, r, close = sample_training_pairs(tensors, 1024, generator)
        self.assertFalse((q == r).any())
        self.assertTrue(((q // 80) == (r // 80)).all())
        self.assertTrue(((q % 80 // 40) != (r % 80 // 40)).all())
        self.assertGreater(int(close.sum()), 300)
        self.assertLess(int(close.sum()), 700)

    def test_synthetic_end_to_end_produces_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parser().parse_args(["--cache-dir", str(self.cache), "--output", str(Path(directory) / "run"),
                "--device", "cpu", "--smoke", "--epochs", "1", "--batch-size", "8", "--examples-per-epoch", "32"])
            result = train(args)
            self.assertEqual(result["candidate"], CANDIDATE)
            self.assertEqual(result["status"], "synthetic_smoke_complete")
            self.assertEqual(result["optimizer_steps"], 4)
            self.assertEqual(result["internal_validation_windows"], 16)
            self.assertFalse(result["heldout_test_accessed"])
            self.assertFalse(result["query_BP_used_for_retrieval"])
            self.assertFalse(result["gate_recalibrated"])
            self.assertEqual(set(result["metrics"]), {"Overall", "MIMIC", "VitalDB"})
            self.assertTrue((args.output / "best.pt").is_file())
            self.assertEqual(len(pd.read_parquet(args.output / "best_internal_validation_predictions.parquet")), 16)
            self.assertEqual(len(pd.read_csv(args.output / "internal_validation_pooled_diagnostics.csv")), 6)

    def test_production_no_epoch_cap_guard(self):
        args = parser().parse_args(["--cache-dir", str(self.cache), "--output", "unused", "--epochs", "1"])
        with self.assertRaisesRegex(ValueError, "no epoch cap"):
            train(args)


if __name__ == "__main__":
    unittest.main()
