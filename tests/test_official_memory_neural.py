"""Small, explicitly synthetic end-to-end fits in a Torch+Parquet environment."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from test_official_memory_contract import cache_fixture, export_fixture, oof_fixtures


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch runtime required")
class TrustModelTests(unittest.TestCase):
    def test_zero_initialization_matches_fixed_blend(self):
        import torch
        from pulsedb_fewshot.official_memory_models import TrustFusion
        for per_bp in (False, True):
            model = TrustFusion(12, per_bp=per_bp)
            features = torch.randn(4, 12)
            base, memory = torch.randn(4, 2) + 100, torch.randn(4, 2) + 110
            alpha = torch.tensor([0., .2, .8, 1.])
            legal = torch.tensor([True, True, True, False])
            prediction, learned_alpha = model(features, base, memory, alpha, legal)
            expected = base + (alpha * legal)[:, None] * (memory - base)
            torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
            self.assertEqual(learned_alpha.shape, (4, 2 if per_bp else 1))

    def test_endpoint_gradient_and_bounds(self):
        import torch
        from pulsedb_fewshot.official_memory_models import TrustFusion
        model = TrustFusion(12)
        features = torch.ones(2, 12)
        base, memory = torch.zeros(2, 2), torch.ones(2, 2)
        prediction, _ = model(features, base, memory, torch.ones(2), torch.ones(2, dtype=torch.bool))
        prediction.sum().backward()
        self.assertGreater(abs(float(model.network[-1].bias.grad.sum())), 0)
        with torch.no_grad():
            model.network[-1].bias.fill_(-100)
        _, alpha = model(features, base, memory, torch.tensor([0., 1.]), torch.ones(2, dtype=torch.bool))
        self.assertTrue(bool(((alpha >= 0) & (alpha <= 1)).all()))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("pyarrow"),
                     "Torch and Parquet runtime required")
class OfficialMemoryFitSmokeTests(unittest.TestCase):
    def test_five_methods_complete_without_test_role(self):
        import torch
        from pulsedb_fewshot.official_memory_train import fit, parser
        torch.set_num_threads(1)
        bank, val = cache_fixture(n_train=96, n_val=3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "base"
            export_fixture(cache / "train", bank)
            export_fixture(cache / "validation", val)
            oof_roots = []
            for fold, pair in enumerate(oof_fixtures(bank)):
                oof = root / f"fold{fold}"
                export_fixture(oof / "source_fit", pair[0])
                export_fixture(oof / "excluded_fold", pair[1])
                oof_roots.append(oof)
            full_bank, test_inputs = cache_fixture(n_train=99, n_val=3)
            for item in (full_bank, test_inputs):
                item["manifest"].update(stage="final", official_test_accessed=True,
                                          official_test_targets_accessed=False, checkpoint_sha256="d" * 64)
            test_inputs["metadata"]["role"] = "test_inputs"
            test_inputs["manifest"]["labels_present"] = False
            del test_inputs["bp"]
            export_fixture(root / "final_cache" / "train", full_bank)
            export_fixture(root / "final_cache" / "test_inputs", test_inputs)
            for method in ("fixed", "v1", "e1", "trust_scalar", "trust_bp"):
                cli = ["--cache-root", str(cache), "--output", str(root / method), "--method", method,
                       "--epochs", "1", "--examples-per-epoch", "32", "--batch-size", "16",
                       "--device", "cpu", "--synthetic-smoke"]
                if method.startswith("trust"):
                    cli += ["--oof-roots", *map(str, oof_roots)]
                run = fit(parser().parse_args(cli))
                self.assertEqual(run["status"], "synthetic_smoke_complete")
                self.assertFalse(run["official_test_accessed"])
                self.assertFalse(run["query_bp_is_prediction_input"])
                self.assertEqual(run["epochs_completed"], 0 if method == "fixed" else 1)
                self.assertEqual(set(run["metrics"]), {"Overall", "MIMIC", "VitalDB"})
                if method.startswith("trust"):
                    self.assertEqual(len(run["crossfit_evidence"]), 3)
                    self.assertEqual(run["trust_scaler"]["fit_role"], "train_encoder_OOF_only")
                self.assertTrue((root / method / "best.pt").is_file())
                self.assertTrue(np.isfinite(run["metrics"]["Overall"]["mean_mae"]))
                final_cli = ["--cache-root", str(root / "final_cache"), "--output", str(root / (method + "_final")),
                             "--method", method, "--stage", "final", "--selection-run", str(root / method),
                             "--examples-per-epoch", "32", "--batch-size", "16", "--device", "cpu", "--synthetic-smoke"]
                final_run = fit(parser().parse_args(final_cli))
                self.assertFalse(final_run["official_test_targets_accessed"])
                self.assertTrue(final_run["official_test_inputs_accessed"])
                self.assertIsNone(final_run["metrics"])
                prediction = pd.read_parquet(root / (method + "_final") / "official_test_input_predictions.parquet")
                self.assertEqual(set(prediction), {"subject_uid", "event_id", "source", "pred_sbp", "pred_dbp"})
                self.assertEqual(len(prediction), len(test_inputs["metadata"]))
                self.assertTrue(np.isfinite(prediction[["pred_sbp", "pred_dbp"]]).all().all())
                self.assertTrue(final_run["frozen_configuration"]["frozen_before_optimizer"])
                if method.startswith("trust"):
                    self.assertEqual(final_run["optimizer_steps"], 0)
                    self.assertTrue(final_run["gate_frozen_from_inner_320_OOF"])


if __name__ == "__main__":
    unittest.main()
