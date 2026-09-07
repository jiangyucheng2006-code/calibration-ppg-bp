"""Synthetic-only contracts for experimental personal measurement memory.

No patient file or trained project checkpoint is read by these tests. Torch is
optional for non-training installations; a skip is not a passed GPU smoke.
"""

from __future__ import annotations

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
    from pulsedb_fewshot.personal_memory_models import (
        PersonalMemoryRelation, reference_prediction, supervised_relation_loss,
    )
    from pulsedb_fewshot.personal_memory_train import (
        load_cache, parser, train, validate_gate,
    )
    from pulsedb_fewshot.training import file_sha256


def synthetic_cache(root: Path) -> Path:
    """Write a small, flagged fixture with two sources and four people."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    frames, arrays = {}, {}
    train_bp = []
    for role, per_person in (("train", 8), ("validation", 2)):
        rows, features, bps, neighbours = [], [], [], []
        for person in range(4):
            for within in range(per_person):
                bp = np.array([115 + person * 8 + within, 65 + person * 3 + within * .5], dtype=np.float32)
                rows.append({"subject_uid": f"s{person}", "event_id": f"{role}-{person}-{within}",
                             "source": "MIMIC" if person < 2 else "VitalDB",
                             "role": "train" if role == "train" else "internal_validation",
                             "target_sbp": bp[0], "target_dbp": bp[1]})
                features.append(rng.normal(size=256).astype(np.float32) * .2)
                bps.append(bp)
                candidates = [person * 8 + x for x in range(8) if role != "train" or x != within]
                neighbours.append(candidates[:5])
        frames[role] = pd.DataFrame(rows)
        arrays[f"{role}_features.npy"] = np.stack(features)
        arrays[f"{role}_bp.npy"] = np.stack(bps)
        arrays[f"{role}_base.npy"] = np.stack(bps) + np.array([3., -2.], dtype=np.float32)
        arrays[f"{role}_knn_indices.npy"] = np.array(neighbours, dtype=np.int64)
        arrays[f"{role}_uniform_indices.npy"] = np.array(neighbours, dtype=np.int64)[:, ::-1].copy()
        arrays[f"{role}_knn_weights.npy"] = np.full((len(rows), 5), .2, dtype=np.float32)
        arrays[f"{role}_support_weight.npy"] = np.full(len(rows), .5, dtype=np.float32)
    # One legal-fallback query remains in evaluation and keeps its base value.
    arrays["validation_knn_indices.npy"][0] = -1
    arrays["validation_uniform_indices.npy"][0] = -1
    arrays["validation_knn_weights.npy"][0] = 0
    manifest = {
        "status": "complete", "protocol_id": "development-calbased-analogue-v1",
        "split_mode": "random_disjoint", "source_parent_split": "meta_train",
        "read_roles": ["train", "internal_validation"], "heldout_test_accessed": False,
        "synthetic_smoke": True, "target_scaler": {"mean": [125., 72.], "std": [20., 10.]},
        "source_checkpoint": {"synthetic": True, "description": "no real checkpoint read"},
        "personal_label_budget_windows": 8, "files": {},
    }
    for name, value in arrays.items():
        np.save(root / name, value, allow_pickle=False)
    for role, frame in frames.items():
        frame.to_parquet(root / f"{role}_metadata.parquet", index=False)
    for path in root.iterdir():
        manifest["files"][path.name] = {"path": path.name, "sha256": file_sha256(path)}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@unittest.skipIf(torch is None, "PyTorch unavailable: training/GPU smoke remains unverified")
class PersonalMemoryModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = PersonalMemoryRelation(4, 6)
        self.query = torch.randn(3, 4)
        self.reference = torch.randn(3, 4)

    def test_zero_initial_delta(self):
        torch.testing.assert_close(self.model(self.query, self.reference), torch.zeros(3, 2))

    def test_antisymmetry_after_nonzero_weights(self):
        torch.nn.init.normal_(self.model.network[-1].weight)
        torch.testing.assert_close(self.model(self.query, self.reference),
                                   -self.model(self.reference, self.query), atol=0, rtol=0)

    def test_identical_feature_zero(self):
        torch.nn.init.normal_(self.model.network[-1].weight)
        torch.testing.assert_close(self.model(self.query, self.query), torch.zeros(3, 2), atol=0, rtol=0)

    def test_prediction_api_does_not_accept_query_bp(self):
        self.assertNotIn("query_bp", inspect.signature(reference_prediction).parameters)
        self.assertNotIn("query_bp", inspect.signature(PersonalMemoryRelation.forward).parameters)

    def test_wrong_feature_dimensions_fail(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 3), torch.randn(2, 3))

    def _predict(self, support=None, mask=None):
        ref = torch.randn(3, 2, 4)
        reference_bp = torch.tensor([[[100., 60.], [120., 70.]]] * 3)
        base = torch.tensor([[130., 80.]] * 3)
        weights = torch.tensor([[.25, .75]] * 3)
        support = torch.ones(3) if support is None else support
        mask = torch.ones(3, 2, dtype=torch.bool) if mask is None else mask
        return reference_prediction(self.model, self.query, ref, reference_bp, base,
                                    weights, mask, support, torch.tensor([20., 10.]))

    def test_initial_prediction_is_weighted_reference_bp(self):
        prediction, _ = self._predict()
        torch.testing.assert_close(prediction, torch.tensor([[115., 67.5]] * 3))

    def test_all_invalid_fallback_is_exact(self):
        prediction, _ = self._predict(mask=torch.zeros(3, 2, dtype=torch.bool))
        torch.testing.assert_close(prediction, torch.tensor([[130., 80.]] * 3), atol=0, rtol=0)

    def test_zero_support_weight_is_exact_baseline(self):
        prediction, _ = self._predict(support=torch.zeros(3))
        torch.testing.assert_close(prediction, torch.tensor([[130., 80.]] * 3), atol=0, rtol=0)

    def test_pair_loss_has_finite_gradients(self):
        prediction, delta = self._predict()
        target = torch.tensor([[118., 68.]] * 3)
        refs = torch.tensor([[[100., 60.], [120., 70.]]] * 3)
        loss, _, _ = supervised_relation_loss(prediction, target, delta, refs,
            torch.ones(3, 2, dtype=torch.bool), torch.tensor([20., 10.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in self.model.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_gate_requires_two_modes_and_same_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            gate = {"status": "proceed_to_train", "heldout_test_accessed": False,
                    "cache_manifest_sha256": {"random_disjoint": "a", "chronological_blocked": "b"}}
            path.write_text(json.dumps(gate))
            self.assertEqual(validate_gate(path, {"split_mode": "random_disjoint"}, "a", False)["status"],
                             "proceed_to_train")
            with self.assertRaises(ValueError):
                validate_gate(path, {"split_mode": "random_disjoint"}, "different", False)

    def test_gate_cannot_be_bypassed_for_real_cache(self):
        with self.assertRaises(ValueError):
            validate_gate(None, {"split_mode": "random_disjoint"}, "a", False)
        with self.assertRaises(ValueError):
            validate_gate(None, {"split_mode": "random_disjoint"}, "a", True)

    def test_cache_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = synthetic_cache(Path(directory) / "cache")
            np.save(cache / "train_bp.npy", np.zeros((32, 2)))
            with self.assertRaisesRegex(ValueError, "cache content changed"):
                load_cache(cache, "pair_single", smoke=True)

    def test_incomplete_real_cohort_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = synthetic_cache(Path(directory) / "cache")
            path = cache / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["synthetic_smoke"] = False
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "requires 320 rows"):
                load_cache(cache, "pair_single", smoke=False)

    def test_synthetic_end_to_end_all_candidates(self):
        from pulsedb_fewshot.personal_memory_models import CANDIDATES
        with tempfile.TemporaryDirectory() as directory:
            cache = synthetic_cache(Path(directory) / "cache")
            for candidate in CANDIDATES:
                with self.subTest(candidate=candidate):
                    args = parser().parse_args([
                        "--cache-dir", str(cache), "--output", str(Path(directory) / candidate),
                        "--candidate", candidate, "--device", "cpu", "--epochs", "1",
                        "--examples-per-epoch", "8", "--batch-size", "4", "--smoke",
                    ])
                    run = train(args)
                    self.assertEqual(run["status"], "synthetic_smoke_complete")
                    self.assertFalse(run["heldout_test_accessed"])
                    self.assertEqual(run["fallback_validation_windows"], 1)
                    self.assertEqual(set(run["metrics"]), {"Overall", "MIMIC", "VitalDB"})
                    predictions = pd.read_parquet(args.output / "best_internal_validation_predictions.parquet")
                    self.assertEqual(len(predictions), 8)
                    np.testing.assert_array_equal(predictions.iloc[0][["pred_sbp", "pred_dbp"]].to_numpy(float),
                                                  np.array([118., 63.]))
                    self.assertTrue((args.output / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
