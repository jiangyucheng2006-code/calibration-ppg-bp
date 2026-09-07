"""E1 single-intervention contracts; synthetic data only."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pulsedb_fewshot.personal_memory_matched import (
    CANDIDATE, load_matched_cache, parser, rematch_training, synthetic_cache, train, training_contract,
)
from pulsedb_fewshot.personal_memory_prepare import prepare_neighbors

try:
    import torch
except ImportError:
    torch = None


def fixture(mode="random_disjoint"):
    rng = np.random.default_rng(81)
    rows = {}
    for role, count in (("train", 80), ("validation", 2)):
        rows[role] = []
        for i in range(count):
            uid = f"{role}{i}"
            start = i * 20 if role == "train" else 1800 + i * 20
            rows[role].append({"subject_uid": "MIMIC:a", "event_id": uid, "source": "MIMIC",
                "window_uid": uid, "waveform_sha256": hashlib.sha256(uid.encode()).hexdigest(),
                "recording_uid": "MIMIC:r", "time_axis_uid": "MIMIC:r", "start_s": float(start),
                "end_s": float(start + 10), "role": "train" if role == "train" else "internal_validation"})
    x = rng.normal(size=(80, 256)).astype(np.float32)
    v = rng.normal(size=(2, 256)).astype(np.float32)
    neighbors = prepare_neighbors(x, v, rows["train"], rows["validation"], mode=mode)
    arrays = {"train_features": x, "validation_features": v}
    for role in rows:
        for target, source in (("indices", "knn_indices"), ("weights", "knn_weights"), ("support", "support_weight")):
            arrays[f"{role}_{target}"] = neighbors[role][source]
    return {"split_mode": mode}, arrays, {role: pd.DataFrame(value) for role, value in rows.items()}


class MatchedDonorTests(unittest.TestCase):
    def test_real_smoke_exact_four_steps_without_synthetic_bypass(self):
        args = parser().parse_args(["--output", "unused", "--smoke-check"])
        self.assertTrue(training_contract(args))
        self.assertEqual((args.epochs, args.examples_per_epoch, args.batch_size), (1, 1024, 256))
        self.assertFalse(args.smoke)
        args.smoke = True
        with self.assertRaisesRegex(ValueError, "distinct"):
            training_contract(args)

    def test_production_epoch_cap_or_changed_seed_is_rejected(self):
        for controls in (["--epochs", "1"], ["--seed", "20260908"]):
            args = parser().parse_args(["--output", "unused", *controls])
            with self.assertRaisesRegex(ValueError, "fixed single-intervention"):
                training_contract(args)

    def test_only_training_donors_and_weights_can_change(self):
        manifest, arrays, metadata = fixture()
        original = {name: value.copy() for name, value in arrays.items()}
        updated, summary = rematch_training(manifest, arrays, metadata)
        self.assertGreater(summary["changed_train_rows"], 0)
        self.assertFalse(summary["alpha_recomputed"])
        self.assertFalse(summary["validation_references_recomputed"])
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], original[name])
            if name not in {"train_indices", "train_weights"}:
                np.testing.assert_array_equal(updated[name], original[name])
        self.assertEqual(summary["lineage_audit"]["cross_role_lineage"], "pass")
        self.assertFalse((updated["train_indices"] == np.arange(80)[:, None]).any())

    def test_chronological_donors_are_past_on_comparable_clock(self):
        manifest, arrays, metadata = fixture("chronological_blocked")
        updated, summary = rematch_training(manifest, arrays, metadata)
        self.assertTrue((updated["train_indices"][:5] == -1).all())
        self.assertTrue((updated["train_indices"][5:] >= 0).all())
        self.assertLess(summary["train_new"]["n_fallback"], summary["train_old"]["n_fallback"])
        for qi, references in enumerate(updated["train_indices"]):
            for ri in references[references >= 0]:
                self.assertLessEqual(metadata["train"].iloc[ri].end_s, metadata["train"].iloc[qi].start_s)
                self.assertEqual(metadata["train"].iloc[ri].time_axis_uid, metadata["train"].iloc[qi].time_axis_uid)

    def test_training_physiological_overlap_is_excluded(self):
        manifest, arrays, metadata = fixture()
        metadata["train"].loc[1, ["start_s", "end_s"]] = [5., 15.]
        # Make the illegal pair as similar as possible; it must remain absent.
        arrays["train_features"][1] = arrays["train_features"][0]
        updated, _ = rematch_training(manifest, arrays, metadata)
        self.assertNotIn(1, updated["train_indices"][0])
        self.assertNotIn(0, updated["train_indices"][1])

    def test_validation_labels_do_not_influence_rematching(self):
        manifest, arrays, metadata = fixture()
        first, summary = rematch_training(manifest, arrays, metadata)
        for role in metadata:
            metadata[role]["target_sbp"] = np.nan
            metadata[role]["target_dbp"] = -99999
        second, changed = rematch_training(manifest, arrays, metadata)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(summary, changed)

    def test_duplicate_hash_and_wrong_role_remain_rejected(self):
        for corruption in ("hash", "role"):
            with self.subTest(corruption=corruption):
                manifest, arrays, metadata = fixture()
                if corruption == "hash":
                    metadata["validation"].loc[0, "waveform_sha256"] = metadata["train"].loc[0, "waveform_sha256"]
                else:
                    metadata["validation"].loc[0, "role"] = "heldout_test"
                with self.assertRaises(ValueError):
                    rematch_training(manifest, arrays, metadata)

    def test_cross_role_interval_overlap_remains_rejected(self):
        manifest, arrays, metadata = fixture()
        metadata["validation"].loc[0, ["start_s", "end_s"]] = [1., 11.]
        with self.assertRaisesRegex(ValueError, "physiological interval overlap"):
            rematch_training(manifest, arrays, metadata)

    def test_deterministic_and_all_three_mismatch_views(self):
        manifest, arrays, metadata = fixture()
        one, summary = rematch_training(manifest, arrays, metadata)
        two, repeated = rematch_training(manifest, arrays, metadata)
        self.assertEqual(summary, repeated)
        for name in one:
            np.testing.assert_array_equal(one[name], two[name])
        for scope in ("train_old", "train_new", "validation_unchanged"):
            self.assertFalse(summary[scope]["query_targets_accessed"])
            self.assertIn("cosine_distance_all_donors", summary[scope])


@unittest.skipIf(torch is None, "PyTorch unavailable; end-to-end training must run on server")
class MatchedTrainingTests(unittest.TestCase):
    def test_synthetic_training_both_modes_keeps_epoch_zero_and_full_coverage(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ("random_disjoint", "chronological_blocked"):
                with self.subTest(mode=mode):
                    cache = synthetic_cache(root / (mode + "_cache"), mode)
                    args = parser().parse_args(["--cache-dir", str(cache), "--output", str(root / mode),
                        "--device", "cpu", "--epochs", "1", "--examples-per-epoch", "16",
                        "--batch-size", "8", "--smoke"])
                    run = train(args)
                    self.assertEqual(run["status"], "synthetic_smoke_complete")
                    self.assertEqual(run["candidate"], CANDIDATE)
                    self.assertEqual(run["optimizer_steps"], 2)
                    self.assertEqual(run["examples_processed"], 16)
                    self.assertEqual(run["internal_validation_windows"], 4)
                    self.assertEqual(set(run["metrics"]), {"Overall", "MIMIC", "VitalDB"})
                    self.assertFalse(run["heldout_test_accessed"])
                    self.assertTrue(run["fixed_v1_validation_references_preserved"])
                    self.assertTrue(run["fixed_v1_alpha_preserved"])
                    self.assertTrue((args.output / "initial_internal_validation_predictions.parquet").is_file())
                    self.assertTrue((args.output / "best.pt").is_file())
                    self.assertTrue((args.output / "best_trained.pt").is_file())
                    self.assertEqual(run["best_trained_epoch"], 1)
                    self.assertEqual(len(pd.read_parquet(args.output / "best_trained_internal_validation_predictions.parquet")), 4)
                    self.assertEqual(len(pd.read_parquet(args.output / "best_internal_validation_predictions.parquet")), 4)
                    self.assertLessEqual(run["metrics"]["Overall"]["mean_mae"], run["initial_metrics"]["Overall"]["mean_mae"])

    def test_synthetic_flag_and_hashes_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = synthetic_cache(Path(tmp) / "cache")
            with self.assertRaisesRegex(ValueError, "synthetic and real"):
                load_matched_cache(cache, smoke=False)
            np.save(cache / "train_features.npy", np.zeros((96, 256)), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "cache content changed"):
                load_matched_cache(cache, smoke=True)


if __name__ == "__main__":
    unittest.main()
