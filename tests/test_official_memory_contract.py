"""NumPy-only contract tests; neural smoke tests run in the project Torch env."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pulsedb_fewshot.official_memory_train import (
    PROTOCOL, CACHE_FILES, TRUST_FEATURE_NAMES, event_digest, sha256_file,
    load_role, audit_roles, audit_crossfit, prepare_pair, memory_inputs, training_std,
)


def cache_fixture(n_train=48, n_val=3):
    rng = np.random.default_rng(12)
    caches = []
    for role, count, shift in (("train", n_train, 0), ("validation", n_val, 2000)):
        rows = []
        for source in ("MIMIC", "VitalDB"):
            for j in range(count):
                uid = f"{source}-{role}-{j}"
                rows.append({"subject_uid": source + "person", "event_id": uid, "source": source,
                             "recording_uid": source + "record", "time_axis_uid": source + "clock",
                             "start_s": float(shift + j * 20), "end_s": float(shift + j * 20 + 10),
                             "waveform_sha256": hashlib.sha256(uid.encode()).hexdigest(), "role": role})
        features = rng.normal(size=(len(rows), 256)).astype(np.float32)
        bp = (np.array([125., 75.]) + rng.normal(size=(len(rows), 2)) * 8).astype(np.float32)
        caches.append({"metadata": pd.DataFrame(rows), "features": features,
                       "base": bp + np.array([3., -2.], dtype=np.float32), "bp": bp,
                       "manifest": {"protocol_id": PROTOCOL, "official_test_accessed": False,
                                    "checkpoint_sha256": "a" * 64, "synthetic_smoke": True,
                                    "official_contract_sha256": "c" * 64,
                                    "bp_units": "mmHg", "fit_transforms_on_source_fit_only": True}})
    return caches


def export_fixture(path, cache):
    path.mkdir(parents=True)
    cache["metadata"].to_parquet(path / "metadata.parquet", index=False)
    for key, filename in (("features", "features.npy"), ("base", "base_bp.npy"), ("bp", "bp.npy")):
        if key in cache and cache[key] is not None:
            np.save(path / filename, cache[key], allow_pickle=False)
    manifest = {**cache["manifest"], "files": {name: sha256_file(path / name) for name in CACHE_FILES if (path / name).exists()}}
    (path / "provenance.json").write_text(json.dumps(manifest), encoding="utf-8")


def oof_fixtures(bank):
    all_index = np.arange(len(bank["metadata"]))
    folds = []
    for fold in range(3):
        pair = []
        fit_ids = bank["metadata"].event_id[all_index % 3 != fold]
        held_ids = bank["metadata"].event_id[all_index % 3 == fold]
        for role, selected in (("source_fit", all_index % 3 != fold), ("excluded_fold", all_index % 3 == fold)):
            cache = {key: np.asarray(bank[key])[selected].copy() for key in ("features", "bp", "base")}
            cache["metadata"] = bank["metadata"].loc[selected].reset_index(drop=True).copy()
            cache["metadata"]["oof_role"] = role
            cache["manifest"] = {**bank["manifest"], "fold": fold,
                                 "checkpoint_sha256": str(fold + 1) * 64,
                                 "encoder_protocol": "model-level-exclusion", "encoder_initialization": "scratch",
                                 "epoch_policy": "prespecified_smoke",
                                 "model_selection_excluded_fold_labels_used": False,
                                 "encoder_fit_event_ids_sha256": event_digest(fit_ids),
                                 "excluded_event_ids_sha256": event_digest(held_ids)}
            pair.append(cache)
        folds.append(pair)
    return folds


class OfficialCacheTests(unittest.TestCase):
    def test_digest_order_invariant(self):
        self.assertEqual(event_digest(["z", "a"]), hashlib.sha256(b"a\nz\n").hexdigest())

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Parquet engine required")
    def test_cache_hashes_and_test_roles(self):
        bank, _ = cache_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train"
            export_fixture(path, bank)
            loaded = load_role(path, "train", synthetic=True)
            np.testing.assert_array_equal(loaded["features"], bank["features"])
            with self.assertRaisesRegex(ValueError, "test/held-out"):
                load_role(path, "test", synthetic=True)
            np.save(path / "base_bp.npy", bank["base"] + 1)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_role(path, "train", synthetic=True)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Parquet engine required")
    def test_synthetic_cannot_be_real(self):
        bank, _ = cache_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train"
            export_fixture(path, bank)
            with self.assertRaisesRegex(ValueError, "synthetic"):
                load_role(path, "train")

    def test_seen_subjects_permitted_exact_reuse_not_permitted(self):
        bank, val = cache_fixture()
        audit, _, _ = audit_roles(bank, val)
        self.assertEqual(audit["cross_role_lineage"], "pass")
        val["metadata"].loc[0, "waveform_sha256"] = bank["metadata"].loc[0, "waveform_sha256"]
        with self.assertRaisesRegex(ValueError, "waveform_sha256 overlap"):
            audit_roles(bank, val)

    def test_global_physiological_overlap_is_caught(self):
        bank, val = cache_fixture()
        val["metadata"].loc[0, ["start_s", "end_s"]] = [5., 15.]
        with self.assertRaisesRegex(ValueError, "physiological interval"):
            audit_roles(bank, val)

    def test_official_counts_not_assumed(self):
        bank, val = cache_fixture()
        with self.assertRaisesRegex(ValueError, "2506"):
            audit_roles(bank, val, expected_counts=(2506, 320, 40))

    def test_e1_only_changes_training_donors(self):
        bank, val = cache_fixture()
        v1, _ = prepare_pair(bank, val)
        e1, _ = prepare_pair(bank, val, matched=True)
        self.assertTrue(np.any(v1["train"]["knn_indices"] != e1["train"]["knn_indices"]))
        np.testing.assert_array_equal(v1["train"]["support_weight"], e1["train"]["support_weight"])
        for key in v1["validation"]:
            np.testing.assert_array_equal(v1["validation"][key], e1["validation"][key])
        for qi, refs in enumerate(e1["train"]["knn_indices"]):
            self.assertNotIn(qi, refs)

    def test_trust_features_do_not_read_query_targets(self):
        bank, val = cache_fixture()
        neighbors, _ = prepare_pair(bank, val)
        expected = memory_inputs(bank, val, neighbors["validation"], training_std(bank))
        del val["bp"]
        val["metadata"]["target_sbp"] = float("nan")
        observed = memory_inputs(bank, val, neighbors["validation"], training_std(bank))
        self.assertEqual(observed["features"].shape[1], len(TRUST_FEATURE_NAMES))
        for key in expected:
            np.testing.assert_array_equal(expected[key], observed[key])

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Parquet engine required")
    def test_final_input_cache_rejects_target_file(self):
        _, query = cache_fixture()
        query["metadata"]["role"] = "test_inputs"
        query["manifest"].update(stage="final", labels_present=False, official_test_accessed=True,
                                  official_test_targets_accessed=False)
        del query["bp"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_inputs"
            export_fixture(path, query)
            loaded = load_role(path, "test_inputs", synthetic=True, allow_test_inputs=True)
            self.assertIsNone(loaded["bp"])
            np.save(path / "bp.npy", np.zeros((len(query["metadata"]), 2)))
            with self.assertRaisesRegex(ValueError, "must not contain BP"):
                load_role(path, "test_inputs", synthetic=True, allow_test_inputs=True)

    def test_insufficient_reference_fallback_coverage(self):
        bank, val = cache_fixture(n_train=4)
        neighbors, _ = prepare_pair(bank, val)
        observed = memory_inputs(bank, val, neighbors["validation"], training_std(bank))
        self.assertFalse(observed["legal"].any())
        np.testing.assert_array_equal(observed["memory"], val["base"])
        self.assertTrue(np.isfinite(observed["features"]).all())


class CrossfitContractTests(unittest.TestCase):
    def test_exact_three_encoder_folds(self):
        bank, _ = cache_fixture()
        folds = oof_fixtures(bank)
        evidence = audit_crossfit(folds, bank)
        self.assertEqual(sum(item["n_excluded"] for item in evidence), len(bank["bp"]))

    def test_missing_or_duplicate_folds_rejected(self):
        bank, _ = cache_fixture()
        folds = oof_fixtures(bank)
        with self.assertRaisesRegex(ValueError, "three"):
            audit_crossfit(folds[:2], bank)
        with self.assertRaisesRegex(ValueError, "more than one"):
            audit_crossfit([folds[0], folds[0], folds[2]], bank)

    def test_pretrained_full_model_is_not_oof(self):
        bank, _ = cache_fixture()
        folds = oof_fixtures(bank)
        folds[0][0]["manifest"]["encoder_initialization"] = "full_fit_checkpoint"
        with self.assertRaisesRegex(ValueError, "scratch"):
            audit_crossfit(folds, bank)

    def test_excluded_label_early_stop_is_rejected(self):
        bank, _ = cache_fixture()
        folds = oof_fixtures(bank)
        folds[0][1]["manifest"]["model_selection_excluded_fold_labels_used"] = True
        with self.assertRaisesRegex(ValueError, "excluded-label"):
            audit_crossfit(folds, bank)

    def test_false_encoder_fit_hash_is_rejected(self):
        bank, _ = cache_fixture()
        folds = oof_fixtures(bank)
        folds[0][0]["manifest"]["encoder_fit_event_ids_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "source-fit hash"):
            audit_crossfit(folds, bank)

    def test_data_selected_epoch_policy_rejected(self):
        bank, _ = cache_fixture()
        folds = oof_fixtures(bank)
        folds[0][0]["manifest"]["epoch_policy"] = "inherited_from_full_inner_fit_outer_validation_selection"
        with self.assertRaisesRegex(ValueError, "excluded-label"):
            audit_crossfit(folds, bank)

    def test_changed_label_for_same_id_is_rejected(self):
        bank, _ = cache_fixture()
        folds = oof_fixtures(bank)
        folds[0][1]["bp"][0, 0] += 10
        with self.assertRaises(AssertionError):
            audit_crossfit(folds, bank)

    def test_changed_lineage_for_same_id_is_rejected(self):
        bank, _ = cache_fixture()
        folds = oof_fixtures(bank)
        folds[0][1]["metadata"].loc[0, "waveform_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "lineage differs"):
            audit_crossfit(folds, bank)


if __name__ == "__main__":
    unittest.main()
