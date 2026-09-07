"""Pure NumPy/stdlib tests for the real-cache personal-memory preparation rules."""
from copy import deepcopy
import hashlib
import inspect
import unittest

import numpy as np

from pulsedb_fewshot.personal_memory_prepare import (
    audit_metadata, diagnostic_predictions, macro_metrics, prepare_neighbors,
    score_probe, stage_evidence, validate_export_manifest,
)


def row(uid, start, *, subject="MIMIC:a", role="train", record="MIMIC:r", axis=None):
    return {"subject_uid": subject, "event_id": uid, "source": subject.split(":")[0],
            "window_uid": uid, "waveform_sha256": hashlib.sha256(uid.encode()).hexdigest(),
            "recording_uid": record, "time_axis_uid": axis or record,
            "start_s": float(start), "end_s": float(start + 10), "role": role}


def fixture(n=80):
    train = [row(f"t{i:03}", 20*i) for i in range(n)]
    val = [row("v0", 20*n + 50, role="internal_validation"),
           row("v1", 20*n + 70, role="internal_validation")]
    rng = np.random.default_rng(31)
    return rng.normal(size=(n, 8)) + .1, rng.normal(size=(2, 8)) + .1, train, val


class PrepareTest(unittest.TestCase):
    def test_export_manifest_provenance(self):
        manifest = {"status": "exported", "protocol_id": "development-calbased-analogue-v1",
                    "split_mode": "random_disjoint", "heldout_test_accessed": False,
                    "source_parent_split": "meta_train", "read_roles": ["train", "internal_validation"],
                    "source_run": "immutable-run", "source_checkpoint_sha256": "a"*64,
                    "source_tree_sha256": "b"*64, "target_scaler": {"mean": [120, 80], "std": [20, 10]}}
        validate_export_manifest(manifest)
        for key, value in (("status", "complete"), ("source_parent_split", "meta_test"),
                           ("heldout_test_accessed", True), ("read_roles", ["train", "heldout_test"]),
                           ("target_scaler", {"mean": [120, 80], "std": [0, 10]}),
                           ("source_checkpoint_sha256", "unknown"), ("source_run", "")):
            with self.subTest(key=key):
                altered = deepcopy(manifest)
                altered[key] = value
                with self.assertRaises(ValueError):
                    validate_export_manifest(altered)

    def result(self, *, mode="random_disjoint", **kwargs):
        return prepare_neighbors(*fixture(), mode=mode, **kwargs)

    def test_complete_train_block_removed(self):
        result = self.result()
        self.assertTrue(result["train"]["valid"].all())
        self.assertTrue((result["train"]["knn_indices"][:40] >= 40).all())
        self.assertTrue((result["train"]["knn_indices"][40:] < 40).all())

    def test_chronological_training_first_block_fallback(self):
        result = self.result(mode="chronological_blocked")
        self.assertFalse(result["train"]["valid"][:40].any())
        self.assertTrue(result["train"]["valid"][40:].all())
        self.assertTrue((result["train"]["knn_indices"][40:] < 40).all())
        self.assertTrue(result["validation"]["valid"].all())

    def test_never_returns_self(self):
        result = self.result()
        self.assertFalse((result["train"]["knn_indices"] == np.arange(80)[:, None]).any())
        self.assertFalse((result["train"]["uniform_indices"] == np.arange(80)[:, None]).any())

    def test_small_history_all_minus_one_and_zero_weight(self):
        result = prepare_neighbors(*fixture(40), mode="random_disjoint")
        self.assertTrue((result["train"]["knn_indices"] == -1).all())
        self.assertTrue((result["train"]["knn_weights"] == 0).all())
        self.assertTrue((result["train"]["support_weight"] == 0).all())
        self.assertIsNone(result["train_distance_cutpoints"]["q95"])

    def test_deterministic(self):
        first, second = self.result(), self.result()
        for role in ("train", "validation"):
            for name in first[role]:
                np.testing.assert_array_equal(first[role][name], second[role][name])

    def test_softmax_weights(self):
        result = self.result()
        for role in ("train", "validation"):
            np.testing.assert_allclose(result[role]["knn_weights"].sum(1), 1, atol=1e-6)
            self.assertTrue((result[role]["knn_weights"] > 0).all())

    def test_uniform_five_spaced_donors(self):
        result = self.result()
        np.testing.assert_array_equal(result["train"]["uniform_indices"][0], [40, 49, 59, 69, 79])

    def test_hash_duplicate_across_mislabelled_subject_detected(self):
        _, _, train, val = fixture()
        val[0]["subject_uid"] = "MIMIC:wrong"
        val[0]["waveform_sha256"] = train[0]["waveform_sha256"]
        with self.assertRaisesRegex(ValueError, "global cross-role waveform"):
            audit_metadata(train, val)

    def test_global_id_duplicate(self):
        _, _, train, val = fixture()
        val[0]["window_uid"] = train[0]["window_uid"]
        with self.assertRaisesRegex(ValueError, "global cross-role window_uid"):
            audit_metadata(train, val)

    def test_global_event_duplicate(self):
        _, _, train, val = fixture()
        val[0]["event_id"] = train[0]["event_id"]
        with self.assertRaisesRegex(ValueError, "global cross-role event_id"):
            audit_metadata(train, val)

    def test_cross_subject_physical_interval_detected(self):
        _, _, train, val = fixture()
        val[0].update(subject_uid="MIMIC:wrong", start_s=1., end_s=11.)
        with self.assertRaisesRegex(ValueError, "global cross-role physiological"):
            audit_metadata(train, val)

    def test_half_open_adjacent_allowed(self):
        _, _, train, val = fixture()
        val[0].update(start_s=10., end_s=20.)
        self.assertEqual(audit_metadata(train, val)["cross_role_lineage"], "pass")

    def test_wrong_role_forbidden(self):
        _, _, train, val = fixture()
        val[0]["role"] = "heldout_test"
        with self.assertRaisesRegex(ValueError, "forbidden"):
            audit_metadata(train, val)

    def test_record_clock_mismatch_fails(self):
        _, _, train, val = fixture()
        val[0]["time_axis_uid"] = "another-clock"
        with self.assertRaisesRegex(ValueError, "inconsistent canonical clocks"):
            audit_metadata(train, val)

    def test_different_record_chrono_fallback_not_guessed(self):
        x, v, train, val = fixture()
        for r in val:
            r.update(recording_uid="MIMIC:new", time_axis_uid="MIMIC:new")
        result = prepare_neighbors(x, v, train, val, mode="chronological_blocked")
        self.assertFalse(result["validation"]["valid"].any())
        self.assertTrue((result["validation"]["knn_indices"] == -1).all())
        self.assertTrue(prepare_neighbors(x, v, train, val, mode="random_disjoint")["validation"]["valid"].all())

    def test_no_neighbors_from_another_person(self):
        x, v, train, val = fixture()
        for i in range(40, 80):
            train[i].update(subject_uid="VitalDB:b", source="VitalDB", recording_uid="VitalDB:r", time_axis_uid="VitalDB:r")
        result = prepare_neighbors(x, v, train, val, mode="random_disjoint")
        self.assertTrue((result["validation"]["knn_indices"] < 40).all())

    def test_60_second_band_respected(self):
        x, v, train, val = fixture()
        val[0].update(start_s=1590., end_s=1600.)
        result = prepare_neighbors(x, v, train, val, mode="chronological_blocked", exclusion_s=60)
        chosen = result["validation"]["knn_indices"][0]
        self.assertTrue(all(train[i]["end_s"] <= 1530 for i in chosen))

    def test_train_time_control_excludes_training_block(self):
        result = self.result()
        self.assertEqual(result["train"]["nearest_time_index"][39], 40)
        self.assertEqual(result["train"]["nearest_time_index"][40], 39)

    def test_validation_features_cannot_change_train_gate(self):
        x, v, train, val = fixture()
        one = prepare_neighbors(x, v, train, val, mode="random_disjoint")
        two = prepare_neighbors(x, -v * 40, train, val, mode="random_disjoint")
        self.assertEqual(one["train_distance_cutpoints"], two["train_distance_cutpoints"])
        np.testing.assert_array_equal(one["train"]["support_weight"], two["train"]["support_weight"])

    def test_metadata_target_columns_do_not_influence_retrieval(self):
        x, v, train, val = fixture()
        first = prepare_neighbors(x, v, train, val, mode="random_disjoint")
        for i, item in enumerate(train + val):
            item.update(target_sbp=1000. + i, target_dbp=-1000. - i)
        second = prepare_neighbors(x, v, train, val, mode="random_disjoint")
        for role in ("train", "validation"):
            for name in first[role]:
                np.testing.assert_array_equal(first[role][name], second[role][name])
        self.assertEqual(first["train_distance_cutpoints"], second["train_distance_cutpoints"])

    def test_zero_distance_gate_is_finite(self):
        x, v, train, val = fixture()
        x[:] = 1
        v[:] = 1
        result = prepare_neighbors(x, v, train, val, mode="random_disjoint")
        self.assertTrue(np.isfinite(result["validation"]["support_weight"]).all())
        self.assertTrue((result["validation"]["support_weight"] == 1).all())

    def test_nonfinite_features_rejected(self):
        x, v, train, val = fixture()
        x[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            prepare_neighbors(x, v, train, val, mode="random_disjoint")

    def test_prediction_api_has_no_query_target(self):
        self.assertEqual(list(inspect.signature(diagnostic_predictions).parameters),
                         ["train_bp", "train_base", "validation_base", "neighbors"])

    def test_residual_and_bp_controls_exact(self):
        result = self.result()
        train_bp = np.tile([130., 80.], (80, 1))
        train_base = train_bp - [2., 1.]
        base = np.tile([125., 75.], (2, 1))
        pred = diagnostic_predictions(train_bp, train_base, base, result["validation"])
        np.testing.assert_allclose(pred["D0"], base)
        np.testing.assert_allclose(pred["D1"], train_bp[:2], atol=1e-5)
        np.testing.assert_allclose(pred["D2"], base + [2., 1.], atol=1e-5)
        np.testing.assert_allclose(pred["D3"], train_bp[:2])

    def test_fallback_unchanged_for_every_method(self):
        result = self.result(mode="chronological_blocked")
        subset = {key: value[:5] for key, value in result["train"].items()}
        base = np.tile([125., 75.], (5, 1))
        pred = diagnostic_predictions(np.ones((80, 2)), np.zeros((80, 2)), base, subset)
        for value in pred.values():
            np.testing.assert_array_equal(value, base)

    def test_macro_not_pooled(self):
        target = np.zeros((4, 2))
        pred = np.array([[10., 10.], [0., 0.], [0., 0.], [0., 0.]])
        metric = macro_metrics(target, pred, ["a", "b", "b", "b"])
        self.assertEqual(metric["mean_mae"], 5.)
        self.assertEqual(metric["n_participants"], 2)

    def test_scoring_keeps_full_coverage(self):
        x, v, train, val = fixture()
        result = prepare_neighbors(x, v, train, val, mode="random_disjoint")
        train_bp = np.tile([130., 80.], (80, 1))
        pred = diagnostic_predictions(train_bp, train_bp, np.tile([129., 79.], (2, 1)), result["validation"])
        overall, groups = score_probe(np.tile([130., 80.], (2, 1)), pred, val, train_bp, train,
                                      result["validation"], result["train_distance_cutpoints"])
        self.assertEqual(overall["D0"]["Overall"]["n_events"], 2)
        self.assertAlmostEqual(overall["D1"]["Overall"]["mean_mae"], 0., places=10)
        self.assertTrue(stage_evidence(overall, groups)["passes_this_mode"])

    def test_stage_group_size_and_method_gate(self):
        metric = {"Overall": {"mean_mae": 3.}}
        overall = {name: deepcopy(metric) for name in ("D0", "D1", "D2", "D3")}
        group = {"method": "D2", "scope": "MIMIC", "group_type": "bp_deviation", "n_participants": 99,
                 "n_events": 999, "gain_mmhg": .5}
        self.assertFalse(stage_evidence(overall, [group])["passes_this_mode"])
        group["n_participants"] = 100
        self.assertTrue(stage_evidence(overall, [group])["passes_this_mode"])
        group["method"] = "D3"
        self.assertFalse(stage_evidence(overall, [group])["passes_this_mode"])

    def test_stage_overall_threshold(self):
        overall = {name: {"Overall": {"mean_mae": 3.}} for name in ("D0", "D1", "D2", "D3")}
        overall["D1"]["Overall"]["mean_mae"] = 2.981
        self.assertFalse(stage_evidence(overall, [])["passes_this_mode"])
        overall["D1"]["Overall"]["mean_mae"] = 2.979
        self.assertTrue(stage_evidence(overall, [])["passes_this_mode"])


if __name__ == "__main__":
    unittest.main()
