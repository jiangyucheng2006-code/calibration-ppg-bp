"""Pure NumPy tests for frozen diagnostics: no local CUDA/Torch dependency."""
from copy import deepcopy
import hashlib
import inspect
import unittest

import numpy as np

from pulsedb_fewshot.personal_memory_diagnostics import (
    _validate_neighbors, assert_reproduction, fit_temporal_terciles, fixed_reliability,
    frozen_distance_alpha, reference_time_gaps, time_matched_references, uniform_bp_prediction,
)
from pulsedb_fewshot.personal_memory_prepare import prepare_neighbors


def row(name, start, role="train", person="MIMIC:a", record="MIMIC:r"):
    return dict(subject_uid=person, source=person.split(":")[0], event_id=name, window_uid=name,
                waveform_sha256=hashlib.sha256(name.encode()).hexdigest(), recording_uid=record,
                time_axis_uid=record, start_s=float(start), end_s=float(start+10), role=role)


def fixture():
    train = [row(f"t{i:03}", 20*i) for i in range(80)]
    val = [row("v0", 1610, "internal_validation"), row("v1", 1650, "internal_validation")]
    rng = np.random.default_rng(31)
    result = prepare_neighbors(rng.normal(size=(80, 8)), rng.normal(size=(2, 8)), train, val,
                               mode="chronological_blocked")
    return train, val, result


class FrozenDiagnosticsTest(unittest.TestCase):
    def reliability(self, weights=None, local=None, legal=None):
        return fixed_reliability(np.array([[110., 70.]]),
            np.tile([120., 80.], (1, 5, 1)) if local is None else local,
            np.full((1, 5), .2) if weights is None else weights,
            np.ones((1, 5), bool) if legal is None else legal, np.array([.8]), np.array([20., 10.]))

    def test_equal_references_preserve_fixed_alpha(self):
        prediction, extra = self.reliability()
        np.testing.assert_allclose(prediction, [[118, 78]])
        np.testing.assert_allclose(extra["alpha"], .8)
        np.testing.assert_allclose(extra["ess"], 5)
        np.testing.assert_allclose(extra["scatter"], 0)

    def test_concentrated_reference_reduces_alpha(self):
        _, extra = self.reliability(weights=np.array([[1, 0, 0, 0, 0]]))
        np.testing.assert_allclose(extra["alpha"], .16)
        np.testing.assert_allclose(extra["ess"], 1)

    def test_scatter_reduces_alpha_and_uses_reference_specific_delta(self):
        local = np.tile([120., 80.], (1, 5, 1))
        local[0, 0] += [40, 20]
        _, extra = self.reliability(local=local)
        self.assertGreater(extra["scatter"][0], 0)
        self.assertLess(extra["alpha"][0], .8)

    def test_zero_reference_fallback_exact(self):
        pred, extra = self.reliability(legal=np.zeros((1, 5), bool))
        np.testing.assert_array_equal(pred, [[110., 70.]])
        np.testing.assert_array_equal(extra["alpha"], [0])

    def test_unit_conversion_does_not_change_alpha(self):
        base = np.array([[110., 70.]])
        local = np.arange(10).reshape(1, 5, 2)+110.
        args = (np.full((1, 5), .2), np.ones((1, 5), bool), np.array([.8]))
        p, x = fixed_reliability(base, local, *args, np.array([20., 10.]))
        q, y = fixed_reliability(base*.133322, local*.133322, *args, np.array([20., 10.])*.133322)
        np.testing.assert_allclose(x["alpha"], y["alpha"])
        np.testing.assert_allclose(p*.133322, q)

    def test_no_query_target_argument(self):
        self.assertNotIn("query_bp", inspect.signature(fixed_reliability).parameters)
        self.assertNotIn("target", inspect.signature(time_matched_references).parameters)

    def test_frozen_gate_scale(self):
        np.testing.assert_allclose(frozen_distance_alpha([0, .1, .2, np.inf], [1, 1, 1, 0], .2),
                                   [1, .5, 0, 0])
        np.testing.assert_array_equal(frozen_distance_alpha([0, 1], [1, 1], None), [0, 0])

    def test_matching_preserves_strata_and_legality(self):
        train, val, n = fixture()
        fit = fit_temporal_terciles(train, n["train"]["knn_indices"])
        matched = time_matched_references(train, val, n["validation"]["knn_indices"], fit, gap_s=0, past_only=True)
        idx = matched["random_indices"]
        self.assertTrue((idx >= 0).all())
        self.assertTrue(all(len(set(x)) == 5 for x in idx))
        gap = reference_time_gaps(train, val, idx)
        strata = np.searchsorted(fit["cuts_s"], gap, side="right")
        np.testing.assert_array_equal(strata, matched["strata"])
        for q, selection in zip(val, idx):
            self.assertTrue(all(train[i]["end_s"] <= q["start_s"] for i in selection))

    def test_matching_deterministic_and_scoring_metadata_ignored(self):
        train, val, n = fixture()
        fit = fit_temporal_terciles(train, n["train"]["knn_indices"])
        a = time_matched_references(train, val, n["validation"]["knn_indices"], fit, gap_s=0, past_only=True)
        changed = deepcopy(val)
        for r in changed:
            r["target_sbp"] = -100000.
        b = time_matched_references(train, changed, n["validation"]["knn_indices"], fit, gap_s=0, past_only=True)
        np.testing.assert_array_equal(a["random_indices"], b["random_indices"])
        reverse = time_matched_references(train, val[::-1], n["validation"]["knn_indices"][::-1], fit,
                                         gap_s=0, past_only=True)
        np.testing.assert_array_equal(a["random_indices"], reverse["random_indices"][::-1])

    def test_cross_record_no_guessed_clock_and_matched_fallback(self):
        train, val, n = fixture()
        changed = deepcopy(val)
        changed[0]["recording_uid"] = changed[0]["time_axis_uid"] = "MIMIC:other"
        fit = fit_temporal_terciles(train, n["train"]["knn_indices"])
        matched = time_matched_references(train, changed, n["validation"]["knn_indices"], fit, gap_s=0, past_only=True)
        self.assertTrue((matched["random_indices"][0] == -1).all())
        self.assertTrue((matched["feature_indices"][0] == -1).all())
        self.assertEqual(matched["fallback_reasons"]["feature_references_cross_record_clock"], 1)

    def test_insufficient_history_keeps_all_queries(self):
        train, val, n = fixture()
        idx = n["validation"]["knn_indices"].copy()
        idx[0] = -1
        fit = fit_temporal_terciles(train, n["train"]["knn_indices"])
        matched = time_matched_references(train, val, idx, fit, gap_s=0, past_only=True)
        base = np.array([[110., 70.], [120., 80.]])
        pred = uniform_bp_prediction(np.tile([125., 85.], (80, 1)), base, matched["random_indices"])
        self.assertEqual(len(pred), len(val))
        np.testing.assert_array_equal(pred[0], base[0])

    def test_loaded_neighbors_independent_clock_audit(self):
        train, val, n = fixture()
        _validate_neighbors(n, train, val, gap_s=0, past_only=True)
        bad = deepcopy(n)
        bad["validation"]["knn_indices"][0, 0] = -1
        with self.assertRaisesRegex(ValueError, "fallback"):
            _validate_neighbors(bad, train, val, gap_s=0, past_only=True)
        with self.assertRaisesRegex(ValueError, "exclusion band|past only"):
            _validate_neighbors(n, train, val, gap_s=10000, past_only=True)

    def test_exact_full_replay_tolerance(self):
        x = np.array([[120., 80.], [130., 90.]])
        self.assertEqual(assert_reproduction(x, x+1e-5)["status"], "pass")
        with self.assertRaisesRegex(ValueError, "reproduction failed"):
            assert_reproduction(x, x+1e-3)


if __name__ == "__main__":
    unittest.main()
