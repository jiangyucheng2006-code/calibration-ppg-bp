import unittest
import numpy as np

from pulsedb_fewshot.post_enrollment_ablations import memory_variant, paired_stratified_interval


class MemoryControlTests(unittest.TestCase):
    def setUp(self):
        self.bank = np.array([[100 + i * 10, 60 + i * 5] for i in range(5)], dtype=float)
        self.base = np.array([[140., 90.], [150., 100.]])
        self.state = {"knn_indices": np.array([[0, 1, 2, 3, 4], [-1] * 5]),
                      "knn_weights": np.array([[.4, .3, .15, .1, .05], [0.] * 5]),
                      "valid": np.array([True, False]), "support_weight": np.array([.8, 0.])}

    def test_default_matches_original_distance_softmax(self):
        memory = (self.bank * self.state["knn_weights"][0, :, None]).sum(0)
        expected = self.base.copy()
        expected[0] += .8 * (memory - self.base[0])
        np.testing.assert_allclose(memory_variant(self.bank, self.base, self.state), expected)

    def test_uniform_retains_donors_and_gate(self):
        out = memory_variant(self.bank, self.base, self.state, mode="distance_uniform")
        np.testing.assert_allclose(out[0], self.base[0] + .8 * (self.bank.mean(0) - self.base[0]))
        np.testing.assert_array_equal(out[1], self.base[1])

    def test_fixed_gate_and_memory_only(self):
        memory = (self.bank * self.state["knn_weights"][0, :, None]).sum(0)
        half = memory_variant(self.bank, self.base, self.state, mode="fixed_half")
        only = memory_variant(self.bank, self.base, self.state, mode="memory_only")
        np.testing.assert_allclose(half[0], .5 * (self.base[0] + memory))
        np.testing.assert_allclose(only[0], memory)
        np.testing.assert_array_equal(half[1], self.base[1])
        np.testing.assert_array_equal(only[1], self.base[1])

    def test_memory_only_does_not_depend_on_head_for_legal_donors(self):
        a = memory_variant(self.bank, self.base, self.state, mode="memory_only")
        b = memory_variant(self.bank, self.base + 123, self.state, mode="memory_only")
        np.testing.assert_allclose(a[0], b[0])

    def test_bad_donors_and_unknown_controls_fail(self):
        with self.assertRaisesRegex(ValueError, "unknown prespecified"):
            memory_variant(self.bank, self.base, self.state, mode="select_best_from_test")
        bad = dict(self.state, knn_indices=self.state["knn_indices"].copy())
        bad["knn_indices"][0, 0] = -1
        with self.assertRaisesRegex(ValueError, "donor indices"):
            memory_variant(self.bank, self.base, bad)


class PairedIntervalTests(unittest.TestCase):
    def test_constant_gains_have_exact_intervals(self):
        ci = paired_stratified_interval([2., 2., 2., 2.], ["MIMIC"] * 2 + ["VitalDB"] * 2)
        for name in ("gain_mmHg", "ci95_low_mmHg", "ci95_high_mmHg"):
            self.assertEqual(ci[name], 2.)
        self.assertEqual(ci["improved_participants"], 4)

    def test_strata_sizes_are_preserved(self):
        ci = paired_stratified_interval([1., 1., 10.], ["MIMIC", "MIMIC", "VitalDB"])
        self.assertEqual(ci["gain_mmHg"], 4.)
        self.assertEqual(ci["ci95_low_mmHg"], 4.)
        self.assertEqual(ci["ci95_high_mmHg"], 4.)

    def test_seed_is_fixed(self):
        with self.assertRaisesRegex(ValueError, "fixed before"):
            paired_stratified_interval([1., 2.], ["MIMIC", "MIMIC"], seed=5)


if __name__ == "__main__":
    unittest.main()
