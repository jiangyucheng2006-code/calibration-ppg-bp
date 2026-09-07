import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "export imports require server PyTorch")
class ReferencePrecisionTests(unittest.TestCase):
    def test_source_float32_representation(self):
        from pulsedb_fewshot.personal_memory_export import check_reference_targets
        raw = np.array([[180.000007, 80.123456789]], dtype=np.float64)
        saved = raw.astype(np.float32)
        saved[0, 0] = np.nextafter(saved[0, 0], np.float32(-np.inf))
        self.assertGreater(abs(float(saved[0, 0]) - raw[0, 0]), 2e-5)
        result = check_reference_targets(saved, raw)
        self.assertLessEqual(result["max_abs_mmhg"], 2e-5)

    def test_material_target_change_rejected(self):
        from pulsedb_fewshot.personal_memory_export import check_reference_targets
        with self.assertRaises(ValueError):
            check_reference_targets([[120.01, 80]], [[120, 80]])
        with self.assertRaises(ValueError):
            check_reference_targets([[np.nan, 80]], [[120, 80]])


if __name__ == "__main__":
    unittest.main()
