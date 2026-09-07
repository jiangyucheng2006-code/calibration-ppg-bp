"""Small aggregate-only report regressions; no checkpoints or held-out data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from pulsedb_fewshot.personal_memory_report import (
    CONTROL, EXPECTED_COUNTS, FROZEN, MODES, NEURAL, SCREEN_ID, combined_report,
)


class CombinedReportTests(unittest.TestCase):
    def fixture(self, root, *, gain=0.05, skipped=False):
        directories = []
        for mode in MODES:
            directory = root / mode
            directory.mkdir()
            reference = FROZEN if skipped or mode == MODES[1] else CONTROL
            selection = {
                "status": "skipped_no_complementarity" if skipped else "complete",
                "screen_id": SCREEN_ID, "split_mode": mode,
                "heldout_test_accessed": False,
                "official_pulsedb_calbased_reproduction": False,
                "reference": reference, "gate_manifest_sha256": "a" * 64,
            }
            (directory / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
            rows = []
            for candidate in (FROZEN,) if skipped else (FROZEN, CONTROL, *NEURAL):
                candidate_gain = gain if candidate == "pair_distance_blend" else 0.0
                for scope, (people, events) in EXPECTED_COUNTS.items():
                    baseline = {"Overall": 4.0, "MIMIC": 3.8, "VitalDB": 4.2}[scope]
                    rows.append({
                        "candidate": candidate, "view": scope, "split_mode": mode,
                        "seed": 20260907, "n_participants": people, "n_events": events,
                        "mean_mae": baseline - candidate_gain,
                        "reference_mean_mae": baseline, "reference_candidate": reference,
                        "gain_mmhg": candidate_gain,
                    })
            pd.DataFrame(rows).to_csv(directory / "comparison_vs_reference.csv", index=False)
            directories.append(directory)
        return directories

    def change_frame(self, directory, operation):
        path = directory / "comparison_vs_reference.csv"
        frame = pd.read_csv(path)
        operation(frame)
        frame.to_csv(path, index=False)

    def test_complete_report_with_native_pandas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = combined_report(self.fixture(root), root / "output")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["eligible_candidates"], [])
            self.assertEqual(result["references"], {MODES[0]: CONTROL, MODES[1]: FROZEN})
            self.assertFalse(result["heldout_test_accessed"])
            combined = pd.read_csv(root / "output/cross_split_comparison.csv")
            self.assertEqual(len(combined), 36)
            self.assertEqual(set(combined["view"]), set(EXPECTED_COUNTS))
            self.assertEqual(len(pd.read_csv(root / "output/promotion_gate.csv")), 4)
            with self.assertRaises(FileExistsError):
                combined_report([root / mode for mode in MODES], root / "output")

    def test_series_view_method_collision_compatible_with_pandas_2(self):
        # pandas 2.x exposes Series.view; pandas 3 removed it. Recreate that
        # attribute on the installed pandas without replacing DataFrame/Series.
        def legacy_view(series, *args, **kwargs):
            raise AssertionError("A report column must never invoke Series.view")

        with tempfile.TemporaryDirectory() as tmp, patch.object(pd.Series, "view", legacy_view, create=True):
            root = Path(tmp)
            row = pd.Series({"view": "Overall"})
            self.assertTrue(callable(row.view))
            self.assertEqual(row["view"], "Overall")
            result = combined_report(self.fixture(root, gain=0.2), root / "output")
            self.assertEqual(result["eligible_candidates"], ["pair_distance_blend"])

    def test_positive_overall_gain_does_not_override_source_decline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = self.fixture(root, gain=0.2)

            def decline(frame):
                mask = frame.candidate.eq("pair_distance_blend") & frame["view"].eq("MIMIC")
                frame.loc[mask, "gain_mmhg"] = -0.01
                frame.loc[mask, "mean_mae"] = frame.loc[mask, "reference_mean_mae"] + 0.01

            self.change_frame(directories[1], decline)
            result = combined_report(directories, root / "output")
            self.assertEqual(result["eligible_candidates"], [])

    def test_complete_scientific_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = combined_report(self.fixture(root, skipped=True), root / "output")
            self.assertEqual(result["status"], "skipped_no_complementarity")
            self.assertEqual(result["eligible_candidates"], [])
            self.assertTrue(pd.read_csv(root / "output/promotion_gate.csv").empty)

    def test_changed_cohort_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = self.fixture(root)
            self.change_frame(directories[0], lambda frame: frame.__setitem__("n_events", 1))
            with self.assertRaisesRegex(ValueError, "source cohort changed"):
                combined_report(directories, root / "output")
            self.assertFalse((root / "output").exists())

    def test_modified_gain_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = self.fixture(root)
            self.change_frame(directories[0], lambda frame: frame.__setitem__("gain_mmhg", 99.0))
            with self.assertRaises(AssertionError):
                combined_report(directories, root / "output")

    def test_two_distinct_modes_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = self.fixture(root)
            with self.assertRaises(ValueError):
                combined_report(directories[:1], root / "output")
            with self.assertRaises(ValueError):
                combined_report([directories[0], directories[0]], root / "output")

    def test_global_gate_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = self.fixture(root)
            path = directories[1] / "selection.json"
            selection = json.loads(path.read_text(encoding="utf-8"))
            selection["gate_manifest_sha256"] = "b" * 64
            path.write_text(json.dumps(selection), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different global scientific gates"):
                combined_report(directories, root / "output")


if __name__ == "__main__":
    unittest.main()
