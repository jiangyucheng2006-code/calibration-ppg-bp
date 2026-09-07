"""Aggregate-only v2 reporting contracts; no patient files or Torch required."""
import unittest

import pandas as pd

from pulsedb_fewshot.personal_memory_v2_report import comparisons, eligibility, CANDIDATES, DIAGNOSTICS, MODES
from pulsedb_fewshot.personal_memory_report import EXPECTED_COUNTS


def fixture(mode="random_disjoint"):
    rows = []
    for candidate in sorted(set(CANDIDATES) | set(DIAGNOSTICS) | {"lora_continued_control"}):
        for scope, (people, events) in EXPECTED_COUNTS.items():
            score = 3.0 if candidate == "lora_continued_control" else 3.1 if candidate == "D0_frozen_lora" else 2.8
            rows.append(dict(candidate=candidate, view=scope, split_mode=mode,
                             n_participants=people, n_events=events, mean_mae=score))
    return pd.DataFrame(rows)


class ReportTests(unittest.TestCase):
    def test_same_reference_for_all_sources(self):
        frame = fixture()
        frame.loc[frame.candidate.eq("D0_frozen_lora") & frame["view"].eq("MIMIC"), "mean_mae"] = 1.0
        table, reference = comparisons(frame)
        self.assertEqual(reference, "lora_continued_control")
        self.assertTrue(table.reference_mean_mae.eq(3.0).all())

    def test_missing_candidate(self):
        with self.assertRaises(ValueError):
            comparisons(fixture().iloc[:-3])

    def test_coverage_rejection(self):
        frame = fixture()
        frame.loc[0, "n_events"] -= 1
        with self.assertRaises(ValueError):
            comparisons(frame)

    def test_both_modes_gate_and_module_gain_distinct(self):
        frame = pd.concat([comparisons(fixture(mode))[0] for mode in MODES])
        decisions = eligibility(frame)
        self.assertTrue(all(r["passes_historical_LoRA_upgrade_gate"] for r in decisions))
        self.assertFalse(any(r["beats_fixed_blend_in_both_modes"] for r in decisions))

    def test_chronological_small_gain_fails(self):
        frame = pd.concat([comparisons(fixture(mode))[0] for mode in MODES], ignore_index=True)
        frame.loc[frame.split_mode.eq(MODES[1]), "gain_vs_reference_mmhg"] = .05
        self.assertFalse(any(r["passes_historical_LoRA_upgrade_gate"] for r in eligibility(frame)))

    def test_one_source_regression_fails(self):
        frame = pd.concat([comparisons(fixture(mode))[0] for mode in MODES], ignore_index=True)
        frame.loc[frame["view"].eq("VitalDB"), "gain_vs_reference_mmhg"] = -.01
        self.assertFalse(any(r["passes_historical_LoRA_upgrade_gate"] for r in eligibility(frame)))

    def test_single_mode_invalid(self):
        with self.assertRaises(ValueError):
            eligibility(comparisons(fixture())[0])

    def test_nonfinite_rejected(self):
        frame = fixture()
        frame.loc[0, "mean_mae"] = float("nan")
        with self.assertRaises(ValueError):
            comparisons(frame)


if __name__ == "__main__":
    unittest.main()
