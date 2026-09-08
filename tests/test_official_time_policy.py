"""Exact sample-span boundary regressions, independent of Torch/HDF5."""
import unittest

import pandas as pd

from pulsedb_fewshot.official_time_policy import sample_span_audit, TIME_BOUNDARY_POLICY


def pair(start=16390.008, end=16400.0):
    return pd.DataFrame([
        dict(source="MIMIC", subject_uid="p1", record_id="record", segment_uid="a",
             start_time_s=16380.016, end_time_s=16390.008, official_role="official_train"),
        dict(source="MIMIC", subject_uid="p1", record_id="record", segment_uid="b",
             start_time_s=start, end_time_s=end, official_role="official_test"),
    ])


class OfficialTimePolicyTests(unittest.TestCase):
    def test_verified_four_boundary_pairs_have_touches_not_positive_overlap(self):
        frames = []
        for source, person, record in (
            ("MIMIC", "p056440", "2167-09-02-15-40"),
            ("VitalDB", "p001612", "c5746"),
            ("VitalDB", "p002463", "c5347"),
            ("VitalDB", "p005002", "c5166"),
        ):
            frames.append(pair().assign(source=source, subject_uid=person, record_id=record))
        result = sample_span_audit(pd.concat(frames), "official_role")
        self.assertEqual(result["time_boundary_policy"], TIME_BOUNDARY_POLICY)
        self.assertEqual(result["cross_role_overlap_pairs"], 0)
        self.assertEqual(result["cross_role_touching_pairs"], 4)

    def test_positive_overlap_even_below_one_sample_is_not_waived(self):
        for overlap in (.008, .001, .000001, 9.):
            result = sample_span_audit(pair(start=16390.008 - overlap), "official_role")
            self.assertEqual(result["cross_role_overlap_pairs"], 1)
            self.assertEqual(result["cross_role_touching_pairs"], 0)

    def test_numerical_roundoff_only_is_reported_as_touch(self):
        result = sample_span_audit(pair(start=16390.008 - 1e-8), "official_role")
        self.assertEqual(result["cross_role_overlap_pairs"], 0)
        self.assertEqual(result["cross_role_touching_pairs"], 1)

    def test_person_label_cannot_hide_same_recording_overlap(self):
        frame = pair(start=16389.)
        frame.loc[1, "subject_uid"] = "mistaken_person"
        self.assertEqual(sample_span_audit(frame, "official_role")["cross_role_overlap_pairs"], 1)

    def test_separate_recordings_are_separate_axes(self):
        frame = pair(start=16389.)
        frame.loc[1, "record_id"] = "other_recording"
        self.assertEqual(sample_span_audit(frame, "official_role")["cross_role_overlap_pairs"], 0)

    def test_invalid_span_rejected(self):
        for start, end in ((1., 1.), (2., 1.), (float("nan"), 1.), (1., float("inf"))):
            with self.assertRaises(ValueError):
                sample_span_audit(pair(start, end), "official_role")


if __name__ == "__main__":
    unittest.main()
