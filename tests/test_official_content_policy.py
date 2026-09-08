"""The official exception permits only known copies, never arbitrary leakage."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pulsedb_fewshot import official_content_policy as policy
from pulsedb_fewshot import official_calbased_train as train
from pulsedb_fewshot.official_memory_train import canonical_rows
from pulsedb_fewshot.personal_memory_prepare import audit_metadata
from test_official_calbased_train import fixture_frame


class OfficialContentPolicyTests(unittest.TestCase):
    def frame(self):
        frame = fixture_frame()
        frame.loc[9, "ppg_content_sha256"] = frame.loc[0, "ppg_content_sha256"]
        frame.loc[2, "ppg_content_sha256"] = frame.loc[1, "ppg_content_sha256"]
        frame["official_content_policy"] = policy.POLICY
        frame["official_duplicate_audit_path"] = "/private/verified_audit.json"
        groups = {frame.loc[0, "ppg_content_sha256"]: frozenset(frame.loc[[0, 9], "segment_uid"]),
                  frame.loc[1, "ppg_content_sha256"]: frozenset(frame.loc[[1, 2], "segment_uid"])}
        return frame, groups

    def test_authorized_pairs_pass_neural_and_memory_consumers_without_dropping(self):
        frame, groups = self.frame()
        with patch.object(policy, "approved_groups", return_value=groups):
            accepted = train.validate_training_frame(frame, smoke=True)
            pd.testing.assert_frame_equal(accepted[frame.columns], frame)
            for stage, fold in (("inner", None), ("oof", 1)):
                bank, exports = train.select_stage(accepted, stage, fold)
                query = exports["validation" if stage == "inner" else "excluded_fold"]
                b = canonical_rows(train.canonical_metadata(bank, "train"), "train")
                q = canonical_rows(train.canonical_metadata(query, "validation"), "internal_validation")
                report = policy.audit_official_metadata(b, q)
                self.assertGreater(report["retained_content_duplicate_groups"], 0)
                self.assertEqual(report["content_policy"], policy.POLICY)
                with self.assertRaisesRegex(ValueError, "duplicate|overlap"):
                    audit_metadata(b, q)  # Old strict default is unchanged.

    def test_unknown_copy_is_not_covered_by_a_known_hash(self):
        frame, groups = self.frame()
        frame.loc[10, "ppg_content_sha256"] = frame.loc[0, "ppg_content_sha256"]
        with patch.object(policy, "approved_groups", return_value=groups):
            with self.assertRaisesRegex(ValueError, "not the verified"):
                train.validate_training_frame(frame, smoke=True)

    def test_reused_identity_still_fails(self):
        frame, groups = self.frame()
        frame.loc[9, "segment_uid"] = frame.loc[0, "segment_uid"]
        with patch.object(policy, "approved_groups", return_value=groups):
            with self.assertRaisesRegex(ValueError, "duplicate official segment"):
                train.validate_training_frame(frame, smoke=True)

    def test_positive_interval_overlap_still_fails(self):
        frame, groups = self.frame()
        frame.loc[9, ["start_time_s", "end_time_s"]] = [5., 14.992]
        with patch.object(policy, "approved_groups", return_value=groups):
            with self.assertRaisesRegex(ValueError, "physiological interval"):
                train.validate_training_frame(frame, smoke=True)

    def test_marker_required_and_consistent(self):
        frame, groups = self.frame()
        with patch.object(policy, "approved_groups", return_value=groups):
            with self.assertRaisesRegex(ValueError, "content crosses"):
                train.validate_training_frame(frame.drop(columns=list(policy.POLICY_COLUMNS)), smoke=True)
            frame.loc[9, "official_content_policy"] = "unknown"
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                train.validate_training_frame(frame, smoke=True)

    def test_unpinned_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.json"
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                policy.approved_groups(str(path))

    def test_canonical_export_retains_explicit_policy(self):
        frame, _ = self.frame()
        result = train.canonical_metadata(frame, "source_fit")
        self.assertTrue(result.official_content_policy.eq(policy.POLICY).all())
        self.assertTrue(result.oof_role.eq("source_fit").all())
        self.assertFalse({"sbp", "dbp"} & set(result))

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py needed for recovery module")
    def test_recovered_shard_is_rehashed_and_row_bijection_checked(self):
        from pulsedb_fewshot.official_calbased_recover import check_array
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.npy"
            values = np.arange(2500, dtype="<f4").reshape(2, 1250)
            np.save(path, values)
            frame = pd.DataFrame({"waveform_row": [0, 1], "ppg_content_sha256":
                                  [hashlib.sha256(v.tobytes()).hexdigest() for v in values]})
            expected = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": 2}
            check_array(path, frame, expected)
            frame.loc[1, "waveform_row"] = 0
            with self.assertRaisesRegex(ValueError, "bijection"):
                check_array(path, frame, expected)


if __name__ == "__main__":
    unittest.main()
