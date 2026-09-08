import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pulsedb_fewshot.official_calbased_data import (
    assign_inner_roles, interval_conflicts, join_index, parse_subject_name,
    read_official_membership, validate_memberships,
)


class OfficialMembershipTests(unittest.TestCase):
    def test_name_uses_matlab_mapping(self):
        self.assertEqual(parse_subject_name("p000160_0"), ("MIMIC", "p000160"))
        self.assertEqual(parse_subject_name("s000001_1"), ("VitalDB", "s000001"))
        for bad in ("../a_0", "p000160_3", "short"):
            with self.assertRaises(ValueError):
                parse_subject_name(bad)

    def test_info_whitelist_and_one_based_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Info.mat"
            with h5py.File(path, "w") as f:
                g = f.create_group("Train_Info")
                name = f.create_dataset("name", data=np.array([ord(c) for c in "p000160_0"], dtype="u2")[:, None])
                index = f.create_dataset("idx", data=np.array([[1, 3]], dtype=float))
                g.create_dataset("Subj_Name", (1, 1), dtype=h5py.ref_dtype)[0, 0] = name.ref
                g.create_dataset("Subj_SegIDX", (1, 1), dtype=h5py.ref_dtype)[0, 0] = index.ref
                g["SegSBP"] = h5py.ExternalLink("DO_NOT_READ.h5", "/hidden")
            out = read_official_membership(path, "official_train")
            self.assertEqual(out.segment_row.tolist(), [0, 2])
            self.assertNotIn("sbp", out)

    def test_membership_integrity(self):
        frame = pd.DataFrame({"source": ["PulseDB_MIMIC"] * 4, "file_subject": ["p000160"] * 4,
                              "segment_row": range(4)})
        validate_memberships(frame.iloc[:3], frame.iloc[3:], {"PulseDB_MIMIC": 1}, 3, 1)
        with self.assertRaises(ValueError):
            validate_memberships(frame.iloc[:3], frame.iloc[:1], {"PulseDB_MIMIC": 1}, 3, 1)

    def test_inner_split_determinism_and_counts(self):
        frame = pd.DataFrame({"subject_uid": ["a"] * 360, "segment_uid": [f"x{i}" for i in range(360)],
                              "record_id": ["r"] * 360, "start_time_s": np.arange(360) * 10})
        a, b = assign_inner_roles(frame), assign_inner_roles(frame.sample(frac=1, random_state=1))
        self.assertEqual(a.inner_role.value_counts().to_dict(), {"train": 320, "internal_validation": 40})
        self.assertEqual(set(a.loc[a.inner_role.eq("train"), "inner_fold"]), {0, 1, 2})
        pd.testing.assert_frame_equal(a.set_index("segment_uid").sort_index(), b.set_index("segment_uid").sort_index())

    def test_join_does_not_silently_drop(self):
        members = pd.DataFrame({"source": ["s"], "file_subject": ["a"], "segment_row": [0]})
        meta = members.assign(segment_uid="x")
        self.assertEqual(join_index(members, meta).segment_uid.tolist(), ["x"])
        with self.assertRaises(ValueError):
            join_index(members, meta.assign(segment_row=1))

    def test_interval_boundary_and_overlap(self):
        frame = pd.DataFrame({"source": ["s"] * 3, "subject_uid": ["a"] * 3, "record_id": ["r"] * 3,
                              "start_time_s": [0., 10., 20.], "duration_s": [10.] * 3,
                              "official_role": ["official_train", "official_test", "official_train"]})
        self.assertEqual(interval_conflicts(frame), 0)
        frame.loc[1, "start_time_s"] = 9
        self.assertEqual(interval_conflicts(frame), 1)


if __name__ == "__main__":
    unittest.main()
