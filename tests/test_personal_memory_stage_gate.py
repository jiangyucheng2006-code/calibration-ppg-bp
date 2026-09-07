import copy
import json
from pathlib import Path
import tempfile
import unittest
from pulsedb_fewshot.personal_memory_stage_gate import run, sha


class GateTests(unittest.TestCase):
    def fixture(self, root, gain=0.0, subgroups=None):
        directories = []
        for mode in ("random_disjoint", "chronological_blocked"):
            directory = root / mode
            directory.mkdir()
            manifest = {"status": "complete", "protocol_id": "development-calbased-analogue-v1",
                        "source_parent_split": "meta_train",
                        "split_mode": mode, "heldout_test_accessed": False,
                        "read_roles": ["train", "internal_validation"]}
            summary = {"status": "complete", "split_mode": mode,
                       "protocol_id": manifest["protocol_id"], "heldout_test_accessed": False,
                       "overall": {"D0": {"Overall": {"mean_mae": 4}},
                                   "D1": {"Overall": {"mean_mae": 5}},
                                   "D2": {"Overall": {"mean_mae": 4-gain}}},
                       "subgroups": subgroups or []}
            (directory / "probe_summary.json").write_text(json.dumps(summary))
            manifest["files"] = {"probe_summary.json": {"path": "probe_summary.json", "sha256": sha(directory / "probe_summary.json")}}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            directories.append(directory)
        return directories

    def test_stop_without_signal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = run(self.fixture(root), root / "gate.json")
            self.assertEqual(result["status"], "stop_no_complementarity")

    def test_overall_signal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = run(self.fixture(root, .03), root / "gate.json")
            self.assertEqual(result["status"], "proceed_to_train")
            self.assertEqual(len(result["cache_manifest_sha256"]), 2)

    def test_predefined_subgroup_only(self):
        cell = {"method": "D2", "group_type": "bp_deviation", "scope": "VitalDB",
                "bp": "SBP", "group": ">20", "n_participants": 100,
                "n_events": 500, "gain_mmhg": .16}
        for change, expected in [({}, "proceed_to_train"), ({"n_participants": 99}, "stop_no_complementarity"),
                                 ({"method": "D3"}, "stop_no_complementarity"),
                                 ({"scope": "Overall"}, "stop_no_complementarity"),
                                 ({"group_type": "posthoc_worst30"}, "stop_no_complementarity")]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                row = {**copy.deepcopy(cell), **change}
                self.assertEqual(run(self.fixture(root, subgroups=[row]), root / "gate.json")["status"], expected)

    def test_both_modes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            directories = self.fixture(root)
            with self.assertRaises(ValueError):
                run(directories[:1], root / "gate.json")
            with self.assertRaises(ValueError):
                run([directories[0], directories[0]], root / "gate.json")
            run(directories, root / "gate.json")
            with self.assertRaises(FileExistsError):
                run(directories, root / "gate.json")

    def test_modified_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            directories = self.fixture(root)
            p = directories[0] / "probe_summary.json"
            p.write_text(p.read_text() + " ")
            with self.assertRaises(ValueError):
                run(directories, root / "gate.json")


if __name__ == "__main__":
    unittest.main()
