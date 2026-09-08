"""Pure-standard-library staging contracts; no real data or BP access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "src/pulsedb_fewshot/official_calbased_stage.py"
SPEC = importlib.util.spec_from_file_location("official_stage_under_test", MODULE_PATH)
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


class OfficialStagingTests(unittest.TestCase):
    def test_subject_whitelist_strict(self):
        self.assertEqual(stage.validate_subjects([("MIMIC", "p000160")], {"MIMIC": 1}), [("MIMIC", "p000160")])
        for rows in [[("MIMIC", "../evil")], [("other", "p000160")],
                     [("MIMIC", "p000160"), ("MIMIC", "p000160")]]:
            with self.assertRaises(ValueError):
                stage.validate_subjects(rows, {"MIMIC": 1})

    def test_copy_and_resume_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "nas.mat", Path(tmp) / "work/file.mat"
            source.write_bytes(b"real raw signal test")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(stage.copy_verified_file(source, target, expected), "copied_verified")
            initial_stat = target.stat()
            self.assertEqual(stage.copy_verified_file(source, target, expected), "verified_existing")
            self.assertEqual(target.stat().st_mtime_ns, initial_stat.st_mtime_ns)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            self.assertEqual(list(target.parent.glob("*.partial")), [])

    def test_existing_mismatch_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "nas.mat", Path(tmp) / "work.mat"
            source.write_bytes(b"new data")
            target.write_bytes(b"old data")
            with self.assertRaisesRegex(ValueError, "preserved"):
                stage.copy_verified_file(source, target, hashlib.sha256(b"new data").hexdigest())
            self.assertEqual(target.read_bytes(), b"old data")
            self.assertEqual(source.read_bytes(), b"new data")

    def test_bad_source_checksum_does_not_install_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "nas.mat", Path(tmp) / "work.mat"
            source.write_bytes(b"bad source")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                stage.copy_verified_file(source, target, "0" * 64)
            self.assertFalse(target.exists())
            self.assertTrue(source.exists())
            self.assertEqual(list(Path(tmp).glob("*.partial")), [])

    def test_same_file_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "data.mat"
            source.write_bytes(b"data")
            with self.assertRaises(ValueError):
                stage.copy_verified_file(source, source, stage.digest_file(source))

    def test_receipt_is_valid_and_no_partial_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            stage.save_json(path, {"status": "copying"})
            stage.save_json(path, {"status": "ready"})
            self.assertEqual(json.loads(path.read_text()), {"status": "ready"})
            self.assertEqual(list(Path(tmp).glob("*.partial")), [])

    def test_plan_only_explicit_roots_and_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "nas", Path(tmp) / "work"
            raw = source / "PulseDB_MIMIC/p000160.mat"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"data")
            subjects = [("MIMIC", "p000160")]
            hashes = {subjects[0]: stage.digest_file(raw)}
            plan = stage.build_plan(subjects, hashes, source, target)
            self.assertEqual(plan[0]["bytes"], 4)
            self.assertFalse(plan[0]["existing_target"])
            self.assertFalse(target.exists())
            with self.assertRaises(ValueError):
                stage.build_plan(subjects, hashes, source, source / "child")

    def test_insufficient_space_fails_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = type("Disk", (), {"free": 99})()
            with patch.object(stage.shutil, "disk_usage", return_value=fake):
                with self.assertRaises(OSError):
                    stage.check_free_space([{"bytes": 80, "existing_target": False}], Path(tmp), 20)
                self.assertEqual(stage.check_free_space([{"bytes": 80, "existing_target": True}], Path(tmp), 20)["copy_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
