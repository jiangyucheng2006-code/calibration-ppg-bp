"""Official runner access-contract tests; optional real-architecture CPU smoke."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pulsedb_fewshot import official_calbased_train as official


def fixture_frame(people=6):
    rows = []
    for person in range(people):
        source = "MIMIC" if person < people // 2 else "VitalDB"
        for i in range(11):
            role = "train" if i < 9 else "internal_validation"
            rows.append({"subject_uid": f"{source}:person{person}", "segment_uid": f"p{person}:w{i}",
                "source": source, "waveform_file": "ppg.npy", "waveform_row": len(rows),
                "ppg_content_sha256": hashlib.sha256(f"{person}:{i}".encode()).hexdigest(),
                "record_id": f"record{person}", "start_time_s": i * 20.,
                "end_time_s": i * 20. + 9.992, "duration_s": 10., "sample_interval_s": .008,
                "inner_role": role, "inner_fold": i % 3 if i < 9 else -1,
                "sbp": 110. + person * 3 + i, "dbp": 65. + person * 2 + i * .4})
    return pd.DataFrame(rows)


class OfficialContractTests(unittest.TestCase):
    def test_valid_seen_subject_overlap(self):
        frame = official.validate_training_frame(fixture_frame(), smoke=True)
        fit, exports = official.select_stage(frame, "inner")
        self.assertEqual(set(fit.subject_uid), set(exports["validation"].subject_uid))
        self.assertFalse(set(fit.segment_uid) & set(exports["validation"].segment_uid))

    def test_formal_counts_enforced(self):
        with self.assertRaisesRegex(ValueError, "2506"):
            official.validate_training_frame(fixture_frame())

    def test_invalid_role_fails(self):
        frame = fixture_frame()
        frame.loc[0, "inner_role"] = "official_test"
        with self.assertRaisesRegex(ValueError, "forbidden inner role"):
            official.validate_training_frame(frame, smoke=True)

    def test_duplicate_window_fails(self):
        frame = fixture_frame()
        frame.loc[9, "segment_uid"] = frame.loc[0, "segment_uid"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            official.validate_training_frame(frame, smoke=True)

    def test_duplicate_content_crossing_roles_fails(self):
        frame = fixture_frame()
        frame.loc[9, "ppg_content_sha256"] = frame.loc[0, "ppg_content_sha256"]
        with self.assertRaisesRegex(ValueError, "content crosses"):
            official.validate_training_frame(frame, smoke=True)

    def test_overlapping_physiological_interval_fails(self):
        frame = fixture_frame()
        frame.loc[9, ["start_time_s", "end_time_s"]] = [5., 14.992]
        with self.assertRaisesRegex(ValueError, "physiological interval"):
            official.validate_training_frame(frame, smoke=True)

    def test_actual_touching_boundary_does_not_add_one_sample(self):
        frame = fixture_frame()
        frame.loc[0, ["start_time_s", "end_time_s"]] = [16380.016, 16390.008]
        frame.loc[9, ["start_time_s", "end_time_s"]] = [16390.008, 16400.]
        accepted = official.validate_training_frame(frame, smoke=True)
        metadata = official.canonical_metadata(accepted, "train")
        self.assertEqual(metadata.loc[0, "end_s"], metadata.loc[9, "start_s"])
        self.assertTrue(metadata.time_boundary_policy.eq(official.TIME_BOUNDARY_POLICY).all())

    def test_real_one_sample_overlap_still_fails(self):
        frame = fixture_frame()
        frame.loc[9, ["start_time_s", "end_time_s"]] = [9.984, 19.976]
        with self.assertRaisesRegex(ValueError, "physiological interval"):
            official.validate_training_frame(frame, smoke=True)

    def test_oof_exact_touching_span_allowed(self):
        frame = fixture_frame()
        frame.loc[1, ["start_time_s", "end_time_s"]] = [9.992, 19.984]
        fit, exports = official.select_stage(frame, "oof", 0)
        self.assertFalse(set(fit.segment_uid) & set(exports["excluded_fold"].segment_uid))

    def test_invalid_float_row_rejected(self):
        frame = fixture_frame().astype({"waveform_row": float})
        frame.loc[0, "waveform_row"] = .5
        with self.assertRaisesRegex(ValueError, "row index"):
            official.validate_training_frame(frame, smoke=True)

    def test_oof_every_inner_row_excluded_once(self):
        frame = official.validate_training_frame(fixture_frame(), smoke=True)
        all_held = []
        for fold in range(3):
            fit, exports = official.select_stage(frame, "oof", fold)
            held = exports["excluded_fold"]
            self.assertFalse(set(fit.segment_uid) & set(held.segment_uid))
            self.assertEqual(set(fit.subject_uid), set(held.subject_uid))
            self.assertFalse(set(fit.segment_uid) & set(exports["outer_validation"].segment_uid))
            all_held.extend(held.segment_uid)
        self.assertEqual(len(all_held), len(set(all_held)))
        self.assertEqual(set(all_held), set(frame.loc[frame.inner_role.eq("train"), "segment_uid"]))

    def test_oof_fold_content_reuse_fails(self):
        frame = fixture_frame()
        frame.loc[1, "ppg_content_sha256"] = frame.loc[0, "ppg_content_sha256"]
        with self.assertRaisesRegex(ValueError, "OOF held fold reuses"):
            official.select_stage(frame, "oof", 0)

    def test_missing_fold_participant_fails(self):
        frame = fixture_frame()
        frame.loc[frame.subject_uid.eq(frame.subject_uid.iloc[0]) & frame.inner_role.eq("train"), "inner_fold"] = 0
        with self.assertRaisesRegex(ValueError, "every registered subject"):
            official.select_stage(frame, "oof", 0)

    def test_only_fit_labels_change_anchors_scaler(self):
        frame = fixture_frame()
        fit, exports = official.select_stage(frame, "oof", 0)
        scaler, anchors, mapping = official.fit_personal_state(fit)
        modified = frame.copy()
        modified.loc[~modified.segment_uid.isin(fit.segment_uid), ["sbp", "dbp"]] = 10000.
        repeated_fit, _ = official.select_stage(modified, "oof", 0)
        again, repeated_anchors, repeated_mapping = official.fit_personal_state(repeated_fit)
        self.assertEqual(scaler, again)
        pd.testing.assert_frame_equal(anchors, repeated_anchors)
        self.assertEqual(mapping, repeated_mapping)

    def test_final_uses_full_official_training(self):
        frame = fixture_frame()
        fit, exports = official.select_stage(frame, "final")
        self.assertEqual(len(fit), len(frame))
        self.assertEqual(set(exports), {"train"})

    def test_prediction_metadata_has_no_labels(self):
        frame = fixture_frame()
        metadata = official.canonical_metadata(frame, "excluded_fold")
        self.assertFalse({"sbp", "dbp", "target_sbp", "target_dbp"} & set(metadata))
        self.assertTrue(metadata.oof_role.eq("excluded_fold").all())
        self.assertTrue(np.allclose(metadata.end_s - metadata.start_s, 9.992))
        self.assertTrue(np.allclose(metadata.duration_s, 10.))

    def test_identity_hash_contract(self):
        expected = hashlib.sha256(b"a\nb\n").hexdigest()
        self.assertEqual(official.event_ids_sha256(["b", "a"]), expected)

    def test_load_store_does_not_read_test_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official.save_json(root / "manifest.json", {"protocol_id": official.PROTOCOL_ID, "status": "ready", "synthetic": True})
            (root / "ppg.npy").touch()
            requested = []
            def read(path):
                requested.append(Path(path).name)
                self.assertEqual(Path(path).name, "train_manifest.parquet")
                return fixture_frame()
            with patch.object(pd, "read_parquet", side_effect=read):
                official.load_store(root, smoke=True)
            self.assertEqual(requested, ["train_manifest.parquet"])

    def test_synthetic_flag_cannot_relax_real_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official.save_json(root / "manifest.json", {"protocol_id": official.PROTOCOL_ID, "status": "ready"})
            with self.assertRaisesRegex(ValueError, "smoke requires"):
                official.load_store(root, smoke=True)

    def test_real_store_requires_declared_sample_span_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official.save_json(root / "manifest.json", {"protocol_id": official.PROTOCOL_ID, "status": "ready"})
            with self.assertRaisesRegex(ValueError, "sample-span time contract"):
                official.load_store(root)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official.save_json(root / "manifest.json", {"protocol_id": official.PROTOCOL_ID, "status": "ready", "synthetic": True})
            frame = fixture_frame()
            frame.loc[0, "waveform_file"] = "../private.npy"
            with patch.object(pd, "read_parquet", return_value=frame):
                with self.assertRaisesRegex(ValueError, "escaping"):
                    official.load_store(root, smoke=True)

    def test_oof_fixed_epochs_cannot_use_selected_inner_epochs(self):
        args = argparse.Namespace(stage="oof", epochs=3, selection_run=None, smoke=False, oof_epochs=25)
        with self.assertRaisesRegex(ValueError, "prespecified"):
            official.resolve_fixed_epochs(args, "hash")

    def test_oof_epochs_prespecified_25_without_validation_selection(self):
        args = argparse.Namespace(stage="oof", epochs=0, selection_run=None, smoke=False, oof_epochs=25)
        epochs, evidence = official.resolve_fixed_epochs(args, "hash")
        self.assertEqual(epochs, 25)
        self.assertEqual(evidence["epoch_selection_role"], "none_prespecified")
        args.oof_epochs = 26
        with self.assertRaisesRegex(ValueError, "prespecified as 25"):
            official.resolve_fixed_epochs(args, "hash")

    def test_inner_epochcap_not_silently_enabled(self):
        args = argparse.Namespace(stage="inner", epochs=25, selection_run=None, smoke=False)
        with self.assertRaisesRegex(ValueError, "no fixed epoch cap"):
            official.resolve_fixed_epochs(args, "hash")

    def test_inner_selected_epochs_reused_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official.save_json(root / "run.json", {"protocol_id": official.PROTOCOL_ID, "stage": "inner",
                "status": "complete", "store_manifest_sha256": "hash", "official_test_targets_accessed": False,
                "best_epoch": 13, "seed": 99, "arguments": {"batch_size": 64, "learning_rate": .0003,
                    "weight_decay": .0001, "huber_delta": .5, "gradient_clip": 5.}})
            args = argparse.Namespace(stage="final", epochs=0, selection_run=root, smoke=False, seed=99,
                batch_size=64, learning_rate=.0003, weight_decay=.0001, huber_delta=.5, gradient_clip=5.)
            epochs, evidence = official.resolve_fixed_epochs(args, "hash")
            self.assertEqual(epochs, 13)
            self.assertEqual(evidence["epoch_selection_role"], "official_train_internal_validation")
            args.epochs = 14
            with self.assertRaisesRegex(ValueError, "must equal"):
                official.resolve_fixed_epochs(args, "hash")
            args.epochs = 0
            args.learning_rate = .01
            with self.assertRaisesRegex(ValueError, "preserve inner selected training setting"):
                official.resolve_fixed_epochs(args, "hash")

    def test_test_labels_rejected(self):
        frame = fixture_frame()
        with patch.object(pd, "read_parquet", return_value=frame):
            with self.assertRaisesRegex(ValueError, "contains targets"):
                official.load_test_inputs(Path("."), frame)

    def test_checkpoint_selection_is_participant_macro(self):
        frame = pd.DataFrame({"subject_uid": ["a", "a", "a", "b"], "sbp": [100.] * 4, "dbp": [60.] * 4})
        prediction = np.array([[100., 60.]] * 3 + [[110., 70.]])
        self.assertEqual(official.participant_score(frame, prediction), 5.)


@unittest.skipUnless(official.torch is not None and importlib.util.find_spec("pyarrow"), "torch/pyarrow runtime required")
class OfficialTorchSmokeTests(unittest.TestCase):
    def build_store(self, root):
        frame = fixture_frame()
        rng = np.random.default_rng(7)
        waveforms = rng.normal(size=(len(frame), 1250)).astype(np.float32)
        frame["ppg_content_sha256"] = [hashlib.sha256(row.tobytes()).hexdigest() for row in waveforms]
        np.save(root / "ppg.npy", waveforms)
        frame.to_parquet(root / "train_manifest.parquet", index=False)
        official.save_json(root / "manifest.json", {"protocol_id": official.PROTOCOL_ID, "status": "ready", "synthetic": True,
            "train_manifest_sha256": official.sha256(root / "train_manifest.parquet")})
        # A trap: inner and OOF must never parse this file.
        (root / "test_inputs.parquet").write_bytes(b"do not read official test")
        return frame

    def test_real_architecture_cpu_inner_and_oof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = self.build_store(root)
            parser = official.parser()
            inner = parser.parse_args(["--store-root", str(root), "--output", str(root / "inner"),
                "--stage", "inner", "--epochs", "1", "--smoke", "--device", "cpu", "--workers", "0", "--batch-size", "16"])
            first = official.run(inner)
            self.assertEqual(first["status"], "complete")
            self.assertFalse(first["official_test_targets_accessed"])
            self.assertEqual(first["best_epoch"], 1)
            self.assertEqual(np.load(root / "inner/cache/train/features.npy").shape, (54, 256))
            oof = parser.parse_args(["--store-root", str(root), "--output", str(root / "oof0"),
                "--stage", "oof", "--fold", "0", "--oof-epochs", "1",
                "--smoke", "--device", "cpu", "--workers", "0", "--batch-size", "16"])
            second = official.run(oof)
            self.assertEqual(second["status"], "complete")
            self.assertEqual(second["epochs_completed"], 1)
            self.assertEqual(second["encoder_protocol"], "model-level-exclusion")
            self.assertFalse(second["official_test_inputs_accessed"])
            metadata = pd.read_parquet(root / "oof0/cache/excluded_fold/metadata.parquet")
            fit = pd.read_parquet(root / "oof0/encoder_fit_manifest.parquet")
            self.assertFalse(set(fit.segment_uid) & set(metadata.event_id))
            self.assertEqual(len(metadata), 18)
            provenance = json.loads((root / "oof0/cache/excluded_fold/provenance.json").read_text())
            self.assertEqual(provenance["encoder_initialization"], "scratch")
            self.assertFalse(provenance["model_selection_excluded_fold_labels_used"])
            self.assertTrue(provenance["fit_transforms_on_source_fit_only"])
            # Final full-training refit can export only explicitly provided test
            # INPUTS. There is no test BP file, even in this synthetic workflow.
            rng = np.random.default_rng(19)
            test_waveforms = rng.normal(size=(6 * 40, 1250)).astype(np.float32)
            np.save(root / "test_ppg.npy", test_waveforms)
            test_rows = []
            for person, (_, group) in enumerate(frame.groupby("subject_uid", sort=True)):
                base = group.iloc[0].to_dict()
                for i in range(40):
                    row = {k: v for k, v in base.items() if k not in {"sbp", "dbp", "inner_role", "inner_fold"}}
                    wave_row = person * 40 + i
                    row.update(segment_uid=f"{row['subject_uid']}:test{i}", waveform_file="test_ppg.npy", waveform_row=wave_row,
                               start_time_s=1000. + i * 20, end_time_s=1009.992 + i * 20,
                               ppg_content_sha256=hashlib.sha256(test_waveforms[wave_row].tobytes()).hexdigest())
                    test_rows.append(row)
            pd.DataFrame(test_rows).to_parquet(root / "test_inputs.parquet", index=False)
            final = parser.parse_args(["--store-root", str(root), "--output", str(root / "final"),
                "--stage", "final", "--selection-run", str(root / "inner"), "--include-test-inputs",
                "--smoke", "--device", "cpu", "--workers", "0", "--batch-size", "16"])
            third = official.run(final)
            self.assertEqual(third["fit_rows"], len(frame))
            self.assertTrue(third["official_test_inputs_accessed"])
            self.assertFalse(third["official_test_targets_accessed"])
            self.assertFalse((root / "final/cache/test_inputs/bp.npy").exists())
            predictions = pd.read_parquet(root / "final/official_test_input_predictions.parquet")
            self.assertEqual(len(predictions), 240)
            self.assertEqual(set(predictions), {"subject_uid", "event_id", "source", "pred_sbp", "pred_dbp"})


if __name__ == "__main__":
    unittest.main()
