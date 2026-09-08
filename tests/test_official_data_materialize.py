"""Small raw-MAT -> waveform shard -> official training-store contract tests.

All signals and BP values are synthetic. Broken external links stand in for
raw BP/ABP fields: PPG materialization must never dereference those fields.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from pulsedb_fewshot import official_calbased_data as data
from pulsedb_fewshot import official_calbased_train as train


class OfficialMaterializationTests(unittest.TestCase):
    def write_subject(self, path, waves, storage):
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            group = handle.create_group("Subj_Wins")
            if storage.startswith("references"):
                refs = handle.create_group("#refs#")
                shape = (1, len(waves)) if storage == "references_row" else (len(waves), 1)
                field = group.create_dataset("PPG_F", shape, dtype=h5py.ref_dtype)
                for index, wave in enumerate(waves):
                    reference = refs.create_dataset(f"ppg_{index}", data=wave[None]).ref
                    if shape[0] == 1:
                        field[0, index] = reference
                    else:
                        field[index, 0] = reference
            elif storage == "dense_rows":
                group.create_dataset("PPG_F", data=waves)
            elif storage == "dense_columns":
                group.create_dataset("PPG_F", data=waves.T)
            else:
                raise ValueError(storage)
            for name in ("SegSBP", "SegDBP", "ABP_Raw", "ABP_F"):
                group[name] = h5py.ExternalLink("DO_NOT_READ_TEST_BP.h5", "/forbidden")

    def fixture(self, root, storage_pair=("references_row", "references_column")):
        random = np.random.default_rng(914)
        raw = {}
        for source, person, storage in zip(
                ("MIMIC", "VitalDB"), ("p001111", "s001111"), storage_pair):
            waves = random.normal(size=(6, 1250)).astype(np.float64)
            path = root / "raw" / source / f"{person}.mat"
            self.write_subject(path, waves, storage)
            raw[source] = (person, path, waves)

        rows, expected = [], {}
        # Deliberately interleave sources and shuffle segment indices. A raw-file
        # groupby must not change metadata order or associate the wrong signal.
        for segment in (3, 0, 5, 1, 4, 2):
            for source in ("VitalDB", "MIMIC"):
                person, path, waves = raw[source]
                uid = f"{source}:{person}:{segment:06d}"
                role = "train" if segment < 4 else "internal_validation"
                rows.append({
                    "source": source, "subject_uid": f"{source}:{person}",
                    "segment_uid": uid, "segment_row": segment,
                    "record_id": f"record-{person}", "raw_file": str(path),
                    "start_time_s": segment * 20., "end_time_s": segment * 20. + 9.992,
                    "duration_s": 10., "sample_interval_s": .008, "n_samples": 1250,
                    "inner_role": role, "inner_fold": segment % 3 if role == "train" else -1,
                    "official_role": "official_train", "protocol_id": data.PROTOCOL_ID,
                    "waveform_file": "official_train_000.npy",
                    "sbp": 111. + segment + (8 if source == "VitalDB" else 0),
                    "dbp": 65. + segment * .4 + (4 if source == "VitalDB" else 0),
                })
                expected[uid] = waves[segment].astype("<f4")
        frame = pd.DataFrame(rows)
        # The mmap row is a locator, not necessarily the DataFrame's ordinal.
        frame["waveform_row"] = np.roll(np.arange(len(frame))[::-1], 2)
        plan_path, signal_path = root / "official_train_000.parquet", root / "official_train_000.npy"
        frame.to_parquet(plan_path, index=False)
        return frame, expected, plan_path, signal_path

    def make_ready_store(self, root, materialized):
        manifest_path = root / "train_manifest.parquet"
        materialized.to_parquet(manifest_path, index=False)
        receipt = {
            "protocol_id": data.PROTOCOL_ID, "status": "ready", "synthetic": True,
            "train_manifest_sha256": train.sha256(manifest_path),
        }
        (root / "manifest.json").write_text(json.dumps(receipt), encoding="utf-8")
        # Reading this trap would fail, so load_store(smoke=True) must restrict
        # itself to the TRAIN manifest and referenced training waveform files.
        (root / "test_inputs.parquet").write_bytes(b"OFFICIAL TEST MUST REMAIN UNOPENED")

    def test_referenced_mat_to_training_store_preserves_order_hash_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            planned, expected, plan_path, signal_path = self.fixture(root)
            result = data.materialize_shard((str(plan_path), str(signal_path)))
            actual = pd.read_parquet(plan_path)
            signals = np.load(signal_path)

            self.assertEqual(result["rows"], len(planned))
            self.assertEqual(result["file"], signal_path.name)
            self.assertEqual(result["sha256"], data.digest_file(signal_path))
            self.assertEqual(signals.shape, (12, 1250))
            self.assertEqual(signals.dtype, np.dtype("<f4"))
            pd.testing.assert_frame_equal(actual[planned.columns], planned)
            for row in actual.itertuples():
                wave = expected[row.segment_uid]
                np.testing.assert_array_equal(signals[row.waveform_row], wave)
                self.assertEqual(row.ppg_content_sha256, hashlib.sha256(wave.tobytes()).hexdigest())
                self.assertEqual(Path(row.waveform_file), Path(signal_path.name))

            self.make_ready_store(root, actual)
            receipt, loaded = train.load_store(root, smoke=True)
            self.assertTrue(receipt["synthetic"])
            self.assertEqual(set(loaded.source), {"MIMIC", "VitalDB"})
            self.assertEqual(loaded.segment_uid.tolist(), planned.segment_uid.tolist())
            fit, exported = train.select_stage(loaded, "inner")
            self.assertEqual(len(fit), 8)
            self.assertEqual(len(exported["validation"]), 4)
            self.assertFalse(set(fit.segment_uid) & set(exported["validation"].segment_uid))

    def test_both_dense_matrix_orientations_materialize_the_correct_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, expected, plan_path, signal_path = self.fixture(root, ("dense_rows", "dense_columns"))
            data.materialize_shard((plan_path, signal_path))
            frame, signals = pd.read_parquet(plan_path), np.load(signal_path)
            for row in frame.itertuples():
                np.testing.assert_array_equal(signals[row.waveform_row], expected[row.segment_uid])

    def test_input_only_materialization_does_not_add_bp_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            planned, _, plan_path, signal_path = self.fixture(root)
            inputs = planned.drop(columns=["sbp", "dbp"])
            inputs["official_role"] = "official_test"
            inputs["inner_role"] = "official_test"
            inputs.to_parquet(plan_path, index=False)
            data.materialize_shard((plan_path, signal_path))
            output = pd.read_parquet(plan_path)
            self.assertFalse({"sbp", "dbp", "SegSBP", "SegDBP", "ABP_Raw"} & set(output))
            self.assertEqual(output.segment_uid.tolist(), inputs.segment_uid.tolist())
            self.assertTrue(output.ppg_content_sha256.str.fullmatch("[0-9a-f]{64}").all())

    def test_manifest_hash_drift_is_rejected_after_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, plan_path, signal_path = self.fixture(root)
            data.materialize_shard((plan_path, signal_path))
            frame = pd.read_parquet(plan_path)
            self.make_ready_store(root, frame)
            frame.loc[0, "sbp"] += 1.
            frame.to_parquet(root / "train_manifest.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "manifest hash changed"):
                train.load_store(root, smoke=True)

    def test_constant_ppg_fails_without_silently_dropping_the_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            planned, _, plan_path, signal_path = self.fixture(root)
            raw = Path(planned.raw_file.iloc[0])
            with h5py.File(raw, "r+") as handle:
                field = handle["Subj_Wins/PPG_F"]
                ref = field[3, 0] if field.shape[1] == 1 else field[0, 3]
                handle[ref][...] = 1.
            with self.assertRaisesRegex(ValueError, "invalid official PPG window"):
                data.materialize_shard((plan_path, signal_path))
            # The failed worker must not produce a falsely accepted hash table.
            self.assertNotIn("ppg_content_sha256", pd.read_parquet(plan_path))


if __name__ == "__main__":
    unittest.main()
