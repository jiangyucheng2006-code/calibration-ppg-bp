"""Protocol, label access, old-run regression and full synthetic pipeline tests."""
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

from pulsedb_fewshot.legacy_enrollment_protocol import (
    METHODS, CONFIG, build_assignment, assign_person, reject_labels, prepare, partition, load_plan,
)


def fixture():
    rows, subjects, waves, labels = [], [], [], []
    rng = np.random.default_rng(402)
    for s in range(12):
        source = "MIMIC" if s < 6 else "VitalDB"
        person = f"{source}:person{s:02d}"
        split = ("meta_train", "meta_validation", "meta_test")[s % 3]
        subjects.append(dict(subject_uid=person, source=source, split=split))
        for i in range(80):
            wave = rng.normal(size=1250).astype(np.float32)
            row = dict(subject_uid=person, source=source, segment_uid=f"{person}:w{i:03d}",
                       waveform_file="ppg.npy", waveform_row=len(waves),
                       ppg_content_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                       record_id=f"record{s}", start_time_s=i * 20., end_time_s=i * 20. + 9.992,
                       duration_s=10., sample_interval_s=.008)
            rows.append(row)
            labels.append(dict(subject_uid=person, source=source, segment_uid=row["segment_uid"],
                               sbp=110 + s * 2 + i * .2, dbp=60 + s + i * .1))
            waves.append(wave)
    return pd.DataFrame(rows), pd.DataFrame(subjects), np.array(waves), pd.DataFrame(labels)


class ContractTests(unittest.TestCase):
    def test_old_outer_assignments_are_preserved(self):
        meta, old, _, _ = fixture()
        result, audit = build_assignment(meta, old)
        actual = result[["subject_uid", "source", "split"]].drop_duplicates().sort_values("subject_uid").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, old.sort_values("subject_uid").reset_index(drop=True))
        self.assertEqual(audit["cross_outer_subject_overlap"], 0)
        self.assertEqual(len(result), len(meta))
        self.assertEqual(set(result.loc[result.split.eq("meta_train"), "personal_role"]), {"population_fit"})

    def test_input_order_cannot_change_assignment(self):
        meta, old, _, _ = fixture()
        a = build_assignment(meta, old)[0].set_index("segment_uid")
        b = build_assignment(meta.sample(frac=1, random_state=3), old.sample(frac=1, random_state=4))[0].set_index("segment_uid")
        cols = ["split", "personal_role", "inner_role"]
        pd.testing.assert_frame_equal(a[cols].sort_index(), b[cols].sort_index())

    def test_target_and_error_columns_cannot_drive_assignment(self):
        meta, old, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "forbidden reference"):
            build_assignment(meta.assign(sbp=123.), old)
        base = build_assignment(meta, old)[0]
        other = build_assignment(meta.assign(previous_error=np.nan), old)[0]
        pd.testing.assert_frame_equal(base[["segment_uid", "personal_role", "inner_role"]],
                                      other[["segment_uid", "personal_role", "inner_role"]])

    def test_ratios_are_90_10_and_nested_80_10_10(self):
        meta, _, _, _ = fixture()
        f = assign_person(meta.loc[meta.subject_uid.eq(meta.subject_uid.iloc[0])])
        self.assertEqual(f.personal_role.value_counts().to_dict(), {"registration": 72, "query": 8})
        self.assertEqual(f.inner_role.value_counts().to_dict(), {"train": 64, "internal_validation": 8, "query": 8})

    def test_duplicate_group_stays_together(self):
        meta, _, _, _ = fixture()
        f = meta.loc[meta.subject_uid.eq(meta.subject_uid.iloc[0])].copy()
        f.loc[1, "ppg_content_sha256"] = f.loc[0, "ppg_content_sha256"]
        actual = assign_person(f)
        self.assertTrue(actual.groupby("ppg_content_sha256").inner_role.nunique().eq(1).all())

    def test_overlapping_intervals_stay_together(self):
        meta, _, _, _ = fixture()
        f = meta.loc[meta.subject_uid.eq(meta.subject_uid.iloc[0])].copy()
        f.loc[1, ["start_time_s", "end_time_s"]] = [5., 14.992]
        actual = assign_person(f).set_index("segment_uid")
        self.assertEqual(actual.loc[f.iloc[0].segment_uid, "inner_role"], actual.loc[f.iloc[1].segment_uid, "inner_role"])

    def test_cross_outer_identity_content_quarantines_whole_people(self):
        meta, old, _, _ = fixture()
        meta.loc[80, "ppg_content_sha256"] = meta.loc[0, "ppg_content_sha256"]
        actual, audit = build_assignment(meta, old)
        self.assertEqual(audit["quarantined_subjects"], sorted(meta.loc[[0, 80], "subject_uid"].tolist()))
        self.assertEqual(len(actual), len(meta) - 160)

    def test_duplicate_and_unknown_person_rejected(self):
        meta, old, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            build_assignment(pd.concat([meta, meta.iloc[:1]]), old)
        with self.assertRaisesRegex(ValueError, "absent from saved"):
            build_assignment(meta, old.iloc[1:])

    def test_only_two_methods_frozen(self):
        self.assertEqual(METHODS, ["new_person_lora", "new_person_lora_memory"])
        self.assertEqual(CONFIG["personal_parameters"], 2048)

    def test_reference_fields_rejected(self):
        for c in ("target_sbp", "SegDBP", "abp_wave"):
            with self.assertRaises(ValueError):
                reject_labels(pd.DataFrame({c: [0]}))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("pyarrow"), "full project runtime required")
class PipelineTest(unittest.TestCase):
    def test_full_paired_pipeline_and_target_access(self):
        import torch
        from pulsedb_fewshot import legacy_enrollment_train as train
        from pulsedb_fewshot import legacy_enrollment_evaluate as evaluate
        from pulsedb_fewshot.official_calbased_train import save_json, sha256
        from pulsedb_fewshot.official_calbased_evaluate import read_exact_targets
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = root / "source"
            store.mkdir()
            meta, old, waves, labels = fixture()
            original_train = meta.groupby("subject_uid", group_keys=False).head(72)
            original_train.merge(labels, on=["subject_uid", "segment_uid", "source"]).assign(inner_role="train").to_parquet(store / "train_manifest.parquet", index=False)
            meta.loc[~meta.segment_uid.isin(original_train.segment_uid)].to_parquet(store / "test_inputs.parquet", index=False)
            np.save(store / "ppg.npy", waves)
            save_json(store / "manifest.json", {"protocol_id": "pulsedb-official-calbased-v1", "status": "ready", "synthetic": True})
            old.to_csv(root / "old.csv", index=False)
            labels.to_parquet(root / "labels.parquet", index=False)
            prep = argparse.Namespace(store_root=store, output=root / "prepare", legacy_split=root / "old.csv", full_index=root / "labels.parquet", synthetic=True)
            read_ids = set()
            def read_allowed(path, frame):
                read_ids.update(frame.segment_uid)
                return read_exact_targets(path, frame)
            with patch("pulsedb_fewshot.official_calbased_evaluate.read_exact_targets", side_effect=read_allowed):
                plan = prepare(prep)
            plan_path = root / "prepare" / "plan.json"
            _, test_inputs = partition(plan_path, "test_inputs", access="personal_test", synthetic=True)
            self.assertFalse(read_ids & set(test_inputs.segment_uid))
            for name in ("test_registration", "test_inputs"):
                with self.assertRaisesRegex(ValueError, "access forbidden"):
                    partition(plan_path, name, access="population", synthetic=True)
            with self.assertRaisesRegex(ValueError, "access forbidden"):
                partition(plan_path, "population_train", access="personal_test", synthetic=True)
            pop = train.parser().parse_args(["--stage", "population", "--plan", str(plan_path), "--output", str(root / "population"), "--synthetic", "--epochs", "1", "--device", "cpu", "--workers", "0"])
            population = train.population(pop)
            self.assertFalse(population["test_registration_accessed"])
            before = population["shared_state_sha256"]
            for cohort in ("validation", "test"):
                runs = []
                for shard in (0, 1):
                    directory = root / f"{cohort}_{shard}"
                    args = train.parser().parse_args(["--stage", "personal", "--cohort", cohort, "--shard", str(shard), "--plan", str(plan_path), "--population-run", str(root / "population"), "--validation-run", str(root / "validation_score"), "--output", str(directory), "--device", "cpu", "--synthetic", "--workers", "0"])
                    report = train.personal(args)
                    self.assertEqual(len(report["prediction_files"]), 2)
                    self.assertTrue(all(r["shared_frozen_verified"] and r["reload_equivalence"] for r in report["profiles"].values()))
                    runs.append(directory)
                args = argparse.Namespace(plan=plan_path, population_run=root / "population", personal_runs=runs,
                                          cohort=cohort, output=root / f"{cohort}_score", synthetic=True)
                result = evaluate.run(args)
                self.assertTrue(result["all_predictions_frozen_before_targets"])
                table = pd.read_csv(args.output / "participant_macro.csv")
                self.assertEqual(len(table), 6)
                self.assertEqual(set(table.Scope), {"Overall", "MIMIC", "VitalDB"})
            self.assertEqual(train.load_population(root / "population", plan_path, synthetic=True)[1]["shared_state_sha256"], before)
            # Wrong old/all-subject checkpoints or changed plan files are rejected.
            run_path = root / "population" / "run.json"
            tampered = json.loads(run_path.read_text())
            tampered["old_checkpoint_used"] = True
            save_json(run_path, tampered)
            with self.assertRaisesRegex(ValueError, "provenance"):
                train.load_population(root / "population", plan_path, synthetic=True)
            plan_path.write_text(plan_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_plan(plan_path, synthetic=True)


if __name__ == "__main__":
    unittest.main()
