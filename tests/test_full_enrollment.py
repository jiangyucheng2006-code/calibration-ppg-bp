"""Full-scope accounting, exact target access, and variable-history contracts."""
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

from test_legacy_enrollment import fixture
from pulsedb_fewshot.legacy_enrollment_protocol import FULL_CONFIG, FULL_PROTOCOL, build_assignment, assign_person


class FullContractTests(unittest.TestCase):
    def test_full_scope_has_no_400_window_or_official_membership_gate(self):
        meta, old, _, _ = fixture()
        keep = np.arange(len(meta)) % 80 < 17
        f, audit = build_assignment(meta.loc[keep], old, full_cohort=True)
        self.assertEqual(f.subject_uid.nunique(), len(old))
        self.assertEqual(len(f), 17 * len(old))
        self.assertEqual(audit["source_pool_subjects"], len(old))
        self.assertEqual(FULL_CONFIG["cohort_scope"], "all_original_subjects_all_valid_windows")

    def test_short_histories_retained_when_three_distinct_roles_are_possible(self):
        meta, _, _, _ = fixture()
        for n in (3, 4, 7, 16, 80):
            actual = assign_person(meta.iloc[:n], full_cohort=True)
            self.assertEqual(len(actual), n)
            self.assertEqual(set(actual.inner_role), {"train", "internal_validation", "query"})
        with self.assertRaisesRegex(ValueError, "insufficient"):
            assign_person(meta.iloc[:2], full_cohort=True)

    def test_short_population_people_not_excluded(self):
        meta, old, _, _ = fixture()
        meta = meta.loc[(np.arange(len(meta)) % 80 == 0) | ~meta.subject_uid.isin(old.loc[old.split.eq("meta_train"), "subject_uid"])]
        f, _ = build_assignment(meta, old, full_cohort=True)
        self.assertEqual(set(f.subject_uid), set(old.subject_uid))

    def test_duplicate_collapse_is_label_blind_and_deterministic(self):
        meta, old, _, _ = fixture()
        meta.loc[1, "ppg_content_sha256"] = meta.loc[0, "ppg_content_sha256"]
        f, audit = build_assignment(meta, old, full_cohort=True)
        self.assertEqual(audit["same_outer_duplicate_rows_collapsed"], 1)
        self.assertEqual(len(f), len(meta) - 1)
        other = build_assignment(meta.sample(frac=1, random_state=5), old, full_cohort=True)[0]
        pd.testing.assert_frame_equal(f.sort_values("segment_uid").reset_index(drop=True),
                                      other.sort_values("segment_uid").reset_index(drop=True))

    def test_memory_chunking_preserves_retrieval_and_fallback(self):
        from pulsedb_fewshot.personal_memory_prepare import prepare_neighbors
        rng = np.random.default_rng(19)
        train = [dict(subject_uid="p", recording_uid="r", time_axis_uid="r", window_uid=f"w{i:03}",
                      start_s=float(i*20), end_s=float(i*20+10)) for i in range(95)]
        query = [dict(train[i], window_uid=f"q{i}", start_s=float(3000+i*20), end_s=float(3010+i*20)) for i in range(10)]
        x, q = rng.normal(size=(95, 256)), rng.normal(size=(10, 256))
        a = prepare_neighbors(x, q, train, query, mode="random_disjoint", audit=False)
        b = prepare_neighbors(x, q, train, query, mode="random_disjoint", audit=False, query_chunk_size=7)
        for role in ("train", "validation"):
            for key in a[role]:
                np.testing.assert_allclose(a[role][key], b[role][key], atol=1e-7, rtol=0)
        for key in a["train_distance_cutpoints"]:
            self.assertAlmostEqual(a["train_distance_cutpoints"][key], b["train_distance_cutpoints"][key], places=12)


@unittest.skipUnless(all(importlib.util.find_spec(p) for p in ("torch", "pyarrow", "h5py")), "server runtime required")
class FullPipelineTests(unittest.TestCase):
    def test_full_raw_materialization_to_frozen_two_method_evaluation(self):
        import h5py
        import pyarrow as pa
        import pyarrow.parquet as pq
        import torch
        from pulsedb_fewshot import full_enrollment_data as data
        from pulsedb_fewshot import legacy_enrollment_train as train
        from pulsedb_fewshot import legacy_enrollment_evaluate as evaluate
        from pulsedb_fewshot.legacy_enrollment_protocol import partition, load_plan
        from pulsedb_fewshot.official_calbased_train import sha256
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meta, old, waves, labels = fixture()
            old.to_csv(root / "split.csv", index=False)
            raw_root = root / "raw"
            tables = []
            for group_id, (subject, g) in enumerate(meta.groupby("subject_uid", sort=True)):
                # Exercise a short eligible target history, a target with too few
                # rows, a one-row population person, and several non-400 budgets.
                n = {0: 1, 1: 2, 2: 3}.get(group_id, 55 + group_id)
                g = g.iloc[:n].copy()
                directory = "PulseDB_MIMIC" if g.source.iloc[0] == "MIMIC" else "PulseDB_Vital"
                file = raw_root / directory / f"person{group_id}.mat"
                file.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(file, "w") as h:
                    out = h.create_group("Subj_Wins").create_dataset("PPG_F", (1, len(g)), dtype=h5py.ref_dtype)
                    for j, index in enumerate(g.index):
                        d = h.create_dataset(f"ref{j}", data=waves[index][None].astype(np.float64))
                        out[0, j] = d.ref
                g = g.drop(columns=["waveform_file", "waveform_row", "ppg_content_sha256"])
                g = g.merge(labels, on=data.KEYS, validate="one_to_one")
                g = g.assign(subject_id=subject.split(":")[1], n_samples=1250, raw_file=str(file),
                             raw_file_sha256=sha256(file), ppg_field="PPG_F", ppg_storage_mode="references",
                             ppg_reference_index=np.arange(len(g)), segment_schema_valid=True, segment_exclusion_reasons="")
                tables.append(pa.Table.from_pandas(g, preserve_index=False))
            with pq.ParquetWriter(root / "index.parquet", tables[0].schema) as writer:
                for table in tables:
                    writer.write_table(table)
            args = argparse.Namespace(full_index=root / "index.parquet", legacy_split=root / "split.csv",
                                      raw_root=raw_root, output=root / "store", workers=1, shards=2, synthetic=True)
            data.materialize(args)
            args.store_root, args.output = root / "store", root / "prepare"
            accessed = set()
            original = data.read_grouped_targets
            def tracked(index, keys):
                accessed.update(keys.segment_uid)
                return original(index, keys)
            with patch.object(data, "read_grouped_targets", side_effect=tracked):
                plan = data.prepare(args)
            path = args.output / "plan.json"
            self.assertTrue(plan["all_original_subjects_accounted_for"])
            self.assertEqual(plan["original_subjects"], 12)
            self.assertEqual(plan["protocol_id"], FULL_PROTOCOL)
            self.assertEqual(plan["audit"]["excluded_people"], 1)
            _, query = partition(path, "test_inputs", access="personal_test", synthetic=True)
            self.assertFalse(accessed & set(query.segment_uid))
            self.assertTrue(load_plan(path, synthetic=True)["all_original_subjects_accounted_for"])
            pop_args = train.parser().parse_args(["--stage", "population", "--plan", str(path), "--output", str(root / "population"),
                "--synthetic", "--epochs", "1", "--device", "cpu", "--workers", "0"])
            result = train.population(pop_args)
            self.assertEqual(result["protocol_id"], FULL_PROTOCOL)
            for cohort in ("validation", "test"):
                personal_runs = []
                for shard in (0, 1):
                    directory = root / f"{cohort}_{shard}"
                    args = train.parser().parse_args(["--stage", "personal", "--cohort", cohort, "--shard", str(shard),
                        "--plan", str(path), "--population-run", str(root / "population"),
                        "--validation-run", str(root / "validation_score"), "--output", str(directory),
                        "--synthetic", "--device", "cpu", "--workers", "0"])
                    result = train.personal(args)
                    self.assertEqual(len(result["prediction_files"]), 2)
                    personal_runs.append(directory)
                args = argparse.Namespace(plan=path, population_run=root / "population", personal_runs=personal_runs,
                                          cohort=cohort, output=root / f"{cohort}_score", synthetic=True)
                receipt = evaluate.run(args)
                self.assertTrue(receipt["all_predictions_frozen_before_targets"])
                self.assertEqual(len(pd.read_csv(args.output / "participant_macro.csv")), 6)
                self.assertNotIn("400 source windows", (args.output / "RESULT_TABLES.md").read_text())


if __name__ == "__main__":
    unittest.main()
