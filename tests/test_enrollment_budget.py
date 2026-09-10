"""Nested budgets, one-group fallback, and all-arm target-isolated execution."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_legacy_enrollment import fixture
from pulsedb_fewshot import enrollment_budget_protocol as budget
from pulsedb_fewshot.legacy_enrollment_protocol import FULL_CONFIG, FULL_PROTOCOL, build_assignment, assign_person
from pulsedb_fewshot.official_calbased_train import sha256, save_json


def synthetic_parent(root):
    """Small valid full-cohort contract; no real person, prior model or BP used."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    meta, old, waves, labels = fixture()
    # One tiny target person exercises all-budget coverage and zero-step fallback.
    tiny = old.loc[old.split.eq("meta_test"), "subject_uid"].iloc[0]
    remove = meta.loc[meta.subject_uid.eq(tiny)].index[3:]
    meta = meta.drop(index=remove)
    groups = {subject: i for i, subject in enumerate(sorted(meta.subject_uid.unique()))}
    meta["full_index_group"] = meta.subject_uid.map(groups)
    old.to_csv(root / "split.csv", index=False)
    store = root / "store"
    store.mkdir()
    np.save(store / "ppg.npy", waves)
    save_json(store / "manifest.json", {"synthetic": True})
    with pq.ParquetWriter(root / "index.parquet", pa.Table.from_pandas(labels.iloc[:1], preserve_index=False).schema) as writer:
        for person in sorted(groups):
            writer.write_table(pa.Table.from_pandas(labels.loc[labels.subject_uid.eq(person)].reset_index(drop=True), preserve_index=False))
    assigned, audit = build_assignment(meta, old, full_cohort=True)
    path = root / "parent"
    path.mkdir()
    parts = {"population_train": assigned.loc[assigned.split.eq("meta_train")].copy()}
    for cohort, role in (("validation", "meta_validation"), ("test", "meta_test")):
        own = assigned.loc[assigned.split.eq(role)]
        parts[f"{cohort}_registration"] = own.loc[own.personal_role.eq("registration")].copy()
        parts[f"{cohort}_inputs"] = own.loc[own.personal_role.eq("query")].copy()
    for name, frame in parts.items():
        if not name.endswith("inputs"):
            frame = frame.merge(labels, on=budget.KEYS, validate="one_to_one")
        frame.to_parquet(path / f"{name}.parquet", index=False)
        audit[name] = budget.frame_audit(frame)
    parts["validation_inputs"][budget.KEYS].merge(labels, on=budget.KEYS).to_parquet(path / "validation_targets.parquet", index=False)
    plan = dict(protocol_id=FULL_PROTOCOL, status="ready", synthetic=True, config=FULL_CONFIG,
        source_root=str(store), full_index=str(root / "index.parquet"), full_index_sha256=sha256(root / "index.parquet"),
        source_manifest_sha256=sha256(store / "manifest.json"), legacy_split_path=str(root / "split.csv"),
        legacy_split_sha256=sha256(root / "split.csv"), methods=budget.METHODS, audit=audit,
        all_original_subjects_accounted_for=True, original_subjects=12,
        validation_subjects=sorted(parts["validation_inputs"].subject_uid.unique()),
        test_subjects=sorted(parts["test_inputs"].subject_uid.unique()),
        files={p.name: sha256(p) for p in path.iterdir()})
    save_json(path / "plan.json", plan)
    save_json(path / "plan_digest.json", {"sha256": sha256(path / "plan.json")})
    return path / "plan.json"


class BudgetContractTests(unittest.TestCase):
    def bank(self, n=80):
        meta, old, _, _ = fixture()
        a = assign_person(meta.iloc[:n], full_cohort=True)
        return a.loc[a.personal_role.eq("registration")], int(a.personal_role.eq("query").sum())

    def test_eight_nested_budgets_and_exact_upper_roles(self):
        bank, queries = self.bank()
        selections, rows = budget.person_budgets(bank, queries)
        self.assertEqual(list(selections), list(range(20, 100, 10)))
        previous = set()
        for percent, frame in selections.items():
            self.assertEqual(len(frame), 80 * percent // 100)
            self.assertTrue(previous.issubset(set(frame.segment_uid)))
            previous = set(frame.segment_uid)
        a = selections[90].sort_values("segment_uid").reset_index(drop=True)
        b = bank[budget.KEYS + ["inner_role"]].sort_values("segment_uid").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual({r["query_windows"] for r in rows}, {8})

    def test_label_error_blind_and_input_order_invariant(self):
        bank, queries = self.bank()
        a, _ = budget.person_budgets(bank, queries)
        b, _ = budget.person_budgets(bank.sample(frac=1, random_state=9), queries)
        for p in a:
            pd.testing.assert_frame_equal(a[p], b[p])
        for column in ("sbp", "dbp", "pred_sbp", "previous_error"):
            with self.assertRaises(ValueError):
                budget.person_budgets(bank.assign(**{column: 999.}), queries)

    def test_tiny_person_retained_and_no_extra_labels_for_inner_validation(self):
        bank, queries = self.bank(n=3)
        frames, audits = budget.person_budgets(bank, queries)
        self.assertEqual(len(frames[20]), 1)
        self.assertTrue(audits[0]["zero_adapter_fallback"])
        self.assertAlmostEqual(audits[0]["actual_fraction"], 1/3)
        self.assertEqual(len(frames[90]), 2)

    def test_complete_duplicate_groups_are_indivisible(self):
        bank, queries = self.bank()
        bank = bank.copy().reset_index(drop=True)
        bank.loc[1, "ppg_content_sha256"] = bank.loc[0, "ppg_content_sha256"]
        # The synthetic duplicate could straddle old inner roles: choose a lower
        # budget to check the newly created complete-group assignment only.
        selections, _ = budget.person_budgets(bank, queries)
        pair = set(bank.iloc[:2].segment_uid)
        for p in budget.PERCENTAGES[:-1]:
            selected = selections[p]
            self.assertIn(len(set(selected.segment_uid) & pair), (0, 2))
            if pair.issubset(set(selected.segment_uid)):
                self.assertEqual(selected.loc[selected.segment_uid.isin(pair)].inner_role.nunique(), 1)


class BudgetPipelineTests(unittest.TestCase):
    def test_all_budgets_fresh_profiles_fixed_queries_and_scoring_gate(self):
        import torch
        from pulsedb_fewshot import legacy_enrollment_train as parent_train
        from pulsedb_fewshot import enrollment_budget_train as train
        from pulsedb_fewshot import enrollment_budget_evaluate as score
        from pulsedb_fewshot import full_enrollment_data as data
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_plan = synthetic_parent(root)
            args = parent_train.parser().parse_args(["--stage", "population", "--plan", str(parent_plan),
                "--output", str(root / "population"), "--synthetic", "--epochs", "1", "--device", "cpu", "--workers", "0"])
            parent_train.population(args)
            before = sha256(root / "population" / "best.pt")
            with patch.object(data, "read_grouped_targets", side_effect=AssertionError("prepare must not read target store")):
                budget.prepare(argparse.Namespace(parent_plan=parent_plan, population_run=root / "population",
                                                  output=root / "prepare", synthetic=True))
            plan = root / "prepare" / "plan.json"
            for p in budget.PERCENTAGES:
                _, checked = budget.partition(plan, "test", p, synthetic=True)
                self.assertTrue(checked.access_role.eq(checked.split + ":registration:" + checked.inner_role).all())
            def personal_args(cohort, p, shard):
                return train.parser().parse_args(["--stage", "personal", "--cohort", cohort, "--percent", str(p),
                    "--shard", str(shard), "--plan", str(plan), "--population-run", str(root / "population"),
                    "--validation-run", str(root / "validation_score"), "--output", str(root / f"{cohort}_p{p}_{shard}"),
                    "--synthetic", "--device", "cpu", "--workers", "0"])
            with self.assertRaises(FileNotFoundError):
                train.run(personal_args("test", 20, 0))
            for cohort in ("validation", "test"):
                for p in budget.PERCENTAGES:
                    for shard in (0, 1):
                        with patch.object(data, "read_grouped_targets", side_effect=AssertionError("fitter cannot read query BP")):
                            receipt = train.run(personal_args(cohort, p, shard))
                        self.assertFalse(receipt["larger_budget_adapter_used"])
                        self.assertFalse(receipt["unused_budget_labels_accessed"])
                        if cohort == "test" and p == 20 and shard == 0:
                            tiny = next(v for v in receipt["profiles"].values() if v["registration_rows"] == 1)
                            self.assertEqual(tiny["refit_optimizer_steps"], 0)
                            self.assertIsNotNone(tiny["selection_fallback"])
                original = data.read_grouped_targets
                def checked_read(index, keys):
                    frozen = json.loads((root / f"{cohort}_score" / "frozen_predictions.json").read_text())
                    self.assertEqual(len(frozen["prediction_files"]), 16)
                    self.assertTrue(frozen["all_sixteen_predictions_frozen_before_targets"])
                    return original(index, keys)
                with patch.object(data, "read_grouped_targets", side_effect=checked_read):
                    receipt = score.run(argparse.Namespace(plan=plan, run_root=root, cohort=cohort,
                        output=root / f"{cohort}_score", synthetic=True))
                self.assertEqual(receipt["percentages"], budget.PERCENTAGES)
                table = pd.read_csv(root / f"{cohort}_score" / "all_budgets_participant_macro.csv")
                self.assertEqual(len(table), 8 * 2 * 3)
                bands = pd.read_csv(root / f"{cohort}_score" / "budget_paired_intervals.csv")
                self.assertEqual(int(bands.primary_family.sum()), 7)
            self.assertEqual(before, sha256(root / "population" / "best.pt"))
            # Mutation is rejected without weakening any older protocol gate.
            text = plan.read_text()
            plan.write_text(text + " ")
            with self.assertRaisesRegex(ValueError, "plan changed"):
                budget.load_plan(plan, synthetic=True)


if __name__ == "__main__":
    unittest.main()
