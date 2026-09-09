"""New-subject exclusion, fresh adapter, frozen shared state and target-access tests."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pulsedb_fewshot.post_enrollment_protocol import select_people, audit_split, linked_quarantine, SEED, PROTOCOL, EXPANDED_PROTOCOL


def frames(people=8):
    train, test, waveforms = [], [], []
    rng = np.random.default_rng(99)
    for p in range(people):
        source = "MIMIC" if p < people // 2 else "VitalDB"
        for i in range(80):
            wave = rng.normal(size=1250).astype(np.float32)
            row = {"subject_uid": f"{source}:person{p}", "segment_uid": f"p{p}:w{i}",
                "source": source, "waveform_file": "ppg.npy", "waveform_row": len(waveforms),
                "ppg_content_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                "record_id": f"record{p}", "start_time_s": i * 20., "end_time_s": i * 20. + 9.992,
                "duration_s": 10., "sample_interval_s": .008}
            waveforms.append(wave)
            if i < 72:
                row.update(inner_role="train" if i < 64 else "internal_validation",
                    inner_fold=i % 3 if i < 64 else -1, sbp=110. + p * 3 + i * .2,
                    dbp=65. + p * 2 + i * .1)
                train.append(row)
            else:
                test.append(row)
    return pd.DataFrame(train), pd.DataFrame(test), np.array(waveforms)


class ProtocolTests(unittest.TestCase):
    def test_selection_is_repeatable_and_order_independent(self):
        train, _, _ = frames()
        chosen = select_people(train, per_source=1)
        self.assertEqual(chosen, select_people(train.sample(frac=1, random_state=14), per_source=1))
        self.assertEqual(len(chosen), 2)

    def test_selection_does_not_use_labels_or_error(self):
        train, _, _ = frames()
        chosen = select_people(train, per_source=1)
        changed = train.copy()
        changed[["sbp", "dbp"]] = np.nan
        changed["previous_error"] = -np.inf
        self.assertEqual(chosen, select_people(changed, per_source=1))

    def test_whole_people_removed_from_all_population_roles(self):
        train, test, _ = frames()
        selected = select_people(train, per_source=1)
        parts, audit = audit_split(train, test, selected, synthetic=True)
        for name in ("population_train", "population_test_inputs"):
            self.assertFalse(set(parts[name].subject_uid) & set(selected))
        for name in ("enrollment_train", "enrollment_test_inputs"):
            self.assertEqual(set(parts[name].subject_uid), set(selected))
        self.assertEqual(sum(len(v) for v in parts.values()), len(train) + len(test))
        self.assertEqual(audit["cross_cohort_subject_overlap"], 0)

    def test_cross_cohort_content_rejected(self):
        train, test, _ = frames()
        selected = select_people(train, per_source=1)
        i = train.loc[train.subject_uid.isin(selected)].index[0]
        j = train.loc[~train.subject_uid.isin(selected)].index[0]
        train.loc[i, "ppg_content_sha256"] = train.loc[j, "ppg_content_sha256"]
        with self.assertRaisesRegex(ValueError, "content crosses"):
            audit_split(train, test, selected, synthetic=True)

    def test_test_labels_cannot_enter_inputs(self):
        train, test, _ = frames()
        test["target_sbp"] = 123.
        with self.assertRaisesRegex(ValueError, "reference labels"):
            audit_split(train, test, select_people(train, per_source=1), synthetic=True)

    def test_duplicate_row_rejected(self):
        train, test, _ = frames()
        test.loc[0, "segment_uid"] = train.iloc[0].segment_uid
        with self.assertRaisesRegex(ValueError, "identities overlap"):
            audit_split(train, test, select_people(train, per_source=1), synthetic=True)

    def test_formal_counts_not_relaxed_by_new_protocol(self):
        train, test, _ = frames()
        with self.assertRaisesRegex(ValueError, "30 of 2506"):
            audit_split(train, test, select_people(train, per_source=1))
        with self.assertRaisesRegex(ValueError, "200 of 2506"):
            audit_split(train, test, select_people(train, per_source=1), protocol_id=EXPANDED_PROTOCOL)

    def test_expanded_selection_keeps_exactly_100_per_source(self):
        identities = pd.DataFrame([
            {"source": source, "subject_uid": f"{source}:{i:05d}"}
            for source, size in (("MIMIC", 1213), ("VitalDB", 1293)) for i in range(size)])
        selected = select_people(identities, per_source=100)
        self.assertEqual(len(selected), 200)
        self.assertEqual(identities.loc[identities.subject_uid.isin(selected)].groupby("source").size().to_dict(), {"MIMIC": 100, "VitalDB": 100})
        self.assertEqual(selected, select_people(identities.sample(frac=1, random_state=8), per_source=100))

    def test_unknown_protocol_is_not_a_leakage_bypass(self):
        train, test, _ = frames()
        with self.assertRaisesRegex(ValueError, "unknown enrollment protocol"):
            audit_split(train, test, select_people(train, per_source=1), synthetic=True, protocol_id="anything")

    def test_linked_identity_is_quarantined_without_resampling(self):
        train, test, _ = frames()
        selected = select_people(train, per_source=2)
        i = train.loc[train.subject_uid.isin(selected)].index[0]
        j = train.loc[~train.subject_uid.isin(selected)].index[0]
        linked_person = train.loc[j, "subject_uid"]
        train.loc[i, "ppg_content_sha256"] = train.loc[j, "ppg_content_sha256"]
        parts, audit = audit_split(train, test, selected, synthetic=True, protocol_id=EXPANDED_PROTOCOL)
        self.assertEqual(audit["quarantined_subjects"], [linked_person])
        self.assertEqual(audit["quarantine_rows"], 80)
        for part in parts.values():
            self.assertNotIn(linked_person, set(part.subject_uid))
        self.assertEqual(set(parts["enrollment_train"].subject_uid), set(selected))
        self.assertEqual(set(parts["enrollment_test_inputs"].subject_uid), set(selected))
        self.assertEqual(sum(len(p) for p in parts.values()) + 80, len(train) + len(test))
        self.assertEqual(audit["cross_cohort_content_overlap"], 0)

    def test_quarantine_follows_transitive_content_links(self):
        metadata = pd.DataFrame({"subject_uid": ["a", "b", "b", "c", "d"],
                                 "ppg_content_sha256": ["h1", "h1", "h2", "h2", "h3"]})
        self.assertEqual(linked_quarantine(metadata, ["a"]), ["b", "c"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch optional on local contract-only runtime")
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_fresh_person_does_not_copy_old_adapters(self):
        import torch
        from pulsedb_fewshot.lora_prs_models import LoraPRSRegressor, PRS_MODELS
        from pulsedb_fewshot.post_enrollment_personal import fresh_person_model
        from pulsedb_fewshot.post_enrollment_population import shared_digest
        old = LoraPRSRegressor(PRS_MODELS["lora_continue"], subject_count=4)
        with torch.no_grad():
            old.base.lora_a.weight.fill_(100)
            old.base.lora_b.weight.fill_(200)
        new = fresh_person_model(old.state_dict(), SEED, "cpu")
        self.assertEqual(torch.count_nonzero(new.base.lora_b.weight).item(), 0)
        self.assertLess(new.base.lora_a.weight.abs().max().item(), .1)
        self.assertEqual(shared_digest(old.state_dict()), shared_digest(new.state_dict()))
        self.assertEqual(sum(p.numel() for p in new.parameters() if p.requires_grad), 2048)

    def test_cached_feature_forward_equals_real_ppg_forward(self):
        import torch
        from pulsedb_fewshot.lora_prs_models import LoraPRSRegressor, PRS_MODELS
        from pulsedb_fewshot.post_enrollment_personal import feature_forward
        model = LoraPRSRegressor(PRS_MODELS["lora_continue"], subject_count=1).eval()
        with torch.no_grad():
            model.base.lora_b.weight.normal_(0, .05)
            x, anchor = torch.randn(5, 1, 1250), torch.randn(5, 2)
            actual = model(x, anchor, subject_index=torch.zeros(5, dtype=torch.long))
            cached = feature_forward(model, model.base.encoder(x), anchor)[0]
        torch.testing.assert_close(actual, cached, atol=1e-6, rtol=1e-6)

    def test_personal_training_does_not_change_shared_state(self):
        import torch
        from pulsedb_fewshot.lora_prs_models import LoraPRSRegressor, PRS_MODELS
        from pulsedb_fewshot.post_enrollment_personal import fresh_person_model, fit_adapter
        from pulsedb_fewshot.post_enrollment_population import shared_digest
        old = LoraPRSRegressor(PRS_MODELS["lora_continue"], subject_count=4)
        # The actual population head has been trained. A completely untrained
        # zero-output head gives the adapter zero gradients by construction.
        with torch.no_grad():
            for parameter in old.base.residual_head.parameters():
                parameter.normal_(0, .03)
        model = fresh_person_model(old.state_dict(), SEED, "cpu")
        before = shared_digest(model.state_dict())
        args = argparse.Namespace(device="cpu", learning_rate=3e-4, weight_decay=1e-4,
            huber_delta=.5, batch_size=8, gradient_clip=5., synthetic=True)
        z = np.random.default_rng(1).normal(size=(32, 256)).astype(np.float32)
        bp = np.tile([140., 90.], (32, 1)).astype(np.float32)
        fit_adapter(model, z, bp, [120., 80.], {"mean": [120., 80.], "std": [15., 10.]}, args, fixed_epochs=2)
        self.assertEqual(before, shared_digest(model.state_dict()))
        self.assertGreater(torch.count_nonzero(model.base.lora_b.weight).item(), 0)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("pyarrow"), "full server dependencies required")
class PipelineSmoke(unittest.TestCase):
    def test_end_to_end_registration_freezing_and_scoring(self):
        self.exercise_pipeline(PROTOCOL)

    def test_expanded_ablations_freezing_and_scoring(self):
        self.exercise_pipeline(EXPANDED_PROTOCOL)

    def exercise_pipeline(self, protocol_id):
        import torch
        from pulsedb_fewshot import post_enrollment_protocol as protocol
        from pulsedb_fewshot import post_enrollment_population as population
        from pulsedb_fewshot import post_enrollment_personal as personal
        from pulsedb_fewshot import post_enrollment_evaluate as evaluate
        from pulsedb_fewshot.official_calbased_train import save_json
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "store"
            store.mkdir()
            train, test, waves = frames()
            train.to_parquet(store / "train_manifest.parquet", index=False)
            test.to_parquet(store / "test_inputs.parquet", index=False)
            np.save(store / "ppg.npy", waves)
            save_json(store / "manifest.json", {"protocol_id": protocol.SOURCE_PROTOCOL, "status": "ready", "synthetic": True})
            prepared = protocol.prepare(store, root / "partitions", synthetic=True, protocol_id=protocol_id)
            plan = root / "partitions" / "plan.json"
            if protocol_id == EXPANDED_PROTOCOL:
                tampered = dict(prepared, methods=list(protocol.METHODS))
                changed_plan = root / "changed_plan.json"
                save_json(changed_plan, tampered)
                with self.assertRaisesRegex(ValueError, "candidate configuration changed"):
                    protocol.load_plan(changed_plan, synthetic=True)
            args = population.parser().parse_args(["--plan", str(plan), "--output", str(root / "inner"),
                "--stage", "inner", "--device", "cpu", "--synthetic", "--epochs", "1", "--workers", "0"])
            seen_reads = []
            original_read = pd.read_parquet
            def guarded(path, *a, **kw):
                seen_reads.append(str(path))
                if Path(path).name.startswith("enrollment") or "full_index" in str(path):
                    raise AssertionError("population trainer accessed enrollment or test targets")
                return original_read(path, *a, **kw)
            with patch.object(pd, "read_parquet", side_effect=guarded):
                population.run(args)
            self.assertFalse(any("enrollment" in Path(p).name for p in seen_reads))
            args.output, args.stage, args.selection_run = root / "final", "final", root / "inner"
            with patch.object(pd, "read_parquet", side_effect=guarded):
                population.run(args)
            runs = []
            for shard in (0, 1):
                output = root / f"personal{shard}"
                runs.append(output)
                pa = personal.parser().parse_args(["--plan", str(plan), "--population-run", str(root / "final"),
                    "--output", str(output), "--shard", str(shard), "--device", "cpu", "--synthetic"])
                def personal_guard(path, *a, **kw):
                    if "full_index" in str(path) or "scored" in str(path):
                        raise AssertionError("personal trainer read test BP")
                    return original_read(path, *a, **kw)
                with patch.object(pd, "read_parquet", side_effect=personal_guard):
                    report = personal.run(pa)
                self.assertEqual(report["status"], "complete")
                self.assertFalse(report["test_targets_accessed"])
                self.assertTrue(all(v["reload_equivalence"] for v in report["profiles"].values()))
                self.assertEqual(len(report["prediction_files"]), len(prepared["methods"]))
                if protocol_id == EXPANDED_PROTOCOL:
                    self.assertTrue(all(v["all_ablation_reload_equivalence"] for v in report["profiles"].values()))
            labels = test[["subject_uid", "segment_uid", "source"]].copy()
            labels["sbp"], labels["dbp"] = 125., 75.
            index = root / "full_index.parquet"
            labels.to_parquet(index, index=False)
            ea = evaluate.parser().parse_args(["--stage", "enrollment", "--plan", str(plan),
                "--population-run", str(root / "final"), "--personal-runs", *map(str, runs),
                "--full-index", str(index), "--output", str(root / "score"), "--device", "cpu", "--synthetic"])
            evaluate.enrollment_score(ea)
            receipt = json.loads((root / "score" / "evaluation_receipt.json").read_text())
            self.assertEqual(receipt["subjects"], len(prepared["selected_subjects"]))
            self.assertEqual(receipt["windows"], len(prepared["selected_subjects"]) * 8)
            self.assertTrue(receipt["all_methods_frozen_before_targets"])
            self.assertEqual(receipt["method_count"], len(prepared["methods"]))
            if protocol_id == PROTOCOL:
                self.assertTrue(receipt["all_four_methods_frozen_before_targets"])
            else:
                intervals = pd.read_csv(root / "score" / "paired_participant_intervals.csv")
                self.assertEqual(len(intervals), 8 * 3 * 3)
                self.assertEqual(intervals.primary_contrast.sum(), 1)
                self.assertTrue(intervals.loc[intervals.Scope.eq("Overall"), "participants"].eq(4).all())
            # Corrupting a profile or a prediction invalidates evaluation rather
            # than allowing a silent re-score of altered model outputs.
            bad = runs[0] / "new_person_lora_predictions.parquet"
            altered = original_read(bad)
            altered["pred_sbp"] = 999.
            altered.to_parquet(bad, index=False)
            ea.output = root / "must_not_score"
            with self.assertRaisesRegex(ValueError, "predictions changed"):
                evaluate.enrollment_score(ea)
            self.assertFalse(ea.output.exists())


if __name__ == "__main__":
    unittest.main()
