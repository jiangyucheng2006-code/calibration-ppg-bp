"""Package aggregate enrollment-budget results and render English reports.

Inputs are previously audited local aggregate CSV/JSON files, not waveforms or
individual predictions. Usage: python scripts/report_enrollment_budget.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "local_archive/enrollment_budget_v1_20260911"
DEST = ROOT / "results/enrollment_budget_v1_20260911"
BUDGETS = list(range(20, 91, 10))
METHODS = {"new_person_lora": "LoRA", "new_person_lora_memory": "LoRA + memory"}
SCOPES = ("Overall", "MIMIC", "VitalDB")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |", *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def main():
    audit = json.loads((STAGE / "saved_prediction_audit.json").read_text())
    assert audit["status"] == "pass" and audit["work_nas_aggregate_identity"]
    for relative, expected in audit["artifact_sha256"].items():
        path = STAGE / "budget_summary.csv" if relative.startswith("prepare/") else STAGE / "final_public_source" / relative.removeprefix("test_score/")
        assert sha(path) == expected, relative
    macro = pd.read_csv(STAGE / "final_public_source/all_budgets_participant_macro.csv")
    diag = pd.read_csv(STAGE / "final_public_source/all_budgets_diagnostics.csv")
    ci = pd.read_csv(STAGE / "final_public_source/budget_paired_intervals.csv")
    counts = pd.read_csv(STAGE / "budget_summary.csv")
    execution = json.loads((STAGE / "execution_receipt.json").read_text())
    current = json.loads((STAGE / "final_public_source/evaluation_receipt.json").read_text())
    parent_dir = ROOT / "results/full_cohort_enrollment_v1_20260910/test"
    parent = json.loads((parent_dir / "evaluation_receipt.json").read_text())
    assert current["checkpoint_sha256"] == parent["checkpoint_sha256"]
    assert current["query_ids_sha256"] == parent["query_ids_sha256"]
    for method in METHODS:
        name = f"{method}_frozen_predictions.parquet"
        assert current["prediction_files"][f"p90_{name}"] == parent["prediction_files"][name]
    previous = pd.read_csv(parent_dir / "participant_macro.csv")
    check_columns = ["Setting", "Scope", "n_participants", "n_events", "sbp_mae", "dbp_mae", "mean_mae"]
    pd.testing.assert_frame_equal(macro.loc[macro.budget_percent.eq(90), check_columns].reset_index(drop=True), previous[check_columns].reset_index(drop=True))

    DEST.mkdir(parents=True, exist_ok=True)
    for source_name, public_name in [("final_public_source", "test"), ("validation_public_source", "validation")]:
        source = STAGE / source_name
        receipt = json.loads((source / "evaluation_receipt.json").read_text())
        assert receipt["status"] == "complete"
        assert receipt["all_sixteen_predictions_frozen_before_targets"]
        assert receipt["plan_sha256"] == current["plan_sha256"]
        assert receipt["checkpoint_sha256"] == current["checkpoint_sha256"]
        for file in source.rglob("*"):
            if file.is_file():
                assert file.suffix in (".csv", ".json"), file.name
                target = DEST / public_name / file.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
    for filename in ("budget_summary.csv", "saved_prediction_audit.json", "execution_receipt.json"):
        shutil.copy2(STAGE / filename, DEST / filename)

    def row(budget, scope, method="new_person_lora_memory"):
        return macro.loc[macro.budget_percent.eq(budget) & macro.Scope.eq(scope) & macro.Setting.eq(method)].iloc[0]

    overall = []
    source_rows = []
    benefit = []
    budget_rows = []
    for p in BUDGETS:
        base, memory = row(p, "Overall", "new_person_lora"), row(p, "Overall")
        overall.append([f"{p}%", f"{base.sbp_mae:.4f}", f"{base.dbp_mae:.4f}", f"{memory.sbp_mae:.4f}", f"{memory.dbp_mae:.4f}", f"{base.mean_mae-memory.mean_mae:.4f}"])
        source_rows.append([f"{p}%", *[f"{row(p, s).sbp_mae:.4f} / {row(p, s).dbp_mae:.4f}" for s in SCOPES]])
        interval = pd.read_csv(DEST / f"test/p{p}/paired_participant_intervals.csv")
        interval = interval.loc[interval.Scope.eq("Overall") & interval.BP.eq("Mean")].iloc[0]
        pc = next(r for r in audit["participant_memory_changes"] if r["budget_percent"] == p)
        benefit.append([f"{p}%", f"{interval.gain_mmHg:.4f}", f"[{interval.ci95_low_mmHg:.4f}, {interval.ci95_high_mmHg:.4f}]", pc["memory_better"], pc["tied"], pc["memory_worse"]])
        c = counts.loc[counts.cohort.eq("test") & counts.budget_percent.eq(p)].set_index("source")
        budget_rows.append([f"{p}%", f"{int(c.registration_windows.sum()):,}", int(c.loc["MIMIC", "median_windows"]), int(c.loc["VitalDB", "median_windows"]), int(c.zero_adapter_fallbacks.sum()), "78,237"])

    contrasts = ci.loc[ci.primary_family].sort_values("budget_percent")
    contrast_rows = [[f"{r.budget_percent}%", f"{r.MAE_increase_vs_90:.4f}", f"[{r.pointwise_ci95_low:.4f}, {r.pointwise_ci95_high:.4f}]", f"[{r.simultaneous_ci95_low:.4f}, {r.simultaneous_ci95_high:.4f}]"] for r in contrasts.itertuples()]
    requested = ["Setting", "BP", "MAE", "R²", "ME", "STD", "≤5 mmHg", "≤10 mmHg", "≤15 mmHg", "AAMI", "BHS"]
    def diagnostic_rows(frame):
        records = []
        for r in frame.to_dict("records"):
            values = {**r, "Setting": f"{METHODS[r['Setting']]} {r['budget_percent']}%"}
            records.append([f"{values[k]:.4f}" if isinstance(values[k], (int, float)) else str(values[k]) for k in requested])
        return records

    for scope in SCOPES:
        sub = diag.loc[diag.Scope.eq(scope)].sort_values(["budget_percent", "Setting", "BP"], key=lambda s: s.map({"SBP": 0, "DBP": 1}) if s.name == "BP" else s)
        psub = macro.loc[macro.Scope.eq(scope)].sort_values(["budget_percent", "Setting"])
        primary = [[f"{r.budget_percent}%", METHODS[r.Setting], f"{r.sbp_mae:.4f}", f"{r.dbp_mae:.4f}", f"{r.mean_mae:.4f}"] for r in psub.itertuples()]
        text = f"# {scope}: all enrollment-budget results\n\n[Main report](README.md) · [Primary CSV](test/{scope}_all_budgets_participant_macro.csv) · [Diagnostic CSV](test/{scope}_all_budgets_diagnostics.csv)\n\nAll eight budgets and both methods use the same people and query windows.\n\n## Primary participant-macro MAE (mmHg)\n\n" + table(["Budget", "Setting", "SBP MAE", "DBP MAE", "Mean MAE"], primary)
        text += "\n\n## Requested full table: pooled-window diagnostics\n\nThese MAEs differ from the primary values because every window, rather than every person, receives equal weight. STD is the sample standard deviation of signed prediction-minus-reference errors (ddof=1), not the between-person SD of MAE. The three threshold columns are percentages.\n\n" + table(requested, diagnostic_rows(sub))
        text += "\n\n\\* Retrospective numerical screens only: AAMI-style |ME| <= 5 mmHg and error STD <= 8 mmHg; BHS-style cumulative percentage grades implemented in [calbased_metrics.py](../../src/pulsedb_fewshot/calbased_metrics.py). PASS is not clinical validation, device certification, or validation of a wrist monitor.\n"
        (DEST / f"{scope}_RESULT_TABLES.md").write_text(text, encoding="utf-8", newline="\n")

    ninety_tables = []
    for scope in SCOPES:
        ninety_tables.append(f"### {scope}\n\n" + table(requested, diagnostic_rows(diag.loc[diag.Scope.eq(scope) & diag.budget_percent.eq(90)])))
    validation = pd.read_csv(DEST / "validation/Overall_all_budgets_participant_macro.csv")
    validation_rows = []
    for p in BUDGETS:
        v = validation.loc[validation.budget_percent.eq(p)].set_index("Setting")
        validation_rows.append([f"{p}%", *[f"{v.loc[m,'sbp_mae']:.4f} / {v.loc[m,'dbp_mae']:.4f}" for m in METHODS]])

    report = f"""# Personal enrollment budget: completed results

Report date: 11 September 2026. Run: `enrollment-budget-v1_20260910-180706`.
All eight budgets, both methods and both evaluation cohorts are complete.
The final scorer finished at **17:07:47 China time** (09:07:47 UTC).
Jobs 1783–1818 all completed with exit 0:0; no new training was submitted while preparing this report.

## Main findings

- With LoRA + reference memory, final participant-macro SBP/DBP MAE falls from **4.5797/2.5934 mmHg at 20%** to **3.2081/1.8336 at 90%**. Mean MAE falls by 1.0657 mmHg (29.71%).
- The 70% and 80% conditions approach the 90% point estimate: mean-MAE gaps are **0.0625** and **0.0512 mmHg**. This is not a demonstrated equivalence or a validated minimum budget.
- The 30% condition has higher observed error than 90%: its mean-MAE gap is **0.7332 mmHg**, with simultaneous 95% interval [0.5131, 0.9533]. The corresponding 50% gap is 0.3544 [0.1344, 0.5745]. These results do not support claiming equal accuracy.
- Memory lowers SBP, DBP and mean MAE at every budget in Overall, MIMIC and VitalDB. At 90%, 626 of 762 people improve, 97 tie and 39 worsen in mean MAE. Improvement is not universal.
- The two 90% prediction files are **byte-identical to the previously reported full-cohort 90% files**. This reproduces the reference condition; it is not an additional independent replication.

Full requested tables: **[Overall](Overall_RESULT_TABLES.md)** · **[MIMIC](MIMIC_RESULT_TABLES.md)** · **[VitalDB](VitalDB_RESULT_TABLES.md)**. Each includes all eight budgets, both methods, MAE, R², ME, STD, error percentages and AAMI/BHS numerical screens.

## 1. What was held fixed

The population model was trained on 3,750 people. The 763 validation and 762 final-evaluation people were absent from population fitting; the three outer subject sets are disjoint. This is the [full-cohort subject-disjoint enrollment protocol](../../docs/PLAN_FULL_COHORT_ENROLLMENT_V1.md), not the historical seen-user or exact official CalBased split.

Each new person supplies labeled enrollment windows for their own BP anchor, rank-4 LoRA parameters and reference memory. Only that allowed subset can be used for personal epoch selection and fresh full-enrollment refitting. The shared ResNet remains frozen. Approximately 10% of each person's eligible windows remain fixed query windows at every budget; other unused windows are not silently added to enrollment or scored as new queries.

| Role/view | People | Fixed query windows |
| --- | ---: | ---: |
| Validation | 763 | 76,909 |
| Final Overall | 762 | 78,237 |
| Final MIMIC | 343 | 54,664 |
| Final VitalDB | 419 | 23,573 |

Budgets are nominal percentages of all eligible personal windows, with label-blind nested complete content/interval groups. The within-person partition is random/content-grouped, **not chronological**. A 20% condition is not 20% of the 90% bank. Group rounding and short-history fallbacks are reported below.

Each budget starts a fresh personal fit from the same verified shared checkpoint, with the same per-person seed, optimizer settings and patience of eight non-improving epochs. No larger-budget adapter is reused. The two methods share the same fitted personal adapter; memory adds retrieval from that person's permitted enrollment bank. Its five nearest donors are not a five-cuff calibration budget.

The shared model was originally selected using the parent's 90%-enrollment validation rule. It is held fixed here, not separately optimized for each smaller budget. All sixteen predictions within a cohort were frozen before the scorer opened query targets. No scored result was fed back into this batch's fitting or budget selection.

## 2. Primary results: every person receives equal weight

MAE units are mmHg. Mean MAE is the average of participant-macro SBP and DBP MAE. Positive memory gain means lower error than the paired LoRA.

{table(["Enrollment", "LoRA SBP", "LoRA DBP", "LoRA + memory SBP", "LoRA + memory DBP", "Memory mean-MAE gain"], overall)}

### LoRA + memory: source-specific results

Each cell is participant-macro **SBP / DBP MAE**, in mmHg. Overall is recomputed over all 762 people, not the arithmetic average of the two source summaries. MIMIC and VitalDB here are internal PulseDB strata, not independent external validation cohorts.

{table(["Enrollment", "Overall", "MIMIC", "VitalDB"], source_rows)}

The Overall memory curve improves at each step, but not every subgroup metric is monotonic: MIMIC SBP is 3.2155 at 80% versus 3.2322 at 90%, while VitalDB SBP is 3.2792 at 70% versus 3.3204 at 80%. Across individuals, 718 improve and 44 worsen from 20% to 90% in memory-model mean MAE. Thus the aggregate curve cannot guarantee improvement for every user or every update.

## 3. How close are smaller budgets to 90%?

The planned paired bootstrap resamples people within PulseDB source strata: 2,000 replicates, seed 20260911, conditional on this shared fit and one nested enrollment ordering. Positive differences mean the smaller budget has higher mean MAE.

{table(["Enrollment", "Mean-MAE increase vs 90%", "Pointwise 95% interval", "Simultaneous 95% interval"], contrast_rows)}

The simultaneous intervals jointly cover the seven prespecified Overall mean-MAE contrasts for LoRA + memory, using a bootstrap maximum absolute centered deviation. The 60% pointwise interval excludes zero, but its simultaneous interval does not; these are different inferential statements. The 70% and 80% intervals include zero. **This does not establish equivalence**: no non-inferiority margin or clinical acceptable-loss threshold was prespecified.

Descriptively, most of the average improvement has occurred by 70%; 50% trades additional error for a smaller bank. These observations can motivate a separately evaluated budget policy, but this final cohort must not be reused to select and confirm that policy. The entire eight-budget curve remains the reported result.

## 4. Does personal memory still contribute?

These are pointwise exploratory, source-stratified participant-bootstrap intervals (2,000 replicates, seed 20260909), not a joint multiple-testing claim. Positive gain is paired LoRA MAE minus memory MAE, averaged over SBP and DBP.

{table(["Enrollment", "Mean-MAE gain", "Pointwise 95% interval", "People improved", "Tied", "Worsened"], benefit)}

All people and query windows remain in these comparisons, including people for whom memory or a small-bank fallback does not improve the prediction. There is no removal of the worst 30% in the headline results. Saved tail-error columns are explicitly retrospective oracle diagnostics, not a deployable rejection rule.

## 5. Requested full diagnostic table: 90% reference

These are **pooled-window** diagnostics, not the participant-macro results above. Every query window receives equal weight. ME = prediction minus reference; STD is the sample SD of signed errors (ddof=1). The three threshold columns are percentages. Consequently, the Overall pooled SBP MAE of 2.7235 is not a new improvement over the primary 3.2081 value: they summarize the same predictions with different weights.

{chr(10).join(ninety_tables)}

All 96 saved diagnostic rows (8 budgets × 2 methods × 3 views × 2 BP targets) meet the implemented AAMI-style numerical screen and BHS-style Grade A percentage cutoffs. **PASS* means only a retrospective numerical screen.** The implementation uses |ME| ≤ 5 mmHg and error STD ≤ 8 mmHg, and Grade A cumulative percentages of at least 60%, 85% and 95% within 5, 10 and 15 mmHg. These are code-level diagnostic rules, not evidence of completing a clinical validation protocol.

In particular, passing these calculations on hospital PPG windows is not wrist-device certification. [ISO 81060-3:2022](https://www.iso.org/standard/71161.html) specifies clinical-investigation requirements for continuous automated non-invasive devices. [BIHS validation guidance](https://bihs.org.uk/blood_pressure_technology/bp_monitor_validations.aspx) also distinguishes its conventional cuff-monitor validation process from wearable endorsement. Neither process was performed here.

## 6. Actual enrollment burden

Counts are labeled 10-s public-data windows, not independent cuff measurements. All final query counts remain fixed.

{table(["Nominal enrollment", "Total bank windows", "MIMIC median/person", "VitalDB median/person", "Zero-adapter fallback people", "Queries"], budget_rows)}

Even at 20%, median bank sizes are 207 MIMIC and 100 VitalDB windows. For histories with only one eligible allocation group, the prespecified method performs zero adapter optimization steps rather than selecting epochs using query labels. Those people and their queries remain included. At 70% and above this particular fallback is no longer needed. Counts, rounding and fallback coverage belong to the protocol; the budget curve does not isolate sample count from every consequence of the small-bank fitting rule.

[Complete budget count/fraction audit](budget_summary.csv) includes validation and both final source strata. The 70% bank uses approximately 22% fewer labeled windows than the 90% bank; this is not a demonstrated reduction of 22% in real cuff visits or study duration.

## 7. Validation cohort (supporting, not a second final test)

Participant-macro SBP / DBP MAE. The validation people helped select the shared parent checkpoint, so their outcomes are development results, not independent confirmation.

{table(["Enrollment", "LoRA", "LoRA + memory"], validation_rows)}

[Validation aggregate files](validation) preserve all source views, diagnostics, frozen-method evidence and budget intervals. No arm was discarded after validation scoring.

## 8. Verification and interpretation limits

- A separate read-only audit recomputed all 48 primary summary rows, all 96 full diagnostic rows and the seven primary paired budget intervals from saved final predictions. Maximum absolute numeric discrepancy was {audit['maximum_absolute_numeric_difference']:.3g}.
- All sixteen final prediction sets have identical query keys and targets, matching frozen-prediction hashes/values. Counts are unchanged; no new score-based exclusion was introduced. Final aggregate files match the work/NAS archive by SHA256.
- The 90% prediction hashes and primary metrics exactly match the previous completed parent run. This reference is repeated, not a newly independent test. The budget batch also reuses the parent's already reported final cohort, and is an exploratory dose-response study rather than a newly untouched confirmation cohort.
- No population or personal model was retrained during result publication. This audit is saved-prediction recomputation, not a multiple-training-seed replication.
- More randomly sampled labeled history is associated with lower average error in this controlled benchmark. It does not establish early-to-late prediction, cross-day durability, continuous online adaptation, motion/contact-pressure robustness, or transfer to MAXREFDES104 wrist data.
- The eight budgets compare accumulated-history registration, not the archived K=1/2/3/5 few-event goal. Public reference labels are ABP-derived; no fixed conversion to a number of cuff readings is warranted.

**Working decision:** retain personal LoRA + reference memory as the candidate method and preserve the complete budget curve. A lower-burden operating point and chronological predict-before-update behavior need separate validation, rather than another module being declared necessary from these scores. No new jobs are submitted in this publication step.

## Files and reproducibility

- [Frozen experiment plan](../../docs/ENROLLMENT_BUDGET_PLAN_20260911.md) and [execution history](../../docs/ENROLLMENT_BUDGET_RUN_20260911.md).
- [Primary aggregate CSV](test/all_budgets_participant_macro.csv), [full diagnostic CSV](test/all_budgets_diagnostics.csv), [all paired budget contrasts](test/budget_paired_intervals.csv).
- [Scoring receipt](test/evaluation_receipt.json), [freeze receipt](test/frozen_predictions.json), [uncertainty contract](test/budget_uncertainty_contract.json).
- [Read-only prediction audit](saved_prediction_audit.json), [job accounting](execution_receipt.json), [statistical review notes](REVIEW_NOTES.md).
- [Read-only audit program](../../scripts/audit_enrollment_budget_results.py) and [aggregate report builder](../../scripts/report_enrollment_budget.py).

Training snapshot: `2b5dd222b7fb469a9f02853b2ab5edf4fc11de64`. Plan SHA256: `{current['plan_sha256']}`. Shared-checkpoint SHA256: `{current['checkpoint_sha256']}`. Fixed-query SHA256: `{current['query_ids_sha256']}`.

Only aggregate tables, checksums, protocols and reporting code are published. Raw waveforms, individual predictions, target labels, personal parameters and reference banks remain private.
"""
    (DEST / "README.md").write_text(report, encoding="utf-8", newline="\n")
    manifest = {"protocol_id": current["protocol_id"], "all_sixteen_final_arms_included": True, "parent_ninety_prediction_hashes_identical": True, "parent_ninety_primary_metrics_identical": True, "copied_final_files_match_read_only_audit": True, "new_training_submitted": False, "artifact_sha256": {file.relative_to(DEST).as_posix(): sha(file) for file in sorted(DEST.rglob("*")) if file.is_file() and file.name != "publication_receipt.json"}}
    (DEST / "publication_receipt.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"report": str(DEST / "README.md"), "primary_rows": len(macro), "diagnostic_rows": len(diag), "artifact_files": len(manifest["artifact_sha256"]), "all_numeric_screens_pass": bool(diag.AAMI.eq("PASS*").all()), "all_bhs_grade_a": bool(diag.BHS.eq("PASS (Grade A)*").all()), "parent_90_exact_match": True}, indent=2))


if __name__ == "__main__":
    main()
