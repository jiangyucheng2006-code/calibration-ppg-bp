# Personal enrollment budget: completed results

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

| Enrollment | LoRA SBP | LoRA DBP | LoRA + memory SBP | LoRA + memory DBP | Memory mean-MAE gain |
| --- | --- | --- | --- | --- | --- |
| 20% | 5.2789 | 2.9869 | 4.5797 | 2.5934 | 0.5464 |
| 30% | 4.8066 | 2.7485 | 4.1265 | 2.3816 | 0.5235 |
| 40% | 4.5747 | 2.5739 | 3.9305 | 2.2189 | 0.4996 |
| 50% | 4.3334 | 2.4396 | 3.6620 | 2.0886 | 0.5112 |
| 60% | 4.1006 | 2.3282 | 3.3956 | 1.9613 | 0.5359 |
| 70% | 3.9677 | 2.2341 | 3.2850 | 1.8817 | 0.5176 |
| 80% | 3.9741 | 2.2235 | 3.2732 | 1.8709 | 0.5267 |
| 90% | 3.8556 | 2.1661 | 3.2081 | 1.8336 | 0.4900 |

### LoRA + memory: source-specific results

Each cell is participant-macro **SBP / DBP MAE**, in mmHg. Overall is recomputed over all 762 people, not the arithmetic average of the two source summaries. MIMIC and VitalDB here are internal PulseDB strata, not independent external validation cohorts.

| Enrollment | Overall | MIMIC | VitalDB |
| --- | --- | --- | --- |
| 20% | 4.5797 / 2.5934 | 4.2675 / 2.4766 | 4.8352 / 2.6891 |
| 30% | 4.1265 / 2.3816 | 3.9052 / 2.2802 | 4.3077 / 2.4646 |
| 40% | 3.9305 / 2.2189 | 3.7642 / 2.1036 | 4.0666 / 2.3134 |
| 50% | 3.6620 / 2.0886 | 3.5346 / 1.9994 | 3.7664 / 2.1616 |
| 60% | 3.3956 / 1.9613 | 3.3594 / 1.9517 | 3.4253 / 1.9692 |
| 70% | 3.2850 / 1.8817 | 3.2920 / 1.8614 | 3.2792 / 1.8983 |
| 80% | 3.2732 / 1.8709 | 3.2155 / 1.8175 | 3.3204 / 1.9146 |
| 90% | 3.2081 / 1.8336 | 3.2322 / 1.7896 | 3.1884 / 1.8695 |

The Overall memory curve improves at each step, but not every subgroup metric is monotonic: MIMIC SBP is 3.2155 at 80% versus 3.2322 at 90%, while VitalDB SBP is 3.2792 at 70% versus 3.3204 at 80%. Across individuals, 718 improve and 44 worsen from 20% to 90% in memory-model mean MAE. Thus the aggregate curve cannot guarantee improvement for every user or every update.

## 3. How close are smaller budgets to 90%?

The planned paired bootstrap resamples people within PulseDB source strata: 2,000 replicates, seed 20260911, conditional on this shared fit and one nested enrollment ordering. Positive differences mean the smaller budget has higher mean MAE.

| Enrollment | Mean-MAE increase vs 90% | Pointwise 95% interval | Simultaneous 95% interval |
| --- | --- | --- | --- |
| 20% | 1.0657 | [0.8747, 1.2855] | [0.8456, 1.2858] |
| 30% | 0.7332 | [0.5493, 0.9472] | [0.5131, 0.9533] |
| 40% | 0.5539 | [0.3919, 0.7420] | [0.3338, 0.7739] |
| 50% | 0.3544 | [0.1960, 0.5439] | [0.1344, 0.5745] |
| 60% | 0.1576 | [0.0348, 0.2656] | [-0.0624, 0.3777] |
| 70% | 0.0625 | [-0.0361, 0.1448] | [-0.1576, 0.2825] |
| 80% | 0.0512 | [-0.0422, 0.1395] | [-0.1689, 0.2713] |

The simultaneous intervals jointly cover the seven prespecified Overall mean-MAE contrasts for LoRA + memory, using a bootstrap maximum absolute centered deviation. The 60% pointwise interval excludes zero, but its simultaneous interval does not; these are different inferential statements. The 70% and 80% intervals include zero. **This does not establish equivalence**: no non-inferiority margin or clinical acceptable-loss threshold was prespecified.

Descriptively, most of the average improvement has occurred by 70%; 50% trades additional error for a smaller bank. These observations can motivate a separately evaluated budget policy, but this final cohort must not be reused to select and confirm that policy. The entire eight-budget curve remains the reported result.

## 4. Does personal memory still contribute?

These are pointwise exploratory, source-stratified participant-bootstrap intervals (2,000 replicates, seed 20260909), not a joint multiple-testing claim. Positive gain is paired LoRA MAE minus memory MAE, averaged over SBP and DBP.

| Enrollment | Mean-MAE gain | Pointwise 95% interval | People improved | Tied | Worsened |
| --- | --- | --- | --- | --- | --- |
| 20% | 0.5464 | [0.4920, 0.6027] | 512 | 207 | 43 |
| 30% | 0.5235 | [0.4743, 0.5751] | 556 | 170 | 36 |
| 40% | 0.4996 | [0.4557, 0.5478] | 576 | 148 | 38 |
| 50% | 0.5112 | [0.4665, 0.5566] | 604 | 124 | 34 |
| 60% | 0.5359 | [0.4856, 0.5892] | 606 | 115 | 41 |
| 70% | 0.5176 | [0.4718, 0.5690] | 611 | 110 | 41 |
| 80% | 0.5267 | [0.4776, 0.5806] | 616 | 107 | 39 |
| 90% | 0.4900 | [0.4513, 0.5320] | 626 | 97 | 39 |

All people and query windows remain in these comparisons, including people for whom memory or a small-bank fallback does not improve the prediction. There is no removal of the worst 30% in the headline results. Saved tail-error columns are explicitly retrospective oracle diagnostics, not a deployable rejection rule.

## 5. Requested full diagnostic table: 90% reference

These are **pooled-window** diagnostics, not the participant-macro results above. Every query window receives equal weight. ME = prediction minus reference; STD is the sample SD of signed errors (ddof=1). The three threshold columns are percentages. Consequently, the Overall pooled SBP MAE of 2.7235 is not a new improvement over the primary 3.2081 value: they summarize the same predictions with different weights.

### Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA 90% | SBP | 3.4251 | 0.9409 | 0.0245 | 5.2910 | 79.2771 | 94.6675 | 98.1262 | PASS* | PASS (Grade A)* |
| LoRA 90% | DBP | 1.8949 | 0.9170 | -0.1212 | 3.6509 | 93.6168 | 98.5697 | 99.3724 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | SBP | 2.7235 | 0.9570 | -0.0559 | 4.5117 | 85.6104 | 96.2294 | 98.5812 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | DBP | 1.5091 | 0.9377 | -0.0970 | 3.1626 | 95.2401 | 98.8305 | 99.4964 | PASS* | PASS (Grade A)* |
### MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA 90% | SBP | 3.4744 | 0.9410 | 0.0763 | 5.3904 | 79.0100 | 94.3564 | 97.9255 | PASS* | PASS (Grade A)* |
| LoRA 90% | DBP | 1.9225 | 0.9154 | -0.1430 | 3.7870 | 93.4911 | 98.3024 | 99.1786 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | SBP | 2.7968 | 0.9568 | -0.0314 | 4.6120 | 85.1310 | 95.9077 | 98.4139 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | DBP | 1.5365 | 0.9386 | -0.0967 | 3.2254 | 94.9180 | 98.6188 | 99.3579 | PASS* | PASS (Grade A)* |
### VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA 90% | SBP | 3.3107 | 0.9300 | -0.0955 | 5.0510 | 79.8965 | 95.3888 | 98.5916 | PASS* | PASS (Grade A)* |
| LoRA 90% | DBP | 1.8309 | 0.9215 | -0.0706 | 3.3134 | 93.9083 | 99.1898 | 99.8218 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | SBP | 2.5535 | 0.9500 | -0.1126 | 4.2694 | 86.7221 | 96.9754 | 98.9692 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | DBP | 1.4455 | 0.9351 | -0.0977 | 3.0120 | 95.9869 | 99.3213 | 99.8176 | PASS* | PASS (Grade A)* |

All 96 saved diagnostic rows (8 budgets × 2 methods × 3 views × 2 BP targets) meet the implemented AAMI-style numerical screen and BHS-style Grade A percentage cutoffs. **PASS* means only a retrospective numerical screen.** The implementation uses |ME| ≤ 5 mmHg and error STD ≤ 8 mmHg, and Grade A cumulative percentages of at least 60%, 85% and 95% within 5, 10 and 15 mmHg. These are code-level diagnostic rules, not evidence of completing a clinical validation protocol.

In particular, passing these calculations on hospital PPG windows is not wrist-device certification. [ISO 81060-3:2022](https://www.iso.org/standard/71161.html) specifies clinical-investigation requirements for continuous automated non-invasive devices. [BIHS validation guidance](https://bihs.org.uk/blood_pressure_technology/bp_monitor_validations.aspx) also distinguishes its conventional cuff-monitor validation process from wearable endorsement. Neither process was performed here.

## 6. Actual enrollment burden

Counts are labeled 10-s public-data windows, not independent cuff measurements. All final query counts remain fixed.

| Nominal enrollment | Total bank windows | MIMIC median/person | VitalDB median/person | Zero-adapter fallback people | Queries |
| --- | --- | --- | --- | --- | --- |
| 20% | 156,127 | 207 | 100 | 50 | 78,237 |
| 30% | 234,297 | 311 | 150 | 37 | 78,237 |
| 40% | 312,545 | 415 | 201 | 23 | 78,237 |
| 50% | 390,857 | 519 | 251 | 14 | 78,237 |
| 60% | 468,948 | 622 | 301 | 14 | 78,237 |
| 70% | 547,126 | 726 | 352 | 0 | 78,237 |
| 80% | 625,389 | 830 | 402 | 0 | 78,237 |
| 90% | 703,869 | 934 | 453 | 0 | 78,237 |

Even at 20%, median bank sizes are 207 MIMIC and 100 VitalDB windows. For histories with only one eligible allocation group, the prespecified method performs zero adapter optimization steps rather than selecting epochs using query labels. Those people and their queries remain included. At 70% and above this particular fallback is no longer needed. Counts, rounding and fallback coverage belong to the protocol; the budget curve does not isolate sample count from every consequence of the small-bank fitting rule.

[Complete budget count/fraction audit](budget_summary.csv) includes validation and both final source strata. The 70% bank uses approximately 22% fewer labeled windows than the 90% bank; this is not a demonstrated reduction of 22% in real cuff visits or study duration.

## 7. Validation cohort (supporting, not a second final test)

Participant-macro SBP / DBP MAE. The validation people helped select the shared parent checkpoint, so their outcomes are development results, not independent confirmation.

| Enrollment | LoRA | LoRA + memory |
| --- | --- | --- |
| 20% | 5.3780 / 3.0502 | 4.6263 / 2.6452 |
| 30% | 5.0529 / 2.8685 | 4.2410 / 2.4524 |
| 40% | 4.7001 / 2.6682 | 3.9288 / 2.2634 |
| 50% | 4.5129 / 2.5390 | 3.7682 / 2.1208 |
| 60% | 4.3869 / 2.4544 | 3.6444 / 2.0451 |
| 70% | 4.2237 / 2.3722 | 3.5019 / 1.9754 |
| 80% | 4.0844 / 2.3141 | 3.4013 / 1.9259 |
| 90% | 4.0121 / 2.2543 | 3.3595 / 1.8714 |

[Validation aggregate files](validation) preserve all source views, diagnostics, frozen-method evidence and budget intervals. No arm was discarded after validation scoring.

## 8. Verification and interpretation limits

- A separate read-only audit recomputed all 48 primary summary rows, all 96 full diagnostic rows and the seven primary paired budget intervals from saved final predictions. Maximum absolute numeric discrepancy was 1.42e-14.
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

Training snapshot: `2b5dd222b7fb469a9f02853b2ab5edf4fc11de64`. Plan SHA256: `49e023ba7a323baf7fd72cebed4bfb46e17ddd7c1e7b56f6f04fbc20366fb665`. Shared-checkpoint SHA256: `5340902ef5bf81c1f8a704380d3376d9b7e1b1df3d509175b0b89c7d494397ed`. Fixed-query SHA256: `fdec318023a1ff0cf2616c51e889eaea5d3f083cf58591d5097be3dcb5f7144a`.

Only aggregate tables, checksums, protocols and reporting code are published. Raw waveforms, individual predictions, target labels, personal parameters and reference banks remain private.
