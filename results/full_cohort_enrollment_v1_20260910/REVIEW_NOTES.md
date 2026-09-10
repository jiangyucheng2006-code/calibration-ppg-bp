## Material Passport

- Source: saved full-cohort enrollment results, frozen receipts and live Slurm accounting.
- Mode: result interpretation and saved-output verification.
- Review date: 11 September 2026.
- Verification status: ANALYZED; metric arithmetic reproduced, neural training not rerun.
- Version: full_cohort_result_review_v1.

## Statistical review

Overall confidence: CAUTION for scientific generalization, with matching saved
and recalculated numerical metrics. The primary comparison is paired mean MAE
across the same 762 participants. The saved 2,000-replicate source-stratified
bootstrap interval for the gain is [0.4513, 0.5320] mmHg. No new p-value,
normality test, multiple-testing adjustment or training-seed interval was
calculated. Parametric t-test assumptions are not asserted. Bootstrap
interpretation assumes useful exchangeability of participants within the two
source strata and is conditional on the fitted checkpoint.

The present review recomputed participant-macro MAE, pooled MAE/R²/ME/STD,
cumulative threshold percentages, numerical-screen categories, paired mean
gains and improvement/tie/worsening counts. Saved interval endpoints were
inspected with their generating code, not independently regenerated here.
The original label store and neural training were not rerun; aggregate
agreement is not a fresh full data-lineage or implementation audit.

## Fallacy scan: 11/11 categories considered

| Category | Assessment in this review |
|---|---|
| Simpson's paradox | No direction reversal between Overall, MIMIC and VitalDB for the paired mean-MAE comparison; other subgroups not newly analyzed. |
| Ecological fallacy | Do not infer universal individual benefit: 626 improve, 97 tie and 39 worsen. |
| Berkson/selection bias | Hospital-derived, preprocessed public signals and enrollment eligibility limit population applicability. |
| Collider bias | No new outcome-based conditioning was performed; upstream eligibility and quality selection remain a limitation, not a demonstrated absence of bias. |
| Base-rate neglect | No diagnostic classification or predictive-value claim; threshold percentages are error summaries. |
| Regression to the mean | Primary comparison retains the whole eligible cohort; saved oracle-tail columns are not the headline evidence. |
| Survivorship bias | All 762 planned eligible test profiles completed; 83 short-history and three provenance exclusions from the full source audit are disclosed. |
| Look-elsewhere effect | Two methods were prespecified in this batch, but the project has extensive prior exploration; secondary intervals are not independent confirmations. |
| Forking paths | Preserve the frozen current contract. Lower-budget/online protocols are separate proposals and cannot be selected by optimizing this final score. |
| Correlation versus causation | The paired prediction comparison supports a numerical performance difference, not a physiological mechanism or clinical benefit. |
| Reverse causality/temporal claims | No causal or future-date conclusion; within-person random windows are not prospective long-term testing. |

No additional per-person filtering, hypothesis shopping, outlier removal,
model fitting or intervention was performed. The distinction between pooled
and participant-macro weighting is explicit in the main report.
