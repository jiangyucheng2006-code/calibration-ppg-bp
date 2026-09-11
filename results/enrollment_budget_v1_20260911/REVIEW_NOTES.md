## Material Passport

- Origin: completed enrollment-budget predictions and frozen scoring artifacts.
- Mode: result validation.
- Date: 2026-09-11.
- Verification status: ANALYZED; numerical summaries and primary intervals were independently recomputed from saved predictions.
- Version: enrollment-budget-results-v1.
- Scope: one shared fit and one nested random-history ordering. No training-seed replication was performed.

## Statistical findings

| Finding | Evidence | Interpretation |
| --- | --- | --- |
| More enrollment history lowers average error | Overall memory mean MAE: 3.5865 at 20%, 2.5209 at 90%; paired increase 1.0657, simultaneous 95% interval [0.8456, 1.2858] | Supports a budget effect within this fixed-query benchmark. |
| 30% is not effectively interchangeable with 90% on these point estimates | Mean-MAE increase 0.7332, simultaneous interval [0.5131, 0.9533] | Do not claim equivalence or a demonstrated minimal personal-label requirement. |
| 70% and 80% approach 90% | Mean-MAE increases 0.0625 and 0.0512; joint intervals include zero | Descriptive plateau candidates only. A null-containing interval is not an equivalence test. |
| Memory adds predictive value at each budget | All Overall/source SBP, DBP and mean MAE comparisons favor memory on their averages | Report paired magnitudes and exploratory intervals, not only pass/fail labels. |
| Not all individuals improve | 90% memory vs LoRA: 626 improve / 97 tie / 39 worsen; memory 20% to 90%: 718 improve / 44 worsen | Do not promise that every user or every future update improves. |

## Design and uncertainty checks

The independent resampling unit is the participant, stratified by PulseDB
source. All arms retain the same participants and query identities. Point
estimates and intervals are conditional on a fixed trained population model;
they do not include new training seeds or a new enrollment-group ordering.
No t-test, normality test, equal-variance test or clinical power calculation
is claimed. The bootstrap relies on exchangeability of sampled people within
source strata; hospital/quality-filtered data limit population transportability.

All seven primary mean-MAE contrasts versus 90% use the prespecified joint
maximum-centered-deviation interval. SBP/DBP, source-specific and within-budget
memory intervals remain pointwise exploratory summaries. No prespecified
non-inferiority margin exists. No alternative test or margin was chosen after
seeing the final scores, and no budget is designated a confirmed winner.

The parent 90% final cohort had already been scored and reported. Its reference
predictions match this run byte-for-byte. The remaining budgets add an
exploratory dose-response evaluation on that same cohort; this is not another
independent confirmation of the entire model-development history.

## Fallacy scan

Coverage: 11/11 checked. Checks address the reported interpretation; they are
not a guarantee that every possible source-data confound has been excluded.

| Issue | Assessment | Handling |
| --- | --- | --- |
| Simpson's paradox | No reversal of the average memory advantage across the two source strata | Report Overall, MIMIC and VitalDB separately. Some adjacent-budget subgroup metrics are non-monotonic and remain visible. |
| Ecological fallacy | Individual benefit cannot be inferred from the aggregate curve | Publish improved/tied/worsened counts and avoid universal-benefit claims. |
| Berkson's/selection bias | Hospital-source and preprocessing eligibility limit generalization | Retain source and device-setting caveats; do not generalize directly to ambulatory volunteers. |
| Collider bias | No new conditioning on prediction error in this budget comparison | Existing source-quality/eligibility selection remains a limitation. No test-error exclusion was introduced. |
| Base-rate neglect | Not a disease classification study | Numerical error-percentage screens are not predictive values or diagnosis. |
| Regression to the mean | No worst-error subset was selected for the headline curve | Retain every eligible participant and the paired LoRA comparison. Oracle tail diagnostics are not a prospective filter. |
| Survivorship bias | All 762 final people and 78,237 queries remain in every arm | Report prespecified zero-adapter fallbacks instead of dropping short-history people. Preserve the parent's 86 eligibility exclusions separately. |
| Look-elsewhere effect | Eight budgets and two methods create multiple comparisons | Publish all results and the seven-contrast joint primary interval; other intervals remain exploratory. |
| Garden of forking paths | Long prior development history and previously scored parent queries | State local prespecification and frozen-code evidence, without claiming independent preregistered confirmation. |
| Correlation/causation confusion | This is a controlled software comparison, not a causal clinical intervention | Attribute differences to this implemented budget policy; do not claim elapsed wear time causes better clinical reliability. |
| Reverse causality / temporal precedence | Personal history was sampled without chronological precedence | Do not call the budget curve prospective online or future-date validation. |

## Reproducibility verdict

Saved-prediction arithmetic: PASS. A separate read-only program verified the
16 frozen prediction sets, 48 participant-macro summary rows, 96 full
diagnostic rows and seven primary budget intervals. The maximum absolute
numeric discrepancy was 1.4210854715202004e-14. Final aggregate files match
work/NAS copies and their local transferred copies.

Training replication: NOT RUN in this publication step. Exact repetition of
the 90% reference under the same checkpoint, personal seed and input ordering
is useful consistency evidence, but not independent-seed evidence.

Clinical validation: NOT PERFORMED. All AAMI/BHS fields remain retrospective
numerical screens. No wrist-data, cross-day or prospective human-subject
result is included.
