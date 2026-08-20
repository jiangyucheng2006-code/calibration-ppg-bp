# Round-8 calibration-relative screening results

## Scope and claim limit

Round 8 is a development-only, single-seed screen on the PulseDB
`meta_validation` participants. The common comparison uses fixed-first `K=5`
support and event-6-onward queries. The locked meta-test was not accessed.

R8-1 through R8-5 were deliberately screened at K=5 before spending compute
on all calibration budgets. R8-7 and R8-8 also produced K=1/2/3/5 outputs, but
K=5 is the only common budget across every Round-8 candidate. K=4 is not a
prespecified primary budget in this project.

Participant-macro MAE is the primary metric. Event-pooled R-squared, bias,
error standard deviation, cumulative absolute-error percentages, and
AAMI/BHS-style fields are secondary diagnostics. The latter are numerical
screens only and do not establish device, clinical, or standards compliance.

## Full-coverage participant-macro result

All full-coverage candidates use the same 697 participants and 103,564 K=5
query events.

| Setting | SBP MAE | DBP MAE | Mean MAE | Change from reference |
|---|---:|---:|---:|---:|
| Quality Gate + Huber (256-D) | 10.803 | 6.168 | 8.485 | reference |
| R8-1 Pairwise delta | 10.743 | 5.987 | 8.365 | -0.120 |
| R8-2 Pairwise delta + causal time | 10.605 | 5.970 | 8.288 | -0.197 |
| R8-3 Pairwise delta + range auxiliary | 10.845 | 6.097 | 8.471 | -0.014 |
| **R8-4 Pairwise delta + causal time + range** | **10.487** | **6.021** | **8.254** | **-0.231** |
| R8-5 R8-4 + generic PPG-change features | 10.606 | 5.948 | 8.277 | -0.208 |
| R8-7 Direct demographics + 256-D | 10.803 | 6.106 | 8.455 | -0.030 |
| R8-8 Quality Gate + Huber 128-D | 10.890 | 6.126 | 8.508 | +0.023 |

R8-4 is the numerical screening winner. Relative to the reference, it reduces
Overall participant-macro SBP MAE by 0.316 mmHg, DBP MAE by 0.146 mmHg, and
their mean by 0.231 mmHg. This is a positive development result, not yet an
independent-seed confirmation.

## Source-stratified result

MIMIC and VitalDB are internal PulseDB source strata, not independent external
validation datasets.

| Scope | Setting | Participants | Queries | SBP MAE | DBP MAE | Mean MAE |
|---|---|---:|---:|---:|---:|---:|
| Overall | Reference | 697 | 103,564 | 10.803 | 6.168 | 8.485 |
| Overall | **R8-4** | 697 | 103,564 | **10.487** | **6.021** | **8.254** |
| MIMIC | Reference | 316 | 80,874 | 11.576 | 6.573 | 9.075 |
| MIMIC | **R8-4** | 316 | 80,874 | **11.355** | **6.348** | **8.852** |
| VitalDB | Reference | 381 | 22,690 | 10.161 | 5.831 | 7.996 |
| VitalDB | **R8-4** | 381 | 22,690 | **9.766** | **5.750** | **7.758** |

R8-4 improves mean MAE by 0.223 mmHg in MIMIC and 0.238 mmHg in VitalDB. The
direction is therefore consistent across both internal source strata in this
single-seed screen.

## What each ablation shows

- Pairwise calibration-relative prediction is useful: R8-1 improves mean MAE
  by 0.120 mmHg.
- Causal query history adds value: R8-2 improves more than R8-1.
- The range task alone is insufficient: R8-3 is essentially neutral.
- The range task becomes useful when combined with pairwise and causal
  information: R8-4 is the best full-coverage candidate.
- Generic handcrafted PPG-change summaries in R8-5 improve DBP but do not
  improve the overall mean beyond R8-4.
- Direct demographic concatenation is nearly neutral, and reducing the PPG
  representation to 128 dimensions is slightly worse than the 256-D
  reference.

These are ablation interpretations from one development seed. They identify
promising components but do not quantify their final uncertainty.

## Beat-similarity threshold sensitivity

The `similarity >= 0.90` sensitivity retains 82,653 of 103,564 Overall queries
(79.81%). Coverage differs substantially by source: 85.45% in MIMIC versus
59.70% in VitalDB. It also produces worse full-retained-set participant-macro
mean MAE than the full-coverage reference. It is therefore not eligible as a
deployable winner and is not carried forward as a filtering rule.

## Decision

R8-4 passes the prespecified single-seed screen because it improves Overall
mean MAE by more than 0.15 mmHg and improves both internal source strata. It
becomes the candidate base for the next development round. Multi-seed
confirmation is intentionally deferred until the next method round determines
whether this calibration-relative design can be improved further. The locked
meta-test remains quarantined.

## Machine-readable tables

- [Full-coverage participant-macro results](../results/round8/participant_macro_full_coverage.csv)
- [Full-coverage event-pooled diagnostics](../results/round8/diagnostic_full_coverage.csv)
- [Similarity-filter coverage](../results/round8/similarity_filter_coverage.csv)
- [Similarity-filter participant-macro results](../results/round8/similarity_filter_participant_macro.csv)
- [Similarity-filter event-pooled diagnostics](../results/round8/similarity_filter_diagnostic.csv)
