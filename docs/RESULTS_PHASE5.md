# Phase-5 development results

Last updated: 2026-08-14.

## Result status

This document reports an **exploratory, single-seed meta-validation snapshot**.
It is not a locked-test, external-validation, clinical, or final-paper result.

- Split: `meta_validation` only.
- Locked meta-test accessed: **no**.
- Participants: 697.
- Common future query events per calibration budget: 103,564.
- Calibration budgets: `K = 1, 2, 3, 5` independent `event120-v1`
  pseudo-cuff events.
- Model-selection seed: `20260813`.
- Primary metric: participant-macro MAE in mmHg.

In PulseDB, a pseudo-cuff event is a temporally separated representative event
with an ABP-derived reference BP. It is not a literal cuff measurement. Events
1--5 form the support-candidate pool, and all K values use the same future query
set beginning at event 6. Query BP labels are evaluator-only.

## Primary comparison

The table reports the participant-macro mean of SBP MAE and DBP MAE. Lower is
better.

| Method | K=1 | K=2 | K=3 | K=5 |
|---|---:|---:|---:|---:|
| **M0: variable-K residual anchor** | **9.998** | **9.435** | **9.119** | **8.688** |
| M1: M0 + FiLM | 10.307 | 9.630 | 9.280 | 8.818 |
| M2: M1 + reliability weighting | 10.622 | 9.807 | 9.381 | 8.873 |
| LoRA adaptation | 10.372 | 10.043 | 9.824 | 9.509 |
| Head-only adaptation | 10.975 | 10.427 | 10.040 | 9.554 |
| Residual-offset correction | 12.305 | 11.190 | 10.593 | 9.931 |
| Calibration-free population network | 11.638 | 11.638 | 11.638 | 11.638 |
| Siamese delta model | 11.702 | -- | -- | -- |
| Last-cuff persistence | 15.023 | 13.825 | 12.993 | 12.733 |
| Full-network adaptation | 15.032 | 13.381 | 12.498 | 11.862 |
| Population BP mean | 13.620 | 13.620 | 13.620 | 13.620 |

M0 is the current single-seed development winner at every K. Adding FiLM in
M1 and query-conditioned reliability weighting in M2 did not improve the
result under the current seed and shared optimization configuration.

The complete rounded metric table is available in
[`results/phase5_meta_validation_summary.csv`](../results/phase5_meta_validation_summary.csv).

## Extended diagnostic table

The first-run prediction artifacts were also re-evaluated with event-pooled
MAE, R², signed mean error, sample SD of error, and cumulative percentages
within 5, 10, and 15 mmHg. The complete 90-row SBP/DBP table is available in
[RESULTS_PHASE5_FIRST_RUN_EXTENDED.md](RESULTS_PHASE5_FIRST_RUN_EXTENDED.md)
and
[`results/phase5_first_run_extended_metrics.csv`](../results/phase5_first_run_extended_metrics.csv).

All 90 method/K/BP rows fail the AAMI Criterion-1-style numerical screen; 19
receive historical BHS Grade C and 71 receive Grade D. These labels are
diagnostic screens only. PulseDB retrospective model evaluation does not meet
the full study-design, reference-measurement, participant-distribution, or
repeated-measure requirements for formal device validation, so no standards
compliance claim is made.

## M0 SBP and DBP results

| K | SBP MAE | DBP MAE | Mean MAE | SBP RMSE | DBP RMSE | SBP bias | DBP bias |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.865 | 7.131 | 9.998 | 15.140 | 8.513 | 2.123 | 0.991 |
| 2 | 12.155 | 6.714 | 9.435 | 14.421 | 8.083 | 2.336 | 0.899 |
| 3 | 11.770 | 6.468 | 9.119 | 14.045 | 7.851 | 2.306 | 0.874 |
| 5 | 11.219 | 6.157 | 8.688 | 13.491 | 7.566 | 2.030 | 0.744 |

Bias is defined as prediction minus reference. M0 retains a positive SBP bias
of approximately 2 mmHg in this development cohort.

## Paired development diagnostics

A post-hoc participant-paired bootstrap used 5,000 resamples and seed
`20260814`. Positive improvement means M0 has lower participant mean MAE than
the comparator. These intervals characterize participants in the current
meta-validation cohort; they do not include training-seed variability and are
not confirmatory locked-test intervals.

| Comparison | K | Improvement (mmHg) | 95% bootstrap interval | Participant win rate |
|---|---:|---:|---:|---:|
| M0 vs population network | 1 | 1.641 | [1.158, 2.132] | 62.3% |
| M0 vs population network | 5 | 2.951 | [2.492, 3.409] | 70.3% |
| M0 vs residual offset | 1 | 2.307 | [1.892, 2.755] | 71.9% |
| M0 vs residual offset | 5 | 1.243 | [1.043, 1.447] | 73.0% |
| M0 vs LoRA | 1 | 0.374 | [0.054, 0.688] | 58.8% |
| M0 vs LoRA | 5 | 0.821 | [0.564, 1.095] | 64.3% |

Within M0, K=5 improved participant mean MAE over K=1 by 1.310 mmHg,
with a 95% participant-bootstrap interval of [1.072, 1.563]. A total of 66.7%
of participants improved. This supports calibration efficiency, while also
showing that additional calibration does not benefit every participant.

## Source and tail diagnostics

| Source | Participants | K=1 | K=2 | K=3 | K=5 |
|---|---:|---:|---:|---:|---:|
| MIMIC | 316 | 10.337 | 9.895 | 9.662 | 9.344 |
| VitalDB | 381 | 9.717 | 9.053 | 8.669 | 8.144 |

M0 improved over the population model within both sources at every K, but its
absolute error was consistently higher in MIMIC. These source groups are both
components of PulseDB and are not independent external validation.

The M0 participant mean-MAE distribution remains broad. At K=5, the median,
90th percentile, and 95th percentile were 7.72, 14.67, and 16.92 mmHg,
respectively. The tail prevents any claim of uniformly stable or clinically
validated BP estimation.

## Optimization and reproducibility audit

| Method | Epochs run | Best epoch | Stop condition |
|---|---:|---:|---|
| Population | 7 | 2 | patience reached |
| Siamese | 6 | 1 | patience reached |
| M0 | 25 | 25 | maximum epoch reached |
| M1 | 9 | 4 | patience reached |
| M2 | 12 | 7 | patience reached |

The formal runner used `--epochs 25 --patience 5`. M0 achieved its best
validation score at epoch 25, so convergence beyond the cap is unknown.

The work and NAS result directories were verified byte-for-byte for the
population, M0, M1, M2, calibration-control, and replacement Siamese runs.
Checkpoint hashes also match their NAS archives. Private artifacts retain run
configuration, environment, data-manifest hash, source-tree hash, history,
checkpoint hash, and participant/event-level predictions. Data, predictions,
and checkpoints are intentionally excluded from this public repository.

## Supported conclusion

Under the current internal development split, event-level few-shot personal
calibration adds value over calibration-free prediction and strong simple
calibration controls. M0 is the strongest current configuration. The evidence
does **not** yet establish multi-seed stability, locked-test generalization,
pressure or motion robustness, device transfer, external validation, clinical
accuracy, or compliance with a medical-device standard.

## Next decision gate

Before any locked-test evaluation:

1. Run a one-factor M0 convergence sensitivity with a higher epoch cap while
   keeping all other settings unchanged.
2. Repeat the population-to-personalization pipeline under at least three
   prespecified seeds; five seeds are preferred if compute permits.
3. Retain M0, M1, M2, LoRA, head-only, residual offset, and last cuff in the
   development comparison so training-seed variability is not confused with
   participant-bootstrap uncertainty.
4. Freeze the final configuration, seed policy, result script, checkpoint rule,
   and source/tail reporting plan.
5. Run the quarantined locked meta-test once.

Robustness training for contact pressure, motion, and device/acquisition shift
begins only after this base personalization gate is passed.

The exact raw-file, segment, event, eligibility, split, and sampler rules used
before this first run are documented in
[DATA_SELECTION_AND_TRAINING_COHORT.md](DATA_SELECTION_AND_TRAINING_COHORT.md).
