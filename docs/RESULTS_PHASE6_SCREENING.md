# Phase-6 fixed-first development screening

Last updated: 2026-08-18.

## Scope and integrity gates

This report compares the fixed-first M0 reference (job 818) with five
single-factor candidates (jobs 826--830). All six runs used seed `20260813`,
events 1--K as calibration support, and the identical event-6-onward query
set. Every job completed with exit code `0:0`; stderr was empty. Each saved
prediction table contains 697 meta-validation participants and 103,564 common
queries for each K. Query keys and targets match exactly across methods, work
and NAS artifacts are byte-identical, and `locked_test_accessed=false` for
every run.

Participant-macro MAE is the primary metric. Event-pooled metrics are secondary
diagnostics. MIMIC and VitalDB below are internal PulseDB source strata, not
independent external validation datasets.

## Primary Overall participant-macro results

Each cell is `(SBP MAE + DBP MAE) / 2`, computed after first averaging errors
within each participant. Units are mmHg.

| Setting | K=1 | K=2 | K=3 | K=5 | Four-K mean | Change vs M0 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-first M0 | 9.559 | 9.189 | 9.018 | 8.782 | 9.137 | -- |
| Huber loss | 9.440 | 9.137 | 8.981 | 8.756 | 9.078 | -0.059 |
| BP-change sampling | 9.810 | 9.428 | 9.218 | 8.948 | 9.351 | +0.214 |
| Median anchor | 9.575 | 9.240 | 9.171 | 8.971 | 9.239 | +0.102 |
| PPG-only quality gate | **9.386** | **9.035** | **8.840** | **8.587** | **8.962** | **-0.175** |
| Age and sex | 9.537 | 9.172 | 9.011 | 8.801 | 9.130 | -0.007 |

The PPG-only quality gate is the only candidate that improves Overall mean MAE
at every K. Its four-K improvement is 0.175 mmHg (about 1.9%). Huber is a small
secondary improvement. BP-change sampling and the median anchor are worse;
age/sex conditioning is effectively neutral.

## Source-stratified primary results

The table shows the four-K average participant-macro mean MAE. Overall is
recomputed over all participants; it is not the arithmetic mean of the two
source rows.

| Setting | Overall | MIMIC | VitalDB |
|---|---:|---:|---:|
| Fixed-first M0 | 9.137 | 9.891 | 8.512 |
| Huber loss | 9.078 | 9.783 | 8.494 |
| BP-change sampling | 9.351 | 10.220 | 8.629 |
| Median anchor | 9.239 | 9.939 | 8.659 |
| PPG-only quality gate | **8.962** | **9.546** | 8.478 |
| Age and sex | 9.130 | 9.923 | **8.472** |

The quality-gate improvement is source-asymmetric: it improves MIMIC by 0.345
mmHg on the four-K average but VitalDB by only 0.034 mmHg. This is why one
single seed is not enough to replace M0 as the confirmed base model.

## Requested K=5 diagnostic table

The full Overall, MIMIC, and VitalDB tables for every setting and K are stored
under `results/phase6_screening/`. The compact table below shows the reference
and leading candidate at K=5. `ME = prediction - reference`.

| Scope / Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Overall / M0 | SBP | 12.093 | 0.484 | -2.028 | 15.680 | 28.15% | 51.82% | 69.42% | FAIL* | Grade D, FAIL* |
| Overall / M0 | DBP | 7.063 | 0.446 | -1.071 | 9.737 | 47.24% | 76.69% | 90.07% | FAIL* | Grade C, FAIL* |
| Overall / quality gate | SBP | **11.745** | **0.513** | -1.232 | **15.309** | 28.73% | 53.08% | 70.68% | FAIL* | Grade D, FAIL* |
| Overall / quality gate | DBP | **6.918** | **0.481** | -0.227 | **9.486** | 47.19% | 77.33% | 90.85% | FAIL* | Grade C, FAIL* |
| MIMIC / M0 | SBP | 12.595 | 0.475 | -2.269 | 16.225 | 26.82% | 49.88% | 67.42% | FAIL* | Grade D, FAIL* |
| MIMIC / M0 | DBP | 7.388 | 0.423 | -1.127 | 10.196 | 45.76% | 74.96% | 88.75% | FAIL* | Grade C, FAIL* |
| MIMIC / quality gate | SBP | **12.163** | **0.508** | -1.606 | **15.774** | 27.70% | 51.55% | 68.96% | FAIL* | Grade D, FAIL* |
| MIMIC / quality gate | DBP | **7.197** | **0.464** | -0.292 | **9.876** | 45.64% | 75.88% | 89.81% | FAIL* | Grade C, FAIL* |
| VitalDB / M0 | SBP | 10.303 | 0.497 | -1.167 | 13.526 | 32.90% | 58.74% | 76.57% | FAIL* | Grade D, FAIL* |
| VitalDB / M0 | DBP | **5.907** | **0.551** | -0.871 | **7.886** | 52.50% | 82.84% | 94.75% | PASS* | Grade B, PASS* |
| VitalDB / quality gate | SBP | **10.256** | **0.507** | 0.098 | **13.442** | 32.42% | 58.53% | 76.83% | FAIL* | Grade D, FAIL* |
| VitalDB / quality gate | DBP | 5.924 | 0.551 | 0.007 | 7.941 | 52.70% | 82.50% | 94.54% | PASS* | Grade B, PASS* |

The asterisks mean retrospective numerical screens only. These rows do not
establish AAMI/ISO/IEEE device compliance or clinical validity.

## Exact worst-30% diagnosis at K=5

The reference tail is defined by participant mean MAE descending, with
`subject_uid` as a deterministic tie-break. Exactly `ceil(0.30 x 697) = 210`
participants are selected. Because this definition uses observed query error,
the retained-70% and routing numbers are oracle diagnostics only.

| Model/cohort | Participants | SBP MAE | DBP MAE | Mean MAE |
|---|---:|---:|---:|---:|
| M0, all | 697 | 11.212 | 6.353 | 8.782 |
| M0, observed-error worst 30% | 210 | 17.691 | 9.738 | 13.714 |
| M0, oracle retained 70% | 487 | 8.418 | 4.894 | 6.656 |
| Quality gate, all | 697 | 10.910 | 6.264 | 8.587 |
| Quality gate on the same reference worst 30% | 210 | 16.609 | 9.233 | 12.921 |
| Quality gate on the same reference retained 70% | 487 | 8.452 | 4.984 | 6.718 |

The quality gate improves the fixed reference tail by 0.793 mmHg, while it is
0.062 mmHg worse on the fixed retained group. An oracle router using M0 for the
retained group and the quality gate for the tail would reach 8.544 mmHg, an
improvement of 0.239 mmHg over M0. That router is not deployable because the
tail membership was obtained from the true query errors.

## Common patterns in the observed-error worst 30%

| Development-only diagnostic | Remaining 70% | Worst 30% | Difference |
|---|---:|---:|---:|
| Mean query SBP | 116.75 | 122.20 | +5.45 mmHg |
| Mean query DBP | 61.67 | 64.94 | +3.27 mmHg |
| Within-participant query SBP SD | 11.82 | 15.24 | +28.9% |
| Within-participant query DBP SD | 6.53 | 8.84 | +35.3% |
| Mean absolute support-to-query SBP change | 13.98 | 19.02 | +36.0% |
| Mean absolute support-to-query DBP change | 7.47 | 10.55 | +41.3% |
| Mean query event index | 71.47 | 99.08 | +38.6% |
| Query SBP outside support range | 56.57% | 64.03% | +7.47 pp |
| Query DBP outside support range | 55.34% | 62.27% | +6.92 pp |
| Filtered-PPG standard deviation | 0.2747 | 0.2765 | nearly unchanged |
| Mean cleaned age | 61.18 | 64.10 | +2.93 years |

The MIMIC tail rate is 38.3%, versus 23.4% in VitalDB. These are associations,
not causal explanations. The strongest patterns involve larger BP drift,
larger within-participant BP variability, and a later prediction horizon. The
simple filtered-PPG amplitude proxy does not separate the groups, so a future
deployable risk model needs richer PPG quality/morphology and model-uncertainty
features.

## Development decision and Round-4 design

- Promote the PPG-only quality gate to a provisional component for further
  screening; do not yet replace M0 because evidence is single-seed and
  source-asymmetric.
- Retain Huber as a secondary robust-loss factor.
- Do not promote BP-change-aware sampling, median anchoring, or age/sex as an
  isolated base-model change.
- Complete a three-factor `quality gate x Huber x participant-tail CVaR`
  ablation. Existing jobs 818, 826, and 829 provide the none, Huber-only, and
  gate-only cells; Round 4 adds CVaR-only, gate+Huber, gate+CVaR, Huber+CVaR,
  and gate+Huber+CVaR.
- CVaR batches contain repeated episodes from distinct, uniformly sampled
  meta-train participants. The objective combines ordinary participant risk
  and the highest-loss 30% participant risk with fixed weight 0.5. Query BP is
  used only as an ordinary meta-train supervision target, never as an
  inference-time routing input.
- A deployable high-risk gate and specialist model remain a later step. Their
  tail labels must come from participant-disjoint, cross-fitted meta-train
  predictions; their inputs may use support BP, PPG morphology/quality,
  embedding shift, time since calibration, and uncertainty, but not query BP
  or query error.
- A second cuff measurement may be studied only at a fixed time or after an
  input-only trigger. The trigger event becomes support and cannot also be
  scored as a query.

## Reproducibility artifacts

- Formal CPU aggregation: Slurm job 839, exit `0:0`, 55 regression tests
  passing before execution, zero-byte stderr, and byte-identical work/NAS
  report archives.
- Complete machine-readable tables:
  `results/phase6_screening/`.
- Tail-aware method and evidence plan:
  [PHASE6B_TAIL_AWARE_PLAN.md](PHASE6B_TAIL_AWARE_PLAN.md).
- Locked meta-test remains quarantined and was not accessed.
