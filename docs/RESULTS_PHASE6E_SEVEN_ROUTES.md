# Phase-6E seven-route continuous-error screen

## Result in one sentence

The causal GRU residual corrector is the best of seven single-seed K=5
development candidates, but its 0.078-mmHg Overall gain is below the frozen
0.15-mmHg promotion threshold. No Phase-6E route is promoted.

## Protocol and integrity

- Reference: Quality Gate + Huber, job 841.
- Split: `meta_validation`; locked meta-test not accessed.
- Calibration/query rule: fixed-first events 1--5 support, event 6 onward
  queries, K=5.
- Evaluation: 697 participants and 103,564 common queries; 316 MIMIC
  participants/80,874 queries and 381 VitalDB participants/22,690 queries.
- Training targets: participant-disjoint five-fold Quality Gate + Huber
  meta-train out-of-fold residuals. Folds 0--3 fit candidates and fold 4
  performs internal selection.
- Promotion gate: at least 0.15-mmHg Overall participant-macro mean-MAE gain
  and improvement in both source strata.
- Jobs 915--924 and corrected audit jobs 925--926 all completed with exit code
  0 and empty stderr. Work and NAS report artifacts are byte-identical.

## Primary participant-macro result

Lower is better. Delta is candidate mean MAE minus the general reference, so a
negative value is an improvement.

| Setting | Overall mean MAE | Delta | MIMIC mean MAE | VitalDB mean MAE |
|---|---:|---:|---:|---:|
| General QGate + Huber | 8.485 | -- | 9.075 | 7.996 |
| R6-1 Ridge residual | 8.557 | +0.072 | 9.138 | 8.076 |
| R6-2 Residual MLP | 8.463 | -0.023 | 9.023 | 7.998 |
| R6-3 Gated residual MLP | 8.498 | +0.013 | 9.084 | 8.012 |
| R6-4 Difficult-2x residual MLP | 8.548 | +0.063 | 9.098 | 8.092 |
| R6-5 Causal GRU residual | **8.408** | **-0.078** | **8.969** | **7.942** |
| R6-6 Supervised MoE | 8.512 | +0.027 | 9.093 | 8.031 |
| R6-7 Morphology cluster MoE | 8.480 | -0.005 | 9.088 | 7.976 |

The causal GRU's participant-macro SBP/DBP/mean MAE is
10.674/6.141/8.408 mmHg Overall, 11.413/6.526/8.969 in MIMIC, and
10.061/5.822/7.942 in VitalDB. Its paired participant-bootstrap mean-MAE
difference versus the reference is -0.0775 mmHg, with an exploratory 95%
interval of -0.1542 to approximately 0.0000. The gain is small and this is one
development seed; it is not confirmatory evidence.

## Morphology-cluster result

The waveform-driven idea was implemented as a frozen-encoder embedding,
meta-train-only PCA/K-means partition, soft cluster assignment, and
cluster-specific residual experts. It did not impose a 70%/30% split and it
produced a prediction for every query.

However, none of the 8/16/32-cluster candidates passed the prespecified 0.75
stability gate: their assignment stabilities were 0.379, 0.389, and 0.439.
The reported 8-cluster result is therefore an explicitly exploratory fallback,
not evidence of stable PPG phenotypes. It improved Overall by only 0.005 mmHg,
worsened MIMIC by 0.013, and improved VitalDB by 0.020; it is ineligible for
promotion.

## Interpretation and decision

Temporal history contains a weak useful signal: the causal GRU improves both
sources and is the only route with an Overall bootstrap interval just excluding
zero in the favorable direction before rounding. Nevertheless, the absolute
gain is only about 0.91% and fails the frozen effect-size gate. Ridge,
difficult-case weighting, and both MoE variants do not improve the general
model. The current base remains Quality Gate + Huber; do not spend multiple
seeds confirming any Phase-6E candidate yet.

The next experiment should make one substantive change rather than enlarge
these shallow residual heads: either learn a temporally causal correction from
the PPG encoder sequence end-to-end, or learn morphology prototypes with a
representation objective that explicitly produces stable participant-
independent clusters before attaching experts. This remains development-only.

## Public aggregate artifacts

- `results/phase6e/participant_macro.csv`: Overall/MIMIC/VitalDB primary
  metrics and oracle tail diagnostics.
- `results/phase6e/pooled_metrics.csv`: requested Setting/BP/MAE/R2/ME/STD,
  5/10/15-mmHg, AAMI-style, and BHS-style retrospective screens.
- `results/phase6e/paired_bootstrap.csv`: paired participant-bootstrap
  differences versus the reference.
- `results/phase6e/run.json`: run scope, winner, gate, and claim limit.

The AAMI/BHS columns are retrospective numerical screens only, not formal
standards compliance or clinical-device validation.
