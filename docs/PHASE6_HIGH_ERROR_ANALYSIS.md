# Phase-6 high-error participant analysis

## Scope and definition

This is a development-only diagnostic analysis of the existing M0 model at
`K=5`, using seed `20260813`. The analysis includes 697 meta-validation
participants and 103,564 common-query events. The locked meta-test was not
accessed.

Participants were ranked by their participant-level mean of SBP MAE and DBP
MAE. The worst 20% tail contains 140 participants with mean MAE greater than or
equal to 11.751 mmHg. This tail is defined using observed validation error and
is therefore an oracle diagnostic group, not an inference-time exclusion rule.

## Main comparison

| Participant group | Mean MAE | SBP MAE | DBP MAE | Within-subject SBP SD | Within-subject DBP SD | Mean absolute support-to-query SBP change | Mean absolute support-to-query DBP change | Query outside support SBP range | Query outside support DBP range | Mean event index | Mean filtered-PPG SD | Mean age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Remaining 80% | 6.935 | 8.936 | 4.933 | 12.056 | 6.679 | 13.704 | 7.288 | 55.9% | 54.4% | 72.5 | 0.2753 | 61.8 |
| Worst 20% | 15.662 | 20.300 | 11.024 | 16.011 | 9.426 | 22.660 | 12.791 | 70.4% | 69.4% | 108.7 | 0.2748 | 63.1 |

All BP and error values are in mmHg; age is in years. Support statistics use
the first five calibration events only.

## Interpretation

The strongest observed pattern is calibration drift rather than a simple PPG
amplitude or age effect. Relative to the remaining participants, the worst
20% group has:

- 65% larger mean support-to-query SBP change and 76% larger DBP change;
- 33% larger within-participant SBP variability and 41% larger DBP variability;
- a 50% later mean event index, indicating a longer or more difficult
  post-calibration trajectory;
- more query targets outside the BP range represented by the five calibration
  events;
- nearly identical filtered-PPG standard deviation and only a 1.3-year higher
  mean age.

These findings support methods that address post-calibration BP change,
outlier-resistant support aggregation, and PPG-only reliability modulation.
They do not establish causality. Filtered-PPG standard deviation is only an
amplitude proxy and cannot rule out morphology changes, motion artefact, poor
contact, or device effects.

## Source association

| Source | Participants | Mean MAE | SBP MAE | DBP MAE | Worst-20% rate |
|---|---:|---:|---:|---:|---:|
| MIMIC | 316 | 9.344 | 12.039 | 6.648 | 26.6% |
| VitalDB | 381 | 8.144 | 10.539 | 5.749 | 14.7% |

The source difference may reflect population, recording, clinical-state, or
device/domain differences and should not be interpreted as a device effect
without source-specific confounder analysis.

## Training response

Phase-6 screens each intervention separately against the original rolling-
support M0 configuration using one development seed:

1. Huber loss;
2. BP-change-aware sampling on meta-train episodes only;
3. coordinate-wise median residual anchor;
4. PPG-only quality gate that never observes query BP or query error;
5. age/sex conditioning with participant-level cleaning and missingness masks.

Age is aggregated as the participant median of valid adult values in 18 to 100
years. Invalid age is retained as `age_z=0` with `age_valid=0`; age
normalization is fitted using meta-train participants only. Sex uses the
participant mode, with ties or unavailable values encoded as unknown. The
fixed-first support experiment is retained separately as a protocol ablation.

Multi-seed and multi-fold experiments remain deferred until a candidate shows
a meaningful development improvement across multiple K values and does not
materially worsen the participant error tail.
