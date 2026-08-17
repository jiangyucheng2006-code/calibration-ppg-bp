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

## BP level and range association

The worst-tail participants have a modest upward shift in average BP, but they
are not simply a group of uniformly high-BP participants.

| Participant-level BP summary | Remaining 80% | Worst 20% |
|---|---:|---:|
| Mean of participant mean SBP | 117.63 | 121.45 |
| Median of participant mean SBP | 115.73 | 118.77 |
| Participants with mean SBP >=130 | 20.3% | 27.1% |
| Participants with mean SBP <110 | 33.6% | 21.4% |
| Mean of participant mean DBP | 62.12 | 64.81 |
| Median of participant mean DBP | 61.57 | 62.78 |
| Participants with mean DBP >=70 | 21.0% | 30.0% |
| Participants with mean DBP <60 | 44.0% | 35.7% |

At the event level, high-range events are over-represented in the worst tail:

| Event range | Remaining 80% | Worst 20% |
|---|---:|---:|
| SBP >=150 | 6.9% | 13.8% |
| SBP <90 | 6.6% | 8.0% |
| DBP >=80 | 7.3% | 13.9% |
| DBP <50 | 15.1% | 13.8% |

Within the worst-tail participants, SBP MAE is 34.0, 23.6, 18.1, 16.5, and
23.6 mmHg in the `<90`, `90--109`, `110--129`, `130--149`, and `>=150`
ranges. DBP MAE is 13.4, 11.1, 8.9, 9.5, and 17.8 mmHg in the `<50`,
`50--59`, `60--69`, `70--79`, and `>=80` ranges. Error therefore rises at
both BP extremes rather than being confined to one high-BP interval.

The Spearman correlation between participant mean MAE and participant mean
SBP/DBP is weak (`rho=0.109/0.094`). In contrast, correlation with
within-participant SBP/DBP variability is `0.470/0.477`, and correlation with
mean support-to-query SBP/DBP change is `0.568/0.587`. The primary association
is therefore BP variability and calibration drift; a mild upward BP shift and
more high-range events are secondary associations.

## Training response

The corrected Phase-6 set screens each intervention separately against the
fixed-first M0 configuration using one development seed. Both meta-training
and validation use the first K events as support and predict event 6 onward:

1. Huber loss;
2. BP-change-aware sampling on meta-train episodes only;
3. coordinate-wise median residual anchor;
4. PPG-only quality gate that never observes query BP or query error;
5. age/sex conditioning with participant-level cleaning and missingness masks.

Age is aggregated as the participant median of valid adult values in 18 to 100
years. Invalid age is retained as `age_z=0` with `age_valid=0`; age
normalization is fitted using meta-train participants only. Sex uses the
participant mode, with ties or unavailable values encoded as unknown. The
completed fixed-first M0 experiment is the common reference for these
candidates. The earlier rolling-support Huber run is retained only as a
historical protocol comparison and is not part of the senior-aligned candidate
set.

Multi-seed and multi-fold experiments remain deferred until a candidate shows
a meaningful development improvement across multiple K values and does not
materially worsen the participant error tail.
