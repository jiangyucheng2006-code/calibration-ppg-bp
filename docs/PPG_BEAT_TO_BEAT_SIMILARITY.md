# Within-window PPG beat-to-beat morphology similarity

## Question

How similar are the individual pulse beats within the 10-second filtered PPG
windows currently used by the calibrated BP experiments?

This is a development-data morphology audit. It does not validate physiological
accuracy, absence of motion artefact, or clinical signal quality.

## Data boundary

- Split: `meta_validation` model inputs only.
- Locked meta-test accessed: no.
- Windows: the same 103,564 K=5 query events used by the current development
  evaluation.
- Participants: 697 (316 PulseDB MIMIC, 381 PulseDB VitalDB).
- Signal: filtered `PPG_F`, 1,250 samples at 125 Hz (10 seconds).
- MIMIC and VitalDB are internal PulseDB source strata, not independent external
  validation datasets.

## Method

Each 10-second signal is lightly smoothed for peak detection. Candidate systolic
peaks are detected under a 0.30-second refractory period. Inter-peak minima define
trough-to-trough complete beats. Beats outside 0.30--2.00 seconds are excluded.
Each retained beat is resampled to 100 points and z-normalized, so the comparison
primarily measures shape rather than absolute amplitude or duration.

The primary window-level statistic is the median Pearson correlation across all
pairs of complete beats in that window. The median correlation to the median-beat
template and normalized template RMSE are secondary checks. Original and inverted
PPG polarity are both considered, with the valid segmentation containing more
complete beats selected. A window requires at least three complete beats.

The 0.80, 0.90, and 0.95 correlation cutoffs below are descriptive summaries only;
they are not universal or clinically validated quality thresholds.

## Results

| Scope | Windows | Participants | Valid windows | Pairwise median | Window p10 | Template median | >=0.80 | >=0.90 | >=0.95 | Participant median | Participant p10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 103,564 | 697 | 100.00% | 0.9923 | 0.5792 | 0.9971 | 83.47% | 79.81% | 74.70% | 0.9905 | 0.5301 |
| PulseDB MIMIC | 80,874 | 316 | 100.00% | 0.9932 | 0.7741 | 0.9974 | 89.27% | 85.45% | 79.95% | 0.9935 | 0.9031 |
| PulseDB VitalDB | 22,690 | 381 | 100.00% | 0.9783 | 0.3088 | 0.9948 | 62.79% | 59.70% | 55.98% | 0.9822 | 0.4725 |

The typical 10-second PPG window has high within-window beat-to-beat morphology
similarity. However, the lower tail is substantial: 20.19% of all windows fall
below 0.90, and the overall window-level 10th percentile is only 0.579. The source
difference is large. MIMIC has both a higher typical similarity and a much better
lower tail than VitalDB; only 59.70% of VitalDB windows reach 0.90, compared with
85.45% of MIMIC windows.

All windows were segmentable, but this must not be interpreted as all windows being
high quality: periodic noise can still produce detectable beats, and a repeated
artefact can itself be morphologically consistent. The next defensible analysis is
to test whether low beat-to-beat similarity predicts BP error after accounting for
source, participant, heart-rate/beat-count, and existing PPG quality-gate features.

## Reproducibility

- Slurm job: 946, completed with exit code `0:0`; stderr was empty.
- Work and NAS result directories were byte-identical.
- Aggregate public results: `results/beat_similarity/summary.csv`.
- Participant/event-level similarity values remain private and are not committed.
- Submitted-code snapshot SHA-256:
  `1f9775d72279d5706bf496c29bb7e62bc04535603567c050b7268cfed781f63b`.

