# Frozen development protocol

## Research question

Can a personalized neural model use only `K = 1, 2, 3, or 5` independent
reference-BP events and PPG waveforms to improve future SBP/DBP prediction over
strong simple calibration controls, while remaining stable under later
contact-pressure, motion, and device/acquisition shifts?

The initial gate concerns few-shot calibration itself. Robustness training and
external validation are subsequent phases.

## Independent calibration event

The unit of calibration is an independent reference-BP event, not a raw PPG
window. For PulseDB, one deterministic representative segment is selected from
each non-overlapping 120-second temporal block. Because the labels are derived
from ABP rather than literal cuff readings, these are described as
**pseudo-cuff events**.

Adjacent or overlapping 10-second segments from the same temporal neighbourhood
cannot be counted as multiple calibration events.

## K-shot construction

The first five chronological events are the fixed support-candidate pool. For
each participant:

- `K=1` exposes event 1;
- `K=2` exposes events 1--2;
- `K=3` exposes events 1--3;
- `K=5` exposes events 1--5.

All K values use the same query set, event 6 onward. Every support event must
precede every query event.

## Split hierarchy

Participants are assigned before model training to source-stratified,
participant-disjoint meta-train, meta-validation, and locked meta-test splits.
Population training, target scaling, learned preprocessing, model selection,
quality thresholds, and hyperparameter selection use no locked-test
participant.

The locked query input artifact excludes query SBP/DBP. Targets are held in a
separate evaluator-only artifact and may be joined only after predictions are
written.

## Evaluation

Primary results are participant-macro metrics. At minimum, report SBP and DBP:

- mean absolute error (MAE);
- root mean squared error (RMSE);
- signed bias;
- participant-level paired uncertainty or confidence intervals.

Pooled event-level metrics are secondary diagnostics. All K comparisons use the
same participants and same common-query events wherever the method is defined.

Development work uses meta-validation only. Locked meta-test scoring is a
one-time confirmatory stage after the final model, comparator, seeds,
preprocessing, adaptation hyperparameters, and checkpoint rule are frozen.

## Stop/go gates

1. Simple calibration and population baselines must be reproducible from saved
   prediction tables.
2. The personalized model must be compared directly with last-cuff persistence,
   support mean, and residual offset.
3. If PPG-aware methods do not beat the simple controls, diagnose temporal BP
   variation, label construction, PPG quality, regression to the mean, and query
   timing before increasing complexity.
4. Robustness training starts only after the base personalization method is
   credible.
5. External validation must use an independently mapped, labelled dataset and
   must not be used to retroactively tune the locked PulseDB experiment.

## Claim limits

- PulseDB pseudo-cuff events do not establish real-world cuff calibration
  usability.
- MIMIC- and VitalDB-derived PulseDB subsets are not independent external
  datasets with respect to their source databases.
- A lower internal MAE is not device validation, clinical validation, or proof
  of standards compliance.
- Pressure, motion, or device robustness must be demonstrated on explicit
  stress/domain groups; synthetic augmentation alone is insufficient evidence.
