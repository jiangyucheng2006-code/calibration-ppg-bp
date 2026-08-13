# Verified project status

Last updated: 2026-08-13.

## Completed gates

- PulseDB v2 controlled extraction and integrity validation completed.
- All 5,361 participant files passed the project schema gate.
- The normalized segment index contains 5,245,454 valid segments.
- Frozen source-stratified participant split:
  - 3,752 meta-train participants;
  - 805 meta-validation participants;
  - 804 quarantined meta-test participants.
- `event120-v1` was selected using development participants only.
- The 120-second protocol yields 3,840 eligible development participants and
  681 eligible locked-test participants under the fixed eligibility rule.
- Development event table: 606,010 representative event rows.
- Locked input table: 104,874 event rows; common-query targets are held in a
  separate evaluator-only artifact.
- Leakage audit passed for subject disjointness, temporal order, common-query
  equality across K, and locked-query label isolation.
- Development and locked-input PPG waveforms were materialized to the hot work
  area without query BP in the model-input store.
- GPU smoke testing passed on an NVIDIA RTX 5080.
- The server regression suite contained 25 passing tests at the training-queue
  freeze.

## Development-only training

The single-seed calibration-free population run completed after early stopping.
Its best meta-validation participant-macro result was:

| Metric | Value |
|---|---:|
| SBP MAE | 14.552 mmHg |
| DBP MAE | 8.725 mmHg |
| Mean of SBP/DBP MAE | 11.638 mmHg |

This is a development result, not a locked-test or final-paper result.

At the status timestamp, M0 was running and M1/M2, calibration controls, and the
corrected first-anchor Siamese comparator were queued. M0 had successfully
written separate `K=1,2,3,5` validation outputs for its first epoch. Finalists
require multiple seeds and reproducibility from saved per-event predictions.

## Current gate

Complete the meta-validation comparison and determine whether M0/M1/M2 add
value beyond last-cuff persistence, residual offset, and architecture-matched
adaptation. Do not submit locked-test scoring yet.

## Not yet established

- no locked-test result;
- no final multi-seed model comparison;
- no real pressure-, motion-, or device-shift robustness result;
- no independent external validation;
- no clinical validation or standards claim.
