# Verified project status

Last updated: 2026-08-14.

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

The single-seed population, M0, M1, M2, calibration-control, and corrected
first-anchor Siamese runs completed successfully. Their formal stderr files are
empty, and the work/NAS result trees and checkpoint archives passed bytewise
verification.

M0 is the current meta-validation winner. Its participant-macro SBP/DBP/mean
MAE was:

| K | SBP MAE | DBP MAE | Mean MAE |
|---:|---:|---:|---:|
| 1 | 12.865 | 7.131 | 9.998 |
| 2 | 12.155 | 6.714 | 9.435 |
| 3 | 11.770 | 6.468 | 9.119 |
| 5 | 11.219 | 6.157 | 8.688 |

All values are mmHg and come from 697 meta-validation participants and the
same 103,564 future query events per K. The locked meta-test was not accessed.
See [RESULTS_PHASE5.md](RESULTS_PHASE5.md) for the comparator table,
uncertainty, source diagnostics, limitations, and next gate.

## Current gate

Resolve M0 convergence beyond the original 25-epoch cap and run the
prespecified repeat-seed comparison. Freeze the final configuration, seed
policy, statistics, and source/tail reporting before submitting a one-time
locked-test evaluation.

## Not yet established

- no locked-test result;
- no final multi-seed model comparison or frozen finalist;
- no real pressure-, motion-, or device-shift robustness result;
- no independent external validation;
- no clinical validation or standards claim.
