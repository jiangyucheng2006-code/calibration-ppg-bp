# Verified project status

Last updated: 2026-08-17.

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

The five prespecified seed pipelines (`20260813`--`20260817`) completed for the
population model, M0, M1, M2, and the shared calibration controls. Scheduler
jobs 782--807 all exited `0:0`; the final aggregation and independently rebuilt
report were archived identically to work and NAS. The locked meta-test was not
accessed.

M0 has the lowest five-seed mean participant-macro MAE at every calibration
budget. Values below are mean ± sample SD across training seeds:

| K | SBP MAE | DBP MAE | Mean MAE |
|---:|---:|---:|---:|
| 1 | 12.920 ± 0.111 | 7.242 ± 0.096 | 10.081 ± 0.096 |
| 2 | 12.240 ± 0.136 | 6.812 ± 0.095 | 9.526 ± 0.112 |
| 3 | 11.894 ± 0.144 | 6.576 ± 0.095 | 9.235 ± 0.116 |
| 5 | 11.318 ± 0.110 | 6.253 ± 0.086 | 8.785 ± 0.094 |

All rows contain the same 697 meta-validation participants and 103,564 future
query events per K. Across K, M0 has mean MAE 9.407 ± 0.104 mmHg, followed by
M1 at 9.453 ± 0.143 and M2 at 9.510 ± 0.102. The M0--M1 difference is small
relative to seed variability; M0 is therefore the parsimonious provisional
finalist, not a conclusively superior model.

The unlimited-epoch runs all stopped by patience-8 early stopping. M0 selected
best epochs 10--33 across seeds, confirming that the original fixed 25-epoch
cap was not adequate for every initialization while showing that the new runs
did converge under the prespecified stopping rule.

The five-seed [extended report](RESULTS_PHASE5_REPEAT5.md) contains the required
Setting/BP/MAE/R²/ME/STD/5--10--15-mmHg/AAMI/BHS columns. All 88 method/K/BP
summary rows fail the AAMI numerical screen in every seed, and no row obtains
a BHS Grade A or B in any seed. These are retrospective numerical screens, not
formal device-validation determinations.

The complete first-run data-selection funnel and acceptance/exclusion rules are
documented in
[DATA_SELECTION_AND_TRAINING_COHORT.md](DATA_SELECTION_AND_TRAINING_COHORT.md).

## Current gate

The convergence and repeat-seed gates are complete. Before a one-time locked-
test evaluation, use development data only to complete the planned residual-
tail analysis, decide whether it motivates one prespecified training change,
and then freeze M0 (or a justified alternative), the seed/checkpoint policy,
statistics, exclusions, and reporting script.

After the repeated-seed development result is available, run a development-
only residual-tail analysis before robustness expansion. It will identify
high-error participants/events and test prespecified associations with source,
BP range, support-query BP change, time since calibration, event history, and
PPG-derived input-only quality/morphology measures. Candidate interventions
include robust loss or reweighting, hard-example training, and an input-only
uncertainty/quality gate with coverage reporting. Query error or reference ABP
must never be used to decide at inference time whether a case is kept. Any rule
is frozen on development data and applied unchanged to locked/external data.

## Not yet established

- no locked-test result or frozen final configuration;
- no real pressure-, motion-, or device-shift robustness result;
- no independent external validation;
- no clinical validation or standards claim.
