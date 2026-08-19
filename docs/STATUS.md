# Verified project status

Last updated: 2026-08-19.

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

The fixed-first single-factor Phase-6 screen is complete and documented in
[RESULTS_PHASE6_SCREENING.md](RESULTS_PHASE6_SCREENING.md). Jobs 826--830 all
completed successfully on the same seed and query set. The PPG-only quality
gate improved the four-K Overall participant-macro mean MAE from 9.137 to
8.962 mmHg and improved all four K values, but most of the gain occurred in
MIMIC; it is a provisional component, not yet a confirmed replacement for M0.

The exact observed-error worst 30% contains 210 of 697 participants at K=5.
Their M0 mean MAE is 13.714 mmHg versus 6.656 mmHg for the oracle retained 70%.
They show more support-to-query BP change, greater within-participant BP
variability, and later query horizons. These are oracle associations and cannot
be used as a deployable filter.

Round 4 is complete and documented in
[RESULTS_PHASE6B_FACTORIAL.md](RESULTS_PHASE6B_FACTORIAL.md). All five new jobs
completed successfully; together with the three existing factorial cells they
provide the full quality-gate x Huber x participant-tail-CVaR comparison.
Quality gate plus Huber has the lowest four-K Overall participant-macro mean
MAE: 8.888 mmHg versus 9.137 for fixed-first M0 and 8.962 for quality gate
alone. The participant-cluster bootstrap difference versus M0 is -0.249 mmHg
(exploratory 95% interval -0.353 to -0.150). The gain is larger in MIMIC
(-0.472) than VitalDB (-0.065; interval includes zero), so this is a
provisional single-seed candidate rather than a frozen final model.

Participant-CVaR does not improve the full cohort, either alone or when added
to the quality gate and/or Huber. The three-factor setting is slightly best on
the fixed observed-error tail at K=5, but worse than quality gate plus Huber on
the full four-K cohort. Because true tail membership uses query error, that
result remains an oracle diagnostic and does not justify a deployment-time
router. The locked meta-test remains untouched.

An additional Phase-6C two-stage experiment has been implemented and submitted
to answer the separate deployment question: can difficult cases be recognised
without seeing their reference BP, and can a dedicated model improve them?
Five source-stratified participant folds inside `meta_train` create K=5
out-of-fold M0 errors and exact within-source worst-30% labels. A 22-feature
input-visible risk MLP is then trained without query BP, true error, source, or
participant identity. Three specialist variants test 4x difficult-group
sampling, difficult-group-only training, and difficult-group-only training
with the PPG quality gate. Each will be evaluated alone, through a frozen hard
router, and through soft risk-weighted fusion with M0.

The new implementation passed 61 server regression tests and a read-only real-
store preflight covering all 3,143 meta-train participants and all 103,564 K=5
meta-validation query rows. The dependent jobs are queued; no Phase-6C result
is reported yet. This prototype is deliberately K=5. It may be extended to
K=1/2/3 only with support-budget-specific features that do not use unavailable
cuff measurements. Promotion requires useful held-out participant AUPRC and a
full-coverage gain in Overall, MIMIC, and VitalDB; a single-seed gain would
still require confirmation.

Phase-6C subsequently completed without scheduler or artifact-integrity
failures. Its input-only risk classifier showed moderate, not yet decisive,
held-out meta-train discrimination (Overall participant AUPRC 0.469;
precision 0.444 and recall 0.442 at the frozen threshold). The best original
specialist route improved K=5 Overall participant-macro mean MAE only slightly,
from 8.783 to 8.736 mmHg. This supports testing the two-stage idea further but
does not yet establish reliable automatic identification or specialist value.

Phase-6D is now submitted as the corrected end-to-end test. It first rebuilds
the five-fold out-of-fold difficult labels with the current winning Quality
Gate + Huber model instead of the older M0 reference. It then retrains the
input-only risk classifier and compares three matched Quality Gate + Huber
specialists: 2x difficult-group weighting, 4x weighting, and difficult-only
training. The final dependent report evaluates identification, specialist
performance on the fixed evaluation-only tail, deployable event-level hard and
soft routing, and clearly separated retrospective/oracle upper bounds. The
full [Phase-6D plan](PHASE6D_RISK_ROUTING_PLAN.md) records the leakage boundary
and promotion gate. Jobs 880--890 were submitted on 2026-08-19; the first two
cross-fitting jobs started on `hpc-2`. No result is reported before the final
report job completes.

## Not yet established

- no locked-test result or frozen final configuration;
- no real pressure-, motion-, or device-shift robustness result;
- no independent external validation;
- no clinical validation or standards claim.
