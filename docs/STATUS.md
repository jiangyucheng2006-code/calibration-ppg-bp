# Verified project status

Last updated: 2026-08-19.

## Round-7 new training submitted

Round 7 supersedes the earlier interpretation that the completed Phase-6E
cluster head fully represented the requested waveform-category system. Nine
new candidates are now submitted under seed `20260821`: six newly trained OOF
residual methods, a separately trained waveform-phenotype router with hard and
soft independent category experts, and a deeper waveform-embedding causal GRU.

The full server suite passed 72 tests before submission. Slurm jobs 932--942
form the new chain; no previous candidate job is substituted into its final
report. The exact submitted-code snapshot is
`event120-v1_round7_nine_routes_20260819-1437.tar.gz`, SHA-256
`56888eb537b573d5d4295f53ee0ae99967a00c23f7f5e5732ad3bf2b3baa9a91`.
The design and leakage boundary are documented in
[ROUND7_NINE_ROUTE_PLAN.md](ROUND7_NINE_ROUTE_PLAN.md).

## Phase-6E result

The seven-route K=5 development screen is complete. All jobs 915--924 and the
cluster-audit correction jobs 925--926 completed with exit code `0:0`, empty
stderr, and byte-identical work/NAS report artifacts. The locked meta-test was
not accessed.

The causal GRU residual corrector is the numerical winner: Overall
participant-macro SBP/DBP/mean MAE is 10.674/6.141/8.408 mmHg versus
10.803/6.168/8.485 for Quality Gate + Huber. MIMIC mean MAE improves from
9.075 to 8.969 and VitalDB from 7.996 to 7.942. The Overall gain is only
0.078 mmHg, below the frozen 0.15-mmHg promotion threshold, so no route is
promoted and Quality Gate + Huber remains the development base.

The morphology-cluster MoE used an explicitly exploratory K=8 fallback because
none of K=8/16/32 reached the 0.75 meta-train stability gate. Its Overall gain
was only 0.005 mmHg and MIMIC worsened. It does not establish stable learned
waveform phenotypes. Full aggregate results are in
[RESULTS_PHASE6E_SEVEN_ROUTES.md](RESULTS_PHASE6E_SEVEN_ROUTES.md) and
`results/phase6e/`.

## Phase-6E seven-route screen submitted

Phase-6E now tests seven K=5 continuous-error corrections against the frozen
Quality Gate + Huber development comparator: ridge residual correction,
residual MLP, confidence-gated residual MLP, difficult-participant 2x weighted
MLP, causal GRU residual correction, supervised mixture-of-experts, and an
unsupervised morphology-cluster mixture-of-experts. The last route learns
waveform phenotypes from frozen PPG embeddings rather than imposing a fixed
70%/30% partition; a new query is softly assigned to cluster-specific residual
experts and every query retains a prediction.

The server regression suite passed 70 tests and all seven routes passed a
real-data end-to-end preflight. The corrected submission chain is jobs
915--924: preparation 915, routes 916--921, waveform embedding 922,
morphology-cluster route 923, and unified report 924. Preparation job 915
completed successfully with empty stderr. The exact submitted-code archive is
`event120-v1_phase6e_seven_routes_postfix_20260819-1351.tar.gz` with SHA-256
`acf357bfc2085656e8e078775f1d5d1fed935940a359b87a917a0fb30864e1d8`.
An earlier dependency chain 900--907 was superseded after job 900 exposed an
NAS permission-preservation incompatibility; jobs 901--907 were cancelled
without running, and the corrected chain also reduces the cluster job's memory
request to the verified node limit.

All candidate targets are derived from participant-disjoint Quality Gate +
Huber out-of-fold meta-train predictions. Folds 0--3 fit candidate parameters,
fold 4 performs internal selection, and meta-validation is evaluated only
after freezing. The locked meta-test remains inaccessible. A candidate advances
only if Overall participant-macro mean MAE improves by at least 0.15 mmHg and
both MIMIC and VitalDB improve; otherwise Phase-6E remains a negative screen.

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

Phase-6D is complete and reported in
[RESULTS_PHASE6D_RISK_ROUTING.md](RESULTS_PHASE6D_RISK_ROUTING.md). Jobs
880--890 all completed with exit code `0:0` and empty stderr. The corrected
Quality Gate + Huber risk classifier achieves Overall participant AUPRC 0.457,
AUROC 0.659, precision 0.470, and recall 0.486. Predicted-high-risk
participants have mean MAE 10.140 versus 7.737 mmHg in the predicted-low group,
showing useful but incomplete separation.

Moderate 2x difficult-participant weighting is the only specialist that
improves the fixed evaluation-only difficult tail in both sources. Fourfold
weighting is too aggressive, and difficult-only training worsens the tail. The
binary event hard route changes Overall mean MAE only from 8.485 to 8.481 mmHg
and worsens VitalDB, so the requested 70%/30% hard-routing system is not
promoted. Continuous event-risk soft fusion is the Phase-6D winner at 8.424
mmHg Overall, 8.986 MIMIC, and 7.958 VitalDB, improving the general model by
0.061/0.089/0.038 mmHg. Exploratory 20,000-repetition paired participant
bootstrap intervals exclude zero in all three scopes, but the gain is small
and comes from the same single-seed development screen. Soft fusion advances
only to independent-seed confirmation; locked meta-test remains untouched.

## Not yet established

- no locked-test result or frozen final configuration;
- no real pressure-, motion-, or device-shift robustness result;
- no independent external validation;
- no clinical validation or standards claim.
