# Verified project status

Last updated: 2026-08-22.

## Round-10 partial end-to-end screen completed

Round 10 is complete. Recovery jobs 1017--1025 and deterministic report job
1026 all completed with exit code `0:0`, every stderr file is empty, and every
work output matches its NAS archive. The full server suite passes 119 tests.
All nine candidates use the identical K=5 fold-4 comparison of 628 participants
and 96,332 queries after folds 0--2 fitting and fold-3 patience-8 early
stopping. Meta-validation and the locked meta-test were not accessed.

T10-8, last-block adaptation with pair-direction and temporal-consistency
objectives, is the numerical winner. Its Overall participant-macro SBP/DBP/
mean MAE is 10.7820/6.1093/8.4457 mmHg, versus 10.8946/6.1697/8.5322 for the
frozen-encoder reference. Mean MAE improves by 0.0865 mmHg Overall, 0.0725 in
MIMIC, and 0.0981 in VitalDB.

Although both internal source strata improve, the Overall gain is below the
prespecified 0.15-mmHg promotion threshold. No Round-10 candidate is promoted
or evaluated on meta-validation. See the complete
[Round-10 result](RESULTS_ROUND10_PARTIAL_END_TO_END.md), the prospective
[plan](ROUND10_PARTIAL_END_TO_END_PLAN.md), and aggregate tables under
`results/round10/`.

The earlier failed chains produced no scientific result. Jobs 991--1003
exposed an argument-validation defect, while jobs 1007--1016 exposed a
mixed-precision loss-boundary defect. Both were corrected and regression-
tested before the accepted recovery run. The exact accepted snapshot is
`event120-v1_round10_dtype_fix_20260821-160940.tar.gz`, SHA-256
`2c239ba85501baca2a561b0c3076778c6661d225e975ccc120178f4df30b6e7d`.

## Round-9 calibration refinement completed

Round 9 is complete. Jobs 975--985 all completed with exit code `0:0`, empty
stderr, and byte-identical work/NAS reports. The screen compares one
architecture-matched R8 reference with eight isolated changes at K=5.

To reduce repeated model-selection pressure on the previously viewed
meta-validation set, folds 0--2 of participant-disjoint meta-train fit each
candidate, fold 3 controls patience-8 early stopping, and fold 4 ranks the
candidates. Meta-validation is not used for training, early stopping,
prediction, scoring, or candidate ranking in this screen. The locked meta-test
is not accessed. The server suite passes 97 tests. The fold-4 comparison covers
628 participants and 96,332 common queries.

No candidate passes the internal promotion gate. R9-1 adaptive fusion is the
numerical winner but improves Overall participant-macro mean MAE by only 0.036
mmHg and worsens VitalDB by 0.003 mmHg. R9-7 is the only method to improve both
source strata, but its Overall gain is only 0.010 mmHg. No Round-9 candidate is
promoted or evaluated on meta-validation. See the complete
[Round-9 result](RESULTS_ROUND9_CALIBRATION_REFINEMENT.md) and aggregate files
under `results/round9/`. The next justified route is partial end-to-end
PPG-encoder adaptation to query-to-calibration BP change.

## Ten-second PPG beat-to-beat similarity audit

The within-window morphology audit is complete for all 103,564 K=5
meta-validation query PPG windows from 697 participants; the locked meta-test
was not accessed. The median within-window pairwise beat correlation is 0.9923
Overall, 0.9932 for PulseDB MIMIC, and 0.9783 for PulseDB VitalDB. However, the
window-level 10th percentiles are 0.5792, 0.7741, and 0.3088, respectively.
Thus typical morphology is highly repeatable, but there is a marked low-
similarity tail, especially in VitalDB. Full aggregate results and method
limitations are documented in
[PPG_BEAT_TO_BEAT_SIMILARITY.md](PPG_BEAT_TO_BEAT_SIMILARITY.md); private
participant/event rows are not published.

The follow-up similarity--error analysis is also complete. In both the Quality
Gate + Huber reference and the R7-5 causal GRU, event-level correlations are
near zero and within-participant comparisons do not show higher error below
0.90 similarity. Participant-level correlations are weakly positive rather
than negative. Therefore the current normalized morphology-similarity score is
not promoted as a quality gate, rejection rule, or specialist-routing feature.
See [PPG_BEAT_SIMILARITY_ERROR_RELATION.md](PPG_BEAT_SIMILARITY_ERROR_RELATION.md).

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

## Round-11A backbone screen completed

All formal jobs 1033--1048 completed with exit code `0:0`, and all 16 work/NAS
result pairs are byte-identical. The patch-Transformer stderr contains only a
PyTorch nested-tensor optimization warning; the other formal stderr files are
empty. Every setting uses the same 628 fold-4 participants and 96,332 K=5
queries after folds 0--2 fitting and fold-3 patience-8 early stopping.

The compact ResNet remains the numerical winner. QGH participant-macro
SBP/DBP/mean MAE is 11.0136/6.2771/8.6453 mmHg Overall, with mean MAE 9.3631
in MIMIC and 8.0489 in VitalDB. InceptionTime is the closest alternative but
worsens the three means by 0.0774/0.0741/0.0801 mmHg. Deeper ResNet, patch
Transformer and Conformer worsen Overall by 0.2583, 0.2679 and 0.3336 mmHg,
respectively. No alternative improves either internal source stratum.

`winner_backbone=resnet_small` and `passes_internal_gate=false`; no new
backbone is promoted. The result supports retaining the smaller encoder and
rejects complexity-for-complexity's-sake under the current protocol. It does
not establish a universal negative conclusion about all attention models.
Meta-validation and the locked meta-test were not accessed. The accepted
report is [RESULTS_ROUND11_BACKBONE_SCREEN.md](RESULTS_ROUND11_BACKBONE_SCREEN.md),
with public aggregate files under `results/round11a/`.

Stage 11B may now use the retained compact ResNet to perform a structural subtraction
ablation of the personal correction MLP, PPG-only quality gate and
query-conditioned support attention, followed by a paired MSE-versus-Huber
loss comparison on the selected minimal structure.
Stage 11C will then compare universal-only, universal + stable personal bias,
universal + dynamic change and the full three-part decomposition. These later
stages are intentionally not submitted before the upstream winner is known.
The complete prospective design is in
[ROUND11_SYSTEMATIC_MODEL_REVISION_PLAN.md](ROUND11_SYSTEMATIC_MODEL_REVISION_PLAN.md).

## Not yet established

- no locked-test result or frozen final configuration;
- no real pressure-, motion-, or device-shift robustness result;
- no independent external validation;
- no clinical validation or standards claim.

## Round-12 literature-derived backbone screen completed

A broad calibrated/personalized PPG-BP literature audit was completed before
new jobs were defined. The central finding is that no audited paper currently
demonstrates PulseDB + PPG-only + participant-disjoint development + strictly
chronological first K<=5 labeled events + no later label access + SBP/DBP
AAMI/BHS acceptance at the same time. Several very low PulseDB errors instead
use same-participant high-volume window splits, multimodal inputs, or ongoing
reference-BP updates. The evidence table and source links are preserved in
[LITERATURE_AUDIT_CALIBRATED_PPG_BP_20260822.md](LITERATURE_AUDIT_CALIBRATED_PPG_BP_20260822.md).

Round 12 therefore performs a controlled architecture-family screen rather
than copying incomparable headline scores. It compares the unchanged compact
ResNet with a causal TCN, a five-shot residual-attention encoder, a compact
CNN-GRU and a 1D residual U-Net. All five use the same PPG-only QGH calibration
head, fixed-first K=5 support, folds 0--2/3/4, Huber loss, quality gate and
single seed. Meta-validation and the locked meta-test remain quarantined. The
prospective specification is in
[ROUND12_LITERATURE_BACKBONE_PLAN.md](ROUND12_LITERATURE_BACKBONE_PLAN.md).

Server verification passed all 131 regression tests. CUDA smoke job 1049
completed with exit code `0:0` after forward propagation, backpropagation and
an optimizer update for all existing and new backbones. Formal jobs 1050--1065
then all completed with exit code `0:0`, all stderr files are empty, and all 16
work/NAS output pairs are byte-identical.

The compact ResNet remains the numerical winner at Overall participant-macro
SBP/DBP/mean MAE 11.0294/6.2592/8.6443 mmHg. The residual U-Net, TCN,
five-shot residual-attention encoder and CNN-GRU worsen Overall mean MAE by
0.1120, 0.1778, 0.4141 and 0.4245 mmHg, respectively, and every candidate also
worsens mean MAE in both internal PulseDB source strata.
`passes_internal_gate=false`; no new architecture advances to meta-validation,
multi-seed confirmation, or the locked meta-test. The accepted aggregate report is
[RESULTS_ROUND12_LITERATURE_BACKBONES.md](RESULTS_ROUND12_LITERATURE_BACKBONES.md),
with machine-readable tables under `results/round12/`.

## Round-8 calibration-relative screen completed

All Round-8 jobs completed successfully with empty stderr, and the locked
meta-test was not accessed. This was a K=5, single-seed development screen
built around the Quality Gate + Huber job-841 reference. Every full-coverage
candidate used the same 697 participants and 103,564 query events.

R8-4, which combines pairwise calibration-relative prediction, causal query
history, and a support-BP-range auxiliary task, is the numerical winner. Its
Overall participant-macro SBP/DBP/mean MAE is
10.487/6.021/8.254 mmHg, compared with 10.803/6.168/8.485 mmHg for the
reference. Mean MAE also improves in both internal PulseDB source strata:
9.075 to 8.852 mmHg in MIMIC and 7.996 to 7.758 mmHg in VitalDB.

The `similarity >=0.90` sensitivity retains only 79.81% of Overall queries and
has strongly source-dependent coverage, so it is not a full-coverage model and
is not promoted. Direct demographic concatenation is nearly neutral, while a
128-dimensional PPG representation is slightly worse than the 256-dimensional
reference. R8-4 advances as a candidate base for a further development round;
multi-seed confirmation is deferred until that method round is complete.

The full result, scope limitations, candidate interpretation, and links to all
public Overall/MIMIC/VitalDB tables are in
[RESULTS_ROUND8_CALIBRATION_RELATIVE.md](RESULTS_ROUND8_CALIBRATION_RELATIVE.md).
The prespecified design remains available in
[ROUND8_CALIBRATION_RELATIVE_PLAN.md](ROUND8_CALIBRATION_RELATIVE_PLAN.md).

## Round-13 final architecture/capacity screen submitted

Round 13 is the final broad development-only test of whether the calibrated
model is limited by its population PPG encoder. Earlier work already tested a
710,530-parameter patch Transformer, a 1,587,330-parameter Conformer, a
3,827,002-parameter deeper ResNet, and a 16.13-million-parameter
residual-attention network; none improved the same-round compact ResNet. The
new round therefore uses controlled changes instead of assuming that more
parameters must help.

Thirteen same-seed candidates now compare the compact ResNet reference with a
depth-only ResNet, a width-only ResNet, base/wide InceptionTime, base/deep/wide
Patch Transformers, two tokenization-only Transformer variants, base/large
Conformer, and ConvNeXt-1D. Every encoder outputs 256 features and feeds the
same K=5 fixed-first Quality Gate + Huber calibration model. Population and
QGH training use common physical microbatches of 32 and 16 with four-step
gradient accumulation, giving effective batches of 128 and 64 for all
candidates. Each epoch samples 99,968 examples, exactly divisible by both
effective batch sizes.

The implementation is commit `7628be5`. Server verification passed all 173
regression tests. CUDA smoke jobs 1072 (RTX 5080) and 1073 (RTX 5070 Ti) both
completed with exit code `0:0`, reproduced every pre-registered parameter
count, and completed forward, backward, and optimizer steps using the formal
32/16 microbatches. The only stderr text is the known PyTorch nested-tensor
optimization warning for norm-first Transformer layers.

Formal jobs 1074--1114 are submitted as 13 independent
population -> QGH -> fold-4 evaluation chains. Jobs 1074 and 1077 began
concurrently on the RTX 5080 and RTX 5070 Ti. Job 1113 generates the common
Overall/MIMIC/VitalDB report only after all 13 evaluations succeed, and job
1114 verifies work/NAS artifacts and archives the Slurm logs. The immutable
source snapshot is `round13_7628be5`, archive SHA-256 is
`b495915484416032d766a4aa89c2f62db9a56743627e28b219b9edf1fc97e21f`, and
source-tree SHA-256 is
`54a5251254e2a92007f6d9e465c7ca627675ad8ea289f59e26d7328062bbce6c`.

The prespecified internal gate is at least 0.15 mmHg improvement in Overall
participant-macro mean MAE versus the same-round compact ResNet, with
improvement in both MIMIC and VitalDB. Meta-validation and the locked meta-test
remain untouched. If no candidate passes, architecture scaling is closed
after this round. The full prospective specification is in
[ROUND13_FINAL_CAPACITY_SCREEN_PLAN.md](ROUND13_FINAL_CAPACITY_SCREEN_PLAN.md).
