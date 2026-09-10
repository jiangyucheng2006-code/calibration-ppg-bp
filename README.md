# Calibration PPG BP

## Current experiment — full-cohort personal enrollment

[The full-cohort enrollment plan](docs/PLAN_FULL_COHORT_ENROLLMENT_V1.md) scans
all5,361 original people, preserving disjoint training/validation/test identities.
Only personal LoRA and the same LoRA plus reference memory are compared.
The incorrectly narrowed2,504-person subset batch was cancelled with authorization.
All 72 server checks and GPU smoke passed. Source-file availability is confirmed
for all 5,361 people; complete-data preparation is running as job1764, with
audit and training queued behind successful prerequisites. Missing work copies
are staged from verified NAS masters rather than excluding their participants.
Full-cohort preparation/training status is recorded in
[verified progress and job IDs](docs/STATUS.md). Assignment counts precede
necessary validity and registration/query eligibility checks. No new result is available yet.

## Latest results — 200-person new-user enrollment, 10 September 2026

The expanded enrollment study and its matched ablations are complete.
**[200 人建档结果与消融总结（中文）](results/post_enrollment_200_v1_20260910/README.md)**
contains all nine settings, three source views, requested metric tables,
paired uncertainty, failure history and verification evidence.

The 200 people (100 MIMIC / 100 VitalDB) were excluded from fresh population
fitting. Each subsequently provided 360 labelled registration windows for a
new personal profile, followed by 40 disjoint test windows. All 8,000 queries
are retained. Values below are participant-macro SBP / DBP MAE in mmHg.

| Method | Overall | MIMIC | VitalDB |
|---|---:|---:|---:|
| Personal LoRA | 3.9049 / 2.1858 | 4.2868 / 2.3628 | 3.5230 / 2.0088 |
| Shared-feature memory only; no personal LoRA | 3.4846 / 1.9178 | 3.8867 / 2.1361 | 3.0825 / 1.6996 |
| LoRA-feature memory only; no prediction-head fusion | 3.3696 / 1.8519 | 3.7708 / 2.0602 | 2.9685 / 1.6436 |
| **Personal LoRA + reference memory** | **3.3312 / 1.8442** | **3.7162 / 2.0353** | **2.9463 / 1.6531** |

The complete candidate reduces Overall mean MAE from **3.0453 to 2.5877**,
a **15.03%** paired improvement over LoRA. The gain is 0.4576 mmHg,
95% participant-paired source-stratified bootstrap CI [0.3930, 0.5232];
186/200 people improve and 14 worsen. Both source-stratified means improve.
The complete candidate has the lowest mean MAE in each scope among the nine
settings, **not the best value for every BP/STD metric**.

Memory-only controls are strong: the complete candidate improves mean MAE by
only 0.1135 over shared-feature retrieval and 0.0230 over adapted-feature
retrieval. Weighting/fusion add modest gains, so the results do not establish
that every component is indispensable. Full LoRA and memory results meet the
saved numerical AAMI/BHS-A screens, not clinical-device certification.

This is a custom whole-person-excluded derivative of official membership,
not the unchanged official benchmark, K-shot cuff calibration or a future-date
wrist study. One content-linked outside identity was quarantined without
reselecting the 200 people; the remaining population is 2,305. Three of the 200
also occurred in the earlier 30-person sample. This is exploratory single-seed
evidence, not a pristine independent confirmation.

[All nine methods and all requested columns](results/post_enrollment_200_v1_20260910/enrollment/RESULT_TABLES.md)
· [Paired intervals](results/post_enrollment_200_v1_20260910/enrollment/paired_participant_intervals.csv)
· [Verified project status](docs/STATUS.md).
No new training was submitted during this result review.

## Previous results and method explanation — 9 September 2026

[中文模型说明与手表采集计划](docs/PERSONAL_LORA_MEMORY_METHOD_ZH.md) explains
the implemented ResNet backbone, personal low-rank feature adaptation, reference
retrieval, new-user enrollment, and the limits of the current evidence.
[Download the Chinese Word report](docs/个人LoRA与参考记忆血压估计研究说明.docx).

Both recent batches are complete. Values below are participant-macro SBP / DBP
MAE in mmHg; **the two cohorts are not paired with each other**.

| Experiment and method | Overall | MIMIC | VitalDB |
|---|---:|---:|---:|
| Official CalBased · LoRA | 3.8207 / 2.0994 | 4.1920 / 2.2802 | 3.4724 / 1.9298 |
| Official CalBased · LoRA + fixed memory | 3.4210 / 1.8858 | 3.8073 / 2.0699 | 3.0587 / 1.7132 |
| New 30-person enrollment · LoRA | 4.0749 / 2.2775 | 3.8486 / 2.2589 | 4.3011 / 2.2961 |
| New 30-person enrollment · LoRA + memory | 3.1515 / 1.7466 | 3.0666 / 1.7583 | 3.2364 / 1.7349 |

- [Official CalBased results](results/official_calbased_v1_20260909/README.md):
  2,506 registered people, 360 training / 40 test windows each. Exact source
  membership is retained, including 35 disclosed cross-role duplicate-content
  rows. Four learned memory variants select epoch 0; their matching scores are
  not evidence of four separate learned improvements.
- [Post-training enrollment results](results/post_enrollment_30_v1_20260909/README.md):
  a fresh population model excludes all 30 selected people. Each then supplies
  360 labelled registration windows for a new personal adapter and reference
  bank, followed by 40 disjoint test windows. Overall mean MAE improves 22.89%
  over matched LoRA; no people are filtered. This is an exploratory derived
  protocol, not the unchanged official benchmark or a few-cuff study.
- Both reports include Overall/MIMIC/VitalDB tables with R², ME, STD, threshold
  percentages, and qualified AAMI/BHS numerical screens. These are not device
  certification, external wrist validation, or proof of prospective stability.

See [verified execution and publication status](docs/STATUS.md). Planned human
data collection remains unstarted and requires the appropriate ethics and
measurement protocol. Raw signals, personal profiles and checkpoints are not
published.

## Completed previous work — registered-user personal memory

The [personal-memory v2 results](docs/RESULTS_PERSONAL_MEMORY_V2.md) are complete
(2026-09-08): all11 jobs and four new fits succeeded. E1 reaches Overall mean
MAE **2.6475 / 3.6334 mmHg**, and fixed E3 refinement reaches **2.6381 / 3.6448**
(random / chronological). No candidate passes the prespecified joint upgrade
rule. E1 random selects its untrained epoch0; E2 does not outperform the old
blend; E3 adds only **0.0094 / 0.0124** improvement over v1 and worsens the
large-DBP-deviation subgroup. LoRA remains the primary reference.

[All requested v2 tables](results/personal_memory_v2/RESULT_TABLES.md) report
Overall/MIMIC/VitalDB separately, with STD and qualified AAMI/BHS fields.
Time-gap controls expose substantial dependence on nearby references; these
controls were applied to v1, not to new E1/E2. No new tasks or held-out access
were performed for this publication. See the
[current status](docs/STATUS.md) and [execution record](docs/PERSONAL_MEMORY_V2_EXECUTION.md).

### Previous v1 screen — historical comparison

The [complete personal-memory results](docs/RESULTS_PERSONAL_MEMORY_V1.md)
are now available: all ten model fits completed, and the final aggregate
report was repaired without retraining or changing predictions.

The memory/LoRA blend is best among the six main settings: Overall
participant-macro mean MAE **2.6475 / 3.6571 mmHg** (random / chronological),
versus paired LoRA **2.8764 / 3.7124**. It does **not** pass the prespecified
joint upgrade rule. Random selects epoch0 (fixed retrieval/fusion, not a
learned relation gain); a nearest-time training-BP diagnostic is stronger
in that mode. The report explains these limits and includes all requested
Overall/MIMIC/VitalDB MAE, R², ME, STD, threshold, AAMI-style and BHS tables.

The [publication-oriented next plan](docs/PLAN_PERSONAL_MEMORY_PAPER.md)
first separates waveform information from time proximity, then proposes
retrieval-matched training, BP-state retrieval and reliability-aware fusion.
The authorized [v2 implementation plan](docs/PLAN_PERSONAL_MEMORY_V2.md)
specifies four new fits (E1/E2 under both modes), frozen temporal controls,
and fixed E3 reliability evaluation. The learned OOF error gate is deferred.
See [execution status](docs/STATUS.md) for verified submission/completion;
implemented code is not itself proof that a model ran or improved.
The current stronger LoRA remains the main reference.

Both recent screens are complete (2026-09-07):

- [Personal feature mechanisms: all 16 results](docs/RESULTS_PERSONAL_FEATURE_MECHANISMS.md).
- [LoRA + PRS continuation: all 10 results](docs/RESULTS_LORA_PRS_CONTINUATION.md).

The strongest result in these screens is **continued LoRA without PRS**:
Overall mean MAE **2.8801 / 3.7124 mmHg**, random-disjoint / chronological-blocked.
Neither alternative personal modules nor PRS exceed their matched LoRA control.
The tables include independent Overall/MIMIC/VitalDB views, STD and qualified
AAMI/BHS numerical screens; these are development results, not clinical validation.

The original [personal-memory research blueprint](docs/PUBLICATION_RESEARCH_BLUEPRINT_20260907.md)
asks whether lawful personal training measurements complement the strong LoRA
parameters. It is a hypothesis, not a demonstrated upgrade or novelty claim.
The [finite executable experiment](docs/PLAN_PERSONAL_MEMORY_V1.md) is complete;
both real-data GPU checks passed and all10 neural/control fits ran. See the
[execution receipt](docs/PERSONAL_MEMORY_EXECUTION_20260907.md) and
[verified status](docs/STATUS.md) for the dated job snapshot.

The [frozen-model diagnostic](docs/RESULTS_PERSONAL_MECHANISMS.md) is complete.
LoRA depends on both correctly matched PPG windows and persistent personal
state, not just the participant BP mean. Natural and cached predictions were
reproduced exactly. Overall/MIMIC/VitalDB tables are available in the report.

The [feature-mechanism plan](docs/PLAN_PERSONAL_FEATURE_MECHANISMS.md) is retained
with its complete results, including the serial-loader recovery of one failed run.

## Earlier completed experiment — 2026-09-06

The [compact personal-profile result](docs/RESULTS_SAME_SUBJECT_PERSONAL_PROFILES.md)
is complete: eight candidates under each of two same-subject development
splits. Paired LoRA remains best at Overall mean MAE **3.0394** mmHg for
random disjoint windows and **3.7945** mmHg for chronological blocking.
The proposed 32D profile reaches **4.0179 / 4.3140** mmHg and does not pass
the prospective upgrade gate. The report explains the useful dynamic branch,
the unhelpful current reliability gate, and the proposed next mechanism tests.

[All Overall/MIMIC/VitalDB tables](results/same_subject_personal_profiles/)
include MAE, R², ME, STD, percentages within 5/10/15 mmHg, and qualified
AAMI/BHS numerical screens. This experiment uses persistent profiles learned
from 320 labelled windows per seen participant. Held-out targets remain sealed.
See [verified project status](docs/STATUS.md) for the experiment record.

Research code for photoplethysmography (PPG) blood-pressure personalization.
The current track studies registered-user persistent personal state, with
intentional participant overlap and disjoint windows across development roles.
The original unseen-user `K = 1, 2, 3, or 5` reference-event experiments remain
as a separate historical track; they are not the current training protocol.

> This repository contains research code, tests, configurations, and Slurm
> scripts. It does not contain PulseDB waveforms, participant data, trained
> checkpoints, or clinical software. The project is not a validated medical
> device and makes no clinical-use claim.

## Original unseen-user study design — separate historical track

- Dataset role: PulseDB v2 for population/meta-training, meta-validation, and a
  quarantined internal meta-test.
- Calibration unit: one temporally separated, ABP-labelled representative event
  (a PulseDB pseudo-cuff event), never an adjacent waveform window.
- Primary calibration budgets: `K = 1, 2, 3, 5`.
- Frozen event protocol: `event120-v1`, using 120-second temporal blocks.
- Fair K comparison: events 1--5 form the support-candidate pool; every K uses
  the identical query set beginning at event 6.
- Split policy: participants are disjoint across meta-train, meta-validation,
  and locked meta-test.
- Primary reporting unit: participant-macro error. Pooled event-level error is
  secondary.

```text
events:  1  2  3  4  5 | 6  7  8 ...
K=1:     S  U  U  U  U | Q  Q  Q ...
K=2:     S  S  U  U  U | Q  Q  Q ...
K=3:     S  S  S  U  U | Q  Q  Q ...
K=5:     S  S  S  S  S | Q  Q  Q ...
```

`S` is a calibration support event, `U` is unused for that K, and `Q` is a
future query. Query BP is evaluator-only and cannot be used for adaptation,
preprocessing selection, early stopping, or model selection.

## Separate same-subject benchmark

The repository also contains a deliberately separate
[`development-calbased-analogue-v1`](docs/PULSEDB_SAME_SUBJECT_ANALOGUE_PLAN.md)
benchmark. It asks an easier, subject-dependent question: after a participant
contributes hundreds of labelled training windows, how well can PPG-only
models predict different 10-second windows from that same participant?

This track uses only the frozen `meta_train` parent split, compares random
disjoint-window and chronological-blocked assignments, audits exact PPG
content across roles, and keeps its final held-out targets sealed during model
selection. Its results must not be presented as unseen-participant
generalization, K=1/2/3/5 cuff calibration, or an exact official PulseDB
CalBased reproduction.

The completed [same-subject dual-split result](docs/RESULTS_SAME_SUBJECT_DUAL_SPLIT.md)
reports all nine settings under both split modes. The participant-mean plus PPG
residual model is best in both: Overall participant-macro SBP/DBP/mean MAE is
7.6485/4.1693/5.9089 mmHg for random disjoint windows and
8.0361/4.3716/6.2039 mmHg for chronological blocked windows. Complete
Overall/MIMIC/VitalDB tables are under
[results/same_subject_dual_split](results/same_subject_dual_split). The
held-out role remains sealed.

The completed [single-component follow-up](docs/RESULTS_SAME_SUBJECT_SINGLE_COMPONENT.md)
compares the paired residual reference with 18 isolated changes on the same
complete internal-validation cohort. The selected seen-subject rank-4 LoRA
adapter reaches Overall participant-macro SBP/DBP/mean MAE of
3.9663/2.2043/3.0853 mmHg, versus 7.7155/4.2092/5.9624 mmHg for the paired
reference, while also improving both PulseDB source strata. The result is a
same-participant random-window development finding based on 320 labelled
training windows per participant; it is not an unseen-participant K=1/2/3/5
calibration claim. Public aggregate tables are under
[results/same_subject_single_component](results/same_subject_single_component),
and the held-out role remains sealed.

The completed [terminal module-combination result](docs/RESULTS_SAME_SUBJECT_COMBINATIONS.md)
keeps LoRA fixed and tests a bounded 15-candidate hierarchy of the five
next-best single components under both random-disjoint and chronological-
blocked assignments. `lora_film_reliability` is numerically lowest under
random disjoint windows at 3.0106 mmHg Overall participant-macro mean MAE,
while `lora_film_attention` is lowest under chronological blocking at 3.7831
mmHg. Neither reaches the prespecified 0.15-mmHg gain in both modes, so the
cross-split selector retains plain LoRA. Complete path-free aggregate tables
are under [results/same_subject_combinations](results/same_subject_combinations),
and no held-out result is released by this screen. The prospective
[plan](docs/PLAN_SAME_SUBJECT_COMBINATION_SCREEN.md) remains available for
comparison with the completed decision.

## Model and baseline matrix

The development comparison contains:

- population BP mean, last-cuff persistence, and support-BP mean;
- a calibration-free multiscale 1-D ResNet population model;
- population residual-offset correction;
- single-anchor Siamese delta prediction;
- head-only, full-network, and LoRA personal fine-tuning;
- `M0`: unweighted variable-K residual anchoring;
- `M1`: M0 plus support-conditioned FiLM modulation;
- `M2`: M1 plus query-conditioned support reliability weighting.

The main research hypothesis concerns the complete event-level protocol and
the M0--M2 progression, not the novelty of FiLM, LoRA, residual correction, or
set pooling in isolation. Robustness to contact pressure, motion, and device or
acquisition shift is a later phase and begins only after the base personalized
model passes the calibration gate.

See [METHODS.md](docs/METHODS.md) for equations and allowed inputs,
[PROTOCOL.md](docs/PROTOCOL.md) for leakage controls, and
[STATUS.md](docs/STATUS.md) for the current verified development state. The
current Phase-5 development comparison is the [five-seed report](docs/RESULTS_PHASE5_REPEAT5.md),
with a [participant-macro table](results/phase5_repeat5_participant_macro.csv),
an [88-row extended summary](results/phase5_repeat5_extended_summary.csv), and
the [440 seed-specific diagnostic rows](results/phase5_repeat5_extended_metrics_by_seed.csv)
needed to reconstruct the summary. The original [single-seed snapshot](docs/RESULTS_PHASE5.md)
is retained as a historical development record. The complete [data-selection
and training-cohort report](docs/DATA_SELECTION_AND_TRAINING_COHORT.md)
documents why the experiment uses a leakage-safe subset of the original
segment rows.

The completed fixed-first Phase-6 single-seed screen is reported in
[RESULTS_PHASE6_SCREENING.md](docs/RESULTS_PHASE6_SCREENING.md). Its complete
Overall, MIMIC, and VitalDB result tables and exact worst-30% oracle diagnostics
are under [results/phase6_screening](results/phase6_screening). MIMIC and
VitalDB are internal PulseDB source strata, not external validation datasets.

The completed fourth-round tail-aware factorial is reported in
[RESULTS_PHASE6B_FACTORIAL.md](docs/RESULTS_PHASE6B_FACTORIAL.md), with public
machine-readable Overall/MIMIC/VitalDB, participant-macro, bootstrap, and
oracle-diagnostic tables under
[results/phase6b_factorial](results/phase6b_factorial). The private
participant-level tail-membership file is not published.

The completed
[Round-8 calibration-relative screen](docs/RESULTS_ROUND8_CALIBRATION_RELATIVE.md)
compares five calibration-relative/causal candidates and three isolated
collaborator-requested checks on the same development protocol. R8-4 is the
single-seed K=5 screening winner, improving Overall participant-macro mean MAE
from 8.485 to 8.254 mmHg and improving both internal PulseDB source strata.
Machine-readable Overall/MIMIC/VitalDB tables are under
[results/round8](results/round8). The locked meta-test remains untouched, and
independent-seed confirmation has not yet been performed.

The exploratory [Round-9 calibration-refinement result](docs/RESULTS_ROUND9_CALIBRATION_REFINEMENT.md)
compares eight isolated calibration-relative refinements with an
architecture-matched reference at K=5 using participant-disjoint internal
meta-train folds. No candidate passes the prespecified promotion gate. Adaptive
population/personal fusion has the lowest Overall mean MAE but improves only
0.036 mmHg and slightly worsens VitalDB; it is not promoted. Aggregate tables
are under [results/round9](results/round9), and the locked meta-test remains
untouched. The original [Round-9 plan](docs/ROUND9_CALIBRATION_REFINEMENT_PLAN.md)
is retained for prospective context.

The completed [Round-10 partial end-to-end result](docs/RESULTS_ROUND10_PARTIAL_END_TO_END.md)
compares nine K=5 encoder-adaptation candidates under a stricter fold-0--2 fit,
fold-3 early-stop, fold-4 selection boundary. T10-8 improves Overall mean MAE
from 8.5322 to 8.4457 mmHg and improves both internal source strata, but its
0.0865-mmHg gain is below the prespecified 0.15-mmHg promotion threshold. No
candidate is promoted or evaluated on meta-validation; the locked meta-test
remains untouched. Aggregate tables are under [results/round10](results/round10),
and the original [Round-10 plan](docs/ROUND10_PARTIAL_END_TO_END_PLAN.md) is
retained for prospective context.

The completed [Round-11A backbone result](docs/RESULTS_ROUND11_BACKBONE_SCREEN.md)
holds the K=5 Quality Gate + Huber calibration head and all data boundaries
fixed while comparing the current compact ResNet, a higher-capacity ResNet,
InceptionTime, a patch Transformer, and a Conformer. The compact ResNet remains
best at 8.6453 mmHg Overall mean participant-macro MAE. Every alternative
worsens Overall, MIMIC, and VitalDB, so no new backbone is promoted. Aggregate
tables are under [results/round11a](results/round11a). Round 11 may proceed to
the prespecified subtractive component ablation on the retained compact
ResNet; meta-validation and the locked meta-test remain untouched.

The completed [Round-12 literature-derived backbone result](docs/RESULTS_ROUND12_LITERATURE_BACKBONES.md)
tests four additional architecture families under the same strict K=5 boundary:
a causal TCN, the residual-attention encoder from a direct PulseDB five-shot
study, a compact CNN-GRU, and a one-dimensional residual U-Net. None improves
the unchanged compact ResNet in Overall, MIMIC, or VitalDB; the closest
alternative, the residual U-Net, worsens Overall mean participant-macro MAE by
0.1120 mmHg. The supporting [literature audit](docs/LITERATURE_AUDIT_CALIBRATED_PPG_BP_20260822.md)
separates true few-shot calibration from same-participant high-volume window
splits. Aggregate tables are under [results/round12](results/round12).
Meta-validation and the locked meta-test remain untouched.

The completed [Round-13 final architecture/capacity result](docs/RESULTS_ROUND13_FINAL_CAPACITY_SCREEN.md)
repeats the compact ResNet reference and tests controlled ResNet depth/width,
InceptionTime width, Transformer depth/width/tokenization, Conformer scale, and
ConvNeXt-1D while keeping the K=5 QGH calibration model, effective batch,
folds, seed, and query set fixed. The wider InceptionTime encoder improves
Overall participant-macro mean MAE from 8.7089 to 8.5197 mmHg and also improves
both internal PulseDB source strata. Its 0.1891-mmHg Overall gain passes the
prespecified single-seed internal gate, so it advances only to independent-seed
confirmation; it does not yet replace the reference. Aggregate tables are
under [results/round13](results/round13). Meta-validation and the locked
meta-test remain untouched.

The completed [Round-14 paired confirmation and complete-method
result](docs/RESULTS_ROUND14_CONFIRMATION_AND_COMPLETE_METHODS.md) shows that
the wider InceptionTime QGH backbone does not confirm its Round-13
architecture gain: its four-new-seed Overall improvement over compact ResNet
is 0.0559 mmHg, below the frozen 0.15-mmHg gate, despite four of four positive
Overall directions. In the separate single-seed complete-method screen,
`calibration_relative` improves participant-macro mean MAE from 8.5107 to
8.3520 mmHg Overall and improves both internal PulseDB source strata. It
passes the exploratory primary gate and advances only to independent-seed
confirmation; it is not a final model. The standards-oriented variant does
not pass either gate, and no complete setting passes the Overall
retrospective AAMI-style numerical screen. Aggregate tables are under
[results/round14](results/round14), and the prospective
[plan](docs/ROUND14_CONFIRMATION_AND_STANDARDS_PLAN.md) is retained for
comparison. Meta-validation and the locked meta-test remain untouched.

## Repository layout

```text
config/                     immutable public acquisition/configuration metadata
docs/                       study protocol, model definitions, and status
results/                    public rounded development-result tables
scripts/                    data-audit, materialization, Slurm, and training tools
src/pulsedb_fewshot/        Python package
tests/                      synthetic and contract-level regression tests
requirements-train.txt      pinned PyTorch training requirement
pyproject.toml              package and test configuration
```

## Installation and tests

The data-contract and audit modules require Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

GPU training additionally requires the PyTorch build specified in
`requirements-train.txt`. The included Slurm scripts reflect the current lab
cluster and should be reviewed before use on another system.

## Data

PulseDB data are intentionally excluded. Obtain the dataset from the
[official PulseDB repository](https://github.com/pulselabteam/PulseDB) and
follow its source-specific terms. Keep raw data and high-frequency processed
arrays outside Git.

The server-side project convention is:

```text
~/work/ppg_bp/   active code, environment, processed data, logs, and checkpoints
~/nas/ppg_bp/    raw-data master and durable archives
```

Training reads from `~/work`; it does not train directly from NAS.

## Reproducibility boundary

Development hyperparameters and checkpoints are selected only on
meta-validation. Locked meta-test scoring must not be submitted until the final
configuration, comparator, seed policy, and evaluation script are frozen.
Saved predictions, run configuration, source snapshot hash, data-manifest hash,
checkpoint hash, environment, and participant-macro metrics are required for an
accepted experiment.

## Current maturity

The complete PulseDB audit, participant split, `event120-v1` construction,
label-isolated query artifacts, waveform materialization, GPU smoke test, and
five-seed Phase-5 development comparison are complete. M0 has the lowest mean
participant-macro MAE at every K. The repeated runs used no epoch cap and all
stopped through patience-8 early stopping, resolving the original convergence
concern. M0 and M1 remain close, so M0 is the parsimonious provisional finalist
rather than a conclusively superior model. No locked-test, robustness,
external-validation, or final-paper result is reported at this stage.

The fixed-first Phase-6 screen is complete. The PPG-only quality gate improved
the four-K participant-macro mean MAE from 9.137 to 8.962 mmHg in the fixed
seed, but the gain was concentrated in MIMIC; it is therefore a provisional
component rather than a confirmed replacement for M0. Round 4 is a
development-only tail-aware factorial screen of quality gating, Huber loss,
and participant-tail CVaR. It is complete: quality gate plus Huber has the best
full-coverage four-K mean (8.888 mmHg versus 9.137 for M0), whereas CVaR does
not add a full-cohort gain. The improvement is still single-seed and
source-asymmetric, so quality gate plus Huber is a provisional candidate, not
a frozen final model. The full method boundary is documented in
[PHASE6B_TAIL_AWARE_PLAN.md](docs/PHASE6B_TAIL_AWARE_PLAN.md). No locked-test,
robustness, external-validation, or final-paper result is reported at this
stage.

A leakage-safe Phase-6C prototype adds explicit hard-case identification and
specialisation at K=5. Five participant-disjoint meta-train cross-fitting runs
create out-of-fold difficult-participant labels; an input-visible risk MLP
then identifies similar query events, while separately trained tail experts
are evaluated through hard routing and soft fusion with M0. Neither the risk
classifier nor the experts use meta-validation or locked-test error labels for
training. This is an exploratory mixture-of-experts screen, not a replacement
for M0 until its classifier and full-coverage gains pass the development gate.

## License and citation

No open-source license has yet been granted. The public repository is intended
for transparent research review; contact the repository owner before reusing or
redistributing the code. Dataset licenses remain separate and are not changed by
this repository.
