# Calibration PPG BP

Research code for leakage-safe, event-level, few-shot personalization of
photoplethysmography (PPG) blood-pressure models.

The primary experiment asks whether a PPG-aware personalized neural model can
outperform strong, simple calibration controls when each new participant
provides only `K = 1, 2, 3, or 5` independent reference-BP events.

> This repository contains research code, tests, configurations, and Slurm
> scripts. It does not contain PulseDB waveforms, participant data, trained
> checkpoints, or clinical software. The project is not a validated medical
> device and makes no clinical-use claim.

## Study design

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
