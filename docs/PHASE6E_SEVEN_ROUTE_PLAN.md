# Phase-6E seven-route continuous-error screen

## Decision and gate

Phase-6E retains Quality Gate + Huber as the K=5 general reference and screens
seven continuous-error models at seed `20260820`. A candidate advances only if
Overall participant-macro mean MAE improves by at least 0.15 mmHg and both
MIMIC and VitalDB improve. No locked-test data are read.

All residual targets come from five-fold Quality Gate + Huber out-of-fold
predictions on meta-train. Folds 0--3 fit model parameters; fold 4 controls
early stopping or ridge regularization. Meta-validation is evaluated once
after each candidate is frozen.

| Route | Candidate | Single defining change |
|---|---|---|
| R6-1 | Ridge residual | Linear prediction of signed SBP/DBP residual |
| R6-2 | Residual MLP | Nonlinear signed residual prediction |
| R6-3 | Gated residual MLP | Learns both correction and shrinkage confidence |
| R6-4 | Difficult-2x residual MLP | Doubles OOF difficult-participant loss weight |
| R6-5 | Causal GRU residual | Uses only the current and earlier query-feature sequence |
| R6-6 | Supervised MoE | BP-residual loss jointly learns a soft gate and four experts |
| R6-7 | Morphology-cluster MoE | Frozen waveform-embedding clusters gate specialist residual heads |

## Morphology-cluster route

R6-7 operationalizes waveform phenotypes without preassigning a difficult
fraction. A frozen population PPG encoder produces a 256-dimensional embedding
for every meta-train and meta-validation query waveform. PCA is fitted on a
meta-train sample. K-means candidates with 8, 16, and 32 clusters are compared
using two-seed assignment stability and minimum cluster fraction, without BP
targets or meta-validation outcomes. The highest stable, non-degenerate K is
frozen. A new query obtains soft distances to these prototypes and combines
cluster-specific residual experts. Source labels are excluded from clustering
and used only for a post-hoc mixing audit.

This is an unsupervised morphology partition followed by supervised residual
experts. It is distinct from R6-6, whose gate is learned directly from BP
residual loss. Participant IDs, query BP, true error, and future queries are
never inference inputs. The model predicts every query; no group is excluded.

## Submitted execution

The corrected dependency chain was submitted on 2026-08-19. Preparation job
915 feeds routes 916--921 and waveform-embedding job 922; cluster-MoE job 923
depends on 922; unified report job 924 depends on all seven candidates. Job 915
completed successfully before this record was published. The final report will
recompute Overall, MIMIC, and VitalDB metrics from the same frozen query keys.

An initial chain 900--907 was superseded because NAS rejected permission-
preserving copies and the cluster route requested 32 GiB on a node with a
verified 31,000-MB limit. The dependent jobs had not run and were cancelled;
the corrected scripts use ordinary archive copies and a 28-GiB cluster job.
