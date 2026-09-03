# Few-shot support-conditioned adapter-bank screen

## Objective

This is a prospective, development-only screen for unseen-participant PPG blood-pressure calibration. It tests whether a new participant can use K=1/2/3/5 fixed-first cuff-reference events to mix a small bank of shared low-rank feature adapters. The bank is learned only from meta-training participants; it does not create or retrieve a participant-ID-specific adapter.

The scientific factor under study is deliberately narrow:

1. number of shared adapter bases M in {5, 10, 15, 20, 25, 30};
2. routing over the five highest-scoring bases (`top5`) or over all bases (`dense`).

All other model, data, optimization, support, query, fold, seed, and reporting choices are held fixed.

## Experimental matrix

| Basis count M | Top-5 routing | Dense routing |
|---:|:---:|:---:|
| 5 | yes | yes |
| 10 | yes | yes |
| 15 | yes | yes |
| 20 | yes | yes |
| 25 | yes | yes |
| 30 | yes | yes |

This produces 12 scheduled candidate jobs but only 11 functionally unique configurations. When M=5, Top-5 and dense routing both use all five bases. Those two jobs are retained as a numerical and hardware consistency control and must not be counted as two independent scientific candidates.

## Model held constant around the tested factor

- Frozen population encoder/regressor: the completed `resnet_small`, 256-dimensional population checkpoint trained on internal folds 0-2 and stopped on fold 3 without fold-4 access.
- Baseline (`m0_reference`): support residual mean anchor plus the standard support-set correction head.
- Candidate: the same M0 model plus a support-conditioned rank-4 low-rank adapter bank applied to the query PPG feature.
- Router input: only the aggregated support PPG/BP/residual context and normalized support count K/5.
- Excluded inputs: participant identity, source label, future query events, query BP, query error, and reference ABP.
- Initialization: adapter output factors and the M0 correction output are zero initialized, so each candidate starts from the same residual-anchor prediction as M0.

## Frozen development protocol

- Event store: `event120-v1` development store.
- Participant roles: meta-train participants only, split by the existing participant-disjoint five-fold table.
- Fit: folds 0, 1, and 2.
- Early stopping: fold 3, patience 8, no epoch-count cap.
- Internal candidate ranking: fold 4, opened only after each checkpoint is frozen.
- Support: fixed chronological events 1..K for K in {1,2,3,5}.
- Query: the identical event-6-and-later common query set for every K and every setting.
- Sampling: participant-balanced; 99,968 sampled episodes per epoch.
- Optimization: standardized-coordinate MSE, AdamW, learning rate 3e-4, weight decay 1e-4.
- Seed: 20260904 for this single-seed screen.
- Meta-validation and locked meta-test: not accessed.

The feature-cache artifact stores query PPG embeddings without query BP columns. Fit, early-stopping, and fold-4 target tables are physically separate. The fold-4 prediction table is generated before its target table is loaded and joined for scoring.

## Primary report

For M0 and all 12 scheduled candidates, report participant-macro SBP MAE, DBP MAE, and their mean:

- for K=1,2,3,5 separately;
- averaged equally over the four K budgets;
- for Overall, MIMIC, and VitalDB strata.

Event-pooled MAE, R-squared, ME, STD, within-5/10/15 mmHg percentages, and retrospective AAMI/BHS numerical screens are secondary diagnostics. They are not compliance claims.

The same query identities and targets must be used for every setting. Also report routing entropy, active-basis count, mean basis weight, and top-1 basis usage to diagnose bank collapse.

## Frozen development promotion gate

A candidate may advance to a later confirmation experiment only if all conditions hold against the paired M0 reference:

1. Overall four-K average mean participant-macro MAE improves by at least 0.15 mmHg;
2. both MIMIC and VitalDB improve;
3. at least three of four K budgets improve;
4. the worst K-specific change is no worse than -0.10 mmHg;
5. neither Overall SBP nor Overall DBP four-K average MAE worsens by more than 0.05 mmHg.

The screen selects a configuration; it does not establish generalization. Any promoted candidate still requires repeated-seed confirmation and a predeclared meta-validation comparison before any locked-test use.

## Interpretation boundary

This experiment is a comparison of a shared adapter dictionary for unseen-user K-shot calibration. It is not the same as persistent same-subject LoRA, because a new participant receives no stored participant-specific parameters. Their support events only determine mixture weights over bases learned from other participants.
