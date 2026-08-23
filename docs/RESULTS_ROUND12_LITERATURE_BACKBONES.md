# Round-12 literature-derived PPG-backbone screen

## Outcome

Round 12 completed successfully, but none of the four literature-derived
backbones improves the compact ResNet reference. The unchanged
`resnet_small` QGH model remains the numerical winner at an Overall
participant-macro SBP/DBP/mean MAE of **11.0294/6.2592/8.6443 mmHg**.

The closest alternative is the 1D residual U-Net at 8.7563 mmHg, which is
0.1120 mmHg worse Overall. TCN, the five-shot residual-attention encoder, and
the compact CNN-GRU are worse by 0.1778, 0.4141, and 0.4245 mmHg,
respectively. Every candidate is also worse in both internal PulseDB source
strata. Consequently, `winner_backbone=resnet_small` and
`passes_internal_gate=false`; no new backbone is promoted.

This is an architecture-family result, not a reproduction or refutation of
the source papers. In particular, it does not test the masked self-supervised
pretraining used by the direct PulseDB five-shot paper.

## Frozen comparison boundary

- task: PPG-only, calibrated K=5 prediction;
- support: the first five eligible pseudo-cuff/reference-BP events;
- query: the common event-6-onward set;
- fitting: `meta_train` internal folds 0--2;
- early stopping: fold 3, patience eight, no epoch-count cap;
- candidate ranking: fold 4;
- seed: `20260826`;
- calibration model: the same frozen-population Quality Gate + Huber (QGH)
  head for every backbone;
- feature output: 256 dimensions for every encoder;
- meta-validation: not used for training, early stopping, prediction, scoring,
  or candidate ranking;
- locked meta-test: not accessed.

All five settings use the same 628 participants and 96,332 query events.
MIMIC contributes 285 participants/74,373 queries and VitalDB contributes
343/21,959. MIMIC and VitalDB are internal PulseDB source strata, not
independent external validation datasets.

## Primary participant-macro result

Negative change versus reference would be better.

| Scope | Backbone | Participants | Queries | SBP MAE | DBP MAE | Mean MAE | Change vs reference |
|---|---|---:|---:|---:|---:|---:|---:|
| Overall | `resnet_small` | 628 | 96,332 | 11.0294 | 6.2592 | **8.6443** | 0.0000 |
| Overall | `resunet_encoder` | 628 | 96,332 | 11.0817 | 6.4308 | 8.7563 | +0.1120 |
| Overall | `tcn_bp` | 628 | 96,332 | 11.1906 | 6.4537 | 8.8222 | +0.1778 |
| Overall | `fewshot_resnet_attention` | 628 | 96,332 | 11.4456 | 6.6713 | 9.0584 | +0.4141 |
| Overall | `bp_crnn` | 628 | 96,332 | 11.6010 | 6.5367 | 9.0689 | +0.4245 |
| MIMIC | `resnet_small` | 285 | 74,373 | 12.0610 | 6.5695 | **9.3153** | 0.0000 |
| MIMIC | `resunet_encoder` | 285 | 74,373 | 12.0148 | 6.8754 | 9.4451 | +0.1298 |
| MIMIC | `tcn_bp` | 285 | 74,373 | 12.1727 | 6.8026 | 9.4876 | +0.1724 |
| MIMIC | `bp_crnn` | 285 | 74,373 | 12.5170 | 6.7794 | 9.6482 | +0.3329 |
| MIMIC | `fewshot_resnet_attention` | 285 | 74,373 | 12.5198 | 7.0662 | 9.7930 | +0.4778 |
| VitalDB | `resnet_small` | 343 | 21,959 | 10.1723 | 6.0014 | **8.0868** | 0.0000 |
| VitalDB | `resunet_encoder` | 343 | 21,959 | 10.3064 | 6.0614 | 8.1839 | +0.0971 |
| VitalDB | `tcn_bp` | 343 | 21,959 | 10.3745 | 6.1639 | 8.2692 | +0.1824 |
| VitalDB | `fewshot_resnet_attention` | 343 | 21,959 | 10.5530 | 6.3431 | 8.4481 | +0.3612 |
| VitalDB | `bp_crnn` | 343 | 21,959 | 10.8399 | 6.3351 | 8.5875 | +0.5007 |

Participant-macro MAE is the primary result. Overall is recomputed from all
eligible participants/events and is not the average of the two source rows.

## Event-pooled diagnostic screen for QGH models

These values are secondary. `AAMI` applies only a retrospective numerical
mean-error/error-SD screen, and `BHS` is the historical cumulative-error grade;
neither constitutes formal device validation. The asterisk in the archived
CSV marks this qualification.

### Overall

| Setting | BP | MAE | R² | ME | STD | <=5 | <=10 | <=15 | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `resnet_small` | SBP | 12.4470 | 0.4523 | -0.9167 | 16.4915 | 28.07% | 51.43% | 68.66% | Fail | D |
| `resnet_small` | DBP | 6.8158 | 0.4719 | -0.9485 | 9.4792 | 48.54% | 77.88% | 91.02% | Fail | C |
| `resunet_encoder` | SBP | 12.4502 | 0.4554 | -0.9252 | 16.4438 | 27.89% | 51.22% | 68.51% | Fail | D |
| `resunet_encoder` | DBP | 7.0187 | 0.4544 | -0.7022 | 9.6578 | 46.51% | 76.37% | 90.85% | Fail | C |
| `tcn_bp` | SBP | 12.6338 | 0.4446 | -0.5017 | 16.6262 | 27.30% | 50.22% | 67.59% | Fail | D |
| `tcn_bp` | DBP | 6.9646 | 0.4558 | -0.7863 | 9.6391 | 47.44% | 77.20% | 90.65% | Fail | C |
| `fewshot_resnet_attention` | SBP | 12.8244 | 0.4223 | -0.7694 | 16.9456 | 27.33% | 50.29% | 67.22% | Fail | D |
| `fewshot_resnet_attention` | DBP | 7.2508 | 0.4210 | -0.4335 | 9.9661 | 45.36% | 75.16% | 89.80% | Fail | C |
| `bp_crnn` | SBP | 12.7380 | 0.4359 | -1.2803 | 16.7142 | 26.92% | 49.78% | 67.32% | Fail | D |
| `bp_crnn` | DBP | 6.9872 | 0.4546 | -0.5590 | 9.6648 | 46.99% | 76.84% | 90.60% | Fail | C |

### MIMIC

| Setting | BP | MAE | R² | ME | STD | <=5 | <=10 | <=15 | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `resnet_small` | SBP | 13.1034 | 0.4303 | -1.1505 | 17.2200 | 26.66% | 49.07% | 66.07% | Fail | D |
| `resnet_small` | DBP | 7.0457 | 0.4444 | -0.9088 | 9.7825 | 47.35% | 76.61% | 90.14% | Fail | C |
| `resunet_encoder` | SBP | 13.0582 | 0.4383 | -1.2577 | 17.0906 | 26.31% | 48.97% | 66.16% | Fail | D |
| `resunet_encoder` | DBP | 7.2951 | 0.4231 | -0.7196 | 9.9859 | 44.90% | 74.71% | 89.93% | Fail | C |
| `tcn_bp` | SBP | 13.2751 | 0.4253 | -0.5357 | 17.3259 | 25.85% | 47.87% | 65.03% | Fail | D |
| `tcn_bp` | DBP | 7.1866 | 0.4300 | -0.6930 | 9.9271 | 46.27% | 76.04% | 89.90% | Fail | C |
| `fewshot_resnet_attention` | SBP | 13.4692 | 0.4031 | -0.8603 | 17.6453 | 26.01% | 47.97% | 64.70% | Fail | D |
| `fewshot_resnet_attention` | DBP | 7.5046 | 0.3917 | -0.3315 | 10.2750 | 43.95% | 73.84% | 88.92% | Fail | C |
| `bp_crnn` | SBP | 13.2842 | 0.4239 | -1.5858 | 17.2823 | 25.86% | 47.82% | 65.02% | Fail | D |
| `bp_crnn` | DBP | 7.1743 | 0.4327 | -0.5694 | 9.9112 | 46.11% | 75.88% | 89.80% | Fail | C |

### VitalDB

| Setting | BP | MAE | R² | ME | STD | <=5 | <=10 | <=15 | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `resnet_small` | SBP | 10.2240 | 0.4870 | -0.1249 | 13.7113 | 32.85% | 59.44% | 77.44% | Fail | D |
| `resnet_small` | DBP | 6.0370 | 0.5528 | -1.0830 | 8.3694 | 52.57% | 82.16% | 94.00% | Fail | B |
| `resunet_encoder` | SBP | 10.3911 | 0.4670 | 0.2008 | 13.9752 | 33.21% | 58.84% | 76.47% | Fail | D |
| `resunet_encoder` | DBP | 6.0824 | 0.5488 | -0.6434 | 8.4525 | 52.00% | 82.00% | 93.98% | Fail | B |
| `tcn_bp` | SBP | 10.4617 | 0.4649 | -0.3865 | 13.9988 | 32.19% | 58.19% | 76.26% | Fail | D |
| `tcn_bp` | DBP | 6.2125 | 0.5296 | -1.1020 | 8.5848 | 51.39% | 81.15% | 93.17% | Fail | B |
| `fewshot_resnet_attention` | SBP | 10.6405 | 0.4399 | -0.4614 | 14.3202 | 31.79% | 58.16% | 75.76% | Fail | D |
| `fewshot_resnet_attention` | DBP | 6.3914 | 0.5065 | -0.7787 | 8.8313 | 50.11% | 79.61% | 92.78% | Fail | B |
| `bp_crnn` | SBP | 10.8882 | 0.4198 | -0.2457 | 14.5800 | 30.48% | 56.42% | 75.10% | Fail | D |
| `bp_crnn` | DBP | 6.3534 | 0.5143 | -0.5238 | 8.7791 | 49.96% | 80.08% | 93.31% | Fail | C |

No Overall or source-stratified QGH setting passes the AAMI-style numerical
screen for both SBP and DBP. VitalDB DBP reaches historical BHS Grade B for
four settings, but the paired SBP remains Grade D; this is not a complete
model-level standards pass.

## Architecture interpretation

- **Residual U-Net is the closest alternative**, but its 1.16-million-parameter
  encoder remains worse than the 665-thousand-parameter reference in Overall,
  MIMIC, and VitalDB.
- **TCN is parameter-efficient** (20,482 population-model parameters including
  its BP head) but loses 0.178 mmHg Overall. This agrees with the audited 2026
  subject-disjoint PulseDB paper's conclusion that ResNet-versus-TCN effects
  are small relative to calibration and data quality.
- **The 16.13-million-parameter residual-attention model is the largest and
  clearly worse.** Its source paper's reported benefit cannot be attributed to
  this architecture alone under the present protocol; self-supervised
  pretraining remains an untested factor.
- **The compact CNN-GRU is also worse.** Its low parameter count does not
  compensate for the information loss associated with adapting its original
  short-input temporal contract to the 10-second PulseDB event waveform.

The result supports retaining the compact ResNet and rejects an architecture
replacement based only on published headline scores. It also supports a
subtractive or training-method-focused next step rather than another broad
backbone sweep.

## Execution and integrity

- jobs 1050--1065 all completed with exit code `0:0`;
- all 16 stderr files are zero bytes;
- every population, QGH, evaluation, and report directory is byte-identical
  between work and NAS;
- all five settings have the same fold-4 query keys and targets, with shared
  SHA-256
  `0f10de3ef63f017848717bccdeb5d80c5973afab5595bb7665c7c2cc91b42036`;
- all five evaluation runs record `meta_validation_accessed=false`,
  `meta_validation_used_for_training=false`,
  `meta_validation_used_for_early_stopping=false`,
  `meta_validation_used_for_candidate_ranking=false`,
  `meta_validation_predictions_generated=false`, `locked_test_accessed=false`,
  `query_bp_model_input=false`, `future_query_model_input=false`, and
  `source_model_input=false`;
- the final selection file records `winner_backbone=resnet_small` and
  `passes_internal_gate=false`.

Accepted aggregate SHA-256 values:

- participant-macro table:
  `465cc4f70978709741338692561213501ee40a55c136c67d671512a4061dc87d`;
- pooled diagnostic table:
  `11224a25bec9f5b3439e0d2dfde602331b8b4ea4a8428111a0027e05b56189a2`;
- comparison table:
  `b87e87d70a18ebb55214819ccf9683fe662289e2f6771c3524c4b466f26217b0`;
- complexity table:
  `c110f2d397bd477994a7053a7c59b49fdc95fa5e1d7ac193ffcbc1e6312b091e`;
- selection JSON:
  `b70805bb9620ba0a0dcc6f7fea3df183b69105bfb330d20438159052c592abf4`.

## Public aggregate artifacts

- [participant-macro table](../results/round12/participant_macro_internal.csv)
- [complete pooled diagnostics](../results/round12/pooled_diagnostics_internal.csv)
- [comparison versus reference](../results/round12/comparison_vs_reference_internal.csv)
- [model complexity](../results/round12/model_complexity.csv)
- [selection record](../results/round12/selection.json)
- [prospective plan](ROUND12_LITERATURE_BACKBONE_PLAN.md)
- [supporting literature audit](LITERATURE_AUDIT_CALIBRATED_PPG_BP_20260822.md)

Participant identifiers, event-level predictions, checkpoints, logs, raw
waveforms, personal paths, and private infrastructure details are not
published.

## Decision and claim limit

No Round-12 candidate advances to meta-validation, multi-seed confirmation, or
the locked meta-test. The compact ResNet remains the architecture reference.
The next method decision should distinguish architecture from training method:
either run the already prespecified subtractive QGH ablation, or separately
test masked self-supervised PPG pretraining on the retained architecture and a
single literature-derived encoder. Any such experiment remains a new
single-seed internal screen and must not reuse Round-12 fold-4 outcomes as
training labels.
