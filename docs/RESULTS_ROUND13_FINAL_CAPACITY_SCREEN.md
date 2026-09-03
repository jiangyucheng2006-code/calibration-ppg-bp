# Round-13 final architecture and capacity screen

## Outcome

Round 13 completed its prespecified single-seed internal screen. The wider
InceptionTime encoder (`inception_time_wide`) is the first architecture
candidate in this series to pass the internal promotion gate. With the same
K=5 Quality Gate + Huber (QGH) calibration model, its Overall
participant-macro SBP/DBP/mean MAE is **10.8759/6.1636/8.5197 mmHg**, compared
with 11.0255/6.3922/8.7089 mmHg for the same-round compact ResNet reference.

The Overall mean-MAE gain is **0.1891 mmHg**. Mean MAE also improves in both
internal PulseDB source strata: by **0.3497 mmHg in MIMIC** and **0.0557 mmHg
in VitalDB**. This satisfies the prespecified requirement of at least 0.15
mmHg Overall improvement plus improvement in both source strata, so
`passes_internal_gate=true`.

This is not yet a final model result. It is one development seed selected on
the repeatedly used internal fold 4. The architecture advances only to
independent-seed confirmation; it has not been evaluated on meta-validation or
the locked meta-test. It also does not pass the retrospective AAMI-style or
historical BHS screen for both SBP and DBP.

## Frozen comparison boundary

- task: PPG-only calibrated K=5 prediction;
- support: the first five eligible pseudo-cuff/reference-BP events;
- query: the common event-6-onward set;
- fitting: `meta_train` internal folds 0--2;
- early stopping: fold 3, patience eight, no epoch-count cap;
- candidate ranking: fold 4;
- seed: `20260827`;
- calibration model: the same frozen-population QGH head for every backbone;
- feature output: 256 dimensions for every encoder;
- population training: physical microbatch 32 with four-step accumulation;
- QGH training: physical microbatch 16 with four-step accumulation;
- sampled examples per epoch: 99,968 for every candidate;
- meta-validation: not used for training, early stopping, prediction, scoring,
  or candidate ranking;
- locked meta-test: not accessed.

All 13 settings use the same 628 participants and 96,332 query events. MIMIC
contributes 285 participants/74,373 queries and VitalDB contributes
343/21,959. MIMIC and VitalDB are internal PulseDB source strata, not
independent external validation datasets.

## Primary participant-macro result

Negative change versus the same-round compact ResNet is better. Overall is
recomputed from all eligible participants/events and is not the average of the
two source rows.

### Overall

| Backbone | Participants | Queries | SBP MAE | DBP MAE | Mean MAE | Change vs reference |
|---|---:|---:|---:|---:|---:|---:|
| `inception_time_wide` | 628 | 96,332 | 10.8759 | 6.1636 | **8.5197** | **-0.1891** |
| `inception_time` | 628 | 96,332 | 11.0233 | 6.2589 | 8.6411 | -0.0678 |
| `resnet_wide1p5` | 628 | 96,332 | 11.1287 | 6.2845 | 8.7066 | -0.0023 |
| `resnet_small` | 628 | 96,332 | 11.0255 | 6.3922 | 8.7089 | 0.0000 |
| `resnet_depth2` | 628 | 96,332 | 11.1990 | 6.3715 | 8.7853 | +0.0764 |
| `patch_transformer_highres` | 628 | 96,332 | 11.3215 | 6.4296 | 8.8756 | +0.1667 |
| `patch_transformer` | 628 | 96,332 | 11.4087 | 6.5237 | 8.9662 | +0.2573 |
| `convnext_1d` | 628 | 96,332 | 11.4755 | 6.5565 | 9.0160 | +0.3071 |
| `patch_transformer_deep` | 628 | 96,332 | 11.5232 | 6.5181 | 9.0206 | +0.3117 |
| `patch_transformer_wide` | 628 | 96,332 | 11.5382 | 6.5594 | 9.0488 | +0.3399 |
| `conformer` | 628 | 96,332 | 11.6181 | 6.6067 | 9.1124 | +0.4035 |
| `conformer_large` | 628 | 96,332 | 11.8264 | 6.7337 | 9.2801 | +0.5712 |
| `patch_transformer_longpatch` | 628 | 96,332 | 11.8955 | 6.6762 | 9.2858 | +0.5770 |

### MIMIC

| Backbone | Participants | Queries | SBP MAE | DBP MAE | Mean MAE | Change vs reference |
|---|---:|---:|---:|---:|---:|---:|
| `inception_time_wide` | 285 | 74,373 | 11.7310 | 6.4670 | **9.0990** | **-0.3497** |
| `inception_time` | 285 | 74,373 | 11.9975 | 6.5956 | 9.2965 | -0.1522 |
| `resnet_wide1p5` | 285 | 74,373 | 12.0862 | 6.5840 | 9.3351 | -0.1136 |
| `resnet_depth2` | 285 | 74,373 | 12.1747 | 6.7032 | 9.4390 | -0.0097 |
| `resnet_small` | 285 | 74,373 | 12.1199 | 6.7775 | 9.4487 | 0.0000 |
| `patch_transformer_highres` | 285 | 74,373 | 12.2828 | 6.7155 | 9.4992 | +0.0505 |
| `patch_transformer` | 285 | 74,373 | 12.3941 | 6.8654 | 9.6297 | +0.1810 |
| `patch_transformer_wide` | 285 | 74,373 | 12.4470 | 6.8327 | 9.6399 | +0.1912 |
| `conformer` | 285 | 74,373 | 12.5152 | 6.7967 | 9.6560 | +0.2073 |
| `convnext_1d` | 285 | 74,373 | 12.4744 | 6.9596 | 9.7170 | +0.2683 |
| `patch_transformer_deep` | 285 | 74,373 | 12.6009 | 6.8796 | 9.7402 | +0.2915 |
| `conformer_large` | 285 | 74,373 | 12.8530 | 7.1260 | 9.9895 | +0.5408 |
| `patch_transformer_longpatch` | 285 | 74,373 | 12.9842 | 6.9991 | 9.9917 | +0.5430 |

### VitalDB

| Backbone | Participants | Queries | SBP MAE | DBP MAE | Mean MAE | Change vs reference |
|---|---:|---:|---:|---:|---:|---:|
| `inception_time_wide` | 343 | 21,959 | 10.1653 | 5.9115 | **8.0384** | **-0.0557** |
| `resnet_small` | 343 | 21,959 | 10.1161 | 6.0721 | 8.0941 | 0.0000 |
| `inception_time` | 343 | 21,959 | 10.2138 | 5.9792 | 8.0965 | +0.0024 |
| `resnet_wide1p5` | 343 | 21,959 | 10.3332 | 6.0356 | 8.1844 | +0.0903 |
| `resnet_depth2` | 343 | 21,959 | 10.3883 | 6.0959 | 8.2421 | +0.1480 |
| `patch_transformer_highres` | 343 | 21,959 | 10.5228 | 6.1921 | 8.3574 | +0.2633 |
| `patch_transformer` | 343 | 21,959 | 10.5900 | 6.2397 | 8.4148 | +0.3207 |
| `patch_transformer_deep` | 343 | 21,959 | 10.6277 | 6.2177 | 8.4227 | +0.3285 |
| `convnext_1d` | 343 | 21,959 | 10.6455 | 6.2215 | 8.4335 | +0.3394 |
| `patch_transformer_wide` | 343 | 21,959 | 10.7831 | 6.3323 | 8.5577 | +0.4635 |
| `conformer` | 343 | 21,959 | 10.8726 | 6.4487 | 8.6607 | +0.5666 |
| `conformer_large` | 343 | 21,959 | 10.9735 | 6.4077 | 8.6906 | +0.5965 |
| `patch_transformer_longpatch` | 343 | 21,959 | 10.9908 | 6.4079 | 8.6994 | +0.6052 |

Participant-macro MAE is the primary result. The winning mean improvement in
VitalDB is driven by DBP: VitalDB participant-macro DBP improves by 0.1607
mmHg, while SBP is 0.0492 mmHg worse. The prespecified gate applies to the
SBP/DBP mean in each source stratum, so the candidate still passes, but this
endpoint-level asymmetry must be checked in the confirmation stage.

## Event-pooled diagnostic comparison

These values are secondary. `AAMI` below applies only a retrospective
mean-error/error-SD numerical screen, and `BHS` is the historical cumulative
error grade; neither is a formal device-validation claim. The complete table
for all population and QGH settings is available in the public aggregate CSV.

### Overall

| Setting | BP | MAE | R² | ME | STD | <=5 | <=10 | <=15 | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `resnet_small` | SBP | 12.4966 | 0.4398 | -1.4621 | 16.6406 | 28.16% | 51.61% | 68.93% | Fail | D |
| `resnet_small` | DBP | 7.0191 | 0.4430 | -0.9887 | 9.7333 | 47.53% | 76.51% | 90.45% | Fail | C |
| `inception_time_wide` | SBP | 12.3256 | 0.4663 | -0.9841 | 16.2748 | 28.25% | 51.68% | 68.86% | Fail | D |
| `inception_time_wide` | DBP | 6.8079 | 0.4737 | -0.6545 | 9.4878 | 48.60% | 78.01% | 91.15% | Fail | C |

### MIMIC

| Setting | BP | MAE | R² | ME | STD | <=5 | <=10 | <=15 | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `resnet_small` | SBP | 13.1960 | 0.4145 | -1.7626 | 17.4079 | 26.52% | 49.09% | 66.31% | Fail | D |
| `resnet_small` | DBP | 7.2701 | 0.4122 | -0.9054 | 10.0649 | 46.38% | 75.03% | 89.48% | Fail | C |
| `inception_time_wide` | SBP | 12.9454 | 0.4469 | -1.3847 | 16.9497 | 26.97% | 49.48% | 66.34% | Fail | D |
| `inception_time_wide` | DBP | 7.0590 | 0.4438 | -0.7309 | 9.8031 | 47.33% | 76.63% | 90.18% | Fail | C |

### VitalDB

| Setting | BP | MAE | R² | ME | STD | <=5 | <=10 | <=15 | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `resnet_small` | SBP | 10.1281 | 0.4890 | -0.4443 | 13.6780 | 33.71% | 60.14% | 77.77% | Fail | D |
| `resnet_small` | DBP | 6.1691 | 0.5353 | -1.2705 | 8.5088 | 51.42% | 81.52% | 93.72% | Fail | B |
| `inception_time_wide` | SBP | 10.2266 | 0.4906 | 0.3725 | 13.6584 | 32.59% | 59.13% | 77.38% | Fail | D |
| `inception_time_wide` | DBP | 5.9575 | 0.5637 | -0.3958 | 8.3269 | 52.91% | 82.68% | 94.40% | Fail | B |

No Overall or source-stratified QGH setting passes the AAMI-style numerical
screen for both SBP and DBP. VitalDB DBP reaches historical BHS Grade B for
the reference and winner, but the paired SBP remains Grade D; therefore there
is no complete model-level standards pass.

## Architecture and capacity interpretation

- **InceptionTime width is the only controlled scale change that passes.**
  Increasing the InceptionTime population encoder from 512,162 to 1,123,954
  parameters improves all three participant-macro mean-MAE views.
- **The base InceptionTime signal is directionally consistent but too small.**
  It improves Overall by 0.0678 mmHg and MIMIC by 0.1522 mmHg, while VitalDB is
  effectively unchanged and slightly worse by 0.0024 mmHg.
- **More parameters are not generally better.** ResNet depth is worse,
  ResNet width is nearly neutral Overall and worse in VitalDB, and the larger
  Transformer, Conformer, and ConvNeXt-1D variants are all worse than the
  compact ResNet.
- **Transformer scaling does not resolve the present limitation.** Increasing
  Transformer depth or width and changing token resolution or patch length
  all worsen the primary metric under the fixed calibration protocol.

The result supports a narrow confirmation of `inception_time_wide`, not a
general move to larger networks and not another unconstrained architecture
sweep.

## Execution and integrity

- all 173 regression tests passed before submission;
- both planned GPU smoke tests reproduced the registered parameter counts and
  completed forward, backward, and optimizer steps at the formal microbatches;
- all 13 population, QGH, and fold-4 evaluation chains completed successfully;
- a post-run SHA-256 inventory verified that all 41 work/archive output pairs
  have identical relative file sets, sizes, and contents;
- one final housekeeping step returned a nonzero status because the archive
  target does not support preserving POSIX permission metadata; the scientific
  outputs had already completed and the subsequent content audit found no
  mismatch;
- all 13 settings have the same fold-4 query keys and targets;
- the selection record states
  `meta_validation_used_for_candidate_ranking=false` and
  `locked_test_accessed=false`;
- the final selection is `winner_backbone=inception_time_wide` and
  `passes_internal_gate=true`.

Accepted public aggregate SHA-256 values:

- participant-macro table:
  `fd4378fe1bbc8f0769bcc03c229183ed67dbf84141a2bc62ae7a1207fbbc2394`;
- pooled diagnostic table:
  `6180af7a9908731c07de6882b57ab30e9f77a0a0c57070929c1f802be9265621`;
- comparison table:
  `0b0a43cd31087593d0a49ca529ae3ae4fc62cb612e56f2d40fac687d5a34d1a8`;
- complexity table:
  `3ccacff810882e86a3ea1472859ca96a97d7874fc3e0cecfc745269f336b1671`;
- selection JSON:
  `33de0f484883eb2bfbb18e1643ca46af6ace6eec96ed5ccc2f1d709fe1f859a0`.

## Public aggregate artifacts

- [participant-macro table](../results/round13/participant_macro_internal.csv)
- [complete pooled diagnostics](../results/round13/pooled_diagnostics_internal.csv)
- [comparison versus reference](../results/round13/comparison_vs_reference_internal.csv)
- [model complexity](../results/round13/model_complexity.csv)
- [selection record](../results/round13/selection.json)
- [prospective plan](ROUND13_FINAL_CAPACITY_SCREEN_PLAN.md)

Participant identifiers, event-level predictions, checkpoints, logs, raw
waveforms, personal paths, and private infrastructure details are not
published.

## Decision and claim limit

`inception_time_wide` advances to a prespecified independent-seed confirmation
against the same compact ResNet reference. It does not yet replace the
reference model, and no meta-validation or locked-test access is justified
until the confirmation reproduces the gain. The confirmation should retain
the same query set and report Overall, MIMIC, and VitalDB from the same saved
predictions, with participant-macro MAE primary. The small VitalDB gain and its
SBP/DBP asymmetry should be treated as explicit confirmation criteria.
