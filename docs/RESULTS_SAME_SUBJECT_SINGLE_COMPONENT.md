# Same-subject single-component screen: result

## Conclusion

All 19 prespecified candidates completed successfully. The selected candidate
is `residual_subject_lora_rank4`, a compact-ResNet residual predictor with a
rank-4 adapter indexed by the already-seen participant.

Its internal-validation participant-macro SBP/DBP/mean MAE is
**3.9663/2.2043/3.0853 mmHg** Overall. The corresponding source-stratified
results are **4.3375/2.3832/3.3604 mmHg** in PulseDB MIMIC and
**3.6055/2.0304/2.8180 mmHg** in PulseDB VitalDB.

Against the paired `residual_reference`, the winner reduces participant-macro
mean MAE by 2.8771 mmHg Overall (48.25%), 2.6213 mmHg in MIMIC (43.82%), and
3.1256 mmHg in VitalDB (52.59%). It therefore passes the prospective discovery
gate of at least 0.15-mmHg Overall improvement plus improvement in both source
strata. `residual_film` is the second-ranked setting at 4.7803 mmHg Overall,
still 1.6950 mmHg worse than the subject-specific LoRA model.

This is a **development-only, seen-participant, random-disjoint-window result**.
It uses 320 labelled training windows per participant and 40 different
internal-validation windows from each of the same participants. It is not an
unseen-participant result, not a K=1/2/3/5 independent-event calibration
result, not an exact reproduction of the official PulseDB CalBased benchmark,
and not an external validation result. The held-out role was not accessed.

## Frozen comparison

- Protocol: `development-calbased-analogue-v1`.
- Screen: `same-subject-single-component-v1`.
- Split mode: `random_disjoint`.
- Seed: `20260831`.
- Cohort: 2,051 participants: 1,011 MIMIC and 1,040 VitalDB.
- Per participant: 320 labelled train-role windows, 40 internal-validation
  windows, and 40 sealed held-out windows.
- Selection set: 82,040 internal-validation windows: 40,440 MIMIC and 41,600
  VitalDB.
- Selection metric: Overall participant-macro mean of SBP and DBP MAE.
- Optimization: Huber loss, no epoch cap, patience of eight non-improving
  epochs.
- Winner convergence: best epoch 65; training stopped after epoch 73 by early
  stopping.
- Held-out access during this screen: **false**.

Every candidate retained the same complete internal-validation cohort. The
two filtering candidates changed training rows only and did not discard hard
validation queries. All candidate run metadata, saved internal-validation
predictions, and pooled diagnostic files were checked against their NAS copies;
all 57 checked files were byte-identical.

## Participant-macro ranking

| Rank | Setting | SBP MAE | DBP MAE | Mean MAE |
|---:|---|---:|---:|---:|
| 1 | `residual_subject_lora_rank4` | 3.9663 | 2.2043 | **3.0853** |
| 2 | `residual_film` | 6.1537 | 3.4069 | 4.7803 |
| 3 | `residual_support_attention` | 6.7819 | 3.7060 | 5.2440 |
| 4 | `residual_multi_event_weighting` | 6.8461 | 3.7381 | 5.2921 |
| 5 | `residual_support_reliability` | 6.8814 | 3.7818 | 5.3316 |
| 6 | `residual_calibration_relative` | 6.8910 | 3.7849 | 5.3379 |
| 7 | `residual_inception_time_wide` | 7.0676 | 3.8611 | 5.4643 |
| 8 | `residual_demographics_direct` | 7.3475 | 4.0240 | 5.6858 |
| 9 | `residual_conformer` | 7.4051 | 4.0385 | 5.7218 |
| 10 | `residual_quality_gate` | 7.6183 | 4.1423 | 5.8803 |
| 11 | `residual_quality_weighted_loss` | 7.7016 | 4.2006 | 5.9511 |
| 12 | `residual_reference` | 7.7155 | 4.2092 | 5.9624 |
| 13 | `residual_soft_moe` | 7.7384 | 4.2067 | 5.9725 |
| 14 | `residual_ppg_quality_filter` | 7.7436 | 4.2174 | 5.9805 |
| 15 | `residual_prototype_moe` | 7.7972 | 4.2554 | 6.0263 |
| 16 | `residual_cnn_bilstm` | 8.0539 | 4.3974 | 6.2256 |
| 17 | `residual_patch_transformer` | 8.1691 | 4.4386 | 6.3038 |
| 18 | `residual_beat_similarity_filter` | 8.2067 | 4.4630 | 6.3348 |
| 19 | `residual_cnn_gru` | 8.7722 | 4.7324 | 6.7523 |

## Winner by PulseDB source

MIMIC and VitalDB are internal PulseDB source strata, not independent external
datasets. Overall was recomputed from all eligible participants and was not
formed by averaging the two source results.

| View | Participants | Events | SBP MAE | DBP MAE | Mean MAE | Oracle worst-30% mean MAE | Remaining-70% mean MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 2,051 | 82,040 | 3.9663 | 2.2043 | **3.0853** | 4.5308 | 2.4649 |
| MIMIC | 1,011 | 40,440 | 4.3375 | 2.3832 | **3.3604** | 5.0083 | 2.6518 |
| VitalDB | 1,040 | 41,600 | 3.6055 | 2.0304 | **2.8180** | 3.9608 | 2.3282 |

The worst-30% and remaining-70% values are retrospective oracle diagnostics
defined from true internal-validation errors. They are not a deployable
screening rule and were not used to select or exclude validation participants.

## Requested event-pooled diagnostic table

Participant-macro MAE above is primary. The following R-squared, signed error,
error standard deviation, threshold percentages, and standards-style fields
are secondary event-pooled diagnostics reconstructed from the same saved
predictions.

| Scope | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Overall | SBP | 3.9663 | 0.9237 | -0.1235 | 5.8465 | 73.47% | 92.73% | 97.42% | PASS* | PASS (Grade A)* |
| Overall | DBP | 2.2043 | 0.9203 | -0.1939 | 3.6169 | 91.17% | 98.20% | 99.37% | PASS* | PASS (Grade A)* |
| MIMIC | SBP | 4.3375 | 0.9201 | -0.1119 | 6.4322 | 70.44% | 90.89% | 96.50% | PASS* | PASS (Grade A)* |
| MIMIC | DBP | 2.3832 | 0.9040 | -0.2927 | 4.1101 | 89.75% | 97.48% | 98.97% | PASS* | PASS (Grade A)* |
| VitalDB | SBP | 3.6055 | 0.9242 | -0.1347 | 5.2145 | 76.41% | 94.53% | 98.31% | PASS* | PASS (Grade A)* |
| VitalDB | DBP | 2.0304 | 0.9376 | -0.0978 | 3.0593 | 92.54% | 98.91% | 99.76% | PASS* | PASS (Grade A)* |

The starred AAMI and BHS entries are retrospective numerical screens only.
They do not establish formal device, clinical, AAMI/ISO/IEEE, or BHS
compliance because this public-dataset experiment was not designed as a formal
device-validation study.

## What the winning method does

The common model first encodes the current 10-second PPG into a 256-dimensional
feature vector. Each already-seen participant has a rank-4 pair of low-rank
matrices that slightly transforms this feature vector before the shared
residual head predicts SBP and DBP change. The final output is:

```text
participant train-role BP mean
+ shared PPG residual
+ participant-indexed low-rank feature adjustment
```

At rank 4, each participant owns 2,048 adapter parameters
(`2 × 256 × 4`), approximately 8 KiB in FP32. Across 2,051 participants,
4,200,448 of the model's 4,898,578 parameters are participant-indexed. This
candidate used `support_count = 0`: it did not derive a new adapter from five
support events at inference. Instead, its adapter table and shared network were
jointly learned from the labelled train-role windows of the same registered
participants.

## Why the result is much better

The result supports the following interpretation, but the causal contribution
of each part still requires ablation:

1. The participant train-role mean removes much of the stable between-person
   BP offset.
2. The shared compact ResNet learns PPG-related deviations around that anchor.
3. The participant-indexed rank-4 adapter can store a different feature
   correction for every already-seen person instead of forcing one mapping to
   fit all 2,051 participants.
4. Random-disjoint validation asks the model to interpolate to different
   windows from known participants; this is substantially easier than adapting
   to a new participant from one to five independent calibration events.

The large gain is therefore scientifically interesting evidence that stable
participant-specific mappings matter, but it cannot yet be attributed solely
to a sample-efficient LoRA mechanism. A subject-mean-only control, a shared
adapter, shuffled participant indices, rank/parameter controls, and a
chronological-blocked confirmation are needed to separate useful personal
mapping from identity memorization and random-window interpolation.

## Decision and next gate

`residual_subject_lora_rank4` passes the prospective random-disjoint discovery
gate. The next justified experiment is a controlled chronological-blocked
confirmation of this single candidate and its paired reference, followed by
the minimum mechanism ablations above. No held-out test should be released and
no module combination should be selected from the present internal-validation
result.

## Public result files

- [`participant_macro_summary.csv`](../results/same_subject_single_component/participant_macro_summary.csv):
  all 19 candidates under Overall, MIMIC, and VitalDB participant-macro views.
- [`event_pooled_diagnostics.csv`](../results/same_subject_single_component/event_pooled_diagnostics.csv):
  requested diagnostic columns for every candidate and scope.
- [`selection.json`](../results/same_subject_single_component/selection.json):
  frozen selection role, winner, score, and held-out-access declaration.
- [`manifest.json`](../results/same_subject_single_component/manifest.json):
  execution, integrity, source-hash, and claim-boundary record.
