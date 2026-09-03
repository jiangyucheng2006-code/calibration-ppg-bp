# Same-subject dual-split development result

## Conclusion

Under both same-subject development splits, `subject_mean_residual_ppg` is the
best of the nine screened settings. It combines each participant's mean BP in
the 320 labelled training windows with a compact-ResNet PPG residual model.

- `random_disjoint`: Overall participant-macro SBP/DBP/mean MAE is
  **7.6485/4.1693/5.9089 mmHg**.
- `chronological_blocked`: Overall participant-macro SBP/DBP/mean MAE is
  **8.0361/4.3716/6.2039 mmHg**.

The chronological result is 0.2949 mmHg worse in mean MAE, showing that random
same-subject window assignment gives an easier estimate of performance. The
largest loss is in chronological VitalDB, where mean MAE rises to 6.7472 mmHg.

These are **internal-validation results from a development-only same-subject
analogue**. They are not unseen-participant results, not K=1/2/3/5 cuff
calibration results, not external validation, and not an exact reproduction of
the official PulseDB CalBased benchmark.

## Frozen protocol and reporting boundary

- Protocol: `development-calbased-analogue-v1`.
- Seed: `20260828`.
- Eligible parent partition: the existing frozen `meta_train` only.
- Retained cohort: 2,051 participants: 1,011 PulseDB MIMIC and 1,040 PulseDB
  VitalDB.
- Per participant and split mode: 320 labelled training windows, 40 internal-
  validation windows, and 40 sealed held-out windows.
- Internal-validation comparison size: 82,040 windows in total: 40,440 MIMIC
  and 41,600 VitalDB.
- Query input for all neural settings: one 10-second PPG window. Participant
  training-label information is used only by explicitly named seen-subject
  methods.
- Selection metric: Overall participant-macro mean of SBP and DBP MAE.
- Held-out test accessed: **no**.

`random_disjoint` assigns non-overlapping windows through a deterministic
participant-specific random procedure. `chronological_blocked` requires the
training windows to precede validation and held-out windows in record/time
order. Results from these modes are reported independently and are never
averaged.

Before training, exact PPG-content overlap across roles was audited and seven
affected participants were excluded. The accepted cohort contains no exact
PPG-content duplicate across or within roles.

## Overall participant-macro comparison

### Random disjoint windows

| Rank | Setting | Main idea | SBP MAE | DBP MAE | Mean MAE |
|---:|---|---|---:|---:|---:|
| 1 | `subject_mean_residual_ppg` | participant train-label mean + compact-ResNet PPG residual | 7.6485 | 4.1693 | **5.9089** |
| 2 | `inception_time_wide` | direct InceptionTime-wide PPG regression | 8.2109 | 4.9570 | 6.5839 |
| 3 | `self_attention_resunet_adaptation` | direct self-attention ResUNet PPG regression | 8.7280 | 5.3343 | 7.0312 |
| 4 | `runet_resunet_encoder_adaptation` | direct rU-Net/ResUNet PPG regression | 9.3066 | 5.6984 | 7.5025 |
| 5 | `patch_transformer` | direct patch-Transformer PPG regression | 9.4143 | 5.7462 | 7.5803 |
| 6 | `compact_resnet` | direct compact-ResNet PPG regression | 9.4550 | 5.7658 | 7.6104 |
| 7 | `subject_train_mean` | participant train-label mean only | 10.4163 | 5.5423 | 7.9793 |
| 8 | `cnn_bilstm_adaptation` | direct CNN-BiLSTM PPG regression | 10.1650 | 6.2162 | 8.1906 |
| 9 | `cnn_transformer_aff_adaptation` | direct CNN-Transformer/AFF PPG regression | 10.7820 | 6.5788 | 8.6804 |

Relative to `subject_train_mean`, the winner reduces Overall mean MAE by
2.0704 mmHg, or 25.95%.

### Chronological blocked windows

| Rank | Setting | Main idea | SBP MAE | DBP MAE | Mean MAE |
|---:|---|---|---:|---:|---:|
| 1 | `subject_mean_residual_ppg` | participant train-label mean + compact-ResNet PPG residual | 8.0361 | 4.3716 | **6.2039** |
| 2 | `inception_time_wide` | direct InceptionTime-wide PPG regression | 9.1245 | 5.5620 | 7.3433 |
| 3 | `subject_train_mean` | participant train-label mean only | 9.7773 | 5.1721 | 7.4747 |
| 4 | `self_attention_resunet_adaptation` | direct self-attention ResUNet PPG regression | 9.3580 | 5.7502 | 7.5541 |
| 5 | `compact_resnet` | direct compact-ResNet PPG regression | 9.6869 | 5.9034 | 7.7951 |
| 6 | `runet_resunet_encoder_adaptation` | direct rU-Net/ResUNet PPG regression | 9.7230 | 5.8998 | 7.8114 |
| 7 | `patch_transformer` | direct patch-Transformer PPG regression | 10.2317 | 6.2880 | 8.2598 |
| 8 | `cnn_bilstm_adaptation` | direct CNN-BiLSTM PPG regression | 10.6321 | 6.3569 | 8.4945 |
| 9 | `cnn_transformer_aff_adaptation` | direct CNN-Transformer/AFF PPG regression | 10.7410 | 6.5593 | 8.6501 |

Relative to `subject_train_mean`, the winner reduces Overall mean MAE by
1.2708 mmHg, or 17.00%.

## Four requested source-by-split results

The participant-macro SBP/DBP/mean MAE is the primary result. R², ME, STD,
threshold percentages, AAMI, and BHS are event-pooled diagnostics computed
from the same saved predictions. MIMIC and VitalDB are internal PulseDB source
strata, not independent external datasets.

### 1. PulseDB MIMIC — random disjoint

Participant-macro SBP/DBP/mean MAE: **7.6699/4.1324/5.9012 mmHg**.

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `subject_mean_residual_ppg` | SBP | 7.6699 | 0.7750 | 0.4492 | 10.7868 | 46.62% | 73.50% | 86.90% | FAIL* | FAIL, Grade C* |
| `subject_mean_residual_ppg` | DBP | 4.1324 | 0.7769 | -0.1015 | 6.2800 | 72.34% | 91.96% | 97.12% | PASS* | PASS, Grade A* |

### 2. PulseDB VitalDB — random disjoint

Participant-macro SBP/DBP/mean MAE: **7.6277/4.2052/5.9165 mmHg**.

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `subject_mean_residual_ppg` | SBP | 7.6277 | 0.6881 | 0.1317 | 10.5786 | 46.17% | 73.30% | 87.01% | FAIL* | FAIL, Grade C* |
| `subject_mean_residual_ppg` | DBP | 4.2052 | 0.7714 | 0.0723 | 5.8572 | 69.57% | 91.61% | 97.68% | PASS* | PASS, Grade A* |

### 3. PulseDB MIMIC — chronological blocked

Participant-macro SBP/DBP/mean MAE: **7.4341/3.8558/5.6449 mmHg**.

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `subject_mean_residual_ppg` | SBP | 7.4341 | 0.7817 | -0.0406 | 10.5877 | 48.44% | 75.12% | 87.64% | FAIL* | FAIL, Grade C* |
| `subject_mean_residual_ppg` | DBP | 3.8558 | 0.7952 | -0.1821 | 5.8472 | 74.96% | 92.44% | 97.32% | PASS* | PASS, Grade A* |

### 4. PulseDB VitalDB — chronological blocked

Participant-macro SBP/DBP/mean MAE: **8.6213/4.8730/6.7472 mmHg**.

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `subject_mean_residual_ppg` | SBP | 8.6213 | 0.5737 | -0.2126 | 11.6051 | 40.90% | 67.55% | 82.47% | FAIL* | FAIL, Grade D* |
| `subject_mean_residual_ppg` | DBP | 4.8730 | 0.6704 | -0.4424 | 6.7330 | 62.98% | 87.96% | 96.58% | PASS* | PASS, Grade A* |

## Interpretation

1. Seen-subject calibration information is useful. Adding the PPG residual to
   the participant's training-label mean clearly outperforms using that mean
   alone in both split modes.
2. Direct PPG regression is weaker than the residual formulation. The best
   direct model is InceptionTime-wide, but its mean MAE is 0.6750 mmHg worse
   than the residual model under random assignment and 1.1394 mmHg worse under
   chronological assignment.
3. The split definition materially changes the conclusion. Random and
   chronological results must remain separate; reporting only the random split
   would hide the later-window degradation in VitalDB.
4. DBP passes both retrospective numerical screens in all four source/split
   views, whereas SBP does not. The main limitation is therefore SBP error
   dispersion, especially chronological VitalDB, rather than a large mean
   bias.

## Standards qualification

The starred AAMI and BHS entries are **retrospective numerical screens only**.
They apply threshold calculations to saved event-level errors but do not
establish formal compliance with a device-validation standard. The dataset,
protocol, reference procedure, sample independence, repeated-measure handling,
and clinical validation design were not constructed as a formal AAMI/ISO or
BHS device-validation study.

## Execution and reporting audit

All 18 model/baseline jobs completed successfully. The two original report
jobs stopped after training because their target-consistency assertion required
bit-exact equality between float32 and float64 serializations. Composite keys
and target values were re-audited: neural target files were exact, while the
subject-mean baseline differed by at most approximately 2.24e-5 mmHg from
serialization precision alone. The reporter now sorts by participant/source/
event keys and uses a 1e-4-mmHg absolute tolerance with zero relative
tolerance. A 0.01-mmHg mismatch still fails the regression test.

The repaired report was rebuilt from the unchanged saved predictions. Work and
NAS report directories are byte-identical. No model was retrained, no new
experiment was submitted, and sealed held-out targets remained inaccessible.

## Public result files

- [`participant_macro.csv`](../results/same_subject_dual_split/participant_macro.csv):
  all nine settings, both splits, and Overall/MIMIC/VitalDB participant-macro
  summaries.
- [`event_pooled_diagnostics.csv`](../results/same_subject_dual_split/event_pooled_diagnostics.csv):
  event-pooled diagnostics for the same settings and views.
- [`four_view_winner.csv`](../results/same_subject_dual_split/four_view_winner.csv):
  the requested four source-by-split winner views in the standard result-table
  format.
- [`manifest.json`](../results/same_subject_dual_split/manifest.json): protocol,
  selection boundary, source-report checksums, and held-out-access declaration.
