# Experiment results by protocol

Verified on 10 September 2026. This index retrieves existing experiments; it does not launch or modify training. All MAEs are in mmHg. Participant-macro MAE is primary. Different cohorts, label budgets, query definitions and split modes must not be ranked as if they were one controlled experiment.

Latest publication, 11 September 2026: the **[20%-90% enrollment-budget study](../results/enrollment_budget_v1_20260911/README.md)**
is complete. It retains the same shared model, 762 final people and 78,237
queries at every budget. Memory SBP/DBP MAE is 4.5797/2.5934 at 20%,
3.2850/1.8817 at 70% and 3.2081/1.8336 at 90%. The 90% reference is an exact
repeat of the parent predictions, not an independent replication. The report
preserves all eight budgets, both methods, three source views and the full
diagnostic tables. No new training was submitted for this publication.

Publication update, 11 September 2026: the full-cohort final evaluation is now
complete. The [completed report](../results/full_cohort_enrollment_v1_20260910/README.md)
contains both final and validation results. Final participant-macro LoRA
SBP/DBP MAE is 3.8556/2.1661; LoRA plus memory is 3.2081/1.8336 mmHg across
762 people. Section 1 below preserves the earlier validation-time snapshot;
it is no longer the current job status.

## 1. Full-cohort enrollment: recorded validation-time snapshot

Run: `full-cohort-enrollment-v1_20260910-150200`.
At 2026-09-10 23:30:26 China time (15:30:26 UTC), test jobs 1770/1771 were running and final scorer 1772 was dependency-pending. The logged profile counts were 32/381 and 36/381. These are build-progress counts, not final test results.

Completed stages:

| Stage | Job | Elapsed | Outcome |
| --- | --- | --- | --- |
| Full waveform materialization | 1764 | 47 min 27 s | Completed, exit 0:0 |
| Cohort preparation and audit | 1765 | 5 min 13 s | Completed, exit 0:0 |
| Fresh population model | 1766 | 6 h 5 min 16 s | Completed; best epoch 5, stopped after epoch 13 |
| Personal validation shard 0 | 1767 | 1 h 25 min 23 s | Completed |
| Personal validation shard 1 | 1768 | 1 h 25 min 14 s | Completed in parallel with shard 0 |
| Validation scoring | 1769 | 11 s | Completed |

All 5,361 original people and 5,245,454 source windows were audited. Original outer subject assignments were unchanged. After outcome-blind exclusions, 5,275 people remain: 3,750 population-training, 763 validation and 762 test. The 86 excluded people comprise 83 with insufficient separable registration/selection/query history and three in cross-outer content/interval-linked components. Exact duplicate rows within an outer role were also collapsed; this did not exclude all affected people. Total excluded windows: 22,502.

The population model fits 3,672,008 windows per epoch. Batch size 64 gives 57,376 optimizer steps per epoch, explaining the approximately 26–31 minutes per epoch. Thirteen epochs entail 47,736,104 window presentations, not thirteen passes over a small pilot. The two requested methods share this one fit and each user's fitted adapter; memory is a paired addition, not another entire backbone retraining.

Validation uses 691,929 registration windows and 76,909 distinct query windows across 763 users (344 MIMIC / 419 VitalDB). Test has 703,869 registration and 78,237 distinct query windows across 762 users (343 / 419). Each target user is absent from population fitting, then enrolls using approximately 90% of their own eligible windows. Personal selection uses only an inner registration subset; query BP does not fit personal parameters or memory. This is substantial enrollment, not K=1/2/3/5 calibration, not zero-calibration, and not a future-date experiment. Validation selects the population checkpoint, so this validation result is not the final test result.

| Scope | Setting | Participants | Queries | SBP MAE | DBP MAE | Mean MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | new_person_lora | 763 | 76909 | 4.0121 | 2.2543 | 3.1332 |
| MIMIC | new_person_lora | 344 | 55567 | 4.1050 | 2.2938 | 3.1994 |
| VitalDB | new_person_lora | 419 | 21342 | 3.9358 | 2.2218 | 3.0788 |
| Overall | new_person_lora_memory | 763 | 76909 | 3.3595 | 1.8714 | 2.6154 |
| MIMIC | new_person_lora_memory | 344 | 55567 | 3.5320 | 1.9479 | 2.7400 |
| VitalDB | new_person_lora_memory | 419 | 21342 | 3.2179 | 1.8086 | 2.5132 |

The primary paired memory improvement is 0.5177 mmHg in mean participant MAE (16.52%); the saved participant-level, source-stratified 2,000-bootstrap interval is [0.4622, 0.5706] mmHg. It is conditional on this fitted model and does not quantify training-seed uncertainty. No new hypothesis test was run for this index.

The complete requested Setting/BP/MAE/R²/ME/STD/≤5/≤10/≤15/AAMI/BHS tables are in [validation diagnostic tables](../results/full_cohort_enrollment_v1_20260910/validation/RESULT_TABLES.md). Their MAE is window-pooled; it differs from participant-macro MAE above because users have unequal query counts. AAMI/BHS are saved retrospective numerical screens, not clinical/device certification. All three source views were recomputed by the original frozen scorer, not averaged from two source scores.

- [Primary aggregate CSV](../results/full_cohort_enrollment_v1_20260910/validation/participant_macro.csv)
- [Paired intervals](../results/full_cohort_enrollment_v1_20260910/validation/paired_participant_intervals.csv)
- [Scoring receipt](../results/full_cohort_enrollment_v1_20260910/validation/evaluation_receipt.json)

## 2. Seen-user overlapping-subject protocol: direct backbone comparison was completed

This is the development CalBased analogue, not exact official CalBased membership. It used 2,051 people: 320 labelled train windows, 40 internal-validation windows and 40 sealed windows per person. Train and internal-validation people intentionally overlap, but exact windows/content do not. Random-disjoint and chronological-blocked results are separate.

Seven direct PPG regressors were compared: compact ResNet, wide InceptionTime, patch Transformer, self-attention ResUNet adaptation, residual U-Net adaptation, CNN-BiLSTM adaptation, and CNN-Transformer/AFF adaptation. Two additional controls were the personal training BP mean and that mean plus a ResNet PPG residual. Thus there were nine settings per split mode, not nine pure backbones. All 18 runs completed. The architecture-family adaptations are not full reproductions of the corresponding papers.

| Split mode | Scope | Setting | SBP MAE | DBP MAE | Mean MAE |
| --- | --- | --- | --- | --- | --- |
| chronological_blocked | MIMIC | subject_mean_residual_ppg | 7.4341 | 3.8558 | 5.6449 |
| chronological_blocked | MIMIC | inception_time_wide | 8.5515 | 4.8617 | 6.7066 |
| chronological_blocked | MIMIC | subject_train_mean | 9.1618 | 4.6239 | 6.8928 |
| chronological_blocked | MIMIC | self_attention_resunet_adaptation | 8.8878 | 5.1414 | 7.0146 |
| chronological_blocked | MIMIC | compact_resnet | 9.3261 | 5.2926 | 7.3093 |
| chronological_blocked | MIMIC | runet_resunet_encoder_adaptation | 9.4157 | 5.3220 | 7.3688 |
| chronological_blocked | MIMIC | patch_transformer | 10.0383 | 5.6810 | 7.8597 |
| chronological_blocked | MIMIC | cnn_bilstm_adaptation | 10.5273 | 5.8585 | 8.1929 |
| chronological_blocked | MIMIC | cnn_transformer_aff_adaptation | 10.5873 | 5.9952 | 8.2913 |
| chronological_blocked | Overall | subject_mean_residual_ppg | 8.0361 | 4.3716 | 6.2039 |
| chronological_blocked | Overall | inception_time_wide | 9.1245 | 5.5620 | 7.3433 |
| chronological_blocked | Overall | subject_train_mean | 9.7773 | 5.1721 | 7.4747 |
| chronological_blocked | Overall | self_attention_resunet_adaptation | 9.3580 | 5.7502 | 7.5541 |
| chronological_blocked | Overall | compact_resnet | 9.6869 | 5.9034 | 7.7951 |
| chronological_blocked | Overall | runet_resunet_encoder_adaptation | 9.7230 | 5.8998 | 7.8114 |
| chronological_blocked | Overall | patch_transformer | 10.2317 | 6.2880 | 8.2598 |
| chronological_blocked | Overall | cnn_bilstm_adaptation | 10.6321 | 6.3569 | 8.4945 |
| chronological_blocked | Overall | cnn_transformer_aff_adaptation | 10.7410 | 6.5593 | 8.6501 |
| chronological_blocked | VitalDB | subject_mean_residual_ppg | 8.6213 | 4.8730 | 6.7472 |
| chronological_blocked | VitalDB | inception_time_wide | 9.6816 | 6.2427 | 7.9621 |
| chronological_blocked | VitalDB | subject_train_mean | 10.3757 | 5.7050 | 8.0404 |
| chronological_blocked | VitalDB | self_attention_resunet_adaptation | 9.8151 | 6.3420 | 8.0785 |
| chronological_blocked | VitalDB | runet_resunet_encoder_adaptation | 10.0217 | 6.4616 | 8.2416 |
| chronological_blocked | VitalDB | compact_resnet | 10.0376 | 6.4972 | 8.2674 |
| chronological_blocked | VitalDB | patch_transformer | 10.4197 | 6.8779 | 8.6488 |
| chronological_blocked | VitalDB | cnn_bilstm_adaptation | 10.7340 | 6.8414 | 8.7877 |
| chronological_blocked | VitalDB | cnn_transformer_aff_adaptation | 10.8904 | 7.1076 | 8.9990 |
| random_disjoint | MIMIC | subject_mean_residual_ppg | 7.6699 | 4.1324 | 5.9012 |
| random_disjoint | MIMIC | inception_time_wide | 8.2629 | 4.7268 | 6.4949 |
| random_disjoint | MIMIC | self_attention_resunet_adaptation | 8.9231 | 5.2045 | 7.0638 |
| random_disjoint | MIMIC | runet_resunet_encoder_adaptation | 9.5201 | 5.5188 | 7.5194 |
| random_disjoint | MIMIC | subject_train_mean | 9.9727 | 5.2203 | 7.5965 |
| random_disjoint | MIMIC | compact_resnet | 9.7192 | 5.6272 | 7.6732 |
| random_disjoint | MIMIC | patch_transformer | 9.7834 | 5.6064 | 7.6949 |
| random_disjoint | MIMIC | cnn_bilstm_adaptation | 10.7222 | 6.1738 | 8.4480 |
| random_disjoint | MIMIC | cnn_transformer_aff_adaptation | 11.5108 | 6.5879 | 9.0494 |
| random_disjoint | Overall | subject_mean_residual_ppg | 7.6485 | 4.1693 | 5.9089 |
| random_disjoint | Overall | inception_time_wide | 8.2109 | 4.9570 | 6.5839 |
| random_disjoint | Overall | self_attention_resunet_adaptation | 8.7280 | 5.3343 | 7.0312 |
| random_disjoint | Overall | runet_resunet_encoder_adaptation | 9.3066 | 5.6984 | 7.5025 |
| random_disjoint | Overall | patch_transformer | 9.4143 | 5.7462 | 7.5803 |
| random_disjoint | Overall | compact_resnet | 9.4550 | 5.7658 | 7.6104 |
| random_disjoint | Overall | subject_train_mean | 10.4163 | 5.5423 | 7.9793 |
| random_disjoint | Overall | cnn_bilstm_adaptation | 10.1650 | 6.2162 | 8.1906 |
| random_disjoint | Overall | cnn_transformer_aff_adaptation | 10.7820 | 6.5788 | 8.6804 |
| random_disjoint | VitalDB | subject_mean_residual_ppg | 7.6277 | 4.2052 | 5.9165 |
| random_disjoint | VitalDB | inception_time_wide | 8.1603 | 5.1807 | 6.6705 |
| random_disjoint | VitalDB | self_attention_resunet_adaptation | 8.5385 | 5.4605 | 6.9995 |
| random_disjoint | VitalDB | patch_transformer | 9.0555 | 5.8822 | 7.4688 |
| random_disjoint | VitalDB | runet_resunet_encoder_adaptation | 9.0990 | 5.8731 | 7.4860 |
| random_disjoint | VitalDB | compact_resnet | 9.1982 | 5.9006 | 7.5494 |
| random_disjoint | VitalDB | cnn_bilstm_adaptation | 9.6234 | 6.2573 | 7.9403 |
| random_disjoint | VitalDB | cnn_transformer_aff_adaptation | 10.0736 | 6.5699 | 8.3218 |
| random_disjoint | VitalDB | subject_train_mean | 10.8476 | 5.8554 | 8.3515 |

The best direct backbone was wide InceptionTime in both modes. The personal-mean-plus-PPG-residual control outperformed every direct regressor. These are internal-validation results; the sealed role was not opened in this screen.

- [Original result narrative](RESULTS_SAME_SUBJECT_DUAL_SPLIT.md)
- [All participant-macro values](../results/same_subject_dual_split/participant_macro.csv)
- [All requested diagnostic columns, both modes and all three scopes](../results/same_subject_dual_split/event_pooled_diagnostics.csv)

## 3. Seen-user single-component screen: 19 completed candidates

Same 2,051-person development analogue, random-disjoint mode, 320 personal labelled training windows and 40 internal-validation windows per user. These are additions/replacements on the personal-mean residual formulation, not a pure-backbone-only ranking. This is where the participant-indexed rank-4 LoRA result was established. A five-window context in some competitors is not a five-label total budget.

| Scope | Setting | Participants | Queries | SBP MAE | DBP MAE | Mean MAE |
| --- | --- | --- | --- | --- | --- | --- |
| MIMIC | residual_subject_lora_rank4 | 1011 | 40440 | 4.3375 | 2.3832 | 3.3604 |
| MIMIC | residual_film | 1011 | 40440 | 6.3258 | 3.4400 | 4.8829 |
| MIMIC | residual_support_attention | 1011 | 40440 | 6.9302 | 3.7155 | 5.3229 |
| MIMIC | residual_multi_event_weighting | 1011 | 40440 | 6.9745 | 3.7312 | 5.3528 |
| MIMIC | residual_support_reliability | 1011 | 40440 | 7.0028 | 3.7907 | 5.3967 |
| MIMIC | residual_calibration_relative | 1011 | 40440 | 7.0383 | 3.8071 | 5.4227 |
| MIMIC | residual_inception_time_wide | 1011 | 40440 | 7.0587 | 3.8194 | 5.4391 |
| MIMIC | residual_demographics_direct | 1011 | 40440 | 7.3775 | 3.9939 | 5.6857 |
| MIMIC | residual_conformer | 1011 | 40440 | 7.4605 | 4.0246 | 5.7426 |
| MIMIC | residual_quality_gate | 1011 | 40440 | 7.6794 | 4.1278 | 5.9036 |
| MIMIC | residual_soft_moe | 1011 | 40440 | 7.6862 | 4.1315 | 5.9089 |
| MIMIC | residual_quality_weighted_loss | 1011 | 40440 | 7.7029 | 4.1601 | 5.9315 |
| MIMIC | residual_reference | 1011 | 40440 | 7.7687 | 4.1947 | 5.9817 |
| MIMIC | residual_ppg_quality_filter | 1011 | 40440 | 7.7976 | 4.1989 | 5.9982 |
| MIMIC | residual_prototype_moe | 1011 | 40440 | 7.8255 | 4.2293 | 6.0274 |
| MIMIC | residual_beat_similarity_filter | 1011 | 40440 | 8.0496 | 4.3127 | 6.1812 |
| MIMIC | residual_cnn_bilstm | 1011 | 40440 | 8.1633 | 4.3956 | 6.2794 |
| MIMIC | residual_patch_transformer | 1011 | 40440 | 8.2088 | 4.3992 | 6.3040 |
| MIMIC | residual_cnn_gru | 1011 | 40440 | 8.9028 | 4.7224 | 6.8126 |
| Overall | residual_subject_lora_rank4 | 2051 | 82040 | 3.9663 | 2.2043 | 3.0853 |
| Overall | residual_film | 2051 | 82040 | 6.1537 | 3.4069 | 4.7803 |
| Overall | residual_support_attention | 2051 | 82040 | 6.7819 | 3.7060 | 5.2440 |
| Overall | residual_multi_event_weighting | 2051 | 82040 | 6.8461 | 3.7381 | 5.2921 |
| Overall | residual_support_reliability | 2051 | 82040 | 6.8814 | 3.7818 | 5.3316 |
| Overall | residual_calibration_relative | 2051 | 82040 | 6.8910 | 3.7849 | 5.3379 |
| Overall | residual_inception_time_wide | 2051 | 82040 | 7.0676 | 3.8611 | 5.4643 |
| Overall | residual_demographics_direct | 2051 | 82040 | 7.3475 | 4.0240 | 5.6858 |
| Overall | residual_conformer | 2051 | 82040 | 7.4051 | 4.0385 | 5.7218 |
| Overall | residual_quality_gate | 2051 | 82040 | 7.6183 | 4.1423 | 5.8803 |
| Overall | residual_quality_weighted_loss | 2051 | 82040 | 7.7016 | 4.2006 | 5.9511 |
| Overall | residual_reference | 2051 | 82040 | 7.7155 | 4.2092 | 5.9624 |
| Overall | residual_soft_moe | 2051 | 82040 | 7.7384 | 4.2067 | 5.9725 |
| Overall | residual_ppg_quality_filter | 2051 | 82040 | 7.7436 | 4.2174 | 5.9805 |
| Overall | residual_prototype_moe | 2051 | 82040 | 7.7972 | 4.2554 | 6.0263 |
| Overall | residual_cnn_bilstm | 2051 | 82040 | 8.0539 | 4.3974 | 6.2256 |
| Overall | residual_patch_transformer | 2051 | 82040 | 8.1691 | 4.4386 | 6.3038 |
| Overall | residual_beat_similarity_filter | 2051 | 82040 | 8.2067 | 4.4630 | 6.3348 |
| Overall | residual_cnn_gru | 2051 | 82040 | 8.7722 | 4.7324 | 6.7523 |
| VitalDB | residual_subject_lora_rank4 | 1040 | 41600 | 3.6055 | 2.0304 | 2.8180 |
| VitalDB | residual_film | 1040 | 41600 | 5.9865 | 3.3747 | 4.6806 |
| VitalDB | residual_support_attention | 1040 | 41600 | 6.6378 | 3.6967 | 5.1672 |
| VitalDB | residual_multi_event_weighting | 1040 | 41600 | 6.7213 | 3.7448 | 5.2331 |
| VitalDB | residual_calibration_relative | 1040 | 41600 | 6.7477 | 3.7633 | 5.2555 |
| VitalDB | residual_support_reliability | 1040 | 41600 | 6.7634 | 3.7731 | 5.2682 |
| VitalDB | residual_inception_time_wide | 1040 | 41600 | 7.0761 | 3.9017 | 5.4889 |
| VitalDB | residual_demographics_direct | 1040 | 41600 | 7.3183 | 4.0533 | 5.6858 |
| VitalDB | residual_conformer | 1040 | 41600 | 7.3512 | 4.0520 | 5.7016 |
| VitalDB | residual_quality_gate | 1040 | 41600 | 7.5588 | 4.1564 | 5.8576 |
| VitalDB | residual_reference | 1040 | 41600 | 7.6638 | 4.2234 | 5.9436 |
| VitalDB | residual_ppg_quality_filter | 1040 | 41600 | 7.6911 | 4.2354 | 5.9632 |
| VitalDB | residual_quality_weighted_loss | 1040 | 41600 | 7.7004 | 4.2400 | 5.9702 |
| VitalDB | residual_prototype_moe | 1040 | 41600 | 7.7697 | 4.2807 | 6.0252 |
| VitalDB | residual_soft_moe | 1040 | 41600 | 7.7891 | 4.2798 | 6.0344 |
| VitalDB | residual_cnn_bilstm | 1040 | 41600 | 7.9475 | 4.3991 | 6.1733 |
| VitalDB | residual_patch_transformer | 1040 | 41600 | 8.1305 | 4.4769 | 6.3037 |
| VitalDB | residual_beat_similarity_filter | 1040 | 41600 | 8.3593 | 4.6091 | 6.4842 |
| VitalDB | residual_cnn_gru | 1040 | 41600 | 8.6453 | 4.7422 | 6.6938 |

- [Original 19-candidate report](RESULTS_SAME_SUBJECT_SINGLE_COMPONENT.md)
- [All requested diagnostic columns](../results/same_subject_single_component/event_pooled_diagnostics.csv)
- [Subsequent combination experiments](RESULTS_SAME_SUBJECT_COMBINATIONS.md)
- [Subsequent personal-memory experiments](RESULTS_PERSONAL_MEMORY_V2.md)

## 4. Subject-disjoint, no-personal-calibration backbone experiments also exist

Rounds 11A, 12 and 13 saved separate `Population` and `QGH` prediction columns. The former is direct PPG-to-BP regression with no target-user support BP, anchor, adapter or memory. The latter adds K=5 Quality Gate + Huber calibration. Older narrative headlines primarily showed QGH, which can obscure the completed calibration-free results.

The code path was checked: `train.py` creates `PopulationDataset` and calls the population model with only `batch["ppg"]`; `round10_end_to_end.py::_encode_metadata` produces uncalibrated PPG predictions, and `round11_backbones.py::_prediction_view` selects the separate population prediction columns. The saved K field belongs to a shared comparison schema and does not make those population predictions calibrated.

All three rounds use the earlier event120-v1 store and person-disjoint internal folds: folds 0–2 fit the network, fold 3 selects the checkpoint, and fold 4 ranks candidates. The fold-4 assessment includes 628 people (285 MIMIC / 343 VitalDB) and 96,332 query events. This is internal development, not the original locked 804-person test. Event eligibility and the event-6-onward query definition are retained even for Population models. These experiments therefore do not constitute an all-raw-window benchmark over the current full cohort.

There are 23 saved Population fits across these three rounds, covering 18 distinct backbone configurations. Repeated references and round-specific seeds must remain separate.

### Round 11A — five population backbones

| Scope | Backbone | SBP MAE | DBP MAE | Mean MAE |
| --- | --- | --- | --- | --- |
| Overall | resnet_small | 14.0885 | 8.8877 | 11.4881 |
| MIMIC | resnet_small | 15.8451 | 9.6170 | 12.7311 |
| VitalDB | resnet_small | 12.6289 | 8.2818 | 10.4553 |
| Overall | resnet_deep | 14.3674 | 8.8660 | 11.6167 |
| MIMIC | resnet_deep | 16.2693 | 9.5009 | 12.8851 |
| VitalDB | resnet_deep | 12.7870 | 8.3384 | 10.5627 |
| Overall | inception_time | 14.1843 | 8.9729 | 11.5786 |
| MIMIC | inception_time | 16.1089 | 9.5123 | 12.8106 |
| VitalDB | inception_time | 12.5851 | 8.5246 | 10.5549 |
| Overall | patch_transformer | 14.3665 | 8.9453 | 11.6559 |
| MIMIC | patch_transformer | 16.3608 | 9.5478 | 12.9543 |
| VitalDB | patch_transformer | 12.7095 | 8.4447 | 10.5771 |
| Overall | conformer | 14.4729 | 9.0721 | 11.7725 |
| MIMIC | conformer | 16.5974 | 9.7176 | 13.1575 |
| VitalDB | conformer | 12.7077 | 8.5357 | 10.6217 |

[Original primary table](../results/round11a/participant_macro_internal.csv) · [Full diagnostic table](../results/round11a/pooled_diagnostics_internal.csv). Select `Model=Population` or the `Population` setting; QGH rows are not calibration-free.

### Round 12 — five population backbones

| Scope | Backbone | SBP MAE | DBP MAE | Mean MAE |
| --- | --- | --- | --- | --- |
| Overall | resnet_small | 14.1130 | 8.7813 | 11.4471 |
| MIMIC | resnet_small | 16.1315 | 9.3290 | 12.7302 |
| VitalDB | resnet_small | 12.4358 | 8.3261 | 10.3810 |
| Overall | tcn_bp | 14.4871 | 8.8940 | 11.6906 |
| MIMIC | tcn_bp | 16.5732 | 9.5000 | 13.0366 |
| VitalDB | tcn_bp | 12.7538 | 8.3905 | 10.5721 |
| Overall | fewshot_resnet_attention | 14.5267 | 9.1441 | 11.8354 |
| MIMIC | fewshot_resnet_attention | 16.5641 | 9.9886 | 13.2764 |
| VitalDB | fewshot_resnet_attention | 12.8337 | 8.4423 | 10.6380 |
| Overall | bp_crnn | 14.3966 | 8.9533 | 11.6749 |
| MIMIC | bp_crnn | 16.3505 | 9.6097 | 12.9801 |
| VitalDB | bp_crnn | 12.7730 | 8.4078 | 10.5904 |
| Overall | resunet_encoder | 14.0685 | 8.8777 | 11.4731 |
| MIMIC | resunet_encoder | 16.0621 | 9.5071 | 12.7846 |
| VitalDB | resunet_encoder | 12.4120 | 8.3548 | 10.3834 |

[Original primary table](../results/round12/participant_macro_internal.csv) · [Full diagnostic table](../results/round12/pooled_diagnostics_internal.csv). Select `Model=Population` or the `Population` setting; QGH rows are not calibration-free.

### Round 13 — thirteen population/capacity configurations

| Scope | Backbone | SBP MAE | DBP MAE | Mean MAE |
| --- | --- | --- | --- | --- |
| Overall | resnet_small | 14.1159 | 8.8510 | 11.4835 |
| MIMIC | resnet_small | 16.0001 | 9.4823 | 12.7412 |
| VitalDB | resnet_small | 12.5503 | 8.3265 | 10.4384 |
| Overall | resnet_depth2 | 14.1449 | 8.8977 | 11.5213 |
| MIMIC | resnet_depth2 | 16.2156 | 9.4656 | 12.8406 |
| VitalDB | resnet_depth2 | 12.4243 | 8.4258 | 10.4251 |
| Overall | resnet_wide1p5 | 14.3430 | 8.8738 | 11.6084 |
| MIMIC | resnet_wide1p5 | 16.3051 | 9.4495 | 12.8773 |
| VitalDB | resnet_wide1p5 | 12.7126 | 8.3954 | 10.5540 |
| Overall | inception_time | 14.1997 | 8.8399 | 11.5198 |
| MIMIC | inception_time | 16.2831 | 9.5105 | 12.8968 |
| VitalDB | inception_time | 12.4686 | 8.2827 | 10.3757 |
| Overall | inception_time_wide | 14.1713 | 8.8205 | 11.4959 |
| MIMIC | inception_time_wide | 16.2767 | 9.5693 | 12.9230 |
| VitalDB | inception_time_wide | 12.4220 | 8.1984 | 10.3102 |
| Overall | patch_transformer | 14.4205 | 8.9987 | 11.7096 |
| MIMIC | patch_transformer | 16.5738 | 9.5828 | 13.0783 |
| VitalDB | patch_transformer | 12.6314 | 8.5133 | 10.5724 |
| Overall | patch_transformer_deep | 14.4312 | 9.0572 | 11.7442 |
| MIMIC | patch_transformer_deep | 16.6262 | 9.7984 | 13.2123 |
| VitalDB | patch_transformer_deep | 12.6075 | 8.4413 | 10.5244 |
| Overall | patch_transformer_wide | 14.4367 | 9.0448 | 11.7407 |
| MIMIC | patch_transformer_wide | 16.4057 | 9.6910 | 13.0484 |
| VitalDB | patch_transformer_wide | 12.8006 | 8.5078 | 10.6542 |
| Overall | patch_transformer_highres | 14.5769 | 9.0220 | 11.7994 |
| MIMIC | patch_transformer_highres | 16.7183 | 9.7519 | 13.2351 |
| VitalDB | patch_transformer_highres | 12.7975 | 8.4155 | 10.6065 |
| Overall | patch_transformer_longpatch | 14.5370 | 9.0713 | 11.8042 |
| MIMIC | patch_transformer_longpatch | 16.6691 | 9.7296 | 13.1993 |
| VitalDB | patch_transformer_longpatch | 12.7655 | 8.5243 | 10.6449 |
| Overall | conformer | 14.4019 | 9.0251 | 11.7135 |
| MIMIC | conformer | 16.4626 | 9.7526 | 13.1076 |
| VitalDB | conformer | 12.6896 | 8.4206 | 10.5551 |
| Overall | conformer_large | 14.5471 | 9.0256 | 11.7864 |
| MIMIC | conformer_large | 16.6494 | 9.6893 | 13.1694 |
| VitalDB | conformer_large | 12.8004 | 8.4740 | 10.6372 |
| Overall | convnext_1d | 14.0975 | 8.8611 | 11.4793 |
| MIMIC | convnext_1d | 15.8569 | 9.4442 | 12.6506 |
| VitalDB | convnext_1d | 12.6355 | 8.3766 | 10.5061 |

[Original primary table](../results/round13/participant_macro_internal.csv) · [Full diagnostic table](../results/round13/pooled_diagnostics_internal.csv). Select `Model=Population` or the `Population` setting; QGH rows are not calibration-free.

The pure population models generally have Overall SBP MAE around 14–14.6 mmHg and DBP MAE around 8.8–9.1 mmHg in these screens. This supports "tested, but limited calibration-free accuracy under this protocol," not "never tested" or "all backbone families are ineffective."

## 5. Subject-excluded substantial enrollment had also been tested before the full cohort

The completed 200-person enrollment experiment removed those people from population fitting, then allowed each to provide 360 labelled registration windows and evaluated 40 other windows. This is neither seen-user population fitting nor few-shot calibration.

| Scope | Setting | Participants | Queries | SBP MAE | DBP MAE | Mean MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | new_person_lora | 200 | 8000 | 3.9049 | 2.1858 | 3.0453 |
| MIMIC | new_person_lora | 100 | 4000 | 4.2868 | 2.3628 | 3.3248 |
| VitalDB | new_person_lora | 100 | 4000 | 3.5230 | 2.0088 | 2.7659 |
| Overall | new_person_lora_memory | 200 | 8000 | 3.3312 | 1.8442 | 2.5877 |
| MIMIC | new_person_lora_memory | 100 | 4000 | 3.7162 | 2.0353 | 2.8757 |
| VitalDB | new_person_lora_memory | 100 | 4000 | 2.9463 | 1.6531 | 2.2997 |

- [200-person complete diagnostic tables and nine matched controls](../results/post_enrollment_200_v1_20260910/enrollment/RESULT_TABLES.md)
- [Earlier 30-person enrollment tables](../results/post_enrollment_30_v1_20260909/enrollment/RESULT_TABLES.md)

## 6. What has and has not been completed

| Question | Evidence status |
| --- | --- |
| Seen users, direct PPG prediction with several backbones | Completed: seven direct backbones in two window-split modes |
| Seen users, personal modules such as LoRA/FiLM/attention/MoE | Completed: 19 single-component candidates plus later combination screens |
| Unseen people, no personal calibration, multiple backbones | Completed: Population rows in rounds 11A–13 |
| Unseen people, K=1/2/3/5 reference events | Completed historical few-shot track; separate results |
| Unseen people, substantial individual enrollment | Completed 30- and 200-person batches; full-cohort validation and final evaluation complete |
| All eligible raw windows, original full-cohort outer split, broad calibration-free backbone comparison | Not established by the earlier event-based screens; no such new sweep was launched here |
| Current full-cohort final test | Completed on 762 people; see the linked final report above |

MIMIC and VitalDB are source strata within PulseDB, not independent external validation datasets. No new candidate, resubmission, test-driven change or GitHub push was performed for the original status/history request. The later publication update preserves the frozen test procedure and the completed results.

## Verification scope

This index was generated from saved aggregate CSVs and checked training/evaluation code. Every setting's MIMIC and VitalDB participant/query counts sum to its saved Overall counts. No historical checkpoint was retrained or raw prediction table rescored here. Current validation CSV/report/receipt files were copied from the completed server scorer and verified against its work and NAS copies. No raw PPG, per-user profile or final test target was downloaded.
