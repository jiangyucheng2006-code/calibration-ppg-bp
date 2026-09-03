# Round-11A PPG-backbone internal screen

All models use folds 0--2 for fitting, fold 3 for patience-8 early stopping, and fold 4 for candidate ranking. Meta-validation and the locked meta-test are not accessed.

## Result and decision

Jobs 1033--1048 all completed with exit code `0:0`. The 16 active result
directories are byte-identical to their NAS archives. All non-Transformer
stderr files are empty; the three patch-Transformer jobs contain only the
PyTorch notice that the nested-tensor optimization is disabled when
`norm_first=True`. This is a performance-path warning, not a numerical or
training failure.

The compact ResNet reference is the best population model and the best
Quality Gate + Huber (QGH) personalized model. No alternative backbone
improves any of the three primary participant-macro scope means:

| Backbone | Overall mean MAE | MIMIC mean MAE | VitalDB mean MAE | Overall change vs compact ResNet |
| --- | ---: | ---: | ---: | ---: |
| compact ResNet | 8.6453 | 9.3631 | 8.0489 | 0.0000 |
| InceptionTime | 8.7227 | 9.4372 | 8.1290 | +0.0774 |
| deeper ResNet | 8.9036 | 9.5944 | 8.3297 | +0.2583 |
| patch Transformer | 8.9132 | 9.4734 | 8.4477 | +0.2679 |
| Conformer | 8.9789 | 9.6823 | 8.3944 | +0.3336 |

Therefore `winner_backbone=resnet_small` and
`passes_internal_gate=false`. **No Round-11A backbone is promoted.** The
result argues against increasing backbone size or replacing the current
encoder with the tested attention-based architectures under this protocol.
It does not establish that every Transformer or Conformer design must fail.

The 8.6453-mmHg compact-ResNet value is from seed 20260825. The Round-10
reference used seed 20260824 and must not be treated as the same paired run;
the 0.1131-mmHg difference between those two reference values is not evidence
that the code or backbone degraded. The valid architecture comparison is the
within-Round-11A comparison above.

MIMIC and VitalDB are internal PulseDB source strata, not independent external
validation datasets. Participant-macro MAE is primary. AAMI/BHS fields below
are retrospective numerical screens and are not device-standard or clinical
compliance claims.

## Participant-macro results

| Backbone | Model | Setting | Scope | N participants | N queries | SBP participant-macro MAE | DBP participant-macro MAE | Mean participant-macro MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet_small | Population | resnet_small \| Population | MIMIC | 285 | 74373 | 15.8451 | 9.6170 | 12.7311 |
| inception_time | Population | inception_time \| Population | MIMIC | 285 | 74373 | 16.1089 | 9.5123 | 12.8106 |
| resnet_deep | Population | resnet_deep \| Population | MIMIC | 285 | 74373 | 16.2693 | 9.5009 | 12.8851 |
| patch_transformer | Population | patch_transformer \| Population | MIMIC | 285 | 74373 | 16.3608 | 9.5478 | 12.9543 |
| conformer | Population | conformer \| Population | MIMIC | 285 | 74373 | 16.5974 | 9.7176 | 13.1575 |
| resnet_small | Population | resnet_small \| Population | Overall | 628 | 96332 | 14.0885 | 8.8877 | 11.4881 |
| inception_time | Population | inception_time \| Population | Overall | 628 | 96332 | 14.1843 | 8.9729 | 11.5786 |
| resnet_deep | Population | resnet_deep \| Population | Overall | 628 | 96332 | 14.3674 | 8.8660 | 11.6167 |
| patch_transformer | Population | patch_transformer \| Population | Overall | 628 | 96332 | 14.3665 | 8.9453 | 11.6559 |
| conformer | Population | conformer \| Population | Overall | 628 | 96332 | 14.4729 | 9.0721 | 11.7725 |
| resnet_small | Population | resnet_small \| Population | VitalDB | 343 | 21959 | 12.6289 | 8.2818 | 10.4553 |
| inception_time | Population | inception_time \| Population | VitalDB | 343 | 21959 | 12.5851 | 8.5246 | 10.5549 |
| resnet_deep | Population | resnet_deep \| Population | VitalDB | 343 | 21959 | 12.7870 | 8.3384 | 10.5627 |
| patch_transformer | Population | patch_transformer \| Population | VitalDB | 343 | 21959 | 12.7095 | 8.4447 | 10.5771 |
| conformer | Population | conformer \| Population | VitalDB | 343 | 21959 | 12.7077 | 8.5357 | 10.6217 |
| resnet_small | QGH | resnet_small \| QGH | MIMIC | 285 | 74373 | 12.0126 | 6.7136 | 9.3631 |
| inception_time | QGH | inception_time \| QGH | MIMIC | 285 | 74373 | 12.0591 | 6.8153 | 9.4372 |
| patch_transformer | QGH | patch_transformer \| QGH | MIMIC | 285 | 74373 | 12.1640 | 6.7827 | 9.4734 |
| resnet_deep | QGH | resnet_deep \| QGH | MIMIC | 285 | 74373 | 12.2990 | 6.8897 | 9.5944 |
| conformer | QGH | conformer \| QGH | MIMIC | 285 | 74373 | 12.4912 | 6.8734 | 9.6823 |
| resnet_small | QGH | resnet_small \| QGH | Overall | 628 | 96332 | 11.0136 | 6.2771 | 8.6453 |
| inception_time | QGH | inception_time \| QGH | Overall | 628 | 96332 | 11.1094 | 6.3360 | 8.7227 |
| resnet_deep | QGH | resnet_deep \| QGH | Overall | 628 | 96332 | 11.3101 | 6.4972 | 8.9036 |
| patch_transformer | QGH | patch_transformer \| QGH | Overall | 628 | 96332 | 11.3376 | 6.4888 | 8.9132 |
| conformer | QGH | conformer \| QGH | Overall | 628 | 96332 | 11.4082 | 6.5496 | 8.9789 |
| resnet_small | QGH | resnet_small \| QGH | VitalDB | 343 | 21959 | 10.1834 | 5.9145 | 8.0489 |
| inception_time | QGH | inception_time \| QGH | VitalDB | 343 | 21959 | 10.3204 | 5.9377 | 8.1290 |
| resnet_deep | QGH | resnet_deep \| QGH | VitalDB | 343 | 21959 | 10.4884 | 6.1710 | 8.3297 |
| conformer | QGH | conformer \| QGH | VitalDB | 343 | 21959 | 10.5084 | 6.2805 | 8.3944 |
| patch_transformer | QGH | patch_transformer \| QGH | VitalDB | 343 | 21959 | 10.6509 | 6.2445 | 8.4477 |

## QGH change versus the current ResNet reference

Negative candidate-minus-reference values are better.

| Backbone | Scope | Reference mean participant-macro MAE | Candidate mean participant-macro MAE | Candidate minus reference |
| --- | --- | --- | --- | --- |
| resnet_small | Overall | 8.6453 | 8.6453 | 0.0000 |
| resnet_small | MIMIC | 9.3631 | 9.3631 | 0.0000 |
| resnet_small | VitalDB | 8.0489 | 8.0489 | 0.0000 |
| resnet_deep | Overall | 8.6453 | 8.9036 | 0.2583 |
| resnet_deep | MIMIC | 9.3631 | 9.5944 | 0.2313 |
| resnet_deep | VitalDB | 8.0489 | 8.3297 | 0.2808 |
| inception_time | Overall | 8.6453 | 8.7227 | 0.0774 |
| inception_time | MIMIC | 9.3631 | 9.4372 | 0.0741 |
| inception_time | VitalDB | 8.0489 | 8.1290 | 0.0801 |
| patch_transformer | Overall | 8.6453 | 8.9132 | 0.2679 |
| patch_transformer | MIMIC | 9.3631 | 9.4734 | 0.1103 |
| patch_transformer | VitalDB | 8.0489 | 8.4477 | 0.3988 |
| conformer | Overall | 8.6453 | 8.9789 | 0.3336 |
| conformer | MIMIC | 9.3631 | 9.6823 | 0.3192 |
| conformer | VitalDB | 8.0489 | 8.3944 | 0.3455 |

## Model complexity

| Backbone | Population parameters | QGH total parameters | QGH trainable parameters |
| --- | --- | --- | --- |
| resnet_small | 665490 | 1127318 | 461828 |
| resnet_deep | 3827002 | 4288830 | 461828 |
| inception_time | 512162 | 973990 | 461828 |
| patch_transformer | 710530 | 1172358 | 461828 |
| conformer | 1587330 | 2049158 | 461828 |

## Event-pooled diagnostics

Participant-macro MAE is primary. AAMI/BHS entries are retrospective numerical screens only.

| Setting | Scope | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet_small \| Population | Overall | SBP | 15.1377 | 0.2423 | -1.1426 | 19.3940 | 21.8442 | 41.5490 | 58.1790 | FAIL* | FAIL (Grade D)* |
| resnet_small \| Population | Overall | DBP | 8.9842 | 0.1864 | 0.7080 | 11.8032 | 35.7192 | 64.0940 | 82.9901 | FAIL* | FAIL (Grade D)* |
| resnet_small \| QGH | Overall | SBP | 12.3329 | 0.4661 | -0.5681 | 16.2987 | 28.0852 | 51.4429 | 68.9189 | FAIL* | FAIL (Grade D)* |
| resnet_small \| QGH | Overall | DBP | 6.8340 | 0.4733 | -0.6316 | 9.4932 | 48.2280 | 77.5256 | 91.3155 | FAIL* | FAIL (Grade C)* |
| resnet_deep \| Population | Overall | SBP | 15.3223 | 0.2323 | -1.0233 | 19.5280 | 21.0055 | 40.4642 | 57.0745 | FAIL* | FAIL (Grade D)* |
| resnet_deep \| Population | Overall | DBP | 8.9478 | 0.1750 | -0.7409 | 11.8841 | 36.0628 | 65.1871 | 83.1229 | FAIL* | FAIL (Grade D)* |
| resnet_deep \| QGH | Overall | SBP | 12.6870 | 0.4335 | -1.1942 | 16.7561 | 27.5101 | 50.7235 | 67.7283 | FAIL* | FAIL (Grade D)* |
| resnet_deep \| QGH | Overall | DBP | 7.0598 | 0.4375 | -0.9037 | 9.7903 | 47.1287 | 76.5312 | 90.1590 | FAIL* | FAIL (Grade C)* |
| inception_time \| Population | Overall | SBP | 15.5056 | 0.2042 | -1.9093 | 19.8179 | 21.4083 | 40.5504 | 57.2292 | FAIL* | FAIL (Grade D)* |
| inception_time \| Population | Overall | DBP | 9.0410 | 0.1580 | -0.2383 | 12.0269 | 35.8894 | 64.9576 | 82.3818 | FAIL* | FAIL (Grade D)* |
| inception_time \| QGH | Overall | SBP | 12.4952 | 0.4544 | 0.2406 | 16.4834 | 27.6596 | 51.0256 | 68.5255 | FAIL* | FAIL (Grade D)* |
| inception_time \| QGH | Overall | DBP | 6.8954 | 0.4547 | -0.7740 | 9.6494 | 48.2851 | 77.8495 | 90.7196 | FAIL* | FAIL (Grade C)* |
| patch_transformer \| Population | Overall | SBP | 15.5378 | 0.2072 | -2.9794 | 19.6480 | 20.8809 | 40.2182 | 56.8970 | FAIL* | FAIL (Grade D)* |
| patch_transformer \| Population | Overall | DBP | 9.0886 | 0.1777 | 0.2912 | 11.8839 | 34.4465 | 63.6476 | 82.6828 | FAIL* | FAIL (Grade D)* |
| patch_transformer \| QGH | Overall | SBP | 12.6273 | 0.4434 | -1.2157 | 16.6072 | 27.1364 | 50.3353 | 67.7636 | FAIL* | FAIL (Grade D)* |
| patch_transformer \| QGH | Overall | DBP | 7.0197 | 0.4432 | -1.3768 | 9.6851 | 47.1837 | 76.6900 | 90.5836 | FAIL* | FAIL (Grade C)* |
| conformer \| Population | Overall | SBP | 15.6900 | 0.1910 | -2.2600 | 19.9472 | 20.7854 | 40.0615 | 56.5523 | FAIL* | FAIL (Grade D)* |
| conformer \| Population | Overall | DBP | 9.2047 | 0.1513 | -0.8639 | 12.0462 | 34.8067 | 63.0901 | 81.6063 | FAIL* | FAIL (Grade D)* |
| conformer \| QGH | Overall | SBP | 12.8580 | 0.4257 | -1.2536 | 16.8679 | 26.7720 | 49.7633 | 67.0224 | FAIL* | FAIL (Grade D)* |
| conformer \| QGH | Overall | DBP | 7.1044 | 0.4378 | -1.2723 | 9.7463 | 46.1685 | 76.2810 | 90.3448 | FAIL* | FAIL (Grade C)* |
| resnet_small \| Population | MIMIC | SBP | 15.8991 | 0.2135 | -1.7099 | 20.2059 | 20.6876 | 39.4928 | 55.6533 | FAIL* | FAIL (Grade D)* |
| resnet_small \| Population | MIMIC | DBP | 9.1998 | 0.1532 | 0.8420 | 12.1000 | 35.3300 | 63.1627 | 82.0983 | FAIL* | FAIL (Grade D)* |
| resnet_small \| QGH | MIMIC | SBP | 12.9356 | 0.4478 | -1.1794 | 16.9504 | 26.8525 | 49.1415 | 66.5430 | FAIL* | FAIL (Grade D)* |
| resnet_small \| QGH | MIMIC | DBP | 7.0929 | 0.4435 | -0.7656 | 9.8033 | 46.7602 | 75.9886 | 90.4226 | FAIL* | FAIL (Grade C)* |
| resnet_deep \| Population | MIMIC | SBP | 16.0835 | 0.2047 | -1.7799 | 20.3142 | 19.7209 | 38.2827 | 54.5319 | FAIL* | FAIL (Grade D)* |
| resnet_deep \| Population | MIMIC | DBP | 9.1335 | 0.1423 | -0.7896 | 12.1819 | 35.9458 | 64.6471 | 82.2987 | FAIL* | FAIL (Grade D)* |
| resnet_deep \| QGH | MIMIC | SBP | 13.3208 | 0.4132 | -1.6213 | 17.4411 | 26.2609 | 48.5835 | 65.2616 | FAIL* | FAIL (Grade D)* |
| resnet_deep \| QGH | MIMIC | DBP | 7.3221 | 0.4058 | -1.0791 | 10.1031 | 45.8675 | 75.2020 | 89.1130 | FAIL* | FAIL (Grade C)* |
| inception_time \| Population | MIMIC | SBP | 16.3734 | 0.1678 | -2.4599 | 20.7137 | 20.0705 | 38.1267 | 54.3961 | FAIL* | FAIL (Grade D)* |
| inception_time \| Population | MIMIC | DBP | 9.2185 | 0.1254 | -0.3347 | 12.3222 | 35.9122 | 64.4011 | 81.4987 | FAIL* | FAIL (Grade D)* |
| inception_time \| QGH | MIMIC | SBP | 13.1079 | 0.4336 | -0.2448 | 17.2064 | 26.5042 | 49.1509 | 66.3050 | FAIL* | FAIL (Grade D)* |
| inception_time \| QGH | MIMIC | DBP | 7.1609 | 0.4198 | -0.9326 | 9.9969 | 47.1219 | 76.4901 | 89.6575 | FAIL* | FAIL (Grade C)* |
| patch_transformer \| Population | MIMIC | SBP | 16.3944 | 0.1733 | -3.9136 | 20.4189 | 19.4425 | 37.7691 | 53.9941 | FAIL* | FAIL (Grade D)* |
| patch_transformer \| Population | MIMIC | DBP | 9.2548 | 0.1530 | 0.4126 | 12.1238 | 34.1979 | 62.9933 | 82.0150 | FAIL* | FAIL (Grade D)* |
| patch_transformer \| QGH | MIMIC | SBP | 13.2221 | 0.4274 | -1.6352 | 17.2251 | 25.7862 | 48.1035 | 65.3786 | FAIL* | FAIL (Grade D)* |
| patch_transformer \| QGH | MIMIC | DBP | 7.2478 | 0.4159 | -1.5383 | 9.9555 | 46.0974 | 75.3526 | 89.7960 | FAIL* | FAIL (Grade C)* |
| conformer \| Population | MIMIC | SBP | 16.5736 | 0.1536 | -2.9513 | 20.8289 | 19.3995 | 37.6266 | 53.6122 | FAIL* | FAIL (Grade D)* |
| conformer \| Population | MIMIC | DBP | 9.3860 | 0.1254 | -0.5493 | 12.3148 | 34.3821 | 62.4218 | 80.9393 | FAIL* | FAIL (Grade D)* |
| conformer \| QGH | MIMIC | SBP | 13.5536 | 0.4041 | -1.5404 | 17.5835 | 25.0736 | 47.2403 | 64.2491 | FAIL* | FAIL (Grade D)* |
| conformer \| QGH | MIMIC | DBP | 7.3264 | 0.4133 | -1.2178 | 10.0224 | 44.7299 | 75.2491 | 89.6495 | FAIL* | FAIL (Grade C)* |
| resnet_small \| Population | VitalDB | SBP | 12.5591 | 0.2822 | 0.7791 | 16.2006 | 25.7616 | 48.5131 | 66.7335 | FAIL* | FAIL (Grade D)* |
| resnet_small \| Population | VitalDB | DBP | 8.2541 | 0.2774 | 0.2542 | 10.7249 | 37.0372 | 67.2481 | 86.0103 | FAIL* | FAIL (Grade D)* |
| resnet_small \| QGH | VitalDB | SBP | 10.2916 | 0.4844 | 1.5024 | 13.6642 | 32.2601 | 59.2377 | 76.9662 | FAIL* | FAIL (Grade D)* |
| resnet_small \| QGH | VitalDB | DBP | 5.9572 | 0.5628 | -0.1779 | 8.3422 | 53.1991 | 82.7315 | 94.3395 | FAIL* | PASS (Grade B)* |
| resnet_deep \| Population | VitalDB | SBP | 12.7443 | 0.2656 | 1.5394 | 16.3330 | 25.3563 | 47.8528 | 65.6861 | FAIL* | FAIL (Grade D)* |
| resnet_deep \| Population | VitalDB | DBP | 8.3190 | 0.2637 | -0.5758 | 10.8134 | 36.4589 | 67.0158 | 85.9147 | FAIL* | FAIL (Grade D)* |
| resnet_deep \| QGH | VitalDB | SBP | 10.5402 | 0.4576 | 0.2523 | 14.0968 | 31.7410 | 57.9717 | 76.0827 | FAIL* | FAIL (Grade D)* |
| resnet_deep \| QGH | VitalDB | DBP | 6.1715 | 0.5327 | -0.3098 | 8.6210 | 51.4003 | 81.0328 | 93.7019 | FAIL* | PASS (Grade B)* |
| inception_time \| Population | VitalDB | SBP | 12.5664 | 0.2761 | -0.0447 | 16.2875 | 25.9393 | 48.7591 | 66.8245 | FAIL* | FAIL (Grade D)* |
| inception_time \| Population | VitalDB | DBP | 8.4399 | 0.2455 | 0.0883 | 10.9616 | 35.8122 | 66.8428 | 85.3727 | FAIL* | FAIL (Grade D)* |
| inception_time \| QGH | VitalDB | SBP | 10.4203 | 0.4835 | 1.8847 | 13.6280 | 31.5725 | 57.3751 | 76.0463 | FAIL* | FAIL (Grade D)* |
| inception_time \| QGH | VitalDB | DBP | 5.9963 | 0.5625 | -0.2370 | 8.3439 | 52.2246 | 82.4537 | 94.3167 | FAIL* | PASS (Grade B)* |
| patch_transformer \| Population | VitalDB | SBP | 12.6365 | 0.2673 | 0.1847 | 16.3852 | 25.7525 | 48.5131 | 66.7289 | FAIL* | FAIL (Grade D)* |
| patch_transformer \| Population | VitalDB | DBP | 8.5259 | 0.2370 | -0.1200 | 11.0231 | 35.2885 | 65.8637 | 84.9447 | FAIL* | FAIL (Grade D)* |
| patch_transformer \| QGH | VitalDB | SBP | 10.6130 | 0.4476 | 0.2050 | 14.2263 | 31.7091 | 57.8943 | 75.8413 | FAIL* | FAIL (Grade D)* |
| patch_transformer \| QGH | VitalDB | DBP | 6.2470 | 0.5220 | -0.8300 | 8.6851 | 50.8630 | 81.2195 | 93.2511 | FAIL* | PASS (Grade B)* |
| conformer \| Population | VitalDB | SBP | 12.6971 | 0.2659 | 0.0810 | 16.4025 | 25.4793 | 48.3082 | 66.5103 | FAIL* | FAIL (Grade D)* |
| conformer \| Population | VitalDB | DBP | 8.5906 | 0.2139 | -1.9295 | 11.0218 | 36.2448 | 65.3536 | 83.8654 | FAIL* | FAIL (Grade D)* |
| conformer \| QGH | VitalDB | SBP | 10.5021 | 0.4546 | -0.2823 | 14.1354 | 32.5242 | 58.3087 | 76.4151 | FAIL* | FAIL (Grade D)* |
| conformer \| QGH | VitalDB | DBP | 6.3526 | 0.5066 | -1.4572 | 8.7444 | 51.0406 | 79.7759 | 92.7000 | FAIL* | PASS (Grade B)* |

Internal numerical winner: **resnet_small**.
Internal promotion gate passed: **False**.

## Reproducibility artifacts

- [Participant-macro results](../results/round11a/participant_macro_internal.csv)
- [Complete pooled diagnostics](../results/round11a/pooled_diagnostics_internal.csv)
- [Comparison with the compact ResNet](../results/round11a/comparison_vs_reference_internal.csv)
- [Model complexity](../results/round11a/model_complexity.csv)
- [Selection record](../results/round11a/selection.json)
- [Prospective Round-11 design](ROUND11_SYSTEMATIC_MODEL_REVISION_PLAN.md)
