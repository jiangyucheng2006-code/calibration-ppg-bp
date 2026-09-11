# MIMIC: all enrollment-budget results

[Main report](README.md) · [Primary CSV](test/MIMIC_all_budgets_participant_macro.csv) · [Diagnostic CSV](test/MIMIC_all_budgets_diagnostics.csv)

All eight budgets and both methods use the same people and query windows.

## Primary participant-macro MAE (mmHg)

| Budget | Setting | SBP MAE | DBP MAE | Mean MAE |
| --- | --- | --- | --- | --- |
| 20% | LoRA | 4.8812 | 2.8393 | 3.8603 |
| 20% | LoRA + memory | 4.2675 | 2.4766 | 3.3720 |
| 30% | LoRA | 4.4689 | 2.5764 | 3.5227 |
| 30% | LoRA + memory | 3.9052 | 2.2802 | 3.0927 |
| 40% | LoRA | 4.3137 | 2.4250 | 3.3693 |
| 40% | LoRA + memory | 3.7642 | 2.1036 | 2.9339 |
| 50% | LoRA | 4.0833 | 2.2865 | 3.1849 |
| 50% | LoRA + memory | 3.5346 | 1.9994 | 2.7670 |
| 60% | LoRA | 4.0006 | 2.2878 | 3.1442 |
| 60% | LoRA + memory | 3.3594 | 1.9517 | 2.6556 |
| 70% | LoRA | 3.9330 | 2.1951 | 3.0640 |
| 70% | LoRA + memory | 3.2920 | 1.8614 | 2.5767 |
| 80% | LoRA | 3.8382 | 2.1437 | 2.9909 |
| 80% | LoRA + memory | 3.2155 | 1.8175 | 2.5165 |
| 90% | LoRA | 3.7949 | 2.0948 | 2.9448 |
| 90% | LoRA + memory | 3.2322 | 1.7896 | 2.5109 |

## Requested full table: pooled-window diagnostics

These MAEs differ from the primary values because every window, rather than every person, receives equal weight. STD is the sample standard deviation of signed prediction-minus-reference errors (ddof=1), not the between-person SD of MAE. The three threshold columns are percentages.

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA 20% | SBP | 4.1374 | 0.9185 | 0.0939 | 6.3347 | 72.6145 | 91.7990 | 96.8023 | PASS* | PASS (Grade A)* |
| LoRA 20% | DBP | 2.3089 | 0.8856 | -0.2440 | 4.3986 | 90.6428 | 97.6694 | 98.9170 | PASS* | PASS (Grade A)* |
| LoRA + memory 20% | SBP | 3.3540 | 0.9400 | -0.0631 | 5.4343 | 80.1862 | 94.1259 | 97.5889 | PASS* | PASS (Grade A)* |
| LoRA + memory 20% | DBP | 1.8352 | 0.9192 | -0.1752 | 3.6992 | 93.0612 | 98.0902 | 99.1823 | PASS* | PASS (Grade A)* |
| LoRA 30% | SBP | 3.8891 | 0.9272 | 0.0311 | 5.9850 | 75.0531 | 92.7631 | 97.1956 | PASS* | PASS (Grade A)* |
| LoRA 30% | DBP | 2.1469 | 0.9010 | -0.2040 | 4.0926 | 91.7880 | 97.9987 | 99.0542 | PASS* | PASS (Grade A)* |
| LoRA + memory 30% | SBP | 3.1672 | 0.9457 | -0.0810 | 5.1686 | 81.9625 | 94.6345 | 97.8908 | PASS* | PASS (Grade A)* |
| LoRA + memory 30% | DBP | 1.7326 | 0.9265 | -0.1428 | 3.5285 | 93.6759 | 98.2950 | 99.2353 | PASS* | PASS (Grade A)* |
| LoRA 40% | SBP | 3.7837 | 0.9302 | 0.0454 | 5.8606 | 76.1781 | 93.2661 | 97.3804 | PASS* | PASS (Grade A)* |
| LoRA 40% | DBP | 2.0948 | 0.9037 | -0.2130 | 4.0361 | 92.1905 | 97.9822 | 99.0396 | PASS* | PASS (Grade A)* |
| LoRA + memory 40% | SBP | 3.0652 | 0.9486 | -0.0604 | 5.0302 | 82.8406 | 95.0424 | 98.0462 | PASS* | PASS (Grade A)* |
| LoRA + memory 40% | DBP | 1.6799 | 0.9295 | -0.1319 | 3.4558 | 93.9960 | 98.3609 | 99.2500 | PASS* | PASS (Grade A)* |
| LoRA 50% | SBP | 3.6817 | 0.9338 | 0.0762 | 5.7070 | 77.0891 | 93.4857 | 97.5670 | PASS* | PASS (Grade A)* |
| LoRA 50% | DBP | 2.0313 | 0.9078 | -0.1885 | 3.9518 | 92.7265 | 98.1725 | 99.0908 | PASS* | PASS (Grade A)* |
| LoRA + memory 50% | SBP | 2.9849 | 0.9510 | -0.0459 | 4.9114 | 83.4425 | 95.2894 | 98.1432 | PASS* | PASS (Grade A)* |
| LoRA + memory 50% | DBP | 1.6371 | 0.9317 | -0.1219 | 3.4009 | 94.2979 | 98.4542 | 99.2737 | PASS* | PASS (Grade A)* |
| LoRA 60% | SBP | 3.6062 | 0.9363 | 0.0515 | 5.5987 | 77.7440 | 93.8479 | 97.6767 | PASS* | PASS (Grade A)* |
| LoRA 60% | DBP | 1.9900 | 0.9106 | -0.1748 | 3.8902 | 92.9771 | 98.2365 | 99.1219 | PASS* | PASS (Grade A)* |
| LoRA + memory 60% | SBP | 2.9140 | 0.9534 | -0.0454 | 4.7888 | 83.9492 | 95.5803 | 98.2658 | PASS* | PASS (Grade A)* |
| LoRA + memory 60% | DBP | 1.6010 | 0.9346 | -0.1134 | 3.3281 | 94.4589 | 98.5255 | 99.3195 | PASS* | PASS (Grade A)* |
| LoRA 70% | SBP | 3.5427 | 0.9381 | 0.0673 | 5.5220 | 78.3807 | 93.9887 | 97.8084 | PASS* | PASS (Grade A)* |
| LoRA 70% | DBP | 1.9555 | 0.9116 | -0.1660 | 3.8689 | 93.1966 | 98.2987 | 99.1347 | PASS* | PASS (Grade A)* |
| LoRA + memory 70% | SBP | 2.8668 | 0.9546 | -0.0349 | 4.7264 | 84.3791 | 95.6553 | 98.3426 | PASS* | PASS (Grade A)* |
| LoRA + memory 70% | DBP | 1.5769 | 0.9354 | -0.1090 | 3.3096 | 94.6693 | 98.5585 | 99.3231 | PASS* | PASS (Grade A)* |
| LoRA 80% | SBP | 3.5007 | 0.9396 | 0.0478 | 5.4533 | 78.6642 | 94.2137 | 97.8560 | PASS* | PASS (Grade A)* |
| LoRA 80% | DBP | 1.9307 | 0.9142 | -0.1654 | 3.8126 | 93.3777 | 98.3481 | 99.1676 | PASS* | PASS (Grade A)* |
| LoRA + memory 80% | SBP | 2.8236 | 0.9558 | -0.0382 | 4.6653 | 84.8182 | 95.8108 | 98.4103 | PASS* | PASS (Grade A)* |
| LoRA + memory 80% | DBP | 1.5505 | 0.9378 | -0.1059 | 3.2471 | 94.8229 | 98.6188 | 99.3286 | PASS* | PASS (Grade A)* |
| LoRA 90% | SBP | 3.4744 | 0.9410 | 0.0763 | 5.3904 | 79.0100 | 94.3564 | 97.9255 | PASS* | PASS (Grade A)* |
| LoRA 90% | DBP | 1.9225 | 0.9154 | -0.1430 | 3.7870 | 93.4911 | 98.3024 | 99.1786 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | SBP | 2.7968 | 0.9568 | -0.0314 | 4.6120 | 85.1310 | 95.9077 | 98.4139 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | DBP | 1.5365 | 0.9386 | -0.0967 | 3.2254 | 94.9180 | 98.6188 | 99.3579 | PASS* | PASS (Grade A)* |

\* Retrospective numerical screens only: AAMI-style |ME| <= 5 mmHg and error STD <= 8 mmHg; BHS-style cumulative percentage grades implemented in [calbased_metrics.py](../../src/pulsedb_fewshot/calbased_metrics.py). PASS is not clinical validation, device certification, or validation of a wrist monitor.
