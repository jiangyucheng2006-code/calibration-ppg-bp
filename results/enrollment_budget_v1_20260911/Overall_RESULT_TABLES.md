# Overall: all enrollment-budget results

[Main report](README.md) · [Primary CSV](test/Overall_all_budgets_participant_macro.csv) · [Diagnostic CSV](test/Overall_all_budgets_diagnostics.csv)

All eight budgets and both methods use the same people and query windows.

## Primary participant-macro MAE (mmHg)

| Budget | Setting | SBP MAE | DBP MAE | Mean MAE |
| --- | --- | --- | --- | --- |
| 20% | LoRA | 5.2789 | 2.9869 | 4.1329 |
| 20% | LoRA + memory | 4.5797 | 2.5934 | 3.5865 |
| 30% | LoRA | 4.8066 | 2.7485 | 3.7775 |
| 30% | LoRA + memory | 4.1265 | 2.3816 | 3.2541 |
| 40% | LoRA | 4.5747 | 2.5739 | 3.5743 |
| 40% | LoRA + memory | 3.9305 | 2.2189 | 3.0747 |
| 50% | LoRA | 4.3334 | 2.4396 | 3.3865 |
| 50% | LoRA + memory | 3.6620 | 2.0886 | 2.8753 |
| 60% | LoRA | 4.1006 | 2.3282 | 3.2144 |
| 60% | LoRA + memory | 3.3956 | 1.9613 | 2.6785 |
| 70% | LoRA | 3.9677 | 2.2341 | 3.1009 |
| 70% | LoRA + memory | 3.2850 | 1.8817 | 2.5833 |
| 80% | LoRA | 3.9741 | 2.2235 | 3.0988 |
| 80% | LoRA + memory | 3.2732 | 1.8709 | 2.5721 |
| 90% | LoRA | 3.8556 | 2.1661 | 3.0108 |
| 90% | LoRA + memory | 3.2081 | 1.8336 | 2.5209 |

## Requested full table: pooled-window diagnostics

These MAEs differ from the primary values because every window, rather than every person, receives equal weight. STD is the sample standard deviation of signed prediction-minus-reference errors (ddof=1), not the between-person SD of MAE. The three threshold columns are percentages.

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA 20% | SBP | 4.2404 | 0.9126 | 0.0424 | 6.4313 | 71.4457 | 91.2484 | 96.7368 | PASS* | PASS (Grade A)* |
| LoRA 20% | DBP | 2.3609 | 0.8843 | -0.1798 | 4.3086 | 89.8526 | 97.7747 | 99.0899 | PASS* | PASS (Grade A)* |
| LoRA + memory 20% | SBP | 3.4034 | 0.9364 | -0.1017 | 5.4882 | 79.5889 | 93.9300 | 97.6686 | PASS* | PASS (Grade A)* |
| LoRA + memory 20% | DBP | 1.8711 | 0.9156 | -0.1626 | 3.6791 | 92.7387 | 98.2681 | 99.3098 | PASS* | PASS (Grade A)* |
| LoRA 30% | SBP | 3.9396 | 0.9232 | -0.0009 | 6.0286 | 74.2488 | 92.6084 | 97.2571 | PASS* | PASS (Grade A)* |
| LoRA 30% | DBP | 2.1781 | 0.8995 | -0.1544 | 4.0168 | 91.3545 | 98.1275 | 99.2152 | PASS* | PASS (Grade A)* |
| LoRA + memory 30% | SBP | 3.1708 | 0.9437 | -0.1029 | 5.1603 | 81.6877 | 94.7288 | 98.0150 | PASS* | PASS (Grade A)* |
| LoRA + memory 30% | DBP | 1.7475 | 0.9237 | -0.1334 | 3.4990 | 93.6245 | 98.4892 | 99.3596 | PASS* | PASS (Grade A)* |
| LoRA 40% | SBP | 3.7975 | 0.9280 | 0.0111 | 5.8389 | 75.7084 | 93.2117 | 97.5446 | PASS* | PASS (Grade A)* |
| LoRA 40% | DBP | 2.1022 | 0.9038 | -0.1659 | 3.9293 | 91.9654 | 98.2208 | 99.2382 | PASS* | PASS (Grade A)* |
| LoRA + memory 40% | SBP | 3.0392 | 0.9475 | -0.0822 | 4.9852 | 82.9045 | 95.1736 | 98.1876 | PASS* | PASS (Grade A)* |
| LoRA + memory 40% | DBP | 1.6780 | 0.9276 | -0.1226 | 3.4098 | 94.0655 | 98.5723 | 99.3916 | PASS* | PASS (Grade A)* |
| LoRA 50% | SBP | 3.6845 | 0.9322 | 0.0312 | 5.6671 | 76.7016 | 93.5427 | 97.7210 | PASS* | PASS (Grade A)* |
| LoRA 50% | DBP | 2.0346 | 0.9081 | -0.1492 | 3.8403 | 92.5521 | 98.4048 | 99.2868 | PASS* | PASS (Grade A)* |
| LoRA + memory 50% | SBP | 2.9402 | 0.9506 | -0.0694 | 4.8360 | 83.6382 | 95.4945 | 98.3141 | PASS* | PASS (Grade A)* |
| LoRA + memory 50% | DBP | 1.6265 | 0.9305 | -0.1149 | 3.3412 | 94.4732 | 98.6848 | 99.4235 | PASS* | PASS (Grade A)* |
| LoRA 60% | SBP | 3.5953 | 0.9354 | 0.0114 | 5.5313 | 77.5145 | 93.9786 | 97.8591 | PASS* | PASS (Grade A)* |
| LoRA 60% | DBP | 1.9874 | 0.9113 | -0.1390 | 3.7740 | 92.8742 | 98.4803 | 99.3200 | PASS* | PASS (Grade A)* |
| LoRA + memory 60% | SBP | 2.8641 | 0.9532 | -0.0695 | 4.7084 | 84.2415 | 95.7910 | 98.4394 | PASS* | PASS (Grade A)* |
| LoRA + memory 60% | DBP | 1.5847 | 0.9335 | -0.1101 | 3.2674 | 94.6688 | 98.7576 | 99.4708 | PASS* | PASS (Grade A)* |
| LoRA 70% | SBP | 3.5149 | 0.9378 | 0.0257 | 5.4264 | 78.3261 | 94.2214 | 98.0124 | PASS* | PASS (Grade A)* |
| LoRA 70% | DBP | 1.9427 | 0.9131 | -0.1326 | 3.7340 | 93.2244 | 98.5442 | 99.3379 | PASS* | PASS (Grade A)* |
| LoRA + memory 70% | SBP | 2.8078 | 0.9547 | -0.0591 | 4.6319 | 84.7361 | 95.9137 | 98.5045 | PASS* | PASS (Grade A)* |
| LoRA + memory 70% | DBP | 1.5552 | 0.9345 | -0.1057 | 3.2419 | 94.9359 | 98.8011 | 99.4747 | PASS* | PASS (Grade A)* |
| LoRA 80% | SBP | 3.4707 | 0.9392 | 0.0040 | 5.3656 | 78.6993 | 94.4809 | 98.0482 | PASS* | PASS (Grade A)* |
| LoRA 80% | DBP | 1.9141 | 0.9155 | -0.1332 | 3.6829 | 93.4379 | 98.5953 | 99.3609 | PASS* | PASS (Grade A)* |
| LoRA + memory 80% | SBP | 2.7597 | 0.9560 | -0.0622 | 4.5659 | 85.2205 | 96.0735 | 98.5710 | PASS* | PASS (Grade A)* |
| LoRA + memory 80% | DBP | 1.5268 | 0.9367 | -0.1029 | 3.1871 | 95.1059 | 98.8496 | 99.4760 | PASS* | PASS (Grade A)* |
| LoRA 90% | SBP | 3.4251 | 0.9409 | 0.0245 | 5.2910 | 79.2771 | 94.6675 | 98.1262 | PASS* | PASS (Grade A)* |
| LoRA 90% | DBP | 1.8949 | 0.9170 | -0.1212 | 3.6509 | 93.6168 | 98.5697 | 99.3724 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | SBP | 2.7235 | 0.9570 | -0.0559 | 4.5117 | 85.6104 | 96.2294 | 98.5812 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | DBP | 1.5091 | 0.9377 | -0.0970 | 3.1626 | 95.2401 | 98.8305 | 99.4964 | PASS* | PASS (Grade A)* |

\* Retrospective numerical screens only: AAMI-style |ME| <= 5 mmHg and error STD <= 8 mmHg; BHS-style cumulative percentage grades implemented in [calbased_metrics.py](../../src/pulsedb_fewshot/calbased_metrics.py). PASS is not clinical validation, device certification, or validation of a wrist monitor.
