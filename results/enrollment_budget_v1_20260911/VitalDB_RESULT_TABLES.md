# VitalDB: all enrollment-budget results

[Main report](README.md) · [Primary CSV](test/VitalDB_all_budgets_participant_macro.csv) · [Diagnostic CSV](test/VitalDB_all_budgets_diagnostics.csv)

All eight budgets and both methods use the same people and query windows.

## Primary participant-macro MAE (mmHg)

| Budget | Setting | SBP MAE | DBP MAE | Mean MAE |
| --- | --- | --- | --- | --- |
| 20% | LoRA | 5.6044 | 3.1077 | 4.3561 |
| 20% | LoRA + memory | 4.8352 | 2.6891 | 3.7621 |
| 30% | LoRA | 5.0830 | 2.8894 | 3.9862 |
| 30% | LoRA + memory | 4.3077 | 2.4646 | 3.3862 |
| 40% | LoRA | 4.7885 | 2.6958 | 3.7421 |
| 40% | LoRA + memory | 4.0666 | 2.3134 | 3.1900 |
| 50% | LoRA | 4.5382 | 2.5650 | 3.5516 |
| 50% | LoRA + memory | 3.7664 | 2.1616 | 2.9640 |
| 60% | LoRA | 4.1825 | 2.3612 | 3.2718 |
| 60% | LoRA + memory | 3.4253 | 1.9692 | 2.6972 |
| 70% | LoRA | 3.9962 | 2.2660 | 3.1311 |
| 70% | LoRA + memory | 3.2792 | 1.8983 | 2.5887 |
| 80% | LoRA | 4.0854 | 2.2888 | 3.1871 |
| 80% | LoRA + memory | 3.3204 | 1.9146 | 2.6175 |
| 90% | LoRA | 3.9053 | 2.2244 | 3.0649 |
| 90% | LoRA + memory | 3.1884 | 1.8695 | 2.5290 |

## Requested full table: pooled-window diagnostics

These MAEs differ from the primary values because every window, rather than every person, receives equal weight. STD is the sample standard deviation of signed prediction-minus-reference errors (ddof=1), not the between-person SD of MAE. The three threshold columns are percentages.

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA 20% | SBP | 4.4795 | 0.8788 | -0.0770 | 6.6487 | 68.7354 | 89.9716 | 96.5851 | PASS* | PASS (Grade A)* |
| LoRA 20% | DBP | 2.4813 | 0.8805 | -0.0309 | 4.0883 | 88.0202 | 98.0189 | 99.4909 | PASS* | PASS (Grade A)* |
| LoRA + memory 20% | SBP | 3.5177 | 0.9136 | -0.1912 | 5.6104 | 78.2039 | 93.4756 | 97.8535 | PASS* | PASS (Grade A)* |
| LoRA + memory 20% | DBP | 1.9543 | 0.9056 | -0.1336 | 3.6321 | 91.9908 | 98.6807 | 99.6055 | PASS* | PASS (Grade A)* |
| LoRA 30% | SBP | 4.0566 | 0.8970 | -0.0751 | 6.1279 | 72.3837 | 92.2496 | 97.3996 | PASS* | PASS (Grade A)* |
| LoRA 30% | DBP | 2.2503 | 0.8950 | -0.0396 | 3.8330 | 90.3491 | 98.4262 | 99.5885 | PASS* | PASS (Grade A)* |
| LoRA + memory 30% | SBP | 3.1792 | 0.9275 | -0.1537 | 5.1408 | 81.0504 | 94.9476 | 98.3031 | PASS* | PASS (Grade A)* |
| LoRA + memory 30% | DBP | 1.7822 | 0.9158 | -0.1115 | 3.4296 | 93.5053 | 98.9395 | 99.6479 | PASS* | PASS (Grade A)* |
| LoRA 40% | SBP | 3.8296 | 0.9082 | -0.0684 | 5.7875 | 74.6193 | 93.0853 | 97.9256 | PASS* | PASS (Grade A)* |
| LoRA 40% | DBP | 2.1195 | 0.9038 | -0.0566 | 3.6674 | 91.4436 | 98.7740 | 99.6988 | PASS* | PASS (Grade A)* |
| LoRA + memory 40% | SBP | 2.9792 | 0.9347 | -0.1329 | 4.8791 | 83.0526 | 95.4779 | 98.5153 | PASS* | PASS (Grade A)* |
| LoRA + memory 40% | DBP | 1.6735 | 0.9220 | -0.1010 | 3.3005 | 94.2264 | 99.0625 | 99.7200 | PASS* | PASS (Grade A)* |
| LoRA 50% | SBP | 3.6910 | 0.9149 | -0.0733 | 5.5722 | 75.8028 | 93.6750 | 98.0783 | PASS* | PASS (Grade A)* |
| LoRA 50% | DBP | 2.0423 | 0.9090 | -0.0580 | 3.5669 | 92.1478 | 98.9437 | 99.7412 | PASS* | PASS (Grade A)* |
| LoRA + memory 50% | SBP | 2.8365 | 0.9405 | -0.1237 | 4.6563 | 84.0920 | 95.9700 | 98.7104 | PASS* | PASS (Grade A)* |
| LoRA + memory 50% | DBP | 1.6021 | 0.9268 | -0.0987 | 3.1985 | 94.8797 | 99.2194 | 99.7709 | PASS* | PASS (Grade A)* |
| LoRA 60% | SBP | 3.5699 | 0.9209 | -0.0817 | 5.3705 | 76.9821 | 94.2816 | 98.2819 | PASS* | PASS (Grade A)* |
| LoRA 60% | DBP | 1.9813 | 0.9130 | -0.0562 | 3.4885 | 92.6356 | 99.0455 | 99.7794 | PASS* | PASS (Grade A)* |
| LoRA + memory 60% | SBP | 2.7485 | 0.9440 | -0.1253 | 4.5161 | 84.9192 | 96.2796 | 98.8419 | PASS* | PASS (Grade A)* |
| LoRA + memory 60% | DBP | 1.5468 | 0.9302 | -0.1025 | 3.1220 | 95.1555 | 99.2958 | 99.8218 | PASS* | PASS (Grade A)* |
| LoRA 70% | SBP | 3.4506 | 0.9259 | -0.0709 | 5.1969 | 78.1996 | 94.7610 | 98.4856 | PASS* | PASS (Grade A)* |
| LoRA 70% | DBP | 1.9131 | 0.9174 | -0.0550 | 3.3993 | 93.2889 | 99.1134 | 99.8091 | PASS* | PASS (Grade A)* |
| LoRA + memory 70% | SBP | 2.6709 | 0.9468 | -0.1151 | 4.4043 | 85.5640 | 96.5130 | 98.8801 | PASS* | PASS (Grade A)* |
| LoRA + memory 70% | DBP | 1.5047 | 0.9322 | -0.0981 | 3.0791 | 95.5542 | 99.3637 | 99.8261 | PASS* | PASS (Grade A)* |
| LoRA 80% | SBP | 3.4011 | 0.9271 | -0.0976 | 5.1553 | 78.7808 | 95.1003 | 98.4940 | PASS* | PASS (Grade A)* |
| LoRA 80% | DBP | 1.8754 | 0.9192 | -0.0585 | 3.3618 | 93.5774 | 99.1685 | 99.8091 | PASS* | PASS (Grade A)* |
| LoRA + memory 80% | SBP | 2.6116 | 0.9486 | -0.1179 | 4.3262 | 86.1537 | 96.6826 | 98.9437 | PASS* | PASS (Grade A)* |
| LoRA + memory 80% | DBP | 1.4717 | 0.9337 | -0.0960 | 3.0434 | 95.7621 | 99.3849 | 99.8176 | PASS* | PASS (Grade A)* |
| LoRA 90% | SBP | 3.3107 | 0.9300 | -0.0955 | 5.0510 | 79.8965 | 95.3888 | 98.5916 | PASS* | PASS (Grade A)* |
| LoRA 90% | DBP | 1.8309 | 0.9215 | -0.0706 | 3.3134 | 93.9083 | 99.1898 | 99.8218 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | SBP | 2.5535 | 0.9500 | -0.1126 | 4.2694 | 86.7221 | 96.9754 | 98.9692 | PASS* | PASS (Grade A)* |
| LoRA + memory 90% | DBP | 1.4455 | 0.9351 | -0.0977 | 3.0120 | 95.9869 | 99.3213 | 99.8176 | PASS* | PASS (Grade A)* |

\* Retrospective numerical screens only: AAMI-style |ME| <= 5 mmHg and error STD <= 8 mmHg; BHS-style cumulative percentage grades implemented in [calbased_metrics.py](../../src/pulsedb_fewshot/calbased_metrics.py). PASS is not clinical validation, device certification, or validation of a wrist monitor.
