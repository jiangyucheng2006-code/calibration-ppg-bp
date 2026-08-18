# Phase-6 fixed-first single-seed screening results

- Development seed: `20260813`
- Split: `meta_validation` only; locked meta-test was not accessed.
- Reference: `Fixed-first M0`.
- Participant-macro MAE is the primary project metric.
- MIMIC and VitalDB are PulseDB source strata, not independent external validation datasets.
- AAMI/BHS entries are retrospective numerical screens only; formal compliance is not established.
- Worst-30% and remaining-70% rows use observed query error and are oracle diagnostics, not deployable filters.

## Participant-macro primary results

| Setting | Scope | K | N participants | N query events | SBP participant-macro MAE | DBP participant-macro MAE | Mean participant-macro MAE | Paired delta vs reference |
|---|---|---|---|---|---|---|---|---|
| Fixed-first M0 | Overall | 1 | 697 | 103564 | 12.135 | 6.983 | 9.559 | 0.000 |
| Fixed-first M0 | Overall | 2 | 697 | 103564 | 11.671 | 6.707 | 9.189 | 0.000 |
| Fixed-first M0 | Overall | 3 | 697 | 103564 | 11.489 | 6.548 | 9.018 | 0.000 |
| Fixed-first M0 | Overall | 5 | 697 | 103564 | 11.212 | 6.353 | 8.782 | 0.000 |
| Huber | Overall | 1 | 697 | 103564 | 12.065 | 6.816 | 9.440 | -0.118 |
| Huber | Overall | 2 | 697 | 103564 | 11.727 | 6.546 | 9.137 | -0.052 |
| Huber | Overall | 3 | 697 | 103564 | 11.549 | 6.412 | 8.981 | -0.038 |
| Huber | Overall | 5 | 697 | 103564 | 11.268 | 6.243 | 8.756 | -0.027 |
| BP-change sampling | Overall | 1 | 697 | 103564 | 12.401 | 7.219 | 9.810 | 0.251 |
| BP-change sampling | Overall | 2 | 697 | 103564 | 11.966 | 6.889 | 9.428 | 0.239 |
| BP-change sampling | Overall | 3 | 697 | 103564 | 11.735 | 6.701 | 9.218 | 0.199 |
| BP-change sampling | Overall | 5 | 697 | 103564 | 11.404 | 6.491 | 8.948 | 0.165 |
| Median anchor | Overall | 1 | 697 | 103564 | 12.164 | 6.985 | 9.575 | 0.016 |
| Median anchor | Overall | 2 | 697 | 103564 | 11.740 | 6.741 | 9.240 | 0.052 |
| Median anchor | Overall | 3 | 697 | 103564 | 11.677 | 6.664 | 9.171 | 0.152 |
| Median anchor | Overall | 5 | 697 | 103564 | 11.465 | 6.477 | 8.971 | 0.189 |
| PPG quality gate | Overall | 1 | 697 | 103564 | 11.891 | 6.880 | 9.386 | -0.173 |
| PPG quality gate | Overall | 2 | 697 | 103564 | 11.473 | 6.596 | 9.035 | -0.154 |
| PPG quality gate | Overall | 3 | 697 | 103564 | 11.232 | 6.448 | 8.840 | -0.178 |
| PPG quality gate | Overall | 5 | 697 | 103564 | 10.910 | 6.264 | 8.587 | -0.195 |
| Age and sex | Overall | 1 | 697 | 103564 | 12.213 | 6.862 | 9.537 | -0.022 |
| Age and sex | Overall | 2 | 697 | 103564 | 11.722 | 6.621 | 9.172 | -0.017 |
| Age and sex | Overall | 3 | 697 | 103564 | 11.540 | 6.481 | 9.011 | -0.008 |
| Age and sex | Overall | 5 | 697 | 103564 | 11.297 | 6.305 | 8.801 | 0.018 |
| Fixed-first M0 | MIMIC | 1 | 316 | 80874 | 13.039 | 7.328 | 10.184 | 0.000 |
| Fixed-first M0 | MIMIC | 2 | 316 | 80874 | 12.744 | 7.121 | 9.933 | 0.000 |
| Fixed-first M0 | MIMIC | 3 | 316 | 80874 | 12.591 | 7.036 | 9.813 | 0.000 |
| Fixed-first M0 | MIMIC | 5 | 316 | 80874 | 12.355 | 6.911 | 9.633 | 0.000 |
| Huber | MIMIC | 1 | 316 | 80874 | 13.011 | 7.107 | 10.059 | -0.125 |
| Huber | MIMIC | 2 | 316 | 80874 | 12.759 | 6.897 | 9.828 | -0.104 |
| Huber | MIMIC | 3 | 316 | 80874 | 12.605 | 6.812 | 9.709 | -0.104 |
| Huber | MIMIC | 5 | 316 | 80874 | 12.375 | 6.700 | 9.537 | -0.096 |
| BP-change sampling | MIMIC | 1 | 316 | 80874 | 13.466 | 7.627 | 10.547 | 0.363 |
| BP-change sampling | MIMIC | 2 | 316 | 80874 | 13.212 | 7.372 | 10.292 | 0.359 |
| BP-change sampling | MIMIC | 3 | 316 | 80874 | 12.997 | 7.267 | 10.132 | 0.319 |
| BP-change sampling | MIMIC | 5 | 316 | 80874 | 12.699 | 7.123 | 9.911 | 0.278 |
| Median anchor | MIMIC | 1 | 316 | 80874 | 13.063 | 7.313 | 10.188 | 0.004 |
| Median anchor | MIMIC | 2 | 316 | 80874 | 12.784 | 7.125 | 9.955 | 0.022 |
| Median anchor | MIMIC | 3 | 316 | 80874 | 12.716 | 7.044 | 9.880 | 0.067 |
| Median anchor | MIMIC | 5 | 316 | 80874 | 12.477 | 6.987 | 9.732 | 0.099 |
| PPG quality gate | MIMIC | 1 | 316 | 80874 | 12.653 | 7.211 | 9.932 | -0.251 |
| PPG quality gate | MIMIC | 2 | 316 | 80874 | 12.254 | 6.948 | 9.601 | -0.332 |
| PPG quality gate | MIMIC | 3 | 316 | 80874 | 12.026 | 6.849 | 9.437 | -0.376 |
| PPG quality gate | MIMIC | 5 | 316 | 80874 | 11.716 | 6.710 | 9.213 | -0.420 |
| Age and sex | MIMIC | 1 | 316 | 80874 | 13.221 | 7.174 | 10.197 | 0.014 |
| Age and sex | MIMIC | 2 | 316 | 80874 | 12.907 | 7.010 | 9.958 | 0.026 |
| Age and sex | MIMIC | 3 | 316 | 80874 | 12.754 | 6.950 | 9.852 | 0.039 |
| Age and sex | MIMIC | 5 | 316 | 80874 | 12.536 | 6.834 | 9.685 | 0.052 |
| Fixed-first M0 | VitalDB | 1 | 381 | 22690 | 11.385 | 6.696 | 9.041 | 0.000 |
| Fixed-first M0 | VitalDB | 2 | 381 | 22690 | 10.781 | 6.363 | 8.572 | 0.000 |
| Fixed-first M0 | VitalDB | 3 | 381 | 22690 | 10.575 | 6.144 | 8.359 | 0.000 |
| Fixed-first M0 | VitalDB | 5 | 381 | 22690 | 10.263 | 5.891 | 8.077 | 0.000 |
| Huber | VitalDB | 1 | 381 | 22690 | 11.281 | 6.574 | 8.928 | -0.113 |
| Huber | VitalDB | 2 | 381 | 22690 | 10.871 | 6.256 | 8.564 | -0.008 |
| Huber | VitalDB | 3 | 381 | 22690 | 10.673 | 6.081 | 8.377 | 0.017 |
| Huber | VitalDB | 5 | 381 | 22690 | 10.350 | 5.864 | 8.107 | 0.030 |
| BP-change sampling | VitalDB | 1 | 381 | 22690 | 11.518 | 6.880 | 9.199 | 0.158 |
| BP-change sampling | VitalDB | 2 | 381 | 22690 | 10.933 | 6.489 | 8.711 | 0.139 |
| BP-change sampling | VitalDB | 3 | 381 | 22690 | 10.688 | 6.232 | 8.460 | 0.101 |
| BP-change sampling | VitalDB | 5 | 381 | 22690 | 10.330 | 5.966 | 8.148 | 0.071 |
| Median anchor | VitalDB | 1 | 381 | 22690 | 11.419 | 6.713 | 9.066 | 0.025 |
| Median anchor | VitalDB | 2 | 381 | 22690 | 10.874 | 6.422 | 8.648 | 0.076 |
| Median anchor | VitalDB | 3 | 381 | 22690 | 10.814 | 6.349 | 8.582 | 0.223 |
| Median anchor | VitalDB | 5 | 381 | 22690 | 10.626 | 6.055 | 8.340 | 0.263 |
| PPG quality gate | VitalDB | 1 | 381 | 22690 | 11.259 | 6.606 | 8.932 | -0.108 |
| PPG quality gate | VitalDB | 2 | 381 | 22690 | 10.826 | 6.305 | 8.566 | -0.006 |
| PPG quality gate | VitalDB | 3 | 381 | 22690 | 10.573 | 6.117 | 8.345 | -0.014 |
| PPG quality gate | VitalDB | 5 | 381 | 22690 | 10.241 | 5.895 | 8.068 | -0.009 |
| Age and sex | VitalDB | 1 | 381 | 22690 | 11.377 | 6.603 | 8.990 | -0.051 |
| Age and sex | VitalDB | 2 | 381 | 22690 | 10.739 | 6.299 | 8.519 | -0.053 |
| Age and sex | VitalDB | 3 | 381 | 22690 | 10.533 | 6.093 | 8.313 | -0.046 |
| Age and sex | VitalDB | 5 | 381 | 22690 | 10.269 | 5.866 | 8.067 | -0.010 |

## Overall event-pooled diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---|---|---|---|---|---|---|---|---|
| Fixed-first M0 (K=1) | SBP | 12.731 | 0.439 | -2.209 | 16.340 | 26.108 | 49.227 | 66.933 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=1) | DBP | 7.495 | 0.401 | -1.141 | 10.125 | 44.003 | 73.713 | 88.669 | FAIL* | FAIL (Grade C)* |
| Fixed-first M0 (K=2) | SBP | 12.385 | 0.463 | -1.887 | 16.020 | 27.159 | 50.853 | 68.255 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=2) | DBP | 7.322 | 0.419 | -1.076 | 9.977 | 45.324 | 74.960 | 89.043 | FAIL* | FAIL (Grade C)* |
| Fixed-first M0 (K=3) | SBP | 12.267 | 0.473 | -1.951 | 15.865 | 27.377 | 51.164 | 68.729 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=3) | DBP | 7.233 | 0.428 | -1.077 | 9.899 | 46.014 | 75.696 | 89.373 | FAIL* | FAIL (Grade C)* |
| Fixed-first M0 (K=5) | SBP | 12.093 | 0.484 | -2.028 | 15.680 | 28.151 | 51.821 | 69.425 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=5) | DBP | 7.063 | 0.446 | -1.071 | 9.737 | 47.239 | 76.686 | 90.065 | FAIL* | FAIL (Grade C)* |
| Huber (K=1) | SBP | 12.574 | 0.446 | -1.059 | 16.350 | 26.941 | 50.148 | 67.718 | FAIL* | FAIL (Grade D)* |
| Huber (K=1) | DBP | 7.200 | 0.439 | -0.516 | 9.850 | 45.661 | 75.408 | 89.713 | FAIL* | FAIL (Grade C)* |
| Huber (K=2) | SBP | 12.348 | 0.464 | -0.691 | 16.101 | 27.097 | 51.102 | 68.721 | FAIL* | FAIL (Grade D)* |
| Huber (K=2) | DBP | 7.071 | 0.453 | -0.456 | 9.728 | 46.782 | 76.302 | 89.939 | FAIL* | FAIL (Grade C)* |
| Huber (K=3) | SBP | 12.204 | 0.477 | -0.662 | 15.908 | 27.352 | 51.448 | 69.183 | FAIL* | FAIL (Grade D)* |
| Huber (K=3) | DBP | 6.989 | 0.463 | -0.424 | 9.636 | 47.306 | 77.033 | 90.222 | FAIL* | FAIL (Grade C)* |
| Huber (K=5) | SBP | 12.019 | 0.489 | -0.760 | 15.725 | 28.140 | 52.227 | 69.861 | FAIL* | FAIL (Grade D)* |
| Huber (K=5) | DBP | 6.857 | 0.476 | -0.427 | 9.523 | 48.608 | 77.755 | 90.635 | FAIL* | FAIL (Grade C)* |
| BP-change sampling (K=1) | SBP | 12.952 | 0.418 | -1.589 | 16.716 | 25.646 | 48.560 | 66.321 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=1) | DBP | 7.770 | 0.364 | -1.366 | 10.409 | 42.392 | 71.795 | 87.526 | FAIL* | FAIL (Grade C)* |
| BP-change sampling (K=2) | SBP | 12.620 | 0.441 | -1.445 | 16.389 | 26.670 | 49.937 | 67.691 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=2) | DBP | 7.549 | 0.390 | -1.328 | 10.200 | 43.798 | 73.553 | 88.168 | FAIL* | FAIL (Grade C)* |
| BP-change sampling (K=3) | SBP | 12.445 | 0.456 | -1.538 | 16.171 | 26.978 | 50.684 | 68.382 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=3) | DBP | 7.423 | 0.403 | -1.334 | 10.081 | 44.760 | 74.414 | 88.667 | FAIL* | FAIL (Grade C)* |
| BP-change sampling (K=5) | SBP | 12.220 | 0.470 | -1.552 | 15.946 | 27.923 | 51.500 | 69.216 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=5) | DBP | 7.228 | 0.427 | -1.302 | 9.879 | 46.087 | 75.533 | 89.560 | FAIL* | FAIL (Grade C)* |
| Median anchor (K=1) | SBP | 12.697 | 0.441 | -2.238 | 16.308 | 26.262 | 49.406 | 67.168 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=1) | DBP | 7.456 | 0.407 | -0.996 | 10.091 | 44.009 | 73.957 | 88.878 | FAIL* | FAIL (Grade C)* |
| Median anchor (K=2) | SBP | 12.379 | 0.464 | -1.808 | 16.022 | 27.163 | 50.905 | 68.308 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=2) | DBP | 7.304 | 0.423 | -0.874 | 9.964 | 45.269 | 75.103 | 89.213 | FAIL* | FAIL (Grade C)* |
| Median anchor (K=3) | SBP | 12.416 | 0.459 | -1.931 | 16.070 | 27.045 | 50.700 | 68.239 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=3) | DBP | 7.305 | 0.418 | -0.997 | 9.994 | 45.441 | 75.259 | 89.294 | FAIL* | FAIL (Grade C)* |
| Median anchor (K=5) | SBP | 12.210 | 0.475 | -2.031 | 15.819 | 27.676 | 51.575 | 68.995 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=5) | DBP | 7.116 | 0.440 | -0.941 | 9.810 | 47.002 | 76.198 | 89.713 | FAIL* | FAIL (Grade C)* |
| PPG quality gate (K=1) | SBP | 12.398 | 0.465 | -1.439 | 16.042 | 27.066 | 50.682 | 68.131 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=1) | DBP | 7.365 | 0.431 | -0.256 | 9.932 | 43.837 | 74.539 | 89.531 | FAIL* | FAIL (Grade C)* |
| PPG quality gate (K=2) | SBP | 12.037 | 0.493 | -1.111 | 15.638 | 28.008 | 51.923 | 69.495 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=2) | DBP | 7.149 | 0.458 | -0.232 | 9.691 | 45.066 | 75.974 | 90.270 | FAIL* | FAIL (Grade C)* |
| PPG quality gate (K=3) | SBP | 11.934 | 0.502 | -1.148 | 15.495 | 28.011 | 52.253 | 69.947 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=3) | DBP | 7.070 | 0.466 | -0.194 | 9.623 | 45.719 | 76.565 | 90.501 | FAIL* | FAIL (Grade C)* |
| PPG quality gate (K=5) | SBP | 11.745 | 0.513 | -1.232 | 15.309 | 28.735 | 53.080 | 70.681 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=5) | DBP | 6.918 | 0.481 | -0.227 | 9.486 | 47.188 | 77.333 | 90.848 | FAIL* | FAIL (Grade C)* |
| Age and sex (K=1) | SBP | 12.887 | 0.426 | -2.780 | 16.446 | 25.931 | 48.687 | 66.298 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=1) | DBP | 7.274 | 0.428 | -1.149 | 9.890 | 45.389 | 74.994 | 89.817 | FAIL* | FAIL (Grade C)* |
| Age and sex (K=2) | SBP | 12.519 | 0.451 | -2.506 | 16.115 | 26.983 | 50.270 | 67.762 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=2) | DBP | 7.128 | 0.443 | -1.075 | 9.768 | 46.437 | 76.045 | 90.103 | FAIL* | FAIL (Grade C)* |
| Age and sex (K=3) | SBP | 12.397 | 0.461 | -2.535 | 15.957 | 27.283 | 50.608 | 68.214 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=3) | DBP | 7.041 | 0.453 | -1.034 | 9.685 | 47.146 | 76.781 | 90.395 | FAIL* | FAIL (Grade C)* |
| Age and sex (K=5) | SBP | 12.238 | 0.472 | -2.621 | 15.785 | 27.847 | 51.190 | 69.008 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=5) | DBP | 6.900 | 0.467 | -1.024 | 9.561 | 48.501 | 77.476 | 90.785 | FAIL* | FAIL (Grade C)* |

## MIMIC event-pooled diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---|---|---|---|---|---|---|---|---|
| Fixed-first M0 (K=1) | SBP | 13.098 | 0.440 | -2.505 | 16.738 | 25.338 | 47.949 | 65.564 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=1) | DBP | 7.723 | 0.391 | -1.216 | 10.464 | 43.128 | 72.703 | 87.760 | FAIL* | FAIL (Grade C)* |
| Fixed-first M0 (K=2) | SBP | 12.839 | 0.457 | -2.120 | 16.527 | 26.123 | 49.198 | 66.528 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=2) | DBP | 7.600 | 0.401 | -1.161 | 10.383 | 44.252 | 73.535 | 87.819 | FAIL* | FAIL (Grade C)* |
| Fixed-first M0 (K=3) | SBP | 12.743 | 0.466 | -2.210 | 16.377 | 26.144 | 49.327 | 66.877 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=3) | DBP | 7.546 | 0.406 | -1.169 | 10.342 | 44.733 | 74.070 | 88.026 | FAIL* | FAIL (Grade C)* |
| Fixed-first M0 (K=5) | SBP | 12.595 | 0.475 | -2.269 | 16.225 | 26.819 | 49.881 | 67.421 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=5) | DBP | 7.388 | 0.423 | -1.127 | 10.196 | 45.763 | 74.959 | 88.752 | FAIL* | FAIL (Grade C)* |
| Huber (K=1) | SBP | 12.921 | 0.448 | -0.985 | 16.769 | 26.224 | 48.998 | 66.443 | FAIL* | FAIL (Grade D)* |
| Huber (K=1) | DBP | 7.366 | 0.435 | -0.585 | 10.131 | 45.131 | 74.701 | 89.060 | FAIL* | FAIL (Grade C)* |
| Huber (K=2) | SBP | 12.771 | 0.460 | -0.598 | 16.602 | 26.087 | 49.653 | 67.179 | FAIL* | FAIL (Grade D)* |
| Huber (K=2) | DBP | 7.302 | 0.440 | -0.520 | 10.089 | 45.874 | 75.135 | 88.945 | FAIL* | FAIL (Grade C)* |
| Huber (K=3) | SBP | 12.644 | 0.473 | -0.583 | 16.408 | 26.160 | 49.862 | 67.543 | FAIL* | FAIL (Grade D)* |
| Huber (K=3) | DBP | 7.248 | 0.447 | -0.479 | 10.025 | 46.147 | 75.751 | 89.135 | FAIL* | FAIL (Grade C)* |
| Huber (K=5) | SBP | 12.480 | 0.482 | -0.676 | 16.261 | 26.827 | 50.490 | 68.148 | FAIL* | FAIL (Grade D)* |
| Huber (K=5) | DBP | 7.129 | 0.458 | -0.448 | 9.930 | 47.304 | 76.402 | 89.500 | FAIL* | FAIL (Grade C)* |
| BP-change sampling (K=1) | SBP | 13.377 | 0.415 | -1.795 | 17.205 | 24.830 | 47.161 | 64.772 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=1) | DBP | 8.009 | 0.356 | -1.272 | 10.759 | 41.338 | 70.560 | 86.673 | FAIL* | FAIL (Grade C)* |
| BP-change sampling (K=2) | SBP | 13.112 | 0.432 | -1.584 | 16.967 | 25.590 | 48.327 | 66.011 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=2) | DBP | 7.843 | 0.372 | -1.255 | 10.623 | 42.465 | 71.998 | 86.983 | FAIL* | FAIL (Grade C)* |
| BP-change sampling (K=3) | SBP | 12.947 | 0.446 | -1.710 | 16.740 | 25.770 | 48.948 | 66.610 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=3) | DBP | 7.747 | 0.382 | -1.258 | 10.541 | 43.335 | 72.655 | 87.393 | FAIL* | FAIL (Grade C)* |
| BP-change sampling (K=5) | SBP | 12.743 | 0.459 | -1.709 | 16.545 | 26.675 | 49.596 | 67.227 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=5) | DBP | 7.558 | 0.404 | -1.196 | 10.352 | 44.576 | 73.686 | 88.277 | FAIL* | FAIL (Grade C)* |
| Median anchor (K=1) | SBP | 13.054 | 0.442 | -2.548 | 16.693 | 25.542 | 48.208 | 65.927 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=1) | DBP | 7.676 | 0.399 | -1.073 | 10.412 | 43.077 | 72.884 | 88.065 | FAIL* | FAIL (Grade C)* |
| Median anchor (K=2) | SBP | 12.808 | 0.459 | -2.130 | 16.491 | 26.195 | 49.326 | 66.705 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=2) | DBP | 7.564 | 0.408 | -1.007 | 10.340 | 44.252 | 73.784 | 88.114 | FAIL* | FAIL (Grade C)* |
| Median anchor (K=3) | SBP | 12.871 | 0.455 | -2.201 | 16.547 | 25.775 | 48.970 | 66.433 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=3) | DBP | 7.578 | 0.401 | -1.032 | 10.401 | 44.323 | 73.942 | 88.174 | FAIL* | FAIL (Grade C)* |
| Median anchor (K=5) | SBP | 12.628 | 0.472 | -2.246 | 16.274 | 26.644 | 50.077 | 67.400 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=5) | DBP | 7.422 | 0.420 | -0.923 | 10.242 | 45.486 | 74.668 | 88.546 | FAIL* | FAIL (Grade C)* |
| PPG quality gate (K=1) | SBP | 12.732 | 0.466 | -1.831 | 16.413 | 26.460 | 49.644 | 66.801 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=1) | DBP | 7.572 | 0.425 | -0.392 | 10.228 | 42.830 | 73.486 | 88.806 | FAIL* | FAIL (Grade C)* |
| PPG quality gate (K=2) | SBP | 12.402 | 0.491 | -1.429 | 16.071 | 27.338 | 50.801 | 68.078 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=2) | DBP | 7.388 | 0.447 | -0.341 | 10.035 | 43.928 | 74.762 | 89.364 | FAIL* | FAIL (Grade C)* |
| PPG quality gate (K=3) | SBP | 12.327 | 0.499 | -1.506 | 15.939 | 27.068 | 50.927 | 68.371 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=3) | DBP | 7.339 | 0.451 | -0.279 | 10.000 | 44.278 | 75.150 | 89.517 | FAIL* | FAIL (Grade C)* |
| PPG quality gate (K=5) | SBP | 12.163 | 0.508 | -1.606 | 15.774 | 27.702 | 51.552 | 68.957 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=5) | DBP | 7.197 | 0.464 | -0.292 | 9.876 | 45.641 | 75.882 | 89.811 | FAIL* | FAIL (Grade C)* |
| Age and sex (K=1) | SBP | 13.288 | 0.424 | -3.089 | 16.874 | 25.117 | 47.377 | 64.774 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=1) | DBP | 7.470 | 0.421 | -1.214 | 10.201 | 44.648 | 74.142 | 89.134 | FAIL* | FAIL (Grade C)* |
| Age and sex (K=2) | SBP | 13.015 | 0.442 | -2.765 | 16.663 | 25.914 | 48.567 | 65.910 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=2) | DBP | 7.364 | 0.430 | -1.161 | 10.128 | 45.592 | 74.789 | 89.186 | FAIL* | FAIL (Grade C)* |
| Age and sex (K=3) | SBP | 12.914 | 0.451 | -2.810 | 16.510 | 26.090 | 48.662 | 66.156 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=3) | DBP | 7.308 | 0.436 | -1.109 | 10.081 | 46.041 | 75.363 | 89.348 | FAIL* | FAIL (Grade C)* |
| Age and sex (K=5) | SBP | 12.773 | 0.460 | -2.890 | 16.360 | 26.622 | 49.175 | 66.848 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=5) | DBP | 7.181 | 0.448 | -1.060 | 9.976 | 47.220 | 76.006 | 89.677 | FAIL* | FAIL (Grade C)* |

## VitalDB event-pooled diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---|---|---|---|---|---|---|---|---|
| Fixed-first M0 (K=1) | SBP | 11.424 | 0.400 | -1.152 | 14.788 | 28.854 | 53.781 | 71.816 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=1) | DBP | 6.683 | 0.442 | -0.876 | 8.806 | 47.122 | 77.312 | 91.908 | FAIL* | FAIL (Grade C)* |
| Fixed-first M0 (K=2) | SBP | 10.769 | 0.460 | -1.054 | 14.033 | 30.851 | 56.752 | 74.412 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=2) | DBP | 6.332 | 0.497 | -0.771 | 8.366 | 49.145 | 80.040 | 93.407 | FAIL* | FAIL (Grade C)* |
| Fixed-first M0 (K=3) | SBP | 10.569 | 0.474 | -1.030 | 13.849 | 31.772 | 57.708 | 75.333 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=3) | DBP | 6.118 | 0.527 | -0.752 | 8.115 | 50.582 | 81.494 | 94.174 | FAIL* | PASS (Grade B)* |
| Fixed-first M0 (K=5) | SBP | 10.303 | 0.497 | -1.167 | 13.526 | 32.896 | 58.735 | 76.567 | FAIL* | FAIL (Grade D)* |
| Fixed-first M0 (K=5) | DBP | 5.907 | 0.551 | -0.871 | 7.886 | 52.503 | 82.843 | 94.747 | PASS* | PASS (Grade B)* |
| Huber (K=1) | SBP | 11.338 | 0.401 | -1.325 | 14.755 | 29.498 | 54.244 | 72.261 | FAIL* | FAIL (Grade D)* |
| Huber (K=1) | DBP | 6.609 | 0.451 | -0.272 | 8.772 | 47.550 | 77.929 | 92.041 | FAIL* | FAIL (Grade C)* |
| Huber (K=2) | SBP | 10.842 | 0.449 | -1.021 | 14.170 | 30.696 | 56.267 | 74.218 | FAIL* | FAIL (Grade D)* |
| Huber (K=2) | DBP | 6.247 | 0.508 | -0.228 | 8.310 | 50.018 | 80.458 | 93.482 | FAIL* | PASS (Grade B)* |
| Huber (K=3) | SBP | 10.637 | 0.465 | -0.945 | 13.975 | 31.600 | 57.104 | 75.029 | FAIL* | FAIL (Grade D)* |
| Huber (K=3) | DBP | 6.064 | 0.533 | -0.227 | 8.095 | 51.437 | 81.600 | 94.094 | FAIL* | PASS (Grade B)* |
| Huber (K=5) | SBP | 10.376 | 0.489 | -1.061 | 13.641 | 32.821 | 58.418 | 75.967 | FAIL* | FAIL (Grade D)* |
| Huber (K=5) | DBP | 5.884 | 0.554 | -0.351 | 7.902 | 53.253 | 82.578 | 94.680 | PASS* | PASS (Grade B)* |
| BP-change sampling (K=1) | SBP | 11.435 | 0.399 | -0.856 | 14.820 | 28.554 | 53.548 | 71.842 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=1) | DBP | 6.915 | 0.397 | -1.701 | 9.040 | 46.148 | 76.197 | 90.564 | FAIL* | FAIL (Grade C)* |
| BP-change sampling (K=2) | SBP | 10.869 | 0.453 | -0.949 | 14.126 | 30.520 | 55.677 | 73.680 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=2) | DBP | 6.501 | 0.465 | -1.588 | 8.518 | 48.550 | 79.092 | 92.389 | FAIL* | FAIL (Grade C)* |
| BP-change sampling (K=3) | SBP | 10.657 | 0.467 | -0.923 | 13.938 | 31.287 | 56.871 | 74.698 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=3) | DBP | 6.268 | 0.499 | -1.603 | 8.228 | 49.837 | 80.683 | 93.208 | FAIL* | FAIL (Grade C)* |
| BP-change sampling (K=5) | SBP | 10.356 | 0.494 | -0.990 | 13.584 | 32.371 | 58.286 | 76.307 | FAIL* | FAIL (Grade D)* |
| BP-change sampling (K=5) | DBP | 6.049 | 0.529 | -1.682 | 7.952 | 51.476 | 82.115 | 94.134 | PASS* | PASS (Grade B)* |
| Median anchor (K=1) | SBP | 11.427 | 0.399 | -1.136 | 14.799 | 28.828 | 53.676 | 71.591 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=1) | DBP | 6.672 | 0.439 | -0.723 | 8.847 | 47.329 | 77.783 | 91.776 | FAIL* | FAIL (Grade C)* |
| Median anchor (K=2) | SBP | 10.848 | 0.451 | -0.659 | 14.162 | 30.613 | 56.532 | 74.024 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=2) | DBP | 6.375 | 0.487 | -0.400 | 8.473 | 48.894 | 79.806 | 93.134 | FAIL* | FAIL (Grade C)* |
| Median anchor (K=3) | SBP | 10.794 | 0.447 | -0.967 | 14.196 | 31.573 | 56.866 | 74.676 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=3) | DBP | 6.331 | 0.494 | -0.871 | 8.381 | 49.427 | 79.952 | 93.283 | FAIL* | FAIL (Grade C)* |
| Median anchor (K=5) | SBP | 10.722 | 0.456 | -1.268 | 14.055 | 31.353 | 56.915 | 74.680 | FAIL* | FAIL (Grade D)* |
| Median anchor (K=5) | DBP | 6.027 | 0.527 | -1.005 | 8.086 | 52.406 | 81.653 | 93.870 | FAIL* | PASS (Grade B)* |
| PPG quality gate (K=1) | SBP | 11.207 | 0.421 | -0.042 | 14.560 | 29.229 | 54.381 | 72.869 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=1) | DBP | 6.626 | 0.450 | 0.228 | 8.778 | 47.426 | 78.294 | 92.115 | FAIL* | FAIL (Grade C)* |
| PPG quality gate (K=2) | SBP | 10.734 | 0.470 | 0.021 | 13.929 | 30.397 | 55.923 | 74.548 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=2) | DBP | 6.298 | 0.504 | 0.157 | 8.339 | 49.123 | 80.295 | 93.499 | FAIL* | FAIL (Grade C)* |
| PPG quality gate (K=3) | SBP | 10.530 | 0.486 | 0.127 | 13.719 | 31.371 | 56.977 | 75.566 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=3) | DBP | 6.113 | 0.529 | 0.111 | 8.133 | 50.855 | 81.609 | 94.006 | FAIL* | PASS (Grade B)* |
| PPG quality gate (K=5) | SBP | 10.256 | 0.507 | 0.098 | 13.442 | 32.415 | 58.528 | 76.827 | FAIL* | FAIL (Grade D)* |
| PPG quality gate (K=5) | DBP | 5.924 | 0.551 | 0.007 | 7.941 | 52.702 | 82.503 | 94.544 | PASS* | PASS (Grade B)* |
| Age and sex (K=1) | SBP | 11.459 | 0.397 | -1.676 | 14.769 | 28.832 | 53.354 | 71.732 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=1) | DBP | 6.578 | 0.456 | -0.916 | 8.688 | 48.030 | 78.030 | 92.252 | FAIL* | FAIL (Grade C)* |
| Age and sex (K=2) | SBP | 10.754 | 0.462 | -1.584 | 13.949 | 30.793 | 56.342 | 74.363 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=2) | DBP | 6.288 | 0.499 | -0.768 | 8.351 | 49.449 | 80.520 | 93.372 | FAIL* | FAIL (Grade C)* |
| Age and sex (K=3) | SBP | 10.551 | 0.477 | -1.553 | 13.762 | 31.534 | 57.545 | 75.549 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=3) | DBP | 6.089 | 0.527 | -0.767 | 8.115 | 51.084 | 81.833 | 94.130 | FAIL* | PASS (Grade B)* |
| Age and sex (K=5) | SBP | 10.329 | 0.496 | -1.661 | 13.493 | 32.212 | 58.369 | 76.703 | FAIL* | FAIL (Grade D)* |
| Age and sex (K=5) | DBP | 5.898 | 0.549 | -0.896 | 7.906 | 53.067 | 82.715 | 94.738 | PASS* | PASS (Grade B)* |

## Method-specific observed-error tail diagnostics

| Setting | Scope | K | N participants | Worst 30% participants | Worst 30% threshold | Worst 30% mean MAE | Remaining 70% mean MAE | P90 participant mean MAE | P95 participant mean MAE | P99 participant mean MAE | Selection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed-first M0 | Overall | 1 | 697 | 210 | 11.007 | 15.028 | 7.200 | 15.462 | 18.763 | 23.631 | observed query error; oracle diagnostic only |
| Fixed-first M0 | Overall | 2 | 697 | 210 | 10.316 | 14.398 | 6.943 | 15.098 | 17.369 | 22.912 | observed query error; oracle diagnostic only |
| Fixed-first M0 | Overall | 3 | 697 | 210 | 10.238 | 14.153 | 6.804 | 14.844 | 16.943 | 22.165 | observed query error; oracle diagnostic only |
| Fixed-first M0 | Overall | 5 | 697 | 210 | 9.969 | 13.714 | 6.656 | 14.429 | 16.249 | 22.252 | observed query error; oracle diagnostic only |
| Huber | Overall | 1 | 697 | 210 | 10.891 | 14.867 | 7.100 | 15.375 | 17.862 | 23.293 | observed query error; oracle diagnostic only |
| Huber | Overall | 2 | 697 | 210 | 10.462 | 14.331 | 6.897 | 14.825 | 16.892 | 23.052 | observed query error; oracle diagnostic only |
| Huber | Overall | 3 | 697 | 210 | 10.381 | 14.066 | 6.788 | 14.588 | 16.442 | 22.501 | observed query error; oracle diagnostic only |
| Huber | Overall | 5 | 697 | 210 | 9.996 | 13.683 | 6.631 | 14.360 | 16.118 | 22.415 | observed query error; oracle diagnostic only |
| BP-change sampling | Overall | 1 | 697 | 210 | 11.062 | 15.620 | 7.304 | 16.018 | 19.376 | 25.995 | observed query error; oracle diagnostic only |
| BP-change sampling | Overall | 2 | 697 | 210 | 10.764 | 15.018 | 7.017 | 15.469 | 18.114 | 25.031 | observed query error; oracle diagnostic only |
| BP-change sampling | Overall | 3 | 697 | 210 | 10.616 | 14.733 | 6.840 | 15.056 | 17.673 | 24.404 | observed query error; oracle diagnostic only |
| BP-change sampling | Overall | 5 | 697 | 210 | 10.287 | 14.116 | 6.719 | 14.384 | 16.793 | 23.498 | observed query error; oracle diagnostic only |
| Median anchor | Overall | 1 | 697 | 210 | 10.889 | 15.079 | 7.201 | 15.380 | 19.116 | 25.447 | observed query error; oracle diagnostic only |
| Median anchor | Overall | 2 | 697 | 210 | 10.385 | 14.535 | 6.957 | 15.044 | 17.829 | 24.022 | observed query error; oracle diagnostic only |
| Median anchor | Overall | 3 | 697 | 210 | 10.380 | 14.459 | 6.890 | 15.146 | 17.913 | 22.686 | observed query error; oracle diagnostic only |
| Median anchor | Overall | 5 | 697 | 210 | 10.207 | 14.130 | 6.747 | 14.875 | 17.199 | 22.461 | observed query error; oracle diagnostic only |
| PPG quality gate | Overall | 1 | 697 | 210 | 10.687 | 14.609 | 7.133 | 15.085 | 17.461 | 23.115 | observed query error; oracle diagnostic only |
| PPG quality gate | Overall | 2 | 697 | 210 | 10.446 | 13.929 | 6.925 | 14.317 | 16.668 | 21.466 | observed query error; oracle diagnostic only |
| PPG quality gate | Overall | 3 | 697 | 210 | 10.178 | 13.622 | 6.778 | 13.960 | 16.445 | 20.883 | observed query error; oracle diagnostic only |
| PPG quality gate | Overall | 5 | 697 | 210 | 9.827 | 13.163 | 6.614 | 13.430 | 15.707 | 19.744 | observed query error; oracle diagnostic only |
| Age and sex | Overall | 1 | 697 | 210 | 10.984 | 15.005 | 7.179 | 15.189 | 17.975 | 24.271 | observed query error; oracle diagnostic only |
| Age and sex | Overall | 2 | 697 | 210 | 10.504 | 14.424 | 6.907 | 15.068 | 17.315 | 23.186 | observed query error; oracle diagnostic only |
| Age and sex | Overall | 3 | 697 | 210 | 10.307 | 14.198 | 6.774 | 14.825 | 17.313 | 22.588 | observed query error; oracle diagnostic only |
| Age and sex | Overall | 5 | 697 | 210 | 9.939 | 13.806 | 6.643 | 14.668 | 16.579 | 22.499 | observed query error; oracle diagnostic only |
| Fixed-first M0 | MIMIC | 1 | 316 | 95 | 12.103 | 16.165 | 7.612 | 16.098 | 20.435 | 24.354 | observed query error; oracle diagnostic only |
| Fixed-first M0 | MIMIC | 2 | 316 | 95 | 12.107 | 15.827 | 7.399 | 16.175 | 19.483 | 24.479 | observed query error; oracle diagnostic only |
| Fixed-first M0 | MIMIC | 3 | 316 | 95 | 11.647 | 15.693 | 7.285 | 16.057 | 19.202 | 23.926 | observed query error; oracle diagnostic only |
| Fixed-first M0 | MIMIC | 5 | 316 | 95 | 11.347 | 15.364 | 7.170 | 15.558 | 18.710 | 23.698 | observed query error; oracle diagnostic only |
| Huber | MIMIC | 1 | 316 | 95 | 11.827 | 16.011 | 7.500 | 16.330 | 19.375 | 24.491 | observed query error; oracle diagnostic only |
| Huber | MIMIC | 2 | 316 | 95 | 11.662 | 15.604 | 7.346 | 16.100 | 18.031 | 24.361 | observed query error; oracle diagnostic only |
| Huber | MIMIC | 3 | 316 | 95 | 11.428 | 15.423 | 7.253 | 15.535 | 18.403 | 23.371 | observed query error; oracle diagnostic only |
| Huber | MIMIC | 5 | 316 | 95 | 11.068 | 15.162 | 7.119 | 15.560 | 18.211 | 23.050 | observed query error; oracle diagnostic only |
| BP-change sampling | MIMIC | 1 | 316 | 95 | 12.226 | 17.004 | 7.771 | 16.967 | 21.422 | 29.550 | observed query error; oracle diagnostic only |
| BP-change sampling | MIMIC | 2 | 316 | 95 | 12.099 | 16.627 | 7.569 | 16.536 | 20.886 | 26.626 | observed query error; oracle diagnostic only |
| BP-change sampling | MIMIC | 3 | 316 | 95 | 11.941 | 16.451 | 7.415 | 16.490 | 20.193 | 28.538 | observed query error; oracle diagnostic only |
| BP-change sampling | MIMIC | 5 | 316 | 95 | 11.595 | 16.039 | 7.277 | 15.808 | 20.004 | 27.025 | observed query error; oracle diagnostic only |
| Median anchor | MIMIC | 1 | 316 | 95 | 12.002 | 16.204 | 7.602 | 16.291 | 20.064 | 26.466 | observed query error; oracle diagnostic only |
| Median anchor | MIMIC | 2 | 316 | 95 | 11.705 | 15.903 | 7.397 | 16.077 | 19.404 | 24.490 | observed query error; oracle diagnostic only |
| Median anchor | MIMIC | 3 | 316 | 95 | 11.691 | 15.844 | 7.317 | 16.217 | 18.828 | 23.499 | observed query error; oracle diagnostic only |
| Median anchor | MIMIC | 5 | 316 | 95 | 11.545 | 15.628 | 7.198 | 16.380 | 19.560 | 22.836 | observed query error; oracle diagnostic only |
| PPG quality gate | MIMIC | 1 | 316 | 95 | 11.595 | 15.584 | 7.503 | 16.050 | 18.229 | 23.260 | observed query error; oracle diagnostic only |
| PPG quality gate | MIMIC | 2 | 316 | 95 | 11.448 | 14.940 | 7.306 | 15.495 | 17.342 | 21.416 | observed query error; oracle diagnostic only |
| PPG quality gate | MIMIC | 3 | 316 | 95 | 11.202 | 14.659 | 7.193 | 15.112 | 16.708 | 22.210 | observed query error; oracle diagnostic only |
| PPG quality gate | MIMIC | 5 | 316 | 95 | 10.882 | 14.241 | 7.052 | 14.637 | 16.582 | 21.620 | observed query error; oracle diagnostic only |
| Age and sex | MIMIC | 1 | 316 | 95 | 11.936 | 16.278 | 7.583 | 16.060 | 20.292 | 25.312 | observed query error; oracle diagnostic only |
| Age and sex | MIMIC | 2 | 316 | 95 | 11.722 | 15.955 | 7.381 | 16.151 | 19.611 | 25.075 | observed query error; oracle diagnostic only |
| Age and sex | MIMIC | 3 | 316 | 95 | 11.651 | 15.846 | 7.276 | 16.426 | 18.879 | 24.047 | observed query error; oracle diagnostic only |
| Age and sex | MIMIC | 5 | 316 | 95 | 11.302 | 15.544 | 7.167 | 16.210 | 17.983 | 23.939 | observed query error; oracle diagnostic only |
| Fixed-first M0 | VitalDB | 1 | 381 | 115 | 10.114 | 13.886 | 6.946 | 14.370 | 17.208 | 22.434 | observed query error; oracle diagnostic only |
| Fixed-first M0 | VitalDB | 2 | 381 | 115 | 9.646 | 12.976 | 6.668 | 13.297 | 15.885 | 20.848 | observed query error; oracle diagnostic only |
| Fixed-first M0 | VitalDB | 3 | 381 | 115 | 9.503 | 12.662 | 6.499 | 13.329 | 15.375 | 20.233 | observed query error; oracle diagnostic only |
| Fixed-first M0 | VitalDB | 5 | 381 | 115 | 9.244 | 12.110 | 6.334 | 12.597 | 14.926 | 18.165 | observed query error; oracle diagnostic only |
| Huber | VitalDB | 1 | 381 | 115 | 10.227 | 13.786 | 6.827 | 13.896 | 17.273 | 22.141 | observed query error; oracle diagnostic only |
| Huber | VitalDB | 2 | 381 | 115 | 9.742 | 13.053 | 6.623 | 13.639 | 16.076 | 20.198 | observed query error; oracle diagnostic only |
| Huber | VitalDB | 3 | 381 | 115 | 9.541 | 12.729 | 6.495 | 13.177 | 15.456 | 20.734 | observed query error; oracle diagnostic only |
| Huber | VitalDB | 5 | 381 | 115 | 9.356 | 12.224 | 6.328 | 12.593 | 14.755 | 18.980 | observed query error; oracle diagnostic only |
| BP-change sampling | VitalDB | 1 | 381 | 115 | 10.238 | 14.291 | 6.997 | 14.725 | 17.727 | 22.742 | observed query error; oracle diagnostic only |
| BP-change sampling | VitalDB | 2 | 381 | 115 | 9.854 | 13.424 | 6.673 | 13.852 | 16.140 | 20.031 | observed query error; oracle diagnostic only |
| BP-change sampling | VitalDB | 3 | 381 | 115 | 9.599 | 13.015 | 6.490 | 13.481 | 15.232 | 19.660 | observed query error; oracle diagnostic only |
| BP-change sampling | VitalDB | 5 | 381 | 115 | 9.438 | 12.303 | 6.352 | 12.813 | 14.778 | 17.927 | observed query error; oracle diagnostic only |
| Median anchor | VitalDB | 1 | 381 | 115 | 10.098 | 13.967 | 6.947 | 14.119 | 17.032 | 23.261 | observed query error; oracle diagnostic only |
| Median anchor | VitalDB | 2 | 381 | 115 | 9.675 | 13.173 | 6.692 | 13.581 | 16.341 | 20.955 | observed query error; oracle diagnostic only |
| Median anchor | VitalDB | 3 | 381 | 115 | 9.517 | 13.028 | 6.660 | 13.731 | 15.920 | 20.434 | observed query error; oracle diagnostic only |
| Median anchor | VitalDB | 5 | 381 | 115 | 9.484 | 12.656 | 6.474 | 12.888 | 15.807 | 19.912 | observed query error; oracle diagnostic only |
| PPG quality gate | VitalDB | 1 | 381 | 115 | 9.877 | 13.614 | 6.909 | 13.837 | 16.868 | 21.707 | observed query error; oracle diagnostic only |
| PPG quality gate | VitalDB | 2 | 381 | 115 | 9.707 | 12.927 | 6.680 | 13.111 | 15.827 | 20.479 | observed query error; oracle diagnostic only |
| PPG quality gate | VitalDB | 3 | 381 | 115 | 9.238 | 12.570 | 6.518 | 12.705 | 15.236 | 19.986 | observed query error; oracle diagnostic only |
| PPG quality gate | VitalDB | 5 | 381 | 115 | 9.214 | 12.057 | 6.344 | 12.535 | 14.113 | 18.584 | observed query error; oracle diagnostic only |
| Age and sex | VitalDB | 1 | 381 | 115 | 10.092 | 13.784 | 6.917 | 14.057 | 16.512 | 21.972 | observed query error; oracle diagnostic only |
| Age and sex | VitalDB | 2 | 381 | 115 | 9.642 | 12.910 | 6.621 | 13.565 | 15.910 | 20.160 | observed query error; oracle diagnostic only |
| Age and sex | VitalDB | 3 | 381 | 115 | 9.317 | 12.577 | 6.469 | 13.332 | 15.470 | 20.119 | observed query error; oracle diagnostic only |
| Age and sex | VitalDB | 5 | 381 | 115 | 9.208 | 12.117 | 6.316 | 12.644 | 14.907 | 18.031 | observed query error; oracle diagnostic only |

## Performance on the reference model's observed-error worst 30%

| Setting | Scope | K | Reference tail setting | Reference-tail participants | Mean MAE on reference worst 30% | Selection |
|---|---|---|---|---|---|---|
| Fixed-first M0 | Overall | 1 | Fixed-first M0 | 210 | 15.028 | reference observed query error; oracle diagnostic only |
| Fixed-first M0 | Overall | 2 | Fixed-first M0 | 210 | 14.398 | reference observed query error; oracle diagnostic only |
| Fixed-first M0 | Overall | 3 | Fixed-first M0 | 210 | 14.153 | reference observed query error; oracle diagnostic only |
| Fixed-first M0 | Overall | 5 | Fixed-first M0 | 210 | 13.714 | reference observed query error; oracle diagnostic only |
| Huber | Overall | 1 | Fixed-first M0 | 210 | 14.632 | reference observed query error; oracle diagnostic only |
| Huber | Overall | 2 | Fixed-first M0 | 210 | 14.078 | reference observed query error; oracle diagnostic only |
| Huber | Overall | 3 | Fixed-first M0 | 210 | 13.896 | reference observed query error; oracle diagnostic only |
| Huber | Overall | 5 | Fixed-first M0 | 210 | 13.524 | reference observed query error; oracle diagnostic only |
| BP-change sampling | Overall | 1 | Fixed-first M0 | 210 | 15.300 | reference observed query error; oracle diagnostic only |
| BP-change sampling | Overall | 2 | Fixed-first M0 | 210 | 14.671 | reference observed query error; oracle diagnostic only |
| BP-change sampling | Overall | 3 | Fixed-first M0 | 210 | 14.439 | reference observed query error; oracle diagnostic only |
| BP-change sampling | Overall | 5 | Fixed-first M0 | 210 | 13.882 | reference observed query error; oracle diagnostic only |
| Median anchor | Overall | 1 | Fixed-first M0 | 210 | 15.054 | reference observed query error; oracle diagnostic only |
| Median anchor | Overall | 2 | Fixed-first M0 | 210 | 14.513 | reference observed query error; oracle diagnostic only |
| Median anchor | Overall | 3 | Fixed-first M0 | 210 | 14.180 | reference observed query error; oracle diagnostic only |
| Median anchor | Overall | 5 | Fixed-first M0 | 210 | 13.877 | reference observed query error; oracle diagnostic only |
| PPG quality gate | Overall | 1 | Fixed-first M0 | 210 | 14.310 | reference observed query error; oracle diagnostic only |
| PPG quality gate | Overall | 2 | Fixed-first M0 | 210 | 13.619 | reference observed query error; oracle diagnostic only |
| PPG quality gate | Overall | 3 | Fixed-first M0 | 210 | 13.362 | reference observed query error; oracle diagnostic only |
| PPG quality gate | Overall | 5 | Fixed-first M0 | 210 | 12.921 | reference observed query error; oracle diagnostic only |
| Age and sex | Overall | 1 | Fixed-first M0 | 210 | 14.893 | reference observed query error; oracle diagnostic only |
| Age and sex | Overall | 2 | Fixed-first M0 | 210 | 14.306 | reference observed query error; oracle diagnostic only |
| Age and sex | Overall | 3 | Fixed-first M0 | 210 | 14.097 | reference observed query error; oracle diagnostic only |
| Age and sex | Overall | 5 | Fixed-first M0 | 210 | 13.698 | reference observed query error; oracle diagnostic only |
| Fixed-first M0 | MIMIC | 1 | Fixed-first M0 | 95 | 16.165 | reference observed query error; oracle diagnostic only |
| Fixed-first M0 | MIMIC | 2 | Fixed-first M0 | 95 | 15.827 | reference observed query error; oracle diagnostic only |
| Fixed-first M0 | MIMIC | 3 | Fixed-first M0 | 95 | 15.693 | reference observed query error; oracle diagnostic only |
| Fixed-first M0 | MIMIC | 5 | Fixed-first M0 | 95 | 15.364 | reference observed query error; oracle diagnostic only |
| Huber | MIMIC | 1 | Fixed-first M0 | 95 | 15.645 | reference observed query error; oracle diagnostic only |
| Huber | MIMIC | 2 | Fixed-first M0 | 95 | 15.351 | reference observed query error; oracle diagnostic only |
| Huber | MIMIC | 3 | Fixed-first M0 | 95 | 15.105 | reference observed query error; oracle diagnostic only |
| Huber | MIMIC | 5 | Fixed-first M0 | 95 | 14.909 | reference observed query error; oracle diagnostic only |
| BP-change sampling | MIMIC | 1 | Fixed-first M0 | 95 | 16.490 | reference observed query error; oracle diagnostic only |
| BP-change sampling | MIMIC | 2 | Fixed-first M0 | 95 | 16.351 | reference observed query error; oracle diagnostic only |
| BP-change sampling | MIMIC | 3 | Fixed-first M0 | 95 | 16.204 | reference observed query error; oracle diagnostic only |
| BP-change sampling | MIMIC | 5 | Fixed-first M0 | 95 | 15.777 | reference observed query error; oracle diagnostic only |
| Median anchor | MIMIC | 1 | Fixed-first M0 | 95 | 16.149 | reference observed query error; oracle diagnostic only |
| Median anchor | MIMIC | 2 | Fixed-first M0 | 95 | 15.896 | reference observed query error; oracle diagnostic only |
| Median anchor | MIMIC | 3 | Fixed-first M0 | 95 | 15.783 | reference observed query error; oracle diagnostic only |
| Median anchor | MIMIC | 5 | Fixed-first M0 | 95 | 15.547 | reference observed query error; oracle diagnostic only |
| PPG quality gate | MIMIC | 1 | Fixed-first M0 | 95 | 15.271 | reference observed query error; oracle diagnostic only |
| PPG quality gate | MIMIC | 2 | Fixed-first M0 | 95 | 14.590 | reference observed query error; oracle diagnostic only |
| PPG quality gate | MIMIC | 3 | Fixed-first M0 | 95 | 14.305 | reference observed query error; oracle diagnostic only |
| PPG quality gate | MIMIC | 5 | Fixed-first M0 | 95 | 13.866 | reference observed query error; oracle diagnostic only |
| Age and sex | MIMIC | 1 | Fixed-first M0 | 95 | 16.143 | reference observed query error; oracle diagnostic only |
| Age and sex | MIMIC | 2 | Fixed-first M0 | 95 | 15.875 | reference observed query error; oracle diagnostic only |
| Age and sex | MIMIC | 3 | Fixed-first M0 | 95 | 15.754 | reference observed query error; oracle diagnostic only |
| Age and sex | MIMIC | 5 | Fixed-first M0 | 95 | 15.490 | reference observed query error; oracle diagnostic only |
| Fixed-first M0 | VitalDB | 1 | Fixed-first M0 | 115 | 13.886 | reference observed query error; oracle diagnostic only |
| Fixed-first M0 | VitalDB | 2 | Fixed-first M0 | 115 | 12.976 | reference observed query error; oracle diagnostic only |
| Fixed-first M0 | VitalDB | 3 | Fixed-first M0 | 115 | 12.662 | reference observed query error; oracle diagnostic only |
| Fixed-first M0 | VitalDB | 5 | Fixed-first M0 | 115 | 12.110 | reference observed query error; oracle diagnostic only |
| Huber | VitalDB | 1 | Fixed-first M0 | 115 | 13.663 | reference observed query error; oracle diagnostic only |
| Huber | VitalDB | 2 | Fixed-first M0 | 115 | 12.920 | reference observed query error; oracle diagnostic only |
| Huber | VitalDB | 3 | Fixed-first M0 | 115 | 12.593 | reference observed query error; oracle diagnostic only |
| Huber | VitalDB | 5 | Fixed-first M0 | 115 | 12.111 | reference observed query error; oracle diagnostic only |
| BP-change sampling | VitalDB | 1 | Fixed-first M0 | 115 | 13.985 | reference observed query error; oracle diagnostic only |
| BP-change sampling | VitalDB | 2 | Fixed-first M0 | 115 | 13.236 | reference observed query error; oracle diagnostic only |
| BP-change sampling | VitalDB | 3 | Fixed-first M0 | 115 | 12.893 | reference observed query error; oracle diagnostic only |
| BP-change sampling | VitalDB | 5 | Fixed-first M0 | 115 | 12.161 | reference observed query error; oracle diagnostic only |
| Median anchor | VitalDB | 1 | Fixed-first M0 | 115 | 13.960 | reference observed query error; oracle diagnostic only |
| Median anchor | VitalDB | 2 | Fixed-first M0 | 115 | 13.159 | reference observed query error; oracle diagnostic only |
| Median anchor | VitalDB | 3 | Fixed-first M0 | 115 | 12.699 | reference observed query error; oracle diagnostic only |
| Median anchor | VitalDB | 5 | Fixed-first M0 | 115 | 12.354 | reference observed query error; oracle diagnostic only |
| PPG quality gate | VitalDB | 1 | Fixed-first M0 | 115 | 13.313 | reference observed query error; oracle diagnostic only |
| PPG quality gate | VitalDB | 2 | Fixed-first M0 | 115 | 12.789 | reference observed query error; oracle diagnostic only |
| PPG quality gate | VitalDB | 3 | Fixed-first M0 | 115 | 12.416 | reference observed query error; oracle diagnostic only |
| PPG quality gate | VitalDB | 5 | Fixed-first M0 | 115 | 11.899 | reference observed query error; oracle diagnostic only |
| Age and sex | VitalDB | 1 | Fixed-first M0 | 115 | 13.663 | reference observed query error; oracle diagnostic only |
| Age and sex | VitalDB | 2 | Fixed-first M0 | 115 | 12.774 | reference observed query error; oracle diagnostic only |
| Age and sex | VitalDB | 3 | Fixed-first M0 | 115 | 12.472 | reference observed query error; oracle diagnostic only |
| Age and sex | VitalDB | 5 | Fixed-first M0 | 115 | 12.034 | reference observed query error; oracle diagnostic only |

No candidate is automatically promoted by this report. Promotion requires a prespecified development decision and later repeated-seed confirmation.
