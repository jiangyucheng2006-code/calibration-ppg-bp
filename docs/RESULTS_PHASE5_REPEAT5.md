# Phase-5 five-seed development results

Last updated: 2026-08-17.

This report covers five prespecified training seeds (`20260813`--
`20260817`) on `meta_validation` only. The locked meta-test was not
accessed. Every method/K/seed row contains the same 697 participants and
103,564 future query events. Calibration uses K=1/2/3/5 independent
`event120-v1` pseudo-cuff/reference-BP events; all K values share queries
beginning at event 6.

## Primary participant-macro comparison

Values are mean ± sample SD across the five training seeds. Within each
seed, SBP and DBP MAE are first averaged per participant; the displayed
mean MAE is then the mean of the SBP and DBP participant-macro MAE.

| Method | K=1 | K=2 | K=3 | K=5 |
|---|---:|---:|---:|---:|
| B0 Population BP mean | 13.620 ± 0.000 | 13.620 ± 0.000 | 13.620 ± 0.000 | 13.620 ± 0.000 |
| B1 Last-cuff persistence | 15.023 ± 0.000 | 13.825 ± 0.000 | 12.993 ± 0.000 | 12.733 ± 0.000 |
| B2 Support-BP mean | 15.023 ± 0.000 | 13.570 ± 0.000 | 12.755 ± 0.000 | 11.948 ± 0.000 |
| B3 Population PPG network | 11.710 ± 0.065 | 11.710 ± 0.065 | 11.710 ± 0.065 | 11.710 ± 0.065 |
| B4 Residual-offset correction | 12.327 ± 0.132 | 11.200 ± 0.153 | 10.624 ± 0.169 | 9.951 ± 0.166 |
| B6 Head-only adaptation | 10.963 ± 0.040 | 10.472 ± 0.087 | 10.105 ± 0.118 | 9.588 ± 0.116 |
| B7 Full-network adaptation | 15.042 ± 0.050 | 13.382 ± 0.147 | 12.520 ± 0.178 | 11.842 ± 0.168 |
| B8 LoRA adaptation | 10.499 ± 0.110 | 10.140 ± 0.104 | 9.944 ± 0.123 | 9.608 ± 0.127 |
| M0 Variable-K residual anchor | 10.081 ± 0.096 | 9.526 ± 0.112 | 9.235 ± 0.116 | 8.785 ± 0.094 |
| M1 M0 + FiLM | 10.190 ± 0.176 | 9.563 ± 0.146 | 9.251 ± 0.135 | 8.807 ± 0.119 |
| M2 M1 + reliability weighting | 10.233 ± 0.081 | 9.641 ± 0.116 | 9.332 ± 0.119 | 8.834 ± 0.104 |

Across the four K budgets, M0 has the lowest average participant-macro
mean MAE (9.407 ± 0.104 mmHg). M1 and M2 are close, so the current evidence supports selecting
M0 primarily for parsimony rather than claiming a decisive advantage
over every more complex variant. M0 remains clearly better than the
calibration-free population network and the prespecified simple or
adaptation controls.

## M0 SBP and DBP detail

| K | SBP MAE | DBP MAE | Mean MAE |
|---:|---:|---:|---:|
| 1 | 12.920 ± 0.111 | 7.242 ± 0.096 | 10.081 ± 0.096 |
| 2 | 12.240 ± 0.136 | 6.812 ± 0.095 | 9.526 ± 0.112 |
| 3 | 11.894 ± 0.144 | 6.576 ± 0.095 | 9.235 ± 0.116 |
| 5 | 11.318 ± 0.110 | 6.253 ± 0.086 | 8.785 ± 0.094 |

## Extended diagnostic table

`ME` is prediction minus reference. Each numeric cell is the mean ±
sample SD across five seed-specific event-pooled metrics. In the `STD`
column, the first number is the mean within-seed sample SD of signed
errors; the second number is its between-seed SD. These pooled metrics
are secondary diagnostics; participant-macro MAE above is primary.

`AAMI` is only a Criterion-1-style numerical screen (`|ME| <= 5 mmHg`
and error `STD <= 8 mmHg`) applied separately within each seed/BP row.
`BHS` is the historical cumulative 5/10/15-mmHg numerical grade, with
Grade A/B displayed as pass. Asterisks mean **numerical screen only;
formal device compliance is not established**. This retrospective
PulseDB evaluation does not meet the full population, reference,
pairing, or repeated-measure requirements of a device-validation study.

The [ISO catalogue](https://www.iso.org/standard/73339.html), checked
2026-08-17, lists ISO 81060-2:2018 as current (confirmed in 2024), with
2020 and 2024 amendments and a replacement draft in development; its
published scope is intermittent cuff-based equipment. The historical
BHS grading source is the [original 1990 protocol](https://pubmed.ncbi.nlm.nih.gov/2168451/).

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg (%) | ≤10 mmHg (%) | ≤15 mmHg (%) | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0 Population BP mean (K=1) | SBP | 17.977 ± 0.000 | -0.004 ± 0.000 | 1.377 ± 0.000 | 22.015 ± 0.000 | 15.98 ± 0.00 | 31.90 ± 0.00 | 47.19 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B0 Population BP mean (K=1) | DBP | 10.199 ± 0.000 | -0.001 ± 0.000 | -0.347 ± 0.000 | 13.166 ± 0.000 | 29.95 ± 0.00 | 57.97 ± 0.00 | 77.58 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B0 Population BP mean (K=2) | SBP | 17.977 ± 0.000 | -0.004 ± 0.000 | 1.377 ± 0.000 | 22.015 ± 0.000 | 15.98 ± 0.00 | 31.90 ± 0.00 | 47.19 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B0 Population BP mean (K=2) | DBP | 10.199 ± 0.000 | -0.001 ± 0.000 | -0.347 ± 0.000 | 13.166 ± 0.000 | 29.95 ± 0.00 | 57.97 ± 0.00 | 77.58 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B0 Population BP mean (K=3) | SBP | 17.977 ± 0.000 | -0.004 ± 0.000 | 1.377 ± 0.000 | 22.015 ± 0.000 | 15.98 ± 0.00 | 31.90 ± 0.00 | 47.19 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B0 Population BP mean (K=3) | DBP | 10.199 ± 0.000 | -0.001 ± 0.000 | -0.347 ± 0.000 | 13.166 ± 0.000 | 29.95 ± 0.00 | 57.97 ± 0.00 | 77.58 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B0 Population BP mean (K=5) | SBP | 17.977 ± 0.000 | -0.004 ± 0.000 | 1.377 ± 0.000 | 22.015 ± 0.000 | 15.98 ± 0.00 | 31.90 ± 0.00 | 47.19 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B0 Population BP mean (K=5) | DBP | 10.199 ± 0.000 | -0.001 ± 0.000 | -0.347 ± 0.000 | 13.166 ± 0.000 | 29.95 ± 0.00 | 57.97 ± 0.00 | 77.58 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B1 Last-cuff persistence (K=1) | SBP | 18.831 ± 0.000 | -0.318 ± 0.000 | 0.198 ± 0.000 | 25.270 ± 0.000 | 20.26 ± 0.00 | 38.20 ± 0.00 | 52.49 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B1 Last-cuff persistence (K=1) | DBP | 10.694 ± 0.000 | -0.253 ± 0.000 | 0.594 ± 0.000 | 14.725 ± 0.000 | 34.93 ± 0.00 | 59.22 ± 0.00 | 75.39 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B1 Last-cuff persistence (K=2) | SBP | 18.104 ± 0.000 | -0.189 ± 0.000 | 1.059 ± 0.000 | 23.981 ± 0.000 | 20.42 ± 0.00 | 38.69 ± 0.00 | 53.61 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B1 Last-cuff persistence (K=2) | DBP | 10.377 ± 0.000 | -0.119 ± 0.000 | 0.573 ± 0.000 | 13.913 ± 0.000 | 34.74 ± 0.00 | 59.51 ± 0.00 | 76.13 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B1 Last-cuff persistence (K=3) | SBP | 17.364 ± 0.000 | -0.084 ± 0.000 | -0.035 ± 0.000 | 22.923 ± 0.000 | 21.12 ± 0.00 | 39.67 ± 0.00 | 54.83 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B1 Last-cuff persistence (K=3) | DBP | 9.958 ± 0.000 | -0.054 ± 0.000 | 0.326 ± 0.000 | 13.512 ± 0.000 | 36.67 ± 0.00 | 61.85 ± 0.00 | 77.68 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B1 Last-cuff persistence (K=5) | SBP | 17.140 ± 0.000 | -0.077 ± 0.000 | -0.691 ± 0.000 | 22.834 ± 0.000 | 22.47 ± 0.00 | 41.10 ± 0.00 | 56.25 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B1 Last-cuff persistence (K=5) | DBP | 9.967 ± 0.000 | -0.055 ± 0.000 | 0.014 ± 0.000 | 13.521 ± 0.000 | 36.67 ± 0.00 | 61.53 ± 0.00 | 77.16 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B2 Support-BP mean (K=1) | SBP | 18.831 ± 0.000 | -0.318 ± 0.000 | 0.198 ± 0.000 | 25.270 ± 0.000 | 20.26 ± 0.00 | 38.20 ± 0.00 | 52.49 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B2 Support-BP mean (K=1) | DBP | 10.694 ± 0.000 | -0.253 ± 0.000 | 0.594 ± 0.000 | 14.725 ± 0.000 | 34.93 ± 0.00 | 59.22 ± 0.00 | 75.39 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B2 Support-BP mean (K=2) | SBP | 17.816 ± 0.000 | -0.161 ± 0.000 | 0.628 ± 0.000 | 23.712 ± 0.000 | 21.04 ± 0.00 | 39.58 ± 0.00 | 54.54 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B2 Support-BP mean (K=2) | DBP | 10.120 ± 0.000 | -0.081 ± 0.000 | 0.583 ± 0.000 | 13.673 ± 0.000 | 36.09 ± 0.00 | 60.83 ± 0.00 | 76.94 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B2 Support-BP mean (K=3) | SBP | 17.166 ± 0.000 | -0.071 ± 0.000 | 0.407 ± 0.000 | 22.777 ± 0.000 | 21.79 ± 0.00 | 40.78 ± 0.00 | 55.64 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B2 Support-BP mean (K=3) | DBP | 9.772 ± 0.000 | -0.009 ± 0.000 | 0.498 ± 0.000 | 13.217 ± 0.000 | 37.24 ± 0.00 | 62.28 ± 0.00 | 78.23 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B2 Support-BP mean (K=5) | SBP | 16.487 ± 0.000 | 0.011 ± 0.000 | -0.031 ± 0.000 | 21.893 ± 0.000 | 22.78 ± 0.00 | 42.22 ± 0.00 | 57.20 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B2 Support-BP mean (K=5) | DBP | 9.412 ± 0.000 | 0.049 ± 0.000 | 0.269 ± 0.000 | 12.837 ± 0.000 | 38.69 ± 0.00 | 64.32 ± 0.00 | 79.71 ± 0.00 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B3 Population PPG network (K=1) | SBP | 15.191 ± 0.186 | 0.207 ± 0.022 | -1.328 ± 0.700 | 19.547 ± 0.225 | 21.63 ± 0.27 | 41.81 ± 0.31 | 59.00 ± 0.60 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B3 Population PPG network (K=1) | DBP | 9.219 ± 0.106 | 0.155 ± 0.031 | -0.228 ± 0.834 | 12.075 ± 0.188 | 34.11 ± 0.57 | 63.25 ± 0.40 | 82.17 ± 0.39 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B3 Population PPG network (K=2) | SBP | 15.191 ± 0.186 | 0.207 ± 0.022 | -1.328 ± 0.700 | 19.547 ± 0.225 | 21.63 ± 0.27 | 41.81 ± 0.31 | 59.00 ± 0.60 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B3 Population PPG network (K=2) | DBP | 9.219 ± 0.106 | 0.155 ± 0.031 | -0.228 ± 0.834 | 12.075 ± 0.188 | 34.11 ± 0.57 | 63.25 ± 0.40 | 82.17 ± 0.39 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B3 Population PPG network (K=3) | SBP | 15.191 ± 0.186 | 0.207 ± 0.022 | -1.328 ± 0.700 | 19.547 ± 0.225 | 21.63 ± 0.27 | 41.81 ± 0.31 | 59.00 ± 0.60 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B3 Population PPG network (K=3) | DBP | 9.219 ± 0.106 | 0.155 ± 0.031 | -0.228 ± 0.834 | 12.075 ± 0.188 | 34.11 ± 0.57 | 63.25 ± 0.40 | 82.17 ± 0.39 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B3 Population PPG network (K=5) | SBP | 15.191 ± 0.186 | 0.207 ± 0.022 | -1.328 ± 0.700 | 19.547 ± 0.225 | 21.63 ± 0.27 | 41.81 ± 0.31 | 59.00 ± 0.60 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B3 Population PPG network (K=5) | DBP | 9.219 ± 0.106 | 0.155 ± 0.031 | -0.228 ± 0.834 | 12.075 ± 0.188 | 34.11 ± 0.57 | 63.25 ± 0.40 | 82.17 ± 0.39 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B4 Residual-offset correction (K=1) | SBP | 16.538 ± 0.257 | 0.020 ± 0.026 | 1.465 ± 0.499 | 21.736 ± 0.292 | 21.43 ± 0.42 | 40.83 ± 0.65 | 56.65 ± 0.69 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B4 Residual-offset correction (K=1) | DBP | 9.307 ± 0.171 | 0.065 ± 0.029 | 0.442 ± 0.090 | 12.722 ± 0.201 | 37.83 ± 1.00 | 64.60 ± 0.93 | 80.95 ± 0.54 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B4 Residual-offset correction (K=2) | SBP | 15.530 ± 0.233 | 0.136 ± 0.026 | 2.008 ± 0.457 | 20.358 ± 0.322 | 22.96 ± 0.43 | 43.41 ± 0.46 | 59.55 ± 0.47 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B4 Residual-offset correction (K=2) | DBP | 8.782 ± 0.217 | 0.178 ± 0.036 | 0.502 ± 0.096 | 11.923 ± 0.261 | 39.90 ± 1.12 | 66.92 ± 1.15 | 82.52 ± 0.79 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B4 Residual-offset correction (K=3) | SBP | 14.942 ± 0.240 | 0.200 ± 0.028 | 1.968 ± 0.491 | 19.584 ± 0.343 | 23.61 ± 0.47 | 44.67 ± 0.52 | 61.03 ± 0.51 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B4 Residual-offset correction (K=3) | DBP | 8.382 ± 0.202 | 0.250 ± 0.031 | 0.484 ± 0.100 | 11.393 ± 0.234 | 41.38 ± 1.14 | 68.81 ± 1.11 | 84.21 ± 0.71 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=1, D=4) |
| B4 Residual-offset correction (K=5) | SBP | 14.335 ± 0.193 | 0.258 ± 0.023 | 1.557 ± 0.433 | 18.895 ± 0.291 | 24.63 ± 0.31 | 46.27 ± 0.38 | 62.90 ± 0.48 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B4 Residual-offset correction (K=5) | DBP | 7.995 ± 0.196 | 0.308 ± 0.029 | 0.388 ± 0.092 | 10.945 ± 0.228 | 43.27 ± 1.10 | 70.96 ± 1.08 | 85.73 ± 0.66 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| B6 Head-only adaptation (K=1) | SBP | 14.712 ± 0.192 | 0.257 ± 0.016 | -0.102 ± 0.215 | 18.971 ± 0.205 | 22.98 ± 0.51 | 43.47 ± 0.79 | 60.21 ± 0.66 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B6 Head-only adaptation (K=1) | DBP | 8.235 ± 0.067 | 0.297 ± 0.014 | -0.061 ± 0.251 | 11.039 ± 0.107 | 40.46 ± 0.39 | 68.98 ± 0.48 | 85.33 ± 0.52 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=2, D=3) |
| B6 Head-only adaptation (K=2) | SBP | 14.307 ± 0.191 | 0.290 ± 0.017 | 0.698 ± 0.194 | 18.532 ± 0.219 | 23.60 ± 0.45 | 44.82 ± 0.79 | 61.87 ± 0.70 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B6 Head-only adaptation (K=2) | DBP | 8.092 ± 0.078 | 0.313 ± 0.012 | 0.204 ± 0.159 | 10.909 ± 0.098 | 41.39 ± 0.78 | 70.01 ± 0.67 | 85.90 ± 0.43 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| B6 Head-only adaptation (K=3) | SBP | 14.006 ± 0.199 | 0.316 ± 0.018 | 0.711 ± 0.205 | 18.186 ± 0.238 | 24.16 ± 0.42 | 45.69 ± 0.75 | 62.91 ± 0.79 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B6 Head-only adaptation (K=3) | DBP | 7.884 ± 0.086 | 0.340 ± 0.010 | 0.242 ± 0.150 | 10.690 ± 0.080 | 42.66 ± 0.92 | 71.37 ± 0.61 | 86.76 ± 0.36 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| B6 Head-only adaptation (K=5) | SBP | 13.526 ± 0.110 | 0.352 ± 0.012 | 0.532 ± 0.178 | 17.716 ± 0.163 | 25.55 ± 0.32 | 47.70 ± 0.36 | 64.84 ± 0.29 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B6 Head-only adaptation (K=5) | DBP | 7.688 ± 0.096 | 0.368 ± 0.010 | 0.236 ± 0.185 | 10.459 ± 0.082 | 43.59 ± 0.95 | 72.43 ± 0.58 | 87.67 ± 0.38 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| B7 Full-network adaptation (K=1) | SBP | 18.820 ± 0.058 | -0.315 ± 0.008 | 0.177 ± 0.177 | 25.246 ± 0.074 | 20.27 ± 0.13 | 38.20 ± 0.16 | 52.44 ± 0.20 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B7 Full-network adaptation (K=1) | DBP | 10.739 ± 0.041 | -0.260 ± 0.008 | 0.555 ± 0.058 | 14.765 ± 0.047 | 34.77 ± 0.23 | 59.07 ± 0.25 | 75.20 ± 0.21 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B7 Full-network adaptation (K=2) | SBP | 17.820 ± 0.168 | -0.157 ± 0.019 | 0.468 ± 0.190 | 23.677 ± 0.195 | 20.87 ± 0.34 | 39.52 ± 0.44 | 54.36 ± 0.48 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B7 Full-network adaptation (K=2) | DBP | 10.124 ± 0.088 | -0.079 ± 0.019 | 0.459 ± 0.043 | 13.667 ± 0.118 | 35.93 ± 0.29 | 60.73 ± 0.35 | 76.94 ± 0.35 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B7 Full-network adaptation (K=3) | SBP | 17.130 ± 0.095 | -0.066 ± 0.013 | 0.251 ± 0.161 | 22.727 ± 0.145 | 21.55 ± 0.13 | 40.61 ± 0.15 | 55.63 ± 0.09 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B7 Full-network adaptation (K=3) | DBP | 9.717 ± 0.052 | -0.003 ± 0.015 | 0.368 ± 0.071 | 13.182 ± 0.100 | 37.41 ± 0.22 | 62.66 ± 0.22 | 78.60 ± 0.24 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B7 Full-network adaptation (K=5) | SBP | 16.519 ± 0.143 | 0.001 ± 0.016 | 0.038 ± 0.198 | 22.008 ± 0.176 | 22.87 ± 0.28 | 42.42 ± 0.34 | 57.43 ± 0.37 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B7 Full-network adaptation (K=5) | DBP | 9.398 ± 0.068 | 0.046 ± 0.014 | 0.250 ± 0.029 | 12.856 ± 0.096 | 38.99 ± 0.23 | 64.41 ± 0.24 | 79.71 ± 0.26 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B8 LoRA adaptation (K=1) | SBP | 14.079 ± 0.170 | 0.321 ± 0.017 | -1.003 ± 0.410 | 18.108 ± 0.220 | 23.74 ± 0.46 | 45.17 ± 0.65 | 62.03 ± 0.52 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B8 LoRA adaptation (K=1) | DBP | 7.949 ± 0.150 | 0.336 ± 0.023 | -0.490 ± 0.448 | 10.705 ± 0.171 | 41.30 ± 0.82 | 70.93 ± 0.84 | 86.98 ± 0.70 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| B8 LoRA adaptation (K=2) | SBP | 13.754 ± 0.152 | 0.346 ± 0.016 | -0.465 ± 0.350 | 17.792 ± 0.217 | 24.39 ± 0.33 | 46.42 ± 0.51 | 63.40 ± 0.37 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B8 LoRA adaptation (K=2) | DBP | 7.865 ± 0.134 | 0.345 ± 0.021 | -0.377 ± 0.391 | 10.641 ± 0.159 | 41.97 ± 0.81 | 71.56 ± 0.68 | 87.17 ± 0.58 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| B8 LoRA adaptation (K=3) | SBP | 13.566 ± 0.171 | 0.361 ± 0.017 | -0.404 ± 0.376 | 17.591 ± 0.229 | 24.89 ± 0.40 | 47.11 ± 0.65 | 64.11 ± 0.47 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B8 LoRA adaptation (K=3) | DBP | 7.732 ± 0.114 | 0.363 ± 0.019 | -0.270 ± 0.377 | 10.497 ± 0.140 | 42.81 ± 0.69 | 72.40 ± 0.59 | 87.68 ± 0.45 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| B8 LoRA adaptation (K=5) | SBP | 13.281 ± 0.160 | 0.381 ± 0.017 | -0.442 ± 0.305 | 17.309 ± 0.238 | 25.71 ± 0.23 | 48.22 ± 0.47 | 65.21 ± 0.41 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| B8 LoRA adaptation (K=5) | DBP | 7.575 ± 0.124 | 0.381 ± 0.019 | -0.221 ± 0.347 | 10.354 ± 0.151 | 44.02 ± 0.77 | 73.40 ± 0.65 | 88.08 ± 0.53 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| M0 Variable-K residual anchor (K=1) | SBP | 13.686 ± 0.059 | 0.326 ± 0.007 | 1.079 ± 0.623 | 18.034 ± 0.085 | 25.97 ± 0.17 | 48.03 ± 0.28 | 64.65 ± 0.32 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M0 Variable-K residual anchor (K=1) | DBP | 7.971 ± 0.081 | 0.315 ± 0.009 | 0.796 ± 0.347 | 10.863 ± 0.063 | 43.02 ± 0.56 | 70.96 ± 0.46 | 86.07 ± 0.30 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| M0 Variable-K residual anchor (K=2) | SBP | 13.170 ± 0.050 | 0.368 ± 0.005 | 1.606 ± 0.583 | 17.418 ± 0.056 | 27.28 ± 0.32 | 49.85 ± 0.27 | 66.38 ± 0.13 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M0 Variable-K residual anchor (K=2) | DBP | 7.770 ± 0.086 | 0.338 ± 0.008 | 0.878 ± 0.250 | 10.673 ± 0.048 | 44.69 ± 0.71 | 72.43 ± 0.72 | 86.66 ± 0.21 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| M0 Variable-K residual anchor (K=3) | SBP | 12.919 ± 0.108 | 0.391 ± 0.009 | 1.609 ± 0.555 | 17.093 ± 0.131 | 27.55 ± 0.49 | 50.55 ± 0.45 | 67.24 ± 0.26 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M0 Variable-K residual anchor (K=3) | DBP | 7.627 ± 0.109 | 0.357 ± 0.011 | 0.910 ± 0.207 | 10.513 ± 0.085 | 45.63 ± 0.86 | 73.32 ± 0.95 | 87.05 ± 0.30 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| M0 Variable-K residual anchor (K=5) | SBP | 12.536 ± 0.089 | 0.420 ± 0.010 | 1.285 ± 0.546 | 16.709 ± 0.137 | 28.75 ± 0.24 | 52.05 ± 0.35 | 68.69 ± 0.28 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M0 Variable-K residual anchor (K=5) | DBP | 7.367 ± 0.099 | 0.390 ± 0.011 | 0.779 ± 0.191 | 10.250 ± 0.087 | 47.32 ± 0.69 | 74.77 ± 0.76 | 87.79 ± 0.31 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| M1 M0 + FiLM (K=1) | SBP | 13.872 ± 0.224 | 0.308 ± 0.023 | 1.049 ± 0.734 | 18.276 ± 0.307 | 25.56 ± 0.49 | 47.47 ± 0.57 | 64.16 ± 0.44 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M1 M0 + FiLM (K=1) | DBP | 8.082 ± 0.162 | 0.296 ± 0.023 | 0.652 ± 0.667 | 11.010 ± 0.135 | 42.44 ± 0.82 | 70.45 ± 1.03 | 85.79 ± 0.68 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=4, D=1) |
| M1 M0 + FiLM (K=2) | SBP | 13.213 ± 0.214 | 0.363 ± 0.021 | 1.493 ± 0.725 | 17.489 ± 0.273 | 27.08 ± 0.52 | 49.68 ± 0.51 | 66.32 ± 0.47 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M1 M0 + FiLM (K=2) | DBP | 7.815 ± 0.150 | 0.331 ± 0.017 | 0.723 ± 0.629 | 10.729 ± 0.092 | 44.23 ± 0.92 | 72.28 ± 1.14 | 86.56 ± 0.50 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| M1 M0 + FiLM (K=3) | SBP | 12.915 ± 0.230 | 0.390 ± 0.022 | 1.456 ± 0.740 | 17.122 ± 0.301 | 27.61 ± 0.44 | 50.59 ± 0.59 | 67.36 ± 0.68 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M1 M0 + FiLM (K=3) | DBP | 7.644 ± 0.150 | 0.355 ± 0.015 | 0.744 ± 0.615 | 10.533 ± 0.077 | 45.31 ± 1.12 | 73.33 ± 1.14 | 87.17 ± 0.48 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| M1 M0 + FiLM (K=5) | SBP | 12.541 ± 0.214 | 0.418 ± 0.019 | 1.139 ± 0.730 | 16.743 ± 0.275 | 28.77 ± 0.47 | 52.05 ± 0.63 | 68.71 ± 0.75 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M1 M0 + FiLM (K=5) | DBP | 7.379 ± 0.110 | 0.389 ± 0.009 | 0.619 ± 0.579 | 10.261 ± 0.047 | 47.01 ± 0.98 | 74.73 ± 0.83 | 87.95 ± 0.31 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| M2 M1 + reliability weighting (K=1) | SBP | 13.836 ± 0.150 | 0.314 ± 0.017 | 1.046 ± 0.726 | 18.189 ± 0.250 | 25.48 ± 0.42 | 47.42 ± 0.36 | 64.19 ± 0.25 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M2 M1 + reliability weighting (K=1) | DBP | 8.142 ± 0.096 | 0.286 ± 0.015 | 0.864 ± 0.407 | 11.085 ± 0.112 | 42.17 ± 0.76 | 70.06 ± 0.54 | 85.60 ± 0.22 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| M2 M1 + reliability weighting (K=2) | SBP | 13.448 ± 0.114 | 0.342 ± 0.014 | 1.523 ± 0.752 | 17.777 ± 0.227 | 26.32 ± 0.44 | 48.95 ± 0.48 | 65.77 ± 0.15 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M2 M1 + reliability weighting (K=2) | DBP | 7.905 ± 0.075 | 0.316 ± 0.009 | 0.953 ± 0.389 | 10.842 ± 0.081 | 43.78 ± 0.99 | 71.78 ± 0.56 | 86.43 ± 0.09 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| M2 M1 + reliability weighting (K=3) | SBP | 13.071 ± 0.105 | 0.376 ± 0.012 | 1.438 ± 0.740 | 17.316 ± 0.193 | 27.05 ± 0.41 | 50.08 ± 0.55 | 66.95 ± 0.24 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M2 M1 + reliability weighting (K=3) | DBP | 7.717 ± 0.069 | 0.345 ± 0.009 | 0.910 ± 0.394 | 10.608 ± 0.090 | 44.73 ± 0.97 | 72.82 ± 0.54 | 87.04 ± 0.13 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |
| M2 M1 + reliability weighting (K=5) | SBP | 12.624 ± 0.113 | 0.407 ± 0.011 | 1.279 ± 0.658 | 16.890 ± 0.169 | 28.76 ± 0.35 | 51.99 ± 0.41 | 68.65 ± 0.26 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=0, D=5) |
| M2 M1 + reliability weighting (K=5) | DBP | 7.490 ± 0.058 | 0.373 ± 0.007 | 0.766 ± 0.315 | 10.393 ± 0.070 | 46.41 ± 0.82 | 74.06 ± 0.52 | 87.58 ± 0.14 | FAIL* (0/5 seeds pass) | FAIL* (0/5 seeds pass; A=0, B=0, C=5, D=0) |

## Interpretation and decision boundary

- M0 improves as K increases and is the lowest-mean method at every K.
- FiLM (M1) and reliability weighting (M2) do not produce a consistent
  average gain over M0 under the shared protocol.
- The unlimited-epoch runs all stopped by patience=8 early stopping;
  therefore the original 25-epoch-cap concern is resolved.
- This is still internal model-selection evidence. It does not establish
  locked-test generalization, pressure/motion/device robustness, external
  validation, clinical accuracy, or standards compliance.

The machine-readable result files retain the seed-specific rows, seed
SDs, participant/event counts, method/K identifiers, and standards
scope needed to reconstruct this table.
