# Phase-5 first-run extended result table

Last updated: 2026-08-15.

This is an exploratory, single-seed (`20260813`) `meta_validation` 
report reconstructed from saved event-level predictions. The locked test 
was not accessed. Every row contains 697 participants and 103,564 common 
future query events. `ME` is prediction minus reference; `STD` uses the 
sample standard deviation (`ddof=1`). MAE, R², ME, STD, and threshold 
percentages in this table are event-pooled diagnostics. The project primary 
MAE remains the participant-macro value reported in `RESULTS_PHASE5.md`.

`AAMI` is only a Criterion-1-style numerical screen: `|ME| <= 5 mmHg` 
and `STD <= 8 mmHg`, evaluated separately for SBP and DBP. The full 
AAMI/ESH/ISO protocol also has design, population, reference-measurement, 
and repeated-measure requirements that this retrospective ML benchmark does 
not satisfy. `BHS` is a historical numerical grade based on cumulative 
percentages within 5/10/15 mmHg; Grade A/B is displayed as PASS and C/D as 
FAIL. Every asterisk therefore means **numerical screen only; formal device 
compliance is not established**.

Primary references: [AAMI/ESH/ISO collaboration statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC5796427/), [current ISO 81060-2:2018 status](https://www.iso.org/standard/73339.html), and the [original BHS protocol](https://pubmed.ncbi.nlm.nih.gov/2168451/).

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg (%) | ≤10 mmHg (%) | ≤15 mmHg (%) | AAMI | BHS |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 Population BP mean (K=1) | SBP | 17.977 | -0.004 | 1.377 | 22.015 | 15.98 | 31.90 | 47.19 | FAIL* | FAIL (Grade D)* |
| B0 Population BP mean (K=1) | DBP | 10.199 | -0.001 | -0.347 | 13.166 | 29.95 | 57.97 | 77.58 | FAIL* | FAIL (Grade D)* |
| B0 Population BP mean (K=2) | SBP | 17.977 | -0.004 | 1.377 | 22.015 | 15.98 | 31.90 | 47.19 | FAIL* | FAIL (Grade D)* |
| B0 Population BP mean (K=2) | DBP | 10.199 | -0.001 | -0.347 | 13.166 | 29.95 | 57.97 | 77.58 | FAIL* | FAIL (Grade D)* |
| B0 Population BP mean (K=3) | SBP | 17.977 | -0.004 | 1.377 | 22.015 | 15.98 | 31.90 | 47.19 | FAIL* | FAIL (Grade D)* |
| B0 Population BP mean (K=3) | DBP | 10.199 | -0.001 | -0.347 | 13.166 | 29.95 | 57.97 | 77.58 | FAIL* | FAIL (Grade D)* |
| B0 Population BP mean (K=5) | SBP | 17.977 | -0.004 | 1.377 | 22.015 | 15.98 | 31.90 | 47.19 | FAIL* | FAIL (Grade D)* |
| B0 Population BP mean (K=5) | DBP | 10.199 | -0.001 | -0.347 | 13.166 | 29.95 | 57.97 | 77.58 | FAIL* | FAIL (Grade D)* |
| B3 Population PPG network (K=1) | SBP | 15.062 | 0.229 | -0.720 | 19.323 | 21.60 | 41.80 | 59.00 | FAIL* | FAIL (Grade D)* |
| B3 Population PPG network (K=1) | DBP | 9.147 | 0.176 | -0.292 | 11.948 | 34.08 | 63.44 | 82.44 | FAIL* | FAIL (Grade D)* |
| B3 Population PPG network (K=2) | SBP | 15.062 | 0.229 | -0.720 | 19.323 | 21.60 | 41.80 | 59.00 | FAIL* | FAIL (Grade D)* |
| B3 Population PPG network (K=2) | DBP | 9.147 | 0.176 | -0.292 | 11.948 | 34.08 | 63.44 | 82.44 | FAIL* | FAIL (Grade D)* |
| B3 Population PPG network (K=3) | SBP | 15.062 | 0.229 | -0.720 | 19.323 | 21.60 | 41.80 | 59.00 | FAIL* | FAIL (Grade D)* |
| B3 Population PPG network (K=3) | DBP | 9.147 | 0.176 | -0.292 | 11.948 | 34.08 | 63.44 | 82.44 | FAIL* | FAIL (Grade D)* |
| B3 Population PPG network (K=5) | SBP | 15.062 | 0.229 | -0.720 | 19.323 | 21.60 | 41.80 | 59.00 | FAIL* | FAIL (Grade D)* |
| B3 Population PPG network (K=5) | DBP | 9.147 | 0.176 | -0.292 | 11.948 | 34.08 | 63.44 | 82.44 | FAIL* | FAIL (Grade D)* |
| B1 Last-cuff persistence (K=1) | SBP | 18.831 | -0.318 | 0.198 | 25.270 | 20.26 | 38.20 | 52.49 | FAIL* | FAIL (Grade D)* |
| B1 Last-cuff persistence (K=1) | DBP | 10.694 | -0.253 | 0.594 | 14.725 | 34.93 | 59.22 | 75.39 | FAIL* | FAIL (Grade D)* |
| B1 Last-cuff persistence (K=2) | SBP | 18.104 | -0.189 | 1.059 | 23.981 | 20.42 | 38.69 | 53.61 | FAIL* | FAIL (Grade D)* |
| B1 Last-cuff persistence (K=2) | DBP | 10.377 | -0.119 | 0.573 | 13.913 | 34.74 | 59.51 | 76.13 | FAIL* | FAIL (Grade D)* |
| B1 Last-cuff persistence (K=3) | SBP | 17.364 | -0.084 | -0.035 | 22.923 | 21.12 | 39.67 | 54.83 | FAIL* | FAIL (Grade D)* |
| B1 Last-cuff persistence (K=3) | DBP | 9.958 | -0.054 | 0.326 | 13.512 | 36.67 | 61.85 | 77.68 | FAIL* | FAIL (Grade D)* |
| B1 Last-cuff persistence (K=5) | SBP | 17.140 | -0.077 | -0.691 | 22.834 | 22.47 | 41.10 | 56.25 | FAIL* | FAIL (Grade D)* |
| B1 Last-cuff persistence (K=5) | DBP | 9.967 | -0.055 | 0.014 | 13.521 | 36.67 | 61.53 | 77.16 | FAIL* | FAIL (Grade D)* |
| B2 Support-BP mean (K=1) | SBP | 18.831 | -0.318 | 0.198 | 25.270 | 20.26 | 38.20 | 52.49 | FAIL* | FAIL (Grade D)* |
| B2 Support-BP mean (K=1) | DBP | 10.694 | -0.253 | 0.594 | 14.725 | 34.93 | 59.22 | 75.39 | FAIL* | FAIL (Grade D)* |
| B2 Support-BP mean (K=2) | SBP | 17.816 | -0.161 | 0.628 | 23.712 | 21.04 | 39.58 | 54.54 | FAIL* | FAIL (Grade D)* |
| B2 Support-BP mean (K=2) | DBP | 10.120 | -0.081 | 0.583 | 13.673 | 36.09 | 60.83 | 76.94 | FAIL* | FAIL (Grade D)* |
| B2 Support-BP mean (K=3) | SBP | 17.166 | -0.071 | 0.407 | 22.777 | 21.79 | 40.78 | 55.64 | FAIL* | FAIL (Grade D)* |
| B2 Support-BP mean (K=3) | DBP | 9.772 | -0.009 | 0.498 | 13.217 | 37.24 | 62.28 | 78.23 | FAIL* | FAIL (Grade D)* |
| B2 Support-BP mean (K=5) | SBP | 16.487 | 0.011 | -0.031 | 21.893 | 22.78 | 42.22 | 57.20 | FAIL* | FAIL (Grade D)* |
| B2 Support-BP mean (K=5) | DBP | 9.412 | 0.049 | 0.269 | 12.837 | 38.69 | 64.32 | 79.71 | FAIL* | FAIL (Grade D)* |
| B4 Residual-offset correction (K=1) | SBP | 16.524 | 0.031 | 0.692 | 21.664 | 21.40 | 40.52 | 56.26 | FAIL* | FAIL (Grade D)* |
| B4 Residual-offset correction (K=1) | DBP | 9.252 | 0.060 | 0.354 | 12.758 | 38.48 | 65.04 | 81.18 | FAIL* | FAIL (Grade D)* |
| B4 Residual-offset correction (K=2) | SBP | 15.548 | 0.142 | 1.268 | 20.347 | 22.75 | 43.00 | 59.09 | FAIL* | FAIL (Grade D)* |
| B4 Residual-offset correction (K=2) | DBP | 8.724 | 0.178 | 0.388 | 11.931 | 40.65 | 67.33 | 82.61 | FAIL* | FAIL (Grade D)* |
| B4 Residual-offset correction (K=3) | SBP | 14.901 | 0.214 | 1.188 | 19.483 | 23.36 | 44.45 | 60.85 | FAIL* | FAIL (Grade D)* |
| B4 Residual-offset correction (K=3) | DBP | 8.323 | 0.252 | 0.357 | 11.377 | 42.02 | 69.15 | 84.34 | FAIL* | FAIL (Grade D)* |
| B4 Residual-offset correction (K=5) | SBP | 14.288 | 0.272 | 0.824 | 18.765 | 24.38 | 46.05 | 62.70 | FAIL* | FAIL (Grade D)* |
| B4 Residual-offset correction (K=5) | DBP | 7.951 | 0.311 | 0.270 | 10.925 | 43.72 | 71.20 | 85.99 | FAIL* | FAIL (Grade C)* |
| B5 Siamese delta (K=1) | SBP | 16.632 | 0.027 | -4.964 | 21.140 | 21.44 | 40.14 | 55.68 | FAIL* | FAIL (Grade D)* |
| B5 Siamese delta (K=1) | DBP | 9.027 | 0.108 | -1.951 | 12.279 | 38.83 | 66.42 | 82.17 | FAIL* | FAIL (Grade D)* |
| B6 Head-only adaptation (K=1) | SBP | 14.681 | 0.265 | -0.396 | 18.874 | 22.84 | 43.13 | 60.21 | FAIL* | FAIL (Grade D)* |
| B6 Head-only adaptation (K=1) | DBP | 8.300 | 0.280 | -0.101 | 11.169 | 40.54 | 68.45 | 84.82 | FAIL* | FAIL (Grade D)* |
| B6 Head-only adaptation (K=2) | SBP | 14.217 | 0.299 | 0.446 | 18.422 | 23.75 | 45.14 | 62.27 | FAIL* | FAIL (Grade D)* |
| B6 Head-only adaptation (K=2) | DBP | 8.116 | 0.299 | 0.085 | 11.021 | 42.07 | 69.67 | 85.32 | FAIL* | FAIL (Grade C)* |
| B6 Head-only adaptation (K=3) | SBP | 13.911 | 0.326 | 0.512 | 18.060 | 24.07 | 45.96 | 63.51 | FAIL* | FAIL (Grade D)* |
| B6 Head-only adaptation (K=3) | DBP | 7.869 | 0.334 | 0.146 | 10.744 | 43.60 | 71.24 | 86.34 | FAIL* | FAIL (Grade C)* |
| B6 Head-only adaptation (K=5) | SBP | 13.488 | 0.358 | 0.298 | 17.638 | 25.35 | 47.64 | 65.08 | FAIL* | FAIL (Grade D)* |
| B6 Head-only adaptation (K=5) | DBP | 7.657 | 0.367 | 0.172 | 10.477 | 44.27 | 72.44 | 87.62 | FAIL* | FAIL (Grade C)* |
| B7 Full-network adaptation (K=1) | SBP | 18.821 | -0.316 | 0.433 | 25.248 | 20.26 | 38.12 | 52.52 | FAIL* | FAIL (Grade D)* |
| B7 Full-network adaptation (K=1) | DBP | 10.698 | -0.251 | 0.503 | 14.719 | 35.03 | 59.26 | 75.27 | FAIL* | FAIL (Grade D)* |
| B7 Full-network adaptation (K=2) | SBP | 17.851 | -0.158 | 0.678 | 23.686 | 20.62 | 39.30 | 54.23 | FAIL* | FAIL (Grade D)* |
| B7 Full-network adaptation (K=2) | DBP | 10.108 | -0.078 | 0.457 | 13.661 | 36.06 | 60.83 | 77.14 | FAIL* | FAIL (Grade D)* |
| B7 Full-network adaptation (K=3) | SBP | 17.077 | -0.057 | 0.393 | 22.633 | 21.57 | 40.62 | 55.72 | FAIL* | FAIL (Grade D)* |
| B7 Full-network adaptation (K=3) | DBP | 9.700 | -0.004 | 0.264 | 13.192 | 37.70 | 62.84 | 78.58 | FAIL* | FAIL (Grade D)* |
| B7 Full-network adaptation (K=5) | SBP | 16.618 | -0.011 | 0.315 | 22.130 | 22.79 | 42.22 | 57.20 | FAIL* | FAIL (Grade D)* |
| B7 Full-network adaptation (K=5) | DBP | 9.448 | 0.036 | 0.269 | 12.924 | 38.83 | 64.26 | 79.47 | FAIL* | FAIL (Grade D)* |
| B8 LoRA adaptation (K=1) | SBP | 13.936 | 0.342 | -0.999 | 17.836 | 23.90 | 45.20 | 62.09 | FAIL* | FAIL (Grade D)* |
| B8 LoRA adaptation (K=1) | DBP | 7.827 | 0.352 | -0.506 | 10.589 | 42.19 | 71.47 | 87.43 | FAIL* | FAIL (Grade C)* |
| B8 LoRA adaptation (K=2) | SBP | 13.646 | 0.363 | -0.502 | 17.564 | 24.30 | 46.39 | 63.59 | FAIL* | FAIL (Grade D)* |
| B8 LoRA adaptation (K=2) | DBP | 7.731 | 0.362 | -0.434 | 10.504 | 42.85 | 72.09 | 87.71 | FAIL* | FAIL (Grade C)* |
| B8 LoRA adaptation (K=3) | SBP | 13.450 | 0.377 | -0.530 | 17.368 | 24.92 | 47.28 | 64.33 | FAIL* | FAIL (Grade D)* |
| B8 LoRA adaptation (K=3) | DBP | 7.618 | 0.378 | -0.309 | 10.376 | 43.59 | 72.74 | 88.09 | FAIL* | FAIL (Grade C)* |
| B8 LoRA adaptation (K=5) | SBP | 13.139 | 0.399 | -0.602 | 17.049 | 25.93 | 48.31 | 65.41 | FAIL* | FAIL (Grade D)* |
| B8 LoRA adaptation (K=5) | DBP | 7.468 | 0.396 | -0.246 | 10.231 | 44.70 | 73.67 | 88.61 | FAIL* | FAIL (Grade C)* |
| M0 Variable-K residual anchor (K=1) | SBP | 13.606 | 0.333 | 1.526 | 17.910 | 26.00 | 48.38 | 65.18 | FAIL* | FAIL (Grade D)* |
| M0 Variable-K residual anchor (K=1) | DBP | 7.884 | 0.326 | 0.950 | 10.768 | 43.59 | 71.28 | 86.34 | FAIL* | FAIL (Grade C)* |
| M0 Variable-K residual anchor (K=2) | SBP | 13.131 | 0.370 | 1.920 | 17.375 | 27.60 | 50.28 | 66.38 | FAIL* | FAIL (Grade D)* |
| M0 Variable-K residual anchor (K=2) | DBP | 7.707 | 0.344 | 0.963 | 10.620 | 45.28 | 72.91 | 86.79 | FAIL* | FAIL (Grade C)* |
| M0 Variable-K residual anchor (K=3) | SBP | 12.780 | 0.401 | 1.956 | 16.919 | 28.16 | 51.31 | 67.61 | FAIL* | FAIL (Grade D)* |
| M0 Variable-K residual anchor (K=3) | DBP | 7.531 | 0.368 | 1.021 | 10.414 | 46.35 | 74.04 | 87.23 | FAIL* | FAIL (Grade C)* |
| M0 Variable-K residual anchor (K=5) | SBP | 12.407 | 0.431 | 1.711 | 16.519 | 29.12 | 52.58 | 69.03 | FAIL* | FAIL (Grade D)* |
| M0 Variable-K residual anchor (K=5) | DBP | 7.275 | 0.401 | 0.927 | 10.146 | 47.94 | 75.33 | 87.94 | FAIL* | FAIL (Grade C)* |
| M1 M0 + FiLM (K=1) | SBP | 14.014 | 0.302 | 0.131 | 18.398 | 25.02 | 46.83 | 63.64 | FAIL* | FAIL (Grade D)* |
| M1 M0 + FiLM (K=1) | DBP | 8.132 | 0.291 | 0.500 | 11.078 | 42.41 | 69.99 | 85.18 | FAIL* | FAIL (Grade C)* |
| M1 M0 + FiLM (K=2) | SBP | 13.316 | 0.361 | 0.579 | 17.582 | 26.64 | 49.04 | 65.69 | FAIL* | FAIL (Grade D)* |
| M1 M0 + FiLM (K=2) | DBP | 7.880 | 0.322 | 0.560 | 10.827 | 44.02 | 71.76 | 85.99 | FAIL* | FAIL (Grade C)* |
| M1 M0 + FiLM (K=3) | SBP | 12.922 | 0.397 | 0.578 | 17.082 | 27.23 | 50.25 | 67.03 | FAIL* | FAIL (Grade D)* |
| M1 M0 + FiLM (K=3) | DBP | 7.694 | 0.350 | 0.592 | 10.597 | 45.14 | 72.72 | 86.72 | FAIL* | FAIL (Grade C)* |
| M1 M0 + FiLM (K=5) | SBP | 12.528 | 0.425 | 0.297 | 16.687 | 28.67 | 51.93 | 68.41 | FAIL* | FAIL (Grade D)* |
| M1 M0 + FiLM (K=5) | DBP | 7.423 | 0.385 | 0.489 | 10.311 | 46.76 | 74.27 | 87.63 | FAIL* | FAIL (Grade C)* |
| M2 M1 + reliability weighting (K=1) | SBP | 14.266 | 0.263 | 1.378 | 18.843 | 25.40 | 47.26 | 63.30 | FAIL* | FAIL (Grade D)* |
| M2 M1 + reliability weighting (K=1) | DBP | 8.315 | 0.245 | 0.795 | 11.411 | 42.10 | 69.22 | 84.76 | FAIL* | FAIL (Grade D)* |
| M2 M1 + reliability weighting (K=2) | SBP | 13.635 | 0.316 | 1.502 | 18.144 | 26.86 | 49.30 | 65.41 | FAIL* | FAIL (Grade D)* |
| M2 M1 + reliability weighting (K=2) | DBP | 7.986 | 0.295 | 0.691 | 11.031 | 43.91 | 71.50 | 85.83 | FAIL* | FAIL (Grade C)* |
| M2 M1 + reliability weighting (K=3) | SBP | 13.142 | 0.362 | 1.364 | 17.531 | 27.82 | 50.69 | 66.92 | FAIL* | FAIL (Grade D)* |
| M2 M1 + reliability weighting (K=3) | DBP | 7.692 | 0.338 | 0.625 | 10.697 | 45.69 | 73.06 | 86.94 | FAIL* | FAIL (Grade C)* |
| M2 M1 + reliability weighting (K=5) | SBP | 12.669 | 0.396 | 1.098 | 17.067 | 29.31 | 52.47 | 68.64 | FAIL* | FAIL (Grade D)* |
| M2 M1 + reliability weighting (K=5) | DBP | 7.437 | 0.372 | 0.492 | 10.422 | 47.36 | 74.23 | 87.92 | FAIL* | FAIL (Grade C)* |

The machine-readable CSV also contains participant-macro MAE, participant 
and event counts, method/K identifiers, aggregation scope, and the formal 
standards limitation.
