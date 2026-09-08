# Personal-memory v2: complete aggregate result tables

Registered-user internal development, seed 20260907; 320 labelled 10-second train windows per person. No held-out evaluation.

Participant-macro MAE is primary. Overall uses the original full-cohort aggregate, not an equal average of sources. MIMIC and VitalDB are internal source strata, not independent external validation.

CSV files retain full precision. All MAE, ME and STD values are mmHg; threshold columns are percentages. Pooled MAE equals participant-macro MAE here because each participant contributes exactly 40 queries.

AAMI/BHS labels marked * are retrospective numerical screens only, not clinical certification, formal device-validation passes or evidence of clinical validity.

## random_disjoint

### Overall

Primary participant-macro results

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | 2051 | 82040 | 3.691313 | 2.068883 | 2.880098 |
| fixed_blend | 2051 | 82040 | 3.391967 | 1.903079 | 2.647523 |
| trained_blend | 2051 | 82040 | 3.391967 | 1.903079 | 2.647523 |
| E3_zero_reliability | 2051 | 82040 | 3.384374 | 1.891913 | 2.638143 |
| E3_trained_reliability | 2051 | 82040 | 3.384374 | 1.891913 | 2.638143 |
| E1_matched_relation | 2051 | 82040 | 3.391967 | 1.903079 | 2.647523 |
| E2_bp_state_metric | 2051 | 82040 | 3.476531 | 1.948570 | 2.712551 |
| lora_continued_control | 2051 | 82040 | 3.685294 | 2.067514 | 2.876404 |

Secondary window-pooled numerical diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 3.691313 | 0.930294 | 0.022515 | 5.587610 | 76.182350 | 93.490980 | 97.737689 | PASS\* | PASS (Grade A)\* |
| D0_frozen_lora | DBP | 2.068883 | 0.925724 | -0.118263 | 3.493801 | 91.868601 | 98.319113 | 99.414920 | PASS\* | PASS (Grade A)\* |
| fixed_blend | SBP | 3.391966 | 0.936049 | -0.090207 | 5.351292 | 79.522184 | 94.001706 | 97.801073 | PASS\* | PASS (Grade A)\* |
| fixed_blend | DBP | 1.903079 | 0.930238 | -0.102790 | 3.386355 | 92.597513 | 98.278888 | 99.367382 | PASS\* | PASS (Grade A)\* |
| trained_blend | SBP | 3.391966 | 0.936049 | -0.090207 | 5.351292 | 79.522184 | 94.001706 | 97.801073 | PASS\* | PASS (Grade A)\* |
| trained_blend | DBP | 1.903079 | 0.930238 | -0.102790 | 3.386355 | 92.597513 | 98.278888 | 99.367382 | PASS\* | PASS (Grade A)\* |
| E3_zero_reliability | SBP | 3.384374 | 0.936596 | -0.073838 | 5.328598 | 79.524622 | 94.072404 | 97.843735 | PASS\* | PASS (Grade A)\* |
| E3_zero_reliability | DBP | 1.891913 | 0.931441 | -0.126497 | 3.356209 | 92.663335 | 98.362994 | 99.423452 | PASS\* | PASS (Grade A)\* |
| E3_trained_reliability | SBP | 3.384374 | 0.936596 | -0.073838 | 5.328598 | 79.524622 | 94.072404 | 97.843735 | PASS\* | PASS (Grade A)\* |
| E3_trained_reliability | DBP | 1.891913 | 0.931441 | -0.126497 | 3.356209 | 92.663335 | 98.362994 | 99.423452 | PASS\* | PASS (Grade A)\* |
| E1_matched_relation | SBP | 3.391966 | 0.936049 | -0.090207 | 5.351292 | 79.522184 | 94.001706 | 97.801073 | PASS\* | PASS (Grade A)\* |
| E1_matched_relation | DBP | 1.903079 | 0.930238 | -0.102790 | 3.386355 | 92.597513 | 98.278888 | 99.367382 | PASS\* | PASS (Grade A)\* |
| E2_bp_state_metric | SBP | 3.476531 | 0.934008 | -0.062973 | 5.436400 | 78.601902 | 93.733545 | 97.746221 | PASS\* | PASS (Grade A)\* |
| E2_bp_state_metric | DBP | 1.948571 | 0.928509 | -0.097365 | 3.428274 | 92.226962 | 98.249634 | 99.383228 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 3.685294 | 0.930408 | 0.024492 | 5.583041 | 76.253047 | 93.501950 | 97.727938 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | DBP | 2.067514 | 0.925668 | -0.101935 | 3.495646 | 91.858849 | 98.300829 | 99.410044 | PASS\* | PASS (Grade A)\* |

### MIMIC

Primary participant-macro results

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | 1011 | 40440 | 4.043434 | 2.245703 | 3.144569 |
| fixed_blend | 1011 | 40440 | 3.770226 | 2.084702 | 2.927464 |
| trained_blend | 1011 | 40440 | 3.770226 | 2.084702 | 2.927464 |
| E3_zero_reliability | 1011 | 40440 | 3.760804 | 2.072009 | 2.916407 |
| E3_trained_reliability | 1011 | 40440 | 3.760804 | 2.072009 | 2.916407 |
| E1_matched_relation | 1011 | 40440 | 3.770226 | 2.084702 | 2.927464 |
| E2_bp_state_metric | 1011 | 40440 | 3.845250 | 2.127265 | 2.986258 |
| lora_continued_control | 1011 | 40440 | 4.037932 | 2.243411 | 3.140671 |

Secondary window-pooled numerical diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 4.043434 | 0.927102 | 0.046850 | 6.144751 | 73.212166 | 91.886746 | 96.980712 | PASS\* | PASS (Grade A)\* |
| D0_frozen_lora | DBP | 2.245703 | 0.910477 | -0.187439 | 3.973828 | 90.378338 | 97.670623 | 99.065282 | PASS\* | PASS (Grade A)\* |
| fixed_blend | SBP | 3.770226 | 0.932539 | -0.089403 | 5.910677 | 76.305638 | 92.467854 | 97.082097 | PASS\* | PASS (Grade A)\* |
| fixed_blend | DBP | 2.084703 | 0.915691 | -0.138515 | 3.858170 | 91.214144 | 97.660732 | 99.038081 | PASS\* | PASS (Grade A)\* |
| trained_blend | SBP | 3.770226 | 0.932539 | -0.089403 | 5.910677 | 76.305638 | 92.467854 | 97.082097 | PASS\* | PASS (Grade A)\* |
| trained_blend | DBP | 2.084703 | 0.915691 | -0.138515 | 3.858170 | 91.214144 | 97.660732 | 99.038081 | PASS\* | PASS (Grade A)\* |
| E3_zero_reliability | SBP | 3.760804 | 0.932986 | -0.059318 | 5.891425 | 76.290801 | 92.581602 | 97.148863 | PASS\* | PASS (Grade A)\* |
| E3_zero_reliability | DBP | 2.072009 | 0.916626 | -0.177743 | 3.835071 | 91.278437 | 97.769535 | 99.097428 | PASS\* | PASS (Grade A)\* |
| E3_trained_reliability | SBP | 3.760804 | 0.932986 | -0.059318 | 5.891425 | 76.290801 | 92.581602 | 97.148863 | PASS\* | PASS (Grade A)\* |
| E3_trained_reliability | DBP | 2.072009 | 0.916626 | -0.177743 | 3.835071 | 91.278437 | 97.769535 | 99.097428 | PASS\* | PASS (Grade A)\* |
| E1_matched_relation | SBP | 3.770226 | 0.932539 | -0.089403 | 5.910677 | 76.305638 | 92.467854 | 97.082097 | PASS\* | PASS (Grade A)\* |
| E1_matched_relation | DBP | 2.084703 | 0.915691 | -0.138515 | 3.858170 | 91.214144 | 97.660732 | 99.038081 | PASS\* | PASS (Grade A)\* |
| E2_bp_state_metric | SBP | 3.845250 | 0.930832 | -0.051798 | 5.985434 | 75.427794 | 92.232938 | 97.052423 | PASS\* | PASS (Grade A)\* |
| E2_bp_state_metric | DBP | 2.127265 | 0.914119 | -0.129385 | 3.894329 | 90.793769 | 97.633531 | 99.062809 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 4.037932 | 0.927178 | 0.086192 | 6.141114 | 73.323442 | 91.889219 | 96.960930 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | DBP | 2.243411 | 0.910320 | -0.164002 | 3.978350 | 90.398121 | 97.660732 | 99.052918 | PASS\* | PASS (Grade A)\* |

### VitalDB

Primary participant-macro results

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | 1040 | 41600 | 3.349010 | 1.896994 | 2.623002 |
| fixed_blend | 1040 | 41600 | 3.024254 | 1.726520 | 2.375387 |
| trained_blend | 1040 | 41600 | 3.024254 | 1.726520 | 2.375387 |
| E3_zero_reliability | 1040 | 41600 | 3.018439 | 1.716840 | 2.367640 |
| E3_trained_reliability | 1040 | 41600 | 3.018439 | 1.716840 | 2.367640 |
| E1_matched_relation | 1040 | 41600 | 3.024254 | 1.726520 | 2.375387 |
| E2_bp_state_metric | 1040 | 41600 | 3.118094 | 1.774859 | 2.446476 |
| lora_continued_control | 1040 | 41600 | 3.342489 | 1.896523 | 2.619506 |

Secondary window-pooled numerical diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 3.349010 | 0.930708 | -0.001141 | 4.986653 | 79.069712 | 95.050481 | 98.473558 | PASS\* | PASS (Grade A)\* |
| D0_frozen_lora | DBP | 1.896994 | 0.941938 | -0.051016 | 2.951790 | 93.317308 | 98.949519 | 99.754808 | PASS\* | PASS (Grade A)\* |
| fixed_blend | SBP | 3.024254 | 0.937244 | -0.090989 | 4.744776 | 82.649038 | 95.492788 | 98.500000 | PASS\* | PASS (Grade A)\* |
| fixed_blend | DBP | 1.726520 | 0.945726 | -0.068061 | 2.853500 | 93.942308 | 98.879808 | 99.687500 | PASS\* | PASS (Grade A)\* |
| trained_blend | SBP | 3.024254 | 0.937244 | -0.090989 | 4.744776 | 82.649038 | 95.492788 | 98.500000 | PASS\* | PASS (Grade A)\* |
| trained_blend | DBP | 1.726520 | 0.945726 | -0.068061 | 2.853500 | 93.942308 | 98.879808 | 99.687500 | PASS\* | PASS (Grade A)\* |
| E3_zero_reliability | SBP | 3.018439 | 0.937963 | -0.087952 | 4.717563 | 82.668269 | 95.521635 | 98.519231 | PASS\* | PASS (Grade A)\* |
| E3_zero_reliability | DBP | 1.716840 | 0.947254 | -0.076681 | 2.812788 | 94.009615 | 98.939904 | 99.740385 | PASS\* | PASS (Grade A)\* |
| E3_trained_reliability | SBP | 3.018439 | 0.937963 | -0.087952 | 4.717563 | 82.668269 | 95.521635 | 98.519231 | PASS\* | PASS (Grade A)\* |
| E3_trained_reliability | DBP | 1.716840 | 0.947254 | -0.076681 | 2.812788 | 94.009615 | 98.939904 | 99.740385 | PASS\* | PASS (Grade A)\* |
| E1_matched_relation | SBP | 3.024254 | 0.937244 | -0.090989 | 4.744776 | 82.649038 | 95.492788 | 98.500000 | PASS\* | PASS (Grade A)\* |
| E1_matched_relation | DBP | 1.726520 | 0.945726 | -0.068061 | 2.853500 | 93.942308 | 98.879808 | 99.687500 | PASS\* | PASS (Grade A)\* |
| E2_bp_state_metric | SBP | 3.118094 | 0.934616 | -0.073837 | 4.843430 | 81.687500 | 95.192308 | 98.420673 | PASS\* | PASS (Grade A)\* |
| E2_bp_state_metric | DBP | 1.774859 | 0.943786 | -0.066238 | 2.904102 | 93.620192 | 98.848558 | 99.694712 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 3.342488 | 0.930881 | -0.035487 | 4.980291 | 79.100962 | 95.069712 | 98.473558 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | DBP | 1.896523 | 0.941995 | -0.041598 | 2.950480 | 93.278846 | 98.923077 | 99.757212 | PASS\* | PASS (Grade A)\* |

## chronological_blocked

### Overall

Primary participant-macro results

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | 2051 | 82040 | 4.777302 | 2.647466 | 3.712384 |
| fixed_blend | 2051 | 82040 | 4.897696 | 2.695645 | 3.796670 |
| trained_blend | 2051 | 82040 | 4.700025 | 2.614252 | 3.657139 |
| E3_zero_reliability | 2051 | 82040 | 4.846354 | 2.664681 | 3.755517 |
| E3_trained_reliability | 2051 | 82040 | 4.687384 | 2.602118 | 3.644751 |
| E1_matched_relation | 2051 | 82040 | 4.673410 | 2.593442 | 3.633426 |
| E2_bp_state_metric | 2051 | 82040 | 4.879563 | 2.696717 | 3.788140 |
| lora_continued_control | 2051 | 82040 | 4.777301 | 2.647466 | 3.712384 |

Secondary window-pooled numerical diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 4.777302 | 0.880409 | 0.206015 | 7.105434 | 67.206241 | 88.133837 | 95.173086 | PASS\* | PASS (Grade A)\* |
| D0_frozen_lora | DBP | 2.647466 | 0.876220 | 0.064872 | 4.375718 | 85.860556 | 96.761336 | 98.895661 | PASS\* | PASS (Grade A)\* |
| fixed_blend | SBP | 4.897695 | 0.873003 | 0.278732 | 7.319925 | 66.418820 | 87.512189 | 94.779376 | PASS\* | PASS (Grade B)\* |
| fixed_blend | DBP | 2.695645 | 0.869648 | 0.072473 | 4.490286 | 85.485129 | 96.368844 | 98.801804 | PASS\* | PASS (Grade A)\* |
| trained_blend | SBP | 4.700025 | 0.880491 | 0.189632 | 7.103450 | 68.057045 | 88.300829 | 95.120673 | PASS\* | PASS (Grade A)\* |
| trained_blend | DBP | 2.614252 | 0.875705 | 0.059094 | 4.384882 | 85.917845 | 96.582155 | 98.896880 | PASS\* | PASS (Grade A)\* |
| E3_zero_reliability | SBP | 4.846354 | 0.875735 | 0.270806 | 7.240937 | 66.795466 | 87.725500 | 94.936616 | PASS\* | PASS (Grade B)\* |
| E3_zero_reliability | DBP | 2.664681 | 0.873027 | 0.057757 | 4.431894 | 85.697221 | 96.491955 | 98.870063 | PASS\* | PASS (Grade A)\* |
| E3_trained_reliability | SBP | 4.687384 | 0.881381 | 0.196301 | 7.076770 | 68.152121 | 88.347148 | 95.166992 | PASS\* | PASS (Grade A)\* |
| E3_trained_reliability | DBP | 2.602118 | 0.877322 | 0.048771 | 4.356392 | 86.023891 | 96.644320 | 98.929790 | PASS\* | PASS (Grade A)\* |
| E1_matched_relation | SBP | 4.673411 | 0.881218 | 0.192053 | 7.081748 | 68.439785 | 88.415407 | 95.162116 | PASS\* | PASS (Grade A)\* |
| E1_matched_relation | DBP | 2.593442 | 0.876196 | 0.052805 | 4.376299 | 86.104339 | 96.552901 | 98.884690 | PASS\* | PASS (Grade A)\* |
| E2_bp_state_metric | SBP | 4.879563 | 0.873290 | 0.250369 | 7.312668 | 66.593125 | 87.570697 | 94.807411 | PASS\* | PASS (Grade B)\* |
| E2_bp_state_metric | DBP | 2.696717 | 0.868993 | 0.044520 | 4.501919 | 85.442467 | 96.370063 | 98.805461 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 4.777302 | 0.880409 | 0.206015 | 7.105434 | 67.206241 | 88.133837 | 95.173086 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | DBP | 2.647466 | 0.876220 | 0.064872 | 4.375718 | 85.860556 | 96.761336 | 98.895661 | PASS\* | PASS (Grade A)\* |

### MIMIC

Primary participant-macro results

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | 1011 | 40440 | 4.511091 | 2.443512 | 3.477302 |
| fixed_blend | 1011 | 40440 | 4.543567 | 2.455852 | 3.499710 |
| trained_blend | 1011 | 40440 | 4.428385 | 2.420707 | 3.424546 |
| E3_zero_reliability | 1011 | 40440 | 4.504903 | 2.428512 | 3.466707 |
| E3_trained_reliability | 1011 | 40440 | 4.412135 | 2.402018 | 3.407077 |
| E1_matched_relation | 1011 | 40440 | 4.399313 | 2.399911 | 3.399612 |
| E2_bp_state_metric | 1011 | 40440 | 4.547395 | 2.473463 | 3.510429 |
| lora_continued_control | 1011 | 40440 | 4.511091 | 2.443512 | 3.477302 |

Secondary window-pooled numerical diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 4.511091 | 0.907683 | 0.162909 | 6.883518 | 70.422849 | 89.364491 | 95.501978 | PASS\* | PASS (Grade A)\* |
| D0_frozen_lora | DBP | 2.443512 | 0.892512 | 0.016900 | 4.237960 | 87.977250 | 96.923838 | 98.805638 | PASS\* | PASS (Grade A)\* |
| fixed_blend | SBP | 4.543567 | 0.903903 | 0.149225 | 7.023390 | 70.272008 | 89.191395 | 95.462413 | PASS\* | PASS (Grade A)\* |
| fixed_blend | DBP | 2.455852 | 0.887440 | 0.001814 | 4.336824 | 87.925321 | 96.597428 | 98.741345 | PASS\* | PASS (Grade A)\* |
| trained_blend | SBP | 4.428385 | 0.907481 | 0.129063 | 6.891772 | 71.092977 | 89.530168 | 95.566271 | PASS\* | PASS (Grade A)\* |
| trained_blend | DBP | 2.420706 | 0.891077 | 0.023108 | 4.266120 | 87.895648 | 96.708704 | 98.805638 | PASS\* | PASS (Grade A)\* |
| E3_zero_reliability | SBP | 4.504903 | 0.905759 | 0.156501 | 6.955052 | 70.573689 | 89.285361 | 95.578635 | PASS\* | PASS (Grade A)\* |
| E3_zero_reliability | DBP | 2.428512 | 0.890814 | -0.017349 | 4.271304 | 88.034125 | 96.698813 | 98.818002 | PASS\* | PASS (Grade A)\* |
| E3_trained_reliability | SBP | 4.412135 | 0.908484 | 0.143822 | 6.853978 | 71.243818 | 89.619189 | 95.640455 | PASS\* | PASS (Grade A)\* |
| E3_trained_reliability | DBP | 2.402018 | 0.893367 | 0.003758 | 4.221087 | 88.014342 | 96.772997 | 98.864985 | PASS\* | PASS (Grade A)\* |
| E1_matched_relation | SBP | 4.399314 | 0.908070 | 0.116024 | 6.870017 | 71.540554 | 89.742829 | 95.610781 | PASS\* | PASS (Grade A)\* |
| E1_matched_relation | DBP | 2.399911 | 0.891458 | 0.011953 | 4.258700 | 88.081108 | 96.706231 | 98.798220 | PASS\* | PASS (Grade A)\* |
| E2_bp_state_metric | SBP | 4.547396 | 0.903591 | 0.141811 | 7.034947 | 70.383284 | 89.139466 | 95.373393 | PASS\* | PASS (Grade A)\* |
| E2_bp_state_metric | DBP | 2.473463 | 0.885396 | -0.023800 | 4.375958 | 87.717606 | 96.577646 | 98.748764 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 4.511091 | 0.907683 | 0.162909 | 6.883518 | 70.422849 | 89.364491 | 95.501978 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | DBP | 2.443512 | 0.892512 | 0.016900 | 4.237960 | 87.977250 | 96.923838 | 98.805638 | PASS\* | PASS (Grade A)\* |

### VitalDB

Primary participant-macro results

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | 1040 | 41600 | 5.036089 | 2.845733 | 3.940911 |
| fixed_blend | 1040 | 41600 | 5.241949 | 2.928752 | 4.085351 |
| trained_blend | 1040 | 41600 | 4.964090 | 2.802401 | 3.883245 |
| E3_zero_reliability | 1040 | 41600 | 5.178284 | 2.894265 | 4.036274 |
| E3_trained_reliability | 1040 | 41600 | 4.954957 | 2.796637 | 3.875797 |
| E1_matched_relation | 1040 | 41600 | 4.939865 | 2.781576 | 3.860720 |
| E2_bp_state_metric | 1040 | 41600 | 5.202469 | 2.913745 | 4.058107 |
| lora_continued_control | 1040 | 41600 | 5.036088 | 2.845733 | 3.940911 |

Secondary window-pooled numerical diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 5.036089 | 0.830506 | 0.247919 | 7.314550 | 64.079327 | 86.937500 | 94.853365 | PASS\* | PASS (Grade B)\* |
| D0_frozen_lora | DBP | 2.845733 | 0.852992 | 0.111505 | 4.505160 | 83.802885 | 96.603365 | 98.983173 | PASS\* | PASS (Grade A)\* |
| fixed_blend | SBP | 5.241949 | 0.816947 | 0.404627 | 7.595073 | 62.673077 | 85.879808 | 94.115385 | PASS\* | PASS (Grade B)\* |
| fixed_blend | DBP | 2.928752 | 0.844440 | 0.141162 | 4.633619 | 83.112981 | 96.146635 | 98.860577 | PASS\* | PASS (Grade A)\* |
| trained_blend | SBP | 4.964089 | 0.831041 | 0.248512 | 7.302949 | 65.105769 | 87.105769 | 94.687500 | PASS\* | PASS (Grade B)\* |
| trained_blend | DBP | 2.802401 | 0.853543 | 0.094077 | 4.497101 | 83.995192 | 96.459135 | 98.985577 | PASS\* | PASS (Grade A)\* |
| E3_zero_reliability | SBP | 5.178284 | 0.821220 | 0.381923 | 7.506840 | 63.122596 | 86.209135 | 94.312500 | PASS\* | PASS (Grade B)\* |
| E3_zero_reliability | DBP | 2.894265 | 0.847938 | 0.130768 | 4.581486 | 83.425481 | 96.290865 | 98.920673 | PASS\* | PASS (Grade A)\* |
| E3_trained_reliability | SBP | 4.954957 | 0.831801 | 0.247317 | 7.286543 | 65.146635 | 87.110577 | 94.706731 | PASS\* | PASS (Grade B)\* |
| E3_trained_reliability | DBP | 2.796637 | 0.854421 | 0.092528 | 4.483630 | 84.088942 | 96.519231 | 98.992788 | PASS\* | PASS (Grade A)\* |
| E1_matched_relation | SBP | 4.939865 | 0.832026 | 0.265962 | 7.280998 | 65.425481 | 87.125000 | 94.725962 | PASS\* | PASS (Grade B)\* |
| E1_matched_relation | DBP | 2.781576 | 0.854179 | 0.092518 | 4.487360 | 84.182692 | 96.403846 | 98.968750 | PASS\* | PASS (Grade A)\* |
| E2_bp_state_metric | SBP | 5.202469 | 0.818197 | 0.355899 | 7.571482 | 62.908654 | 86.045673 | 94.257212 | PASS\* | PASS (Grade B)\* |
| E2_bp_state_metric | DBP | 2.913745 | 0.845397 | 0.110934 | 4.620162 | 83.230769 | 96.168269 | 98.860577 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 5.036089 | 0.830506 | 0.247919 | 7.314550 | 64.079327 | 86.937500 | 94.853365 | PASS\* | PASS (Grade B)\* |
| lora_continued_control | DBP | 2.845733 | 0.852992 | 0.111505 | 4.505160 | 83.802885 | 96.603365 | 98.983173 | PASS\* | PASS (Grade A)\* |
