# Official CalBased frozen test results

Exact official same-subject randomized membership; 360 labelled training windows and 40 test windows per person.
Participant-macro MAE is primary. Pooled AAMI/BHS fields are retrospective numerical screens, not clinical device certification.
All methods and predictions were frozen before the test labels were joined. No candidate is promoted or selected by this scorer.

## Source-duplicate disclosure

Exact official membership is retained, including verified source duplicates. Internal validation and OOF are index-disjoint, not strictly waveform-content-disjoint. No official test target row is accessed for fitting or adaptation.
35 official test windows repeat selected training PPG content. These and all other official members are retained; no deduplicated subset is substituted.
The predeclared inner splits are unchanged; their known content duplicates are retained and internal/OOF scores must not be described as strictly content-independent.

## Overall

### Participant-macro MAE (mmHg)

| Setting | SBP MAE | DBP MAE | Mean MAE | Subjects | Windows |
|---|---:|---:|---:|---:|---:|
| lora | 3.8207 | 2.0994 | 2.9601 | 2506 | 100240 |
| fixed | 3.4210 | 1.8858 | 2.6534 | 2506 | 100240 |
| v1 | 3.4210 | 1.8858 | 2.6534 | 2506 | 100240 |
| e1 | 3.4210 | 1.8858 | 2.6534 | 2506 | 100240 |
| trust_scalar | 3.4210 | 1.8858 | 2.6534 | 2506 | 100240 |
| trust_bp | 3.4210 | 1.8858 | 2.6534 | 2506 | 100240 |

### Window-pooled numerical diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora | SBP | 3.8207 | 0.9237 | 0.1978 | 5.7986 | 75.4030 | 93.0357 | 97.4631 | PASS* | PASS (Grade A)* |
| lora | DBP | 2.0994 | 0.9185 | 0.0296 | 3.6076 | 91.5623 | 98.3679 | 99.4094 | PASS* | PASS (Grade A)* |
| fixed | SBP | 3.4210 | 0.9318 | -0.0451 | 5.4852 | 79.5531 | 93.8627 | 97.6397 | PASS* | PASS (Grade A)* |
| fixed | DBP | 1.8858 | 0.9255 | -0.0512 | 3.4493 | 92.5648 | 98.3669 | 99.3755 | PASS* | PASS (Grade A)* |
| v1 | SBP | 3.4210 | 0.9318 | -0.0451 | 5.4852 | 79.5531 | 93.8627 | 97.6397 | PASS* | PASS (Grade A)* |
| v1 | DBP | 1.8858 | 0.9255 | -0.0512 | 3.4493 | 92.5648 | 98.3669 | 99.3755 | PASS* | PASS (Grade A)* |
| e1 | SBP | 3.4210 | 0.9318 | -0.0451 | 5.4852 | 79.5531 | 93.8627 | 97.6397 | PASS* | PASS (Grade A)* |
| e1 | DBP | 1.8858 | 0.9255 | -0.0512 | 3.4493 | 92.5648 | 98.3669 | 99.3755 | PASS* | PASS (Grade A)* |
| trust_scalar | SBP | 3.4210 | 0.9318 | -0.0451 | 5.4852 | 79.5531 | 93.8627 | 97.6397 | PASS* | PASS (Grade A)* |
| trust_scalar | DBP | 1.8858 | 0.9255 | -0.0512 | 3.4493 | 92.5648 | 98.3669 | 99.3755 | PASS* | PASS (Grade A)* |
| trust_bp | SBP | 3.4210 | 0.9318 | -0.0451 | 5.4852 | 79.5531 | 93.8627 | 97.6397 | PASS* | PASS (Grade A)* |
| trust_bp | DBP | 1.8858 | 0.9255 | -0.0512 | 3.4493 | 92.5648 | 98.3669 | 99.3755 | PASS* | PASS (Grade A)* |

## MIMIC

### Participant-macro MAE (mmHg)

| Setting | SBP MAE | DBP MAE | Mean MAE | Subjects | Windows |
|---|---:|---:|---:|---:|---:|
| lora | 4.1920 | 2.2802 | 3.2361 | 1213 | 48520 |
| fixed | 3.8073 | 2.0699 | 2.9386 | 1213 | 48520 |
| v1 | 3.8073 | 2.0699 | 2.9386 | 1213 | 48520 |
| e1 | 3.8073 | 2.0699 | 2.9386 | 1213 | 48520 |
| trust_scalar | 3.8073 | 2.0699 | 2.9386 | 1213 | 48520 |
| trust_bp | 3.8073 | 2.0699 | 2.9386 | 1213 | 48520 |

### Window-pooled numerical diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora | SBP | 4.1920 | 0.9215 | 0.2407 | 6.3309 | 72.1764 | 91.4138 | 96.6488 | PASS* | PASS (Grade A)* |
| lora | DBP | 2.2802 | 0.9056 | -0.0423 | 4.0314 | 90.2288 | 97.6979 | 99.0334 | PASS* | PASS (Grade A)* |
| fixed | SBP | 3.8073 | 0.9296 | -0.0575 | 6.0005 | 76.2758 | 92.3743 | 96.8982 | PASS* | PASS (Grade A)* |
| fixed | DBP | 2.0699 | 0.9134 | -0.0809 | 3.8611 | 91.1974 | 97.7226 | 99.0272 | PASS* | PASS (Grade A)* |
| v1 | SBP | 3.8073 | 0.9296 | -0.0575 | 6.0005 | 76.2758 | 92.3743 | 96.8982 | PASS* | PASS (Grade A)* |
| v1 | DBP | 2.0699 | 0.9134 | -0.0809 | 3.8611 | 91.1974 | 97.7226 | 99.0272 | PASS* | PASS (Grade A)* |
| e1 | SBP | 3.8073 | 0.9296 | -0.0575 | 6.0005 | 76.2758 | 92.3743 | 96.8982 | PASS* | PASS (Grade A)* |
| e1 | DBP | 2.0699 | 0.9134 | -0.0809 | 3.8611 | 91.1974 | 97.7226 | 99.0272 | PASS* | PASS (Grade A)* |
| trust_scalar | SBP | 3.8073 | 0.9296 | -0.0575 | 6.0005 | 76.2758 | 92.3743 | 96.8982 | PASS* | PASS (Grade A)* |
| trust_scalar | DBP | 2.0699 | 0.9134 | -0.0809 | 3.8611 | 91.1974 | 97.7226 | 99.0272 | PASS* | PASS (Grade A)* |
| trust_bp | SBP | 3.8073 | 0.9296 | -0.0575 | 6.0005 | 76.2758 | 92.3743 | 96.8982 | PASS* | PASS (Grade A)* |
| trust_bp | DBP | 2.0699 | 0.9134 | -0.0809 | 3.8611 | 91.1974 | 97.7226 | 99.0272 | PASS* | PASS (Grade A)* |

## VitalDB

### Participant-macro MAE (mmHg)

| Setting | SBP MAE | DBP MAE | Mean MAE | Subjects | Windows |
|---|---:|---:|---:|---:|---:|
| lora | 3.4724 | 1.9298 | 2.7011 | 1293 | 51720 |
| fixed | 3.0587 | 1.7132 | 2.3860 | 1293 | 51720 |
| v1 | 3.0587 | 1.7132 | 2.3860 | 1293 | 51720 |
| e1 | 3.0587 | 1.7132 | 2.3860 | 1293 | 51720 |
| trust_scalar | 3.0587 | 1.7132 | 2.3860 | 1293 | 51720 |
| trust_bp | 3.0587 | 1.7132 | 2.3860 | 1293 | 51720 |

### Window-pooled numerical diagnostics

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora | SBP | 3.4724 | 0.9223 | 0.1577 | 5.2500 | 78.4300 | 94.5572 | 98.2270 | PASS* | PASS (Grade A)* |
| lora | DBP | 1.9298 | 0.9315 | 0.0971 | 3.1573 | 92.8132 | 98.9965 | 99.7622 | PASS* | PASS (Grade A)* |
| fixed | SBP | 3.0587 | 0.9309 | -0.0334 | 4.9533 | 82.6276 | 95.2591 | 98.3353 | PASS* | PASS (Grade A)* |
| fixed | DBP | 1.7132 | 0.9377 | -0.0233 | 3.0120 | 93.8476 | 98.9714 | 99.7022 | PASS* | PASS (Grade A)* |
| v1 | SBP | 3.0587 | 0.9309 | -0.0334 | 4.9533 | 82.6276 | 95.2591 | 98.3353 | PASS* | PASS (Grade A)* |
| v1 | DBP | 1.7132 | 0.9377 | -0.0233 | 3.0120 | 93.8476 | 98.9714 | 99.7022 | PASS* | PASS (Grade A)* |
| e1 | SBP | 3.0587 | 0.9309 | -0.0334 | 4.9533 | 82.6276 | 95.2591 | 98.3353 | PASS* | PASS (Grade A)* |
| e1 | DBP | 1.7132 | 0.9377 | -0.0233 | 3.0120 | 93.8476 | 98.9714 | 99.7022 | PASS* | PASS (Grade A)* |
| trust_scalar | SBP | 3.0587 | 0.9309 | -0.0334 | 4.9533 | 82.6276 | 95.2591 | 98.3353 | PASS* | PASS (Grade A)* |
| trust_scalar | DBP | 1.7132 | 0.9377 | -0.0233 | 3.0120 | 93.8476 | 98.9714 | 99.7022 | PASS* | PASS (Grade A)* |
| trust_bp | SBP | 3.0587 | 0.9309 | -0.0334 | 4.9533 | 82.6276 | 95.2591 | 98.3353 | PASS* | PASS (Grade A)* |
| trust_bp | DBP | 1.7132 | 0.9377 | -0.0233 | 3.0120 | 93.8476 | 98.9714 | 99.7022 | PASS* | PASS (Grade A)* |
