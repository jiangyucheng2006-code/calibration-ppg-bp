# Remaining-population frozen benchmark

Protocol: post-enrollment-200-v1. Participant-macro MAE is primary.
The threshold fields below are retrospective numerical AAMI/BHS screens, not device certification.
All windows and participants are retained. Each profile uses 360 labelled registration windows, not 360 independent cuff events.
Random-window evaluation does not establish chronological or long-term performance.

Source-duplicate audit: `{"test_rows_with_registration_content": 34}`

## Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_lora | SBP | 3.8672 | 0.9241 | 0.7391 | 5.7385 | 74.7733 | 93.0586 | 97.5033 | PASS* | PASS (Grade A)* |
| population_lora | DBP | 2.1512 | 0.9158 | 0.3703 | 3.6431 | 91.5423 | 98.3200 | 99.4197 | PASS* | PASS (Grade A)* |

## MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_lora | SBP | 4.1996 | 0.9227 | 0.6624 | 6.2544 | 71.8705 | 91.5108 | 96.7244 | PASS* | PASS (Grade A)* |
| population_lora | DBP | 2.3116 | 0.9030 | 0.2486 | 4.0629 | 90.2585 | 97.6192 | 99.0760 | PASS* | PASS (Grade A)* |

## VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_lora | SBP | 3.5574 | 0.9218 | 0.8106 | 5.2109 | 77.4790 | 94.5013 | 98.2293 | PASS* | PASS (Grade A)* |
| population_lora | DBP | 2.0016 | 0.9282 | 0.4838 | 3.1986 | 92.7389 | 98.9732 | 99.7402 | PASS* | PASS (Grade A)* |
