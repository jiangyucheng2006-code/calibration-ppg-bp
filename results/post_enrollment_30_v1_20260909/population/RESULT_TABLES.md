# Remaining-population frozen benchmark

Protocol: post-enrollment-30-v1. Participant-macro MAE is primary.
The threshold fields below are retrospective numerical AAMI/BHS screens, not device certification.
All windows and participants are retained. Each profile uses 360 labelled registration windows, not 360 independent cuff events.
Random-window evaluation does not establish chronological or long-term performance.

Source-duplicate audit: `{"test_rows_with_registration_content": 35}`

## Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_lora | SBP | 3.7752 | 0.9264 | -0.1215 | 5.6912 | 75.6300 | 93.2946 | 97.6171 | PASS* | PASS (Grade A)* |
| population_lora | DBP | 2.0810 | 0.9188 | -0.1887 | 3.5991 | 91.9315 | 98.4178 | 99.4285 | PASS* | PASS (Grade A)* |

## MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_lora | SBP | 4.1031 | 0.9250 | -0.0206 | 6.1890 | 72.9570 | 91.7446 | 96.8531 | PASS* | PASS (Grade A)* |
| population_lora | DBP | 2.2476 | 0.9060 | -0.2272 | 4.0266 | 90.6907 | 97.7567 | 99.0943 | PASS* | PASS (Grade A)* |

## VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_lora | SBP | 3.4679 | 0.9240 | -0.2161 | 5.1796 | 78.1358 | 94.7477 | 98.3333 | PASS* | PASS (Grade A)* |
| population_lora | DBP | 1.9248 | 0.9317 | -0.1526 | 3.1455 | 93.0947 | 99.0376 | 99.7418 | PASS* | PASS (Grade A)* |
