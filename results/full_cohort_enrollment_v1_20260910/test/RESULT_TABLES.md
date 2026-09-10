# full-cohort-enrollment-v1: test

Protocol: full-cohort-enrollment-v1. Participant-macro MAE is primary.
The threshold fields below are retrospective numerical AAMI/BHS screens, not device certification.
All frozen eligible queries and participants are retained. Registration uses approximately 90% of each person's eligible source windows; indivisible overlap groups and minimum one-window roles can change the exact fraction. Consult the cohort and budget audit for exclusions and actual counts. These are not independent cuff events.
Random-window evaluation does not establish chronological or long-term performance.

Source-duplicate audit: `{"cross_outer_subject_overlap": 0, "cross_role_content_overlap": 0}`

## Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_person_lora | SBP | 3.4251 | 0.9409 | 0.0245 | 5.2910 | 79.2771 | 94.6675 | 98.1262 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 1.8949 | 0.9170 | -0.1212 | 3.6509 | 93.6168 | 98.5697 | 99.3724 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 2.7235 | 0.9570 | -0.0559 | 4.5117 | 85.6104 | 96.2294 | 98.5812 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 1.5091 | 0.9377 | -0.0970 | 3.1626 | 95.2401 | 98.8305 | 99.4964 | PASS* | PASS (Grade A)* |

## MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_person_lora | SBP | 3.4744 | 0.9410 | 0.0763 | 5.3904 | 79.0100 | 94.3564 | 97.9255 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 1.9225 | 0.9154 | -0.1430 | 3.7870 | 93.4911 | 98.3024 | 99.1786 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 2.7968 | 0.9568 | -0.0314 | 4.6120 | 85.1310 | 95.9077 | 98.4139 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 1.5365 | 0.9386 | -0.0967 | 3.2254 | 94.9180 | 98.6188 | 99.3579 | PASS* | PASS (Grade A)* |

## VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_person_lora | SBP | 3.3107 | 0.9300 | -0.0955 | 5.0510 | 79.8965 | 95.3888 | 98.5916 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 1.8309 | 0.9215 | -0.0706 | 3.3134 | 93.9083 | 99.1898 | 99.8218 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 2.5535 | 0.9500 | -0.1126 | 4.2694 | 86.7221 | 96.9754 | 98.9692 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 1.4455 | 0.9351 | -0.0977 | 3.0120 | 95.9869 | 99.3213 | 99.8176 | PASS* | PASS (Grade A)* |
