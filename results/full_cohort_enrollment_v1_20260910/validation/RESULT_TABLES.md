# full-cohort-enrollment-v1: validation

Protocol: full-cohort-enrollment-v1. Participant-macro MAE is primary.
The threshold fields below are retrospective numerical AAMI/BHS screens, not device certification.
All frozen eligible queries and participants are retained. Registration uses approximately 90% of each person's eligible source windows; indivisible overlap groups and minimum one-window roles can change the exact fraction. Consult the cohort and budget audit for exclusions and actual counts. These are not independent cuff events.
Random-window evaluation does not establish chronological or long-term performance.

Source-duplicate audit: `{"cross_outer_subject_overlap": 0, "cross_role_content_overlap": 0}`

## Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_person_lora | SBP | 3.4564 | 0.9393 | 0.0194 | 5.2259 | 78.6475 | 94.5455 | 98.0965 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 1.9383 | 0.9290 | -0.0909 | 3.3992 | 93.1725 | 98.5814 | 99.4006 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 2.7970 | 0.9549 | -0.0655 | 4.5067 | 84.6806 | 95.9940 | 98.5918 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 1.5590 | 0.9478 | -0.0709 | 2.9148 | 94.8108 | 98.7817 | 99.5176 | PASS* | PASS (Grade A)* |

## MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_person_lora | SBP | 3.4699 | 0.9406 | 0.0491 | 5.3166 | 78.8202 | 94.5093 | 97.9556 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 1.9384 | 0.9274 | -0.1074 | 3.5525 | 93.2586 | 98.4307 | 99.2712 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 2.8535 | 0.9550 | -0.0602 | 4.6284 | 84.3252 | 95.7511 | 98.4667 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 1.5765 | 0.9464 | -0.0722 | 3.0540 | 94.6029 | 98.6107 | 99.4187 | PASS* | PASS (Grade A)* |

## VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_person_lora | SBP | 3.4210 | 0.9311 | -0.0578 | 4.9813 | 78.1979 | 94.6397 | 98.4631 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 1.9378 | 0.9338 | -0.0480 | 2.9626 | 92.9482 | 98.9739 | 99.7376 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 2.6499 | 0.9516 | -0.0793 | 4.1732 | 85.6058 | 96.6264 | 98.9176 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 1.5132 | 0.9522 | -0.0675 | 2.5167 | 95.3519 | 99.2269 | 99.7751 | PASS* | PASS (Grade A)* |
