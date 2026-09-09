# Thirty-subject post-training enrollment

Protocol: post-enrollment-30-v1. Participant-macro MAE is primary.
The threshold fields below are retrospective numerical AAMI/BHS screens, not device certification.
All windows and participants are retained. Each profile uses 360 labelled registration windows, not 360 independent cuff events.
Random-window evaluation does not establish chronological or long-term performance.

Source-duplicate audit: `{"cross_population_subject_overlap": 0, "test_rows_with_registration_content": 0}`

## Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| personal_mean | SBP | 11.5854 | 0.5326 | 0.1398 | 15.6052 | 29.6667 | 54.5000 | 72.2500 | FAIL* | FAIL (Grade D)* |
| personal_mean | DBP | 6.0402 | 0.5358 | 0.0931 | 8.1810 | 52.0000 | 81.8333 | 93.6667 | FAIL* | PASS (Grade B)* |
| shared_with_anchor | SBP | 10.8464 | 0.5966 | 1.4025 | 14.4293 | 32.9167 | 56.1667 | 73.6667 | FAIL* | FAIL (Grade D)* |
| shared_with_anchor | DBP | 5.7255 | 0.5943 | 0.7996 | 7.6069 | 53.6667 | 83.5000 | 95.6667 | PASS* | PASS (Grade B)* |
| new_person_lora | SBP | 4.0749 | 0.9272 | 0.1723 | 6.1558 | 72.9167 | 92.7500 | 96.6667 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 2.2775 | 0.9161 | -0.0599 | 3.4770 | 89.7500 | 97.6667 | 99.3333 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 3.1515 | 0.9562 | 0.0108 | 4.7765 | 80.5000 | 95.6667 | 97.8333 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 1.7466 | 0.9472 | -0.0849 | 2.7576 | 92.4167 | 98.8333 | 99.8333 | PASS* | PASS (Grade A)* |

## MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| personal_mean | SBP | 10.5278 | 0.6606 | 0.6735 | 13.8871 | 31.5000 | 59.0000 | 75.3333 | FAIL* | FAIL (Grade D)* |
| personal_mean | DBP | 5.2594 | 0.4859 | 0.6231 | 7.0042 | 57.3333 | 86.0000 | 96.8333 | PASS* | PASS (Grade B)* |
| shared_with_anchor | SBP | 10.7799 | 0.6557 | 3.0014 | 13.6772 | 33.3333 | 54.5000 | 72.3333 | FAIL* | FAIL (Grade D)* |
| shared_with_anchor | DBP | 5.5361 | 0.4528 | 1.8172 | 7.0229 | 54.3333 | 84.3333 | 96.0000 | PASS* | PASS (Grade B)* |
| new_person_lora | SBP | 3.8486 | 0.9418 | 0.5098 | 5.7337 | 74.5000 | 94.3333 | 97.0000 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 2.2589 | 0.8635 | 0.0950 | 3.6215 | 91.1667 | 97.3333 | 98.8333 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 3.0666 | 0.9597 | 0.1817 | 4.7874 | 81.3333 | 96.5000 | 98.1667 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 1.7583 | 0.9128 | -0.0419 | 2.8959 | 92.8333 | 98.6667 | 99.8333 | PASS* | PASS (Grade A)* |

## VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| personal_mean | SBP | 12.6429 | 0.3759 | -0.3939 | 17.1473 | 27.8333 | 50.0000 | 69.1667 | FAIL* | FAIL (Grade D)* |
| personal_mean | DBP | 6.8209 | 0.5238 | -0.4369 | 9.1842 | 46.6667 | 77.6667 | 90.5000 | FAIL* | FAIL (Grade C)* |
| shared_with_anchor | SBP | 10.9128 | 0.5235 | -0.1965 | 14.9857 | 32.5000 | 57.8333 | 75.0000 | FAIL* | FAIL (Grade D)* |
| shared_with_anchor | DBP | 5.9150 | 0.6368 | -0.2181 | 8.0268 | 53.0000 | 82.6667 | 95.3333 | FAIL* | PASS (Grade B)* |
| new_person_lora | SBP | 4.3011 | 0.9093 | -0.1653 | 6.5381 | 71.3333 | 91.1667 | 96.3333 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 2.2961 | 0.9376 | -0.2148 | 3.3220 | 88.3333 | 98.0000 | 99.8333 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 3.2364 | 0.9518 | -0.1600 | 4.7635 | 79.6667 | 94.8333 | 97.5000 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 1.7349 | 0.9614 | -0.1279 | 2.6136 | 92.0000 | 99.0000 | 99.8333 | PASS* | PASS (Grade A)* |
