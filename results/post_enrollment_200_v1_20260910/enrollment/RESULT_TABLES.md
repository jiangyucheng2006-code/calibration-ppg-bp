# 200-subject post-training enrollment

Protocol: post-enrollment-200-v1. Participant-macro MAE is primary.
The threshold fields below are retrospective numerical AAMI/BHS screens, not device certification.
All windows and participants are retained. Each profile uses 360 labelled registration windows, not 360 independent cuff events.
Random-window evaluation does not establish chronological or long-term performance.

Source-duplicate audit: `{"cross_population_subject_overlap": 0, "test_rows_with_registration_content": 0}`

## Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| personal_mean | SBP | 10.5711 | 0.5580 | -0.1104 | 14.0837 | 32.7875 | 58.8250 | 75.9625 | FAIL* | FAIL (Grade D)* |
| personal_mean | DBP | 5.6356 | 0.6457 | -0.1681 | 7.6536 | 56.7750 | 84.2375 | 94.5875 | PASS* | PASS (Grade B)* |
| shared_with_anchor | SBP | 9.9821 | 0.6210 | 1.4193 | 12.9636 | 32.5375 | 60.7500 | 78.7000 | FAIL* | FAIL (Grade D)* |
| shared_with_anchor | DBP | 5.3499 | 0.6949 | 0.8200 | 7.0563 | 57.0375 | 86.4375 | 95.9625 | PASS* | PASS (Grade B)* |
| new_person_lora | SBP | 3.9049 | 0.9260 | 0.0705 | 5.7634 | 74.0125 | 93.0500 | 97.4750 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 2.1858 | 0.9269 | -0.1161 | 3.4760 | 91.3125 | 98.4000 | 99.4500 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 3.3312 | 0.9414 | -0.1154 | 5.1275 | 79.5875 | 94.6625 | 98.0250 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 1.8442 | 0.9388 | -0.1379 | 3.1782 | 93.4750 | 98.5750 | 99.4875 | PASS* | PASS (Grade A)* |
| shared_memory_only | SBP | 3.4846 | 0.9349 | -0.2956 | 5.3987 | 78.7000 | 93.9250 | 97.8125 | PASS* | PASS (Grade A)* |
| shared_memory_only | DBP | 1.9178 | 0.9352 | -0.1920 | 3.2676 | 92.7500 | 98.5250 | 99.5625 | PASS* | PASS (Grade A)* |
| shared_memory_blend | SBP | 4.3714 | 0.9053 | -0.0724 | 6.5202 | 70.0625 | 89.9625 | 96.2250 | PASS* | PASS (Grade A)* |
| shared_memory_blend | DBP | 2.3521 | 0.9146 | 0.0145 | 3.7585 | 88.8625 | 97.9000 | 99.3875 | PASS* | PASS (Grade A)* |
| lora_memory_only | SBP | 3.3696 | 0.9382 | -0.1718 | 5.2647 | 79.8000 | 94.5875 | 97.9000 | PASS* | PASS (Grade A)* |
| lora_memory_only | DBP | 1.8519 | 0.9377 | -0.1524 | 3.2061 | 93.1750 | 98.6250 | 99.5500 | PASS* | PASS (Grade A)* |
| lora_memory_uniform | SBP | 3.3919 | 0.9395 | -0.1208 | 5.2080 | 78.9625 | 94.4000 | 97.8750 | PASS* | PASS (Grade A)* |
| lora_memory_uniform | DBP | 1.8740 | 0.9374 | -0.1426 | 3.2152 | 93.2500 | 98.5750 | 99.5000 | PASS* | PASS (Grade A)* |
| lora_memory_fixed_half | SBP | 3.4078 | 0.9402 | -0.0506 | 5.1789 | 79.0250 | 94.6250 | 98.0625 | PASS* | PASS (Grade A)* |
| lora_memory_fixed_half | DBP | 1.8801 | 0.9400 | -0.1343 | 3.1479 | 93.4125 | 98.6750 | 99.5625 | PASS* | PASS (Grade A)* |

## MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| personal_mean | SBP | 10.5242 | 0.6135 | -0.2036 | 14.1081 | 34.1000 | 59.4500 | 75.9250 | FAIL* | FAIL (Grade D)* |
| personal_mean | DBP | 5.6006 | 0.6719 | -0.2661 | 7.8010 | 57.7000 | 84.6000 | 94.5000 | PASS* | PASS (Grade B)* |
| shared_with_anchor | SBP | 10.3415 | 0.6465 | 2.0587 | 13.3350 | 31.6500 | 58.9750 | 77.4250 | FAIL* | FAIL (Grade D)* |
| shared_with_anchor | DBP | 5.6239 | 0.6946 | 1.3819 | 7.4038 | 54.4000 | 85.2250 | 95.1500 | PASS* | PASS (Grade B)* |
| new_person_lora | SBP | 4.2868 | 0.9234 | 0.1955 | 6.2786 | 71.0750 | 91.3750 | 96.5500 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 2.3628 | 0.9161 | -0.1720 | 3.9437 | 89.9750 | 97.8000 | 99.1750 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 3.7162 | 0.9387 | -0.0539 | 5.6179 | 76.1250 | 93.1750 | 97.3750 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 2.0353 | 0.9277 | -0.1422 | 3.6624 | 92.1000 | 98.0250 | 99.2250 | PASS* | PASS (Grade A)* |
| shared_memory_only | SBP | 3.8867 | 0.9312 | -0.2782 | 5.9477 | 74.8000 | 92.4750 | 97.3000 | PASS* | PASS (Grade A)* |
| shared_memory_only | DBP | 2.1361 | 0.9231 | -0.1781 | 3.7745 | 91.0000 | 97.8750 | 99.3500 | PASS* | PASS (Grade A)* |
| shared_memory_blend | SBP | 4.7699 | 0.9068 | 0.0837 | 6.9294 | 66.0000 | 88.2000 | 95.8500 | PASS* | PASS (Grade A)* |
| shared_memory_blend | DBP | 2.5645 | 0.9053 | 0.1933 | 4.1903 | 87.0500 | 97.3250 | 99.1750 | PASS* | PASS (Grade A)* |
| lora_memory_only | SBP | 3.7708 | 0.9344 | -0.1061 | 5.8103 | 76.1500 | 93.3250 | 97.3750 | PASS* | PASS (Grade A)* |
| lora_memory_only | DBP | 2.0602 | 0.9260 | -0.1331 | 3.7036 | 91.8750 | 98.0000 | 99.3250 | PASS* | PASS (Grade A)* |
| lora_memory_uniform | SBP | 3.7779 | 0.9367 | -0.0599 | 5.7091 | 75.5500 | 92.8750 | 97.2500 | PASS* | PASS (Grade A)* |
| lora_memory_uniform | DBP | 2.0638 | 0.9262 | -0.1449 | 3.6992 | 91.9250 | 97.9750 | 99.2500 | PASS* | PASS (Grade A)* |
| lora_memory_fixed_half | SBP | 3.7842 | 0.9368 | 0.0447 | 5.7040 | 75.4250 | 93.1250 | 97.5000 | PASS* | PASS (Grade A)* |
| lora_memory_fixed_half | DBP | 2.0637 | 0.9292 | -0.1525 | 3.6218 | 92.0000 | 98.1500 | 99.3750 | PASS* | PASS (Grade A)* |

## VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| personal_mean | SBP | 10.6180 | 0.4321 | -0.0172 | 14.0604 | 31.4750 | 58.2000 | 76.0000 | FAIL* | FAIL (Grade D)* |
| personal_mean | DBP | 5.6706 | 0.6110 | -0.0702 | 7.5030 | 55.8500 | 83.8750 | 94.6750 | PASS* | PASS (Grade B)* |
| shared_with_anchor | SBP | 9.6226 | 0.5458 | 0.7798 | 12.5503 | 33.4250 | 62.5250 | 79.9750 | FAIL* | FAIL (Grade D)* |
| shared_with_anchor | DBP | 5.0759 | 0.6945 | 0.2581 | 6.6444 | 59.6750 | 87.6500 | 96.7750 | PASS* | PASS (Grade B)* |
| new_person_lora | SBP | 3.5230 | 0.9225 | -0.0545 | 5.1952 | 76.9500 | 94.7250 | 98.4000 | PASS* | PASS (Grade A)* |
| new_person_lora | DBP | 2.0088 | 0.9405 | -0.0601 | 2.9341 | 92.6500 | 99.0000 | 99.7250 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | SBP | 2.9463 | 0.9395 | -0.1768 | 4.5848 | 83.0500 | 96.1500 | 98.6750 | PASS* | PASS (Grade A)* |
| new_person_lora_memory | DBP | 1.6531 | 0.9530 | -0.1337 | 2.6059 | 94.8500 | 99.1250 | 99.7500 | PASS* | PASS (Grade A)* |
| shared_memory_only | SBP | 3.0825 | 0.9339 | -0.3129 | 4.7877 | 82.6000 | 95.3750 | 98.3250 | PASS* | PASS (Grade A)* |
| shared_memory_only | DBP | 1.6996 | 0.9506 | -0.2059 | 2.6665 | 94.5000 | 99.1750 | 99.7750 | PASS* | PASS (Grade A)* |
| shared_memory_blend | SBP | 3.9728 | 0.8937 | -0.2285 | 6.0804 | 74.1250 | 91.7250 | 96.6000 | PASS* | PASS (Grade A)* |
| shared_memory_blend | DBP | 2.1397 | 0.9263 | -0.1643 | 3.2610 | 90.6750 | 98.4750 | 99.6000 | PASS* | PASS (Grade A)* |
| lora_memory_only | SBP | 2.9685 | 0.9376 | -0.2376 | 4.6553 | 83.4500 | 95.8500 | 98.4250 | PASS* | PASS (Grade A)* |
| lora_memory_only | DBP | 1.6436 | 0.9525 | -0.1718 | 2.6159 | 94.4750 | 99.2500 | 99.7750 | PASS* | PASS (Grade A)* |
| lora_memory_uniform | SBP | 3.0059 | 0.9377 | -0.1817 | 4.6531 | 82.3750 | 95.9250 | 98.5000 | PASS* | PASS (Grade A)* |
| lora_memory_uniform | DBP | 1.6842 | 0.9515 | -0.1403 | 2.6444 | 94.5750 | 99.1750 | 99.7500 | PASS* | PASS (Grade A)* |
| lora_memory_fixed_half | SBP | 3.0314 | 0.9393 | -0.1460 | 4.5930 | 82.6250 | 96.1250 | 98.6250 | PASS* | PASS (Grade A)* |
| lora_memory_fixed_half | DBP | 1.6965 | 0.9536 | -0.1160 | 2.5890 | 94.8250 | 99.2000 | 99.7500 | PASS* | PASS (Grade A)* |
