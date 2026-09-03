# Round-10 partial end-to-end internal screen

## Result and decision

Round 10 completed successfully after recovery from the documented
mixed-precision implementation failure. Jobs 1017--1025 and deterministic
report job 1026 all completed with exit code `0:0`; every stderr file is empty,
and the work and NAS result trees are byte-identical.

Folds 0--2 fit the models, fold 3 controls patience-8 early stopping, and fold
4 ranks candidates. Every setting uses the same 628 participants and 96,332
K=5 event-6-onward queries. MIMIC contains 285 participants and 74,373 queries;
VitalDB contains 343 participants and 21,959 queries. Source counts sum exactly
to Overall. Query keys and BP targets match exactly across all nine settings.
Meta-validation and the locked meta-test were not accessed.

T10-8, last-block adaptation with pair-direction and temporal-consistency
objectives, is the numerical winner. It reduces Overall participant-macro mean
MAE from 8.5322 to 8.4457 mmHg and improves both source strata. Its 0.0865-mmHg
Overall gain is nevertheless below the prespecified 0.15-mmHg promotion
threshold. Therefore **no Round-10 candidate is promoted or evaluated on
meta-validation**.

| Scope | T10-0 reference mean MAE | T10-8 mean MAE | Improvement |
| --- | ---: | ---: | ---: |
| Overall | 8.5322 | 8.4457 | 0.0865 |
| MIMIC | 9.2121 | 9.1397 | 0.0725 |
| VitalDB | 7.9672 | 7.8690 | 0.0981 |

The following tables are development-only internal results. MIMIC and VitalDB
are PulseDB source strata, not independent external validation datasets.

## Participant-macro primary results

| Setting | Scope | N participants | N queries | SBP participant-macro MAE | DBP participant-macro MAE | Mean participant-macro MAE |
| --- | --- | --- | --- | --- | --- | --- |
| T10-6 Last block + temporal consistency | MIMIC | 285 | 74373 | 11.9000 | 6.3390 | 9.1195 |
| T10-8 Last block + direction + temporal | MIMIC | 285 | 74373 | 11.9177 | 6.3617 | 9.1397 |
| T10-4 Full-encoder adaptation | MIMIC | 285 | 74373 | 11.9790 | 6.4149 | 9.1969 |
| T10-0 Frozen-encoder reference | MIMIC | 285 | 74373 | 12.0271 | 6.3972 | 9.2121 |
| T10-5 Last block + pair direction | MIMIC | 285 | 74373 | 12.0215 | 6.4213 | 9.2214 |
| T10-2 Last-block adaptation | MIMIC | 285 | 74373 | 12.0317 | 6.4202 | 9.2259 |
| T10-3 Last-two-block adaptation | MIMIC | 285 | 74373 | 12.0606 | 6.4146 | 9.2376 |
| T10-7 Last block + adaptive fusion | MIMIC | 285 | 74373 | 12.0590 | 6.4274 | 9.2432 |
| T10-1 Projection-only adaptation | MIMIC | 285 | 74373 | 12.1482 | 6.4777 | 9.3129 |
| T10-8 Last block + direction + temporal | Overall | 628 | 96332 | 10.7820 | 6.1093 | 8.4457 |
| T10-6 Last block + temporal consistency | Overall | 628 | 96332 | 10.7961 | 6.1158 | 8.4559 |
| T10-5 Last block + pair direction | Overall | 628 | 96332 | 10.8534 | 6.1835 | 8.5185 |
| T10-2 Last-block adaptation | Overall | 628 | 96332 | 10.8764 | 6.1834 | 8.5299 |
| T10-0 Frozen-encoder reference | Overall | 628 | 96332 | 10.8946 | 6.1697 | 8.5322 |
| T10-3 Last-two-block adaptation | Overall | 628 | 96332 | 10.9251 | 6.1747 | 8.5499 |
| T10-7 Last block + adaptive fusion | Overall | 628 | 96332 | 10.9254 | 6.1999 | 8.5626 |
| T10-4 Full-encoder adaptation | Overall | 628 | 96332 | 10.9326 | 6.2155 | 8.5740 |
| T10-1 Projection-only adaptation | Overall | 628 | 96332 | 10.9585 | 6.2113 | 8.5849 |
| T10-8 Last block + direction + temporal | VitalDB | 343 | 21959 | 9.8384 | 5.8996 | 7.8690 |
| T10-6 Last block + temporal consistency | VitalDB | 343 | 21959 | 9.8788 | 5.9303 | 7.9046 |
| T10-5 Last block + pair direction | VitalDB | 343 | 21959 | 9.8828 | 5.9859 | 7.9344 |
| T10-2 Last-block adaptation | VitalDB | 343 | 21959 | 9.9164 | 5.9867 | 7.9515 |
| T10-0 Frozen-encoder reference | VitalDB | 343 | 21959 | 9.9537 | 5.9807 | 7.9672 |
| T10-3 Last-two-block adaptation | VitalDB | 343 | 21959 | 9.9816 | 5.9753 | 7.9785 |
| T10-1 Projection-only adaptation | VitalDB | 343 | 21959 | 9.9700 | 5.9900 | 7.9800 |
| T10-7 Last block + adaptive fusion | VitalDB | 343 | 21959 | 9.9834 | 6.0108 | 7.9971 |
| T10-4 Full-encoder adaptation | VitalDB | 343 | 21959 | 10.0632 | 6.0497 | 8.0565 |

## Change versus the frozen-encoder reference

Negative candidate-minus-reference values are better.

| Setting | Scope | Reference mean participant-macro MAE | Candidate mean participant-macro MAE | Candidate minus reference |
| --- | --- | --- | --- | --- |
| T10-0 Frozen-encoder reference | Overall | 8.5322 | 8.5322 | 0.0000 |
| T10-0 Frozen-encoder reference | MIMIC | 9.2121 | 9.2121 | 0.0000 |
| T10-0 Frozen-encoder reference | VitalDB | 7.9672 | 7.9672 | 0.0000 |
| T10-1 Projection-only adaptation | Overall | 8.5322 | 8.5849 | 0.0527 |
| T10-1 Projection-only adaptation | MIMIC | 9.2121 | 9.3129 | 0.1008 |
| T10-1 Projection-only adaptation | VitalDB | 7.9672 | 7.9800 | 0.0128 |
| T10-2 Last-block adaptation | Overall | 8.5322 | 8.5299 | -0.0023 |
| T10-2 Last-block adaptation | MIMIC | 9.2121 | 9.2259 | 0.0138 |
| T10-2 Last-block adaptation | VitalDB | 7.9672 | 7.9515 | -0.0156 |
| T10-3 Last-two-block adaptation | Overall | 8.5322 | 8.5499 | 0.0177 |
| T10-3 Last-two-block adaptation | MIMIC | 9.2121 | 9.2376 | 0.0255 |
| T10-3 Last-two-block adaptation | VitalDB | 7.9672 | 7.9785 | 0.0113 |
| T10-4 Full-encoder adaptation | Overall | 8.5322 | 8.5740 | 0.0419 |
| T10-4 Full-encoder adaptation | MIMIC | 9.2121 | 9.1969 | -0.0152 |
| T10-4 Full-encoder adaptation | VitalDB | 7.9672 | 8.0565 | 0.0893 |
| T10-5 Last block + pair direction | Overall | 8.5322 | 8.5185 | -0.0137 |
| T10-5 Last block + pair direction | MIMIC | 9.2121 | 9.2214 | 0.0093 |
| T10-5 Last block + pair direction | VitalDB | 7.9672 | 7.9344 | -0.0328 |
| T10-6 Last block + temporal consistency | Overall | 8.5322 | 8.4559 | -0.0763 |
| T10-6 Last block + temporal consistency | MIMIC | 9.2121 | 9.1195 | -0.0927 |
| T10-6 Last block + temporal consistency | VitalDB | 7.9672 | 7.9046 | -0.0626 |
| T10-7 Last block + adaptive fusion | Overall | 8.5322 | 8.5626 | 0.0304 |
| T10-7 Last block + adaptive fusion | MIMIC | 9.2121 | 9.2432 | 0.0311 |
| T10-7 Last block + adaptive fusion | VitalDB | 7.9672 | 7.9971 | 0.0299 |
| T10-8 Last block + direction + temporal | Overall | 8.5322 | 8.4457 | -0.0865 |
| T10-8 Last block + direction + temporal | MIMIC | 9.2121 | 9.1397 | -0.0725 |
| T10-8 Last block + direction + temporal | VitalDB | 7.9672 | 7.8690 | -0.0981 |

## Event-pooled diagnostics

Participant-macro MAE is primary. AAMI/BHS entries are retrospective numerical screens only.

| Setting | Scope | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T10-0 Frozen-encoder reference | Overall | SBP | 12.5627 | 0.4455 | -2.3313 | 16.4551 | 27.8827 | 50.8066 | 67.9317 | FAIL* | FAIL (Grade D)* |
| T10-0 Frozen-encoder reference | Overall | DBP | 6.7114 | 0.4915 | -1.0927 | 9.2840 | 49.0771 | 78.2056 | 91.4732 | FAIL* | FAIL (Grade C)* |
| T10-1 Projection-only adaptation | Overall | SBP | 12.6168 | 0.4416 | -2.7460 | 16.4510 | 27.6544 | 50.5367 | 67.6463 | FAIL* | FAIL (Grade D)* |
| T10-1 Projection-only adaptation | Overall | DBP | 6.7613 | 0.4827 | -1.6101 | 9.2901 | 48.6308 | 78.2357 | 91.4546 | FAIL* | FAIL (Grade C)* |
| T10-2 Last-block adaptation | Overall | SBP | 12.4928 | 0.4512 | -2.6093 | 16.3261 | 27.8796 | 51.0215 | 68.2182 | FAIL* | FAIL (Grade D)* |
| T10-2 Last-block adaptation | Overall | DBP | 6.7017 | 0.4907 | -1.4839 | 9.2366 | 48.9329 | 78.5191 | 91.6061 | FAIL* | FAIL (Grade C)* |
| T10-3 Last-two-block adaptation | Overall | SBP | 12.4496 | 0.4514 | -0.5055 | 16.5236 | 28.1537 | 51.1543 | 68.4248 | FAIL* | FAIL (Grade D)* |
| T10-3 Last-two-block adaptation | Overall | DBP | 6.7091 | 0.4931 | -0.6913 | 9.3077 | 49.0502 | 78.2834 | 91.4857 | FAIL* | FAIL (Grade C)* |
| T10-4 Full-encoder adaptation | Overall | SBP | 12.5063 | 0.4401 | -0.8905 | 16.6771 | 28.5263 | 51.4139 | 68.5266 | FAIL* | FAIL (Grade D)* |
| T10-4 Full-encoder adaptation | Overall | DBP | 6.7553 | 0.4828 | -1.0637 | 9.3675 | 49.1124 | 78.0457 | 91.1774 | FAIL* | FAIL (Grade C)* |
| T10-5 Last block + pair direction | Overall | SBP | 12.4809 | 0.4520 | -2.5355 | 16.3262 | 27.8973 | 51.0402 | 68.2504 | FAIL* | FAIL (Grade D)* |
| T10-5 Last block + pair direction | Overall | DBP | 6.7032 | 0.4901 | -1.5016 | 9.2400 | 48.9920 | 78.5710 | 91.5636 | FAIL* | FAIL (Grade C)* |
| T10-6 Last block + temporal consistency | Overall | SBP | 12.3738 | 0.4616 | -1.6703 | 16.2907 | 28.0987 | 51.5135 | 68.6843 | FAIL* | FAIL (Grade D)* |
| T10-6 Last block + temporal consistency | Overall | DBP | 6.6367 | 0.5034 | -0.7958 | 9.2033 | 49.6273 | 78.7028 | 91.6331 | FAIL* | FAIL (Grade C)* |
| T10-7 Last block + adaptive fusion | Overall | SBP | 12.5771 | 0.4438 | -3.0165 | 16.3698 | 27.6668 | 50.7921 | 67.9701 | FAIL* | FAIL (Grade D)* |
| T10-7 Last block + adaptive fusion | Overall | DBP | 6.7218 | 0.4869 | -1.6039 | 9.2519 | 48.9879 | 78.4308 | 91.5023 | FAIL* | FAIL (Grade C)* |
| T10-8 Last block + direction + temporal | Overall | SBP | 12.3852 | 0.4583 | -1.9047 | 16.3154 | 28.0260 | 51.7087 | 68.6615 | FAIL* | FAIL (Grade D)* |
| T10-8 Last block + direction + temporal | Overall | DBP | 6.6828 | 0.4972 | -0.6396 | 9.2739 | 49.3740 | 78.3872 | 91.4110 | FAIL* | FAIL (Grade C)* |
| T10-0 Frozen-encoder reference | MIMIC | SBP | 13.3381 | 0.4158 | -2.8932 | 17.2366 | 26.0189 | 47.9287 | 64.8622 | FAIL* | FAIL (Grade D)* |
| T10-0 Frozen-encoder reference | MIMIC | DBP | 6.8993 | 0.4709 | -1.0309 | 9.5325 | 48.1317 | 77.1006 | 90.8125 | FAIL* | FAIL (Grade C)* |
| T10-1 Projection-only adaptation | MIMIC | SBP | 13.4110 | 0.4112 | -3.3787 | 17.2175 | 25.7056 | 47.5589 | 64.5234 | FAIL* | FAIL (Grade D)* |
| T10-1 Projection-only adaptation | MIMIC | DBP | 6.9652 | 0.4593 | -1.7178 | 9.5385 | 47.5455 | 77.0844 | 90.7601 | FAIL* | FAIL (Grade C)* |
| T10-2 Last-block adaptation | MIMIC | SBP | 13.2641 | 0.4222 | -3.1210 | 17.0978 | 26.0175 | 48.0981 | 65.1460 | FAIL* | FAIL (Grade D)* |
| T10-2 Last-block adaptation | MIMIC | DBP | 6.8838 | 0.4700 | -1.5166 | 9.4749 | 47.9757 | 77.5053 | 90.9591 | FAIL* | FAIL (Grade C)* |
| T10-3 Last-two-block adaptation | MIMIC | SBP | 13.1751 | 0.4242 | -0.8857 | 17.3284 | 26.4545 | 48.4342 | 65.5695 | FAIL* | FAIL (Grade D)* |
| T10-3 Last-two-block adaptation | MIMIC | DBP | 6.8993 | 0.4727 | -0.5932 | 9.5531 | 48.0658 | 77.1718 | 90.8071 | FAIL* | FAIL (Grade C)* |
| T10-4 Full-encoder adaptation | MIMIC | SBP | 13.2291 | 0.4124 | -1.1763 | 17.4888 | 26.8942 | 48.8242 | 65.7080 | FAIL* | FAIL (Grade D)* |
| T10-4 Full-encoder adaptation | MIMIC | DBP | 6.9420 | 0.4618 | -1.0051 | 9.6176 | 48.3280 | 76.9217 | 90.4522 | FAIL* | FAIL (Grade C)* |
| T10-5 Last block + pair direction | MIMIC | SBP | 13.2597 | 0.4225 | -3.0449 | 17.1069 | 26.0323 | 48.1277 | 65.1688 | FAIL* | FAIL (Grade D)* |
| T10-5 Last block + pair direction | MIMIC | DBP | 6.8868 | 0.4690 | -1.5309 | 9.4819 | 48.0551 | 77.5550 | 90.9040 | FAIL* | FAIL (Grade C)* |
| T10-6 Last block + temporal consistency | MIMIC | SBP | 13.1203 | 0.4335 | -2.1635 | 17.0733 | 26.2918 | 48.7435 | 65.7147 | FAIL* | FAIL (Grade D)* |
| T10-6 Last block + temporal consistency | MIMIC | DBP | 6.8113 | 0.4846 | -0.6780 | 9.4383 | 48.7583 | 77.6048 | 90.9819 | FAIL* | FAIL (Grade C)* |
| T10-7 Last block + adaptive fusion | MIMIC | SBP | 13.3542 | 0.4143 | -3.4577 | 17.1546 | 25.8239 | 47.8305 | 64.9604 | FAIL* | FAIL (Grade D)* |
| T10-7 Last block + adaptive fusion | MIMIC | DBP | 6.9036 | 0.4662 | -1.6093 | 9.4945 | 47.9986 | 77.4421 | 90.8448 | FAIL* | FAIL (Grade C)* |
| T10-8 Last block + direction + temporal | MIMIC | SBP | 13.1261 | 0.4308 | -2.3526 | 17.0907 | 26.1923 | 48.9425 | 65.8196 | FAIL* | FAIL (Grade D)* |
| T10-8 Last block + direction + temporal | MIMIC | DBP | 6.8728 | 0.4767 | -0.6265 | 9.5144 | 48.4558 | 77.2740 | 90.7103 | FAIL* | FAIL (Grade C)* |
| T10-0 Frozen-encoder reference | VitalDB | SBP | 9.9363 | 0.5168 | -0.4280 | 13.3008 | 34.1955 | 60.5538 | 78.3278 | FAIL* | FAIL (Grade D)* |
| T10-0 Frozen-encoder reference | VitalDB | DBP | 6.0751 | 0.5479 | -1.3022 | 8.3849 | 52.2792 | 81.9482 | 93.7110 | FAIL* | PASS (Grade B)* |
| T10-1 Projection-only adaptation | VitalDB | SBP | 9.9271 | 0.5152 | -0.6033 | 13.3152 | 34.2547 | 60.6221 | 78.2231 | FAIL* | FAIL (Grade D)* |
| T10-1 Projection-only adaptation | VitalDB | DBP | 6.0706 | 0.5489 | -1.2451 | 8.3841 | 52.3066 | 82.1349 | 93.8066 | FAIL* | PASS (Grade B)* |
| T10-2 Last-block adaptation | VitalDB | SBP | 9.8802 | 0.5196 | -0.8763 | 13.2402 | 34.1864 | 60.9226 | 78.6238 | FAIL* | FAIL (Grade D)* |
| T10-2 Last-block adaptation | VitalDB | DBP | 6.0850 | 0.5474 | -1.3730 | 8.3787 | 52.1745 | 81.9527 | 93.7975 | FAIL* | PASS (Grade B)* |
| T10-3 Last-two-block adaptation | VitalDB | SBP | 9.9924 | 0.5109 | 0.7822 | 13.3651 | 33.9086 | 60.3670 | 78.0955 | FAIL* | FAIL (Grade D)* |
| T10-3 Last-two-block adaptation | VitalDB | DBP | 6.0650 | 0.5488 | -1.0234 | 8.4151 | 52.3840 | 82.0484 | 93.7839 | FAIL* | PASS (Grade B)* |
| T10-4 Full-encoder adaptation | VitalDB | SBP | 10.0583 | 0.5007 | 0.0775 | 13.5275 | 34.0544 | 60.1849 | 78.0728 | FAIL* | FAIL (Grade D)* |
| T10-4 Full-encoder adaptation | VitalDB | DBP | 6.1231 | 0.5403 | -1.2619 | 8.4628 | 51.7692 | 81.8525 | 93.6336 | FAIL* | PASS (Grade B)* |
| T10-5 Last block + pair direction | VitalDB | SBP | 9.8431 | 0.5226 | -0.8100 | 13.2019 | 34.2138 | 60.9044 | 78.6876 | FAIL* | FAIL (Grade D)* |
| T10-5 Last block + pair direction | VitalDB | DBP | 6.0812 | 0.5479 | -1.4023 | 8.3685 | 52.1654 | 82.0119 | 93.7975 | FAIL* | PASS (Grade B)* |
| T10-6 Last block + temporal consistency | VitalDB | SBP | 9.8454 | 0.5270 | -0.0002 | 13.1663 | 34.2183 | 60.8953 | 78.7422 | FAIL* | FAIL (Grade D)* |
| T10-6 Last block + temporal consistency | VitalDB | DBP | 6.0456 | 0.5536 | -1.1949 | 8.3465 | 52.5707 | 82.4218 | 93.8385 | FAIL* | PASS (Grade B)* |
| T10-7 Last block + adaptive fusion | VitalDB | SBP | 9.9454 | 0.5135 | -1.5222 | 13.2656 | 33.9086 | 60.8224 | 78.1639 | FAIL* | FAIL (Grade D)* |
| T10-7 Last block + adaptive fusion | VitalDB | DBP | 6.1063 | 0.5434 | -1.5857 | 8.3787 | 52.3384 | 81.7797 | 93.7292 | FAIL* | PASS (Grade B)* |
| T10-8 Last block + direction + temporal | VitalDB | SBP | 9.8760 | 0.5207 | -0.3876 | 13.2475 | 34.2365 | 61.0775 | 78.2868 | FAIL* | FAIL (Grade D)* |
| T10-8 Last block + direction + temporal | VitalDB | DBP | 6.0390 | 0.5531 | -0.6837 | 8.4086 | 52.4842 | 82.1577 | 93.7839 | FAIL* | PASS (Grade B)* |

## Reproducibility and public artifacts

- [Participant-macro results](../results/round10/participant_macro_internal.csv)
- [Comparison with the frozen reference](../results/round10/comparison_vs_reference_internal.csv)
- [Complete pooled diagnostics](../results/round10/pooled_diagnostics_internal.csv)
- [Selection record](../results/round10/selection.json)
- [Prospective Round-10 plan](ROUND10_PARTIAL_END_TO_END_PLAN.md)

The AAMI/BHS columns are retrospective numerical screens only. They do not
establish device, clinical, or standards compliance.

Internal winner: **T10-8 Last block + direction + temporal**.

Internal promotion gate passed: **False**.
