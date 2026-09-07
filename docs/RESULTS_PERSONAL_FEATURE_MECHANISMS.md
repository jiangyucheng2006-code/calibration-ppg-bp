# 个人特征适配机制：完整结果

更新：2026-09-07。本文为正式开发结果汇总，不是封存测试集结果。

## 结论

8 个候选 × 2 种划分全部完成。没有替代方法超过同期 rank-4 个人 LoRA 并满足预设升级门槛，保留 `subject_lora_rank4`。
共享适配、压缩个人参数、非线性响应都未产生确认性提升；不能将模型改名或更复杂视为创新有效。

原随机 `shared_bilinear64` 作业 1486 因 DataLoader 共享内存错误失败。仅该项重跑为 1565，已 COMPLETED 0:0，用时 5:54:57；随机报告 1566 与最终报告 1567 已完成。
重跑使用原科学代码快照和相同种子/模型/数据/优化规则，仅 workers=4 改为 0，从 epoch 0 重训，并非最佳 checkpoint 的精确断点续训。不承诺不同数据加载策略逐位等价；原失败记录保留。

## 统一实验边界

- 数据协议：`development-calbased-analogue-v1`，只使用父级 meta-train 人群。
- 2,051 人：MIMIC 1,011 人、VitalDB 1,040 人；每人 320 个带 BP 标签的 10 秒训练窗口、40 个内部验证窗口、40 个仍封存窗口。
- 每种划分验证覆盖全部 82,040 个窗口，没有剔除最差 30% 或低质量验证窗口。
- 同一人的训练与内部验证允许重合身份，但窗口/生理区间/重复波形按既有数据审计分离。随机划分反映已知用户数据内插值；时间划分更接近使用过去训练片段预测较后片段。
- 本批为单种子开发探索；内部验证可选 checkpoint，但不能更新个人状态或充当输入。没有打开封存 held-out 或独立 meta-test。
- 这是自建 CalBased analogue，不是官方 PulseDB CalBased 固定划分的严格复现；也不是新用户 K-shot、320 次袖带校准或独立外部验证。
- MIMIC/VitalDB 是 PulseDB 内部分来源分层。Overall 保留原报告对全体受试者重算的结果，不将两来源简单平均。
- 升级门槛：同一方法在两种划分的 Overall mean MAE 均改善至少 0.15 mmHg，并在两个来源均改善。

## 候选及两种划分总体结果

| Setting | 改变 | Random mean MAE | Chronological mean MAE |
| --- | --- | --- | --- |
| subject_lora_rank4 | 个人线性 rank-4 特征变换（2,048 参数/人） | 3.0473 | 3.7994 |
| shared_lora_rank4 | 所有人共用一个 rank-4 适配器 | 6.0726 | 6.2500 |
| subject_lora_rank1 | 缩小至个人 rank-1（512 参数/人） | 3.2959 | 3.9562 |
| output_profile32 | 32D 个人代码在输出端修正（34 参数/人） | 4.0816 | 4.3698 |
| feature_affine32 | 个人代码控制特征尺度与平移（34 参数/人） | 3.7380 | 4.1372 |
| shared_bilinear32 | 共享方向、个人32D系数（34 参数/人） | 3.4871 | 3.9973 |
| shared_bilinear64 | 共享方向、个人64D系数（66 参数/人） | 3.3239 | 3.9042 |
| subject_nonlinear_rank4 | 相同2,048参数，rank-4中加入非线性 | 3.0525 | 3.8829 |

单位 mmHg，mean MAE=(SBP MAE+DBP MAE)/2，受试者等权；越低越好。

## 全部受试者等权结果

### random_disjoint

| candidate | view | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | MIMIC | 1011 | 40440 | 4.2728 | 2.3738 | 3.3233 |
| subject_nonlinear_rank4 | MIMIC | 1011 | 40440 | 4.3022 | 2.3644 | 3.3333 |
| subject_lora_rank1 | MIMIC | 1011 | 40440 | 4.5788 | 2.5862 | 3.5825 |
| shared_bilinear64 | MIMIC | 1011 | 40440 | 4.6238 | 2.5821 | 3.6030 |
| shared_bilinear32 | MIMIC | 1011 | 40440 | 4.8282 | 2.6926 | 3.7604 |
| feature_affine32 | MIMIC | 1011 | 40440 | 5.1601 | 2.8450 | 4.0025 |
| output_profile32 | MIMIC | 1011 | 40440 | 5.6080 | 3.0817 | 4.3449 |
| shared_lora_rank4 | MIMIC | 1011 | 40440 | 7.8631 | 4.2371 | 6.0501 |
| subject_lora_rank4 | Overall | 2051 | 82040 | 3.9056 | 2.1890 | 3.0473 |
| subject_nonlinear_rank4 | Overall | 2051 | 82040 | 3.9231 | 2.1820 | 3.0525 |
| subject_lora_rank1 | Overall | 2051 | 82040 | 4.2052 | 2.3866 | 3.2959 |
| shared_bilinear64 | Overall | 2051 | 82040 | 4.2543 | 2.3935 | 3.3239 |
| shared_bilinear32 | Overall | 2051 | 82040 | 4.4605 | 2.5137 | 3.4871 |
| feature_affine32 | Overall | 2051 | 82040 | 4.8025 | 2.6735 | 3.7380 |
| output_profile32 | Overall | 2051 | 82040 | 5.2492 | 2.9141 | 4.0816 |
| shared_lora_rank4 | Overall | 2051 | 82040 | 7.8625 | 4.2826 | 6.0726 |
| subject_lora_rank4 | VitalDB | 1040 | 41600 | 3.5486 | 2.0094 | 2.7790 |
| subject_nonlinear_rank4 | VitalDB | 1040 | 41600 | 3.5545 | 2.0047 | 2.7796 |
| subject_lora_rank1 | VitalDB | 1040 | 41600 | 3.8420 | 2.1925 | 3.0172 |
| shared_bilinear64 | VitalDB | 1040 | 41600 | 3.8951 | 2.2101 | 3.0526 |
| shared_bilinear32 | VitalDB | 1040 | 41600 | 4.1031 | 2.3398 | 3.2215 |
| feature_affine32 | VitalDB | 1040 | 41600 | 4.4549 | 2.5068 | 3.4809 |
| output_profile32 | VitalDB | 1040 | 41600 | 4.9004 | 2.7512 | 3.8258 |
| shared_lora_rank4 | VitalDB | 1040 | 41600 | 7.8620 | 4.3269 | 6.0944 |

### chronological_blocked

| candidate | view | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | MIMIC | 1011 | 40440 | 4.6699 | 2.5140 | 3.5920 |
| subject_nonlinear_rank4 | MIMIC | 1011 | 40440 | 4.7142 | 2.5487 | 3.6314 |
| shared_bilinear64 | MIMIC | 1011 | 40440 | 4.8251 | 2.5878 | 3.7064 |
| subject_lora_rank1 | MIMIC | 1011 | 40440 | 4.8549 | 2.7074 | 3.7812 |
| shared_bilinear32 | MIMIC | 1011 | 40440 | 4.9141 | 2.6625 | 3.7883 |
| feature_affine32 | MIMIC | 1011 | 40440 | 5.1909 | 2.7804 | 3.9856 |
| output_profile32 | MIMIC | 1011 | 40440 | 5.6127 | 2.9448 | 4.2787 |
| shared_lora_rank4 | MIMIC | 1011 | 40440 | 7.3415 | 3.8649 | 5.6032 |
| subject_lora_rank4 | Overall | 2051 | 82040 | 4.9023 | 2.6965 | 3.7994 |
| subject_nonlinear_rank4 | Overall | 2051 | 82040 | 4.9957 | 2.7702 | 3.8829 |
| shared_bilinear64 | Overall | 2051 | 82040 | 5.0136 | 2.7947 | 3.9042 |
| subject_lora_rank1 | Overall | 2051 | 82040 | 5.0628 | 2.8496 | 3.9562 |
| shared_bilinear32 | Overall | 2051 | 82040 | 5.1327 | 2.8619 | 3.9973 |
| feature_affine32 | Overall | 2051 | 82040 | 5.3514 | 2.9229 | 4.1372 |
| output_profile32 | Overall | 2051 | 82040 | 5.6557 | 3.0840 | 4.3698 |
| shared_lora_rank4 | Overall | 2051 | 82040 | 8.0892 | 4.4107 | 6.2500 |
| subject_lora_rank4 | VitalDB | 1040 | 41600 | 5.1282 | 2.8738 | 4.0010 |
| shared_bilinear64 | VitalDB | 1040 | 41600 | 5.1969 | 2.9958 | 4.0963 |
| subject_lora_rank1 | VitalDB | 1040 | 41600 | 5.2649 | 2.9877 | 4.1263 |
| subject_nonlinear_rank4 | VitalDB | 1040 | 41600 | 5.2693 | 2.9856 | 4.1274 |
| shared_bilinear32 | VitalDB | 1040 | 41600 | 5.3452 | 3.0557 | 4.2004 |
| feature_affine32 | VitalDB | 1040 | 41600 | 5.5075 | 3.0615 | 4.2845 |
| output_profile32 | VitalDB | 1040 | 41600 | 5.6975 | 3.2193 | 4.4584 |
| shared_lora_rank4 | VitalDB | 1040 | 41600 | 8.8161 | 4.9413 | 6.8787 |

## 要求的完整指标表

下表来自同一批预测的逐窗口诊断：ME=预测−参考血压；STD 为预测误差的样本标准差，不是跨种子的 MAE 标准差。阈值列为累计百分比。
每人恰有40个验证窗口，所以本批 MAE 与受试者等权值仅有数值舍入差；R²、ME、STD和阈值列仍按逐窗口定义计算。
`AAMI*`/`BHS*` 只表示回顾性数值筛查，既不等于临床或设备认证，也不是正式标准验证研究通过。

### random_disjoint · Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | SBP | 3.9056 | 0.9249 | 0.0683 | 5.8011 | 74.0566 | 92.9876 | 97.4707 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank4 | DBP | 2.1890 | 0.9209 | -0.0976 | 3.6062 | 91.2006 | 98.2045 | 99.3698 | PASS\* | PASS (Grade A)\* |
| shared_lora_rank4 | SBP | 7.8625 | 0.7345 | -0.5334 | 10.8915 | 45.0219 | 72.3147 | 86.2896 | FAIL\* | FAIL (Grade C)\* |
| shared_lora_rank4 | DBP | 4.2826 | 0.7674 | -0.3757 | 6.1754 | 69.6490 | 91.3774 | 97.3098 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank1 | SBP | 4.2052 | 0.9157 | 0.1756 | 6.1412 | 70.9447 | 91.7882 | 97.1331 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank1 | DBP | 2.3866 | 0.9089 | -0.1414 | 3.8679 | 89.6794 | 97.9510 | 99.2479 | PASS\* | PASS (Grade A)\* |
| output_profile32 | SBP | 5.2492 | 0.8741 | -0.1578 | 7.5081 | 61.4968 | 86.7357 | 94.9037 | PASS\* | PASS (Grade B)\* |
| output_profile32 | DBP | 2.9141 | 0.8773 | -0.1906 | 4.4893 | 84.3747 | 96.8613 | 98.9871 | PASS\* | PASS (Grade A)\* |
| feature_affine32 | SBP | 4.8025 | 0.8929 | 0.2377 | 6.9235 | 65.3474 | 88.9493 | 95.9837 | PASS\* | PASS (Grade A)\* |
| feature_affine32 | DBP | 2.6735 | 0.8932 | -0.1355 | 4.1901 | 86.7906 | 97.3805 | 99.1175 | PASS\* | PASS (Grade A)\* |
| shared_bilinear32 | SBP | 4.4605 | 0.9059 | -0.2256 | 6.4893 | 68.5897 | 90.6643 | 96.6663 | PASS\* | PASS (Grade A)\* |
| shared_bilinear32 | DBP | 2.5137 | 0.9024 | -0.0712 | 4.0064 | 88.6190 | 97.7499 | 99.2138 | PASS\* | PASS (Grade A)\* |
| shared_bilinear64 | SBP | 4.2543 | 0.9130 | 0.1381 | 6.2408 | 70.5997 | 91.5383 | 96.9954 | PASS\* | PASS (Grade A)\* |
| shared_bilinear64 | DBP | 2.3935 | 0.9095 | -0.1431 | 3.8556 | 89.5015 | 97.9364 | 99.2540 | PASS\* | PASS (Grade A)\* |
| subject_nonlinear_rank4 | SBP | 3.9231 | 0.9228 | -0.0571 | 5.8790 | 74.1541 | 92.7133 | 97.4220 | PASS\* | PASS (Grade A)\* |
| subject_nonlinear_rank4 | DBP | 2.1820 | 0.9202 | -0.1402 | 3.6213 | 91.1555 | 98.1838 | 99.3759 | PASS\* | PASS (Grade A)\* |

### random_disjoint · MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | SBP | 4.2728 | 0.9212 | 0.1551 | 6.3877 | 70.9322 | 91.1474 | 96.5974 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank4 | DBP | 2.3738 | 0.9044 | -0.1564 | 4.1077 | 89.7255 | 97.4036 | 99.0010 | PASS\* | PASS (Grade A)\* |
| shared_lora_rank4 | SBP | 7.8631 | 0.7666 | -0.5684 | 10.9818 | 45.5935 | 72.5025 | 86.2735 | FAIL\* | FAIL (Grade C)\* |
| shared_lora_rank4 | DBP | 4.2371 | 0.7687 | -0.5116 | 6.3745 | 71.4243 | 91.6098 | 97.0005 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank1 | SBP | 4.5788 | 0.9121 | 0.3096 | 6.7409 | 68.0440 | 89.8294 | 96.1944 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank1 | DBP | 2.5862 | 0.8898 | -0.2140 | 4.4090 | 88.1553 | 97.1785 | 98.8056 | PASS\* | PASS (Grade A)\* |
| output_profile32 | SBP | 5.6080 | 0.8750 | 0.0812 | 8.0469 | 59.0603 | 84.7206 | 93.8798 | FAIL\* | PASS (Grade B)\* |
| output_profile32 | DBP | 3.0817 | 0.8608 | -0.2293 | 4.9553 | 83.3531 | 96.0856 | 98.5608 | PASS\* | PASS (Grade A)\* |
| feature_affine32 | SBP | 5.1601 | 0.8922 | 0.2606 | 7.4678 | 62.7003 | 87.1217 | 94.9728 | PASS\* | PASS (Grade B)\* |
| feature_affine32 | DBP | 2.8450 | 0.8769 | -0.2478 | 4.6578 | 85.5292 | 96.6864 | 98.6597 | PASS\* | PASS (Grade A)\* |
| shared_bilinear32 | SBP | 4.8282 | 0.9029 | -0.2045 | 7.0897 | 65.8877 | 88.7141 | 95.6503 | PASS\* | PASS (Grade A)\* |
| shared_bilinear32 | DBP | 2.6926 | 0.8847 | -0.2262 | 4.5094 | 87.2453 | 96.9758 | 98.7933 | PASS\* | PASS (Grade A)\* |
| shared_bilinear64 | SBP | 4.6238 | 0.9095 | 0.2339 | 6.8431 | 67.7399 | 89.6884 | 96.0509 | PASS\* | PASS (Grade A)\* |
| shared_bilinear64 | DBP | 2.5821 | 0.8921 | -0.1807 | 4.3634 | 88.0119 | 97.1958 | 98.8229 | PASS\* | PASS (Grade A)\* |
| subject_nonlinear_rank4 | SBP | 4.3022 | 0.9192 | 0.0787 | 6.4689 | 71.0213 | 90.8531 | 96.5232 | PASS\* | PASS (Grade A)\* |
| subject_nonlinear_rank4 | DBP | 2.3644 | 0.9044 | -0.1847 | 4.1058 | 89.6810 | 97.4827 | 99.0282 | PASS\* | PASS (Grade A)\* |

### random_disjoint · VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | SBP | 3.5486 | 0.9256 | -0.0160 | 5.1661 | 77.0938 | 94.7764 | 98.3197 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank4 | DBP | 2.0094 | 0.9384 | -0.0405 | 3.0394 | 92.6346 | 98.9832 | 99.7284 | PASS\* | PASS (Grade A)\* |
| shared_lora_rank4 | SBP | 7.8620 | 0.6741 | -0.4994 | 10.8031 | 44.4663 | 72.1322 | 86.3053 | FAIL\* | FAIL (Grade C)\* |
| shared_lora_rank4 | DBP | 4.3269 | 0.7620 | -0.2437 | 5.9726 | 67.9231 | 91.1514 | 97.6106 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank1 | SBP | 3.8420 | 0.9159 | 0.0454 | 5.4927 | 73.7644 | 93.6923 | 98.0457 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank1 | DBP | 2.1925 | 0.9294 | -0.0708 | 3.2554 | 91.1611 | 98.7019 | 99.6779 | PASS\* | PASS (Grade A)\* |
| output_profile32 | SBP | 4.9004 | 0.8655 | -0.3901 | 6.9366 | 63.8654 | 88.6947 | 95.8990 | PASS\* | PASS (Grade A)\* |
| output_profile32 | DBP | 2.7512 | 0.8941 | -0.1529 | 3.9841 | 85.3678 | 97.6154 | 99.4014 | PASS\* | PASS (Grade A)\* |
| feature_affine32 | SBP | 4.4549 | 0.8875 | 0.2155 | 6.3497 | 67.9207 | 90.7260 | 96.9663 | PASS\* | PASS (Grade A)\* |
| feature_affine32 | DBP | 2.5068 | 0.9100 | -0.0263 | 3.6756 | 88.0168 | 98.0553 | 99.5625 | PASS\* | PASS (Grade A)\* |
| shared_bilinear32 | SBP | 4.1031 | 0.9046 | -0.2462 | 5.8469 | 71.2163 | 92.5601 | 97.6538 | PASS\* | PASS (Grade A)\* |
| shared_bilinear32 | DBP | 2.3398 | 0.9211 | 0.0795 | 3.4411 | 89.9543 | 98.5024 | 99.6226 | PASS\* | PASS (Grade A)\* |
| shared_bilinear64 | SBP | 3.8951 | 0.9129 | 0.0449 | 5.5921 | 73.3798 | 93.3365 | 97.9135 | PASS\* | PASS (Grade A)\* |
| shared_bilinear64 | DBP | 2.2101 | 0.9279 | -0.1065 | 3.2873 | 90.9495 | 98.6562 | 99.6731 | PASS\* | PASS (Grade A)\* |
| subject_nonlinear_rank4 | SBP | 3.5545 | 0.9234 | -0.1891 | 5.2391 | 77.1995 | 94.5216 | 98.2957 | PASS\* | PASS (Grade A)\* |
| subject_nonlinear_rank4 | DBP | 2.0047 | 0.9368 | -0.0970 | 3.0775 | 92.5889 | 98.8654 | 99.7139 | PASS\* | PASS (Grade A)\* |

### chronological_blocked · Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | SBP | 4.9023 | 0.8730 | 0.1900 | 7.3232 | 66.3067 | 87.8072 | 94.9086 | PASS\* | PASS (Grade B)\* |
| subject_lora_rank4 | DBP | 2.6965 | 0.8731 | 0.0292 | 4.4307 | 85.6509 | 96.5346 | 98.8079 | PASS\* | PASS (Grade A)\* |
| shared_lora_rank4 | SBP | 8.0892 | 0.7052 | 0.2679 | 11.1576 | 44.3893 | 71.1177 | 84.7526 | FAIL\* | FAIL (Grade D)\* |
| shared_lora_rank4 | DBP | 4.4107 | 0.7384 | -0.1540 | 6.3602 | 68.5519 | 89.9208 | 96.8333 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank1 | SBP | 5.0628 | 0.8712 | -0.2251 | 7.3749 | 64.0383 | 87.1563 | 94.7782 | PASS\* | PASS (Grade B)\* |
| subject_lora_rank1 | DBP | 2.8496 | 0.8670 | -0.2050 | 4.5314 | 84.4088 | 96.3396 | 98.8140 | PASS\* | PASS (Grade A)\* |
| output_profile32 | SBP | 5.6557 | 0.8477 | 0.5405 | 8.0037 | 58.5251 | 84.2028 | 93.5605 | FAIL\* | PASS (Grade B)\* |
| output_profile32 | DBP | 3.0840 | 0.8549 | -0.0130 | 4.7373 | 82.3830 | 95.9739 | 98.6970 | PASS\* | PASS (Grade A)\* |
| feature_affine32 | SBP | 5.3514 | 0.8622 | 0.5490 | 7.6099 | 60.9434 | 85.8350 | 94.4198 | PASS\* | PASS (Grade B)\* |
| feature_affine32 | DBP | 2.9229 | 0.8655 | -0.0370 | 4.5621 | 83.6348 | 96.3603 | 98.8420 | PASS\* | PASS (Grade A)\* |
| shared_bilinear32 | SBP | 5.1327 | 0.8689 | 0.1795 | 7.4396 | 63.4569 | 86.7613 | 94.6953 | PASS\* | PASS (Grade B)\* |
| shared_bilinear32 | DBP | 2.8619 | 0.8634 | -0.0715 | 4.5965 | 84.3979 | 96.2104 | 98.7604 | PASS\* | PASS (Grade A)\* |
| shared_bilinear64 | SBP | 5.0136 | 0.8749 | 0.0300 | 7.2695 | 64.1724 | 87.3038 | 95.0902 | PASS\* | PASS (Grade A)\* |
| shared_bilinear64 | DBP | 2.7947 | 0.8692 | -0.0863 | 4.4972 | 85.0622 | 96.4639 | 98.8140 | PASS\* | PASS (Grade A)\* |
| subject_nonlinear_rank4 | SBP | 4.9957 | 0.8670 | -0.1738 | 7.4933 | 65.5375 | 87.2830 | 94.6112 | PASS\* | PASS (Grade B)\* |
| subject_nonlinear_rank4 | DBP | 2.7702 | 0.8675 | -0.1836 | 4.5241 | 85.2462 | 96.3164 | 98.7043 | PASS\* | PASS (Grade A)\* |

### chronological_blocked · MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | SBP | 4.6699 | 0.9012 | 0.2639 | 7.1184 | 68.9590 | 88.9095 | 95.3314 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank4 | DBP | 2.5140 | 0.8895 | 0.0069 | 4.2969 | 87.5717 | 96.5307 | 98.6944 | PASS\* | PASS (Grade A)\* |
| shared_lora_rank4 | SBP | 7.3415 | 0.7855 | 0.4297 | 10.4865 | 49.5697 | 76.0089 | 87.6830 | FAIL\* | FAIL (Grade C)\* |
| shared_lora_rank4 | DBP | 3.8649 | 0.7954 | -0.0033 | 5.8469 | 75.0173 | 92.3393 | 97.3269 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank1 | SBP | 4.8549 | 0.8977 | -0.2766 | 7.2432 | 66.4367 | 88.0069 | 95.1261 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank1 | DBP | 2.7074 | 0.8789 | -0.1301 | 4.4959 | 86.3922 | 96.1845 | 98.5262 | PASS\* | PASS (Grade A)\* |
| output_profile32 | SBP | 5.6127 | 0.8705 | 0.5472 | 8.1376 | 59.6637 | 84.6439 | 93.5015 | FAIL\* | PASS (Grade B)\* |
| output_profile32 | DBP | 2.9448 | 0.8679 | 0.0192 | 4.6980 | 84.4041 | 95.8605 | 98.3160 | PASS\* | PASS (Grade A)\* |
| feature_affine32 | SBP | 5.1909 | 0.8864 | 0.5490 | 7.6195 | 63.3828 | 86.7112 | 94.5203 | PASS\* | PASS (Grade B)\* |
| feature_affine32 | DBP | 2.7804 | 0.8776 | -0.0398 | 4.5220 | 85.3586 | 96.2933 | 98.5435 | PASS\* | PASS (Grade A)\* |
| shared_bilinear32 | SBP | 4.9141 | 0.8982 | 0.0712 | 7.2288 | 66.1177 | 87.7423 | 94.7948 | PASS\* | PASS (Grade B)\* |
| shared_bilinear32 | DBP | 2.6625 | 0.8784 | -0.1008 | 4.5060 | 86.7878 | 96.2611 | 98.5064 | PASS\* | PASS (Grade A)\* |
| shared_bilinear64 | SBP | 4.8251 | 0.9005 | 0.0598 | 7.1466 | 66.5727 | 88.0045 | 95.2621 | PASS\* | PASS (Grade A)\* |
| shared_bilinear64 | DBP | 2.5878 | 0.8854 | -0.0896 | 4.3758 | 87.2478 | 96.5752 | 98.7141 | PASS\* | PASS (Grade A)\* |
| subject_nonlinear_rank4 | SBP | 4.7142 | 0.9004 | -0.2560 | 7.1487 | 68.8724 | 88.6177 | 94.8764 | PASS\* | PASS (Grade B)\* |
| subject_nonlinear_rank4 | DBP | 2.5487 | 0.8884 | -0.2379 | 4.3112 | 87.4308 | 96.3205 | 98.6078 | PASS\* | PASS (Grade A)\* |

### chronological_blocked · VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | SBP | 5.1282 | 0.8212 | 0.1181 | 7.5163 | 63.7284 | 86.7356 | 94.4976 | PASS\* | PASS (Grade B)\* |
| subject_lora_rank4 | DBP | 2.8738 | 0.8497 | 0.0509 | 4.5569 | 83.7837 | 96.5385 | 98.9183 | PASS\* | PASS (Grade A)\* |
| shared_lora_rank4 | SBP | 8.8161 | 0.5615 | 0.1106 | 11.7714 | 39.3534 | 66.3630 | 81.9038 | FAIL\* | FAIL (Grade D)\* |
| shared_lora_rank4 | DBP | 4.9413 | 0.6627 | -0.3004 | 6.8192 | 62.2668 | 87.5697 | 96.3534 | PASS\* | PASS (Grade A)\* |
| subject_lora_rank1 | SBP | 5.2649 | 0.8219 | -0.1750 | 7.5005 | 61.7067 | 86.3293 | 94.4399 | PASS\* | PASS (Grade B)\* |
| subject_lora_rank1 | DBP | 2.9877 | 0.8486 | -0.2778 | 4.5645 | 82.4808 | 96.4904 | 99.0938 | PASS\* | PASS (Grade A)\* |
| output_profile32 | SBP | 5.6975 | 0.8030 | 0.5340 | 7.8715 | 57.4183 | 83.7740 | 93.6178 | PASS\* | PASS (Grade B)\* |
| output_profile32 | DBP | 3.2193 | 0.8349 | -0.0443 | 4.7751 | 80.4183 | 96.0841 | 99.0673 | PASS\* | PASS (Grade A)\* |
| feature_affine32 | SBP | 5.5075 | 0.8162 | 0.5491 | 7.6006 | 58.5721 | 84.9832 | 94.3221 | PASS\* | PASS (Grade B)\* |
| feature_affine32 | DBP | 3.0615 | 0.8468 | -0.0343 | 4.6007 | 81.9591 | 96.4255 | 99.1322 | PASS\* | PASS (Grade A)\* |
| shared_bilinear32 | SBP | 5.3452 | 0.8152 | 0.2849 | 7.6375 | 60.8702 | 85.8077 | 94.5986 | PASS\* | PASS (Grade B)\* |
| shared_bilinear32 | DBP | 3.0557 | 0.8413 | -0.0430 | 4.6827 | 82.0745 | 96.1611 | 99.0072 | PASS\* | PASS (Grade A)\* |
| shared_bilinear64 | SBP | 5.1969 | 0.8273 | 0.0011 | 7.3869 | 61.8389 | 86.6226 | 94.9231 | PASS\* | PASS (Grade B)\* |
| shared_bilinear64 | DBP | 2.9958 | 0.8460 | -0.0830 | 4.6122 | 82.9375 | 96.3558 | 98.9111 | PASS\* | PASS (Grade A)\* |
| subject_nonlinear_rank4 | SBP | 5.2693 | 0.8068 | -0.0939 | 7.8130 | 62.2957 | 85.9856 | 94.3534 | PASS\* | PASS (Grade B)\* |
| subject_nonlinear_rank4 | DBP | 2.9856 | 0.8385 | -0.1308 | 4.7212 | 83.1226 | 96.3125 | 98.7981 | PASS\* | PASS (Grade A)\* |

## 解释与下一步

- 先前冻结诊断支持正确个人状态与当前PPG均有作用，本次训练比较进一步说明：直接保留个人特征变换比已测的输出压缩修正更准确。
- shared 与 personal 也改变个人容量，因此不能把它们的差距全部归因为身份机制；rank4 线性/非线性是较干净的同容量比较，非线性没有获益。
- 不将失败的 shared_bilinear64 最佳中间 checkpoint 当正式结果。本表使用成功重跑及完整验证输出。
- 后续应提出有限、有明确证据门槛的假设，保留强 LoRA 参照；本批不支持再无区别叠模块。

## 可复核文件

- [跨划分比较](../results/personal_feature_mechanisms/cross_split_comparison.csv)；[升级门槛](../results/personal_feature_mechanisms/promotion_gate.csv)；[最终选择](../results/personal_feature_mechanisms/selection.json)。
- [数据与报告核查](../results/personal_feature_mechanisms/publication_audit.json)；[作业完成记录](../results/personal_feature_mechanisms/job_states.csv)。
- [随机总体诊断](../results/personal_feature_mechanisms/random_disjoint/event_pooled_overall.csv)、[MIMIC](../results/personal_feature_mechanisms/random_disjoint/event_pooled_mimic.csv)、[VitalDB](../results/personal_feature_mechanisms/random_disjoint/event_pooled_vitaldb.csv)。
- [时间总体诊断](../results/personal_feature_mechanisms/chronological_blocked/event_pooled_overall.csv)、[MIMIC](../results/personal_feature_mechanisms/chronological_blocked/event_pooled_mimic.csv)、[VitalDB](../results/personal_feature_mechanisms/chronological_blocked/event_pooled_vitaldb.csv)。

本次发布仅复制并核查服务器已完成的聚合报告，不下载受试者级预测或原始波形；不会将聚合核查描述成重新运行了全部患者预测。
已有多轮内部验证探索存在选择偏差；正式论文的效应和不确定性需要后续冻结方案及独立确认。
