# LoRA + PRS 继续训练：完整结果

更新：2026-09-07。本文为正式开发结果汇总，不是封存测试集结果。

## 结论

5 个候选 × 2 种划分全部完成。两个划分的最低 Overall mean MAE 均来自单纯继续训练 LoRA（`lora_continue`）；四种 PRS 组合均未胜过这个同期对照。
相对原初始化的下降不能自动算作 PRS 模块收益。进一步降低学习率继续训练本身已有收益，PRS 的额外效果必须与同期继续训练的 LoRA 比。

随机报告 1505、时间报告 1512、最终报告 1513 均 COMPLETED 0:0；最终选择仍保留 LoRA。

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
| lora_continue | 只继续训练原 LoRA，同期对照 | 2.8801 | 3.7124 |
| lora_prs_bias | LoRA + 每人两个输出偏差 | 2.8850 | 3.7135 |
| lora_prs_dynamic | LoRA + 32D个人代码与波形相关输出修正 | 2.8937 | 3.7130 |
| lora_prs_dynamic_shrink | 上述动态修正加抑制过大修正的惩罚 | 2.8912 | 3.7128 |
| lora_prs_dynamic_frozen | 冻结原 LoRA，仅训练新个人修正 | 2.9087 | 3.7392 |

单位 mmHg，mean MAE=(SBP MAE+DBP MAE)/2，受试者等权；越低越好。

### 单纯继续训练的收益

| split_mode | initial_reference_mean_mae | mean_mae | gain_vs_initial |
| --- | --- | --- | --- |
| random_disjoint | 3.0394 | 2.8801 | 0.1593 |
| chronological_blocked | 3.7945 | 3.7124 | 0.0821 |

两个划分相对初始化的收益并未同时达到0.15 mmHg，因此这也不能被写成通过旧结构升级门槛的新方法。

## 全部受试者等权结果

### random_disjoint

| candidate | view | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- | --- |
| lora_continue | MIMIC | 1011 | 40440 | 4.0434 | 2.2457 | 3.1446 |
| lora_prs_bias | MIMIC | 1011 | 40440 | 4.0492 | 2.2498 | 3.1495 |
| lora_prs_dynamic_shrink | MIMIC | 1011 | 40440 | 4.0611 | 2.2563 | 3.1587 |
| lora_prs_dynamic | MIMIC | 1011 | 40440 | 4.0632 | 2.2588 | 3.1610 |
| lora_prs_dynamic_frozen | MIMIC | 1011 | 40440 | 4.0894 | 2.2681 | 3.1787 |
| lora_continue | Overall | 2051 | 82040 | 3.6913 | 2.0689 | 2.8801 |
| lora_prs_bias | Overall | 2051 | 82040 | 3.6974 | 2.0726 | 2.8850 |
| lora_prs_dynamic_shrink | Overall | 2051 | 82040 | 3.7034 | 2.0791 | 2.8912 |
| lora_prs_dynamic | Overall | 2051 | 82040 | 3.7058 | 2.0816 | 2.8937 |
| lora_prs_dynamic_frozen | Overall | 2051 | 82040 | 3.7274 | 2.0900 | 2.9087 |
| lora_continue | VitalDB | 1040 | 41600 | 3.3490 | 1.8970 | 2.6230 |
| lora_prs_bias | VitalDB | 1040 | 41600 | 3.3553 | 1.9004 | 2.6279 |
| lora_prs_dynamic_shrink | VitalDB | 1040 | 41600 | 3.3558 | 1.9067 | 2.6313 |
| lora_prs_dynamic | VitalDB | 1040 | 41600 | 3.3583 | 1.9094 | 2.6338 |
| lora_prs_dynamic_frozen | VitalDB | 1040 | 41600 | 3.3755 | 1.9169 | 2.6462 |

### chronological_blocked

| candidate | view | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- | --- |
| lora_prs_dynamic | MIMIC | 1011 | 40440 | 4.5055 | 2.4472 | 3.4763 |
| lora_continue | MIMIC | 1011 | 40440 | 4.5111 | 2.4435 | 3.4773 |
| lora_prs_dynamic_shrink | MIMIC | 1011 | 40440 | 4.5105 | 2.4458 | 3.4782 |
| lora_prs_bias | MIMIC | 1011 | 40440 | 4.5134 | 2.4460 | 3.4797 |
| lora_prs_dynamic_frozen | MIMIC | 1011 | 40440 | 4.5480 | 2.4726 | 3.5103 |
| lora_continue | Overall | 2051 | 82040 | 4.7773 | 2.6475 | 3.7124 |
| lora_prs_dynamic_shrink | Overall | 2051 | 82040 | 4.7761 | 2.6496 | 3.7128 |
| lora_prs_dynamic | Overall | 2051 | 82040 | 4.7743 | 2.6517 | 3.7130 |
| lora_prs_bias | Overall | 2051 | 82040 | 4.7779 | 2.6491 | 3.7135 |
| lora_prs_dynamic_frozen | Overall | 2051 | 82040 | 4.8101 | 2.6683 | 3.7392 |
| lora_prs_bias | VitalDB | 1040 | 41600 | 5.0351 | 2.8465 | 3.9408 |
| lora_continue | VitalDB | 1040 | 41600 | 5.0361 | 2.8457 | 3.9409 |
| lora_prs_dynamic_shrink | VitalDB | 1040 | 41600 | 5.0342 | 2.8477 | 3.9410 |
| lora_prs_dynamic | VitalDB | 1040 | 41600 | 5.0356 | 2.8505 | 3.9430 |
| lora_prs_dynamic_frozen | VitalDB | 1040 | 41600 | 5.0649 | 2.8586 | 3.9617 |

## 要求的完整指标表

下表来自同一批预测的逐窗口诊断：ME=预测−参考血压；STD 为预测误差的样本标准差，不是跨种子的 MAE 标准差。阈值列为累计百分比。
每人恰有40个验证窗口，所以本批 MAE 与受试者等权值仅有数值舍入差；R²、ME、STD和阈值列仍按逐窗口定义计算。
`AAMI*`/`BHS*` 只表示回顾性数值筛查，既不等于临床或设备认证，也不是正式标准验证研究通过。

### random_disjoint · Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora_continue | SBP | 3.6913 | 0.9303 | 0.0225 | 5.5876 | 76.1824 | 93.4910 | 97.7377 | PASS\* | PASS (Grade A)\* |
| lora_continue | DBP | 2.0689 | 0.9257 | -0.1183 | 3.4938 | 91.8686 | 98.3191 | 99.4149 | PASS\* | PASS (Grade A)\* |
| lora_prs_bias | SBP | 3.6974 | 0.9302 | 0.0197 | 5.5911 | 76.0763 | 93.5129 | 97.7426 | PASS\* | PASS (Grade A)\* |
| lora_prs_bias | DBP | 2.0726 | 0.9256 | -0.1121 | 3.4961 | 91.8613 | 98.3057 | 99.4088 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic | SBP | 3.7058 | 0.9300 | 0.0298 | 5.6007 | 76.0690 | 93.4532 | 97.7401 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic | DBP | 2.0816 | 0.9253 | -0.0138 | 3.5051 | 91.8150 | 98.3118 | 99.4088 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_shrink | SBP | 3.7034 | 0.9300 | 0.0223 | 5.5985 | 76.1056 | 93.4593 | 97.7292 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_shrink | DBP | 2.0791 | 0.9254 | -0.0122 | 3.5027 | 91.8308 | 98.3228 | 99.4100 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_frozen | SBP | 3.7274 | 0.9292 | 0.0458 | 5.6320 | 75.8411 | 93.3922 | 97.7401 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_frozen | DBP | 2.0900 | 0.9248 | -0.1267 | 3.5157 | 91.6882 | 98.2972 | 99.3991 | PASS\* | PASS (Grade A)\* |

### random_disjoint · MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora_continue | SBP | 4.0434 | 0.9271 | 0.0468 | 6.1448 | 73.2122 | 91.8867 | 96.9807 | PASS\* | PASS (Grade A)\* |
| lora_continue | DBP | 2.2457 | 0.9105 | -0.1874 | 3.9738 | 90.3783 | 97.6706 | 99.0653 | PASS\* | PASS (Grade A)\* |
| lora_prs_bias | SBP | 4.0492 | 0.9270 | 0.0498 | 6.1479 | 73.1182 | 91.8991 | 97.0005 | PASS\* | PASS (Grade A)\* |
| lora_prs_bias | DBP | 2.2498 | 0.9104 | -0.1831 | 3.9768 | 90.3462 | 97.6558 | 99.0554 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic | SBP | 4.0632 | 0.9266 | 0.0462 | 6.1655 | 73.1009 | 91.8373 | 96.9609 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic | DBP | 2.2588 | 0.9100 | -0.0752 | 3.9876 | 90.3759 | 97.6954 | 99.0504 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_shrink | SBP | 4.0611 | 0.9267 | 0.0403 | 6.1628 | 73.1083 | 91.8249 | 96.9510 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_shrink | DBP | 2.2563 | 0.9101 | -0.0752 | 3.9857 | 90.4055 | 97.7052 | 99.0554 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_frozen | SBP | 4.0894 | 0.9255 | 0.0809 | 6.2127 | 72.7522 | 91.7507 | 96.9881 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_frozen | DBP | 2.2681 | 0.9093 | -0.1862 | 3.9996 | 90.2646 | 97.6607 | 99.0307 | PASS\* | PASS (Grade A)\* |

### random_disjoint · VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora_continue | SBP | 3.3490 | 0.9307 | -0.0011 | 4.9867 | 79.0697 | 95.0505 | 98.4736 | PASS\* | PASS (Grade A)\* |
| lora_continue | DBP | 1.8970 | 0.9419 | -0.0510 | 2.9518 | 93.3173 | 98.9495 | 99.7548 | PASS\* | PASS (Grade A)\* |
| lora_prs_bias | SBP | 3.3553 | 0.9306 | -0.0096 | 4.9905 | 78.9519 | 95.0817 | 98.4639 | PASS\* | PASS (Grade A)\* |
| lora_prs_bias | DBP | 1.9004 | 0.9419 | -0.0431 | 2.9532 | 93.3341 | 98.9375 | 99.7524 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic | SBP | 3.3583 | 0.9306 | 0.0138 | 4.9909 | 78.9543 | 95.0240 | 98.4976 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic | DBP | 1.9094 | 0.9416 | 0.0458 | 2.9605 | 93.2139 | 98.9111 | 99.7572 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_shrink | SBP | 3.3558 | 0.9306 | 0.0048 | 4.9892 | 79.0192 | 95.0481 | 98.4856 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_shrink | DBP | 1.9067 | 0.9417 | 0.0491 | 2.9573 | 93.2163 | 98.9231 | 99.7548 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_frozen | SBP | 3.3755 | 0.9302 | 0.0118 | 5.0032 | 78.8438 | 94.9880 | 98.4712 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_frozen | DBP | 1.9169 | 0.9412 | -0.0689 | 2.9696 | 93.0721 | 98.9159 | 99.7572 | PASS\* | PASS (Grade A)\* |

### chronological_blocked · Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora_continue | SBP | 4.7773 | 0.8804 | 0.2060 | 7.1054 | 67.2062 | 88.1338 | 95.1731 | PASS\* | PASS (Grade A)\* |
| lora_continue | DBP | 2.6475 | 0.8762 | 0.0649 | 4.3757 | 85.8606 | 96.7613 | 98.8957 | PASS\* | PASS (Grade A)\* |
| lora_prs_bias | SBP | 4.7779 | 0.8805 | 0.2147 | 7.1033 | 67.1221 | 88.1448 | 95.1731 | PASS\* | PASS (Grade A)\* |
| lora_prs_bias | DBP | 2.6491 | 0.8762 | 0.0707 | 4.3754 | 85.8630 | 96.7674 | 98.8993 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic | SBP | 4.7743 | 0.8806 | 0.0946 | 7.1026 | 67.1453 | 88.1984 | 95.1499 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic | DBP | 2.6517 | 0.8759 | 0.0642 | 4.3816 | 85.8752 | 96.7382 | 98.8847 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_shrink | SBP | 4.7761 | 0.8806 | 0.0921 | 7.1016 | 67.0990 | 88.1680 | 95.1621 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_shrink | DBP | 2.6496 | 0.8761 | 0.0649 | 4.3777 | 85.9057 | 96.7516 | 98.8896 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_frozen | SBP | 4.8101 | 0.8795 | 0.0946 | 7.1347 | 66.8174 | 88.0583 | 95.1390 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_frozen | DBP | 2.6683 | 0.8746 | 0.1075 | 4.4027 | 85.6667 | 96.7199 | 98.8883 | PASS\* | PASS (Grade A)\* |

### chronological_blocked · MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora_continue | SBP | 4.5111 | 0.9077 | 0.1629 | 6.8835 | 70.4228 | 89.3645 | 95.5020 | PASS\* | PASS (Grade A)\* |
| lora_continue | DBP | 2.4435 | 0.8925 | 0.0169 | 4.2380 | 87.9773 | 96.9238 | 98.8056 | PASS\* | PASS (Grade A)\* |
| lora_prs_bias | SBP | 4.5134 | 0.9077 | 0.1696 | 6.8836 | 70.3264 | 89.4115 | 95.4748 | PASS\* | PASS (Grade A)\* |
| lora_prs_bias | DBP | 2.4460 | 0.8924 | 0.0182 | 4.2408 | 87.9698 | 96.9560 | 98.8081 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic | SBP | 4.5055 | 0.9078 | 0.0273 | 6.8813 | 70.3882 | 89.4733 | 95.4773 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic | DBP | 2.4472 | 0.8921 | 0.0136 | 4.2468 | 88.0143 | 96.9510 | 98.7834 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_shrink | SBP | 4.5105 | 0.9077 | 0.0276 | 6.8839 | 70.3684 | 89.4288 | 95.4748 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_shrink | DBP | 2.4458 | 0.8922 | 0.0144 | 4.2438 | 88.0391 | 96.9411 | 98.7834 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_frozen | SBP | 4.5480 | 0.9071 | 0.0112 | 6.9070 | 70.0964 | 89.3348 | 95.4698 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_frozen | DBP | 2.4726 | 0.8896 | 0.0523 | 4.2940 | 87.7473 | 96.8917 | 98.7413 | PASS\* | PASS (Grade A)\* |

### chronological_blocked · VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora_continue | SBP | 5.0361 | 0.8305 | 0.2479 | 7.3145 | 64.0793 | 86.9375 | 94.8534 | PASS\* | PASS (Grade B)\* |
| lora_continue | DBP | 2.8457 | 0.8530 | 0.1115 | 4.5052 | 83.8029 | 96.6034 | 98.9832 | PASS\* | PASS (Grade A)\* |
| lora_prs_bias | SBP | 5.0351 | 0.8307 | 0.2585 | 7.3104 | 64.0072 | 86.9135 | 94.8798 | PASS\* | PASS (Grade B)\* |
| lora_prs_bias | DBP | 2.8465 | 0.8532 | 0.1216 | 4.5018 | 83.8149 | 96.5841 | 98.9880 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic | SBP | 5.0356 | 0.8308 | 0.1601 | 7.3108 | 63.9928 | 86.9591 | 94.8317 | PASS\* | PASS (Grade B)\* |
| lora_prs_dynamic | DBP | 2.8505 | 0.8528 | 0.1135 | 4.5083 | 83.7957 | 96.5312 | 98.9832 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_shrink | SBP | 5.0342 | 0.8310 | 0.1548 | 7.3065 | 63.9207 | 86.9423 | 94.8582 | PASS\* | PASS (Grade B)\* |
| lora_prs_dynamic_shrink | DBP | 2.8477 | 0.8531 | 0.1140 | 4.5036 | 83.8317 | 96.5673 | 98.9928 | PASS\* | PASS (Grade A)\* |
| lora_prs_dynamic_frozen | SBP | 5.0649 | 0.8290 | 0.1758 | 7.3484 | 63.6298 | 86.8173 | 94.8173 | PASS\* | PASS (Grade B)\* |
| lora_prs_dynamic_frozen | DBP | 2.8586 | 0.8529 | 0.1612 | 4.5053 | 83.6442 | 96.5529 | 99.0312 | PASS\* | PASS (Grade A)\* |

## 解释与下一步

- 保留 LoRA 的已有个人特征映射是必要起点；在其输出再加 PRS 并没有本批所需的独立收益。
- 冻结整个基模型只训练新修正，不等于已经验证“只调个人 LoRA”和“只调共享网络”的分离优化；二者仍是不同的待测机制问题。
- 目前结果不能证明临床连续监测、新用户少样本适应或真实压力/运动/跨设备鲁棒性。

## 可复核文件

- [跨划分比较](../results/lora_prs_continuation/cross_split_comparison.csv)；[升级门槛](../results/lora_prs_continuation/promotion_gate.csv)；[最终选择](../results/lora_prs_continuation/selection.json)。
- [数据与报告核查](../results/lora_prs_continuation/publication_audit.json)；[作业完成记录](../results/lora_prs_continuation/job_states.csv)。
- [随机总体诊断](../results/lora_prs_continuation/random_disjoint/event_pooled_overall.csv)、[MIMIC](../results/lora_prs_continuation/random_disjoint/event_pooled_mimic.csv)、[VitalDB](../results/lora_prs_continuation/random_disjoint/event_pooled_vitaldb.csv)。
- [时间总体诊断](../results/lora_prs_continuation/chronological_blocked/event_pooled_overall.csv)、[MIMIC](../results/lora_prs_continuation/chronological_blocked/event_pooled_mimic.csv)、[VitalDB](../results/lora_prs_continuation/chronological_blocked/event_pooled_vitaldb.csv)。

本次发布仅复制并核查服务器已完成的聚合报告，不下载受试者级预测或原始波形；不会将聚合核查描述成重新运行了全部患者预测。
已有多轮内部验证探索存在选择偏差；正式论文的效应和不确定性需要后续冻结方案及独立确认。
