# 个人历史记忆增强：完整开发结果

更新：2026-09-07。协议：`development-calbased-analogue-v1`；种子：20260907。

## 结论

本批 10 项模型训练和两份分划分报告已完成。最后跨划分汇总的 `Series.view` 列名冲突已修复；只重新生成汇总，不重训、不更改预测或数据划分。
在六个主表设置中，`pair_distance_blend` 是两种划分的数值最佳候选。随机划分改善较大，时间划分改善较小；两个内部来源的 SBP 和 DBP 均有改善。额外时间邻近诊断 D3 在随机划分更低，见后文，不应把主表第一名称为全部诊断方法的第一名。
但没有候选达到两种划分均改善至少 0.15 mmHg、且两个来源均改善的预设联合门槛，因此不正式替换配对 LoRA 主参照。
单种子、多轮内部验证探索尚不支持统计显著、独立测试优越性或临床有效性结论。

## 实验范围与信息预算

- 只使用父级 meta-train 中 2,051 名已注册受试者：MIMIC 1,011 人、VitalDB 1,040 人。两者是 PulseDB 内部分来源分层，不是独立外部验证。
- 每人 320 个带 BP 标签的 10 秒训练窗口、40 个内部验证窗口、40 个仍封存窗口。每种划分保留全部 82,040 个验证窗口，未剔除最差 30% 或其他困难窗口。
- 个人 LoRA 状态与参考记忆只来自允许的训练数据。检索五个参考不等于仅使用五个标签，也不是新用户 K=5 或五次真实袖带校准。
- 训练与验证身份有意重合，具体窗口/生理区间/重复波形仍按既有审计分离。随机划分允许较晚训练参考，属于已注册用户数据内插值，不是严格前瞻预测。
- 时间划分只使用当前查询之前且满足同记录时间约束的参考；无足够合格参考时回退基础预测，保留查询计入结果。
- 内部验证用于 checkpoint 选择，不能更新个人记忆或进入预测输入；held-out 与独立 meta-test 仍封存。当前并非官方 PulseDB CalBased 固定划分复现。
- 比较参照按每种划分 Overall mean MAE 选择冻结/继续训练 LoRA 中较强者，然后固定该参照用于两个来源，禁止按来源分别挑选。时间划分两参照仅有浮点级差别。

## 六个设置与总体比较

| Setting | 含义 | Random mean MAE | Chronological mean MAE |
| --- | --- | --- | --- |
| D0_frozen_lora | 原冻结个人 LoRA；无需新增训练 | 2.8801 | 3.7124 |
| lora_continued_control | 同个人标签预算，继续训练 LoRA 的强对照 | 2.8764 | 3.7124 |
| pair_single | 单个相似历史参考 + 关系修正 | 3.0375 | 3.9663 |
| pair_uniform | 多个参考等权 + 关系修正 | 3.1895 | 3.7679 |
| pair_retrieved | 多个相似参考按特征距离加权 + 关系修正 | 2.6973 | 3.6965 |
| pair_distance_blend | 按 PPG 距离融合参考记忆预测与冻结 LoRA | 2.6475 | 3.6571 |

单位 mmHg。mean MAE=(SBP MAE+DBP MAE)/2，受试者等权；越低越好。六个设置包含冻结参照，实际新增模型拟合为每种划分五项，共十项。

### 最佳融合与正确配对参照

| split_mode | view | reference_candidate | sbp_mae | dbp_mae | mean_mae | reference_mean_mae | gain_mmhg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_disjoint | Overall | lora_continued_control | 3.3920 | 1.9031 | 2.6475 | 2.8764 | 0.2289 |
| random_disjoint | MIMIC | lora_continued_control | 3.7702 | 2.0847 | 2.9275 | 3.1407 | 0.2132 |
| random_disjoint | VitalDB | lora_continued_control | 3.0243 | 1.7265 | 2.3754 | 2.6195 | 0.2441 |
| chronological_blocked | Overall | D0_frozen_lora | 4.7000 | 2.6143 | 3.6571 | 3.7124 | 0.0552 |
| chronological_blocked | MIMIC | D0_frozen_lora | 4.4284 | 2.4207 | 3.4245 | 3.4773 | 0.0528 |
| chronological_blocked | VitalDB | D0_frozen_lora | 4.9641 | 2.8024 | 3.8832 | 3.9409 | 0.0577 |

gain_mmhg=配对参照 mean MAE−候选 mean MAE，正数表示改善。

## 全部受试者等权结果

### random_disjoint

| candidate | view | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | Overall | 2051 | 82040 | 3.6913 | 2.0689 | 2.8801 |
| D0_frozen_lora | MIMIC | 1011 | 40440 | 4.0434 | 2.2457 | 3.1446 |
| D0_frozen_lora | VitalDB | 1040 | 41600 | 3.3490 | 1.8970 | 2.6230 |
| lora_continued_control | Overall | 2051 | 82040 | 3.6853 | 2.0675 | 2.8764 |
| lora_continued_control | MIMIC | 1011 | 40440 | 4.0379 | 2.2434 | 3.1407 |
| lora_continued_control | VitalDB | 1040 | 41600 | 3.3425 | 1.8965 | 2.6195 |
| pair_single | Overall | 2051 | 82040 | 3.8894 | 2.1856 | 3.0375 |
| pair_single | MIMIC | 1011 | 40440 | 4.4143 | 2.4335 | 3.4239 |
| pair_single | VitalDB | 1040 | 41600 | 3.3792 | 1.9447 | 2.6619 |
| pair_uniform | Overall | 2051 | 82040 | 4.0577 | 2.3212 | 3.1895 |
| pair_uniform | MIMIC | 1011 | 40440 | 4.4442 | 2.5626 | 3.5034 |
| pair_uniform | VitalDB | 1040 | 41600 | 3.6821 | 2.0866 | 2.8843 |
| pair_retrieved | Overall | 2051 | 82040 | 3.4527 | 1.9420 | 2.6973 |
| pair_retrieved | MIMIC | 1011 | 40440 | 3.8393 | 2.1275 | 2.9834 |
| pair_retrieved | VitalDB | 1040 | 41600 | 3.0768 | 1.7616 | 2.4192 |
| pair_distance_blend | Overall | 2051 | 82040 | 3.3920 | 1.9031 | 2.6475 |
| pair_distance_blend | MIMIC | 1011 | 40440 | 3.7702 | 2.0847 | 2.9275 |
| pair_distance_blend | VitalDB | 1040 | 41600 | 3.0243 | 1.7265 | 2.3754 |

### chronological_blocked

| candidate | view | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | Overall | 2051 | 82040 | 4.7773 | 2.6475 | 3.7124 |
| D0_frozen_lora | MIMIC | 1011 | 40440 | 4.5111 | 2.4435 | 3.4773 |
| D0_frozen_lora | VitalDB | 1040 | 41600 | 5.0361 | 2.8457 | 3.9409 |
| lora_continued_control | Overall | 2051 | 82040 | 4.7773 | 2.6475 | 3.7124 |
| lora_continued_control | MIMIC | 1011 | 40440 | 4.5111 | 2.4435 | 3.4773 |
| lora_continued_control | VitalDB | 1040 | 41600 | 5.0361 | 2.8457 | 3.9409 |
| pair_single | Overall | 2051 | 82040 | 5.1005 | 2.8320 | 3.9663 |
| pair_single | MIMIC | 1011 | 40440 | 4.8622 | 2.6518 | 3.7570 |
| pair_single | VitalDB | 1040 | 41600 | 5.3322 | 3.0072 | 4.1697 |
| pair_uniform | Overall | 2051 | 82040 | 4.8336 | 2.7021 | 3.7679 |
| pair_uniform | MIMIC | 1011 | 40440 | 4.6478 | 2.5548 | 3.6013 |
| pair_uniform | VitalDB | 1040 | 41600 | 5.0143 | 2.8453 | 3.9298 |
| pair_retrieved | Overall | 2051 | 82040 | 4.7494 | 2.6437 | 3.6965 |
| pair_retrieved | MIMIC | 1011 | 40440 | 4.4722 | 2.4536 | 3.4629 |
| pair_retrieved | VitalDB | 1040 | 41600 | 5.0188 | 2.8284 | 3.9236 |
| pair_distance_blend | Overall | 2051 | 82040 | 4.7000 | 2.6143 | 3.6571 |
| pair_distance_blend | MIMIC | 1011 | 40440 | 4.4284 | 2.4207 | 3.4245 |
| pair_distance_blend | VitalDB | 1040 | 41600 | 4.9641 | 2.8024 | 3.8832 |

## 完整诊断指标表

下表使用同一批完整验证预测。MAE、ME、STD 单位均为 mmHg；ME=预测−参考，STD 是逐窗口误差的样本标准差，不是多种子 MAE 的标准差。阈值列是累计百分比。
每人均保留 40 个窗口，因此逐窗口 MAE 与受试者等权 MAE 仅有舍入差；R²、STD、阈值比例仍按逐窗口定义。Overall 为原报告对全体数据重算，未将 MIMIC/VitalDB 简单平均。
`AAMI*` 仅是 |ME|≤5、误差 STD≤8 的回顾性数值筛查，不包含正式标准要求的完整试验设计。`BHS*` 按三档比例给出 A/B/C/D，A/B 显示 PASS。二者均不代表设备认证或临床标准验证通过。

### random_disjoint · Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 3.6913 | 0.9303 | 0.0225 | 5.5876 | 76.1824 | 93.4910 | 97.7377 | PASS\* | PASS (Grade A)\* |
| D0_frozen_lora | DBP | 2.0689 | 0.9257 | -0.1183 | 3.4938 | 91.8686 | 98.3191 | 99.4149 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 3.6853 | 0.9304 | 0.0245 | 5.5830 | 76.2530 | 93.5020 | 97.7279 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | DBP | 2.0675 | 0.9257 | -0.1019 | 3.4956 | 91.8588 | 98.3008 | 99.4100 | PASS\* | PASS (Grade A)\* |
| pair_single | SBP | 3.8894 | 0.9103 | -0.1003 | 6.3379 | 75.8435 | 91.6882 | 96.4493 | PASS\* | PASS (Grade A)\* |
| pair_single | DBP | 2.1856 | 0.8984 | -0.0612 | 4.0873 | 90.1938 | 97.4074 | 98.9822 | PASS\* | PASS (Grade A)\* |
| pair_uniform | SBP | 4.0577 | 0.9211 | 0.0963 | 5.9422 | 72.4598 | 92.3367 | 97.3696 | PASS\* | PASS (Grade A)\* |
| pair_uniform | DBP | 2.3212 | 0.9153 | 0.0081 | 3.7320 | 90.0475 | 98.0497 | 99.3576 | PASS\* | PASS (Grade A)\* |
| pair_retrieved | SBP | 3.4527 | 0.9322 | -0.1075 | 5.5103 | 79.2955 | 93.6897 | 97.5670 | PASS\* | PASS (Grade A)\* |
| pair_retrieved | DBP | 1.9420 | 0.9254 | -0.0933 | 3.5024 | 92.3623 | 98.0741 | 99.2626 | PASS\* | PASS (Grade A)\* |
| pair_distance_blend | SBP | 3.3920 | 0.9360 | -0.0902 | 5.3513 | 79.5222 | 94.0017 | 97.8011 | PASS\* | PASS (Grade A)\* |
| pair_distance_blend | DBP | 1.9031 | 0.9302 | -0.1028 | 3.3864 | 92.5975 | 98.2789 | 99.3674 | PASS\* | PASS (Grade A)\* |

### random_disjoint · MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 4.0434 | 0.9271 | 0.0468 | 6.1448 | 73.2122 | 91.8867 | 96.9807 | PASS\* | PASS (Grade A)\* |
| D0_frozen_lora | DBP | 2.2457 | 0.9105 | -0.1874 | 3.9738 | 90.3783 | 97.6706 | 99.0653 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 4.0379 | 0.9272 | 0.0862 | 6.1411 | 73.3234 | 91.8892 | 96.9609 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | DBP | 2.2434 | 0.9103 | -0.1640 | 3.9783 | 90.3981 | 97.6607 | 99.0529 | PASS\* | PASS (Grade A)\* |
| pair_single | SBP | 4.4143 | 0.9026 | -0.1028 | 7.1005 | 72.0129 | 89.6958 | 95.3363 | PASS\* | PASS (Grade A)\* |
| pair_single | DBP | 2.4335 | 0.8717 | -0.0669 | 4.7615 | 88.6152 | 96.5010 | 98.4941 | PASS\* | PASS (Grade A)\* |
| pair_uniform | SBP | 4.4442 | 0.9175 | -0.0188 | 6.5378 | 69.2062 | 90.4451 | 96.4367 | PASS\* | PASS (Grade A)\* |
| pair_uniform | DBP | 2.5626 | 0.8972 | 0.0012 | 4.2638 | 87.9575 | 97.2354 | 98.9763 | PASS\* | PASS (Grade A)\* |
| pair_retrieved | SBP | 3.8393 | 0.9292 | -0.1213 | 6.0551 | 75.9941 | 92.1241 | 96.8002 | PASS\* | PASS (Grade A)\* |
| pair_retrieved | DBP | 2.1275 | 0.9109 | -0.1219 | 3.9670 | 91.0163 | 97.3961 | 98.9095 | PASS\* | PASS (Grade A)\* |
| pair_distance_blend | SBP | 3.7702 | 0.9325 | -0.0894 | 5.9107 | 76.3056 | 92.4679 | 97.0821 | PASS\* | PASS (Grade A)\* |
| pair_distance_blend | DBP | 2.0847 | 0.9157 | -0.1385 | 3.8582 | 91.2141 | 97.6607 | 99.0381 | PASS\* | PASS (Grade A)\* |

### random_disjoint · VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 3.3490 | 0.9307 | -0.0011 | 4.9867 | 79.0697 | 95.0505 | 98.4736 | PASS\* | PASS (Grade A)\* |
| D0_frozen_lora | DBP | 1.8970 | 0.9419 | -0.0510 | 2.9518 | 93.3173 | 98.9495 | 99.7548 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 3.3425 | 0.9309 | -0.0355 | 4.9803 | 79.1010 | 95.0697 | 98.4736 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | DBP | 1.8965 | 0.9420 | -0.0416 | 2.9505 | 93.2788 | 98.9231 | 99.7572 | PASS\* | PASS (Grade A)\* |
| pair_single | SBP | 3.3792 | 0.9158 | -0.0979 | 5.4961 | 79.5673 | 93.6250 | 97.5312 | PASS\* | PASS (Grade A)\* |
| pair_single | DBP | 1.9447 | 0.9273 | -0.0556 | 3.3027 | 91.7284 | 98.2885 | 99.4567 | PASS\* | PASS (Grade A)\* |
| pair_uniform | SBP | 3.6821 | 0.9217 | 0.2083 | 5.2971 | 75.6226 | 94.1755 | 98.2764 | PASS\* | PASS (Grade A)\* |
| pair_uniform | DBP | 2.0866 | 0.9347 | 0.0148 | 3.1297 | 92.0793 | 98.8413 | 99.7284 | PASS\* | PASS (Grade A)\* |
| pair_retrieved | SBP | 3.0768 | 0.9324 | -0.0941 | 4.9232 | 82.5048 | 95.2115 | 98.3125 | PASS\* | PASS (Grade A)\* |
| pair_retrieved | DBP | 1.7616 | 0.9407 | -0.0655 | 2.9819 | 93.6707 | 98.7332 | 99.6058 | PASS\* | PASS (Grade A)\* |
| pair_distance_blend | SBP | 3.0243 | 0.9372 | -0.0910 | 4.7448 | 82.6490 | 95.4928 | 98.5000 | PASS\* | PASS (Grade A)\* |
| pair_distance_blend | DBP | 1.7265 | 0.9457 | -0.0681 | 2.8535 | 93.9423 | 98.8798 | 99.6875 | PASS\* | PASS (Grade A)\* |

### chronological_blocked · Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 4.7773 | 0.8804 | 0.2060 | 7.1054 | 67.2062 | 88.1338 | 95.1731 | PASS\* | PASS (Grade A)\* |
| D0_frozen_lora | DBP | 2.6475 | 0.8762 | 0.0649 | 4.3757 | 85.8606 | 96.7613 | 98.8957 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 4.7773 | 0.8804 | 0.2060 | 7.1054 | 67.2062 | 88.1338 | 95.1731 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | DBP | 2.6475 | 0.8762 | 0.0649 | 4.3757 | 85.8606 | 96.7613 | 98.8957 | PASS\* | PASS (Grade A)\* |
| pair_single | SBP | 5.1005 | 0.8602 | 0.1523 | 7.6829 | 65.0987 | 86.6041 | 94.0273 | PASS\* | PASS (Grade B)\* |
| pair_single | DBP | 2.8320 | 0.8534 | 0.0333 | 4.7620 | 84.3137 | 95.8459 | 98.5836 | PASS\* | PASS (Grade A)\* |
| pair_uniform | SBP | 4.8336 | 0.8795 | 0.2694 | 7.1304 | 66.2591 | 88.1509 | 95.2353 | PASS\* | PASS (Grade A)\* |
| pair_uniform | DBP | 2.7021 | 0.8764 | 0.0885 | 4.3718 | 85.7058 | 96.7065 | 98.9188 | PASS\* | PASS (Grade A)\* |
| pair_retrieved | SBP | 4.7494 | 0.8774 | 0.1807 | 7.1947 | 67.8657 | 88.0839 | 94.9366 | PASS\* | PASS (Grade B)\* |
| pair_retrieved | DBP | 2.6437 | 0.8732 | 0.0616 | 4.4292 | 85.6131 | 96.3835 | 98.8250 | PASS\* | PASS (Grade A)\* |
| pair_distance_blend | SBP | 4.7000 | 0.8805 | 0.1896 | 7.1035 | 68.0570 | 88.3008 | 95.1207 | PASS\* | PASS (Grade A)\* |
| pair_distance_blend | DBP | 2.6143 | 0.8757 | 0.0591 | 4.3849 | 85.9178 | 96.5822 | 98.8969 | PASS\* | PASS (Grade A)\* |

### chronological_blocked · MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 4.5111 | 0.9077 | 0.1629 | 6.8835 | 70.4228 | 89.3645 | 95.5020 | PASS\* | PASS (Grade A)\* |
| D0_frozen_lora | DBP | 2.4435 | 0.8925 | 0.0169 | 4.2380 | 87.9773 | 96.9238 | 98.8056 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 4.5111 | 0.9077 | 0.1629 | 6.8835 | 70.4228 | 89.3645 | 95.5020 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | DBP | 2.4435 | 0.8925 | 0.0169 | 4.2380 | 87.9773 | 96.9238 | 98.8056 | PASS\* | PASS (Grade A)\* |
| pair_single | SBP | 4.8622 | 0.8888 | 0.0971 | 7.5563 | 67.8215 | 87.8314 | 94.4708 | PASS\* | PASS (Grade B)\* |
| pair_single | DBP | 2.6518 | 0.8673 | -0.0051 | 4.7095 | 86.5010 | 96.0188 | 98.3828 | PASS\* | PASS (Grade A)\* |
| pair_uniform | SBP | 4.6478 | 0.9043 | 0.3260 | 7.0031 | 68.6869 | 89.0925 | 95.3264 | PASS\* | PASS (Grade A)\* |
| pair_uniform | DBP | 2.5548 | 0.8916 | 0.0934 | 4.2552 | 87.3195 | 96.7433 | 98.8501 | PASS\* | PASS (Grade A)\* |
| pair_retrieved | SBP | 4.4722 | 0.9048 | 0.1212 | 6.9902 | 70.9248 | 89.3843 | 95.4080 | PASS\* | PASS (Grade A)\* |
| pair_retrieved | DBP | 2.4536 | 0.8877 | 0.0345 | 4.3307 | 87.5519 | 96.5356 | 98.7092 | PASS\* | PASS (Grade A)\* |
| pair_distance_blend | SBP | 4.4284 | 0.9075 | 0.1291 | 6.8918 | 71.0930 | 89.5302 | 95.5663 | PASS\* | PASS (Grade A)\* |
| pair_distance_blend | DBP | 2.4207 | 0.8911 | 0.0231 | 4.2661 | 87.8956 | 96.7087 | 98.8056 | PASS\* | PASS (Grade A)\* |

### chronological_blocked · VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_frozen_lora | SBP | 5.0361 | 0.8305 | 0.2479 | 7.3145 | 64.0793 | 86.9375 | 94.8534 | PASS\* | PASS (Grade B)\* |
| D0_frozen_lora | DBP | 2.8457 | 0.8530 | 0.1115 | 4.5052 | 83.8029 | 96.6034 | 98.9832 | PASS\* | PASS (Grade A)\* |
| lora_continued_control | SBP | 5.0361 | 0.8305 | 0.2479 | 7.3145 | 64.0793 | 86.9375 | 94.8534 | PASS\* | PASS (Grade B)\* |
| lora_continued_control | DBP | 2.8457 | 0.8530 | 0.1115 | 4.5052 | 83.8029 | 96.6034 | 98.9832 | PASS\* | PASS (Grade A)\* |
| pair_single | SBP | 5.3322 | 0.8072 | 0.2059 | 7.8038 | 62.4519 | 85.4111 | 93.5962 | PASS\* | PASS (Grade B)\* |
| pair_single | DBP | 3.0072 | 0.8323 | 0.0707 | 4.8123 | 82.1875 | 95.6779 | 98.7788 | PASS\* | PASS (Grade A)\* |
| pair_uniform | SBP | 5.0143 | 0.8335 | 0.2143 | 7.2516 | 63.8990 | 87.2356 | 95.1466 | PASS\* | PASS (Grade A)\* |
| pair_uniform | DBP | 2.8453 | 0.8545 | 0.0838 | 4.4822 | 84.1370 | 96.6707 | 98.9856 | PASS\* | PASS (Grade A)\* |
| pair_retrieved | SBP | 5.0188 | 0.8271 | 0.2386 | 7.3876 | 64.8918 | 86.8197 | 94.4784 | PASS\* | PASS (Grade B)\* |
| pair_retrieved | DBP | 2.8284 | 0.8519 | 0.0879 | 4.5228 | 83.7284 | 96.2356 | 98.9375 | PASS\* | PASS (Grade A)\* |
| pair_distance_blend | SBP | 4.9641 | 0.8310 | 0.2485 | 7.3029 | 65.1058 | 87.1058 | 94.6875 | PASS\* | PASS (Grade B)\* |
| pair_distance_blend | DBP | 2.8024 | 0.8535 | 0.0941 | 4.4971 | 83.9952 | 96.4591 | 98.9856 | PASS\* | PASS (Grade A)\* |

## 为什么部分任务只需几十秒

四种关系模块复用冻结的 256 维 LoRA 特征，只有 65,730 个共享可训练参数，并在 GPU 中批量读取缓存；不是从头训练完整原始 PPG 主干。关系模块每项约 38 秒至 2 分 12 秒；两个继续训练 LoRA 对照约 57 分 34 秒和 30 分 57 秒。
随机融合实际完成八轮、6,256 次优化更新，但最佳 checkpoint 为 epoch 0：关系修正初始为零，其优势来自历史参考检索与固定距离融合，而非新增关系网络训练。
时间融合最佳为 epoch 1：相对自身初始 mean MAE 3.796670 降到 3.657139，说明有学习收益；但相对更强 LoRA 参照的优势仍只有约 0.0552 mmHg。不能把自身初始化的下降全部归为超越 LoRA。

## 机制诊断与时间邻近性

诊断 D3 使用合法训练记录中时间最近的 BP，在随机划分 mean MAE 为 2.285079，优于当前融合候选。这提示短时血压延续性与时间邻近性可能贡献较大，不能只将随机划分增益解释为新的波形生理映射。D3 使用记录时间，并非纯 PPG 部署输入。
随机划分简单特征邻居 BP 检索 D1 的 mean MAE 为 2.697324；排除查询前后 60 秒内参考后变为 3.158958，冻结 LoRA 保持 2.880098。该检查支持近时间参考敏感性，不自动等于违反当前窗口划分规则。
60 秒实验针对冻结诊断方法，不是完整融合候选的同条件测试，不能据此断言完整融合在间隔约束下必然失败。训练编码器已见过训练标签，块排除检索属于普通监督训练，不能称为编码器级 OOF。

## 决策与论文边界

保留个人记忆作为有信号的候选方向，但尚不能以新关系网络为已验证核心贡献。优先检验长期/时间间隔条件下记忆是否仍有独立价值，并保留 LoRA、时间邻近 BP、简单检索和固定融合对照。
相关下一步应先写明有限假设、信息预算、时间约束、选择规则和失败退出条件，再执行确认；不能通过删除困难受试者或放宽既定门槛把当前结果变成成功。
多种子与受试者层配对不确定性、机制对照、封存评估以及真实条件鲁棒性仍待完成；当前结果不保证可发表性。

## 可复核文件

- [跨划分比较](../results/personal_memory_v1/final/cross_split_comparison.csv)、[联合升级门槛](../results/personal_memory_v1/final/promotion_gate.csv)、[最终选择](../results/personal_memory_v1/final/selection.json)。
- [随机受试者等权表](../results/personal_memory_v1/random_disjoint/participant_macro_summary.csv)、[时间受试者等权表](../results/personal_memory_v1/chronological_blocked/participant_macro_summary.csv)。
- [随机完整指标](../results/personal_memory_v1/random_disjoint/event_pooled_diagnostics_all_scopes.csv)、[时间完整指标](../results/personal_memory_v1/chronological_blocked/event_pooled_diagnostics_all_scopes.csv)。
- [发布核查](../results/personal_memory_v1/publication_audit.json)记录聚合一致性、数据来源文件散列和联合门槛复算。

公开内容仅含聚合数据、方法与核查记录，不含原始波形、个人标识、逐窗口预测、个人权重、服务器路径或组会幻灯片。发布核查不等于重新推理或重新拟合。

[执行与机制诊断证据](../results/personal_memory_v1/execution_evidence.json)记录修复范围、运行状态及相关诊断来源。
