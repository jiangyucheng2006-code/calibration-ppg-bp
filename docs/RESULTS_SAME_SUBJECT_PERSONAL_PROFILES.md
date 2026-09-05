# 紧凑个人状态模型：完整实验结果

更新日期：2026-09-06。训练及最终报告于2026-09-05 14:53（UTC+8）完成。

## 结论

16个训练任务和3个报告任务均完成，Slurm退出码均为0:0，19份对应stderr为空。
LoRA rank-4在两种划分、三个数据范围均优于所有新Profile变体。预设32D主模型
未达到升级条件，保留LoRA作为当前准确率参考。这个结论来自完整结果；它并非仅因
0.15 mmHg升级门槛被拒绝，而是主模型的误差实际高于LoRA。

随机划分：LoRA mean MAE为3.0394，主模型为4.0179，相差0.9785 mmHg。
时间划分：LoRA为3.7945，主模型为4.3140，相差0.5195 mmHg。
新Profile中，随机划分最好的64D版本为3.9031；时间划分最好的无门控32D版本为4.2311。
两者是控制组中的数值观察，不替换预先指定的主假设。

## 数据与解释边界

- 这是持续使用已注册受试者个人信息的同受试者开发实验。2,051人中MIMIC为1,011人，
  VitalDB为1,040人；每人320个有标签训练窗口、40个内部验证窗口，另40个held-out窗口封存。
- 每种划分的内部验证均为82,040个10秒窗口；同一划分中全部模型使用完全相同的窗口和标签。
- random_disjoint为同人随机不重叠窗口；chronological_blocked按时间分块，训练与验证窗口不同。
  随机划分不代表严格的未来预测，两种结果分别呈现。它们均为自定义CalBased analogue，
  不宣称逐项复现官方PulseDB CalBased。
- seed=20260904；共用compact ResNet、Huber训练、patience=8及无epoch上限。
  本轮为单seed开发比较，未提供重复训练的不确定性结论。
- 五个历史support来自训练角色，并不意味着模型总共只接触过五个标签。
  LoRA及个人code已从每人320个有标签训练窗口学习；本轮不证明新用户K=1/2/3/5校准。
- MIMIC/VitalDB是PulseDB内部来源分层，不能称为独立外部验证。

## 全部模型：Overall participant-macro mean MAE

mean MAE为SBP和DBP两者MAE的平均，单位mmHg。完整三个范围的SBP/DBP见后表。


| candidate | 说明 | 时间划分mean MAE | 随机划分mean MAE |
| --- | --- | --- | --- |
| subject_lora_rank4 | LoRA rank-4 | 3.7945 | 3.0394 |
| personal_profile_code64_reliability | 64D完整容量对照 | 4.2535 | 3.9031 |
| personal_profile_code32_no_reliability | 32D个人状态，无可靠性门控 | 4.2311 | 3.9806 |
| personal_profile_code32_reliability | 32D完整主模型 | 4.3140 | 4.0179 |
| personal_profile_code32_no_support | 32D个人状态，无历史支持 | 4.3112 | 4.0729 |
| personal_profile_support_only | 仅历史支持信息 | 5.0035 | 4.6337 |
| personal_profile_code32_stable_only | 32D仅稳定个人修正 | 5.0482 | 5.1583 |
| residual_reference | 普通PPG残差参考 | 6.1128 | 6.0181 |


## 本轮已经得到的结构证据

1. **动态个人修正有价值。** 在同时无门控且有支持信息的32D设置中，移除动态分支后，
   随机/时间划分mean MAE分别由3.9806/4.2311升至5.1583/5.0482。
   这支持保留依赖当前波形的个体交互，但不独立证明其生理解释。
2. **目前可靠性门控没有收益。** 添加门控后，3.9806/4.2311变为4.0179/4.3140，
   分别恶化0.0374/0.0829。小差异仍需要重复训练确认；不应把门控自动纳入最终模型。
3. **历史support增益有限。** 同为32D、无门控且有动态分支，加入历史描述后，
   4.0729/4.3112降为3.9806/4.2311，改善0.0923/0.0801。
   no_support依然输入当前PPG，也保存个人状态与训练BP均值，绝不是无PPG模型。
4. **增加个人状态容量有些帮助，但没有追上LoRA。** 完整32D改64D后，
   4.0179/4.3140降为3.9031/4.2535。个人可训练参数分别为34和66，LoRA为2,048。
   参数数目只计算个人可训练状态；不包括共享网络、支持描述缓存和身份映射。
   在本研究2,051人规模下，总模型参数分别为813,054、889,630与4,898,578。
   少参数并不等于更快推理，本轮没有测量端到端时延。
5. **更小的随机到时间误差差距不等于更强的最终预测。** 主模型两种划分只差0.2960，
   小于LoRA的0.7550，但主模型在两种条件下的绝对误差都更高。

## 原因判断与下一步建议

已核查代码：LoRA根据个人参数直接变换256维PPG特征，再送入预测头；当前Profile将
个人code与PPG特征拼接，通过共享网络输出稳定和动态修正。因此两者既有个人参数
容量差异，也有个体化作用位置和交互形式差异。现有结果无法断言究竟哪一个是差距主因。

建议优先完成一个小而可解释的后续验证，而非再启动全面模块组合：

1. 保留当前seen-user、两种不重叠窗口划分和封存角色。对LoRA及32D无门控对照，
   做同受试者内PPG打乱、真正不输入PPG的个人偏差基线、共享适配器及交换个人参数
   的机制检查，确认收益来自当前波形变化还是身份信息。扰动检查仅作机制诊断，
   模型在自然数据上的结论仍用正常输入评分。
2. 若继续改进Profile，将假设集中在**让个人状态改变波形特征的映射**。比较当前
   输出端残差、特征端逐通道调制、由个人系数控制的共享线性变换，并加入低rank LoRA
   容量对照。固定训练预算，报告个人参数及全模型参数，区分作用位置与参数量。
   FiLM和低秩变换均有既有工作，不能仅因改名或重组就主张新颖性；这只是待验证方案。
3. 为节省资源，先选有限候选做开发筛选，再对有实质收益的方案与LoRA做配对多seed确认。
   当前结果不支持直接给完整32D模型安排大规模确认，也不支持宣称其与LoRA非劣效。
4. 当前任务优先维持用户选择的已注册用户方案。新用户少量校准可作为另一个后续课题；
   它不自动成为本轮失败后的必做下一步。压力、运动和设备验证要在最终方法与使用场景
   确定后，配合具有可信条件标注的数据设计。

本次只发布结果和建议，没有提交新的训练任务，也没有打开held-out。

## 完整指标表

以下主表按受试者平均；诊断表按全部窗口重新计算。每人均有40个窗口，因此两种MAE
在数值上基本相同。R²、ME、STD和三个误差范围占比均由相应范围的预测重新计算；
Overall并非两个来源指标的简单平均。ME=预测−参考，STD为有符号误差的样本标准差。

诊断表中的AAMI、BHS直接保留既有评分程序的回顾性数值筛查：PASS*、FAIL*和等级。
星号表示它们不是临床认证或正式标准符合性证据；BHS等级单独列明，不把B级写成A级。
其中最差30%/剩余70%字段若在CSV中出现，属于按真实验证误差事后排序的oracle诊断，
不代表已经能在部署时识别或剔除这部分人；主表保留全部2,051名受试者。


### random_disjoint — Overall


Participant-macro MAE（mmHg）：

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | 2051 | 82040 | 3.9018 | 2.1771 | 3.0394 |
| personal_profile_code64_reliability | 2051 | 82040 | 5.0188 | 2.7873 | 3.9031 |
| personal_profile_code32_no_reliability | 2051 | 82040 | 5.1134 | 2.8478 | 3.9806 |
| personal_profile_code32_reliability | 2051 | 82040 | 5.1591 | 2.8768 | 4.0179 |
| personal_profile_code32_no_support | 2051 | 82040 | 5.2357 | 2.9101 | 4.0729 |
| personal_profile_support_only | 2051 | 82040 | 5.9602 | 3.3072 | 4.6337 |
| personal_profile_code32_stable_only | 2051 | 82040 | 6.6935 | 3.6231 | 5.1583 |
| residual_reference | 2051 | 82040 | 7.8023 | 4.2340 | 6.0181 |

窗口汇总诊断；≤5/10/15的数值单位为百分比：

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual_reference | SBP | 7.8023 | 0.7375 | -0.1688 | 10.8427 | 45.2316 | 72.6962 | 86.3701 | FAIL* | FAIL (Grade C)* |
| residual_reference | DBP | 4.2340 | 0.7708 | -0.2713 | 6.1343 | 70.2694 | 91.5553 | 97.3659 | PASS* | PASS (Grade A)* |
| subject_lora_rank4 | SBP | 3.9018 | 0.9248 | 0.0193 | 5.8048 | 74.0492 | 92.9205 | 97.5780 | PASS* | PASS (Grade A)* |
| subject_lora_rank4 | DBP | 2.1771 | 0.9203 | -0.1215 | 3.6195 | 91.0922 | 98.2106 | 99.3467 | PASS* | PASS (Grade A)* |
| personal_profile_support_only | SBP | 5.9602 | 0.8427 | -0.1566 | 8.3925 | 55.9264 | 82.6889 | 92.9595 | FAIL* | PASS (Grade B)* |
| personal_profile_support_only | DBP | 3.3072 | 0.8508 | -0.3756 | 4.9403 | 79.9768 | 95.5790 | 98.6507 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_support | SBP | 5.2357 | 0.8749 | -0.2299 | 7.4818 | 61.5285 | 86.7345 | 94.9854 | PASS* | PASS (Grade B)* |
| personal_profile_code32_no_support | DBP | 2.9101 | 0.8747 | -0.2105 | 4.5348 | 84.4978 | 96.8601 | 98.9408 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_reliability | SBP | 5.1134 | 0.8798 | -0.1342 | 7.3356 | 62.5366 | 87.4086 | 95.2426 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_reliability | DBP | 2.8478 | 0.8805 | -0.2374 | 4.4277 | 85.1792 | 97.0758 | 98.9590 | PASS* | PASS (Grade A)* |
| personal_profile_code32_reliability | SBP | 5.1591 | 0.8777 | 0.1861 | 7.3975 | 62.4122 | 87.0819 | 95.1231 | PASS* | PASS (Grade A)* |
| personal_profile_code32_reliability | DBP | 2.8768 | 0.8797 | -0.0388 | 4.4488 | 84.5758 | 96.9454 | 99.0115 | PASS* | PASS (Grade A)* |
| personal_profile_code64_reliability | SBP | 5.0188 | 0.8837 | -0.0119 | 7.2167 | 63.4313 | 87.8352 | 95.4790 | PASS* | PASS (Grade A)* |
| personal_profile_code64_reliability | DBP | 2.7873 | 0.8844 | -0.2468 | 4.3541 | 85.6619 | 97.1660 | 99.0456 | PASS* | PASS (Grade A)* |
| personal_profile_code32_stable_only | SBP | 6.6935 | 0.8012 | -0.0566 | 9.4356 | 51.0093 | 78.9127 | 90.8862 | FAIL* | PASS (Grade B)* |
| personal_profile_code32_stable_only | DBP | 3.6231 | 0.8194 | -0.3660 | 5.4393 | 76.9161 | 94.5783 | 98.2387 | PASS* | PASS (Grade A)* |

### random_disjoint — MIMIC


Participant-macro MAE（mmHg）：

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | 1011 | 40440 | 4.2769 | 2.3643 | 3.3206 |
| personal_profile_code64_reliability | 1011 | 40440 | 5.4054 | 2.9718 | 4.1886 |
| personal_profile_code32_no_reliability | 1011 | 40440 | 5.5039 | 3.0375 | 4.2707 |
| personal_profile_code32_reliability | 1011 | 40440 | 5.5275 | 3.0601 | 4.2938 |
| personal_profile_code32_no_support | 1011 | 40440 | 5.5905 | 3.0812 | 4.3359 |
| personal_profile_support_only | 1011 | 40440 | 6.1829 | 3.3749 | 4.7789 |
| personal_profile_code32_stable_only | 1011 | 40440 | 7.1219 | 3.8066 | 5.4643 |
| residual_reference | 1011 | 40440 | 7.8116 | 4.1891 | 6.0004 |

窗口汇总诊断；≤5/10/15的数值单位为百分比：

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual_reference | SBP | 7.8116 | 0.7691 | -0.2412 | 10.9340 | 45.7938 | 72.6855 | 86.3477 | FAIL* | FAIL (Grade C)* |
| residual_reference | DBP | 4.1891 | 0.7715 | -0.4830 | 6.3371 | 71.8497 | 91.7161 | 97.0450 | PASS* | PASS (Grade A)* |
| subject_lora_rank4 | SBP | 4.2769 | 0.9209 | -0.0570 | 6.3995 | 70.9570 | 91.1573 | 96.7384 | PASS* | PASS (Grade A)* |
| subject_lora_rank4 | DBP | 2.3643 | 0.9034 | -0.2669 | 4.1231 | 89.5549 | 97.5025 | 98.9491 | PASS* | PASS (Grade A)* |
| personal_profile_support_only | SBP | 6.1829 | 0.8513 | -0.0380 | 8.7751 | 54.7082 | 81.5628 | 92.1241 | FAIL* | PASS (Grade B)* |
| personal_profile_support_only | DBP | 3.3749 | 0.8408 | -0.4952 | 5.2823 | 80.2596 | 95.0371 | 98.2319 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_support | SBP | 5.5905 | 0.8764 | -0.0974 | 7.9994 | 59.1024 | 84.9085 | 93.8749 | PASS* | PASS (Grade B)* |
| personal_profile_code32_no_support | DBP | 3.0812 | 0.8565 | -0.3948 | 5.0217 | 83.4248 | 96.0633 | 98.4792 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_reliability | SBP | 5.5039 | 0.8787 | 0.1165 | 7.9272 | 59.9580 | 85.2300 | 94.0702 | PASS* | PASS (Grade B)* |
| personal_profile_code32_no_reliability | DBP | 3.0375 | 0.8619 | -0.2898 | 4.9327 | 83.8823 | 96.2413 | 98.4842 | PASS* | PASS (Grade A)* |
| personal_profile_code32_reliability | SBP | 5.5275 | 0.8779 | 0.2292 | 7.9488 | 59.8986 | 85.0742 | 94.0974 | PASS* | PASS (Grade B)* |
| personal_profile_code32_reliability | DBP | 3.0601 | 0.8619 | -0.1582 | 4.9378 | 83.2245 | 96.1449 | 98.5485 | PASS* | PASS (Grade A)* |
| personal_profile_code64_reliability | SBP | 5.4054 | 0.8828 | -0.0214 | 7.7908 | 60.6157 | 85.8383 | 94.4115 | PASS* | PASS (Grade B)* |
| personal_profile_code64_reliability | DBP | 2.9718 | 0.8664 | -0.4648 | 4.8385 | 84.3348 | 96.3254 | 98.5608 | PASS* | PASS (Grade A)* |
| personal_profile_code32_stable_only | SBP | 7.1219 | 0.8052 | 0.2988 | 10.0400 | 48.8304 | 76.6518 | 89.3645 | FAIL* | FAIL (Grade C)* |
| personal_profile_code32_stable_only | DBP | 3.8066 | 0.7973 | -0.4459 | 5.9701 | 76.1375 | 93.6820 | 97.6360 | PASS* | PASS (Grade A)* |

### random_disjoint — VitalDB


Participant-macro MAE（mmHg）：

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | 1040 | 41600 | 3.5372 | 1.9951 | 2.7661 |
| personal_profile_code64_reliability | 1040 | 41600 | 4.6430 | 2.6079 | 3.6254 |
| personal_profile_code32_no_reliability | 1040 | 41600 | 4.7337 | 2.6634 | 3.6986 |
| personal_profile_code32_reliability | 1040 | 41600 | 4.8011 | 2.6985 | 3.7498 |
| personal_profile_code32_no_support | 1040 | 41600 | 4.8908 | 2.7437 | 3.8173 |
| personal_profile_support_only | 1040 | 41600 | 5.7437 | 3.2414 | 4.4925 |
| personal_profile_code32_stable_only | 1040 | 41600 | 6.2770 | 3.4448 | 4.8609 |
| residual_reference | 1040 | 41600 | 7.7931 | 4.2776 | 6.0354 |

窗口汇总诊断；≤5/10/15的数值单位为百分比：

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual_reference | SBP | 7.7931 | 0.6778 | -0.0984 | 10.7528 | 44.6851 | 72.7067 | 86.3918 | FAIL* | FAIL (Grade C)* |
| residual_reference | DBP | 4.2776 | 0.7662 | -0.0655 | 5.9232 | 68.7332 | 91.3990 | 97.6779 | PASS* | PASS (Grade A)* |
| subject_lora_rank4 | SBP | 3.5372 | 0.9258 | 0.0935 | 5.1604 | 77.0553 | 94.6346 | 98.3942 | PASS* | PASS (Grade A)* |
| subject_lora_rank4 | DBP | 1.9951 | 0.9382 | 0.0199 | 3.0447 | 92.5865 | 98.8990 | 99.7332 | PASS* | PASS (Grade A)* |
| personal_profile_support_only | SBP | 5.7437 | 0.8214 | -0.2719 | 8.0015 | 57.1106 | 83.7837 | 93.7716 | FAIL* | PASS (Grade B)* |
| personal_profile_support_only | DBP | 3.2414 | 0.8598 | -0.2593 | 4.5806 | 79.7019 | 96.1058 | 99.0577 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_support | SBP | 4.8908 | 0.8655 | -0.3587 | 6.9393 | 63.8870 | 88.5096 | 96.0649 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_support | DBP | 2.7437 | 0.8936 | -0.0313 | 3.9970 | 85.5409 | 97.6346 | 99.3894 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_reliability | SBP | 4.7337 | 0.8744 | -0.3780 | 6.7019 | 65.0433 | 89.5264 | 96.3822 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_reliability | DBP | 2.6634 | 0.8998 | -0.1865 | 3.8734 | 86.4399 | 97.8870 | 99.4207 | PASS* | PASS (Grade A)* |
| personal_profile_code32_reliability | SBP | 4.8011 | 0.8704 | 0.1442 | 6.8188 | 64.8558 | 89.0337 | 96.1202 | PASS* | PASS (Grade A)* |
| personal_profile_code32_reliability | DBP | 2.6985 | 0.8980 | 0.0774 | 3.9120 | 85.8894 | 97.7236 | 99.4615 | PASS* | PASS (Grade A)* |
| personal_profile_code64_reliability | SBP | 4.6430 | 0.8782 | -0.0026 | 6.6110 | 66.1683 | 89.7764 | 96.5168 | PASS* | PASS (Grade A)* |
| personal_profile_code64_reliability | DBP | 2.6079 | 0.9031 | -0.0349 | 3.8130 | 86.9519 | 97.9832 | 99.5168 | PASS* | PASS (Grade A)* |
| personal_profile_code32_stable_only | SBP | 6.2770 | 0.7840 | -0.4021 | 8.7947 | 53.1274 | 81.1106 | 92.3654 | FAIL* | PASS (Grade B)* |
| personal_profile_code32_stable_only | DBP | 3.4448 | 0.8416 | -0.2884 | 4.8671 | 77.6731 | 95.4495 | 98.8245 | PASS* | PASS (Grade A)* |

### chronological_blocked — Overall


Participant-macro MAE（mmHg）：

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | 2051 | 82040 | 4.8789 | 2.7101 | 3.7945 |
| personal_profile_code32_no_reliability | 2051 | 82040 | 5.4717 | 2.9906 | 4.2311 |
| personal_profile_code64_reliability | 2051 | 82040 | 5.4982 | 3.0087 | 4.2535 |
| personal_profile_code32_no_support | 2051 | 82040 | 5.5843 | 3.0381 | 4.3112 |
| personal_profile_code32_reliability | 2051 | 82040 | 5.5740 | 3.0540 | 4.3140 |
| personal_profile_support_only | 2051 | 82040 | 6.4424 | 3.5645 | 5.0035 |
| personal_profile_code32_stable_only | 2051 | 82040 | 6.5280 | 3.5683 | 5.0482 |
| residual_reference | 2051 | 82040 | 7.9399 | 4.2856 | 6.1128 |

窗口汇总诊断；≤5/10/15的数值单位为百分比：

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual_reference | SBP | 7.9399 | 0.7129 | 0.1015 | 11.0136 | 45.4534 | 71.9539 | 85.1682 | FAIL* | FAIL (Grade C)* |
| residual_reference | DBP | 4.2856 | 0.7488 | -0.0262 | 6.2342 | 69.5978 | 90.5814 | 97.0295 | PASS* | PASS (Grade A)* |
| subject_lora_rank4 | SBP | 4.8789 | 0.8780 | 0.0704 | 7.1779 | 65.9373 | 87.9632 | 95.0975 | PASS* | PASS (Grade A)* |
| subject_lora_rank4 | DBP | 2.7101 | 0.8726 | 0.0210 | 4.4403 | 85.4608 | 96.6443 | 98.8762 | PASS* | PASS (Grade A)* |
| personal_profile_support_only | SBP | 6.4424 | 0.8084 | 0.5225 | 8.9824 | 53.0680 | 79.8549 | 91.1190 | FAIL* | PASS (Grade B)* |
| personal_profile_support_only | DBP | 3.5645 | 0.8186 | 0.0290 | 5.2974 | 76.8308 | 94.0529 | 98.3459 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_support | SBP | 5.5843 | 0.8515 | 0.2637 | 7.9163 | 59.0712 | 84.5770 | 93.7165 | PASS* | PASS (Grade B)* |
| personal_profile_code32_no_support | DBP | 3.0381 | 0.8597 | -0.0618 | 4.6584 | 82.5634 | 96.0787 | 98.7555 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_reliability | SBP | 5.4717 | 0.8594 | 0.0597 | 7.7075 | 59.5648 | 85.1621 | 94.2235 | PASS* | PASS (Grade B)* |
| personal_profile_code32_no_reliability | DBP | 2.9906 | 0.8626 | -0.1113 | 4.6085 | 82.9876 | 96.2530 | 98.8554 | PASS* | PASS (Grade A)* |
| personal_profile_code32_reliability | SBP | 5.5740 | 0.8539 | 0.4450 | 7.8440 | 58.8493 | 84.7014 | 93.7311 | PASS* | PASS (Grade B)* |
| personal_profile_code32_reliability | DBP | 3.0540 | 0.8572 | 0.0296 | 4.6999 | 82.4915 | 96.1799 | 98.7994 | PASS* | PASS (Grade A)* |
| personal_profile_code64_reliability | SBP | 5.4982 | 0.8563 | 0.3263 | 7.7853 | 59.6928 | 84.9549 | 93.9968 | PASS* | PASS (Grade B)* |
| personal_profile_code64_reliability | DBP | 3.0087 | 0.8605 | -0.0945 | 4.6455 | 82.9656 | 96.2031 | 98.8103 | PASS* | PASS (Grade A)* |
| personal_profile_code32_stable_only | SBP | 6.5280 | 0.8068 | 0.3706 | 9.0266 | 51.4871 | 79.3467 | 91.2799 | FAIL* | PASS (Grade B)* |
| personal_profile_code32_stable_only | DBP | 3.5683 | 0.8199 | -0.2243 | 5.2733 | 77.1550 | 94.5051 | 98.3874 | PASS* | PASS (Grade A)* |

### chronological_blocked — MIMIC


Participant-macro MAE（mmHg）：

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | 1011 | 40440 | 4.6312 | 2.5143 | 3.5728 |
| personal_profile_code32_no_reliability | 1011 | 40440 | 5.3711 | 2.8362 | 4.1036 |
| personal_profile_code64_reliability | 1011 | 40440 | 5.3906 | 2.8760 | 4.1333 |
| personal_profile_code32_no_support | 1011 | 40440 | 5.4929 | 2.8934 | 4.1931 |
| personal_profile_code32_reliability | 1011 | 40440 | 5.5030 | 2.9359 | 4.2195 |
| personal_profile_support_only | 1011 | 40440 | 6.2693 | 3.3088 | 4.7890 |
| personal_profile_code32_stable_only | 1011 | 40440 | 6.5409 | 3.4202 | 4.9806 |
| residual_reference | 1011 | 40440 | 7.1302 | 3.7268 | 5.4285 |

窗口汇总诊断；≤5/10/15的数值单位为百分比：

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual_reference | SBP | 7.1302 | 0.7953 | 0.3140 | 10.2485 | 50.9842 | 77.1266 | 88.1775 | FAIL* | FAIL (Grade C)* |
| residual_reference | DBP | 3.7268 | 0.8054 | 0.1118 | 5.7011 | 76.2587 | 93.1058 | 97.4926 | PASS* | PASS (Grade A)* |
| subject_lora_rank4 | SBP | 4.6312 | 0.9057 | 0.1424 | 6.9561 | 68.8304 | 89.2582 | 95.4698 | PASS* | PASS (Grade A)* |
| subject_lora_rank4 | DBP | 2.5143 | 0.8878 | 0.0031 | 4.3299 | 87.5025 | 96.7829 | 98.7760 | PASS* | PASS (Grade A)* |
| personal_profile_support_only | SBP | 6.2693 | 0.8414 | 0.4604 | 9.0123 | 55.5687 | 81.2290 | 91.3798 | FAIL* | PASS (Grade B)* |
| personal_profile_support_only | DBP | 3.3088 | 0.8412 | -0.0125 | 5.1512 | 80.4253 | 94.5772 | 98.1231 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_support | SBP | 5.4929 | 0.8760 | 0.2932 | 7.9747 | 60.6998 | 85.2448 | 93.7957 | PASS* | PASS (Grade B)* |
| personal_profile_code32_no_support | DBP | 2.8934 | 0.8738 | -0.0002 | 4.5925 | 84.6414 | 95.9125 | 98.4149 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_reliability | SBP | 5.3711 | 0.8832 | 0.2521 | 7.7403 | 61.3106 | 85.7938 | 94.2334 | PASS* | PASS (Grade B)* |
| personal_profile_code32_no_reliability | DBP | 2.8362 | 0.8762 | -0.0367 | 4.5472 | 84.9555 | 96.0682 | 98.4965 | PASS* | PASS (Grade A)* |
| personal_profile_code32_reliability | SBP | 5.5030 | 0.8770 | 0.3405 | 7.9412 | 60.6380 | 85.0643 | 93.5262 | PASS* | PASS (Grade B)* |
| personal_profile_code32_reliability | DBP | 2.9359 | 0.8684 | -0.0231 | 4.6891 | 84.2483 | 96.0856 | 98.3778 | PASS* | PASS (Grade A)* |
| personal_profile_code64_reliability | SBP | 5.3906 | 0.8798 | 0.4396 | 7.8452 | 61.6568 | 85.9347 | 93.8526 | PASS* | PASS (Grade B)* |
| personal_profile_code64_reliability | DBP | 2.8760 | 0.8716 | -0.0307 | 4.6318 | 84.8170 | 96.0163 | 98.4570 | PASS* | PASS (Grade A)* |
| personal_profile_code32_stable_only | SBP | 6.5409 | 0.8347 | 0.3512 | 9.2065 | 52.4753 | 79.6118 | 91.0930 | FAIL* | PASS (Grade B)* |
| personal_profile_code32_stable_only | DBP | 3.4202 | 0.8338 | -0.1552 | 5.2678 | 79.8343 | 94.7404 | 97.9896 | PASS* | PASS (Grade A)* |

### chronological_blocked — VitalDB


Participant-macro MAE（mmHg）：

| candidate | n_participants | n_events | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | 1040 | 41600 | 5.1196 | 2.9003 | 4.0100 |
| personal_profile_code32_no_reliability | 1040 | 41600 | 5.5695 | 3.1407 | 4.3551 |
| personal_profile_code64_reliability | 1040 | 41600 | 5.6029 | 3.1377 | 4.3703 |
| personal_profile_code32_reliability | 1040 | 41600 | 5.6430 | 3.1687 | 4.4059 |
| personal_profile_code32_no_support | 1040 | 41600 | 5.6732 | 3.1788 | 4.4260 |
| personal_profile_code32_stable_only | 1040 | 41600 | 6.5155 | 3.7122 | 5.1139 |
| personal_profile_support_only | 1040 | 41600 | 6.6108 | 3.8130 | 5.2119 |
| residual_reference | 1040 | 41600 | 8.7271 | 4.8289 | 6.7780 |

窗口汇总诊断；≤5/10/15的数值单位为百分比：

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual_reference | SBP | 8.7271 | 0.5664 | -0.1051 | 11.7061 | 40.0769 | 66.9255 | 82.2428 | FAIL* | FAIL (Grade D)* |
| residual_reference | DBP | 4.8289 | 0.6740 | -0.1604 | 6.7093 | 63.1226 | 88.1274 | 96.5793 | PASS* | PASS (Grade A)* |
| subject_lora_rank4 | SBP | 5.1196 | 0.8274 | 0.0005 | 7.3865 | 63.1250 | 86.7043 | 94.7356 | PASS* | PASS (Grade B)* |
| subject_lora_rank4 | DBP | 2.9003 | 0.8505 | 0.0383 | 4.5451 | 83.4760 | 96.5096 | 98.9736 | PASS* | PASS (Grade A)* |
| personal_profile_support_only | SBP | 6.6108 | 0.7453 | 0.5829 | 8.9530 | 50.6370 | 78.5192 | 90.8654 | FAIL* | PASS (Grade B)* |
| personal_profile_support_only | DBP | 3.8130 | 0.7861 | 0.0693 | 5.4356 | 73.3365 | 93.5433 | 98.5625 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_support | SBP | 5.6732 | 0.8044 | 0.2351 | 7.8591 | 57.4880 | 83.9279 | 93.6394 | PASS* | PASS (Grade B)* |
| personal_profile_code32_no_support | DBP | 3.1788 | 0.8386 | -0.1217 | 4.7208 | 80.5433 | 96.2404 | 99.0865 | PASS* | PASS (Grade A)* |
| personal_profile_code32_no_reliability | SBP | 5.5695 | 0.8137 | -0.1273 | 7.6709 | 57.8678 | 84.5481 | 94.2139 | PASS* | PASS (Grade B)* |
| personal_profile_code32_no_reliability | DBP | 3.1407 | 0.8421 | -0.1838 | 4.6663 | 81.0745 | 96.4327 | 99.2043 | PASS* | PASS (Grade A)* |
| personal_profile_code32_reliability | SBP | 5.6430 | 0.8091 | 0.5466 | 7.7472 | 57.1106 | 84.3486 | 93.9303 | PASS* | PASS (Grade B)* |
| personal_profile_code32_reliability | DBP | 3.1687 | 0.8394 | 0.0810 | 4.7098 | 80.7837 | 96.2716 | 99.2091 | PASS* | PASS (Grade A)* |
| personal_profile_code64_reliability | SBP | 5.6029 | 0.8110 | 0.2161 | 7.7252 | 57.7837 | 84.0024 | 94.1370 | PASS* | PASS (Grade B)* |
| personal_profile_code64_reliability | DBP | 3.1377 | 0.8428 | -0.1564 | 4.6581 | 81.1659 | 96.3846 | 99.1538 | PASS* | PASS (Grade A)* |
| personal_profile_code32_stable_only | SBP | 6.5155 | 0.7518 | 0.3894 | 8.8483 | 50.5264 | 79.0889 | 91.4615 | FAIL* | PASS (Grade B)* |
| personal_profile_code32_stable_only | DBP | 3.7122 | 0.7978 | -0.2915 | 5.2778 | 74.5505 | 94.2764 | 98.7740 | PASS* | PASS (Grade A)* |

## 完成证据与复现

训练任务1398–1405、1407–1414；报告1406、1415、1416。
训练代码版本9bc4aad。已重新读取16份验证预测，逐一检查窗口与标签完全一致，
并重算三个范围的MAE、R²、ME、STD及误差范围比例，全部匹配报告。
运行元数据、预测、个人映射/档案与诊断文件均与NAS归档字节一致；三份报告目录亦一致。
本次没有重新散列checkpoint二进制文件。完整检查项和报告SHA-256见
[verification.json](../results/same_subject_personal_profiles/verification.json)。

[全部聚合结果](../results/same_subject_personal_profiles/)包含48行participant-macro记录、
96行窗口诊断以及六份按划分/来源拆开的表格。原始波形、个人数据和checkpoint不公开。
