# LoRA个人化机制：冻结模型诊断

日期：2026-09-06。诊断任务1474–1477均完成，未训练新模型、未访问封存测试集。

## 结论

现有LoRA的好结果不是仅依靠个人血压均值：同一人的PPG窗口打乱后，随机划分mean MAE
从3.0394升至10.6230/10.5665；时间划分从3.7945升至5.4063/5.3846。
只用训练血压均值分别为7.9793/7.4747。保留正确PPG和血压均值、但换成另一人的
个人参数时，分别为11.0013/10.2862。这些结果支持模型依赖**当前窗口信息和正确的个人参数**。

32D no-support模型同样明显依赖PPG和个人参数，但正常误差为4.0729/4.3112，仍差于LoRA。
所以“有没有使用个人信息”不能解释两者全部差距。代码检查确认：LoRA直接改变256D波形
特征的映射，每人2,048个训练参数；32D输出修正模型每人34个。**特征变换形式与个人容量
是需要继续检验的解释，不是本诊断已经证明的唯一原因。**

## 怎么测的

- 使用之前完成的1399/1408 LoRA和1401/1410 no-support个人模型，各自在原划分下检查。
- 每种划分2,051人、82,040个内部验证窗口；MIMIC1,011人、VitalDB1,040人。
- 先复现原checkpoint预测，再验证缓存PPG特征重放与完整网络一致。
  四组最终检查的两项最大差异均为0.0 mmHg；checkpoint SHA-256已核对。
- 两次同人PPG置换均保证无固定点；仅使用身份和随机数决定配对，不用BP标签。
- 换人实验在同一数据来源内交换个人参数，始终保留接收者正确的PPG和训练血压均值。
- 关闭适配时，LoRA的B矩阵置零，Profile的个人code/bias置零；每次恢复，源checkpoint不修改。
- 训练均值、训练中位数对照只读320个train窗口标签；验证标签只负责评分。

首次随机LoRA诊断1472在另一型号GPU上触发预设数值复现门槛：单点最大差0.3321、
平均差0.000891 mmHg，未进入扰动比较。未放宽门槛；换回原RTX5070Ti后精确复现。
1473是旧缓存检查版本的时间LoRA诊断；正式表仅使用加入缓存一致性检查的1474–1477。

## Overall结果

下表为受试者等权的(SBP MAE+DBP MAE)/2，单位mmHg，越低越好。
natural=正常；ppg_permuted_1/2=同人窗口置换；personal_state_swapped=换人参数；
personal_state_zero=关闭个人适配；train_bp_mean/median=只用训练标签均值/中位数。

| candidate | condition | chronological_blocked | random_disjoint |
| --- | --- | --- | --- |
| personal_profile_code32_no_support | natural | 4.3112 | 4.0729 |
| personal_profile_code32_no_support | personal_state_swapped | 11.0692 | 11.2581 |
| personal_profile_code32_no_support | personal_state_zero | 8.5006 | 8.3424 |
| personal_profile_code32_no_support | ppg_permuted_1 | 5.7424 | 10.2018 |
| personal_profile_code32_no_support | ppg_permuted_2 | 5.7192 | 10.1548 |
| personal_profile_code32_no_support | train_bp_mean | 7.4747 | 7.9793 |
| personal_profile_code32_no_support | train_bp_median | 7.5041 | 7.8627 |
| subject_lora_rank4 | natural | 3.7945 | 3.0394 |
| subject_lora_rank4 | personal_state_swapped | 10.2862 | 11.0013 |
| subject_lora_rank4 | personal_state_zero | 7.2952 | 7.4993 |
| subject_lora_rank4 | ppg_permuted_1 | 5.4063 | 10.6230 |
| subject_lora_rank4 | ppg_permuted_2 | 5.3846 | 10.5665 |
| subject_lora_rank4 | train_bp_mean | 7.4747 | 7.9793 |
| subject_lora_rank4 | train_bp_median | 7.5041 | 7.8627 |

## 分来源结果与完整指标

以下保留每种来源和划分的同一批预测，不剔除困难人群。完整标准列
Setting、BP、MAE、R²、ME、STD、≤5/10/15 mmHg百分比、AAMI、BHS见：

- [随机 Overall](../results/personal_mechanisms/random_disjoint/overall_diagnostics.csv)、[MIMIC](../results/personal_mechanisms/random_disjoint/mimic_diagnostics.csv)、[VitalDB](../results/personal_mechanisms/random_disjoint/vitaldb_diagnostics.csv)。
- [时间 Overall](../results/personal_mechanisms/chronological_blocked/overall_diagnostics.csv)、[MIMIC](../results/personal_mechanisms/chronological_blocked/mimic_diagnostics.csv)、[VitalDB](../results/personal_mechanisms/chronological_blocked/vitaldb_diagnostics.csv)。

STD是逐窗口预测误差的样本标准差，不是MAE跨人的标准差。AAMI/BHS带星号仅表示
离线数值筛查，不是临床标准验证通过。主比较是participant-macro指标；这里每人相同
40个窗口，因此MAE与逐窗口汇总相符，R²/STD仍应按对应定义解释。

| candidate | split_mode | condition | Scope | sbp_mae | dbp_mae | mean_mae |
| --- | --- | --- | --- | --- | --- | --- |
| subject_lora_rank4 | random_disjoint | natural | MIMIC | 4.2769 | 2.3643 | 3.3206 |
| subject_lora_rank4 | random_disjoint | natural | VitalDB | 3.5372 | 1.9951 | 2.7661 |
| subject_lora_rank4 | random_disjoint | ppg_permuted_1 | MIMIC | 13.0445 | 6.7814 | 9.9130 |
| subject_lora_rank4 | random_disjoint | ppg_permuted_1 | VitalDB | 14.6531 | 7.9734 | 11.3132 |
| subject_lora_rank4 | random_disjoint | ppg_permuted_2 | MIMIC | 12.9918 | 6.7510 | 9.8714 |
| subject_lora_rank4 | random_disjoint | ppg_permuted_2 | VitalDB | 14.5601 | 7.9241 | 11.2421 |
| subject_lora_rank4 | random_disjoint | personal_state_swapped | MIMIC | 14.4438 | 7.5490 | 10.9964 |
| subject_lora_rank4 | random_disjoint | personal_state_swapped | VitalDB | 14.1215 | 7.8907 | 11.0061 |
| subject_lora_rank4 | random_disjoint | personal_state_zero | MIMIC | 9.8864 | 5.3174 | 7.6019 |
| subject_lora_rank4 | random_disjoint | personal_state_zero | VitalDB | 9.5835 | 5.2158 | 7.3997 |
| subject_lora_rank4 | random_disjoint | train_bp_mean | MIMIC | 9.9727 | 5.2203 | 7.5965 |
| subject_lora_rank4 | random_disjoint | train_bp_mean | VitalDB | 10.8476 | 5.8554 | 8.3515 |
| subject_lora_rank4 | random_disjoint | train_bp_median | MIMIC | 9.8334 | 5.1018 | 7.4676 |
| subject_lora_rank4 | random_disjoint | train_bp_median | VitalDB | 10.7106 | 5.7828 | 8.2467 |
| personal_profile_code32_no_support | random_disjoint | natural | MIMIC | 5.5905 | 3.0812 | 4.3359 |
| personal_profile_code32_no_support | random_disjoint | natural | VitalDB | 4.8908 | 2.7437 | 3.8173 |
| personal_profile_code32_no_support | random_disjoint | ppg_permuted_1 | MIMIC | 12.4541 | 6.3873 | 9.4207 |
| personal_profile_code32_no_support | random_disjoint | ppg_permuted_1 | VitalDB | 14.2604 | 7.6618 | 10.9611 |
| personal_profile_code32_no_support | random_disjoint | ppg_permuted_2 | MIMIC | 12.3912 | 6.3461 | 9.3687 |
| personal_profile_code32_no_support | random_disjoint | ppg_permuted_2 | VitalDB | 14.2060 | 7.6323 | 10.9191 |
| personal_profile_code32_no_support | random_disjoint | personal_state_swapped | MIMIC | 14.5728 | 7.6880 | 11.1304 |
| personal_profile_code32_no_support | random_disjoint | personal_state_swapped | VitalDB | 14.6297 | 8.1348 | 11.3823 |
| personal_profile_code32_no_support | random_disjoint | personal_state_zero | MIMIC | 11.1898 | 6.0933 | 8.6415 |
| personal_profile_code32_no_support | random_disjoint | personal_state_zero | VitalDB | 10.3270 | 5.7761 | 8.0515 |
| personal_profile_code32_no_support | random_disjoint | train_bp_mean | MIMIC | 9.9727 | 5.2203 | 7.5965 |
| personal_profile_code32_no_support | random_disjoint | train_bp_mean | VitalDB | 10.8476 | 5.8554 | 8.3515 |
| personal_profile_code32_no_support | random_disjoint | train_bp_median | MIMIC | 9.8334 | 5.1018 | 7.4676 |
| personal_profile_code32_no_support | random_disjoint | train_bp_median | VitalDB | 10.7106 | 5.7828 | 8.2467 |
| personal_profile_code32_no_support | chronological_blocked | natural | MIMIC | 5.4929 | 2.8934 | 4.1931 |
| personal_profile_code32_no_support | chronological_blocked | natural | VitalDB | 5.6732 | 3.1788 | 4.4260 |
| personal_profile_code32_no_support | chronological_blocked | ppg_permuted_1 | MIMIC | 6.7671 | 3.5182 | 5.1426 |
| personal_profile_code32_no_support | chronological_blocked | ppg_permuted_1 | VitalDB | 8.1127 | 4.5380 | 6.3254 |
| personal_profile_code32_no_support | chronological_blocked | ppg_permuted_2 | MIMIC | 6.7604 | 3.5084 | 5.1344 |
| personal_profile_code32_no_support | chronological_blocked | ppg_permuted_2 | VitalDB | 8.0670 | 4.5085 | 6.2877 |
| personal_profile_code32_no_support | chronological_blocked | personal_state_swapped | MIMIC | 13.3820 | 6.9452 | 10.1636 |
| personal_profile_code32_no_support | chronological_blocked | personal_state_swapped | VitalDB | 15.2744 | 8.6246 | 11.9495 |
| personal_profile_code32_no_support | chronological_blocked | personal_state_zero | MIMIC | 11.2028 | 5.8827 | 8.5427 |
| personal_profile_code32_no_support | chronological_blocked | personal_state_zero | VitalDB | 10.8741 | 6.0453 | 8.4597 |
| personal_profile_code32_no_support | chronological_blocked | train_bp_mean | MIMIC | 9.1618 | 4.6239 | 6.8928 |
| personal_profile_code32_no_support | chronological_blocked | train_bp_mean | VitalDB | 10.3757 | 5.7050 | 8.0404 |
| personal_profile_code32_no_support | chronological_blocked | train_bp_median | MIMIC | 9.2474 | 4.6353 | 6.9413 |
| personal_profile_code32_no_support | chronological_blocked | train_bp_median | VitalDB | 10.4033 | 5.6992 | 8.0512 |
| subject_lora_rank4 | chronological_blocked | natural | MIMIC | 4.6312 | 2.5143 | 3.5728 |
| subject_lora_rank4 | chronological_blocked | natural | VitalDB | 5.1196 | 2.9003 | 4.0100 |
| subject_lora_rank4 | chronological_blocked | ppg_permuted_1 | MIMIC | 6.2073 | 3.2600 | 4.7337 |
| subject_lora_rank4 | chronological_blocked | ppg_permuted_1 | VitalDB | 7.7858 | 4.3347 | 6.0603 |
| subject_lora_rank4 | chronological_blocked | ppg_permuted_2 | MIMIC | 6.2216 | 3.2593 | 4.7404 |
| subject_lora_rank4 | chronological_blocked | ppg_permuted_2 | VitalDB | 7.7151 | 4.3065 | 6.0108 |
| subject_lora_rank4 | chronological_blocked | personal_state_swapped | MIMIC | 12.5990 | 6.4431 | 9.5210 |
| subject_lora_rank4 | chronological_blocked | personal_state_swapped | VitalDB | 14.1978 | 7.8622 | 11.0300 |
| subject_lora_rank4 | chronological_blocked | personal_state_zero | MIMIC | 9.4772 | 4.9407 | 7.2090 |
| subject_lora_rank4 | chronological_blocked | personal_state_zero | VitalDB | 9.4719 | 5.2863 | 7.3791 |
| subject_lora_rank4 | chronological_blocked | train_bp_mean | MIMIC | 9.1618 | 4.6239 | 6.8928 |
| subject_lora_rank4 | chronological_blocked | train_bp_mean | VitalDB | 10.3757 | 5.7050 | 8.0404 |
| subject_lora_rank4 | chronological_blocked | train_bp_median | MIMIC | 9.2474 | 4.6353 | 6.9413 |
| subject_lora_rank4 | chronological_blocked | train_bp_median | VitalDB | 10.4033 | 5.6992 | 8.0512 |

## 不能从这个结果推出什么

1. 不能把误差上升分配为“PPG贡献百分之几、身份贡献百分之几”。网络存在共同适应，
   扰动也可能把输入送到训练分布外。关闭已训练的适配不等于从头训练无适配模型。
2. 随机/时间划分内的PPG置换幅度和血压变化范围不同，不能拿二者误差增幅直接比较
   生理信息量。时间窗口置换可能仍保留较近状态。
3. 这不是新用户少样本校准，也不是官方PulseDB CalBased分割的严格复现。
   每人仍使用320个带BP标签的10秒训练窗口，不等于320次袖带测量。
4. 对同一内部验证集反复探索会带来选择偏差；新机制是否有效仍需完整对照和后续确认。

## 已冻结的下一轮

[训练计划及公式](PLAN_PERSONAL_FEATURE_MECHANISMS.md)包含8个候选、两种划分。
重点检验相同个人参数量下的非线性个人响应，并用共享适配、rank1、输出修正、
特征仿射和共享双线性方向比较机制。参数共享思路有先例，不能仅改名宣称原创。
输出Profile与新特征模块虽有相同个人参数预算，但共享头和描述符输入不同，属于
架构比较，不是严格仅移动一个模块的位置。最干净的单变量比较是rank4线性/非线性。
训练主干与个人参数均联合优化，不能表述为完全复现原LoRA冻结主干微调方案。
