# 官方 CalBased 六方法结果

2026 年 9 月 9 日完成核验。全部正式任务正常完成，最终评分任务于当日
06:42:21 CST 结束。此处公开的是保存预测后的汇总结果，不是新的重训练。

## 主要结论

固定个人参考记忆在三个统计范围都改善了配对 LoRA。另四个学习型候选
v1、E1、scalar trust、BP-specific trust 虽然实际训练了 8 轮，但内部验证
均选择第 0 轮的固定记忆初始化。其相同分数不能被解释为四个新模块分别有效。

| 方法 | Overall SBP / DBP MAE | MIMIC SBP / DBP MAE | VitalDB SBP / DBP MAE |
|---|---:|---:|---:|
| LoRA | 3.8207 / 2.0994 | 4.1920 / 2.2802 | 3.4724 / 1.9298 |
| LoRA + 固定参考记忆 | 3.4210 / 1.8858 | 3.8073 / 2.0699 | 3.0587 / 1.7132 |

单位 mmHg，主指标为 participant-macro MAE。Overall 综合 MAE 从 2.9601
降至 2.6534。全部候选、全部来源的标准格式见 [完整指标表](RESULT_TABLES.md)。

## 划分与限制

- 保留官方 CalBased 的原始角色成员：2,506 人，每人 360 训练、40 测试窗口；
  902,160 训练窗口，100,240 测试窗口。受试者有意重合。
- 审计中确认的 35 个跨 TRAIN/TEST 重复内容行按用户授权保留，没有自行删除。
  因而不能声称全部角色严格波形内容不重复。内部选择与 OOF 也有已披露的
  原生重复内容；具体限制见 [评价收据](evaluation_receipt.json) 与
  [持续实验记录](../../docs/STATUS.md)。
- 内部训练轮数选择在合法训练角色内完成，正式测试预测冻结后由独立评分步骤
  加入标签。评价收据记录 test_based_model_selection=false。
- MIMIC 与 VitalDB 是 PulseDB 内部来源分层，不是外部数据集验证。
- AAMI 和 BHS 列只代表本项目的数值筛查，不表示完成标准全部条件或临床认证。
- 这是已有档案受试者基准；不能用来证明新用户免校准。另见
  [30 人后注册实验](../post_enrollment_30_v1_20260909/README.md)。

## 文件入口

- [主要宏平均指标](participant_macro.csv)
- [完整诊断指标](diagnostic_tables.csv)
- [Overall](Overall_diagnostics.csv)、[MIMIC](MIMIC_diagnostics.csv)、[VitalDB](VitalDB_diagnostics.csv)
- [中文模型与后续采集说明](../../docs/PERSONAL_LORA_MEMORY_METHOD_ZH.md)

本目录不含原始波形、逐人预测或个人模型参数。
