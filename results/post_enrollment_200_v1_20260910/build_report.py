"""Rebuild the human-readable report from verified public aggregate files."""
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
verification = json.loads((ROOT / "verification_summary.json").read_text(encoding="utf-8"))
assert verification["status"] == "pass"
for original, expected in verification["archived_aggregate_sha256"].items():
    stage, filename = original.split("/")
    local = ROOT / {"evaluate": "enrollment", "population_benchmark": "population"}[stage] / filename
    assert hashlib.sha256(local.read_bytes()).hexdigest() == expected
assert hashlib.sha256((ROOT / "inspect_completed_run.py").read_bytes()).hexdigest() == verification["inspector_sha256"]

macro = pd.read_csv(ROOT / "enrollment/participant_macro.csv")
diagnostic = pd.read_csv(ROOT / "enrollment/diagnostic_tables.csv")
intervals = pd.read_csv(ROOT / "enrollment/paired_participant_intervals.csv")
population = pd.read_csv(ROOT / "population/participant_macro.csv")
labels = {
    "personal_mean": "个人 BP 均值（不使用当前 PPG）",
    "shared_with_anchor": "共享网络 + 个人 BP 基准",
    "new_person_lora": "单独个人 LoRA",
    "new_person_lora_memory": "个人 LoRA + 参考记忆（完整方案）",
    "shared_memory_only": "共享特征检索记忆（无个人 LoRA）",
    "shared_memory_blend": "共享网络预测 + 记忆融合（无个人 LoRA）",
    "lora_memory_only": "LoRA 特征检索记忆（不融合预测头）",
    "lora_memory_uniform": "LoRA + 记忆，邻居等权",
    "lora_memory_fixed_half": "LoRA + 记忆，固定各占一半",
}


def row(method, scope="Overall"):
    return macro.loc[macro.Setting.eq(method) & macro.Scope.eq(scope)].iloc[0]


def pair(method, scope):
    value = row(method, scope)
    return f"{value.sbp_mae:.4f} / {value.dbp_mae:.4f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, values)) + " |" for values in rows])


primary = row("new_person_lora_memory")
baseline = row("new_person_lora")
gain = baseline.mean_mae - primary.mean_mae
reduction = gain / baseline.mean_mae * 100
summary_rows = [[labels[method], *[pair(method, scope) for scope in ("Overall", "MIMIC", "VitalDB")], f"{row(method).mean_mae:.4f}"] for method in labels]
comparison_rows = []
for scope in ("Overall", "MIMIC", "VitalDB"):
    ref, candidate = row("new_person_lora", scope), row("new_person_lora_memory", scope)
    ci = intervals.loc[intervals.Setting.eq("new_person_lora_memory") & intervals.Scope.eq(scope) & intervals.BP.eq("Mean")].iloc[0]
    comparison_rows.append([scope, f"{ref.mean_mae:.4f}", f"{candidate.mean_mae:.4f}", f"{ci.gain_mmHg:.4f}",
        f"{100 * ci.gain_mmHg / ref.mean_mae:.2f}%", f"[{ci.ci95_low_mmHg:.4f}, {ci.ci95_high_mmHg:.4f}]",
        f"{int(ci.improved_participants)}/{int(ci.participants)}"])

lines = ["# 200 人新用户建档与个人参考记忆：结果及消融", "",
    "核验日期：2026-09-10。协议：`post-enrollment-200-v1`。全部阶段正常完成；本次只整理和发布结果，没有提交新训练。", "",
    "## 结论", "",
    f"从公共模型训练中完整排除 200 人，再为这些人分别建立个人 LoRA 和参考记忆后，完整方案的 Overall SBP/DBP MAE 为 **{primary.sbp_mae:.4f}/{primary.dbp_mae:.4f} mmHg**。相较同一批人的单独 LoRA，SBP/DBP 平均 MAE 从 {baseline.mean_mae:.4f} 降至 {primary.mean_mae:.4f}，降低 **{reduction:.2f}%**。186/200 人改善，14 人变差；没有剔除任何一个人。", "",
    "完整方案在本轮九种设置中取得最低 Overall 平均 MAE，MIMIC 和 VitalDB 的平均 MAE 也分别最低。不过，记忆检索本身已经很强；加权和融合带来的额外收益小于加入记忆的收益，不能说每个模块都不可缺少。保留 LoRA + 个人参考记忆为候选方法，同时保留两个 memory-only 对照。", "",
    "## 一、这次究竟怎样训练和测试", "",
    "1. 固定随机种子 20260909，按来源各抽取 100 人，仅用身份和来源选人，不按血压或预测误差选人。这 200 人完全不参加公共模型的训练、内部验证、目标标准化或公共个人参数拟合。",
    "2. 内容审计发现一名外部 MIMIC 身份与入选者存在 5 组相同 PPG 内容。保留原来的 200 人，按预先记录的无标签关联隔离规则，额外排除该外部身份的全部 400 行，剩余公共人群为 2,305 人。未删除或改写原始数据。",
    "3. 公共模型从零训练。每名公共人群受试者先使用 320/40 个允许训练窗口进行内部选轮，再按选出的 35 轮，用全部 360 个允许训练窗口重新训练。公共人群测试集包含 92,200 个窗口，单独评分。",
    "4. 公共模型完成后，对被排除的 200 人逐人建档：每人保留原官方成员关系中的 360 个带 BP 标签的注册窗口，剩余 40 个窗口测试。注册内部 320/40 选个人训练轮数，再重新用全部 360 个窗口拟合该人的专属参数和参考库。测试 BP 不参与这些步骤。",
    "5. 九种输出冻结后才关联测试 BP 评分。全部 200 人、8,000 个测试窗口保留；不筛掉最差 30%，也不按误差重新选人。三个来源视图由同一批预测分别重算。", "",
    "| 人群 | MIMIC / VitalDB 人数 | 训练或注册窗口 | 测试窗口 |",
    "|---|---:|---:|---:|",
    "| 剩余公共人群 | 1,112 / 1,193 | 829,800 | 92,200 |",
    "| 公共模型未见的建档人群 | 100 / 100 | 72,000 | 8,000 |", "",
    "这是保留官方个人内成员关系的**整人排除衍生实验**，不是原封不动的 2,506 人官方基准。它仍是不同窗口的测试，不是严格早期建档、未来日期测试。360 个 ABP 标注的 10 秒窗口不等于 360 次独立袖带测量，也不是 K=1/2/3/5 次袖带校准。", "",
    "## 二、模型与消融设置", "",
    "共享部分为 `resnet_small`，输出 256 维 PPG 特征。新用户的个人 LoRA 是特征空间 rank-4 低秩变换，每人仅训练 2,048 个参数；共享网络、输出头、标准化参数及 BN 状态保持不变。它不是 Transformer attention 上的 LoRA。", "",
    "参考库只保存该人的允许注册记录。完整方案在个人特征空间按余弦相似度取 top-5，用温度 0.1 的 softmax 加权历史 BP，再按注册数据估计的距离可信程度，与当前 LoRA 预测融合。距离尺度由本人注册数据内部 leave-40-block 估计，不使用测试 BP。", "",
    "| 设置标识 | 本轮改变的部分 |",
    "|---|---|",
    "| `personal_mean` | 只用该人注册 BP 的平均值，不读取当前测试 PPG |",
    "| `shared_with_anchor` | 共享 PPG 网络加该人 BP 基准，不使用个人低秩参数 |",
    "| `new_person_lora` | 该人独立训练的特征 LoRA，不检索记忆 |",
    "| `new_person_lora_memory` | 个人 LoRA + top-5 相似度加权参考记忆 + 距离融合 |",
    "| `shared_memory_only` | 不使用个人 LoRA；共享特征找该人历史邻居，直接输出邻居 BP |",
    "| `shared_memory_blend` | 不使用个人 LoRA；共享预测与共享特征记忆按同一规则融合 |",
    "| `lora_memory_only` | 保留 LoRA 特征检索，只输出邻居 BP，不融合当前预测头 |",
    "| `lora_memory_uniform` | 同一组个人 top-5 邻居等权平均，其余不变 |",
    "| `lora_memory_fixed_half` | 保留相似度权重，记忆和 LoRA 预测固定各占 50% |", "",
    "九种设置共享同一公共模型和相应个人建档预算；不是九次公共主干重训。所有个人参数均实际训练：选中轮数 6–1,225，中位数 284.5，无 epoch-0 个人档案。由于复用冻结的 PPG 特征，个人训练快不代表没有训练。", "",
    "## 三、全部设置结果", "",
    "主指标为先计算每人的 MAE，再对人平均（participant-macro）。下表每个双值为 **SBP / DBP MAE，单位 mmHg**；最后一列为 Overall 两种 BP 的平均 MAE，越低越好。",
    "",
    table(["方法", "Overall（200 人）", "MIMIC（100 人）", "VitalDB（100 人）", "Overall 平均 MAE"], summary_rows), "",
    "## 四、完整方案相较配对 LoRA 的改善", "",
    table(["Scope", "LoRA 平均 MAE", "完整方案平均 MAE", "降低 mmHg", "相对降低", "降低量 95% CI", "改善人数"], comparison_rows), "",
    "预先指定的主要比较为 Overall 的 SBP/DBP 平均 MAE。置信区间按受试者配对，并在 MIMIC/VitalDB 内分别重采样，2,000 次 bootstrap，固定种子 20260909。不能把 8,000 个窗口当成 8,000 个独立人。该区间只描述本次固定模型下的人群抽样不确定性，不包含重复训练种子的变化。其余区间是探索性逐项区间，不作多重检验通过的声明。", "",
    "具体到 BP，完整方案改善 SBP 的人数为 177/200，DBP 为 183/200；两者平均改善 186/200。平均 MAE 的改善中位数为 0.3605 mmHg，最差的一人反而增加 0.5538 mmHg。各来源均为 93/100 人平均改善，并非所有人都改善。", "",
    "[完整配对区间](enrollment/paired_participant_intervals.csv) · [固定区间计算约定](enrollment/uncertainty_contract.json)", "",
    "## 五、按要求报告 MAE、R²、ME、STD、误差比例及数值筛查", "",
    "MAE 以人平均为主；因本轮每人同为 40 个测试窗口，窗口合并 MAE 与人平均 MAE 相同。R²、ME、STD 和阈值比例在对应来源的全部窗口上计算。ME = 预测 − 参考；STD 是有符号误差的样本标准差（ddof=1），不是训练种子或个人 MAE 的标准差。", "",
    "下列 AAMI/BHS 沿用本项目既定脚本的回顾性数值筛查：AAMI 栏检查 |ME|≤5 且 STD≤8；BHS 按 ≤5/10/15 mmHg 的比例分级。表中的 PASS* / Grade A* 不代表临床设备认证、完整标准合规或实际袖带/手表验证。", ""]

columns = ["Setting", "BP", "MAE", "R²", "ME", "STD", "≤5 mmHg", "≤10 mmHg", "≤15 mmHg", "AAMI", "BHS"]
for scope in ("Overall", "MIMIC", "VitalDB"):
    selected = diagnostic.loc[diagnostic.Scope.eq(scope) & diagnostic.Setting.isin(("new_person_lora", "new_person_lora_memory"))]
    rows = []
    for entry in selected.to_dict("records"):
        entry["Setting"] = {"new_person_lora": "LoRA", "new_person_lora_memory": "LoRA + 记忆"}[entry["Setting"]]
        values = []
        for col in columns:
            value = entry[col]
            if col in ("MAE", "R²", "ME", "STD"):
                value = f"{value:.4f}"
            elif col.startswith("≤"):
                value = f"{Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}%"
            values.append(value)
        rows.append(values)
    lines += [f"### {scope}", "", table(columns, rows), ""]

lines += ["[九种设置全部 54 行结果表](enrollment/RESULT_TABLES.md) · [Overall CSV](enrollment/Overall_diagnostics.csv) · [MIMIC CSV](enrollment/MIMIC_diagnostics.csv) · [VitalDB CSV](enrollment/VitalDB_diagnostics.csv)", "",
    "## 六、消融实际说明了什么", "",
    f"- **个人建档有作用。** 共享网络加个人 BP 基准的 Overall 平均 MAE 为 {row('shared_with_anchor').mean_mae:.4f}，单独 LoRA 为 {baseline.mean_mae:.4f}；两者平均 MAE 的配对改善覆盖 200/200 人。这支持在本预算下建立个人参数，而不是直接给新用户套用共享网络。",
    f"- **参考检索是强对照。** 不训练个人 LoRA、只用共享特征检索本人参考 BP，平均 MAE 已达 {row('shared_memory_only').mean_mae:.4f}，优于单独 LoRA。完整方案再降到 {primary.mean_mae:.4f}，增益仅 {row('shared_memory_only').mean_mae - primary.mean_mae:.4f} mmHg（138/200 人改善）。不能把完整方案比 LoRA 好的全部收益都归因于低秩参数与记忆的特殊协同。",
    f"- **融合预测头的增益较小。** 保留 LoRA 特征但只用记忆输出时，平均 MAE 为 {row('lora_memory_only').mean_mae:.4f}，完整方案仅再改善 {row('lora_memory_only').mean_mae - primary.mean_mae:.4f}（122/200 人改善）。VitalDB 的 DBP 在该 memory-only 对照中为 1.6436，反而略好于完整方案的 1.6531。",
    f"- **权重与距离融合只带来小幅增益。** 相较邻居等权和固定一半融合，完整方案分别降低 Overall 平均 MAE {row('lora_memory_uniform').mean_mae - primary.mean_mae:.4f} 和 {row('lora_memory_fixed_half').mean_mae - primary.mean_mae:.4f} mmHg；不能据此声称巨大提升。固定一半融合的 Overall DBP STD 也略低于完整方案。",
    f"- **不是模块越多越好。** 无 LoRA 的共享预测加记忆融合，平均 MAE 为 {row('shared_memory_blend').mean_mae:.4f}，比共享特征的纯记忆输出更差。保留该负结果，而不只展示最佳模型。", "",
    "本节新增的模块间直接配对描述是结果解释，不是新增的预注册主要假设；未为它们重新搜索参数或新增显著性声明。", "",
    "## 七、剩余公共人群的单独基准", "",
    table(["Scope", "人数", "SBP MAE", "DBP MAE", "平均 MAE"], [[entry.Scope, str(entry.n_participants), f"{entry.sbp_mae:.4f}", f"{entry.dbp_mae:.4f}", f"{entry.mean_mae:.4f}"] for entry in population.itertuples()]), "",
    "这是 2,305 名已经参加公共模型拟合者的测试结果，不是上述 200 名后加入者，也不是对新用户零校准能力的证明。它与旧 2,506 人官方结果、30 人建档结果不能直接作配对提升比较。", "",
    "[公共人群完整数值表](population/RESULT_TABLES.md)", "",
    "## 八、完成状态与结果核验", "",
    "最终评价于 **2026-09-09 22:54:55（北京时间）**完成。2026-09-10 查询时本用户队列为空。", "",
    "| Stage | Job | 结果 | 用时 |",
    "|---|---:|---|---:|",
    "| 修订后 GPU smoke | 1735 | COMPLETED 0:0，52 项检查及真实 GPU 检查通过 | 00:00:43 |",
    "| 分区及内容关联审计 | 1736 | COMPLETED 0:0 | 00:00:38 |",
    "| 公共模型内部选轮 | 1737 | COMPLETED 0:0，43 轮选中 35 轮 | 02:25:03 |",
    "| 公共模型最终重训 | 1738 | COMPLETED 0:0，重新训练 35 轮 | 02:03:44 |",
    "| 公共人群评价 | 1739 | COMPLETED 0:0 | 00:00:40 |",
    "| 100 名个人档案，第 1 组 | 1740 | COMPLETED 0:0 | 00:11:24 |",
    "| 100 名个人档案，第 2 组 | 1741 | COMPLETED 0:0 | 00:12:26 |",
    "| 九种输出冻结与评分 | 1742 | COMPLETED 0:0 | 00:00:10 |", "",
    "本次只读核验独立重算 27 行新用户主指标、54 行数值诊断、3 行公共人群主指标、6 行公共人群数值诊断和 72 行既定配对区间，均与保存结果一致。九种设置的受试者、查询键及参考 BP 完全匹配；冻结预测与个人分组输出逐值相同。", "",
    "200 人的共享状态冻结和保存后重载一致性检查全部通过；800 份个人档案、参考库、注册元数据和消融状态文件的哈希与完成记录相符。完整方案 8,000 个查询均有可用参考；没有为了美化误差拒绝窗口。24 份汇总和完成凭据在 work/NAS/本地内容哈希一致。原始信号未在本次复核中重新扫描；内容隔离事实来自冻结的准备审计。", "",
    "初始任务 1728 曾因跨身份相同内容停止，1729–1734 被依赖取消；这属于已记录的准备阶段问题，不能计为训练成功。修订后 1735–1742 全部成功，未把失败任务混入本结果。", "",
    "代码快照：`78ca7f50de8a311a428747758d4be0fbdc39e94f`。源树 SHA-256：`9c8cec628e820052c4c92fca95835f87e697a4c7d8d95469c03809e94fbe9fa3`。",
    "",
    "[核验摘要](verification_summary.json) · [只读核验程序](inspect_completed_run.py) · [报告重建程序](build_report.py) · [实验计划](../../docs/PLAN_POST_ENROLLMENT_200_V1.md) · [持续实验状态](../../docs/STATUS.md)", "",
    "## 九、保留结论与尚未完成的验证", "",
    "可以保留“公共模型 + 新用户独立建档 + 个人参考记忆”这一方向：效果不只出现在旧的训练参与者或原来的 30 人样本里，在本次 200 人建档实验中仍优于配对 LoRA。", "",
    "但这仍是一个种子、一次固定划分下的探索性结果。200 人中有 3 人参加过先前 30 人实验，且整个数据集此前已被用于方法探索，不能称为完全独立的未接触确认集。公共人群仍保留 34 个已披露的官方跨角色相同内容窗口；200 名建档者自身注册/测试相同内容为 0，公共人与建档者的身份及相同内容交集为 0。", "",
    "此前 30 人完整方案为 3.1515/1.7466，本次为 3.3312/1.8442；人群和公共模型不同，不宜用这个差值判断模型退步。应看各自相同人群内相较 LoRA 的改善方向和大小。", "",
    "下一阶段优先验证早期注册、后期或跨日测试；配套个人均值、最近一次参考 BP、单 LoRA、共享特征纯记忆、个人特征纯记忆和完整方案。在腕部采集前固定独立袖带事件定义、参考配对、日期分组与个人更新时点。当前没有开展该时间外推实验、真实手表实验或多种子重复，也不能据此保证现实用户每天都准确。", "",
    "本次未提交新任务。发布内容仅含正式汇总、代码和核验记录；原始 PPG、逐人预测、身份清单、LoRA 权重与个人参考库不上传 GitHub。", ""]

(ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
print("REPORT_REBUILT=yes; aggregate hashes verified=24")
print(f"PRIMARY_GAIN={gain:.12f}; RELATIVE_REDUCTION={reduction:.6f}%")
