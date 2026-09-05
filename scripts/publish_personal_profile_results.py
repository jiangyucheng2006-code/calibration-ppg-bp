"""Export verified, aggregate-only compact personal-profile results and report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODES = {"random_disjoint": 1406, "chronological_blocked": 1415}
SCOPES = ["Overall", "MIMIC", "VitalDB"]
REF = "subject_lora_rank4"
PRIMARY = "personal_profile_code32_reliability"
NAMES = {
    REF: "LoRA rank-4",
    "residual_reference": "普通PPG残差参考",
    "personal_profile_support_only": "仅历史支持信息",
    "personal_profile_code32_no_support": "32D个人状态，无历史支持",
    "personal_profile_code32_no_reliability": "32D个人状态，无可靠性门控",
    PRIMARY: "32D完整主模型",
    "personal_profile_code64_reliability": "64D完整容量对照",
    "personal_profile_code32_stable_only": "32D仅稳定个人修正",
}


def table(frame: pd.DataFrame) -> str:
    def cell(value):
        return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        *["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def build(source: Path, repo: Path) -> None:
    audit = json.loads((source / "audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "pass" and audit["training_run_count"] == 16
    macro_frames, pooled_frames = [], []
    for mode, job in MODES.items():
        folder = source / f"same-subject-personal-profile-v1_{mode}_report_seed20260904_job{job}"
        for name, expected in audit["report_checksums"][mode].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
        selection = json.loads((folder / "selection.json").read_text())
        assert selection["status"] == "complete" and selection["heldout_test_accessed"] is False
        m = pd.read_csv(folder / "participant_macro_summary.csv").drop(columns="run_dir")
        m = m.rename(columns={"view": "Scope"})
        p = pd.read_csv(folder / "event_pooled_diagnostics_all_scopes.csv")
        p.insert(0, "split_mode", mode)
        assert set(m.candidate) == set(NAMES) == set(p.Setting)
        assert len(m) == 24 and len(p) == 48
        assert not m.duplicated(["candidate", "Scope"]).any()
        assert not p.duplicated(["Setting", "Scope", "BP"]).any()
        for candidate in NAMES:
            rows = m.loc[m.candidate.eq(candidate)].set_index("Scope")
            assert set(rows.index) == set(SCOPES)
            for count in ["n_participants", "n_events"]:
                assert rows.loc["Overall", count] == rows.loc[["MIMIC", "VitalDB"], count].sum()
        macro_frames.append(m)
        pooled_frames.append(p)
    macro = pd.concat(macro_frames, ignore_index=True)
    pooled = pd.concat(pooled_frames, ignore_index=True)
    final_dir = source / "same-subject-personal-profile-v1_final_report_seed20260904_job1416"
    selection = json.loads((final_dir / "selection.json").read_text())
    assert selection["primary_candidate"]["passes_robust_gate"] is False
    assert selection["numerical_best"]["candidate"] == REF
    assert selection["heldout_test_accessed"] is False
    comparison = pd.read_csv(final_dir / "personal_profile_comparison.csv")
    comparison = comparison.drop(columns=[c for c in comparison if c.startswith("run_dir")])
    assert not comparison.passes_robust_gate.any()
    output = repo / "results/same_subject_personal_profiles"
    output.mkdir(parents=True, exist_ok=True)
    macro.to_csv(output / "participant_macro.csv", index=False)
    pooled.to_csv(output / "event_pooled_diagnostics.csv", index=False)
    comparison.to_csv(output / "cross_split_comparison.csv", index=False)
    for mode in MODES:
        directory = output / mode
        directory.mkdir(exist_ok=True)
        for scope in SCOPES:
            p = pooled.loc[pooled.split_mode.eq(mode) & pooled.Scope.eq(scope)]
            p.drop(columns=["split_mode", "Scope"]).to_csv(directory / f"{scope.lower()}_diagnostics.csv", index=False)
    for name, obj in [("selection.json", selection), ("verification.json", audit)]:
        (output / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    overall = macro.loc[macro.Scope.eq("Overall")].pivot(index="candidate", columns="split_mode", values="mean_mae")
    overall = overall.rename(columns={"random_disjoint": "随机划分mean MAE", "chronological_blocked": "时间划分mean MAE"})
    overall = overall.sort_values("随机划分mean MAE").reset_index()
    overall.insert(1, "说明", overall.candidate.map(NAMES))
    narrative = """# 紧凑个人状态模型：完整实验结果

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

"""
    parts = [narrative, table(overall)]
    parts += ["""

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
"""]
    for mode in MODES:
        for scope in SCOPES:
            m = macro.loc[macro.split_mode.eq(mode) & macro.Scope.eq(scope)]
            p = pooled.loc[pooled.split_mode.eq(mode) & pooled.Scope.eq(scope)]
            parts += [f"\n### {mode} — {scope}\n", "\nParticipant-macro MAE（mmHg）：\n",
                      table(m[["candidate", "n_participants", "n_events", "sbp_mae", "dbp_mae", "mean_mae"]]),
                      "\n窗口汇总诊断；≤5/10/15的数值单位为百分比：\n",
                      table(p.drop(columns=["split_mode", "Scope"]))]
    parts += ["\n## 完成证据与复现\n\n训练任务1398–1405、1407–1414；报告1406、1415、1416。\n"
              "训练代码版本9bc4aad。已重新读取16份验证预测，逐一检查窗口与标签完全一致，\n"
              "并重算三个范围的MAE、R²、ME、STD及误差范围比例，全部匹配报告。\n"
              "运行元数据、预测、个人映射/档案与诊断文件均与NAS归档字节一致；三份报告目录亦一致。\n"
              "本次没有重新散列checkpoint二进制文件。完整检查项和报告SHA-256见\n"
              "[verification.json](../results/same_subject_personal_profiles/verification.json)。\n\n"
              "[全部聚合结果](../results/same_subject_personal_profiles/)包含48行participant-macro记录、\n"
              "96行窗口诊断以及六份按划分/来源拆开的表格。原始波形、个人数据和checkpoint不公开。\n"]
    (repo / "docs/RESULTS_SAME_SUBJECT_PERSONAL_PROFILES.md").write_text("\n".join(parts), encoding="utf-8")
    (output / "README.md").write_text(
        "# Compact personal-profile results\n\n"
        "[完整结果与解释](../../docs/RESULTS_SAME_SUBJECT_PERSONAL_PROFILES.md)\n\n"
        "Eight candidates, two same-subject development splits, seed 20260904. "
        "LoRA remains the accuracy reference; no profile passes the upgrade gate.\n\n"
        "`participant_macro.csv`: 48 rows; `event_pooled_diagnostics.csv`: 96 rows. "
        "Each split subdirectory contains Overall/MIMIC/VitalDB diagnostic tables. "
        "AAMI/BHS fields are retrospective numerical screens, not clinical compliance. "
        "`verification.json` records aggregate audit evidence; held-out remains sealed.\n",
        encoding="utf-8",
    )
    for file in output.rglob("*"):
        if file.is_file():
            text = file.read_text(encoding="utf-8")
            assert "/home/" not in text and "C:\\" not in text
            assert "subject_uid" not in text or "tie-break" in text
    print(json.dumps({"status": "pass", "participant_rows": len(macro), "diagnostic_rows": len(pooled), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.repo)
