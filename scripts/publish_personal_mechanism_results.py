"""Publish only aggregates from the four verified frozen diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def table(f):
    def cell(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(f.columns) + " |",
        "| " + " | ".join(["---"] * len(f.columns)) + " |",
        *["| " + " | ".join(cell(v) for v in row) + " |" for row in f.itertuples(index=False, name=None)]])


def build(source, repo):
    macros, pooleds, evidence = [], [], []
    for job in [1474, 1475, 1476, 1477]:
        folder = source / str(job)
        d = json.loads((folder / "diagnostic.json").read_text())
        assert d["status"] == "complete" and d["heldout_test_accessed"] is False
        assert d["parameters_updated"] is False
        assert d["natural_max_absolute_difference"] == d["cache_max_absolute_difference"] == 0
        m = pd.read_csv(folder / "participant_macro.csv")
        p = pd.read_csv(folder / "pooled_diagnostics.csv")
        assert len(m) == 21 and len(p) == 42
        assert not m.duplicated(["candidate", "split_mode", "condition", "Scope"]).any()
        assert not p.duplicated(["candidate", "split_mode", "Setting", "Scope", "BP"]).any()
        assert set(m.Scope) == {"Overall", "MIMIC", "VitalDB"}
        assert set(p.Scope) == set(m.Scope)
        assert m.loc[m.Scope.eq("Overall"), "n_participants"].eq(2051).all()
        assert m.loc[m.Scope.eq("Overall"), "n_events"].eq(82040).all()
        macros.append(m)
        pooleds.append(p)
        evidence.append({"job_id": job, **d, "aggregate_sha256": {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ["participant_macro.csv", "pooled_diagnostics.csv"]}})
    macro = pd.concat(macros, ignore_index=True)
    pooled = pd.concat(pooleds, ignore_index=True)
    output = repo / "results/personal_mechanisms"
    output.mkdir(parents=True, exist_ok=True)
    macro.to_csv(output / "participant_macro.csv", index=False)
    pooled.to_csv(output / "event_pooled_diagnostics.csv", index=False)
    (output / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for mode in ["random_disjoint", "chronological_blocked"]:
        folder = output / mode
        folder.mkdir(exist_ok=True)
        for scope in ["Overall", "MIMIC", "VitalDB"]:
            f = pooled.loc[pooled.split_mode.eq(mode) & pooled.Scope.eq(scope)].copy()
            f["Setting"] = f.candidate + ":" + f.Setting
            f.drop(columns=["candidate", "split_mode", "Scope"]).to_csv(folder / f"{scope.lower()}_diagnostics.csv", index=False)
    overall = macro.loc[macro.Scope.eq("Overall")].pivot(index=["candidate", "condition"], columns="split_mode", values="mean_mae").reset_index()
    overall.columns.name = None
    text = """# LoRA个人化机制：冻结模型诊断

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

"""
    text += table(overall)
    text += """

## 分来源结果与完整指标

以下保留每种来源和划分的同一批预测，不剔除困难人群。完整标准列
Setting、BP、MAE、R²、ME、STD、≤5/10/15 mmHg百分比、AAMI、BHS见：

- [随机 Overall](../results/personal_mechanisms/random_disjoint/overall_diagnostics.csv)、[MIMIC](../results/personal_mechanisms/random_disjoint/mimic_diagnostics.csv)、[VitalDB](../results/personal_mechanisms/random_disjoint/vitaldb_diagnostics.csv)。
- [时间 Overall](../results/personal_mechanisms/chronological_blocked/overall_diagnostics.csv)、[MIMIC](../results/personal_mechanisms/chronological_blocked/mimic_diagnostics.csv)、[VitalDB](../results/personal_mechanisms/chronological_blocked/vitaldb_diagnostics.csv)。

STD是逐窗口预测误差的样本标准差，不是MAE跨人的标准差。AAMI/BHS带星号仅表示
离线数值筛查，不是临床标准验证通过。主比较是participant-macro指标；这里每人相同
40个窗口，因此MAE与逐窗口汇总相符，R²/STD仍应按对应定义解释。

"""
    bysource = macro.loc[~macro.Scope.eq("Overall"), ["candidate", "split_mode", "condition", "Scope", "sbp_mae", "dbp_mae", "mean_mae"]]
    text += table(bysource)
    text += """

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
"""
    (repo / "docs/RESULTS_PERSONAL_MECHANISMS.md").write_text(text, encoding="utf-8")
    print(json.dumps({"macro_rows": len(macro), "pooled_rows": len(pooled), "diagnostics": len(evidence)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    args = p.parse_args()
    build(args.source, args.repo)
