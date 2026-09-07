"""Validate aggregate-only personal-memory reports and render a public summary.

No network, prediction-level data, model fitting, or held-out access is used.
The server must first generate the repaired final report from the unchanged
completed mode reports. This script checks those aggregates; it does not claim
to recompute patient-level predictions. Requires NumPy and pandas only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


SCREEN = "personal-memory-v1"
SEED = 20260907
MODES = ("random_disjoint", "chronological_blocked")
SCOPES = {"Overall": (2051, 82040), "MIMIC": (1011, 40440), "VitalDB": (1040, 41600)}
FROZEN = "D0_frozen_lora"
CONTROL = "lora_continued_control"
NEURAL = ("pair_single", "pair_uniform", "pair_retrieved", "pair_distance_blend")
CANDIDATES = (FROZEN, CONTROL, *NEURAL)
DIAGNOSTICS = ["Scope", "Setting", "BP", "MAE", "R²", "ME", "STD", "≤5 mmHg", "≤10 mmHg", "≤15 mmHg", "AAMI", "BHS"]
MACRO = ["candidate", "view", "split_mode", "seed", "n_participants", "n_events", "sbp_mae", "dbp_mae", "mean_mae", "sbp_bias", "dbp_bias"]
METRICS = ("sbp_mae", "dbp_mae", "mean_mae", "sbp_bias", "dbp_bias")
KEYS = ["candidate", "split_mode", "view"]
FORBIDDEN = re.compile(r"/home/|/srv/|[A-Za-z]:[/\\]|(?:subject_uid|subject_id|event_id|target_sbp|target_dbp|pred_sbp|pred_dbp|password|private_key|access_token)\b", re.I)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same(actual, expected, *, tolerance=2e-5) -> None:
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)


def qualified_standards(row: pd.Series) -> tuple[str, str]:
    """Recheck the report's historical numerical screen, not device compliance."""
    aami = "PASS*" if abs(float(row["ME"])) <= 5 and float(row["STD"]) <= 8 else "FAIL*"
    percentages = row[["≤5 mmHg", "≤10 mmHg", "≤15 mmHg"]].to_numpy(float)
    grade = next((g for g, threshold in (("A", [60, 85, 95]), ("B", [50, 75, 90]),
                                       ("C", [40, 65, 85]))
                  if np.all(percentages >= threshold)), "D")
    return aami, f"{'PASS' if grade in {'A', 'B'} else 'FAIL'} (Grade {grade})*"


def validate_selection(selection: dict, mode: str | None = None) -> None:
    require(selection.get("status") == "complete", "report is not complete")
    require(selection.get("screen_id") == SCREEN and selection.get("seed") == SEED, "screen/seed mismatch")
    require(selection.get("heldout_test_accessed") is False, "held-out access must be explicitly false")
    require(selection.get("official_pulsedb_calbased_reproduction") is False, "incorrect official benchmark scope")
    require(bool(re.fullmatch("[0-9a-f]{64}", selection.get("gate_manifest_sha256", ""))), "missing gate identity")
    if mode is not None:
        require(selection.get("split_mode") == mode, "selection split mismatch")
        require(selection.get("complete_query_coverage") is True, "filtered/incomplete query cohort")
        require(selection.get("selection_role") == "internal_validation", "wrong checkpoint selection role")
        runs = selection.get("runs", [])
        require(len(runs) == 5 and {run.get("candidate") for run in runs} == {CONTROL, *NEURAL}, "five completed fits required")
        require(all(run.get("status") == "complete" for run in runs), "failed/skipped fit included")
        for run in runs:
            require(bool(re.fullmatch("[0-9a-f]{64}", run.get("checkpoint_sha256", ""))), "missing checkpoint identity")


def check_comparison(comparison: pd.DataFrame, primary: pd.DataFrame, reference: str) -> None:
    require(len(comparison) == len(primary) and not comparison.duplicated(KEYS).any(), "comparison coverage mismatch")
    merged = primary.merge(comparison, on=KEYS, suffixes=("_source", "_comparison"), validate="one_to_one")
    require(len(merged) == len(primary), "comparison keys changed")
    for column in (*METRICS, "n_participants", "n_events", "seed"):
        same(merged[f"{column}_source"], merged[f"{column}_comparison"], tolerance=1e-6)
    reference_rows = primary.loc[primary["candidate"].eq(reference)].set_index("view")
    require(comparison["reference_candidate"].eq(reference).all(), "reference changed across sources")
    for _, row in comparison.iterrows():
        expected = float(reference_rows.loc[row["view"], "mean_mae"])
        same(row["reference_mean_mae"], expected, tolerance=1e-6)
        same(row["gain_mmhg"], expected - row["mean_mae"], tolerance=1e-6)


def validate(macro: dict, pooled: dict, comparisons: dict, mode_selections: dict,
             combined: pd.DataFrame, gate: pd.DataFrame, final_selection: dict) -> dict:
    references = {}
    for mode in MODES:
        selection = mode_selections[mode]
        validate_selection(selection, mode)
        primary, diagnostics = macro[mode], pooled[mode]
        require(list(primary.columns) == MACRO, "unexpected/private participant-macro columns")
        require(len(primary) == 18 and not primary.duplicated(["candidate", "view"]).any(), "missing/repeated macro rows")
        require(set(primary["candidate"]) == set(CANDIDATES) and set(primary["view"]) == set(SCOPES), "candidate/scope mismatch")
        require(primary["split_mode"].eq(mode).all() and primary["seed"].eq(SEED).all(), "mixed mode/seed")
        require(np.isfinite(primary[list(METRICS)].to_numpy(float)).all(), "non-finite macro metric")
        require((primary[["sbp_mae", "dbp_mae", "mean_mae"]] >= 0).all().all(), "negative MAE")
        require(list(diagnostics.columns) == DIAGNOSTICS and len(diagnostics) == 36, "diagnostic schema/size mismatch")
        require(not diagnostics.duplicated(["Scope", "Setting", "BP"]).any(), "repeated diagnostic rows")
        require(set(diagnostics["Scope"]) == set(SCOPES) and set(diagnostics["Setting"]) == set(CANDIDATES), "diagnostic candidate/scope mismatch")
        for candidate in CANDIDATES:
            rows = primary.loc[primary["candidate"].eq(candidate)].set_index("view")
            for scope, counts in SCOPES.items():
                row = rows.loc[scope]
                require(tuple(row[["n_participants", "n_events"]].to_numpy(float)) == counts, "full fixed cohort not retained")
                same(row["mean_mae"], (row["sbp_mae"] + row["dbp_mae"]) / 2)
                diagnostic = diagnostics.loc[diagnostics["Setting"].eq(candidate) & diagnostics["Scope"].eq(scope)].set_index("BP")
                require(set(diagnostic.index) == {"SBP", "DBP"}, "missing BP target")
                same(diagnostic.loc[["SBP", "DBP"], "MAE"].to_numpy(float), [row["sbp_mae"], row["dbp_mae"]])
                same(diagnostic.loc[["SBP", "DBP"], "ME"].to_numpy(float), [row["sbp_bias"], row["dbp_bias"]])
                numeric = diagnostic[["MAE", "R²", "ME", "STD", "≤5 mmHg", "≤10 mmHg", "≤15 mmHg"]].to_numpy(float)
                require(np.isfinite(numeric).all(), "non-finite pooled metric")
                require(diagnostic["STD"].ge(0).all() and diagnostic["R²"].le(1 + 1e-10).all(), "invalid STD/R-squared")
                percentages = diagnostic[["≤5 mmHg", "≤10 mmHg", "≤15 mmHg"]].to_numpy(float)
                require(np.all((percentages >= 0) & (percentages <= 100)) and np.all(np.diff(percentages, axis=1) >= 0), "invalid cumulative percentages")
                for _, metric_row in diagnostic.iterrows():
                    require(tuple(metric_row[["AAMI", "BHS"]]) == qualified_standards(metric_row), "AAMI/BHS numerical labels disagree with values")
                    require(abs(metric_row["ME"]) <= metric_row["MAE"] + 1e-5, "mean signed error exceeds MAE")
            for column in ("n_participants", "n_events"):
                require(rows.loc[["MIMIC", "VitalDB"], column].sum() == rows.loc["Overall", column], "source counts do not sum to Overall")
            # This is a consistency check, not construction of Overall results.
            # Render the original full-cohort row; never average sources equally.
            for metric in METRICS:
                expected = np.average(rows.loc[["MIMIC", "VitalDB"], metric], weights=rows.loc[["MIMIC", "VitalDB"], "n_participants"])
                same(rows.loc["Overall", metric], expected)
        options = primary.loc[primary["candidate"].isin((FROZEN, CONTROL)) & primary["view"].eq("Overall")]
        reference = str(options.sort_values(["mean_mae", "candidate"], kind="mergesort").iloc[0]["candidate"])
        require(selection.get("reference") == reference, "paired reference is not stronger Overall frozen/continued LoRA")
        references[mode] = reference
        check_comparison(comparisons[mode], primary, reference)
    validate_selection(final_selection)
    require(final_selection.get("selection_is_development_only") is True, "final result must remain development-only")
    require(final_selection.get("references") == references, "final reference mapping changed")
    require(len({final_selection["gate_manifest_sha256"], *(x["gate_manifest_sha256"] for x in mode_selections.values())}) == 1, "gate identity differs across reports")
    require(len(combined) == 36 and not combined.duplicated(KEYS).any(), "final comparison coverage mismatch")
    for mode in MODES:
        check_comparison(combined.loc[combined["split_mode"].eq(mode)], macro[mode], references[mode])
    require(set(gate["candidate"]) == set(NEURAL) and len(gate) == 4, "joint gate candidate mismatch")
    eligible = []
    for candidate in NEURAL:
        rows = combined.loc[combined["candidate"].eq(candidate)]
        overall = rows.loc[rows["view"].eq("Overall")]
        sources = rows.loc[~rows["view"].eq("Overall")]
        passed = bool(overall["gain_mmhg"].ge(.15).all() and sources["gain_mmhg"].gt(0).all())
        reported = gate.loc[gate["candidate"].eq(candidate)].iloc[0]
        require(str(reported["passes_accuracy_gate"]).lower() in {"true", "false"}, "invalid gate boolean")
        require((str(reported["passes_accuracy_gate"]).lower() == "true") == passed, "joint promotion gate is wrong")
        same(reported["mean_across_modes"], overall["mean_mae"].mean(), tolerance=1e-6)
        if passed:
            eligible.append(candidate)
    require(set(final_selection.get("eligible_candidates", [])) == set(eligible), "eligible candidate list differs from paired evidence")
    return {"status": "pass", "screen_id": SCREEN, "settings_per_mode": 6, "completed_model_fits": 10,
            "participant_macro_rows": 36, "pooled_diagnostic_rows": 72, "fixed_full_cohort_checked": True,
            "metric_and_qualified_standard_consistency_checked": True, "paired_reference_and_joint_gate_recomputed": True,
            "references": references, "eligible_candidates": eligible, "heldout_test_accessed": False,
            "raw_prediction_recomputation": False,
            "scope": "Aggregate-only cross-check of completed server reports; no private predictions or signals read."}


def table(frame: pd.DataFrame) -> str:
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("*", "\\*")
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                      "| " + " | ".join("---" for _ in frame.columns) + " |",
                      *("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))])


def render(macro, pooled, combined, final_selection, *, execution_evidence=False) -> str:
    meanings = {FROZEN: "原冻结个人 LoRA；无需新增训练", CONTROL: "同个人标签预算，继续训练 LoRA 的强对照",
                "pair_single": "单个相似历史参考 + 关系修正", "pair_uniform": "多个参考等权 + 关系修正",
                "pair_retrieved": "多个相似参考按特征距离加权 + 关系修正",
                "pair_distance_blend": "按 PPG 距离融合参考记忆预测与冻结 LoRA"}
    eligible = final_selection["eligible_candidates"]
    lines = ["# 个人历史记忆增强：完整开发结果", "", "更新：2026-09-07。协议：`development-calbased-analogue-v1`；种子：20260907。", "",
             "## 结论", "",
             "本批 10 项模型训练和两份分划分报告已完成。最后跨划分汇总的 `Series.view` 列名冲突已修复；只重新生成汇总，不重训、不更改预测或数据划分。",
             "在六个主表设置中，`pair_distance_blend` 是两种划分的数值最佳候选。随机划分改善较大，时间划分改善较小；两个内部来源的 SBP 和 DBP 均有改善。额外时间邻近诊断 D3 在随机划分更低，见后文，不应把主表第一名称为全部诊断方法的第一名。",
             ("通过联合门槛的开发候选：" + ", ".join(eligible) + "。仍需独立确认。") if eligible else
             "但没有候选达到两种划分均改善至少 0.15 mmHg、且两个来源均改善的预设联合门槛，因此不正式替换配对 LoRA 主参照。",
             "单种子、多轮内部验证探索尚不支持统计显著、独立测试优越性或临床有效性结论。", "",
             "## 实验范围与信息预算", "",
             "- 只使用父级 meta-train 中 2,051 名已注册受试者：MIMIC 1,011 人、VitalDB 1,040 人。两者是 PulseDB 内部分来源分层，不是独立外部验证。",
             "- 每人 320 个带 BP 标签的 10 秒训练窗口、40 个内部验证窗口、40 个仍封存窗口。每种划分保留全部 82,040 个验证窗口，未剔除最差 30% 或其他困难窗口。",
             "- 个人 LoRA 状态与参考记忆只来自允许的训练数据。检索五个参考不等于仅使用五个标签，也不是新用户 K=5 或五次真实袖带校准。",
             "- 训练与验证身份有意重合，具体窗口/生理区间/重复波形仍按既有审计分离。随机划分允许较晚训练参考，属于已注册用户数据内插值，不是严格前瞻预测。",
             "- 时间划分只使用当前查询之前且满足同记录时间约束的参考；无足够合格参考时回退基础预测，保留查询计入结果。",
             "- 内部验证用于 checkpoint 选择，不能更新个人记忆或进入预测输入；held-out 与独立 meta-test 仍封存。当前并非官方 PulseDB CalBased 固定划分复现。",
             "- 比较参照按每种划分 Overall mean MAE 选择冻结/继续训练 LoRA 中较强者，然后固定该参照用于两个来源，禁止按来源分别挑选。时间划分两参照仅有浮点级差别。", "",
             "## 六个设置与总体比较", ""]
    summary = []
    for candidate in CANDIDATES:
        row = {"Setting": candidate, "含义": meanings[candidate]}
        for mode, label in zip(MODES, ("Random mean MAE", "Chronological mean MAE")):
            row[label] = float(macro[mode].loc[macro[mode]["candidate"].eq(candidate) & macro[mode]["view"].eq("Overall"), "mean_mae"].iloc[0])
        summary.append(row)
    lines += [table(pd.DataFrame(summary)), "", "单位 mmHg。mean MAE=(SBP MAE+DBP MAE)/2，受试者等权；越低越好。六个设置包含冻结参照，实际新增模型拟合为每种划分五项，共十项。", "",
              "### 最佳融合与正确配对参照", ""]
    blend = combined.loc[combined["candidate"].eq("pair_distance_blend"), ["split_mode", "view", "reference_candidate", "sbp_mae", "dbp_mae", "mean_mae", "reference_mean_mae", "gain_mmhg"]]
    lines += [table(blend), "", "gain_mmhg=配对参照 mean MAE−候选 mean MAE，正数表示改善。", "",
              "## 全部受试者等权结果", ""]
    for mode in MODES:
        lines += [f"### {mode}", "", table(macro[mode][["candidate", "view", "n_participants", "n_events", "sbp_mae", "dbp_mae", "mean_mae"]]), ""]
    lines += ["## 完整诊断指标表", "",
              "下表使用同一批完整验证预测。MAE、ME、STD 单位均为 mmHg；ME=预测−参考，STD 是逐窗口误差的样本标准差，不是多种子 MAE 的标准差。阈值列是累计百分比。",
              "每人均保留 40 个窗口，因此逐窗口 MAE 与受试者等权 MAE 仅有舍入差；R²、STD、阈值比例仍按逐窗口定义。Overall 为原报告对全体数据重算，未将 MIMIC/VitalDB 简单平均。",
              "`AAMI*` 仅是 |ME|≤5、误差 STD≤8 的回顾性数值筛查，不包含正式标准要求的完整试验设计。`BHS*` 按三档比例给出 A/B/C/D，A/B 显示 PASS。二者均不代表设备认证或临床标准验证通过。", ""]
    for mode in MODES:
        for scope in SCOPES:
            lines += [f"### {mode} · {scope}", "", table(pooled[mode].loc[pooled[mode]["Scope"].eq(scope), DIAGNOSTICS[1:]]), ""]
    lines += ["## 为什么部分任务只需几十秒", "",
              "四种关系模块复用冻结的 256 维 LoRA 特征，只有 65,730 个共享可训练参数，并在 GPU 中批量读取缓存；不是从头训练完整原始 PPG 主干。关系模块每项约 38 秒至 2 分 12 秒；两个继续训练 LoRA 对照约 57 分 34 秒和 30 分 57 秒。",
              "随机融合实际完成八轮、6,256 次优化更新，但最佳 checkpoint 为 epoch 0：关系修正初始为零，其优势来自历史参考检索与固定距离融合，而非新增关系网络训练。",
              "时间融合最佳为 epoch 1：相对自身初始 mean MAE 3.796670 降到 3.657139，说明有学习收益；但相对更强 LoRA 参照的优势仍只有约 0.0552 mmHg。不能把自身初始化的下降全部归为超越 LoRA。", "",
              "## 机制诊断与时间邻近性", "",
              "诊断 D3 使用合法训练记录中时间最近的 BP，在随机划分 mean MAE 为 2.285079，优于当前融合候选。这提示短时血压延续性与时间邻近性可能贡献较大，不能只将随机划分增益解释为新的波形生理映射。D3 使用记录时间，并非纯 PPG 部署输入。",
              "随机划分简单特征邻居 BP 检索 D1 的 mean MAE 为 2.697324；排除查询前后 60 秒内参考后变为 3.158958，冻结 LoRA 保持 2.880098。该检查支持近时间参考敏感性，不自动等于违反当前窗口划分规则。",
              "60 秒实验针对冻结诊断方法，不是完整融合候选的同条件测试，不能据此断言完整融合在间隔约束下必然失败。训练编码器已见过训练标签，块排除检索属于普通监督训练，不能称为编码器级 OOF。", "",
              "## 决策与论文边界", "",
              "保留个人记忆作为有信号的候选方向，但尚不能以新关系网络为已验证核心贡献。优先检验长期/时间间隔条件下记忆是否仍有独立价值，并保留 LoRA、时间邻近 BP、简单检索和固定融合对照。",
              "相关下一步应先写明有限假设、信息预算、时间约束、选择规则和失败退出条件，再执行确认；不能通过删除困难受试者或放宽既定门槛把当前结果变成成功。",
              "多种子与受试者层配对不确定性、机制对照、封存评估以及真实条件鲁棒性仍待完成；当前结果不保证可发表性。", "",
              "## 可复核文件", "",
              "- [跨划分比较](../results/personal_memory_v1/final/cross_split_comparison.csv)、[联合升级门槛](../results/personal_memory_v1/final/promotion_gate.csv)、[最终选择](../results/personal_memory_v1/final/selection.json)。",
              "- [随机受试者等权表](../results/personal_memory_v1/random_disjoint/participant_macro_summary.csv)、[时间受试者等权表](../results/personal_memory_v1/chronological_blocked/participant_macro_summary.csv)。",
              "- [随机完整指标](../results/personal_memory_v1/random_disjoint/event_pooled_diagnostics_all_scopes.csv)、[时间完整指标](../results/personal_memory_v1/chronological_blocked/event_pooled_diagnostics_all_scopes.csv)。",
              "- [发布核查](../results/personal_memory_v1/publication_audit.json)记录聚合一致性、数据来源文件散列和联合门槛复算。",
              "", "公开内容仅含聚合数据、方法与核查记录，不含原始波形、个人标识、逐窗口预测、个人权重、服务器路径或组会幻灯片。发布核查不等于重新推理或重新拟合。", ""]
    if execution_evidence:
        lines += ["[执行与机制诊断证据](../results/personal_memory_v1/execution_evidence.json)记录修复范围、运行状态及相关诊断来源。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--render-only", action="store_true", help="Accepted for consistency; this publisher is always offline.")
    parser.add_argument("--check-only", action="store_true", help="Validate without writing generated documents.")
    args = parser.parse_args()
    directory = args.root / "results" / "personal_memory_v1"
    provenance = []

    def read(relative: str):
        path = directory / relative
        payload = path.read_bytes()
        text = payload.decode("utf-8-sig")
        require(FORBIDDEN.search(text) is None, "private fields/absolute paths in public input: " + relative)
        provenance.append({"relative_path": relative, "sha256": sha256(payload).hexdigest()})
        return json.loads(text) if path.suffix == ".json" else pd.read_csv(path)

    macro, pooled, comparisons, selections = {}, {}, {}, {}
    for mode in MODES:
        macro[mode] = read(f"{mode}/participant_macro_summary.csv")
        pooled[mode] = read(f"{mode}/event_pooled_diagnostics_all_scopes.csv")
        comparisons[mode] = read(f"{mode}/comparison_vs_reference.csv")
        selections[mode] = read(f"{mode}/selection.json")
    combined = read("final/cross_split_comparison.csv")
    gate = read("final/promotion_gate.csv")
    final_selection = read("final/selection.json")
    evidence_exists = (directory / "execution_evidence.json").is_file()
    if evidence_exists:
        read("execution_evidence.json")
    audit = validate(macro, pooled, comparisons, selections, combined, gate, final_selection)
    audit.update({"checked_utc": datetime.now(timezone.utc).isoformat(), "source_aggregate_files": provenance})
    if not args.check_only:
        content = render(macro, pooled, combined, final_selection, execution_evidence=evidence_exists)
        (args.root / "docs").mkdir(parents=True, exist_ok=True)
        (args.root / "docs" / "RESULTS_PERSONAL_MEMORY_V1.md").write_text(content, encoding="utf-8")
        (directory / "publication_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
