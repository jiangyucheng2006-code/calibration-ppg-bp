"""Publish only audited aggregate results for two completed personal studies.

Requires NumPy/pandas and the system SSH client; never reads patient-level
predictions, waveforms or checkpoints. Server locations are CLI arguments and
are not embedded in public artifacts. --render-only validates the published
aggregate tables and regenerates the two formal reports without any network.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

import numpy as np
import pandas as pd


MODES = ("random_disjoint", "chronological_blocked")
SCOPES = {"Overall": (2051, 82040), "MIMIC": (1011, 40440), "VitalDB": (1040, 41600)}
FEATURE_NAMES = ["subject_lora_rank4", "shared_lora_rank4", "subject_lora_rank1", "output_profile32",
                 "feature_affine32", "shared_bilinear32", "shared_bilinear64", "subject_nonlinear_rank4"]
PRS_NAMES = ["lora_continue", "lora_prs_bias", "lora_prs_dynamic", "lora_prs_dynamic_shrink", "lora_prs_dynamic_frozen"]
SCREENS = {
    "personal_feature_mechanisms": {"id": "personal-feature-mechanisms-v1", "seed": 20260906,
        "names": FEATURE_NAMES, "reference": "subject_lora_rank4", "random": 1566, "chrono": 1497, "final": 1567,
        "document": "RESULTS_PERSONAL_FEATURE_MECHANISMS.md"},
    "lora_prs_continuation": {"id": "lora-prs-continuation-v1", "seed": 20260907,
        "names": PRS_NAMES, "reference": "lora_continue", "random": 1505, "chrono": 1512, "final": 1513,
        "document": "RESULTS_LORA_PRS_CONTINUATION.md"},
}
DIAG_COLUMNS = ["Scope", "Setting", "BP", "MAE", "R²", "ME", "STD", "≤5 mmHg", "≤10 mmHg", "≤15 mmHg", "AAMI", "BHS"]
MACRO_COLUMNS = ["candidate", "backbone", "split_mode", "seed", "view", "n_participants", "n_events",
                 "sbp_mae", "dbp_mae", "mean_mae", "sbp_bias", "dbp_bias", "run_id"]


def fetch(host: str, root: str, relative: str) -> bytes:
    path = str(PurePosixPath(root) / relative)
    return subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                                    "cat " + shlex.quote(path)], timeout=60)


def sanitize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "run_dir" in frame:
        frame["run_id"] = frame.pop("run_dir").map(lambda x: PurePosixPath(str(x)).name)
    return frame.drop(columns=["worst_30_mean_mae", "retained_70_mean_mae", "runner"], errors="ignore")


def csv_bytes(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(StringIO(payload.decode("utf-8-sig")))


def validate(spec, macro, diagnostics, comparison, gate, selection):
    names = set(spec["names"])
    if selection.get("status") != "complete" or selection.get("heldout_test_accessed") is not False:
        raise ValueError("final selection is incomplete or held-out access is not explicitly false")
    if selection.get("official_pulsedb_calbased_reproduction") is not False:
        raise ValueError("incorrect official-benchmark scope")
    if selection.get("screen_id") != spec["id"] or selection.get("reference") != spec["reference"]:
        raise ValueError("final selection screen/reference mismatch")
    if selection.get("seed") != spec["seed"]:
        raise ValueError("selection seed mismatch")
    for mode in MODES:
        primary, pooled = macro[mode], diagnostics[mode]
        if len(primary) != len(names) * 3 or primary.duplicated(["candidate", "view"]).any():
            raise ValueError("missing or repeated participant-macro rows")
        if set(primary.candidate) != names or set(primary.view) != set(SCOPES):
            raise ValueError("candidate or source coverage mismatch")
        if not primary.split_mode.eq(mode).all() or not primary.seed.eq(spec["seed"]).all():
            raise ValueError("mixed split or seed")
        if list(pooled.columns) != DIAG_COLUMNS or len(pooled) != len(names) * 6:
            raise ValueError("diagnostic table schema/size mismatch")
        if pooled.duplicated(["Scope", "Setting", "BP"]).any():
            raise ValueError("repeated diagnostics")
        for candidate in spec["names"]:
            rows = primary.loc[primary.candidate.eq(candidate)].set_index("view")
            for scope, counts in SCOPES.items():
                row = rows.loc[scope]
                if (int(row.n_participants), int(row.n_events)) != counts:
                    raise ValueError("incomplete cohort or filtered validation queries")
                np.testing.assert_allclose(row.mean_mae, (row.sbp_mae + row.dbp_mae) / 2, atol=1e-5, rtol=0)
                diagnostic = pooled.loc[pooled.Setting.eq(candidate) & pooled.Scope.eq(scope)].set_index("BP")
                if set(diagnostic.index) != {"SBP", "DBP"}:
                    raise ValueError("missing BP target")
                np.testing.assert_allclose(diagnostic.loc[["SBP", "DBP"], "MAE"].to_numpy(float),
                                           [row.sbp_mae, row.dbp_mae], atol=2e-5, rtol=0)
                numerics = diagnostic[["MAE", "R²", "ME", "STD", "≤5 mmHg", "≤10 mmHg", "≤15 mmHg"]].to_numpy(float)
                if not np.isfinite(numerics).all():
                    raise ValueError("non-finite diagnostic")
                percentages = diagnostic[["≤5 mmHg", "≤10 mmHg", "≤15 mmHg"]].to_numpy(float)
                if (percentages < 0).any() or (percentages > 100).any() or (np.diff(percentages, axis=1) < 0).any():
                    raise ValueError("invalid cumulative percentages")
                for column in ("AAMI", "BHS"):
                    if not diagnostic[column].astype(str).str.endswith("*").all():
                        raise ValueError("numerical screens must be qualified, not clinical certification")
            if rows.loc[["MIMIC", "VitalDB"], "n_participants"].sum() != rows.loc["Overall", "n_participants"]:
                raise ValueError("source participants do not sum to Overall")
            if rows.loc[["MIMIC", "VitalDB"], "n_events"].sum() != rows.loc["Overall", "n_events"]:
                raise ValueError("source windows do not sum to Overall")
            # Validation consistency only: publication retains the original
            # recomputed Overall row, not a newly averaged source result.
            for column in ("sbp_mae", "dbp_mae", "mean_mae"):
                weighted = np.average(rows.loc[["MIMIC", "VitalDB"], column],
                                      weights=rows.loc[["MIMIC", "VitalDB"], "n_participants"])
                np.testing.assert_allclose(rows.loc["Overall", column], weighted, atol=2e-5, rtol=0)
    all_macro = pd.concat(list(macro.values()), ignore_index=True)
    keys = ["candidate", "split_mode", "view"]
    if len(comparison) != len(all_macro) or comparison.duplicated(keys).any():
        raise ValueError("cross-split comparison coverage mismatch")
    merged = all_macro.merge(comparison, on=keys, suffixes=("_split", "_final"), validate="one_to_one")
    for metric in ("sbp_mae", "dbp_mae", "mean_mae"):
        np.testing.assert_allclose(merged[f"{metric}_split"], merged[f"{metric}_final"], atol=1e-6, rtol=0)
    gain_column = "gain_vs_continued" if spec["id"].startswith("lora-prs") else "gain_mmhg"
    expected_pass = {}
    for candidate in spec["names"]:
        data = comparison.loc[comparison.candidate.eq(candidate)]
        for _, row in data.iterrows():
            reference = all_macro.loc[all_macro.candidate.eq(spec["reference"]) &
                                      all_macro.split_mode.eq(row.split_mode) & all_macro.view.eq(row.view)].iloc[0]
            np.testing.assert_allclose(row[gain_column], reference.mean_mae - row.mean_mae, atol=1e-6, rtol=0)
        expected_pass[candidate] = (candidate != spec["reference"] and
            data.loc[data.view.eq("Overall"), gain_column].ge(0.15).all() and
            data.loc[~data.view.eq("Overall"), gain_column].gt(0).all())
    if set(gate.candidate) != names or gate.candidate.duplicated().any():
        raise ValueError("promotion gate candidate mismatch")
    for _, row in gate.iterrows():
        actual = str(row.passes_accuracy_gate).lower() == "true"
        if actual != bool(expected_pass[row.candidate]):
            raise ValueError("promotion gate differs from recalculation")
    if set(selection["eligible_candidates"]) != {k for k, v in expected_pass.items() if v}:
        raise ValueError("final eligible candidate set differs from aggregate evidence")
    return {"candidate_count": len(names), "split_count": 2,
            "participant_macro_rows": len(all_macro), "diagnostic_rows": sum(map(len, diagnostics.values())),
            "fixed_cohort_and_counts": True, "split_final_consistency": True,
            "paired_gate_recomputed": True, "heldout_test_accessed": False,
            "raw_prediction_recomputation": False,
            "claim_limit": "Existing aggregate reports cross-checked; patient-level source predictions not downloaded."}


def table(frame: pd.DataFrame, precision=4) -> str:
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{precision}f}"
        return str(value).replace("|", "\\|").replace("*", "\\*")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    return "\n".join([header, divider] + ["| " + " | ".join(cell(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)])


def render(key, spec, macro, diagnostics, comparison, selection):
    feature = key == "personal_feature_mechanisms"
    title = "个人特征适配机制：完整结果" if feature else "LoRA + PRS 继续训练：完整结果"
    lines = [f"# {title}", "", "更新：2026-09-07。本文为正式开发结果汇总，不是封存测试集结果。", "", "## 结论", ""]
    if feature:
        lines += ["8 个候选 × 2 种划分全部完成。没有替代方法超过同期 rank-4 个人 LoRA 并满足预设升级门槛，保留 `subject_lora_rank4`。",
                  "共享适配、压缩个人参数、非线性响应都未产生确认性提升；不能将模型改名或更复杂视为创新有效。", "",
                  "原随机 `shared_bilinear64` 作业 1486 因 DataLoader 共享内存错误失败。仅该项重跑为 1565，已 COMPLETED 0:0，用时 5:54:57；随机报告 1566 与最终报告 1567 已完成。",
                  "重跑使用原科学代码快照和相同种子/模型/数据/优化规则，仅 workers=4 改为 0，从 epoch 0 重训，并非最佳 checkpoint 的精确断点续训。不承诺不同数据加载策略逐位等价；原失败记录保留。"]
    else:
        lines += ["5 个候选 × 2 种划分全部完成。两个划分的最低 Overall mean MAE 均来自单纯继续训练 LoRA（`lora_continue`）；四种 PRS 组合均未胜过这个同期对照。",
                  "相对原初始化的下降不能自动算作 PRS 模块收益。进一步降低学习率继续训练本身已有收益，PRS 的额外效果必须与同期继续训练的 LoRA 比。", "",
                  "随机报告 1505、时间报告 1512、最终报告 1513 均 COMPLETED 0:0；最终选择仍保留 LoRA。"]
    lines += ["", "## 统一实验边界", "",
              "- 数据协议：`development-calbased-analogue-v1`，只使用父级 meta-train 人群。",
              "- 2,051 人：MIMIC 1,011 人、VitalDB 1,040 人；每人 320 个带 BP 标签的 10 秒训练窗口、40 个内部验证窗口、40 个仍封存窗口。",
              "- 每种划分验证覆盖全部 82,040 个窗口，没有剔除最差 30% 或低质量验证窗口。",
              "- 同一人的训练与内部验证允许重合身份，但窗口/生理区间/重复波形按既有数据审计分离。随机划分反映已知用户数据内插值；时间划分更接近使用过去训练片段预测较后片段。",
              "- 本批为单种子开发探索；内部验证可选 checkpoint，但不能更新个人状态或充当输入。没有打开封存 held-out 或独立 meta-test。",
              "- 这是自建 CalBased analogue，不是官方 PulseDB CalBased 固定划分的严格复现；也不是新用户 K-shot、320 次袖带校准或独立外部验证。",
              "- MIMIC/VitalDB 是 PulseDB 内部分来源分层。Overall 保留原报告对全体受试者重算的结果，不将两来源简单平均。",
              "- 升级门槛：同一方法在两种划分的 Overall mean MAE 均改善至少 0.15 mmHg，并在两个来源均改善。", "",
              "## 候选及两种划分总体结果", ""]
    meanings = ({"subject_lora_rank4": "个人线性 rank-4 特征变换（2,048 参数/人）",
                 "shared_lora_rank4": "所有人共用一个 rank-4 适配器",
                 "subject_lora_rank1": "缩小至个人 rank-1（512 参数/人）",
                 "output_profile32": "32D 个人代码在输出端修正（34 参数/人）",
                 "feature_affine32": "个人代码控制特征尺度与平移（34 参数/人）",
                 "shared_bilinear32": "共享方向、个人32D系数（34 参数/人）",
                 "shared_bilinear64": "共享方向、个人64D系数（66 参数/人）",
                 "subject_nonlinear_rank4": "相同2,048参数，rank-4中加入非线性"} if feature else
                {"lora_continue": "只继续训练原 LoRA，同期对照",
                 "lora_prs_bias": "LoRA + 每人两个输出偏差",
                 "lora_prs_dynamic": "LoRA + 32D个人代码与波形相关输出修正",
                 "lora_prs_dynamic_shrink": "上述动态修正加抑制过大修正的惩罚",
                 "lora_prs_dynamic_frozen": "冻结原 LoRA，仅训练新个人修正"})
    summary = []
    for candidate in spec["names"]:
        row = {"Setting": candidate, "改变": meanings[candidate]}
        for mode, label in zip(MODES, ("Random mean MAE", "Chronological mean MAE")):
            row[label] = float(macro[mode].loc[macro[mode].candidate.eq(candidate) & macro[mode].view.eq("Overall"), "mean_mae"].iloc[0])
        summary.append(row)
    lines += [table(pd.DataFrame(summary)), "", "单位 mmHg，mean MAE=(SBP MAE+DBP MAE)/2，受试者等权；越低越好。", ""]
    if not feature:
        baseline = comparison.loc[comparison.candidate.eq("lora_continue") & comparison.view.eq("Overall"),
                                  ["split_mode", "initial_reference_mean_mae", "mean_mae", "gain_vs_initial"]]
        lines += ["### 单纯继续训练的收益", "", table(baseline), "",
                  "两个划分相对初始化的收益并未同时达到0.15 mmHg，因此这也不能被写成通过旧结构升级门槛的新方法。", ""]
    lines += ["## 全部受试者等权结果", ""]
    for mode in MODES:
        display = macro[mode][["candidate", "view", "n_participants", "n_events", "sbp_mae", "dbp_mae", "mean_mae"]]
        lines += [f"### {mode}", "", table(display), ""]
    lines += ["## 要求的完整指标表", "",
              "下表来自同一批预测的逐窗口诊断：ME=预测−参考血压；STD 为预测误差的样本标准差，不是跨种子的 MAE 标准差。阈值列为累计百分比。",
              "每人恰有40个验证窗口，所以本批 MAE 与受试者等权值仅有数值舍入差；R²、ME、STD和阈值列仍按逐窗口定义计算。",
              "`AAMI*`/`BHS*` 只表示回顾性数值筛查，既不等于临床或设备认证，也不是正式标准验证研究通过。", ""]
    for mode in MODES:
        for scope in SCOPES:
            display = diagnostics[mode].loc[diagnostics[mode].Scope.eq(scope), DIAG_COLUMNS[1:]]
            lines += [f"### {mode} · {scope}", "", table(display), ""]
    lines += ["## 解释与下一步", ""]
    if feature:
        lines += ["- 先前冻结诊断支持正确个人状态与当前PPG均有作用，本次训练比较进一步说明：直接保留个人特征变换比已测的输出压缩修正更准确。",
                  "- shared 与 personal 也改变个人容量，因此不能把它们的差距全部归因为身份机制；rank4 线性/非线性是较干净的同容量比较，非线性没有获益。",
                  "- 不将失败的 shared_bilinear64 最佳中间 checkpoint 当正式结果。本表使用成功重跑及完整验证输出。",
                  "- 后续应提出有限、有明确证据门槛的假设，保留强 LoRA 参照；本批不支持再无区别叠模块。"]
    else:
        lines += ["- 保留 LoRA 的已有个人特征映射是必要起点；在其输出再加 PRS 并没有本批所需的独立收益。",
                  "- 冻结整个基模型只训练新修正，不等于已经验证“只调个人 LoRA”和“只调共享网络”的分离优化；二者仍是不同的待测机制问题。",
                  "- 目前结果不能证明临床连续监测、新用户少样本适应或真实压力/运动/跨设备鲁棒性。"]
    lines += ["", "## 可复核文件", "",
              f"- [跨划分比较](../results/{key}/cross_split_comparison.csv)；[升级门槛](../results/{key}/promotion_gate.csv)；[最终选择](../results/{key}/selection.json)。",
              f"- [数据与报告核查](../results/{key}/publication_audit.json)；[作业完成记录](../results/{key}/job_states.csv)。",
              f"- [随机总体诊断](../results/{key}/random_disjoint/event_pooled_overall.csv)、[MIMIC](../results/{key}/random_disjoint/event_pooled_mimic.csv)、[VitalDB](../results/{key}/random_disjoint/event_pooled_vitaldb.csv)。",
              f"- [时间总体诊断](../results/{key}/chronological_blocked/event_pooled_overall.csv)、[MIMIC](../results/{key}/chronological_blocked/event_pooled_mimic.csv)、[VitalDB](../results/{key}/chronological_blocked/event_pooled_vitaldb.csv)。",
              "", "本次发布仅复制并核查服务器已完成的聚合报告，不下载受试者级预测或原始波形；不会将聚合核查描述成重新运行了全部患者预测。",
              "已有多轮内部验证探索存在选择偏差；正式论文的效应和不确定性需要后续冻结方案及独立确认。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--host")
    parser.add_argument("--server-output-root")
    parser.add_argument("--archive-output-root")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only and not all((args.host, args.server_output_root, args.archive_output_root)):
        parser.error("fetching requires explicit host, server-output-root and archive-output-root")
    for key, spec in SCREENS.items():
        output = args.project_root / "results" / key
        macro, diagnostics, provenance = {}, {}, []
        if args.render_only:
            for mode in MODES:
                macro[mode] = pd.read_csv(output / mode / "participant_macro_summary.csv")
                diagnostics[mode] = pd.read_csv(output / mode / "event_pooled_diagnostics_all_scopes.csv")
            comparison = pd.read_csv(output / "cross_split_comparison.csv")
            gate = pd.read_csv(output / "promotion_gate.csv")
            selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        else:
            paths = {}
            for mode, job in zip(MODES, (spec["random"], spec["chrono"])):
                folder = f"{spec['id']}_{mode}_report_seed{spec['seed']}_job{job}"
                for filename in ("participant_macro_summary.csv", "event_pooled_diagnostics_all_scopes.csv"):
                    paths[(mode, filename)] = folder + "/" + filename
            final = f"{spec['id']}_final_report_seed{spec['seed']}_job{spec['final']}"
            for filename in ("cross_split_comparison.csv", "promotion_gate.csv", "selection.json"):
                paths[("final", filename)] = final + "/" + filename
            def get(item):
                label, relative = item
                payload = fetch(args.host, args.server_output_root, relative)
                archive = fetch(args.host, args.archive_output_root, relative)
                if payload != archive:
                    raise ValueError("work/NAS aggregate mismatch: " + relative)
                return label, payload, {"source_report": relative, "sha256": sha256(payload).hexdigest(), "work_nas_identical": True}
            with ThreadPoolExecutor(max_workers=4) as executor:
                fetched = list(executor.map(get, paths.items()))
            raw = {label: payload for label, payload, _ in fetched}
            provenance = [record for _, _, record in fetched]
            for mode in MODES:
                macro[mode] = sanitize(csv_bytes(raw[(mode, "participant_macro_summary.csv")]))[MACRO_COLUMNS]
                diagnostics[mode] = csv_bytes(raw[(mode, "event_pooled_diagnostics_all_scopes.csv")])
            comparison = sanitize(csv_bytes(raw[("final", "cross_split_comparison.csv")]))
            gate = csv_bytes(raw[("final", "promotion_gate.csv")])
            selection = json.loads(raw[("final", "selection.json")])
        checks = validate(spec, macro, diagnostics, comparison, gate, selection)
        if not args.render_only:
            jobs = {str(spec[x]) for x in ("random", "chrono", "final")}
            for frame in macro.values():
                jobs.update(re.search(r"_job(\d+)$", run).group(1) for run in frame.run_id)
            if key == "personal_feature_mechanisms":
                jobs.add("1486")  # Preserve the failed historical attempt, never score it.
            command = "sacct -j " + ",".join(sorted(jobs, key=int)) + " --noheader --parsable2 --format=JobIDRaw,State,ExitCode,Elapsed"
            accounting = subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host, command], timeout=60).decode()
            rows = [line.split("|")[:4] for line in accounting.splitlines() if line.split("|")[0] in jobs]
            states = pd.DataFrame(rows, columns=["job_id", "state", "exit_code", "elapsed"])
            if set(states.job_id) != jobs or states.job_id.duplicated().any():
                raise ValueError("missing/duplicate job completion record")
            completed = states.loc[~states.job_id.eq("1486")]
            if not completed.state.eq("COMPLETED").all() or not completed.exit_code.eq("0:0").all():
                raise ValueError("a selected training/report job is not successfully complete")
            output.mkdir(parents=True, exist_ok=True)
            for mode in MODES:
                destination = output / mode
                destination.mkdir(exist_ok=True)
                macro[mode].to_csv(destination / "participant_macro_summary.csv", index=False)
                diagnostics[mode].to_csv(destination / "event_pooled_diagnostics_all_scopes.csv", index=False)
                for scope in SCOPES:
                    diagnostics[mode].loc[diagnostics[mode].Scope.eq(scope)].to_csv(destination / f"event_pooled_{scope.lower()}.csv", index=False)
            comparison.to_csv(output / "cross_split_comparison.csv", index=False)
            gate.to_csv(output / "promotion_gate.csv", index=False)
            states.to_csv(output / "job_states.csv", index=False)
            (output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            checks.update({"checked_utc": datetime.now(timezone.utc).isoformat(), "source_reports": provenance,
                           "selected_training_report_jobs_completed": True,
                           "public_data": "aggregate only; source absolute run paths reduced to run IDs"})
            (output / "publication_audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        content = render(key, spec, macro, diagnostics, comparison, selection)
        (args.project_root / "docs" / spec["document"]).write_text(content, encoding="utf-8")
        print(json.dumps({"study": key, **checks, "source_reports": len(provenance)}, ensure_ascii=True))
        for mode in MODES:
            print(macro[mode].loc[macro[mode].view.eq("Overall"), ["candidate", "sbp_mae", "dbp_mae", "mean_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
