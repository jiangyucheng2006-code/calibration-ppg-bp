# 从强 LoRA 基线到可检验的方法论文

日期：2026-09-07。状态：**研究方案与第一阶段诊断原型；不是已验证的新模型，不是投稿成稿。**

后续执行更新：用户已批准开展试验。[本轮可执行方案](PLAN_PERSONAL_MEMORY_V1.md)
明确固定训练距离门控与普通监督关系学习的首轮实现，不声称它们是 OOF；
本文件末尾“未提交/未上传”的文字记录的是最初方案回合，不是最新任务状态。
实际提交与完成情况以 [STATUS](STATUS.md) 为准。

本轮按用户要求直接提出方法和论文方案，不要求用户先完成论文写作访谈。
未承诺发表、未声明首创/SOTA、未改变数据划分或既有晋升门槛。

## 1. 结论与当前证据

当前缺少的不是一个有低误差的模型，而是相对强基线有证据支持的额外贡献。
能发表不等于每项 MAE 必须第一；但如果精度、可靠性、数据负担、实际成本、机制证据均无
实质增量，仅改名字或堆模块不足以支持方法论文。方法论文和评测/机制论文是不同路线，不能
在看见失败的最终测试结果后临时更换成功标准。

本次重新读取的证据：

- `RESULTS_PERSONAL_MECHANISMS.md`：打乱同一个人的当前 PPG，LoRA 随机划分 mean MAE
  由 3.0394 增至 10.6230；只用个人训练 BP 均值为 7.9793。交换个人参数也明显恶化。
  支持模型同时依赖当前波形与正确个人参数；不是个人均值预测器，但不证明因果生理机制。
- `local_archive/status_20260907_lora_prs/RESULTS_REVIEW.md`：同批继续训练 LoRA 的
  random/chronological mean MAE 为 2.880098 / 3.712384 mmHg；四个 PRS 组合均未超过它。
  完整 Overall/MIMIC/VitalDB 指标和标准数值筛查保存在该报告。不能将继续训练收益归于 PRS。
- `same_subject_component_train.py` 的 `build_support_bank`：从每人训练数据的六个均匀
  位置构建候选，每次查询使用五个非自身支持；旧 attention 用低维波形描述符。不是从全部
  320 条个人记录中按当前学习特征检索。这是本项目实现差异，不是全领域首创证明。

已检验的替代包括 PRS、低维个人代码、特征仿射、共享双线性基底、非线性 rank-4、多个
LoRA 组合、早期差分损失及聚类专家。不要将这些重新命名为尚未尝试的创新。

## 2. 保持研究对象不变

- 问题：已注册用户提供个人训练记录后，预测该人的其他 PPG 窗口。
- 数据：PulseDB v2 的 `development-calbased-analogue-v1`，仅 parent meta_train。
- 2,051 人，MIMIC 1,011、VitalDB 1,040；每人 320 条训练、40 条内部验证、40 条封存窗口。
- 两个独立实验条件：`random_disjoint` 与 `chronological_blocked`。
- 保留每个人的 LoRA A/B、身份映射、train-only BP 锚点。每个窗口 10 s，不是一次独立袖带测量。
- 主输入为 PPG，允许已授权的个人历史 PPG/BP；不用 query 的 ABP/真实 BP、ECG 或未来标签。
- 这不是官方 CalBased 精确复现、未见用户少样本、真正跨设备或临床设备验证。
- 本轮不强制转向未见用户 K-shot；也不删除最差 30% 来提高主结果。

## 3. 推荐的单一主候选：个人测量记忆与参数记忆互补

暂用描述性名称：**Personal Memory–Augmented Calibration（个人测量记忆增强校准）**。
该名称只标识候选，不表示这些通用概念由本项目首次提出。

一句话假设：LoRA 将个人规律压缩进参数，但某些当前波形附近的历史测量关系仍可能有
可利用的增量；只有历史确实能提供可靠参考时，模型才应利用它，而不是无条件加输出修正。

这个假设可能失败：LoRA 可能已学完可用信息；形态接近未必 BP 接近；训练记忆可能只帮助
随机插值而无法帮助后续时间窗口。低 MAE 本身不证明存在可补充的记忆信息。

### 3.1 参数路径：保留已验证的 LoRA

\[
z=E(x),\qquad v_s(x)=z+B_sA_sz/r,\qquad p_s(x)=\mu_s+H(v_s(x)).
\]

其中 x 是当前 10 s PPG；E/H 为共享网络，A_s/B_s 为用户 s 的个人参数；r=4；
mu_s 只由个人训练 BP 得到。沿用源模型的标准化和反标准化，最终 SBP/DBP 单位为 mmHg。
当前实现共享网络与个人 A/B 联合训练，应明确这与冻结主干后 LoRA 微调的文献实现不完全相同。

### 3.2 测量记忆：只存该人的合法训练记录

\[
\mathcal M_s=\{(v_s(x_j),y_j,\mathrm{key}_j,\mathrm{record}_j,t_j)\}_{j\in\mathrm{train}(s)}.
\]

检索只看个人身份、PPG 特征及合规性元数据，不看 query BP。第一版固定从最多 320 条合法
训练记忆中取 m=5 个近邻；**m 是每次检索条数，不是 K-shot 校准预算**。预算仍是 320 条。
memory 更大并不意味着新增血压测量，因为这些记录已经用于个人模型训练；但推理时显式
访问历史 BP 是新增输入机制，必须在比较表里说明。

### 3.3 局部关系网络：学习相对变化，不机械复制参考 BP

\[
p_{\rm mem}(q)=\sum_{j\in N_s(q)}w_j\,[y_j+g_\phi(v_q,v_j)].
\]

g_phi 为共享小型成对网络，输出 SBP/DBP 差值；起步输入为两个特征、差值和逐元素乘积，
一层 64 维隐层后输出两维。不增加每人的新神经参数，个人性来自已保留的 A/B 及个人 memory。
这只是首个有界设计，不展开宽度、深度、参考条数的大扫参。

训练目标由 train 内不同窗口的 BP 差构成，不用 internal-validation 的标签拟合网络。
可以用 `(G(vq,vj)-G(vj,vq))/2` 保证交换配对时变号、相同输入时为零；
**该约束是结构检查，不是新的物理规律，也不自动保证局部预测准确或时间一致性。**

权重起步采用冻结特征上的 cosine-softmax，温度 0.1；比较均匀权重，以隔离检索收益。
在发现简单检索确有信息之前，不增加可学习特征空间或复杂的多专家分支。

### 3.4 支持距离约束融合：参考不合适时保留强基线

\[
\hat y(q)=(1-a_q)\,p_s(q)+a_q\,p_{\rm mem}(q),\qquad a_q\in[0,1].
\]

a_q 只依据 query-reference 特征距离、邻居分散程度、多个候选预测的一致性等推理可得量。
训练内部嵌套划分拟合融合，验证集只评分和选 checkpoint。不能用真实 query BP 决定走哪条路。
没有合法近邻时 a_q=0，所有 query 仍有基线预测，不通过删样本提高主指标。

“支持距离”只是特征空间代理指标，不等于真实生理状态覆盖，更不是 conformal coverage 保证。
不能事先保证门控不会错误抑制真实 BP 变化；这必须测。

### 3.5 必须正视的代数等价

如果 g(q,j) 仅等于现有模型两次预测的差：

\[
\sum_jw_j[y_j+p_s(q)-p_s(x_j)]
=p_s(q)+\sum_jw_j[y_j-p_s(x_j)].
\]

那么方法严格等价于**加权残差校正**，不是独立的新差分网络。这一项列为 baseline。
最终方法只有在独立关系网络或支持距离约束带来可重复增量时，才可能有方法贡献。
即使独立关系网络有效，也必须与 PPG2BP-Net / deduction learning 的先例比较。

## 4. 第一阶段先回答“记忆有没有增量”

| 编号 | 方法 | 目的 | 当前状态 |
| --- | --- | --- | --- |
| D0 | 冻结的强 LoRA；需要训练时另加等预算 continued LoRA | 真实竞争基线 | 已有结果，后续仍须同一输入/平台重放 |
| D1 | 同人 PPG-feature kNN BP | 看历史测量本身能否提供局部信息 | 本轮实现无数据诊断接口与合成测试 |
| D2 | LoRA + 同人 kNN 残差 | 排除只是普通残差修正 | 本轮实现无数据诊断接口与合成测试 |
| D3 | 同人最近时间的 train BP | 排除收益仅来自时间相近 | 待接入真实 manifest 后实现/运行 |

初始固定 m=5，不进行验证集上大量融合权重搜索。D1 不一定要总体击败 D0；但它应在提前
定义的特征距离/个人 BP 变化分组中提供稳定可利用的信息。若 D1/D2 均明显劣于 D0，且没有
预先定义分组的互补证据，停止该路线，不再自动堆关系网络。

之后才执行有限神经比较：

1. 同预算 continued LoRA；
2. 单参考成对网络；
3. 多参考关系网络 + 均匀权重；
4. 按 PPG 特征检索的多参考关系网络；
5. 完整支持距离约束融合。

每个步骤只解释一个机制。不能把新模型多训练的 epoch 或更多 forward/backward 计算带来的
收益当作模块贡献。记录训练时长、实际步数与样本暴露；时间不同时另报计算预算。

## 5. 防止“检索到答案”的实施契约

1. memory 只来自 train role；内部验证/held-out 数据永不进入 bank。
2. 内部训练 query 对应的整段时间 block 从其 memory 中排除，不仅仅删一个 ID。
3. 全局检查 query 与 memory 的 ID、原始内容 hash、相同 recording 的生理时间区间；
   不能先按个人过滤再做检查，否则错误的身份标签可能掩盖重复。
4. chronological 条件仅允许时间上早于当前 query 的合法历史；时间轴不可跨不同记录
   盲比。缺少可比较时间基准时失败并报告，不把行号当真实时间。
5. 提前固定额外 60 s 排除带作为邻近性敏感性分析；不合格的 query 回退 LoRA，保留100%覆盖。
6. 学习关系网络/门控时使用 train 内部的 memory-fit/query-fit/meta-fit 分区或交叉拟合。
   同一监督样本的标签不能既用于训练该分支，又被当作“未见误差”训练门控；编码器也
   必须遵守对应内层排除，不能只更换 reference bank 却继续用见过查询标签的表征冒充 OOF。
7. 内层 checkpoint 的特征空间不混用；每折独立检索并拟合。OOF 与最终 320 条拟合模型
   的残差分布差异须单独评估，不能直接假设可迁移。
8. 纯 in-sample train residual 只能是 D2 诊断输入，不能称其为未见残差；LoRA 拟合得好时
   这些残差可能接近零。必须记录预测 provenance。
9. 冻结 checkpoint、feature transform、subject mapping、memory manifest 与参数后再评估。
   额外开封测试集属于最终确认步骤，不在本轮执行。

本轮新增 `personal_memory_probe.py` 只检查显式提供的数组、角色与 lineage 元数据；它不能
证明上游模型没有见过标签、hash 正确、记录时间已对齐或原始数据没有近重复。

## 6. 评价与停止规则

主目标继续是原有的全人群精度升级：相对**配对且等条件的强 LoRA**，两种划分 Overall
participant-macro mean MAE 均下降至少 0.15 mmHg，且两种来源均正向改善。不得事后放宽旧门槛。
达到开发门槛也不等于发表，需要候选冻结后的确认。

正式结果照常提供 Overall/MIMIC/VitalDB 三份，随机与时间分别报告：Setting、BP、MAE、R²、
ME、STD、≤5/10/15 mmHg、带限定的 AAMI/BHS 数值筛查。增加成对受试者 bootstrap 不确定性；
确认阶段固定三个随机种子。不能把大量开发筛选后的最佳值当作独立验证证据。

次级诊断在看新结果前定义：

- 个人训练 BP 范围内/外的 query；范围由 train 定义，query BP 仅评分时分组，不参与路由；
- 相对个人 train BP 中位数的变化幅度：0–10、10–20、>20 mmHg，SBP/DBP 分开；
- 特征距离分组、同人近邻时间间隔、两来源分层、最差尾部误差；
- 60 s 排除带后是否仍有收益；有新记录时另做跨记录分析，缺数据不冒称跨会话验证；
- 不拒绝任何人的总体误差；若以后做拒绝预测，应另报覆盖率—误差曲线。

若只提升 random：结论限于已见用户已见分布内插值，不能写成长期漂移鲁棒。
若只提升时间/困难子组而总体不达旧门槛：如实列为探索结果，不算旧目标成功；是否建立新的
有临床/工程依据的主目标，必须在新的确认试验前单独确定，不能追着结果改标准。

memory 的 320×256 float32 特征约320 KiB/人，远大于 LoRA 的8 KiB/人；另有标签/元数据。
它是潜在精度换空间，不是压缩方案。计入真实延迟、训练计算、患者历史存储与访问隐私风险。

## 7. 文献边界与检索记录

检索日期2026-09-07；定向检索，不是系统综述或全网穷尽。
检索词包括 PPG LoRA blood pressure personalization、PPG blood pressure memory retrieval、
intra-subject contrastive blood pressure、shared personalized gradient conflict。
查阅来源为出版方/作者机构、PubMed/Europe PMC、arXiv 与正式会议论文入口。
Crossref `/works/{doi}` 核验下列前三项，3/3返回 DOI、题目、期刊有效元数据。
浏览器部分 Nature/PMC 页面返回访问检查；不将失败响应算全文阅读。

| 最近邻 | 已核验内容与证据等级 | 对本方案的限制 |
| --- | --- | --- |
| Li et al. 2026, [LoRA PPG-BP](https://doi.org/10.1109/JBHI.2026.3665810)；[作者机构摘要](https://research.cuhk.edu.hk/en/publications/few-shot-personalized-blood-pressure-estimation-from-photoplethys/) | 摘要/元数据：Transformer、PPSP、采样率稳健 LoRA；UCI/Queensland/CAS-BP，50样本个人化；未获得全文 | 个人LoRA、脉压约束、采样率适应不是本项目原创；不能推断未读到的受试者交集 |
| Joung et al. 2023, [PPG2BP-Net](https://doi.org/10.1038/s41598-023-35492-y) | 成对 PPG 与校准 BP 的比较网络；[PubMed](https://pubmed.ncbi.nlm.nih.gov/37244974/)及原文可检索段落 | 参考PPG+差分预测有直接先例，必须有单参考/多参考控制 |
| Mekonnen et al. 2024, [Mixed deduction learning](https://doi.org/10.1038/s41598-024-75583-y) | [PMC原文索引](https://pmc.ncbi.nlm.nih.gov/articles/PMC11467377/)：个人9轮数据与群体数据、成对推理；目标验证15人/88测量 | 个人历史+群体知识不是新思路；不跨协议直接比较MAE |
| Wang et al. 2024, [ΔBP-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC11969577/) | PulseDB、40s内BP变化、self-contrastive masking；DOI10.1109/JBHI.2024.3422023 | 变化预测本身已有人做；不能与绝对BP指标混为一谈 |
| Tae et al. 2026, [Change Point–Aware Evaluation and Re-Calibration](https://arxiv.org/html/2608.18639v1) | 已读HTML：PPG变化检测、在线聚类触发新增校准；arXiv稿标注MLHC/PMLR | 变化触发校准非空白；本候选不向测试阶段索取新BP，这是范围差异，不是已证实优势 |
| Wu et al. 2026, [P2E-VQ](https://arxiv.org/html/2608.14656v1) | 已读HTML：训练记忆中的ECG-linked特征检索，PPG-only下游分类 | PPG检索增强已有先例；本方案限于个人PPG-BP回归记忆，不使用ECG |

尚未找到上述近邻中完全等同于本候选的实现，但不能据此断言“第一次”或“市面没有”。
方法贡献的候选差异是：**持续个人参数与全部合法个人测量记忆的互补、独立局部关系学习、
在不获取新测试期 BP 的约束下评估支持距离融合是否有效**。这些差异还没有实证成立。

共享/个人交替优化作为备用机制检查保留，不作为另一个本轮主模型：
[PCGrad](https://arxiv.org/abs/2001.06782)、[FedRep](https://proceedings.mlr.press/v139/collins21a.html)
和[2026 FedSPC预印本](https://arxiv.org/html/2606.13748v1)已有相关先例。
若执行，应先量化“更新甲的共享参数是否真实伤害乙”，而不只计算负梯度夹角。

## 8. 论文题目、主线与写法

主候选暂定题目：

**Personal Memory–Augmented Blood Pressure Estimation from PPG in Registered Users**

中文：**面向已注册用户的个人测量记忆增强 PPG 血压估计**。

若后续严格时间验证确实有收益，可将标题改为更聚焦的支持距离/局部关系表述。
当前不要在标题写 robust、few-shot、clinical-grade、state-of-the-art 或 guaranteed。
论文必须明确采用并引用 LoRA；题目不含 LoRA 不意味着可以隐去方法来源。

供Introduction组织用的研究主线，不是已成立结论：

> 已注册用户的参数化个体模型已有较低平均误差，但仍需检验其是否遗漏与当前波形相关的
> 局部历史测量信息。本研究区分参数记忆与显式测量记忆，在不增加个人标签预算的条件下，
> 检验个人参考检索、局部关系预测与支持距离约束融合能否提升后续窗口估计。

建议实证IMRaD结构，正文约5,000英文词（仅规划，不含尚未存在的结果）：

| 章节 | 词数 | 写什么/需要什么证据 |
| --- | ---: | --- |
| Introduction | 650 | 问题是个人化后的剩余误差，不是“PPG无人做到”；界定已注册用户与标签预算 |
| Related work | 550 | LoRA、参考比较网络、deduction learning、变化监测/记忆检索；说明最近邻差异 |
| Method | 1300 | 两种记忆、查询合法性、成对网络、融合、训练内层划分与复杂度 |
| Experimental design | 700 | 两种split、相同预算、强LoRA+简单检索对照、指标与冻结规则 |
| Results | 1000 | 三来源完整表、时间/距离/变化幅度、逐个机制消融；新结果全部待生成 |
| Discussion | 600 | 支持或否定什么、参考缺失/外推失败、标签与存储负担、非临床边界 |
| Conclusion | 200 | 只总结已验证结论，不把开发性观察写成临床有效 |

过渡逻辑：问题与近邻边界 → 可检验假设 → 实现及公平比较 → 证据 → 反例和适用范围。
投稿前补齐数据可用性、代码/版本/划分清单、伦理依据、作者贡献、资金/利益冲突及目标期刊
要求的工具使用披露。不能预填伦理豁免或批准编号，必须核对数据来源和机构要求。

最少四组图表：完整主指标；机制拆解；性能随个人参考距离/时间/BP变化的变化；失败例与资源。
不把几十轮所有失败模型塞进正文，完整筛选历史放补充材料。

若该路线无增益，备用题目可为 **What Drives Accurate Personalized PPG-Based Blood Pressure
Estimation? A Controlled Study of Waveforms, Personal States, and Evaluation Protocols**。
这是机制/评测论文，需要比现有受试者重合批评更深入的对照和最终确认，不能仅靠现有几张
破坏实验表保证录用。

## 9. 本轮实际完成与未完成

- 已完成：历史代码/结果核查、定向近邻文献检索、上述模型与有限试验设计。
- 第一阶段实现：独立的NumPy个人记忆诊断接口及合成单元测试，验证输入/角色/时间/重复等
  软件约束；不包含经过真实数据训练的局部关系网络或门控。
- 验证：本地 Python 3.12.14 / NumPy 2.3.5 下36个合成单元测试通过；未安装额外依赖。
  执行方式为项目根目录设置 `PYTHONPATH=src`，再运行
  `python -m unittest discover -s tests -p test_personal_memory_probe.py`。
  诊断版采用固定均匀top-5；上文softmax/成对网络属于后续神经候选，未混称已实现。
  诊断遇到不足的合法历史会报错而不静默丢掉query；实际评分驱动程序须记录并回退LoRA。
- 尚未完成：真实记忆特征提取/全数据诊断、关系网络训练、收益验证、最终封存测试和论文成稿。
- 本轮未提交新Slurm训练、未变更正在恢复的旧任务、未上传GitHub；没有任何新模型MAE。
- 下一步：接入审核过的真实train特征/manifest运行D0–D3；根据预定停止规则决定是否训练神经候选。
