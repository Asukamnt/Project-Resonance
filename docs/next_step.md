## Next Step（行动清单）：把“成功原型”升级成“可公开/可合作/可发表”的证据链

> 本文目的：把我们之前讨论过的**所有还该做的事**汇总成一个可执行的清单，按 **Week5–Week8** 的目标来推进，并且每一项都有**验收口径/止损条件**，避免发散拖垮节奏。

---

## 0. 当前状态（我们已经完成了什么）
- **范式成立**：token 不落地的 **wave→wave** 推理闭环（输入波形→模型→输出波形→外部 FFT 解码评测）。
- **三任务可复现**：
  - Task1（Mirror）：IID=1.00，OOD-length=1.00
  - Task2（Bracket）：IID=0.96，OOD-length=0.84，OOD-noise(10dB)=0.97（P0 最小证明 + P1 多轴 OOD）
  - Task3（Mod）：EM=0.315（单步 A%B）
- **工程可复现**：统一产物（preds/metrics）、config/seed/commit 元数据、pytest 全绿。

参考文档：
- 规范清单：`docs/spec_from_pdf.md`
- 原始周计划：`docs/plan.pdf` / `docs/plan_text.txt`
- 迭代事实记录：`docs/iteration_log.md`
- 决策表（MET 护栏）：`docs/decision_table_phase4_5_6.md`

---

## 1. 我们“还没做完”的核心缺口（按 Week5–Week8）

### Week5（Scaling Up Complexity / Curriculum）
- **Task3 多步/组合**：计划里有 A%B%C 等组合推理，但当前实现**明确只支持单步**（需要作为能力边界写清，并在后续补齐）。
- **课程/规则选择系统化**：目前多靠启发式逐条修复；需要把“选规则/选课程”变成可证伪外循环（TwoPhase NI 的落点）。

### Week6（OOD Evaluation & Tuning）
- **负对照（negative controls）标准化**：label shuffle / phase scramble / control-signal shuffle 等（证明不是捷径）。
- **全任务 OOD 总报告**：一次性汇总 Task1/2/3 的 IID + 多轴 OOD（至少两轴）并给出曲线或表格。

### Week7（Ablations & Robustness）
- **≥5 个核心消融套件**：需要可一键复现，输出统一 CSV/Markdown 表，避免“隐式消融”无法对外说清。

### Week8（Final Integration & Documentation）
- **发布资产化**：risk log、milestone log、artifacts（复现 seeds/示例音频/可选 checkpoint）等。
- **对外一键复现入口**：README 顶部结果表 + 一键命令（让外人 3 分钟内验证主张）。

---

## 2. 最小证据任务（MET）总览（主线 + 副线）

### 主线（必须优先）：把 Phase1（本 repo 的“证据链”）做成论文级
- **MET-PHASE1-NEG**：负对照套件（至少 2 种）+ 结果表
- **MET-PHASE1-ABL**：5 个核心消融 + 结果表
- **MET-PHASE1-REPORT**：统一总评估报告（Task1/2/3 + IID/OOD 多轴）+ 一键命令

### 副线（二选一，不得抢主线资源）
1) **TwoPhase NI（Selector/Subject）**：让“选规则/选课程”成为可证伪外层优化（见 `docs/decision_table_phase4_5_6.md` Phase4 表格行）。
2) **Big Wave Model（最小 scaling study）**：3 点曲线、2 seeds、等 FLOPs 对照，输出“scaling 是否带来 OOD 增益”的结论。

---

## 3. 5 个核心消融（Week7 的最低可交付）
建议优先这 5 个（都与你们已经证实“会影响 OOD”的关键因素相关）：
1) **无 RoPE**（Attention 不用 RoPE）
2) **有/无绝对 pos_emb**（或保持当前“无 pos_emb”对照一个“有 pos_emb”分支）
3) **无 cls_guidance（Task2）**
4) **无 answer-window-only loss（Task2/Task3）**
5) **无连续渲染修复（P0 OLD tile vs NEW continuous）**

**验收（Pass）**：每个消融给出 IID + OOD（至少两轴）的表格/曲线，并能指出“主要掉在哪个失败模式”（空输出/偏置/能量塌陷/写入失败）。  
**止损（Fail）**：若消融结果噪声极大（seed 方差覆盖结论），先降低问题规模（tiny manifest/limit），再补更多 seeds。

---

## 4. 负对照（Week6 的最低可交付）
至少两种：
- **Label shuffle**：训练标签随机置换，期望 IID/OOD 都崩（证明评测口径没漏洞）。
- **Phase/randomization stress**：对输入或目标做相位扰动（或答案窗随机化），期望模型不应“凭静默段/长度”投机。

**验收（Pass）**：负对照能明显击穿性能，并且诊断字段能解释“为什么崩”。  
**止损（Fail）**：若负对照仍高分，优先怀疑 scorer/评测口径/数据泄漏。

---

## 5. 统一总报告（Week8 的最低可交付）
对外需要一个“单命令总入口”，产出：
- 三任务结果表（IID + OOD-length + OOD-noise + 其它已实现 OOD）
- 关键诊断汇总（pred_empty_rate、disagree_rate、answer_rms 等）
- 运行元数据（config/seed/commit/manifest/机器信息）

**验收（Pass）**：外部用户按 README 一条命令能复现实验并得到同结构的 `metrics.json`/汇总表。  
**止损（Fail）**：如果一条命令太重，允许拆成两条：`train` 与 `evaluate`，但必须提供脚本化 wrapper。

---

## 6. 7 天日程板（按周目标推进：Week6→Week7→Week8）
> 原则：每天只交付一个“可验收物”，其余都算噪声；80% 做主线，20% 做传播与协作入口。

### Day 1（Week6）：负对照 v0
- 交付：`NEG-1` label shuffle 结果 + 小表格（IID/OOD）
- 验收：负对照显著击穿；失败模式可解释

### Day 2（Week6）：负对照 v1 + 诊断固化
- 交付：`NEG-2` phase/答案窗随机化（或等价扰动）+ 诊断字段汇总
- 验收：负对照击穿 + 诊断能区分失败类型

### Day 3（Week7）：消融套件 v0（先跑 2 个）
- 交付：RoPE ablation + cls_guidance ablation（含 OOD 多轴表）
- 验收：结论方向明确（哪怕初版只跑 tiny/limit）

### Day 4（Week7）：消融套件 v1（补齐到 ≥5）
- 交付：pos_emb/answer-window-only/continuous-render 等剩余消融
- 验收：≥5 个消融齐全，输出统一表格

### Day 5（Week8）：统一总报告 v0
- 交付：`report.md` 或 `report.json`（三任务 + 多轴 OOD 汇总）+ 一键入口
- 验收：本地可复现，产物结构稳定

### Day 6（Week8）：发布资产化 v0
- 交付：`docs/risk_log.md`、`docs/milestone_log.md`、`artifacts/`（seeds/示例音频/可选 checkpoint）
- 验收：README 指向这些资产；最小自检脚本/命令可跑

### Day 7（对外）：传播与合作入口（低成本）
- 交付：README 顶部“结果表 + 复现命令 + P0/P1 证据链”；X/知乎/B站任选其一发首帖（只讲证据链与下一步）
- 验收：外部 3 分钟能判断“这事是真的”，并知道怎么复现/怎么合作

---

## 7. 对外沟通护栏（避免夸张叙事）
- 只讲可复现事实：**P0 最小证明 + P1 多轴 OOD + 三任务结果表 + 一键复现**。
- 明确边界：Task3 当前是**单步 mod**，多步/组合属于后续 Week5+ 工作。
- “想法”可以写进 roadmap，但**对外主张必须绑定证据链**。


