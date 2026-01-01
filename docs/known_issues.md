# Known Issues & Fix Plan

> **创建日期**: 2026-01-01
> **来源**: 外部代码审查

---

## 🔴 P0: 必须修复（影响评测正确性）

### ~~1) Task3 manifest 文件名不一致~~ ✅ 已修复

**修复内容**：
- `evaluate_model.py`: 添加 `find_manifest()` 自动探测
- `evaluate.py`: 同上
- `scripts/generate_artifacts.py`: 同上

**修复日期**: 2026-01-01

---

### ~~2) unfold 丢尾巴~~ ✅ 已修复

**修复内容**：
- 添加 `safe_unfold()` 函数到以下文件：
  - `evaluate_model.py`
  - `scripts/generate_artifacts.py`
  - `experiments/run_ablations.py`
- 所有 `unfold` 调用已替换为 `safe_unfold`

**修复日期**: 2026-01-01

---

### ~~3) Task3 答案长度泄漏~~ ✅ 已修复

**问题**（o3-pro 审查发现）：
- `prepare_task3_samples()` 把 `ans_len_aligned` 写进输入总长度
- 模型可通过输入长度推断答案位数（1位/2位）

**修复**：
- 新增 `max_answer_symbols` 配置项（默认 2）
- 输入使用固定最大长度，不再泄漏实际答案位数

**修复日期**: 2026-01-01

---

### ~~4) CTC decode 全序列 vs 答案窗口~~ ✅ 已修复

**修复内容**：
- `evaluate_model.py` 现在对 Task3 mod 计算答案窗口起始帧
- CTC 解码只在答案窗口内进行，与训练管线一致
- 计算逻辑：expr_len_aligned + thinking_gap_aligned → answer_start_frame

**修复日期**: 2026-01-01

---

## 🟡 P1: 建议修复

### ~~5) --tasks 参数和实际评测不一致~~ ✅ 已修复

**修复内容**：
- 新增 `--checkpoint-dir` 参数
- 自动匹配 `{task}_*.pt`, `*_{task}.pt`, `{task}.pt` 等模式
- 按修改时间选择最新的 checkpoint

**用法**：
```bash
# 单任务
python evaluate_model.py --checkpoint runs/mirror.pt --tasks mirror

# 多任务自动匹配
python evaluate_model.py --checkpoint-dir runs/ --tasks mirror bracket mod
```

**修复日期**: 2026-01-01

---

## 🟢 P2: 可选改进

### ~~6) 一键复现脚本~~ ✅ 已实现

**产出**：
- `scripts/repro_tiny.py` - 主脚本
- `scripts/repro_tiny.ps1` - Windows
- `scripts/repro_tiny.sh` - Linux/macOS

**用法**：
```bash
python scripts/repro_tiny.py  # 5 分钟验证
```

**修复日期**: 2026-01-01

### ~~7) 常见坑文档~~ ✅ 已完成

**产出**：README.md / README_CN.md 新增 FAQ 部分
- 采样率问题
- 随机种子
- 显存不足
- 评测全 0

**修复日期**: 2026-01-01

---

## 📊 修复状态

| 优先级 | Issue | 状态 |
|--------|-------|------|
| P0 | manifest 文件名 | ✅ 已修复 |
| P0 | unfold 丢尾巴 | ✅ 已修复 |
| P0 | **答案长度泄漏** | ✅ 已修复 |
| P0 | CTC 答案窗口 | ✅ 已修复 |
| P1 | tasks 参数 | ✅ 已修复 |
| P2 | 复现脚本 | ✅ 已实现 |
| P2 | 坑文档 | ✅ 已完成 |

---

## 📐 评测口径对齐

### 权威评测脚本

| 用途 | 脚本 | 说明 |
|------|------|------|
| Oracle/协议验证 | `evaluate.py` | 编解码闭环，不测模型 |
| Model 能力评测 | `evaluate_model.py` | 禁用所有 guidance |
| 波形重建质量 | `scripts/stft_sdr_eval.py` | STFT-SDR 指标 |

### 评测约定

- **Model EM 必须禁用所有训练时的 guidance**（cls_guidance, remainder_guidance 等）
- **Task3 使用答案窗口 CTC 解码**，不是全序列
- **所有报告数字必须注明**：脚本、split、seed、limit

### Window-only vs Full Sequence

| 场景 | 方式 | 当前状态 |
|------|------|----------|
| Task3 训练 | 答案窗口 | ✅ 已实现 |
| Task3 评测 | 答案窗口 | ✅ 已对齐 |
| Task2 训练/评测 | 固定答案窗口 | ✅ 无泄漏 |

---

## 🧪 对照实验计划

### 已完成

| 实验 | 目的 | 状态 |
|------|------|------|
| 随机映射负对照 | 排除捷径学习 | ✅ |
| STFT-SDR | 证明波形重建 | ✅ |
| 10-seed 显著性 | 统计置信 | ✅ |
| wav2vec2 baseline | 强 baseline 参考 | ✅ |

### 待完成（非阻塞）

| 实验 | 目的 | 优先级 |
|------|------|--------|
| Decoder ceiling | 分离 decoder vs model 贡献 | P2 |
| ~~Model-only (no guidance)~~ | 纯模型能力上限 | ✅ 已确认 |
| Oracle-guided upper bound | 渲染上限参考 | P2 |
| Chirp 信号 | 连续域特有逻辑 | P3 |
| Time Warping | 拓扑鲁棒性 | P3 |

---

## 🎯 当前待办

### P0：发布阻塞

| 事项 | 说明 | 状态 |
|------|------|------|
| ~~docs/ 公开决策~~ | 已公开 overview.md, known_issues.md, iteration_log.md | ✅ 已完成 |
| arXiv endorsement | 需要找人背书才能提交 | ⏳ 待处理 |

### P1：提升可信度（非阻塞）

| 事项 | 耗时 | 状态 |
|------|------|------|
| ~~端到端闭环图~~ | 30min | ✅ 已完成 |
| ~~CTC 答案窗口对齐~~ | 1-2h | ✅ 已完成 |
| ~~Model-only (no guidance) 对照~~ | - | ✅ 已确认（eval_model 不调用任何 guidance 函数） |

### P2：后续扩展

| 事项 | 说明 | 状态 |
|------|------|------|
| 信道扰动压力测试 | AWGN、相位偏移等 | ⏳ |
| thinking gap ablation | gap vs EM 曲线 | ⏳ |
| 纯 SSM baseline | 解释 hybrid 优势 | ⏳ |

---

> **最后更新**: 2026-01-01

