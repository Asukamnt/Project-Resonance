# Jericho: End-to-End Reasoning on Raw Physical Waveforms

**[English](README.md)** | **中文**

<p align="center">
  <strong>跨物理波形域（音频 / 光学 / 射频）的端到端符号推理，无需文本中间表示</strong>
</p>

---

## 这是什么？

**Jericho** 是一个实验性框架，验证一个核心假设：

> **神经网络可以直接在不同物理域的连续波形上（音频、光学/IPD、射频）完成符号推理任务，全程不经过离散化的文本/token 中间表示。**

传统的语音理解流程是：`音频 → ASR → 文本 → LLM → 文本 → TTS → 音频`

Jericho 的流程是：`波形 → 神经网络 → 波形`

我们设计了三个递进难度的任务，并在三个物理域上完成了验证：

| 任务 | 输入 | 输出 | 验证的能力 |
|------|------|------|-----------|
| **Task 1: Mirror** | 符号序列波形 | 相同的符号序列波形 | 波形编解码闭环 |
| **Task 2: Bracket** | 括号表达式波形 | 括号匹配结果波形 | 结构推理 |
| **Task 3: Mod** | 数学表达式波形 | 取模运算结果波形 | 算术推理 |

**支持的物理域**：音频（正弦波）· 光学/IPD（强度-相位）· 射频（幅度调制）

### 端到端闭环

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Manifest   │───▶│   Synth     │───▶│ Mini-JMamba │───▶│ FFT Decode  │
│ (符号序列)   │    │ (符号→波形)  │    │  (推理)      │    │ (波形→符号)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                   │                   │
                         ▼                   ▼                   ▼
                    输入波形 ──────▶ 输出波形 ──────▶ Exact Match
```

| 评测类型 | 脚本 | 测什么 |
|----------|------|--------|
| **Oracle EM** | `evaluate.py` | 编解码协议正确性（不测模型） |
| **Model EM** | `evaluate_model.py` | 模型推理能力（禁用所有 guidance） |

---

## 为什么这很重要？

> **核心主张**：符号仅用于监督与评测，推理发生在连续隐状态轨迹中——不是"换皮 token"。

1. **信息保真**：离散化丢失相位与时序微结构。我们直接在波形上推理，保留完整信号。

2. **因果流式**：SSM 架构天然因果，每帧输出只依赖过去，延迟 = 帧长。

3. **跨域迁移**：同一模型在 Audio / Optical / RF 三种物理波形间迁移成功。

详细实验设置与统计方法见 [`docs/iteration_log.md`](docs/iteration_log.md)

---

## 核心组件

- **Mini-JMamba**：12 层 Mamba-2/Attention 混合架构，直接处理原始波形
- **多域编码器**：音频、光学（IPD）、射频三个物理域的符号-波形映射
- **Scorer 解码器**：基于 FFT 的频率识别，用于评估
- **Manifest 系统**：可复现的数据生成与拆分
- **跨域流水线**：跨物理域的训练和推理
- **闭环评测**：从 manifest 到合成、推理、解码、Exact Match 的完整流水线

---

## 开发历程

| 日期 | 里程碑 | 说明 |
|------|--------|------|
| 2025-12-26 | **Stage A 框架搭建** | Task 1 编解码闭环、Scorer、测试基础设施 |
| 2025-12-28 | **Task 2 OOD 突破** | 括号匹配任务、RoPE + 连续波形生成 |
| 2025-12-29 | **Phase 1 完成** | 评估工具、消融实验、负对照验证 |
| 2025-12-31 | **跨域发布** | 音频/光学/射频三域、迁移学习验证 |
| 2026-01-01 | **代码质量修复** | 答案长度泄漏修复、unfold 尾部修复、一键复现脚本 |

---

## 当前状态

### 🎉 核心突破

| 实验 | 结果 | 意义 |
|------|------|------|
| **单域推理** | Mini-JMamba 45% vs wav2vec2 22%¹ | 小模型优势 |
| **跨域推理** | IPD→Audio IID 98.7% | 跨物理域成功 |
| **跨域迁移** | +1.7pp (p<0.05, 10-seed) | 统计显著 |
| **三角验证** | Audio↔IPD↔RF 6/6 | 载体无关证据 |

> ¹ wav2vec2 用于验证"通用语音预训练是否适合波形推理"，非公平对比。结论：任务特化架构更优。

### ✅ 已完成

- Phase 1: Audio 域单域推理
- Phase 2: IPD（光学）域单域推理  
- Phase 3: 跨域推理（IPD→Audio）
- Phase 4: 跨域迁移验证
- 三物理域完整验证（Audio / IPD / RF）
- 完整测试套件（187 用例）全部通过

---

## 实验结果

> Model EM 评测禁用所有训练时的 guidance，纯模型输出 → FFT 解码。详见 [`docs/iteration_log.md`](docs/iteration_log.md)

### 单域推理（Audio，Task 3 Mod）

| 模型 | 参数量 | IID EM | 
|------|--------|--------|
| wav2vec2-base¹ | 94.57M | 22% |
| Transformer | 1.2M | 41% |
| **Mini-JMamba** | **0.94M** | **45%** |

### 跨域推理（IPD → Audio）

| 指标 | 结果 |
|------|------|
| IID EM | 98.7% ± 1.5% |
| OOD EM | 67.3% ± 2.5% |

### 跨域迁移

| 方向 | Δ EM | 统计显著性 |
|------|------|-----------|
| Audio → IPD | +1.7pp | ✅ 95% CI 不含 0 |
| Audio → RF | +0.3pp | 收敛加速 9 epochs |

---

## 快速开始

### 环境配置

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest -q
```

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q
```

### 运行示例

```powershell
# 生成 Task3 (Mod) manifest
python -m jericho.data.make_task3_manifest --out manifests/task3_tiny.jsonl --seed 321 --preset tiny --balance-remainder

# 训练 Mini-JMamba
python .\train.py --config configs\task3_mod_stable.yaml --manifest manifests\task3_tiny.jsonl --split iid_test --limit 200

# Oracle/Protocol 闭环验证（验证编码→解码系统正确性，非模型能力）
python .\evaluate.py --stage final --tasks mirror bracket mod

# 训练后评估模型能力（需要先通过 train.py 生成 checkpoint）
# python .\evaluate_model.py --checkpoint runs\your_run\mod_seed123_epoch50.pt --tasks mod --splits iid_test --limit 50
```

> **评测口径说明**：
> - **Oracle EM**：系统闭环验证，编码→解码一致性（`evaluate.py`）
> - **Model EM**：训练模型能力，模型预测准确率（`evaluate_model.py`）
> 
> Oracle EM = 1.0 证明评测协议正确；Model EM 才反映模型真实能力。

---

## 详细文档

📖 **[技术概览 (docs/overview.md)](docs/overview.md)** — 完整动机、设计哲学、关键概念解释

📋 **[已知问题 (docs/known_issues.md)](docs/known_issues.md)** — 评测口径、对照计划、bug 状态

📊 **[实验日志 (docs/iteration_log.md)](docs/iteration_log.md)** — 完整可复现信息

<details>
<summary><strong>目录结构</strong></summary>

- `src/jericho/symbols.py`：符号表、频率映射与正弦波形合成
- `src/jericho/domains/`：多域波形编码器（音频、光学/IPD、射频）
- `src/jericho/scorer.py`：基于 FFT 的频率识别与 exact match 评分
- `src/jericho/models/mini_jmamba.py`：Mini-JMamba 模型实现（Mamba-2 + Attention）
- `src/jericho/pipelines/`：各任务和物理域的训练/推理流水线
- `src/jericho/data/`：Manifest 生成工具
- `train.py`：统一训练 CLI
- `evaluate.py`：Oracle/Protocol 闭环评估（系统验收）
- `evaluate_model.py`：模型能力评估（需要 checkpoint）
- `tests/`：完整测试套件（187 个用例）

</details>

<details>
<summary><strong>Manifest 格式说明</strong></summary>

- 文件格式：JSON Lines
- 字段：`split`, `symbols`, `length`, `difficulty_tag`, `example_id`, `seed`, `sequence_seed`
- 默认拆分：`train=500`, `val=100`, `iid_test=100`, `ood_length=100`, `ood_symbol=100`
- 符号与长度范围：
  - `train/val/iid_test`：符号 A–E，长度 1–8
  - `ood_length`：符号 A–E，长度 9–12
  - `ood_symbol`：符号 A–F（至少出现一次 F），长度 1–8

</details>

<details>
<summary><strong>完整训练命令参考</strong></summary>

```powershell
# Task 1: Identity baseline
python .\train.py --model identity --manifest manifests\task1.jsonl --split iid_test --outdir runs\identity_demo --limit 50

# Task 2: Bracket matching
python .\train.py --config configs\task2_bracket_stable.yaml --task bracket --model mini_jmamba --manifest manifests\task2_tiny.jsonl --split iid_test --epochs 50

# Task 3: Mod with thinking gap
python .\train.py --task mod --model mini_jmamba --manifest manifests\task3_easy.jsonl --split iid_test --limit 200 --epochs 50 --pretrain-mirror-epochs 30 --thinking-gap-s 0.5 --thinking-gap-align 160 --outdir runs\mini_jmamba_mod_week4

# Task 3: 使用配置文件
python .\train.py --config configs\task3_mod_stable.yaml --manifest manifests\task3_tiny.jsonl --split iid_test --limit 200
```

</details>

<details>
<summary><strong>Oracle Baselines</strong></summary>

```powershell
# Task 3 Mod oracle（直接输出正确答案）
python .\train.py --task mod --model oracle_mod --manifest manifests\task3.jsonl --split iid_test --outdir runs\oracle_mod_iid --limit 50
```

</details>

---

## 相关概念

本项目是 **Cross-Wave Physical Reasoning (CWPR)** 研究范式的一部分，探索在任意物理波形上进行端到端推理的可能性。

---

## 常见问题 (FAQ)

<details>
<summary><strong>采样率问题</strong></summary>

- Audio 域固定使用 16kHz 采样率
- 所有 `encode_symbols_to_wave` 调用必须使用 `sr=16000`
- 混用不同采样率会导致 FFT 解码失败

</details>

<details>
<summary><strong>随机种子</strong></summary>

- 使用 `--seed` 参数确保可复现性
- 不同 PyTorch 版本可能有轻微数值差异（< 1%）
- 跨平台（Windows/Linux）可能有浮点误差

</details>

<details>
<summary><strong>显存不足</strong></summary>

如果遇到 CUDA OOM：
- 减小 `--batch-size`（建议 4-8）
- 使用 `--limit` 减少样本数
- 尝试 `--device cpu`（慢但可用）

</details>

<details>
<summary><strong>评测结果全 0</strong></summary>

常见原因：
1. Manifest 文件路径错误
2. Split 名称拼写错误（`iid_test` 不是 `iid-test`）
3. Checkpoint 与任务不匹配

</details>

---

## 复现与最优配置

本仓库提供的配置文件是**基础配置**，可以验证系统正常运行并获得合理结果。

> ⚠️ **注意**：由于文件较大，demo checkpoints 和音频示例未包含在本仓库中。请使用 `train.py` 自行训练生成 checkpoint。

如果你需要：
- 📊 论文中报告的最优超参数
- 🔬 更多实验细节和消融结果
- 🤝 合作或交流

请通过以下方式联系我：
- 📧 Email: 928112278@qq.com
- 💬 GitHub Issues: 欢迎提问

---

## 引用

如果你使用了这个项目，请引用：

```
@misc{jericho2025,
  author = {王柏毅},
  title = {Jericho: End-to-End Reasoning on Raw Physical Waveforms},
  year = {2025},
  url = {https://github.com/Asukamnt/Project-Resonance}
}
```

---

## 许可证

MIT License
