# Jericho: End-to-End Reasoning on Raw Audio Waveforms

<p align="center">
  <strong>无需文本中间表示，直接在音频波形上完成符号推理</strong>
</p>

---

## 这是什么？

**Jericho** 是一个实验性框架，验证一个核心假设：

> **神经网络可以直接在连续音频波形上完成符号推理任务，全程不经过离散化的文本/token 中间表示。**

传统的语音理解流程是：`音频 → ASR → 文本 → LLM → 文本 → TTS → 音频`

Jericho 的流程是：`音频 → 神经网络 → 音频`

我们设计了三个递进难度的任务来验证这个假设：

| 任务 | 输入 | 输出 | 验证的能力 |
|------|------|------|-----------|
| **Task 1: Mirror** | 符号序列的音频 | 相同的符号序列音频 | 音频编解码闭环 |
| **Task 2: Bracket** | 括号表达式音频 | 括号匹配结果音频 | 结构推理 |
| **Task 3: Mod** | 数学表达式音频 | 取模运算结果音频 | 算术推理 |

---

## 为什么这很重要？

1. **信息保真度**：离散化（tokenization）会丢失波形中的相位、时序微结构等信息。直接在波形上推理可能保留更多信息。

2. **延迟与流式处理**：不需要等待完整的 token 序列，可以做因果/流式推理。

3. **跨波域泛化**：如果模型能在音频波形上推理，理论上同样的架构可以扩展到其他物理波形（RF、振动、光学信号）。

---

## 核心组件

- **Mini-JMamba**：12 层 SSM/Attention 混合架构，直接处理原始波形
- **符号-音频编码器**：将离散符号映射为正弦波音频
- **Scorer 解码器**：基于 FFT 的频率识别，用于评估
- **Manifest 系统**：可复现的数据生成与拆分
- **闭环评测**：从 manifest 到合成、推理、解码、Exact Match 的完整流水线

---

## 当前状态

- ✅ Task 1 (Mirror)：EM = 1.00 (IID)
- ✅ Task 2 (Bracket)：audio_acc = 0.96 (IID), 0.84 (OOD-length), 0.97 (OOD-noise)
- ✅ Task 3 (Mod)：EM = 0.315 (超过 baseline +0.19)
- ✅ Mini-JMamba 模型集成（RoPE 位置编码）
- ✅ 完整的训练/评估流水线
- ✅ 多轴 OOD 评测（length, noise）
- ✅ 119 个测试用例全部通过

---

## 实验结果

| 任务 | IID | OOD-length | OOD-noise | Baseline |
|------|-----|------------|-----------|----------|
| Task 1 (Mirror) | 1.00 | 1.00 | - | - |
| Task 2 (Bracket) | 0.96 | 0.84 | 0.97 | 0.50 |
| Task 3 (Mod) | 0.315 | - | - | 0.125 |

---

## 快速开始

### 环境配置（Windows PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest -q
```

### 运行示例

```powershell
# 生成 Task3 (Mod) manifest
python -m jericho.data.make_task3_manifest --out manifests/task3_tiny.jsonl --seed 321 --preset tiny --balance-remainder

# 训练 Mini-JMamba
python .\train.py --config configs\task3_mod_stable.yaml --manifest manifests\task3_tiny.jsonl --split iid_test --limit 200

# Oracle/Protocol 闭环验证（不是模型能力；详见 docs/metrics_protocol.md）
python .\evaluate.py --stage final --tasks mirror bracket mod

# 训练模型能力（Model EM）：用 checkpoint 输出 Oracle vs Model 对比总表
python .\evaluate_model.py --checkpoint artifacts\checkpoints\mirror_demo_seed42_epoch15.pt --tasks mirror bracket mod --splits iid_test ood_length --limit 50 --device cpu
```

---

## 详细文档

<details>
<summary><strong>目录结构</strong></summary>

- `src/jericho/symbols.py`：符号表、频率映射与正弦音频合成
- `src/jericho/scorer.py`：基于 FFT 的频率识别与 exact match 评分
- `src/jericho/models/mini_jmamba.py`：Mini-JMamba 模型实现
- `src/jericho/pipelines/`：各任务的训练/推理流水线
- `src/jericho/data/`：Manifest 生成工具
- `train.py`：统一训练 CLI
- `evaluate.py`：Oracle/Protocol 闭环评估（系统验收）
- `evaluate_model.py`：Oracle vs Model 对比总表（模型能力验收）
- `docs/metrics_protocol.md`：评测口径协议（Oracle EM vs Model EM）
- `tests/`：完整测试套件

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

## 发布 Artifacts

| 文件 | 说明 |
|------|------|
| `artifacts/checkpoints/mirror_demo_seed42_epoch15.pt` | Mirror 任务 demo checkpoint，**Model EM = 1.0** (IID & OOD-length) |
| `artifacts/checkpoints/mod_demo_seed42_epoch20.pt` | Mod 任务 demo checkpoint（概念示例，训练不充分） |
| `artifacts/audio_examples/` | 60 个 WAV 示例（input/target/output 三元组） |

> **注意**：`mirror_demo` 是完整训练的能力证明；`mod_demo` 仅作为训练流程示例，需更长训练时间才能达到论文级 Model EM。如需完整训练结果，请参考"复现与最优配置"节。

---

## 复现与最优配置

本仓库提供的配置文件是**基础配置**，可以验证系统正常运行并获得合理结果。

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
@misc{jericho2024,
  author = {王柏毅},
  title = {Jericho: End-to-End Reasoning on Raw Audio Waveforms},
  year = {2025},
  url = {https://github.com/Asukamnt/Project-Resonance}
}
```

---

## 许可证

MIT License
