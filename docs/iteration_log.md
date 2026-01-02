# Iteration Log（改动方向记录）

本文件用于记录 Jericho 第一篇论文的核心开发进度。

> **注意**：探索性实验（睡眠机制、图灵机模拟、XOR分析等）已移至 `docs/私人计划/research_explorations.md`

---

## Week 1（Stage A / Task1）— 符号音频编码与 scorer 闭环
- **目标**：打通 Task1 的 symbol→audio→symbol，可测可复现。
- **改动方向**：用 tone（随机相位正弦）编码符号；FFT 最近频率解码；Exact Match 评测。
- **改动落点**：`symbols.py` / `scorer.py` / `tests/test_task1_roundtrip.py`
- **验收/结果**：`pytest -q` 全绿，roundtrip 100%。
- **结论与下一步**：具备可测地基，为 manifests/批量评估/baseline 做准备。

## Week 2（Stage A / Task1）— manifests + evaluate + identity baseline
- **目标**：S3/S19/S20：可复现 manifest（IID/OOD），批量评估与统一产物格式；identity baseline 作为 sanity check。
- **改动方向**：用 JSONL manifest + seed 复现；train/evaluate 统一输出 `preds.jsonl` + `metrics.json`。
- **改动落点**：`data/make_manifest.py`、`data/utils.py`、`evaluate.py`、`baselines/identity.py`、`train.py`、对应 tests。
- **验收/结果**：identity EM≈1.0；全 split 评估可跑；pytest 全绿。
- **结论与下一步**：Stage A 完成；开始集成可训练模型（Mini-JMamba）。

## Week 3（Stage B / S9）— Mini-JMamba 可训练闭环（音频帧 + CTC）
- **目标**：S9：12 层（10×SSM + 2×Attention）Mini-JMamba；接入训练/评估闭环。
- **改动方向**：
  - 输入改为 10ms 帧（160 samples）序列；输出重建帧；
  - 增加 `symbol_logits` + `CTCLoss` 辅助监督（训练用，不作为推理中间表征）；
  - 可选 `mamba_ssm` 后端（可用则启用，不可用自动回退）。
- **改动落点**：`models/mini_jmamba.py`、`pipelines/mini_jmamba_audio.py`、`train.py`、`tests/test_mini_jmamba_smoke.py`
- **验收/结果**：Task1 `iid_test/ood_length` 可训练到 EM=1.0；pytest 全绿。
- **结论与下一步**：Task1 端到端训练稳定；进入 Week4 引入 Task3。

## Week 4（Task3 / Mod）— 数据+基线闭环 → 训练启发式迭代
- **目标**：引入 Task3（A%B 单步），建立数据、oracle baseline 与可训练闭环，推进 EM 超过弱基线。
- **改动方向（先完成闭环）**：
  - 扩展符号频率表：digits 0-9、`%`，并预留括号 `(` `)` 频率位；
  - Task3 解析与余数目标生成；Task3 manifest（train/val/iid_test/ood_digits，B≠0）；
  - oracle baseline：decode→compute→encode，验证数据/评分器正确性。
- **改动落点**：`symbols.py`、`task3/utils.py`、`data/make_task3_manifest.py`、`baselines/oracle_mod.py`、`train.py`、tests。
- **验收/结果**：oracle_mod 在 IID EM=1.0；pytest 全绿。

### Week4 训练启发式（关键改进）

9) **🎉 突破：RemainderHead (attention pooling) + 移除 detach() + epochs 100**
   - **动机**：`remainder_acc_eval` 仅 ~0.2（接近随机猜测），说明模型"会读题、会写答案、但不会算"。
   - **落点**：新增 `RemainderHead` 类（attention pooling + 3 层 MLP）；移除所有 `.detach()` 调用
   - **验收/结果**：em_post: 0.18 → 0.29 ✅
   - **结论与下一步**：Task3 (Mod) 里程碑达成（EM=0.29 > baseline+0.15）。进入 Task2 (Bracket)。

---

## 当前状态快照

### 🚀 进度概览

| 任务 | 状态 | 核心指标 | 目标达成 |
|------|------|----------|----------|
| Task1 (Mirror) | ✅ 完成 | EM = 1.0 | ✅ >95% |
| Task2 (Bracket) IID | ✅ 达标 | audio_acc = **0.96** | ✅ +46% over baseline |
| Task2 (Bracket) OOD-length | ✅ 突破 | audio_acc = **0.84** | ✅ +34% over baseline |
| Task2 (Bracket) OOD-noise | ✅ 突破 | audio_acc = **0.97** | ✅ +47% over baseline |
| Task3 (Mod) | ✅ **重大突破** | **EM = 0.45** (disjoint holdout) | ✅ 远超 baseline 🚀 |

---

## Week 5（Task2 / Bracket）— OOD 泛化突破 🎉

16) **🔥 关键 Bug 发现：tile 导致 X 频率失真**
   - **根因**：
     - V (1900 Hz)：160 样本 ≈ 19.0 个完整周期，重复时相位连续 ✓
     - X (1950 Hz)：160 样本 ≈ 19.5 个周期（不完整），重复时相位不连续 ✗
   - **落点**：修改 `apply_cls_guidance_to_frames`：直接生成完整长度的连续正弦波
   - **结果**：
     ```
     # IID: audio_acc: 0.50 → 0.96 (+46%)
     # OOD: audio_acc: 0.50 → 0.84 (+34%)
     ```

- **结论与下一步**：
  - ✅ **Task2 IID + OOD 双达标**！
  - 技术突破：RoPE + cls_guidance + 连续波形生成

---

## Phase 2（IPD 光域）— Task2 Bracket Optical (2025-12-31)

### Phase2 P0 Gate 最终状态

- **可审计结论**：
  - ✅ IID：3/3 seeds 显著超过 majority baseline（76.5%~93.5% vs 52%）
  - ⚠️ OOD：1/3 seeds 超过 57%（seed 456 = 58%），2/3 seeds 略低（52.5%, 53%）
  - ✅ OOD 不全 V：3/3 seeds 满足
- **决策**：软通过，推进 Phase 3

---

## Phase 3（跨域：Light → Sound）

### Phase3 D2-D4: 跨域模型训练 — P0 Gate 完全通过！🎉 (2025-12-31)

| Seed | IID EM | OOD EM | IID ≥70% | OOD ≥65% |
|------|--------|--------|----------|----------|
| 42 | **97%** | **65%** | ✅ | ✅ |
| 123 | **100%** | **67%** | ✅ | ✅ |
| 456 | **99%** | **70%** | ✅ | ✅ |
| **Mean** | **98.7%** | **67.3%** | **3/3** | **3/3** |

**P0 Gate 状态**：**完全通过** ✅

**关键结论**：🚀 **跨物理域推理验证成功**

---

## Phase 4: Few-shot 跨域迁移验证 (2025-12-31)

### 结果

| Seed | Transfer 最终 | Scratch 最终 | Diff |
|------|--------------|--------------|------|
| 42   | 96%          | 94%          | +2pp |
| 123  | 99%          | 92%          | +7pp |
| 456  | 99%          | 98%          | +1pp |
| **Mean** | **98.0%**| **94.7%**    | **+3.3pp** |

### 10-Seed Bootstrap CI ✅

**Bootstrap CI (n=1000, 95%)**:
- **Mean ΔEM: +1.70 pp**
- **95% CI: [+0.10, +3.40] pp**
- **CI 不含 0 → ✅ 效果统计显著**

**Phase 4 P0 Gate: ✅ PASSED**

### 三角验证矩阵（最终状态）

```
       Audio  IPD   RF
Audio   [D]   ✅    ✅
IPD     ✅    [D]   ✅
RF      ✅    ✅    [D]

跨域迁移: 6/6 (100%) ✅ COMPLETE
```

---

## D8: wav2vec2 Baseline 实验 (2025-12-31)

### 结果 (3 seeds)

| Model | Setting | Params | IID EM |
|-------|---------|--------|--------|
| wav2vec2-base | frozen | 97.3M | **13%** ± 0% |
| wav2vec2-base | partial | 97.3M | **13%** ± 0% |
| wav2vec2-base | full | 97.3M | **13%** ± 0% |
| Mini-JMamba | — | 0.94M | **45%** ± 2% |

**结论**：通用语音预训练（wav2vec2）不适合波形推理任务。Mini-JMamba +32pp，100× 参数效率。

---

## D10 P2 Ablation Studies（2026-01-01）

### Thinking Gap Ablation ✅

| Gap (s) | Audio EM | CTC EM |
|---------|----------|--------|
| 0.0 | 9.50% | 45.00% |
| **0.5** | 9.00% | **48.25%** |
| 2.0 | 10.75% | 44.75% |

**推荐值：0.5s**

### Architecture Ablation ✅

| 架构 | Audio EM | CTC EM |
|------|----------|--------|
| **Mini-JMamba (10 SSM + 2 Attn)** | 10.5% | **45.5%** |
| Pure SSM (12 SSM + 0 Attn) | 2.25% | 0.0% ❌ |

**结论**：Pure SSM 完全失败（CTC EM=0%）— Attention 层对符号解码至关重要

### Channel Noise Robustness ✅

| 扰动 | 相对 Clean 变化 |
|------|----------------|
| AWGN 5-30dB | 无变化 |
| 相位偏移 | 无变化 |
| **结论** | 对 AWGN 极其鲁棒 |

---

## D12: OOD-Length 崩溃分析 (2026-01-02)

| Split | 输出维度 | Model EM |
|-------|----------|----------|
| iid_test | 100% 单位数 | 40.0% ± 5.0% |
| ood_digits | 100% 单位数 | 39.7% ± 2.1% |
| ood_length | 77.5% 双位数 | 2.7% ± 0.3% |

**根因**：崩溃主因是**输出维度外推**（1→2 位数），而非输入长度增加。

---

## D21 论文改进实验

### P1: 现代基线对比

| 模型 | 参数量 | Val Accuracy (30 epochs) |
|------|--------|--------------------------|
| Transformer | 1.23M | 19% |
| LSTM | 0.64M | 21% |
| S4 (simplified) | 0.54M | 31% |
| Hyena | 0.84M | 40% |
| **Mini-JMamba** | **0.94M** | **93.3%** |

**结论**：Mini-JMamba 比最佳基线 Hyena 高出 **53.3 个百分点**

### P2: 资源效率对比

| 模型 | 参数量 | 延迟 (ms) | 吞吐量 (batch=8) |
|------|--------|-----------|------------------|
| Mini-JMamba | 0.94M | 24.25 | 177.9 sps |
| Transformer | 1.23M | 17.78 | 163.1 sps |
| LSTM | 0.64M | 157.28 | 30.9 sps |

---

## D22-D24: 真实语音验证（Google Speech Commands）

### 实验设置

| 配置 | 值 |
|------|-----|
| 数据集 | Google Speech Commands v0.02（数字 0-9） |
| 训练样本 | 17,500 |
| 验证样本 | 3,750 |
| 测试样本 | 3,750 |

### 结果（3-seed 统计）🎉

| Seed | Val Acc | Test Acc |
|------|---------|----------|
| 42 | 92.3% | 91.84% |
| 123 | 92.9% | 91.76% |
| 456 | 92.3% | 91.39% |
| **Mean ± Std** | **92.5%** | **91.68% ± 0.25%** |

**关键发现**：
1. **真实人声验证成功**：91.7% >> 10%（随机猜测）
2. **证明迁移能力**：合成数据训练的架构在真实数据上有效
3. **解决论文硬伤**：不再是"仅合成数据"

---

## 当前论文状态总结 (2026-01-02)

### 核心贡献

| 贡献 | 证据 | 状态 |
|------|------|------|
| **1. 波形推理可行性** | Task3 Mod: Mini-JMamba 45% vs wav2vec2 13% | ✅ |
| **2. 跨域推理** | IPD→Audio IID 98.7% | ✅ |
| **3. 跨域迁移** | +1.7pp (95% CI, 10-seed) | ✅ |
| **4. 三角验证** | Audio↔IPD↔RF 6/6 跨域边 | ✅ |
| **5. 真实人声验证** | Google Speech Commands **91.7%** 🚀 | ✅ |

### 测试覆盖

- **总测试数**：187 个
- **状态**：全部通过 ✅

---

*最后更新：2026-01-02*
