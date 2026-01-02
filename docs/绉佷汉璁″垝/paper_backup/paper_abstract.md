# Abstract 初稿

---

## Version 1 (English, 150 words) — NeurIPS 2025 Target

Can neural networks reason without tokens? We show they can—directly on physical waveforms.

**Jericho** performs end-to-end inference across Audio, Optical, and RF domains using Mini-JMamba, a 0.94M-parameter SSM-Attention hybrid. Key results:

1. **Single-domain reasoning**: 45% EM on modular arithmetic, vs 13% for wav2vec2 (97M params, full fine-tuning)
2. **Cross-domain reasoning**: 98.7% IID accuracy for IPD→Audio inference
3. **Cross-domain transfer**: 9/9 triangular validation edges verified, +1.7pp gain (p<0.05)
4. **Robustness**: 100% EM at 0 dB SNR with simulated noise + reverberation

Importantly, pretrained speech models (wav2vec2) achieve only 13% even with full fine-tuning—barely above chance. This demonstrates that **task-specialized architectures are essential** for waveform reasoning; general pretraining does not transfer.

Our work opens a new paradigm: **modality-agnostic reasoning systems** that operate natively in physical signal spaces, bypassing the token bottleneck.

---

## Version 2 (中文, 180 字) — NeurIPS 2025

神经网络能否不依赖 token 进行推理？答案是肯定的——直接在物理波形上。

**Jericho** 使用 Mini-JMamba（0.94M 参数）实现 Audio/Optical/RF 三域端到端推理：

1. **单域推理**：取模任务 EM=45%，wav2vec2 全参数微调仅 13%（97M 参数）
2. **跨域推理**：IPD→Audio 达 98.7% IID 准确率
3. **跨域迁移**：三角验证 9/9 全通过，+1.7pp（p<0.05）
4. **鲁棒性**：0 dB SNR + 混响仍达 100% EM

关键发现：**通用语音预训练无法迁移到波形推理**——wav2vec2 即使全量微调也只有 13%，接近随机猜测。这证明任务特化架构的必要性。

我们的工作开启新范式：**在物理信号空间原生运行的模态无关推理系统**。

---

## Version 3 (Punchy, 100 words)

Can neural networks reason without tokens? Yes—directly on raw waveforms.

**Jericho** + Mini-JMamba (0.94M params) achieves:
- 45% EM on audio modular arithmetic (vs 13% for wav2vec2 with 97M params)
- 98.7% cross-domain accuracy (IPD→Audio)
- 9/9 triangular validation, +1.7pp transfer gain (p<0.05)
- 100% EM at 0 dB SNR (noise robustness)

Key insight: **General speech pretraining doesn't transfer to waveform reasoning.** wav2vec2 achieves only 13% even with full fine-tuning—barely above chance. Task-specialized architectures are essential.

This opens a new paradigm: modality-agnostic reasoning in physical signal space.

---

## Title Candidates

| Style | Title |
|-------|-------|
| **Discovery** | "Reasoning Without Symbols: Emergent Dynamics in Waveform-Native Neural Networks" |
| **Paradigm** | "From Tokens to Trajectories: End-to-End Reasoning on Physical Waveforms" |
| **Punchy** | "Jericho: Neural Reasoning Directly on Raw Waveforms" |
| **Biological** | "Internal Time Base and Cross-Domain Transfer in Waveform Reasoning" |

**Recommended**: "Reasoning Without Symbols" — 直接点明核心 insight

