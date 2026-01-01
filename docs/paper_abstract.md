# Abstract 初稿

---

## Version 1 (English, 180 words)

Current neural reasoning systems rely on discrete symbolic representations (text, tokens), creating a "modality bottleneck" that loses continuous signal information. We ask: **can neural networks reason directly on raw physical waveforms, without tokenization?**

We introduce **Jericho**, a framework for end-to-end waveform reasoning across physical domains. Using Mini-JMamba, a lightweight SSM-Attention hybrid (0.94M parameters), we demonstrate:

1. **Single-domain reasoning**: 45% exact match on arithmetic modulo tasks, outperforming Transformer (41%) and wav2vec2 (22%)
2. **Cross-domain reasoning**: 98.7% IID accuracy for IPD→Audio inference
3. **Cross-domain transfer**: Statistically significant gains (+1.7pp, p<0.05) across Audio↔IPD↔RF triangular validation

Notably, our models exhibit **biologically reminiscent dynamics**: an internal time base revealed by frequency-dependent stretch asymmetry (TSAE), time-scale separation effects, and a division of labor between continuous state evolution (SSM) and discrete binding (Attention). 

We do not claim biological equivalence; rather, these phenomena provide **testable signatures** of emergent internal dynamics beyond token-level processing, opening new paths toward biologically-grounded artificial intelligence.

---

## Version 2 (中文, 200 字)

当前神经推理系统依赖离散符号表示（文本/token），形成"模态瓶颈"，丢失连续信号信息。我们提出核心问题：**神经网络能否直接在原始物理波形上推理？**

我们提出 **Jericho**——一个跨物理域的端到端波形推理框架。使用 Mini-JMamba（0.94M 参数的 SSM-Attention 混合架构），我们验证了：

1. **单域推理**：取模任务 EM=45%，优于 Transformer（41%）和 wav2vec2（22%）
2. **跨域推理**：IPD→Audio 达到 98.7% IID 准确率
3. **跨域迁移**：Audio↔IPD↔RF 三角验证 6/6 有效，+1.7pp（p<0.05）

关键发现：模型自然呈现出**类生物动力学特征**——频率依赖的时间尺度不对称效应（TSAE）、时间常数分离（Thinking Gap）、以及连续承载+离散绑定的分工（SSM+Attention）。

这些现象为"从底层演化类人智能"提供了**可证伪的测试路径**。

---

## Version 3 (Punchy, 120 words)

Can neural networks reason without tokens? We show they can—directly on physical waveforms.

**Jericho** performs end-to-end inference across Audio, Optical, and RF domains using Mini-JMamba, a 0.94M-parameter SSM-Attention hybrid. Key results:
- Single-domain: 45% EM (vs 22% wav2vec2)  
- Cross-domain: 98.7% IID accuracy
- Transfer: +1.7pp gain (p<0.05), 6/6 triangular validation

Unexpectedly, our models exhibit **biologically reminiscent dynamics**: internal time base (TSAE), time-scale separation, and continuous-discrete binding. We don't claim biological equivalence—but these phenomena are **testable signatures** of emergent dynamics beyond token processing.

Waveform reasoning isn't just possible; it reveals new paths toward biologically-grounded AI.

---

## Title Candidates

| Style | Title |
|-------|-------|
| **Discovery** | "Reasoning Without Symbols: Emergent Dynamics in Waveform-Native Neural Networks" |
| **Paradigm** | "From Tokens to Trajectories: End-to-End Reasoning on Physical Waveforms" |
| **Punchy** | "Jericho: Neural Reasoning Directly on Raw Waveforms" |
| **Biological** | "Internal Time Base and Cross-Domain Transfer in Waveform Reasoning" |

**Recommended**: "Reasoning Without Symbols" — 直接点明核心 insight

