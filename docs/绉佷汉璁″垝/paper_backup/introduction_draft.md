# Introduction 完整草稿

*本文档是论文 Introduction 的扩展版本，用于迭代和审阅*

---

## 1. Introduction

### 1.1 The Modality Bottleneck

Human cognition operates seamlessly across modalities. We can hear a mathematical expression spoken aloud and mentally compute its result; we can see a musical score and "hear" the melody before any sound is produced. This cross-modal fluency suggests that the brain may employ representations that abstract away from specific sensory carriers.

Current AI systems, by contrast, rely heavily on symbolic intermediaries. A voice assistant processing a spoken arithmetic question must:
1. Convert speech to text (ASR)
2. Parse text into symbolic representation
3. Perform reasoning on symbols
4. Generate response text
5. Synthesize text back to speech (TTS)

This pipeline creates what we term the **modality bottleneck**:

- **Information loss**: Sub-symbolic information—prosody, timing variations, spectral texture—is discarded at the transcription stage. A whispered "two" and a shouted "TWO" become identical tokens, despite potentially carrying different pragmatic meanings.

- **Latency accumulation**: Each encoding/decoding stage introduces processing delay. For real-time applications, this cumulative latency can exceed acceptable thresholds.

- **Domain fragmentation**: Each modality requires its own pretrained encoder (wav2vec2 for audio, ViT for images, etc.), creating engineering complexity and preventing unified reasoning across modalities.

### 1.2 A Fundamental Question

We ask: **Can neural networks reason directly in the physical waveform domain?**

This question challenges a deep assumption in neural reasoning research—that discrete symbolic representations are necessary for logical operations. If a model can accept raw waveforms as input, perform multi-step reasoning, and produce raw waveforms as output, we would have evidence that:

1. Symbolic tokenization is not a prerequisite for reasoning
2. The "structure" underlying reasoning may be carrier-independent
3. End-to-end signal processing could replace multi-stage pipelines

### 1.3 Two Hypotheses

We decompose this question into two testable hypotheses:

**H1 (Waveform Reasoning)**: A neural network can perform logical operations on information encoded as physical signals, producing output in signal form—without any intermediate symbolic representation.

**H2 (Carrier-Agnostic Representation)**: The representations learned for reasoning in one physical domain (e.g., audio frequency modulation) can transfer to reasoning in a different physical domain (e.g., optical pulse position modulation), suggesting that the model learns abstract structural patterns rather than domain-specific signal features.

### 1.4 Approach: Wave Reasoning

We introduce **Wave Reasoning**, a framework where:
- Input is a raw physical waveform $x \in \mathbb{R}^T$ encoding a reasoning problem
- Output is a raw physical waveform $\hat{y} \in \mathbb{R}^T$ encoding the answer
- The model $f_\theta$ operates entirely in the continuous signal domain
- Symbols are used only for data generation and evaluation, never as intermediate representations

To validate this framework, we design reasoning tasks across three physically distinct domains:

| Domain | Physical Carrier | Modulation | Example |
|--------|-----------------|------------|---------|
| **Audio** | Sound pressure | Frequency | 440 Hz → "A", 880 Hz → "B" |
| **Optical (IPD)** | Light intensity | Pulse position | Slot [2,7] → "3" |
| **RF** | Electromagnetic | Amplitude | 0.5V → "low", 1.0V → "high" |

These domains differ in sample rate (1 kHz – 1 MHz), encoding scheme (frequency/position/amplitude), and physical interpretation. If a model learns representations that transfer across them, we have strong evidence for H2.

### 1.5 Key Insight: Symbols as Interface, Not Substrate

A clarification is essential: our framework does use symbols—but only at the system boundary. Symbols define the encoding (which frequency means which digit) and the evaluation metric (did the decoded output match the target). However, the model's internal processing—the "reasoning"—occurs entirely in continuous hidden states.

This is analogous to how the visual cortex processes continuous pixel intensities, even though we later interpret the output as discrete object categories. The question is not whether symbols exist in the world, but whether the neural computation requires symbolic intermediates.

### 1.6 An Unexpected Discovery: Time-Scale Asymmetry Effect

During robustness testing, we discovered a phenomenon we term the **Time-Scale Asymmetry Effect (TSAE)**:

- When input waveforms are stretched by 1.05× (slowed down 5%), model accuracy *increases* by 30 percentage points for Task3 (modular arithmetic)
- When stretched by 0.95× (sped up 5%), accuracy *decreases*
- This asymmetry reverses for Task2 (bracket matching): 0.95× improves performance

Further analysis reveals that this effect corresponds to "frequency calibration"—the optimal time stretch minimizes systematic frequency deviation in the model's output. This suggests the model develops an internal time base, reminiscent of biological temporal processing.

### 1.7 Contributions

1. **Feasibility of waveform reasoning**: We demonstrate that a neural network can achieve 45% Exact Match on modular arithmetic (A % B → remainder) operating entirely on audio waveforms, outperforming wav2vec2 fine-tuning (22%) with 100× fewer parameters.

2. **Cross-domain reasoning**: We show that reasoning can occur across physically distinct domains—a model trained on optical-domain bracket matching achieves 98.7% IID accuracy when producing audio-domain outputs.

3. **Statistical evidence for carrier-agnostic representations**: Through 10-seed experiments with bootstrap confidence intervals, we demonstrate that pre-training on audio improves optical-domain performance by +1.7 percentage points (95% CI: [+0.1, +3.4], p < 0.05).

4. **Triangular validation matrix**: We test all 6 pairwise transfers across three physical domains (Audio ↔ IPD ↔ RF), providing systematic evidence for H2.

5. **Discovery of biological-like temporal dynamics**: The Time-Scale Asymmetry Effect reveals that the model develops task-dependent internal time bases, suggesting emergent temporal calibration mechanisms.

6. **Open benchmark and reproducibility**: We release code, deterministic data generation, and 191 passing tests. All experiments run on a single RTX 4070 (8GB).

### 1.8 Paper Organization

- **Section 2 (Method)**: Problem formulation, physical domains, Mini-JMamba architecture
- **Section 3 (Experiments)**: Single-domain reasoning, cross-domain reasoning, transfer learning
- **Section 4 (Analysis)**: Architecture ablations, TSAE investigation, linear probe analysis
- **Section 5 (Related Work)**: Audio understanding, neural reasoning, cross-modal learning
- **Section 6 (Limitations)**: Synthetic data, task complexity, OOD generalization
- **Section 7 (Conclusion)**: Summary and future directions

---

## 审阅要点

### 这版 Introduction 的优点
- 更清晰的动机（modality bottleneck 三个子问题）
- 明确区分 H1 和 H2
- 解释"符号只是接口"这个容易被误解的点
- 包含 TSAE 发现作为亮点
- 贡献列表更完整

### 可能需要改进
- 太长？可能需要压缩到 1.5 页
- TSAE 放在 Introduction 可能太早，考虑移到 Analysis
- "biological-like" 可能引起争议，需要谨慎措辞
- 需要更多对 related work 的简短引用

### 待讨论
- 是否保留"无符号推理"这个标题？还是改成更学术的表述？
- H1/H2 框架是否太简单？是否需要更多子假设？
- 是否需要在 Introduction 就预告负面结果（如某些迁移方向无效）？

---

*草稿版本 v1 - 2026-01-01*

