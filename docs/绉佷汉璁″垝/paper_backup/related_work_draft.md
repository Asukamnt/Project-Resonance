# Related Work 完整草稿

*本文档是论文 Related Work 的扩展版本*

---

## 5. Related Work

Our work sits at the intersection of several research areas: audio/speech understanding, neural reasoning, cross-modal learning, and efficient sequence modeling. We discuss each in turn.

### 5.1 Audio and Speech Understanding

**Self-supervised speech models.** wav2vec 2.0 [1] and HuBERT [2] learn powerful speech representations through contrastive or masked prediction objectives on large unlabeled corpora. Whisper [3] demonstrates remarkable robustness through weakly-supervised training on 680K hours of labeled data. These models achieve state-of-the-art results on speech recognition and related tasks.

However, these approaches are fundamentally **recognition-oriented**: they extract discrete linguistic units (phonemes, words) from continuous signals. Our work asks the converse question—can models perform **reasoning** directly on continuous signals without extracting discrete intermediates?

**Audio generation.** AudioLM [4] and MusicLM [5] generate realistic audio by operating on discrete audio tokens (e.g., from SoundStream or w2v-BERT). While impressive, this approach still requires discretization. Recent work on continuous audio diffusion [6] operates more directly on waveforms, but focuses on generation quality rather than reasoning.

**Key distinction**: We require models to both *understand* and *produce* continuous waveforms, with logical computation in between—a capability not addressed by recognition or generation models alone.

### 5.2 Neural Reasoning and Arithmetic

**Symbolic reasoning with neural networks.** Neural networks notoriously struggle with systematic reasoning [7, 8]. Chain-of-thought prompting [9] and scratchpad methods [10] improve reasoning by generating intermediate steps, but these operate entirely in discrete token space.

**Neural arithmetic.** Learning arithmetic operations remains challenging for neural networks [11, 12]. The Neural Arithmetic Logic Unit (NALU) [13] and its successors [14] propose specialized architectures for arithmetic, but assume symbolic inputs. Recent work shows transformers can learn modular arithmetic with sufficient scale [15], again in token space.

**Key distinction**: We encode arithmetic problems as *physical waveforms* and require waveform outputs. This tests whether reasoning can occur in continuous representations, not whether networks can manipulate symbols.

### 5.3 Cross-Modal and Multi-Modal Learning

**Representation alignment.** CLIP [16] and ALIGN [17] learn joint vision-language representations through contrastive learning. ImageBind [18] extends this to six modalities by binding to images. These approaches align representations across modalities but do not transfer *reasoning capabilities*.

**Multi-modal reasoning.** Flamingo [19], GPT-4V [20], and Gemini [21] perform impressive multi-modal reasoning, but convert all inputs to a shared token space before reasoning. The reasoning itself still occurs in discrete representations.

**Cross-domain transfer.** Domain adaptation [22] and transfer learning [23] typically operate within related domains (e.g., news → social media text). Transfer across physically distinct signal domains (audio → optical → RF) is less explored.

**Key distinction**: We test whether *reasoning representations*—not just perceptual features—transfer across physically distinct carriers. A model pre-trained on audio-domain bracket matching should help with optical-domain bracket matching if it learns carrier-agnostic structural representations.

### 5.4 State Space Models and Efficient Sequence Modeling

**Linear-time sequence models.** Transformers' O(T²) attention limits scalability for long sequences. State space models (S4 [24], Mamba [25]) achieve O(T) complexity while maintaining long-range dependencies. These models excel at audio, DNA, and time-series tasks.

**Hybrid architectures.** Combining SSM and attention layers [26, 27] can capture both efficient long-range modeling and explicit cross-position interactions. Our Mini-JMamba uses this hybrid design: SSM layers for temporal modeling, attention layers for symbol alignment.

**Key distinction**: We apply SSM-attention hybrids to a novel task—cross-domain waveform reasoning—demonstrating their utility beyond standard sequence modeling benchmarks.

### 5.5 Continuous vs. Discrete Representations in Cognition

**Cognitive science perspective.** The debate between symbolic [28] and subsymbolic [29] representations in cognition remains active. Connectionist models [30] suggest that discrete-seeming behavior can emerge from continuous dynamics.

**Neural coding.** Biological neural systems process continuous signals (membrane potentials, firing rates) yet support discrete-seeming cognition [31]. This suggests that symbolic reasoning may not require symbolic substrates.

**Key distinction**: Our work provides empirical evidence for this hypothesis in artificial systems—models can perform logical operations on continuous waveforms without explicit symbolic intermediaries.

### 5.6 Positioning Our Work

| Aspect | Prior Work | Our Work |
|--------|-----------|----------|
| **Input** | Discrete tokens or extracted features | Raw physical waveforms |
| **Output** | Discrete tokens or class labels | Raw physical waveforms |
| **Reasoning** | In token space | In continuous hidden states |
| **Transfer** | Within modality or aligned spaces | Across physically distinct domains |
| **Goal** | Recognition or generation | End-to-end reasoning |

To our knowledge, we are the first to:
1. Demonstrate logical reasoning entirely in the waveform domain
2. Test reasoning transfer across physically distinct signal modalities
3. Provide statistical evidence for carrier-agnostic reasoning representations

---

## References (to expand)

[1] Baevski et al. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS 2020.

[2] Hsu et al. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. TASLP 2021.

[3] Radford et al. Robust Speech Recognition via Large-Scale Weak Supervision. arXiv 2022.

[4] Borsos et al. AudioLM: a Language Modeling Approach to Audio Generation. arXiv 2022.

[5] Agostinelli et al. MusicLM: Generating Music From Text. arXiv 2023.

[6] Kong et al. DiffWave: A Versatile Diffusion Model for Audio Synthesis. ICLR 2021.

[7] Lake & Baroni. Generalization without Systematicity. ICML 2018.

[8] Kim & Linzen. COGS: A Compositional Generalization Challenge. EMNLP 2020.

[9] Wei et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.

[10] Nye et al. Show Your Work: Scratchpads for Intermediate Computation. arXiv 2021.

[11] Saxton et al. Analysing Mathematical Reasoning Abilities of Neural Models. ICLR 2019.

[12] Nogueira et al. Investigating the Limitations of Transformers with Simple Arithmetic Tasks. arXiv 2021.

[13] Trask et al. Neural Arithmetic Logic Units. NeurIPS 2018.

[14] Madsen & Johansen. Neural Arithmetic Units. ICLR 2020.

[15] Power et al. Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv 2022.

[16] Radford et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.

[17] Jia et al. Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision. ICML 2021.

[18] Girdhar et al. ImageBind: One Embedding Space To Bind Them All. CVPR 2023.

[19] Alayrac et al. Flamingo: a Visual Language Model for Few-Shot Learning. NeurIPS 2022.

[20] OpenAI. GPT-4 Technical Report. arXiv 2023.

[21] Gemini Team. Gemini: A Family of Highly Capable Multimodal Models. arXiv 2023.

[22] Ben-David et al. A Theory of Learning from Different Domains. Machine Learning 2010.

[23] Pan & Yang. A Survey on Transfer Learning. TKDE 2010.

[24] Gu et al. Efficiently Modeling Long Sequences with Structured State Spaces. ICLR 2022.

[25] Gu & Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv 2023.

[26] Dao et al. Hungry Hungry Hippos: Towards Language Modeling with State Space Models. ICLR 2023.

[27] Lieber et al. Jamba: A Hybrid Transformer-Mamba Language Model. arXiv 2024.

[28] Fodor & Pylyshyn. Connectionism and Cognitive Architecture: A Critical Analysis. Cognition 1988.

[29] Smolensky. On the Proper Treatment of Connectionism. Behavioral and Brain Sciences 1988.

[30] Rumelhart et al. Parallel Distributed Processing. MIT Press 1986.

[31] Rieke et al. Spikes: Exploring the Neural Code. MIT Press 1999.

---

## 审阅要点

### 覆盖的领域
- ✅ 语音/音频理解（wav2vec2, HuBERT, Whisper）
- ✅ 音频生成（AudioLM, diffusion）
- ✅ 神经推理（CoT, scratchpad, NALU）
- ✅ 跨模态学习（CLIP, ImageBind, Flamingo）
- ✅ 状态空间模型（S4, Mamba）
- ✅ 认知科学视角（符号 vs 连接主义）

### 可能遗漏
- [ ] 物理信号处理/通信领域的相关工作？
- [ ] 神经符号系统（Neuro-symbolic AI）？
- [ ] 元学习/少样本学习？

### 关键定位语句
每个小节都有 "Key distinction" 来明确我们的工作与现有工作的区别。

### 表格总结
提供了清晰的对比表格，方便读者快速理解定位。

---

*草稿版本 v1 - 2026-01-01*

