# Method & Experiments 完整草稿

*本文档是论文 Method 和 Experiments 的扩展版本*

---

## 2. Method

### 2.1 Problem Formulation

We formalize **waveform reasoning** as a sequence-to-sequence problem in the continuous signal domain.

**Definition (Waveform Reasoning Task).** Given:
- Input waveform $x \in \mathbb{R}^{T_{in}}$ encoding a reasoning problem
- Target waveform $y \in \mathbb{R}^{T_{out}}$ encoding the correct answer

The model $f_\theta: \mathbb{R}^{T_{in}} \to \mathbb{R}^{T_{out}}$ must produce $\hat{y} = f_\theta(x)$ such that decoding $\hat{y}$ yields the correct symbolic answer.

**Evaluation Metric: Exact Match (EM).** We use a deterministic decoder $D: \mathbb{R}^T \to \Sigma^*$ that maps waveforms to symbol sequences. The Exact Match score is:

$$\text{EM} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[D(\hat{y}_i) = D(y_i)]$$

where $\Sigma$ is the symbol vocabulary and $D$ uses FFT-based frequency detection for audio, pulse position detection for optical, and amplitude thresholding for RF.

**Key Design Choice: Symbols as Interface.** Symbols $\Sigma$ define the encoding scheme and evaluation metric, but never appear as model inputs or outputs. The model's internal computation occurs entirely in continuous hidden states $h_t \in \mathbb{R}^d$.

### 2.2 Physical Domains

We validate across three physically distinct domains to test H2 (carrier-agnostic representations):

#### 2.2.1 Audio Domain (Frequency Modulation)

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Symbol duration | 100 ms (1,600 samples) |
| Encoding | Symbol → sine tone frequency |
| Vocabulary | 10 digits (0-9) + operators (+, -, %, etc.) |
| Frequency range | 200 Hz – 2,000 Hz |

**Encoding function:**
$$x_{\text{symbol}}(t) = \sin(2\pi f_{\text{symbol}} \cdot t + \phi)$$

where $f_{\text{symbol}}$ maps each symbol to a unique frequency, and $\phi$ is random phase (for training robustness) or fixed (for evaluation stability).

#### 2.2.2 Optical Domain (Intensity-Pulse Domain, IPD)

| Parameter | Value |
|-----------|-------|
| Sample rate | 1,000 Hz |
| Symbol duration | 100 ms (100 samples) |
| Encoding | Symbol → 2-of-10 pulse position pattern |
| Vocabulary | 10 digits (0-9) |
| Pulse width | 10 ms |

**Encoding function:** Each symbol activates exactly 2 of 10 time slots in a "2-of-10" code, providing error detection capability:

$$x_{\text{symbol}}(t) = \begin{cases} 1 & \text{if } t \in \text{slots}(s) \\ 0 & \text{otherwise} \end{cases}$$

#### 2.2.3 RF Domain (Amplitude Shift Keying)

| Parameter | Value |
|-----------|-------|
| Sample rate | 1,000,000 Hz (1 MHz) |
| Symbol duration | 1 ms (1,000 samples) |
| Encoding | Symbol → carrier amplitude level |
| Vocabulary | 10 digits (0-9) |
| Carrier frequency | 100 kHz |

**Encoding function:**
$$x_{\text{symbol}}(t) = A_{\text{symbol}} \cdot \sin(2\pi f_c \cdot t)$$

where $A_{\text{symbol}} \in \{0.1, 0.2, ..., 1.0\}$ maps digits to amplitude levels.

### 2.3 Reasoning Tasks

We design three reasoning tasks of increasing complexity:

#### Task 1: Mirror (Copy)
- **Input**: Sequence of symbols $[s_1, s_2, ..., s_n]$
- **Output**: Same sequence $[s_1, s_2, ..., s_n]$
- **Purpose**: Baseline task testing waveform encoding/decoding fidelity
- **Difficulty**: Trivial reasoning, tests signal processing

#### Task 2: Bracket Matching
- **Input**: Bracket sequence, e.g., `( ( ) ( ) )`
- **Output**: Binary `V` (valid) or `X` (invalid)
- **Purpose**: Tests structural reasoning (stack-based computation)
- **Difficulty**: Requires tracking nested structure

#### Task 3: Modular Arithmetic
- **Input**: Expression `A % B` where A, B are multi-digit numbers
- **Output**: Remainder (1-2 digits)
- **Purpose**: Tests arithmetic reasoning
- **Difficulty**: Requires multi-step computation

**Task Structure (Task 3 Example):**
```
[Expression Symbols] [Thinking Gap] [Answer Window]
   "1 7 % 5"          (silence)        "2"
```

The "thinking gap" (configurable, default 200ms) provides time for the model to compute before producing output.

### 2.4 Mini-JMamba Architecture

We use a lightweight SSM-Attention hybrid architecture optimized for waveform sequence modeling.

#### Architecture Overview

```
Raw Waveform (T samples)
        ↓
┌─────────────────────────┐
│   Frame Embedding       │  frame_size=160, hop=160
│   Linear(160 → 256)     │  → (T/160) frames × 256 dim
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│   SSM Blocks × 10       │  Mamba-style selective SSM
│   - State dim: 16       │  Long-range temporal modeling
│   - Expand: 2           │  O(T) complexity
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│   Attention Blocks × 2  │  Standard multi-head attention
│   - Heads: 8            │  Cross-position alignment
│   - Dropout: 0.1        │  Symbol matching
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│   Output Heads          │
│   - Symbol logits       │  → CTC loss
│   - Frame reconstruction│  → Waveform output
└─────────────────────────┘
```

#### Design Rationale

**Why SSM + Attention hybrid?**

- **SSM layers (10)**: Efficient O(T) long-range modeling. Audio at 16kHz with 1-second inputs has 16,000 timesteps—prohibitive for O(T²) attention. Mamba's selective state space mechanism captures temporal dependencies while maintaining linear scaling.

- **Attention layers (2)**: Explicit cross-position alignment. For tasks like bracket matching, the model must compare positions (opening vs. closing brackets). Attention provides direct position-to-position interaction that SSM's sequential nature may struggle with.

**Parameter count**: 0.94M (compared to wav2vec2-base's 94.57M—100× smaller)

### 2.5 Training

#### Loss Function

$$\mathcal{L} = \lambda_{\text{CTC}} \mathcal{L}_{\text{CTC}} + \lambda_{\text{recon}} \mathcal{L}_{\text{recon}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$$

- **CTC Loss**: Aligns predicted symbol sequence with target
- **Reconstruction Loss**: MSE between predicted and target waveforms (answer window only for Task 3)
- **Auxiliary Losses**: Frame-level CE, blank penalty, etc.

#### Curriculum

1. **Mirror pre-training** (optional): 5 epochs on copy task
2. **Main training**: 50 epochs on target task
3. **Warmup**: Audio reconstruction loss ramped in over 10 epochs

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Schedule | Cosine with warmup |
| Batch size | 32 |
| Gradient clipping | 1.0 |

---

## 3. Experiments

### 3.1 Experimental Setup

**Hardware**: Single NVIDIA RTX 4070 (8GB VRAM)

**Software**: PyTorch 2.0, Python 3.11

**Reproducibility**: Fixed seeds, deterministic data generation, 191 passing pytest cases

### 3.2 Single-Domain Reasoning (H1 Validation)

**Research Question**: Can a neural network perform logical reasoning entirely in the waveform domain?

#### 3.2.1 Task 3: Modular Arithmetic (Audio Domain)

| Model | Parameters | IID EM | OOD-Digits EM |
|-------|------------|--------|---------------|
| LSTM baseline | 0.44M | 42% | — |
| Transformer baseline | 1.23M | 41% | — |
| **Mini-JMamba** | **0.94M** | **45%** | **40%** |
| wav2vec2-base (fine-tuned) | 94.57M | 22% | — |

**Key findings**:
- Mini-JMamba achieves best performance with 100× fewer parameters than wav2vec2
- wav2vec2's poor performance suggests that general speech representations don't transfer well to reasoning tasks
- OOD-Digits (longer inputs, same output dimension) shows 5pp degradation, indicating some length generalization

#### 3.2.2 Task 2: Bracket Matching (Audio Domain)

| Split | EM |
|-------|-----|
| IID | 100% |
| OOD-Length | 95% |
| OOD-Symbol | 98% |

Bracket matching is easier than modular arithmetic for this architecture.

#### 3.2.3 Waveform Reconstruction Quality

To verify the model actually reconstructs waveforms (not just classifies):

| Metric | Value | Threshold |
|--------|-------|-----------|
| STFT-SDR | 32.84 dB | > 15 dB |
| Waveform EM | 98% | — |

The model reconstructs high-fidelity waveforms, not just symbol classifications.

### 3.3 Cross-Domain Reasoning

**Research Question**: Can the same model reason when input and output are in different physical domains?

#### 3.3.1 IPD → Audio (Bracket Matching)

| Metric | Value |
|--------|-------|
| IID EM | 98.7% ± 1.5% |
| OOD-Length EM | 67.3% ± 2.5% |
| Seeds passing | 3/3 |

The model successfully reasons across physically distinct domains!

### 3.4 Cross-Domain Transfer (H2 Validation)

**Research Question**: Do representations learned in one domain transfer to others?

#### 3.4.1 Transfer Matrix (All Pairwise Directions)

| Source → Target | Δ EM (vs Scratch) | Seeds w/ improvement | Significance |
|-----------------|-------------------|---------------------|--------------|
| Audio → IPD | **+1.7 pp** | 7/10 | **p < 0.05** |
| Audio → RF | +0.3 pp | 3/3 | — |
| IPD → Audio | +0.5 pp | 2/3 | — |
| IPD → RF | ±0 | 2/3 | — |
| RF → Audio | -1.0 pp | 1/3 | — |
| RF → IPD | -2.0 pp | 0/3 | — |

#### 3.4.2 Statistical Analysis (Audio → IPD, 10 Seeds)

| Statistic | Value |
|-----------|-------|
| Mean Δ EM | +1.70 pp |
| Std | 1.89 pp |
| 95% Bootstrap CI | [+0.10, +3.40] pp |
| CI excludes zero | ✓ |
| Interpretation | **Statistically significant** |

**Key finding**: Audio pre-training provides a statistically significant benefit for IPD reasoning, supporting H2 (carrier-agnostic representations).

#### 3.4.3 Linear Probe Analysis

To understand *what* transfers, we train linear probes on frozen hidden states:

| Model | Mean Probe Accuracy |
|-------|---------------------|
| Transfer (Audio → IPD) | 99.3% |
| Scratch (IPD only) | 96.0% |
| **Δ** | **+3.3 pp** |

Transfer models learn more linearly separable representations, suggesting higher-quality internal structure.

### 3.5 Negative Controls

#### 3.5.1 Random Mapping Control

To rule out shortcut learning (e.g., using silence patterns, energy distribution):

| Condition | EM |
|-----------|-----|
| Same mapping (train = test) | 98% |
| Different mapping (train ≠ test) | 50% |
| **Δ** | **48 pp** |

The model genuinely learns symbol-waveform associations, not trivial shortcuts.

#### 3.5.2 Phase Scrambling Control

| Condition | EM |
|-----------|-----|
| Original phase | 98% |
| Scrambled phase | 97% |
| **Δ** | 1 pp |

Phase information is not critical—the model relies on frequency/amplitude structure.

### 3.6 Robustness Analysis

#### 3.6.1 Channel Noise (Task 3)

| Perturbation | EM |
|--------------|-----|
| Clean | 45% |
| AWGN 20dB | 45% |
| AWGN 10dB | 44% |
| AWGN 5dB | 42% |
| Phase offset ±30° | 45% |

Strong robustness to additive noise and phase perturbation.

#### 3.6.2 Time-Scale Asymmetry Effect (TSAE)

**Unexpected discovery**: Performance varies asymmetrically with time stretch:

| Time Stretch | Task 3 EM | Task 2 EM |
|--------------|-----------|-----------|
| 0.92× | 10% | **95%** |
| 0.95× | 15% | **98%** |
| 1.00× | 45% | 100% |
| 1.02× | 55% | 97% |
| **1.05×** | **75%** | 85% |
| 1.08× | 60% | 75% |

**Analysis**: The optimal stretch factor differs by task:
- Task 3 (low frequencies): optimal at 1.05×
- Task 2 (high frequencies): optimal at 0.95×

This corresponds to "frequency calibration"—the time stretch that minimizes systematic frequency deviation in model output.

**Interpretation**: The model develops an internal time base that requires calibration. This is reminiscent of biological temporal processing mechanisms.

---

## 审阅要点

### 覆盖内容
- ✅ 完整问题定义（数学形式化）
- ✅ 三个物理域详细参数
- ✅ 三个推理任务定义
- ✅ Mini-JMamba 架构图 + 设计理由
- ✅ 训练细节（损失、课程、超参）
- ✅ 单域推理结果
- ✅ 跨域推理结果
- ✅ 跨域迁移 + 统计检验
- ✅ 负对照实验
- ✅ 鲁棒性 + TSAE 发现

### 可能需要补充
- [ ] 数据集大小/划分比例
- [ ] 训练曲线图
- [ ] 混淆矩阵
- [ ] 更多 OOD 分析

### 表格统计
- 13 个表格
- 涵盖所有核心结果

---

*草稿版本 v1 - 2026-01-01*

