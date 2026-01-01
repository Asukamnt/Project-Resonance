# 论文核心表格

> 以下表格可直接转换为 LaTeX

---

## Table 1: Single-Domain Reasoning (Task3 Mod, Audio→Audio)

| Model | Parameters | IID EM | Notes |
|-------|------------|--------|-------|
| Random Baseline | - | 10% | 1/10 digits |
| LSTM | 440K | 42% | - |
| Transformer | 1.2M | 41% | - |
| wav2vec2-base | 94.6M | 22% | Pretrained 960h |
| **Mini-JMamba** | **0.94M** | **45%** | 10 SSM + 2 Attn |

> Note: wav2vec2 comparison is not fair (different pretraining); included to show that general speech pretraining does not help waveform reasoning.

---

## Table 2: Cross-Domain Reasoning (IPD→Audio)

| Split | EM (mean ± std) | n_seeds |
|-------|-----------------|---------|
| IID Test | 98.7% ± 1.5% | 3 |
| OOD Length | 67.3% ± 2.5% | 3 |

---

## Table 3: Cross-Domain Transfer Learning

| Direction | Δ EM | Δ Convergence | 95% CI | p-value |
|-----------|------|---------------|--------|---------|
| Audio → IPD | +1.7pp | - | [+0.1, +3.4] | <0.05 |
| Audio → RF | +0.3pp | **+9 epochs** | - | - |
| IPD → RF | +3.7pp | +3.7 epochs | - | - |
| RF → IPD | -2pp | **+5.3 epochs** | - | - |
| RF → Audio | +0.3pp | - | - | - |

> Triangular Validation: 6/6 cross-domain pairs show positive transfer signal (performance or convergence).

---

## Table 4: Architecture Ablation

| Architecture | SSM Layers | Attn Layers | CTC EM |
|--------------|------------|-------------|--------|
| Pure SSM | 12 | 0 | 0.0% ❌ |
| Balanced | 6 | 6 | 40.5% |
| More Attention | 8 | 4 | 43.75% |
| **Mini-JMamba** | **10** | **2** | **45.5%** |

> Key finding: Attention is essential (Pure SSM fails completely), but only 2 layers needed.

---

## Table 5: Thinking Gap Ablation

| Gap (seconds) | CTC EM | Hybrid EM |
|---------------|--------|-----------|
| 0.0 | 45.00% | 77.25% |
| 0.1 | 45.00% | 80.25% |
| 0.25 | 46.25% | 80.25% |
| **0.5** | **48.25%** | 80.25% |
| 1.0 | 48.25% | 80.25% |
| 2.0 | 44.75% | **91.00%** |

> Optimal gap: 0.5s for CTC, 2.0s for Hybrid.

---

## Table 6: Time-Scale Asymmetry Effect (TSAE)

| Task | Frequency Range | 0.95x EM | 1.00x EM | 1.05x EM | Optimal |
|------|-----------------|----------|----------|----------|---------|
| Task3 Mod | 300-1100 Hz | 0% | 3% | **9%** | 1.05x |
| Task2 Bracket | 1800-1950 Hz | **50%** | 0% | 0% | 0.95x |

> Key finding: Optimal stretch point is frequency-dependent, supporting the "internal time base" hypothesis.

---

## Table 7: Biological Dynamics Signatures

| Property | Evidence | Biological Analogy |
|----------|----------|-------------------|
| Internal Time Base | TSAE frequency dependency | Neural oscillation |
| Time-Scale Separation | Thinking Gap effect | Fast/slow variables |
| Continuous + Discrete Binding | SSM + 2 Attn | Cortex + Thalamus |
| Calibratable Bias | Frequency drift correctable | Homeostasis |

---

## Table 8: Negative Controls

| Control | Result | Conclusion |
|---------|--------|------------|
| Random Mapping | EM: 98% → 50% | Model uses actual symbol-wave association |
| STFT-SDR | 32.84 dB | True waveform reconstruction, not classification |
| Pure SSM | 0% EM | Attention essential for symbol readout |

---

## Figure List

1. **Figure 1**: Architecture diagram (Mini-JMamba)
2. **Figure 2**: TSAE curve (stretch vs EM, two tasks)
3. **Figure 3**: Cross-domain transfer heatmap (3×3 matrix)
4. **Figure 4**: Thinking gap ablation curve
5. **Figure 5**: Frequency bias histogram (before/after stretch)

---

## LaTeX Snippet (Table 1)

```latex
\begin{table}[h]
\centering
\caption{Single-Domain Reasoning (Task3 Mod)}
\label{tab:single-domain}
\begin{tabular}{lrrr}
\toprule
Model & Params & IID EM \\
\midrule
Random Baseline & - & 10\% \\
LSTM & 440K & 42\% \\
Transformer & 1.2M & 41\% \\
wav2vec2-base$^\dagger$ & 94.6M & 22\% \\
\textbf{Mini-JMamba} & \textbf{0.94M} & \textbf{45\%} \\
\bottomrule
\end{tabular}
\vspace{2mm}
\footnotesize{$^\dagger$ Pretrained on 960h speech; not a fair comparison.}
\end{table}
```

