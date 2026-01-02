# Appendix

## A. Implementation Details

### A.1 Model Architecture

**Mini-JMamba** consists of:
- Frame Encoder: 1D Conv (kernel=3, stride=1) → LayerNorm → ReLU
- 4 SSM Blocks (Mamba-style selective state space)
- 2 Attention Blocks (single-head self-attention)
- Attention Pooling → Linear Classification Head

| Component | Parameter Count |
|-----------|-----------------|
| Frame Encoder | 49,280 |
| SSM Blocks (×4) | 524,288 |
| Attention Blocks (×2) | 262,144 |
| Classification Head | 1,536 |
| **Total** | **942,848** |

### A.2 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-2 |
| Batch Size | 32 |
| Max Epochs | 50 (early stopping) |
| LR Scheduler | ReduceLROnPlateau |
| Patience | 5 epochs |
| Seed | 42 (primary), 10-seed sweep for significance |

### A.3 Data Generation

**Symbol Encoding:**
- Sample Rate: 16,000 Hz
- Symbol Duration: 0.1s (1,600 samples)
- Frequency Mapping: A=220Hz, B=440Hz, ..., 9=880Hz
- Silence: 0.05s between symbols

**Task3 Modular Arithmetic:**
- Expression Format: `digit + digit % modulus`
- Train: 270 expressions (disjoint)
- Val: 30 expressions (disjoint)
- IID Test: 100 expressions (disjoint)
- OOD Digits: Longer expressions (same output range)
- OOD Length: 2-digit remainders (unseen output dimension)

## B. Extended Experimental Results

### B.1 State Dynamics Analysis

**SSM Hidden State Norms:**

| Sequence Length | Mean Norm | Growth Rate |
|-----------------|-----------|-------------|
| 4 symbols | 12.3 | - |
| 8 symbols | 18.7 | +52% |
| 13 symbols | 24.1 | +29% (inflection) |
| 16 symbols | 28.9 | +20% |
| 32 symbols | 45.2 | +56% |

**Key Finding:** Linear growth with inflection at ~13 symbols suggests continuous information integration without explosion.

### B.2 Pruning Strategy Comparison

**10-Seed k-Sweep (32 symbols):**

| Keep Ratio (k) | Mean Max-Prob | Δ vs k=1.0 | 95% CI |
|----------------|---------------|------------|--------|
| 1.0 (baseline) | 97.4% | - | - |
| 0.7 | 97.2% | -0.2pp | [-0.8, +0.4] |
| 0.5 | 94.8% | -2.6pp | [-4.1, -1.1] |
| 0.3 | 82.9% | -14.5pp | [-17.2, -11.8] |

**Conclusion:** For short sequences, full state retention is optimal.

**Long Sequence Pruning (k=0.7):**

| Sequence Length | Baseline Prob | Pruned Prob | Δ |
|-----------------|---------------|-------------|---|
| 32 symbols | 97.4% | 97.8% | +0.4pp |
| 64 symbols | 89.2% | 91.4% | **+2.2pp** |
| 96 symbols | 82.1% | 83.6% | +1.5pp |
| 128 symbols | 74.3% | 76.0% | +1.7pp |

**Conclusion:** Pruning becomes beneficial beyond ~64 symbols.

### B.3 Dropout vs Selective Pruning

| Method | Keep Ratio | Mean Max-Prob | Δ vs Baseline |
|--------|------------|---------------|---------------|
| Baseline | 100% | 97.4% | - |
| Random Dropout | 50% | 49.2% | -48.2pp |
| Selective Pruning | 50% | 94.8% | -2.6pp |

**Conclusion:** Selective pruning vastly outperforms random dropout, validating the importance of preserving high-energy channels.

### B.4 Dynamic k Training

Training a learnable keep ratio via sigmoid gate:

| Epoch | Learned k | Train Loss | Val EM |
|-------|-----------|------------|--------|
| 0 | 0.50 | 2.34 | 12% |
| 10 | 0.68 | 0.89 | 38% |
| 20 | 0.70 | 0.45 | 42% |
| 30 | 0.7048 | 0.42 | 43% |

**Convergence:** Model learns k ≈ 0.7 without manual tuning.

## C. Reproducibility

### C.1 Hardware

- GPU: NVIDIA RTX 2060 (6GB)
- CPU: Intel i5-9400F
- RAM: 16GB
- Training Time: ~30 min per model

### C.2 Software

- Python 3.11
- PyTorch 2.6.0 + CUDA 12.4
- Key packages: `torchaudio`, `scipy`, `numpy`

### C.3 Random Seeds

All experiments use fixed seeds for reproducibility:
- Primary seed: 42
- 10-seed sweep: [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]

### C.4 Commands

**Training:**
```bash
python train.py --task mod --epochs 50 --seed 42
```

**Evaluation:**
```bash
python evaluate.py --checkpoint runs/best.pt --split iid_test
```

**Cross-Domain Transfer:**
```bash
python train.py --task mod --domain ipd --pretrained runs/audio_best.pt
```

## D. Turing Machine Simulation Experiments

### D.1 Task Design

We designed 5 levels of progressive algorithmic tasks to probe SSM computational boundaries:

| Level | Task | Input Example | Output | Complexity |
|-------|------|---------------|--------|------------|
| 1 | Counting | `AAAA` | `4` | O(n) linear accumulation |
| 2 | Parity (XOR) | `ABBA` | `0` (even) | O(n) discrete flip |
| 3 | Bracket Depth | `((()))` | `3` | O(n) stack tracking |
| 4 | Binary Addition | `101+011` | `1000` | O(n) with carry |
| 5 | Program Execution | `INC INC DEC` | `1` | O(n) state machine |

### D.2 Results

| Level | Task | Train Acc | Val Acc | Test EM | Status |
|-------|------|-----------|---------|---------|--------|
| 1 | Counting | 100% | 100% | **100%** | ✓ Solved |
| 2 | Parity (XOR) | 52% | 50% | **~50%** | ✗ Random guess |
| 3 | Bracket Depth | 96.5% | 94% | **96.5%** | ✓ Solved |
| 4 | Binary Addition | 78% | 72% | **75%** | △ Partial |
| 5 | Program Execution | 65% | 58% | **62%** | △ Partial |

### D.3 Key Finding: Computational Boundary

**SSM succeeds at:** Accumulative operations (counting, bracket depth)
- These require monotonic state updates: `h_{t+1} = h_t + δ`

**SSM fails at:** Discrete-flip operations (XOR parity)
- These require modular state reset: `h_{t+1} = (h_t + 1) mod 2`

**Root cause:** SSM's continuous state dynamics cannot implement hard thresholds. When bracket depth increases from 2→3, it's a linear step (+1). When XOR flips from odd→even, it's a discontinuous jump (1→0), which continuous functions struggle to represent.

**Analogy to Kahneman's Dual-Process Theory:**
- SSM ≈ System 1 (fast, parallel, approximate)
- Discrete logic ≈ System 2 (slow, serial, precise)

Full analysis with hidden state visualizations reserved for follow-up work.

---

## E. Limitations

### E.1 OOD-Length Collapse

The model achieves only ~2-3% EM on `ood_length` splits where outputs require 2-digit remainders. Analysis reveals this is primarily due to **output dimension extrapolation** (never seen during training), not input length.

### E.2 Synthetic Data

All experiments use synthesized waveforms. While robustness tests (noise, reverb) suggest real-world applicability, validation with physical hardware (microphone, laser, RF antenna) remains future work.

### E.3 Task Complexity

Current tasks are relatively simple (modular arithmetic, bracket matching). Extension to complex reasoning (multi-step, compositional) is ongoing.

## F. Inference Efficiency Comparison

We benchmark all baseline architectures with comparable parameter counts:

| Model | Params | Latency (ms) | Throughput (samples/s) |
|-------|--------|--------------|------------------------|
| **Mini-JMamba** | **0.94M** | 24.25 | 41.2 |
| Transformer | 1.23M | 17.78 | 56.2 |
| LSTM | 0.64M | 157.28 | 6.4 |
| S4 (simplified) | 0.54M | 178.29 | 5.6 |
| Hyena | 0.84M | 18.92 | 52.8 |

*Benchmark: batch size = 1, sequence length = 100 frames, CPU inference (no GPU).*

**Key observations:**
- Mini-JMamba achieves competitive throughput with fewer parameters than Transformer
- S4 and LSTM are significantly slower due to sequential dependencies
- Hyena is fastest due to FFT-based long convolutions

Note: S4 uses simplified Python implementation (not optimized CUDA kernels). With proper CUDA implementation, S4 would be much faster.

---

## G. Compute Budget

| Experiment | GPU Hours |
|------------|-----------|
| Single model training | 0.5h |
| 10-seed sweep | 5h |
| Cross-domain transfer (9 directions) | 4.5h |
| State dynamics analysis | 2h |
| Pruning experiments | 8h |
| Turing Machine experiments | 3h |
| Baseline comparison | 2h |
| **Total** | **~25 GPU hours** |

All experiments were conducted on a single consumer GPU (RTX 4070, 8GB).

