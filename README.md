# Jericho: End-to-End Reasoning on Raw Physical Waveforms

**English** | **[中文](README_CN.md)**

<p align="center">
  <strong>Cross-domain symbolic reasoning directly on physical waveforms (Audio / Optical / RF), without text intermediaries</strong>
</p>

---

## What is this?

**Jericho** is an experimental framework that validates a core hypothesis:

> **Neural networks can perform symbolic reasoning directly on continuous physical waveforms across different domains (Audio, Optical/IPD, RF), without discrete text/token intermediate representations.**

Traditional speech understanding: `Audio → ASR → Text → LLM → Text → TTS → Audio`

Jericho's approach: `Waveform → Neural Network → Waveform`

We designed three progressively challenging tasks, validated across three physical domains:

| Task | Input | Output | Validated Capability |
|------|-------|--------|---------------------|
| **Task 1: Mirror** | Symbol sequence waveform | Same symbol sequence waveform | Waveform codec roundtrip |
| **Task 2: Bracket** | Bracket expression waveform | Matching result waveform | Structural reasoning |
| **Task 3: Mod** | Math expression waveform | Modulo result waveform | Arithmetic reasoning |

**Supported Physical Domains**: Audio (sinusoidal) · Optical/IPD (intensity-phase) · RF (amplitude modulation)

---

## Why does this matter?

1. **Information Fidelity**: Tokenization loses phase, temporal microstructure, and other waveform information. Direct waveform reasoning preserves more information.

2. **Latency & Streaming**: No need to wait for complete token sequences; enables causal/streaming inference.

3. **Cross-Domain Generalization**: We have validated that the same architecture can reason across different physical waveforms — Audio ↔ Optical ↔ RF transfer learning works with statistical significance.

---

## Core Components

- **Mini-JMamba**: 12-layer Mamba-2/Attention hybrid architecture, processing raw waveforms directly
- **Multi-Domain Encoders**: Symbol-to-waveform mapping for Audio, Optical (IPD), and RF domains
- **Scorer Decoder**: FFT-based frequency identification for evaluation
- **Manifest System**: Reproducible data generation and splitting
- **Cross-Domain Pipelines**: Training and inference across physical domains
- **Closed-Loop Evaluation**: Complete pipeline from manifest to synthesis, inference, decoding, and Exact Match

---

## Development Timeline

| Date | Milestone | Description |
|------|-----------|-------------|
| 2025-12-26 | **Stage A Framework** | Task 1 codec roundtrip, Scorer, test infrastructure |
| 2025-12-28 | **Task 2 OOD Breakthrough** | Bracket matching, RoPE + continuous waveform generation |
| 2025-12-29 | **Phase 1 Complete** | Evaluation tools, ablations, negative controls |
| 2025-12-31 | **Cross-Domain Release** | Audio/Optical/RF domains, transfer learning validated |

---

## Current Status

### 🎉 Key Breakthroughs

| Experiment | Result | Significance |
|------------|--------|--------------|
| **Single-Domain Reasoning** | Mini-JMamba 45% vs wav2vec2 22%¹ | Small model advantage |
| **Cross-Domain Reasoning** | IPD→Audio IID 98.7% | Cross-physical-domain success |
| **Cross-Domain Transfer** | +1.7pp (p<0.05, 10-seed) | Statistically significant |
| **Triangle Validation** | Audio↔IPD↔RF 6/6 | Carrier-agnostic evidence |

> ¹ wav2vec2 uses frozen feature extractor + linear head (94.57M params); Mini-JMamba is fully trained (0.94M params). Different setups, for reference only.

### ✅ Completed

- Phase 1: Audio domain single-domain reasoning
- Phase 2: IPD (optical) domain single-domain reasoning
- Phase 3: Cross-domain reasoning (IPD→Audio)
- Phase 4: Cross-domain transfer validation
- Full validation across three physical domains (Audio / IPD / RF)
- 191 test cases all passing

---

## Experimental Results

### Single-Domain Reasoning (Audio, Task 3 Mod)

| Model | Parameters | IID EM | 
|-------|------------|--------|
| wav2vec2-base¹ | 94.57M | 22% |
| Transformer | 1.2M | 41% |
| **Mini-JMamba** | **0.94M** | **45%** |

### Cross-Domain Reasoning (IPD → Audio)

| Metric | Result |
|--------|--------|
| IID EM | 98.7% ± 1.5% |
| OOD EM | 67.3% ± 2.5% |

### Cross-Domain Transfer

| Direction | Δ EM | Statistical Significance |
|-----------|------|-------------------------|
| Audio → IPD | +1.7pp | ✅ 95% CI excludes 0 |
| Audio → RF | +0.3pp | Convergence accelerated by 9 epochs |

---

## Quick Start

### Environment Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest -q
```

### Running Examples

```powershell
# Generate Task3 (Mod) manifest
python -m jericho.data.make_task3_manifest --out manifests/task3_tiny.jsonl --seed 321 --preset tiny --balance-remainder

# Train Mini-JMamba
python .\train.py --config configs\task3_mod_stable.yaml --manifest manifests\task3_tiny.jsonl --split iid_test --limit 200

# Oracle/Protocol validation (verifies encode→decode correctness, not model capability)
python .\evaluate.py --stage final --tasks mirror bracket mod

# Model capability evaluation (requires checkpoint from train.py)
# python .\evaluate_model.py --checkpoint runs\your_run\mod_seed123_epoch50.pt --tasks mod --splits iid_test --limit 50
```

> **Evaluation Metrics Clarification**:
> - **Oracle EM**: System roundtrip validation, encode→decode consistency (`evaluate.py`)
> - **Model EM**: Trained model capability, prediction accuracy (`evaluate_model.py`)
> 
> Oracle EM = 1.0 proves the evaluation protocol is correct; Model EM reflects actual model capability.

---

## Documentation

<details>
<summary><strong>Directory Structure</strong></summary>

- `src/jericho/symbols.py`: Symbol table, frequency mapping, sinusoidal waveform synthesis
- `src/jericho/domains/`: Multi-domain waveform encoders (Audio, Optical/IPD, RF)
- `src/jericho/scorer.py`: FFT-based frequency identification and exact match scoring
- `src/jericho/models/mini_jmamba.py`: Mini-JMamba model implementation (Mamba-2 + Attention)
- `src/jericho/pipelines/`: Training/inference pipelines for each task and domain
- `src/jericho/data/`: Manifest generation tools
- `train.py`: Unified training CLI
- `evaluate.py`: Oracle/Protocol closed-loop evaluation (system validation)
- `evaluate_model.py`: Model capability evaluation (requires checkpoint)
- `tests/`: Complete test suite (191 cases)

</details>

<details>
<summary><strong>Manifest Format</strong></summary>

- File format: JSON Lines
- Fields: `split`, `symbols`, `length`, `difficulty_tag`, `example_id`, `seed`, `sequence_seed`
- Default splits: `train=500`, `val=100`, `iid_test=100`, `ood_length=100`, `ood_symbol=100`
- Symbol and length ranges:
  - `train/val/iid_test`: Symbols A–E, length 1–8
  - `ood_length`: Symbols A–E, length 9–12
  - `ood_symbol`: Symbols A–F (at least one F), length 1–8

</details>

<details>
<summary><strong>Full Training Commands</strong></summary>

```powershell
# Task 1: Identity baseline
python .\train.py --model identity --manifest manifests\task1.jsonl --split iid_test --outdir runs\identity_demo --limit 50

# Task 2: Bracket matching
python .\train.py --config configs\task2_bracket_stable.yaml --task bracket --model mini_jmamba --manifest manifests\task2_tiny.jsonl --split iid_test --epochs 50

# Task 3: Mod with thinking gap
python .\train.py --task mod --model mini_jmamba --manifest manifests\task3_easy.jsonl --split iid_test --limit 200 --epochs 50 --pretrain-mirror-epochs 30 --thinking-gap-s 0.5 --thinking-gap-align 160 --outdir runs\mini_jmamba_mod_week4

# Task 3: Using config file
python .\train.py --config configs\task3_mod_stable.yaml --manifest manifests\task3_tiny.jsonl --split iid_test --limit 200
```

</details>

<details>
<summary><strong>Oracle Baselines</strong></summary>

```powershell
# Task 3 Mod oracle (outputs correct answer directly)
python .\train.py --task mod --model oracle_mod --manifest manifests\task3.jsonl --split iid_test --outdir runs\oracle_mod_iid --limit 50
```

</details>

---

## Related Concepts

This project is part of the **Cross-Wave Physical Reasoning (CWPR)** research paradigm, exploring end-to-end reasoning on arbitrary physical waveforms.

---

## Reproduction & Optimal Configurations

The configuration files in this repository are **baseline configurations** that verify the system works correctly and produces reasonable results.

> ⚠️ **Note**: Due to file size, demo checkpoints and audio examples are not included. Use `train.py` to generate your own checkpoints.

If you need:
- 📊 Optimal hyperparameters reported in papers
- 🔬 More experimental details and ablation results
- 🤝 Collaboration or discussion

Please contact me:
- 📧 Email: 928112278@qq.com
- 💬 GitHub Issues: Questions welcome

---

## Citation

If you use this project, please cite:

```
@misc{jericho2025,
  author = {Baiyi Wang},
  title = {Jericho: End-to-End Reasoning on Raw Physical Waveforms},
  year = {2025},
  url = {https://github.com/Asukamnt/Project-Resonance}
}
```

---

## License

MIT License
