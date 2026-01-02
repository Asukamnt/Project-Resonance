# Jericho: Reasoning is Resonance
### *— Cross-Domain Waveform Reasoning Without Tokens*

**English** | **[中文](README_CN.md)**

<p align="center">
  <img src="docs/paper/figures/main/fig0_cover_trajectories.png" alt="Trajectories of Thought" width="600"/>
</p>

<p align="center">
  <em>"Different bodies, same soul — where audio, light, and radio waves converge."</em>
</p>

<p align="center">
  <img src="docs/paper/figures/supp/video_S4_thought_trajectories.gif" alt="Cross-Domain Resonance" width="500"/>
</p>

---

## Core Discovery

**Neural networks can reason directly on raw waveforms, and the learned representations *resonate* across physically distinct carriers.**

| Traditional Pipeline | Jericho |
|---------------------|---------|
| `Audio → ASR → Text → LLM → Text → TTS → Audio` | `Waveform → Neural Network → Waveform` |

<p align="center">
  <img src="docs/paper/figures/main/fig6_cross_domain.png" alt="Cross-Domain Matrix" width="700"/>
</p>

---

## Key Results

<p align="center">
  <img src="docs/paper/figures/main/fig2_transfer_matrix.png" alt="Transfer Matrix" width="600"/>
</p>

| Experiment | Result | Significance |
|------------|--------|--------------|
| **Single-Domain Reasoning** | Mini-JMamba 45% vs wav2vec2 13% | Task-specialized architecture wins |
| **Cross-Domain Reasoning** | IPD→Audio IID 98.7% | Reasoning transfers across physics |
| **Cross-Domain Transfer** | +1.7pp (p<0.05, 10-seed) | Statistically significant resonance |
| **Real Human Speech** | 91.7% ± 0.3% (3-seed) | Generalizes to naturalistic audio |
| **Triangle Validation** | Audio↔IPD↔RF 9/9 edges | Carrier-agnostic representation |

> **Why "Resonance"?** The model's internal clock synchronizes with external signal rhythms. See the TSAE heatmap below — the bright diagonal is where silicon minds *resonate* with physical waves.

<p align="center">
  <img src="docs/paper/figures/main/fig5_tsae_resonance.png" alt="TSAE Resonance" width="550"/>
</p>

---

## What is this?

**Jericho** is an experimental framework validating a fundamental hypothesis:

> **Neural networks can perform symbolic reasoning directly on continuous physical waveforms across different domains (Audio, Optical/IPD, RF), without discrete token intermediaries.**

### Three Tasks × Three Domains

| Task | Input | Output | Capability |
|------|-------|--------|------------|
| **Mirror** | Symbol sequence waveform | Same symbols | Codec roundtrip |
| **Bracket** | Bracket expression | Match result | Structural reasoning |
| **Mod** | Math expression | Modulo result | Arithmetic reasoning |

| Domain | Encoding | Sample Rate |
|--------|----------|-------------|
| **Audio** | Frequency Modulation | 16 kHz |
| **Optical (IPD)** | Pulse Position | 1 kHz |
| **RF** | Amplitude Shift Keying | 1 MHz |

---

## Architecture

<p align="center">
  <img src="docs/paper/figures/main/fig1_architecture.png" alt="Mini-JMamba Architecture" width="600"/>
</p>

**Mini-JMamba**: 0.94M parameters, 10 SSM + 2 Attention layers

```
Input Waveform → Frame Embedding → [SSM Block]×10 → [Attention]×2 → Output Waveform
```

---

## OOD Collapse Analysis

<p align="center">
  <img src="docs/paper/figures/main/fig3_trajectory_comparison.png" alt="OOD Trajectory" width="700"/>
</p>

When output dimensionality changes (1-digit → 2-digit remainders), hidden states drift into unexplored latent regions:

<p align="center">
  <img src="docs/paper/figures/main/fig4_endpoint_distribution.png" alt="Endpoint Distribution" width="500"/>
</p>

---

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -e .
pytest -q  # 199 tests should pass

# Train
python train.py --config configs/task3_mod_stable.yaml --manifest manifests/task3_tiny.jsonl

# Evaluate
python evaluate.py --stage final --tasks mirror bracket mod
```

---

## Documentation

- 📖 **[Technical Overview](docs/overview.md)** — Full motivation and design
- 📊 **[Experiment Log](docs/iteration_log.md)** — Complete reproducibility info
- 📋 **[Known Issues](docs/known_issues.md)** — Limitations and future work

---

## Supplementary Materials

### Animations

| Video | Description |
|-------|-------------|
| [video_S1](docs/paper/figures/supp/video_S1_ood_collapse.gif) | OOD collapse dynamics |
| [video_S2](docs/paper/figures/supp/video_S2_multi_task.gif) | Multi-task trajectory evolution |
| [video_S3](docs/paper/figures/supp/video_S3_cross_domain.gif) | Cross-domain synchronization |
| [video_S4](docs/paper/figures/supp/video_S4_thought_trajectories.gif) | 3D thought trajectories |

### Additional Figures

See [`docs/paper/figures/README.md`](docs/paper/figures/README.md) for complete figure index.

---

## Citation

```bibtex
@misc{jericho2026,
  author = {Baiyi Wang},
  title = {Jericho: Reasoning is Resonance — Cross-Domain Waveform Reasoning Without Tokens},
  year = {2026},
  url = {https://github.com/Asukamnt/Project-Resonance}
}
```

---

## Contact

- 📧 Email: 928112278@qq.com
- 💬 GitHub Issues welcome

---

## License

MIT License
