# Limitations and Future Work

## Known Limitations

### 1. Output Dimension Generalization

**Observation**: On Task3 (Arithmetic Mod), models exhibit severe performance degradation (93.3% EM decay) when the output dimension changes from 1-digit to 2-digit remainders.

| Split | Output Dim | Model EM |
|-------|-----------|----------|
| iid_test | 100% 1-digit | 40.0% ± 5.0% |
| ood_digits | 100% 1-digit | 39.7% ± 2.1% |
| ood_length | 77.5% 2-digit | 2.7% ± 0.3% |

**Analysis**: 
- The `ood_digits` split has longer inputs (5 symbols vs 4) but maintains 1-digit outputs → EM remains stable at 39.7%
- The `ood_length` split has even longer inputs (7 symbols) AND 2-digit outputs → EM collapses to 2.7%
- **Conclusion**: The collapse is primarily caused by **output dimension shift**, not input length increase

**Cause**: The model was never exposed to 2-digit remainder outputs during training. This is a fundamental limitation of current end-to-end training: the output space must be seen during training.

**Mitigation Strategies** (Future Work):
1. Train with mixed 1-digit and 2-digit remainders
2. Use curriculum learning from shorter to longer outputs
3. Explore length-generalizable architectures (e.g., RoPE with extrapolation)

### 2. Synthetic Data Gap

**Observation**: All experiments use synthetic audio generated from symbol sequences. Real-world deployment would face:
- Recording artifacts (echoes, background noise, clipping)
- Speaker/microphone variability
- Timing jitter and phase shifts

**Mitigation Evidence**:
- Channel noise ablation shows robustness to AWGN (5-30 dB SNR): No EM degradation
- Simulated reverberation + bandpass filtering: Protocol EM remains 100%
- TSAE (Time-Scale Alignment Effect) suggests models develop internal time bases that may be sensitive to temporal distortions

**Future Work**:
- Hardware-in-the-loop validation with real speakers/microphones
- Test with diverse acoustic environments
- Explore adaptive time-warping mechanisms

### 3. Task Complexity Ceiling

**Observation**: Current tasks (Mirror, Bracket, Mod) are relatively simple compared to real-world reasoning. We have not yet demonstrated:
- Multi-step chained reasoning (A → B → C)
- Working memory for long-horizon tasks
- Compositional generalization across task types

**Future Work**:
- Extend to more complex arithmetic (multiplication, division)
- Add multi-step tasks requiring intermediate state tracking
- Test compositional transfer (train on A and B, test on A+B)

### 4. Architecture-Specific Findings

**Observation**: Mini-JMamba (SSM + 2-Attn hybrid) outperforms pure Transformer and LSTM on OOD-digits (+17-19pp). However:
- All architectures collapse similarly on output dimension shifts
- This suggests the limitation is fundamental to fixed-vocabulary output learning, not specific to any architecture

### 5. Evaluation Metric Limitations

**Observation**: Exact Match (EM) is strict:
- Partial correctness (e.g., 9 out of 10 digits correct) still counts as 0%
- May underestimate model capabilities for longer sequences

**Future Work**:
- Report token-level accuracy in addition to EM
- Use edit distance (Levenshtein) for finer-grained analysis

---

## Summary Table

| Limitation | Severity | Evidence | Mitigation Path |
|------------|----------|----------|-----------------|
| Output dimension generalization | **High** | 93.3% EM decay | Mixed training |
| Synthetic data gap | Medium | Channel noise OK | Hardware validation |
| Task complexity ceiling | Medium | Simple tasks only | Harder tasks |
| Architecture-agnostic collapse | Low | All models collapse | Fundamental research |
| EM metric strictness | Low | Partial credit lost | Add token accuracy |

---

## Honest Claims

Based on these limitations, we make the following honest claims:

1. ✅ **Claim**: Neural networks can perform symbolic reasoning directly on raw waveforms
   - **Evidence**: Task1 EM=100%, Task2 IID=96%, Task3 IID=45%
   - **Scope**: Limited to training distribution

2. ✅ **Claim**: Cross-domain transfer is possible (Audio → Optical → RF)
   - **Evidence**: 9/9 edges verified, few-shot acceleration 1.5-3x
   - **Scope**: Same task structure, different physical modalities

3. ⚠️ **Limited Claim**: Models generalize to longer inputs within output dimension
   - **Evidence**: ood_digits EM=39.7% (same as IID)
   - **Limitation**: Collapses when output dimension changes

4. ❌ **Cannot Claim**: Models generalize to arbitrary output dimensions
   - **Evidence**: ood_length EM=2.7%
   - **Honest Statement**: "Output dimension must be seen during training"

---

*Last updated: 2026-01-02*

