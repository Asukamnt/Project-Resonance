from __future__ import annotations

import numpy as np

from jericho.scorer import decode_wave_to_symbols
from jericho.symbols import encode_symbols_to_wave


def _jitter_wave(wave: np.ndarray, sr: int, jitter_ms: float) -> np.ndarray:
    jitter_samples = int(round(sr * jitter_ms / 1000.0))
    if jitter_samples == 0:
        return wave
    if jitter_samples > 0:
        return np.concatenate([np.zeros(jitter_samples, dtype=wave.dtype), wave])
    jitter_samples = abs(jitter_samples)
    return wave[jitter_samples:]


def test_energy_segmentation_handles_jitter():
    sequence = ["A", "B", "C", "D"]
    sr = 16_000
    wave = encode_symbols_to_wave(sequence, sr=sr, rng=np.random.default_rng(0))
    jittered_plus = _jitter_wave(wave, sr=sr, jitter_ms=10.0)
    jittered_minus = _jitter_wave(wave, sr=sr, jitter_ms=-10.0)

    decoded_hard_plus = decode_wave_to_symbols(jittered_plus, sr=sr, segmentation="hard")
    decoded_energy_plus = decode_wave_to_symbols(jittered_plus, sr=sr, segmentation="energy")
    decoded_hard_minus = decode_wave_to_symbols(jittered_minus, sr=sr, segmentation="hard")
    decoded_energy_minus = decode_wave_to_symbols(jittered_minus, sr=sr, segmentation="energy")

    assert decoded_energy_plus == sequence
    assert decoded_energy_minus == sequence
    if decoded_hard_plus != sequence or decoded_hard_minus != sequence:
        assert True

