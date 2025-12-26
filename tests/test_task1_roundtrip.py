from __future__ import annotations

import numpy as np

from jericho.scorer import decode_wave_to_symbols, exact_match
from jericho.symbols import SYMBOLS, encode_symbols_to_wave


def test_roundtrip_exact_match():
    rng = np.random.default_rng(2025)
    phase_rng = np.random.default_rng(7)

    for _ in range(25):
        length = int(rng.integers(1, 9))
        sequence = rng.choice(SYMBOLS, size=length, replace=True).tolist()
        wave = encode_symbols_to_wave(sequence, rng=phase_rng)
        decoded = decode_wave_to_symbols(wave)
        assert decoded == sequence
        assert exact_match(decoded, sequence) == 1.0


def test_encode_reproducible_with_seed():
    sequence = ["A", "C", "E", "B"]
    wave_ref = encode_symbols_to_wave(sequence, rng=np.random.default_rng(42))
    wave_repeat = encode_symbols_to_wave(sequence, rng=np.random.default_rng(42))

    assert np.array_equal(wave_ref, wave_repeat)
    assert decode_wave_to_symbols(wave_ref) == sequence
    assert decode_wave_to_symbols(wave_repeat) == sequence

