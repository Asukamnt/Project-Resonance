"""Oracle baseline for Task3 Arithmetic Mod."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..scorer import decode_wave_to_symbols
from ..symbols import GAP_DUR, SR, TONE_DUR, encode_symbols_to_wave
from ..task3 import target_symbols_for_task3


def predict_wave_oracle_mod(
    input_wave: np.ndarray,
    *,
    sr: int = SR,
    tone_dur: float = TONE_DUR,
    gap_dur: float = GAP_DUR,
    rng_seed: int = 0,
) -> np.ndarray:
    """Decode input audio, compute modulo remainder and re-encode as audio."""
    tokens = decode_wave_to_symbols(input_wave)
    target_tokens = target_symbols_for_task3(tokens)
    rng = np.random.default_rng(rng_seed)
    target_wave = encode_symbols_to_wave(
        target_tokens, sr=sr, tone_dur=tone_dur, gap_dur=gap_dur, rng=rng
    )
    if target_wave.size < input_wave.size:
        pad = np.zeros(input_wave.size - target_wave.size, dtype=np.float32)
        target_wave = np.concatenate([target_wave, pad])
    else:
        target_wave = target_wave[: input_wave.size]
    return target_wave.astype(np.float32, copy=False)


__all__ = ["predict_wave_oracle_mod"]

