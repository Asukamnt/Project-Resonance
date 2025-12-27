"""Identity baseline operating at the symbol level."""

from __future__ import annotations

import numpy as np

from ..scorer import decode_wave_to_symbols
from ..symbols import GAP_DUR, SR, TONE_DUR, encode_symbols_to_wave


def predict_wave_identity(
    input_wave: np.ndarray,
    *,
    sr: int = SR,
    tone_dur: float = TONE_DUR,
    gap_dur: float = GAP_DUR,
    rng_seed: int | None = 0,
) -> np.ndarray:
    """Return an output waveform whose decoded symbols match the input.

    Parameters
    ----------
    input_wave:
        Incoming waveform encoded according to ``encode_symbols_to_wave``.
    sr, tone_dur, gap_dur:
        Audio configuration mirroring the encoder/decoder defaults.
    rng_seed:
        Optional seed to control the re-encoding phase randomness. Defaults to 0
        for deterministic behaviour.
    """

    decoded_symbols = decode_wave_to_symbols(
        input_wave,
        sr=sr,
        tone_dur=tone_dur,
        gap_dur=gap_dur,
    )
    rng = None
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    return encode_symbols_to_wave(
        decoded_symbols,
        sr=sr,
        tone_dur=tone_dur,
        gap_dur=gap_dur,
        rng=rng,
    )


__all__ = ["predict_wave_identity"]

