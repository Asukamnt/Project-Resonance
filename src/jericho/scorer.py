"""Scoring utilities for Task1 round-trip validation."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .symbols import GAP_DUR, SR, SYMBOL2FREQ, SYMBOLS, TONE_DUR


def _nearest_symbol(frequency: float) -> str:
    """Return the symbol whose carrier frequency is closest to ``frequency``."""
    return min(SYMBOLS, key=lambda symbol: abs(SYMBOL2FREQ[symbol] - frequency))


def decode_wave_to_symbols(
    wave: np.ndarray,
    sr: int = SR,
    tone_dur: float = TONE_DUR,
    gap_dur: float = GAP_DUR,
) -> List[str]:
    """Decode an encoded waveform back into its symbol sequence.

    Parameters
    ----------
    wave:
        Input waveform (float array) produced by ``encode_symbols_to_wave``.
    sr:
        Sampling rate used during encoding.
    tone_dur:
        Duration of each tone in seconds.
    gap_dur:
        Duration of the silent gap between tones in seconds.

    Returns
    -------
    list[str]
        Decoded symbol sequence.
    """

    if wave.size == 0:
        return []

    tone_samples = int(round(sr * tone_dur))
    gap_samples = int(round(sr * gap_dur))

    stride = tone_samples + gap_samples
    # Reverse of encode: L = n * tone - gap (no trailing gap).
    num_symbols = (len(wave) + gap_samples) // stride if stride > 0 else len(wave) // tone_samples
    num_symbols = max(num_symbols, 1)

    freqs = np.fft.rfftfreq(tone_samples, d=1.0 / sr)
    decoded: list[str] = []

    for idx in range(num_symbols):
        start = idx * stride
        end = start + tone_samples
        tone_segment = wave[start:end]

        if tone_segment.size < tone_samples:
            # Pad with zeros to maintain FFT shape if rounding trimmed samples.
            tone_segment = np.pad(tone_segment, (0, tone_samples - tone_segment.size))

        spectrum = np.fft.rfft(tone_segment)
        magnitude = np.abs(spectrum)
        if magnitude.size > 0:
            magnitude[0] = 0.0  # Ignore DC component.
        dominant_idx = int(np.argmax(magnitude))
        dominant_freq = freqs[dominant_idx]
        decoded.append(_nearest_symbol(dominant_freq))

    return decoded


def exact_match(pred_symbols: Iterable[str], gold_symbols: Iterable[str]) -> float:
    """Return ``1.0`` when predictions exactly match the gold sequence, else ``0.0``."""
    return 1.0 if list(pred_symbols) == list(gold_symbols) else 0.0


__all__ = [
    "decode_wave_to_symbols",
    "exact_match",
]

