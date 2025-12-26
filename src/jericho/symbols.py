"""Symbol definitions and audio encoding utilities for Project Resonance Task1."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Core symbol configuration for Stage A MVP.
SYMBOLS: Sequence[str] = ("A", "B", "C", "D", "E")

# Frequencies (Hz) spaced to avoid overlap within 16 kHz sampling.
SYMBOL2FREQ = {
    "A": 440.0,
    "B": 560.0,
    "C": 720.0,
    "D": 920.0,
    "E": 1150.0,
}

# Audio generation defaults.
SR: int = 16_000
TONE_DUR: float = 0.10  # seconds
GAP_DUR: float = 0.05  # seconds
_AMPLITUDE: float = 0.8
_TWO_PI: float = 2.0 * np.pi


def _validate_symbols(symbols: Iterable[str]) -> None:
    """Ensure symbols are part of the configured alphabet."""
    invalid = {symbol for symbol in symbols if symbol not in SYMBOL2FREQ}
    if invalid:
        raise ValueError(f"Unsupported symbols for encoding: {sorted(invalid)}")


def encode_symbols_to_wave(
    symbols: Sequence[str],
    sr: int = SR,
    tone_dur: float = TONE_DUR,
    gap_dur: float = GAP_DUR,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Encode a symbol sequence into a concatenated sine-wave audio signal.

    Each symbol is rendered as a sine tone with random phase followed by a fixed
    duration gap of silence (except after the final tone). The output waveform is
    normalised to float32 with a maximum amplitude bounded by `_AMPLITUDE`.

    Parameters
    ----------
    symbols:
        Sequence of symbol tokens drawn from ``SYMBOLS``.
    sr:
        Sampling rate in Hz.
    tone_dur:
        Duration in seconds for each tone segment.
    gap_dur:
        Duration in seconds for silent gaps between tones.
    rng:
        Optional NumPy Generator used for phase randomisation. If ``None``, a
        default generator is created for this call.

    Returns
    -------
    np.ndarray
        Concatenated waveform array of dtype float32.
    """

    if len(symbols) == 0:
        return np.zeros(0, dtype=np.float32)

    _validate_symbols(symbols)

    generator = rng if rng is not None else np.random.default_rng()
    tone_samples = int(round(sr * tone_dur))
    gap_samples = int(round(sr * gap_dur))

    time_axis = np.arange(tone_samples) / sr
    segments: list[np.ndarray] = []

    for idx, symbol in enumerate(symbols):
        frequency = SYMBOL2FREQ[symbol]
        phase = generator.uniform(0.0, _TWO_PI)
        tone = _AMPLITUDE * np.sin(_TWO_PI * frequency * time_axis + phase)
        segments.append(tone.astype(np.float32))

        if idx != len(symbols) - 1 and gap_samples > 0:
            segments.append(np.zeros(gap_samples, dtype=np.float32))

    if not segments:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(segments, dtype=np.float32)


__all__ = [
    "SYMBOLS",
    "SYMBOL2FREQ",
    "SR",
    "TONE_DUR",
    "GAP_DUR",
    "encode_symbols_to_wave",
]

