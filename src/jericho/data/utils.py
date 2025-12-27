"""Helper utilities for dataset-driven audio synthesis."""

from __future__ import annotations

import numpy as np

from .manifest import ManifestEntry
from ..symbols import encode_symbols_to_wave


def synthesise_entry_wave(entry: ManifestEntry) -> np.ndarray:
    """Recreate the input waveform for a manifest entry."""
    rng = np.random.default_rng(entry.sequence_seed)
    return encode_symbols_to_wave(entry.symbols, rng=rng)


__all__ = ["synthesise_entry_wave"]

