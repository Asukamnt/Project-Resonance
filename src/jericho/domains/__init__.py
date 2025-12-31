"""Wave domain abstractions for cross-domain reasoning.

Phase2: Optical Intensity Domain (Light→Light)
Phase3: Cross-domain (Light→Sound)
Phase4: Platform layer (WaveDomain × WaveTask)
"""

from .base import WaveDomain
from .optical_intensity import (
    OpticalConfig,
    OpticalIntensityDomain,
    OPTICAL_SYMBOL_VOCAB,
    OPTICAL_SLOT_MAPPING,
)

__all__ = [
    "WaveDomain",
    "OpticalConfig",
    "OpticalIntensityDomain",
    "OPTICAL_SYMBOL_VOCAB",
    "OPTICAL_SLOT_MAPPING",
]

