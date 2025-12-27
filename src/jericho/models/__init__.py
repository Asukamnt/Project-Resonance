"""Model registry for Project Resonance."""

from __future__ import annotations

from typing import Dict, Type

from .mini_jmamba import MiniJMamba, MiniJMambaConfig
from .losses import MultiResolutionSTFTConfig, multi_resolution_stft_loss

MODEL_REGISTRY: Dict[str, Type[MiniJMamba]] = {
    "mini_jmamba": MiniJMamba,
}


__all__ = [
    "MODEL_REGISTRY",
    "MiniJMamba",
    "MiniJMambaConfig",
    "MultiResolutionSTFTConfig",
    "multi_resolution_stft_loss",
]

