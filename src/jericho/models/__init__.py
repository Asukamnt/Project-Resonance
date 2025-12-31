"""Model registry for Project Resonance."""

from __future__ import annotations

from typing import Dict, Type, Union

from .mini_jmamba import MiniJMamba, MiniJMambaConfig
from .losses import MultiResolutionSTFTConfig, multi_resolution_stft_loss
from .baselines import (
    BaselineConfig,
    TransformerBaseline,
    LSTMBaseline,
    count_parameters,
)

MODEL_REGISTRY: Dict[str, Type[Union[MiniJMamba, TransformerBaseline, LSTMBaseline]]] = {
    "mini_jmamba": MiniJMamba,
    "transformer": TransformerBaseline,
    "lstm": LSTMBaseline,
}


__all__ = [
    "MODEL_REGISTRY",
    "MiniJMamba",
    "MiniJMambaConfig",
    "MultiResolutionSTFTConfig",
    "multi_resolution_stft_loss",
    "BaselineConfig",
    "TransformerBaseline",
    "LSTMBaseline",
    "count_parameters",
]

