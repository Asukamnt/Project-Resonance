"""Baselines for Project Resonance."""

from .identity import predict_wave_identity
from .oracle_mod import predict_wave_oracle_mod

__all__ = ["predict_wave_identity", "predict_wave_oracle_mod"]

