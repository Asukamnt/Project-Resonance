"""Abstract base class for wave domains.

This module defines the WaveDomain interface that all domain implementations
must follow. It enables Phase4's platform layer (WaveDomain × WaveTask).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

import numpy as np


class WaveDomain(ABC):
    """Abstract base class for wave domain implementations.
    
    A WaveDomain defines how symbols are encoded into waveforms and decoded
    back. It also specifies how physical/channel distortions are applied.
    
    Core interface (Phase4 platform layer):
    - encode(symbols) -> wave: Symbol sequence to waveform
    - decode(wave) -> symbols: Waveform to symbol sequence  
    - channel(wave, ood_params) -> wave': Apply physical distortions
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sampling rate in Hz."""
        pass

    @property
    @abstractmethod
    def symbol_duration(self) -> float:
        """Duration of each symbol in seconds."""
        pass

    @property
    @abstractmethod
    def samples_per_symbol(self) -> int:
        """Number of samples per symbol."""
        pass

    @abstractmethod
    def encode(self, symbols: Sequence[str]) -> np.ndarray:
        """Encode a symbol sequence into a waveform.
        
        Parameters
        ----------
        symbols : Sequence[str]
            Symbol sequence to encode.
            
        Returns
        -------
        np.ndarray
            Waveform array (1D, float32, domain-specific constraints).
        """
        pass

    @abstractmethod
    def decode(self, wave: np.ndarray) -> List[str]:
        """Decode a waveform back into a symbol sequence.
        
        Parameters
        ----------
        wave : np.ndarray
            Waveform to decode.
            
        Returns
        -------
        List[str]
            Decoded symbol sequence.
        """
        pass

    @abstractmethod
    def channel(self, wave: np.ndarray, ood_params: Dict[str, Any]) -> np.ndarray:
        """Apply channel distortions to a waveform.
        
        Parameters
        ----------
        wave : np.ndarray
            Input waveform.
        ood_params : Dict[str, Any]
            OOD parameters (noise, drift, bandwidth, etc.).
            
        Returns
        -------
        np.ndarray
            Distorted waveform.
        """
        pass

    def oracle_roundtrip(self, symbols: Sequence[str]) -> List[str]:
        """Test encode/decode roundtrip (Oracle EM validation).
        
        Parameters
        ----------
        symbols : Sequence[str]
            Input symbols.
            
        Returns
        -------
        List[str]
            Decoded symbols after encode→decode.
        """
        wave = self.encode(symbols)
        return self.decode(wave)


__all__ = ["WaveDomain"]

