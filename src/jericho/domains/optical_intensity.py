"""Optical Intensity Domain implementation for Phase2.

This module implements MPPM (Multi-Pulse Position Modulation) 2-of-10 encoding
for optical intensity waveforms. Each symbol is encoded as exactly 2 pulses
in a 10-slot window, providing C(10,2)=45 unique symbols.

Key specifications (v0.1):
- Sample rate: 1000 Hz (1 kHz)
- Symbol duration: 100 ms = 100 samples
- Slots per symbol: 10 (each slot = 10 samples)
- Pulses per symbol: 2 (constant energy encoding)
- Value range: [0, 1] (non-negative, physically realistic)

Reference: docs/phase2_light_to_light.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .base import WaveDomain


# =============================================================================
# Frozen slot mapping table (v0.1, Appendix A)
# SINGLE SOURCE OF TRUTH - DO NOT REGENERATE AT RUNTIME
# =============================================================================

OPTICAL_SLOT_MAPPING: Dict[str, Tuple[int, int]] = {
    # Task2 outputs (strong separation)
    "V": (0, 1),
    "X": (8, 9),
    # Task2 inputs (strong separation)
    "(": (2, 3),
    ")": (6, 7),
    # Task3 operator (strong separation)
    "%": (4, 5),
    # Task3 digits
    "0": (0, 2),
    "1": (1, 3),
    "2": (2, 4),
    "3": (3, 5),
    "4": (4, 6),
    "5": (5, 7),
    "6": (6, 8),
    "7": (7, 9),
    "8": (0, 8),
    "9": (1, 9),  # OOD-symbol holdout
    # Task1 letters
    "A": (0, 3),
    "B": (1, 4),
    "C": (2, 5),
    "D": (3, 6),
    "E": (4, 7),
    "F": (5, 8),  # OOD-symbol holdout
}

# Ordered vocabulary for consistent iteration
OPTICAL_SYMBOL_VOCAB: Tuple[str, ...] = (
    # Task2
    "(", ")", "V", "X",
    # Task3
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "%",
    # Task1
    "A", "B", "C", "D", "E", "F",
)

# Reverse mapping for decode: (i, j) -> symbol
_SLOTS_TO_SYMBOL: Dict[Tuple[int, int], str] = {
    slots: symbol for symbol, slots in OPTICAL_SLOT_MAPPING.items()
}

# OOD-symbol holdout symbols (excluded from iid_train/val)
OOD_SYMBOL_HOLDOUT: Tuple[str, ...] = ("F", "9")


@dataclass
class OpticalConfig:
    """Configuration for Optical Intensity Domain.
    
    These parameters define the MPPM 2-of-10 encoding scheme.
    """
    
    sample_rate: int = 1000  # 1 kHz (visible light / LED communication)
    symbol_duration: float = 0.1  # 100 ms per symbol
    num_slots: int = 10  # Slots per symbol window
    pulses_per_symbol: int = 2  # Constant energy (2-of-N)
    pulse_amplitude: float = 1.0  # Pulse high value
    baseline: float = 0.0  # Empty slot value
    gap_duration: float = 0.0  # No gap in v0.1
    
    # Derived values (computed in __post_init__)
    samples_per_symbol: int = field(init=False)
    slot_samples: int = field(init=False)
    
    def __post_init__(self) -> None:
        self.samples_per_symbol = int(round(self.sample_rate * self.symbol_duration))
        self.slot_samples = self.samples_per_symbol // self.num_slots
        
        # Validate alignment
        if self.samples_per_symbol != self.num_slots * self.slot_samples:
            raise ValueError(
                f"samples_per_symbol ({self.samples_per_symbol}) must be exactly "
                f"divisible by num_slots ({self.num_slots})"
            )


class OpticalIntensityDomain(WaveDomain):
    """Optical Intensity Domain using MPPM 2-of-10 encoding.
    
    This domain encodes symbols as pulse patterns in an intensity-only
    (non-negative) waveform, suitable for visible light communication
    or LED-based signaling.
    
    Key properties:
    - Non-negative output: wave[t] >= 0
    - Constant energy per symbol: exactly 2 pulses
    - Robust to gain/bias drift: normalized energy matching decode
    """
    
    def __init__(self, config: OpticalConfig | None = None) -> None:
        self.config = config or OpticalConfig()
        self._validate_vocab()
    
    def _validate_vocab(self) -> None:
        """Ensure all vocab symbols have valid slot mappings."""
        for symbol in OPTICAL_SYMBOL_VOCAB:
            if symbol not in OPTICAL_SLOT_MAPPING:
                raise ValueError(f"Symbol '{symbol}' missing from OPTICAL_SLOT_MAPPING")
            slots = OPTICAL_SLOT_MAPPING[symbol]
            if len(slots) != 2:
                raise ValueError(f"Symbol '{symbol}' must have exactly 2 slots")
            if not (0 <= slots[0] < slots[1] < self.config.num_slots):
                raise ValueError(
                    f"Symbol '{symbol}' slots {slots} invalid for "
                    f"num_slots={self.config.num_slots}"
                )
    
    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate
    
    @property
    def symbol_duration(self) -> float:
        return self.config.symbol_duration
    
    @property
    def samples_per_symbol(self) -> int:
        return self.config.samples_per_symbol
    
    def encode(self, symbols: Sequence[str]) -> np.ndarray:
        """Encode symbols to optical intensity waveform using MPPM 2-of-10.
        
        Each symbol is encoded as a 100-sample window with pulses at
        exactly 2 of the 10 slots (10 samples each).
        
        Parameters
        ----------
        symbols : Sequence[str]
            Symbol sequence (must be in OPTICAL_SYMBOL_VOCAB).
            
        Returns
        -------
        np.ndarray
            Waveform array, shape (len(symbols) * samples_per_symbol,),
            dtype float32, values in [0, 1].
        """
        if len(symbols) == 0:
            return np.zeros(0, dtype=np.float32)
        
        cfg = self.config
        total_samples = len(symbols) * cfg.samples_per_symbol
        wave = np.full(total_samples, cfg.baseline, dtype=np.float32)
        
        for sym_idx, symbol in enumerate(symbols):
            if symbol not in OPTICAL_SLOT_MAPPING:
                raise ValueError(f"Unknown symbol: '{symbol}'")
            
            slot_i, slot_j = OPTICAL_SLOT_MAPPING[symbol]
            base = sym_idx * cfg.samples_per_symbol
            
            # Set pulse at slot_i
            start_i = base + slot_i * cfg.slot_samples
            end_i = start_i + cfg.slot_samples
            wave[start_i:end_i] = cfg.pulse_amplitude
            
            # Set pulse at slot_j
            start_j = base + slot_j * cfg.slot_samples
            end_j = start_j + cfg.slot_samples
            wave[start_j:end_j] = cfg.pulse_amplitude
        
        return wave
    
    def decode(self, wave: np.ndarray) -> List[str]:
        """Decode optical waveform to symbols using normalized energy matching.
        
        The decode algorithm:
        1. Segment by fixed symbol window (T_sym = 100 samples)
        2. Compute slot energy vector e[k] for each window
        3. Normalize: e = (e - min(e)) / (sum(e) + eps) to resist drift
        4. Find top-2 slots and match to symbol table
        
        Parameters
        ----------
        wave : np.ndarray
            Input waveform.
            
        Returns
        -------
        List[str]
            Decoded symbols. Returns "?" for low-confidence decodes.
        """
        if wave.size == 0:
            return []
        
        cfg = self.config
        num_symbols = wave.size // cfg.samples_per_symbol
        
        if num_symbols == 0:
            return []
        
        decoded: List[str] = []
        eps = 1e-8
        
        for sym_idx in range(num_symbols):
            base = sym_idx * cfg.samples_per_symbol
            window = wave[base : base + cfg.samples_per_symbol]
            
            # Compute slot energies
            slot_energies = np.zeros(cfg.num_slots, dtype=np.float32)
            for slot_idx in range(cfg.num_slots):
                slot_start = slot_idx * cfg.slot_samples
                slot_end = slot_start + cfg.slot_samples
                slot_data = window[slot_start:slot_end]
                slot_energies[slot_idx] = float(np.sum(slot_data ** 2))
            
            # Normalize to resist drift
            # Step 1: Remove bias (subtract min)
            slot_energies = slot_energies - np.min(slot_energies)
            # Step 2: Normalize by total (resist gain drift)
            total_energy = np.sum(slot_energies)
            if total_energy > eps:
                slot_energies = slot_energies / total_energy
            
            # Find top-2 slots
            top2_indices = np.argsort(slot_energies)[-2:]
            slot_i, slot_j = sorted(top2_indices)
            
            # Look up symbol
            key = (int(slot_i), int(slot_j))
            if key in _SLOTS_TO_SYMBOL:
                decoded.append(_SLOTS_TO_SYMBOL[key])
            else:
                # Unknown pattern - output blank marker
                decoded.append("?")
        
        return decoded
    
    def channel(self, wave: np.ndarray, ood_params: Dict[str, Any]) -> np.ndarray:
        """Apply channel distortions to simulate OOD conditions.
        
        Supported OOD parameters:
        - noise_std: Additive Gaussian noise standard deviation
        - gain_drift: Multiplicative gain drift (linear ramp)
        - bias_drift: Additive baseline drift
        - bandwidth_hz: Low-pass filter cutoff (simulates sensor bandwidth)
        - dropout_prob: Random dropout probability (segment zeroing)
        
        All distortions are applied in order, then clipped to [0, 1].
        
        Parameters
        ----------
        wave : np.ndarray
            Input waveform.
        ood_params : Dict[str, Any]
            OOD distortion parameters.
            
        Returns
        -------
        np.ndarray
            Distorted waveform, clipped to [0, 1].
        """
        if wave.size == 0:
            return wave.copy()
        
        result = wave.astype(np.float32).copy()
        rng = np.random.default_rng(ood_params.get("seed"))
        
        # 1. Additive noise
        noise_std = ood_params.get("noise_std", 0.0)
        if noise_std > 0:
            noise = rng.normal(0, noise_std, size=result.shape).astype(np.float32)
            result = result + noise
        
        # 2. Gain drift (linear ramp)
        gain_drift = ood_params.get("gain_drift", 0.0)
        if gain_drift != 0:
            # Create linear ramp from (1 - drift/2) to (1 + drift/2)
            ramp = np.linspace(1 - gain_drift / 2, 1 + gain_drift / 2, len(result))
            result = result * ramp.astype(np.float32)
        
        # 3. Bias drift (linear ramp)
        bias_drift = ood_params.get("bias_drift", 0.0)
        if bias_drift != 0:
            # Create linear ramp from 0 to bias_drift
            ramp = np.linspace(0, bias_drift, len(result))
            result = result + ramp.astype(np.float32)
        
        # 4. Bandwidth limiting (simple moving average low-pass)
        bandwidth_hz = ood_params.get("bandwidth_hz", 0)
        if bandwidth_hz > 0 and bandwidth_hz < self.config.sample_rate / 2:
            # Simple approximation: moving average with window based on cutoff
            window_size = max(1, int(self.config.sample_rate / bandwidth_hz / 2))
            if window_size > 1:
                kernel = np.ones(window_size, dtype=np.float32) / window_size
                result = np.convolve(result, kernel, mode="same").astype(np.float32)
        
        # 5. Dropout (random segment zeroing)
        dropout_prob = ood_params.get("dropout_prob", 0.0)
        if dropout_prob > 0:
            # Apply dropout at slot level for more realistic behavior
            num_slots_total = len(result) // self.config.slot_samples
            for slot_idx in range(num_slots_total):
                if rng.random() < dropout_prob:
                    start = slot_idx * self.config.slot_samples
                    end = start + self.config.slot_samples
                    result[start:end] = 0.0
        
        # Final clip to [0, 1] (physical non-negative constraint + saturation)
        result = np.clip(result, 0.0, 1.0)
        
        return result
    
    def get_frame_params(self) -> Dict[str, int]:
        """Get recommended frame parameters for model training.
        
        Returns parameters aligned to slot structure for proper
        integration with MiniJMamba.
        
        Returns
        -------
        Dict[str, int]
            frame_size and hop_size aligned to slot_samples.
        """
        return {
            "frame_size": self.config.slot_samples,
            "hop_size": self.config.slot_samples,
        }


__all__ = [
    "OpticalConfig",
    "OpticalIntensityDomain", 
    "OPTICAL_SYMBOL_VOCAB",
    "OPTICAL_SLOT_MAPPING",
    "OOD_SYMBOL_HOLDOUT",
]

