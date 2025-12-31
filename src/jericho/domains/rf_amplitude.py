"""RF-Amplitude Domain (ASK Modulation).

Third physical wave domain for cross-domain transfer verification.

Encoding scheme:
- Carrier frequency: 100 kHz (simulated via 1 MHz sampling)
- Modulation: Amplitude Shift Keying (ASK)
- Symbol mapping: amplitude levels (0.1 to 1.0)

This domain is intentionally different from:
- Audio: frequency-based encoding
- IPD: pulse-position encoding

RF uses amplitude modulation on a fixed carrier frequency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RFConfig:
    """Configuration for RF-Amplitude domain."""
    
    # Carrier
    carrier_freq: float = 100_000  # 100 kHz carrier
    sample_rate: int = 1_000_000   # 1 MHz sampling (10× carrier)
    
    # Symbol timing
    symbol_duration: float = 0.001  # 1 ms per symbol (1000 samples)
    
    # Amplitude levels (10 levels for digits + special symbols)
    amplitude_levels: int = 16  # 0-9, (, ), V, X, %, +
    min_amplitude: float = 0.1
    max_amplitude: float = 1.0
    
    # Frame parameters (for model input)
    frame_size: int = 100   # 100 samples per frame = 0.1 ms
    hop_size: int = 100
    
    # Channel simulation
    noise_std: float = 0.0  # Additive noise


# Symbol to amplitude mapping
RF_SYMBOL_TO_AMPLITUDE: Dict[str, float] = {
    "0": 0.10,
    "1": 0.16,
    "2": 0.22,
    "3": 0.28,
    "4": 0.34,
    "5": 0.40,
    "6": 0.46,
    "7": 0.52,
    "8": 0.58,
    "9": 0.64,
    "(": 0.70,
    ")": 0.76,
    "V": 0.82,
    "X": 0.88,
    "%": 0.94,
    "+": 1.00,
}

RF_AMPLITUDE_TO_SYMBOL: Dict[float, str] = {v: k for k, v in RF_SYMBOL_TO_AMPLITUDE.items()}


class RFAmplitudeDomain:
    """RF-Amplitude domain encoder/decoder using ASK modulation."""
    
    def __init__(self, config: Optional[RFConfig] = None):
        self.config = config or RFConfig()
        self.samples_per_symbol = int(self.config.sample_rate * self.config.symbol_duration)
        
        # Create amplitude lookup
        self.sym2amp = RF_SYMBOL_TO_AMPLITUDE.copy()
        self.amp2sym = RF_AMPLITUDE_TO_SYMBOL.copy()
        
        # Precompute carrier wave for one symbol
        t = np.arange(self.samples_per_symbol) / self.config.sample_rate
        self.carrier = np.sin(2 * np.pi * self.config.carrier_freq * t).astype(np.float32)
    
    def encode(self, symbols: List[str]) -> np.ndarray:
        """Encode symbols to RF waveform using ASK modulation.
        
        Args:
            symbols: List of symbols to encode
            
        Returns:
            RF waveform (1D numpy array)
        """
        if not symbols:
            return np.zeros(0, dtype=np.float32)
        
        total_samples = len(symbols) * self.samples_per_symbol
        wave = np.zeros(total_samples, dtype=np.float32)
        
        for i, sym in enumerate(symbols):
            if sym not in self.sym2amp:
                raise ValueError(f"Unknown symbol: {sym}")
            
            amplitude = self.sym2amp[sym]
            start = i * self.samples_per_symbol
            end = start + self.samples_per_symbol
            
            # ASK: carrier × amplitude
            wave[start:end] = self.carrier * amplitude
        
        # Add channel noise if configured
        if self.config.noise_std > 0:
            wave += np.random.randn(len(wave)).astype(np.float32) * self.config.noise_std
        
        return wave
    
    def decode(self, wave: np.ndarray) -> List[str]:
        """Decode RF waveform back to symbols.
        
        Uses envelope detection (absolute value + low-pass) to extract amplitude.
        
        Args:
            wave: RF waveform
            
        Returns:
            List of decoded symbols
        """
        if wave.size == 0:
            return []
        
        num_symbols = wave.size // self.samples_per_symbol
        symbols = []
        
        for i in range(num_symbols):
            start = i * self.samples_per_symbol
            end = start + self.samples_per_symbol
            segment = wave[start:end]
            
            # Envelope detection: RMS amplitude
            envelope = np.sqrt(np.mean(segment ** 2)) * np.sqrt(2)  # Convert RMS to peak
            
            # Find closest amplitude level
            best_sym = "?"
            best_dist = float("inf")
            
            for amp, sym in self.amp2sym.items():
                dist = abs(envelope - amp)
                if dist < best_dist:
                    best_dist = dist
                    best_sym = sym
            
            symbols.append(best_sym)
        
        return symbols
    
    def frame(self, wave: np.ndarray) -> np.ndarray:
        """Convert waveform to frames for model input.
        
        Args:
            wave: RF waveform
            
        Returns:
            Framed waveform (num_frames, frame_size)
        """
        num_frames = len(wave) // self.config.frame_size
        if num_frames == 0:
            return np.zeros((1, self.config.frame_size), dtype=np.float32)
        
        frames = wave[:num_frames * self.config.frame_size].reshape(num_frames, self.config.frame_size)
        return frames.astype(np.float32)
    
    def unframe(self, frames: np.ndarray) -> np.ndarray:
        """Convert frames back to waveform.
        
        Args:
            frames: Framed data (num_frames, frame_size)
            
        Returns:
            Waveform (1D array)
        """
        return frames.flatten()


def test_rf_roundtrip():
    """Quick test for RF domain roundtrip."""
    domain = RFAmplitudeDomain()
    
    test_cases = [
        ["(", "(", ")", ")"],
        ["V"],
        ["X"],
        ["1", "2", "3"],
        ["(", ")", "(", ")"],
    ]
    
    passed = 0
    for symbols in test_cases:
        wave = domain.encode(symbols)
        decoded = domain.decode(wave)
        
        if decoded == symbols:
            passed += 1
            print(f"[OK] {symbols} -> {decoded}")
        else:
            print(f"[FAIL] {symbols} -> {decoded}")
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


if __name__ == "__main__":
    test_rf_roundtrip()

