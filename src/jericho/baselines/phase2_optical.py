"""Phase2 baseline models for optical domain tasks.

These baselines define the minimum performance thresholds that models
must exceed to pass the Phase2 Gate.

Baselines (frozen, v0.1):
- Task1 (Mirror): Identity copy of input symbols
- Task2 (Bracket): Always output "V" (always predict valid)
- Task3 (Mod): Always output "0" (no computation)

Reference: docs/phase2_light_to_light.md Section 6.2
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from jericho.domains.optical_intensity import (
    OpticalIntensityDomain,
    OpticalConfig,
)


class Phase2IdentityBaseline:
    """Task1 (Mirror) baseline: copy input symbols.
    
    This baseline achieves perfect performance on Mirror task
    but serves as a sanity check for the model pipeline.
    """
    
    def __init__(self, domain: OpticalIntensityDomain | None = None) -> None:
        self.domain = domain or OpticalIntensityDomain()
    
    def __call__(self, input_symbols: Sequence[str]) -> List[str]:
        """Return input symbols unchanged."""
        return list(input_symbols)
    
    def predict_wave(self, input_wave: np.ndarray) -> np.ndarray:
        """Decode input and re-encode (roundtrip)."""
        symbols = self.domain.decode(input_wave)
        return self.domain.encode(symbols)


class Phase2AlwaysVBaseline:
    """Task2 (Bracket) baseline: always output "V" (valid).
    
    Expected accuracy on balanced dataset: ~50%
    This is the minimum threshold models must exceed.
    """
    
    def __init__(self, domain: OpticalIntensityDomain | None = None) -> None:
        self.domain = domain or OpticalIntensityDomain()
    
    def __call__(self, input_symbols: Sequence[str]) -> List[str]:
        """Always return ["V"]."""
        return ["V"]
    
    def predict_wave(self, input_wave: np.ndarray) -> np.ndarray:
        """Return waveform encoding "V"."""
        return self.domain.encode(["V"])


class Phase2AlwaysZeroBaseline:
    """Task3 (Mod) baseline: always output "0".
    
    Expected accuracy: 1/8 = 12.5% (for mod 8)
    This is the minimum threshold models must exceed.
    """
    
    def __init__(self, domain: OpticalIntensityDomain | None = None) -> None:
        self.domain = domain or OpticalIntensityDomain()
    
    def __call__(self, input_symbols: Sequence[str]) -> List[str]:
        """Always return ["0"]."""
        return ["0"]
    
    def predict_wave(self, input_wave: np.ndarray) -> np.ndarray:
        """Return waveform encoding "0"."""
        return self.domain.encode(["0"])


def get_baseline_for_task(task: str, domain: OpticalIntensityDomain | None = None):
    """Get the appropriate baseline for a task.
    
    Parameters
    ----------
    task : str
        Task name: "mirror", "bracket", or "mod"
    domain : OpticalIntensityDomain, optional
        Domain instance to use.
        
    Returns
    -------
    Baseline instance for the specified task.
    """
    domain = domain or OpticalIntensityDomain()
    
    task_lower = task.lower()
    if task_lower in ("mirror", "task1"):
        return Phase2IdentityBaseline(domain)
    elif task_lower in ("bracket", "task2"):
        return Phase2AlwaysVBaseline(domain)
    elif task_lower in ("mod", "task3"):
        return Phase2AlwaysZeroBaseline(domain)
    else:
        raise ValueError(f"Unknown task: {task}")


__all__ = [
    "Phase2IdentityBaseline",
    "Phase2AlwaysVBaseline", 
    "Phase2AlwaysZeroBaseline",
    "get_baseline_for_task",
]

