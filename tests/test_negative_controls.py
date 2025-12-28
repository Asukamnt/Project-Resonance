"""Tests for S7 negative controls."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from negative_controls import (
    label_shuffle_control,
    phase_scramble_control,
    random_mapping_control,
    run_all_controls,
    encode_symbols_with_mapping,
    phase_scramble,
    create_random_mapping,
)
from jericho.symbols import SYMBOL2FREQ
from jericho.scorer import decode_wave_to_symbols


class TestNegativeControlHelpers:
    """Test helper functions for negative controls."""
    
    def test_encode_with_custom_mapping(self):
        """Test encoding with custom frequency mapping."""
        custom_mapping = {"A": 1000.0, "B": 2000.0}
        wave = encode_symbols_with_mapping(["A", "B"], custom_mapping)
        
        # Should produce non-empty wave
        assert wave.size > 0
        
        # Should NOT decode correctly with standard mapping
        decoded = decode_wave_to_symbols(wave)
        assert decoded != ["A", "B"], "Custom mapping should not decode correctly"
    
    def test_phase_scramble_preserves_length(self):
        """Test that phase scramble preserves wave length."""
        import numpy as np
        
        wave = np.sin(2 * np.pi * 440 * np.arange(1600) / 16000).astype(np.float32)
        scrambled = phase_scramble(wave)
        
        assert scrambled.shape == wave.shape
    
    def test_create_random_mapping(self):
        """Test random mapping creation."""
        import numpy as np
        
        symbols = ["A", "B", "C"]
        mapping = create_random_mapping(symbols, freq_range=(1000.0, 2000.0))
        
        assert len(mapping) == 3
        assert all(s in mapping for s in symbols)
        assert all(1000.0 <= f <= 2000.0 for f in mapping.values())


class TestLabelShuffleControl:
    """Test label shuffle negative control."""
    
    def test_label_shuffle_mirror(self):
        """Label shuffle should break decoding for mirror task."""
        result = label_shuffle_control("mirror", seed=42, n_samples=50)
        
        assert result.is_valid, f"Label shuffle should pass: acc={result.accuracy:.3f}"
        assert result.accuracy < 0.3, "Accuracy should be low with shuffled mapping"
    
    def test_label_shuffle_bracket(self):
        """Label shuffle should break decoding for bracket task."""
        result = label_shuffle_control("bracket", seed=42, n_samples=50)
        
        assert result.is_valid, f"Label shuffle should pass: acc={result.accuracy:.3f}"
        assert result.accuracy < 0.3, "Accuracy should be low with shuffled mapping"
    
    def test_label_shuffle_mod(self):
        """Label shuffle should break decoding for mod task."""
        result = label_shuffle_control("mod", seed=42, n_samples=50)
        
        assert result.is_valid, f"Label shuffle should pass: acc={result.accuracy:.3f}"
        assert result.accuracy < 0.3, "Accuracy should be low with shuffled mapping"


class TestPhaseScrambleControl:
    """Test phase scramble negative control."""
    
    def test_phase_scramble_mirror_multi_symbol(self):
        """Phase scramble should break multi-symbol sequence decoding."""
        result = phase_scramble_control("mirror", seed=42, n_samples=50)
        
        # Multi-symbol: phase scramble should break segmentation
        assert result.is_valid, f"Phase scramble should pass for mirror: acc={result.accuracy:.3f}"
        assert result.accuracy < 0.5, "Multi-symbol accuracy should be low after phase scramble"
    
    def test_phase_scramble_single_symbol_still_works(self):
        """Phase scramble should NOT break single symbol detection (FFT only needs magnitude)."""
        result = phase_scramble_control("bracket", seed=42, n_samples=50)
        
        # Single symbol: FFT magnitude is preserved, should still decode
        assert result.is_valid, f"Phase scramble should pass for bracket: acc={result.accuracy:.3f}"
        assert result.accuracy > 0.8, "Single symbol should still decode after phase scramble"


class TestRandomMappingControl:
    """Test random frequency mapping negative control."""
    
    def test_random_mapping_mirror(self):
        """Random mapping should break decoding for mirror task."""
        result = random_mapping_control("mirror", seed=42, n_samples=50)
        
        assert result.is_valid, f"Random mapping should pass: acc={result.accuracy:.3f}"
        assert result.accuracy < 0.3, "Accuracy should be very low with random mapping"
    
    def test_random_mapping_bracket(self):
        """Random mapping should break decoding for bracket task."""
        result = random_mapping_control("bracket", seed=42, n_samples=50)
        
        assert result.is_valid, f"Random mapping should pass: acc={result.accuracy:.3f}"
        assert result.accuracy < 0.3, "Accuracy should be very low with random mapping"
    
    def test_random_mapping_mod(self):
        """Random mapping should break decoding for mod task."""
        result = random_mapping_control("mod", seed=42, n_samples=50)
        
        assert result.is_valid, f"Random mapping should pass: acc={result.accuracy:.3f}"
        assert result.accuracy < 0.3, "Accuracy should be very low with random mapping"


class TestRunAllControls:
    """Test the full negative control suite."""
    
    def test_run_all_controls_passes(self):
        """All negative controls should pass."""
        results = run_all_controls(
            tasks=["mirror", "bracket", "mod"],
            controls=["label_shuffle", "phase_scramble", "random_mapping"],
            seed=42,
            n_samples=30,
        )
        
        assert len(results) == 9, "Should have 3 tasks × 3 controls = 9 results"
        
        failed = [r for r in results if not r.is_valid]
        assert len(failed) == 0, f"All controls should pass, but failed: {[(r.task, r.control) for r in failed]}"

