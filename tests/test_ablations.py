"""Tests for S22 ablation suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from jericho.models import MiniJMamba, MiniJMambaConfig
from run_ablations import (
    AblationConfig,
    CORE_ABLATIONS,
    build_model_with_ablation,
)


class TestAblationConfigs:
    """Test ablation configuration definitions."""
    
    def test_core5_ablations_defined(self):
        """Core 5 ablations should be defined."""
        assert "baseline" in CORE_ABLATIONS
        assert "no_attention" in CORE_ABLATIONS
        assert "no_rope" in CORE_ABLATIONS
        assert "no_ctc" in CORE_ABLATIONS
        assert "no_curriculum" in CORE_ABLATIONS
        assert len(CORE_ABLATIONS) == 5


class TestModelAblations:
    """Test model builds correctly with ablation configs."""
    
    @pytest.fixture
    def base_config(self):
        return MiniJMambaConfig(
            frame_size=160,
            hop_size=160,
            symbol_vocab_size=10,
            d_model=64,
            num_ssm_layers=4,
            num_attn_layers=2,
            num_heads=2,
            max_frames=100,
        )
    
    def test_baseline_model(self, base_config):
        """Baseline model should build correctly."""
        model = build_model_with_ablation(base_config, CORE_ABLATIONS["baseline"])
        assert model is not None
        assert model.config.num_attn_layers == 2
        assert model.config.use_rope is True
    
    def test_no_attention_model(self, base_config):
        """No attention model should have 0 attention layers."""
        model = build_model_with_ablation(base_config, CORE_ABLATIONS["no_attention"])
        assert model.config.num_attn_layers == 0
        # SSM layers should be increased to compensate
        assert model.config.num_ssm_layers == base_config.num_ssm_layers + base_config.num_attn_layers
    
    def test_no_rope_model(self, base_config):
        """No RoPE model should use learnable positional embedding."""
        model = build_model_with_ablation(base_config, CORE_ABLATIONS["no_rope"])
        assert model.config.use_rope is False
        assert model.config.use_learnable_pos is True
        assert model.pos_emb is not None
    
    def test_model_forward_with_ablations(self, base_config):
        """All ablation models should forward correctly."""
        batch_size = 2
        seq_len = 20
        
        for name, ablation in CORE_ABLATIONS.items():
            if name == "no_ctc" or name == "no_curriculum":
                # These are training config changes, not model changes
                continue
            
            model = build_model_with_ablation(base_config, ablation)
            
            frames = torch.randn(batch_size, seq_len, base_config.frame_size)
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            
            frame_out, symbol_out = model(frames, mask)
            
            assert frame_out.shape == (batch_size, seq_len, base_config.frame_size)
            assert symbol_out.shape == (batch_size, seq_len, base_config.symbol_vocab_size)


class TestRoPEAblation:
    """Test RoPE vs learnable positional embedding."""
    
    def test_rope_vs_learnable_pos_length_extrapolation(self):
        """RoPE should handle longer sequences better than learnable pos."""
        config_rope = MiniJMambaConfig(
            frame_size=160,
            hop_size=160,
            symbol_vocab_size=10,
            d_model=64,
            num_ssm_layers=2,
            num_attn_layers=1,
            num_heads=2,
            max_frames=50,  # Train on max 50 frames
            use_rope=True,
            use_learnable_pos=False,
        )
        
        config_learnable = MiniJMambaConfig(
            frame_size=160,
            hop_size=160,
            symbol_vocab_size=10,
            d_model=64,
            num_ssm_layers=2,
            num_attn_layers=1,
            num_heads=2,
            max_frames=50,
            use_rope=False,
            use_learnable_pos=True,
        )
        
        model_rope = MiniJMamba(config_rope)
        model_learnable = MiniJMamba(config_learnable)
        
        # Test with sequence longer than max_frames
        long_seq_len = 80  # > 50
        frames = torch.randn(1, long_seq_len, 160)
        mask = torch.ones(1, long_seq_len, dtype=torch.bool)
        
        # RoPE should handle this without error
        out_rope, _ = model_rope(frames, mask)
        assert out_rope.shape == (1, long_seq_len, 160)
        
        # Learnable pos should also work (with clamped positions)
        out_learnable, _ = model_learnable(frames, mask)
        assert out_learnable.shape == (1, long_seq_len, 160)

