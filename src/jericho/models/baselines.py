"""Baseline models for comparison: Transformer and LSTM.

These models have similar parameter counts to Mini-JMamba for fair comparison.
The forward interface matches MiniJMamba: (frames, mask, return_hidden) -> (frame_outputs, symbol_logits, hidden_states)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class BaselineConfig:
    """Configuration for baseline models."""
    frame_size: int
    hop_size: int
    symbol_vocab_size: int
    d_model: int = 128
    num_layers: int = 6  # Total layers
    num_heads: int = 4
    max_frames: int = 256
    dropout: float = 0.1
    
    def to_dict(self) -> dict:
        """Convert config to dict for serialization."""
        return asdict(self)


class TransformerBaseline(nn.Module):
    """Transformer baseline with similar parameter count to Mini-JMamba.
    
    Architecture:
    - Frame embedding (linear projection)
    - Sinusoidal positional encoding
    - N Transformer encoder layers
    - CTC output head
    - Frame reconstruction head
    """
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        
        # Frame embedding
        self.frame_embed = nn.Linear(config.frame_size, config.d_model)
        
        # Positional encoding (sinusoidal, not learned)
        self.register_buffer(
            "pos_encoding",
            self._create_sinusoidal_encoding(config.max_frames, config.d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Output heads
        self.symbol_head = nn.Linear(config.d_model, config.symbol_vocab_size)
        self.recon_head = nn.Linear(config.d_model, config.frame_size)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        frames : (B, T, frame_size)
        mask : (B, T) bool, True = valid
        return_hidden : if True, return hidden states as third element
        
        Returns
        -------
        If return_hidden=False: (frame_outputs, symbol_logits)
        If return_hidden=True: (frame_outputs, symbol_logits, hidden_states)
        
        This matches MiniJMamba's interface for drop-in compatibility.
        """
        B, T, _ = frames.shape
        
        # Dynamically extend positional encoding if needed
        if T > self.pos_encoding.size(1):
            new_pe = self._create_sinusoidal_encoding(T, self.config.d_model).to(frames.device)
            self.register_buffer("pos_encoding", new_pe)
        
        # Embed frames
        x = self.frame_embed(frames)  # (B, T, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :]
        x = self.dropout(x)
        
        # Create attention mask for padding
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # Transformer expects True = masked
        
        # Encode
        hidden = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Output heads
        symbol_logits = self.symbol_head(hidden)
        frame_outputs = self.recon_head(hidden)
        
        if return_hidden:
            return frame_outputs, symbol_logits, hidden
        return frame_outputs, symbol_logits


class LSTMBaseline(nn.Module):
    """Bidirectional LSTM baseline with similar parameter count to Mini-JMamba.
    
    Architecture:
    - Frame embedding (linear projection)
    - N bidirectional LSTM layers
    - CTC output head
    - Frame reconstruction head
    """
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        
        # Frame embedding
        self.frame_embed = nn.Linear(config.frame_size, config.d_model)
        
        # Bidirectional LSTM (hidden_size = d_model // 2 so concat = d_model)
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model // 2,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )
        
        # Output heads
        self.symbol_head = nn.Linear(config.d_model, config.symbol_vocab_size)
        self.recon_head = nn.Linear(config.d_model, config.frame_size)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        frames : (B, T, frame_size)
        mask : (B, T) bool, True = valid
        return_hidden : if True, return hidden states as third element
        
        Returns
        -------
        If return_hidden=False: (frame_outputs, symbol_logits)
        If return_hidden=True: (frame_outputs, symbol_logits, hidden_states)
        
        This matches MiniJMamba's interface for drop-in compatibility.
        """
        B, T, _ = frames.shape
        
        # Embed frames
        x = self.frame_embed(frames)  # (B, T, d_model)
        x = self.dropout(x)
        
        # Pack sequence if mask provided
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward
        hidden, _ = self.lstm(x)
        
        # Unpack if packed
        if mask is not None:
            hidden, _ = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True, total_length=T)
        
        # Output heads
        symbol_logits = self.symbol_head(hidden)
        frame_outputs = self.recon_head(hidden)
        
        if return_hidden:
            return frame_outputs, symbol_logits, hidden
        return frame_outputs, symbol_logits


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_comparable_configs(frame_size: int = 160, symbol_vocab_size: int = 12):
    """Create configs with similar parameter counts for fair comparison.
    
    Returns dict of model name -> (model_class, config)
    """
    # Mini-JMamba: d_model=128, 10 SSM + 2 Attn layers ≈ 1M params
    # Transformer: 6 layers with d_model=128 ≈ 1M params
    # LSTM: 4 layers bidirectional with d_model=128 ≈ 1M params
    
    base_config = BaselineConfig(
        frame_size=frame_size,
        hop_size=frame_size,
        symbol_vocab_size=symbol_vocab_size,
        d_model=128,
        num_heads=4,
        max_frames=256,
        dropout=0.1,
    )
    
    transformer_config = BaselineConfig(
        **{**base_config.__dict__, "num_layers": 6}
    )
    
    lstm_config = BaselineConfig(
        **{**base_config.__dict__, "num_layers": 4}
    )
    
    return {
        "transformer": (TransformerBaseline, transformer_config),
        "lstm": (LSTMBaseline, lstm_config),
    }

