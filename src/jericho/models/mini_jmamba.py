"""Mini-JMamba approximation implemented with PyTorch primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:
    from mamba_ssm import Mamba  # type: ignore

    _MAMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Mamba = None
    _MAMBA_AVAILABLE = False


@dataclass
class MiniJMambaConfig:
    frame_size: int
    hop_size: int
    symbol_vocab_size: int
    d_model: int = 128
    num_ssm_layers: int = 10
    num_attn_layers: int = 2
    num_heads: int = 4
    max_frames: int = 256
    dropout: float = 0.1
    attn_dropout: float = 0.1


class SSMLikeBlock(nn.Module):
    """Light-weight approximation of an SSM layer with optional Mamba backend."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.use_mamba = False
        if _MAMBA_AVAILABLE:
            try:
                self.mamba = Mamba(
                    d_model=d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                )
                self.use_mamba = True
            except Exception:
                self.use_mamba = False
        if not self.use_mamba:
            self.in_proj = nn.Linear(d_model, 2 * d_model)
            self.conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=3,
                padding=1,
                groups=d_model,
            )
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        if self.use_mamba:
            y = self.mamba(y)
        else:
            gate, candidate = self.in_proj(y).chunk(2, dim=-1)
            gate = torch.sigmoid(gate)
            candidate = candidate * gate
            candidate = candidate.transpose(1, 2)
            candidate = self.conv(candidate)
            candidate = torch.relu(candidate)
            candidate = candidate.transpose(1, 2)
            y = self.out_proj(candidate)
            y = self.dropout(y)
        y = residual + y
        if mask is not None:
            y = y.masked_fill(~mask.unsqueeze(-1), 0.0)
        return y


class AttentionBlock(nn.Module):
    """Multi-head self-attention block with feed-forward network."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        y = self.norm1(x)
        key_padding = None
        if mask is not None:
            key_padding = ~mask
        attn_out, _ = self.attn(y, y, y, key_padding_mask=key_padding, need_weights=False)
        y = residual + self.dropout(attn_out)
        residual = y
        y = self.ff(self.norm2(y))
        y = residual + y
        if mask is not None:
            y = y.masked_fill(~mask.unsqueeze(-1), 0.0)
        return y


class MiniJMamba(nn.Module):
    """Stack of SSM-like and attention blocks approximating Mini-JMamba."""

    def __init__(self, config: MiniJMambaConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.frame_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, config.max_frames, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

        total_layers = config.num_ssm_layers + config.num_attn_layers
        if total_layers <= 0:
            raise ValueError("Total number of layers must be positive.")

        attn_positions = set()
        if config.num_attn_layers > 0:
            interval = total_layers // (config.num_attn_layers + 1)
            interval = max(1, interval)
            pos = interval
            for _ in range(config.num_attn_layers):
                attn_positions.add(min(pos, total_layers - 1))
                pos += interval

        self.layers = nn.ModuleList()
        ssm_remaining = config.num_ssm_layers
        attn_remaining = config.num_attn_layers
        for layer_idx in range(total_layers):
            if layer_idx in attn_positions and attn_remaining > 0:
                self.layers.append(
                    AttentionBlock(
                        config.d_model,
                        config.num_heads,
                        dropout=config.dropout,
                        attn_dropout=config.attn_dropout,
                    )
                )
                attn_remaining -= 1
            elif ssm_remaining > 0:
                self.layers.append(SSMLikeBlock(config.d_model, config.dropout))
                ssm_remaining -= 1
            else:
                self.layers.append(
                    AttentionBlock(
                        config.d_model,
                        config.num_heads,
                        dropout=config.dropout,
                        attn_dropout=config.attn_dropout,
                    )
                )

        self.final_norm = nn.LayerNorm(config.d_model)
        self.frame_head = nn.Linear(config.d_model, config.frame_size)
        self.symbol_head = nn.Linear(config.d_model, config.symbol_vocab_size)

    def forward(
        self,
        frames: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        *,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        frames:
            Tensor of shape (batch, seq_len, frame_size).
        padding_mask:
            Bool tensor where ``True`` denotes valid frames. Shape (batch, seq_len).
        """
        batch_size, seq_len, feat_dim = frames.shape
        if feat_dim != self.config.frame_size:
            raise ValueError(
                f"Expected frame_size={self.config.frame_size}, received {feat_dim}"
            )
        if seq_len > self.config.max_frames:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured max_frames {self.config.max_frames}"
            )

        x = self.input_proj(frames)
        x = x + self.pos_emb[:, :seq_len, :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, padding_mask)

        x = self.final_norm(x)
        frame_outputs = self.frame_head(x)
        symbol_logits = self.symbol_head(x)
        if padding_mask is not None:
            frame_outputs = frame_outputs.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
            symbol_logits = symbol_logits.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        if return_hidden:
            return frame_outputs, symbol_logits, x
        return frame_outputs, symbol_logits


__all__ = ["MiniJMamba", "MiniJMambaConfig", "SSMLikeBlock", "AttentionBlock"]

