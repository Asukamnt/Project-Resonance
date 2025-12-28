"""Mini-JMamba approximation implemented with PyTorch primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from mamba_ssm import Mamba  # type: ignore

    _MAMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Mamba = None
    _MAMBA_AVAILABLE = False


# =============================================================================
# RoPE (Rotary Position Embedding) - 支持任意长度的相对位置编码
# =============================================================================


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding for relative position encoding.
    
    RoPE encodes position by rotating query/key vectors, enabling:
    - Relative position awareness (not absolute)
    - Extrapolation to arbitrary sequence lengths
    - No learnable parameters (pure geometric transformation)
    """

    def __init__(self, dim: int, base: float = 10000.0):
        """
        Parameters
        ----------
        dim:
            Head dimension (must be even).
        base:
            Base for the exponential frequency computation.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dimension, got {dim}")
        self.dim = dim
        self.base = base
        # Precompute inverse frequencies: 1 / (base^(2i/dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings for a given sequence length.
        
        Returns
        -------
        cos, sin:
            Both of shape (1, seq_len, 1, dim).
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        cos = emb.cos().unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim)
        sin = emb.sin().unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.
    
    Parameters
    ----------
    q, k:
        Shape (batch, seq_len, num_heads, head_dim).
    cos, sin:
        Shape (1, seq_len, 1, head_dim).
    
    Returns
    -------
    q_embed, k_embed:
        Rotated query and key tensors.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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
    # Ablation options
    use_rope: bool = True  # If False, use learnable absolute positional embedding
    use_learnable_pos: bool = False  # If True (and use_rope=False), add learnable pos emb


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
    """Multi-head self-attention block with optional RoPE and feed-forward network.
    
    By default uses Rotary Position Embedding (RoPE) for relative position encoding,
    enabling generalization to arbitrary sequence lengths.
    
    For ablation studies, can disable RoPE (use_rope=False).
    """

    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float, 
        attn_dropout: float,
        use_rope: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        if self.head_dim * num_heads != d_model:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.norm1 = nn.LayerNorm(d_model)
        
        # Q/K/V projections (separate for RoPE application)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RoPE for relative position encoding (optional for ablation)
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim)
        else:
            self.rope = None
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        y = self.norm1(x)
        
        # Compute Q, K, V
        q = self.q_proj(y).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(y).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(y).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE to Q and K (if enabled)
        if self.rope is not None:
            cos, sin = self.rope(seq_len, x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        
        # Apply key padding mask
        if mask is not None:
            # mask: (B, T) where True = valid, False = padding
            # Need to create attention mask: (B, 1, 1, T)
            attn_mask = ~mask.unsqueeze(1).unsqueeze(2)  # True = masked
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # (B, H, T, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)
        
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
        
        # Optional learnable positional embedding (for ablation: no_rope)
        # Only used when use_rope=False and use_learnable_pos=True
        if not config.use_rope and config.use_learnable_pos:
            self.pos_emb = nn.Embedding(config.max_frames, config.d_model)
        else:
            self.pos_emb = None
        
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
                        use_rope=config.use_rope,  # Pass RoPE config for ablation
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
                        use_rope=config.use_rope,
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
        # NOTE: 移除 max_frames 硬限制，现在支持任意长度
        # (max_frames 仅作为配置参考，不再强制检查)

        x = self.input_proj(frames)
        
        # Optional learnable positional embedding (for ablation)
        if self.pos_emb is not None:
            # Clamp position indices to max_frames for OOD lengths
            positions = torch.arange(seq_len, device=frames.device)
            positions = positions.clamp(max=self.config.max_frames - 1)
            x = x + self.pos_emb(positions).unsqueeze(0)
        
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


__all__ = [
    "MiniJMamba",
    "MiniJMambaConfig",
    "SSMLikeBlock",
    "AttentionBlock",
    "RotaryPositionEmbedding",
    "apply_rotary_pos_emb",
]

