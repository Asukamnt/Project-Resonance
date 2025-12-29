"""Mini-JMamba pipeline for Task3 Arithmetic Mod."""

from __future__ import annotations

from collections import Counter
import math
from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence

from jericho.data import ManifestEntry, synthesise_entry_wave
from jericho.models import (
    MiniJMamba,
    MiniJMambaConfig,
    MultiResolutionSTFTConfig,
    multi_resolution_stft_loss,
)
from jericho.scorer import decode_wave_to_symbols, exact_match
from jericho.task3 import MOD_OPERATOR, parse_mod_expression, target_symbols_for_task3, synthesise_task3_target_wave
from jericho.symbols import GAP_DUR, SR, SYMBOL2FREQ, TONE_DUR, encode_symbols_to_wave

from .mini_jmamba_audio import frame_wave

DIGIT_SYMBOLS = list("0123456789")
TASK3_SYMBOLS = DIGIT_SYMBOLS + [MOD_OPERATOR]
DIGIT_FREQS = torch.tensor([SYMBOL2FREQ[symbol] for symbol in DIGIT_SYMBOLS], dtype=torch.float32)


@dataclass
class Task3TrainingConfig:
    frame_size: int = 160
    hop_size: int = 160
    d_model: int = 128
    num_ssm_layers: int = 10
    num_attn_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    ctc_weight: float = 0.3
    ctc_weight_schedule: str = "linear"
    ctc_weight_start: float = 1.2
    ctc_weight_end: float = 0.4
    frame_ce_weight: float = 3.0
    frame_ce_blank_weight: float = 0.05
    frame_recon_weight: float = 0.35
    symbol_guidance_weight: float = 2.0
    blank_penalty_weight: float = 1.0
    symbol_warmup_epochs: int = 20
    thinking_gap_s: float = 0.5
    thinking_gap_align: int = 160
    answer_window_only: bool = True
    single_digit_remainder: bool = True
    remainder_ce_weight: float = 2.0
    remainder_guidance_weight: float = 0.3
    remainder_guidance_blank_floor: float = 0.0
    answer_digit_mass_floor: float = 0.3
    answer_blank_margin: float = 1.0
    answer_blank_margin_weight: float = 1.0
    remainder_head: str = "attn_hidden"  # 默认使用新的 attention-based head
    remainder_gru_hidden: int = 128
    remainder_attn_heads: int = 4
    remainder_attn_dropout: float = 0.1
    pretrain_remainder_epochs: int = 0
    pretrain_remainder_lr: float = 1e-3
    pretrain_remainder_freeze_backbone: bool = False  # 默认不冻结，端到端训练
    l1_weight: float = 1.0
    stft_fft_sizes: Tuple[int, ...] = (512, 1024, 2048)
    stft_hop_scale: float = 0.25
    stft_win_scale: float = 1.0
    stft_sc_weight: float = 1.0
    stft_mag_weight: float = 1.0
    stft_eps: float = 1e-7
    ctc_blank_id: int = 0
    decoder_segmentation: str = "energy"
    decoder_normalize: bool = True
    render_mode: str = "tone_bank_soft"
    render_weight: float = 1.0
    render_fixed_phase: float = 0.0
    pretrain_mirror_epochs: int = 0
    pretrain_mirror_answer_window_only: bool = False
    pretrain_mirror_ctc_weight: float = 1.2
    pretrain_mirror_frame_ce_weight: float = 2.5
    pretrain_mirror_blank_penalty_weight: float = 1.5
    pretrain_mirror_audio_weight: float = 0.0
    mod_expr_ctc_weight: float = 0.1
    mod_expr_ctc_weight_end: float = 0.1
    mod_expr_frame_ce_weight: float = 0.0
    mod_expr_blank_penalty_weight: float = 0.0
    mod_expr_ctc_weight_start: float | None = None  # optional override for schedule

    def __post_init__(self) -> None:
        self.stft_fft_sizes = tuple(self.stft_fft_sizes)

    def build_stft_config(self) -> MultiResolutionSTFTConfig:
        return MultiResolutionSTFTConfig(
            fft_sizes=self.stft_fft_sizes,
            hop_scale=self.stft_hop_scale,
            win_scale=self.stft_win_scale,
            spectral_convergence_weight=self.stft_sc_weight,
            log_mag_weight=self.stft_mag_weight,
            eps=self.stft_eps,
        )

    def ctc_weight_for_epoch(self, epoch_idx: int, total_epochs: int) -> float:
        schedule = self.ctc_weight_schedule.lower()
        if schedule == "linear":
            if total_epochs <= 1:
                return max(0.0, self.ctc_weight_start)
            frac = min(1.0, max(0.0, epoch_idx / max(1, total_epochs - 1)))
            weight = self.ctc_weight_start + frac * (self.ctc_weight_end - self.ctc_weight_start)
            return max(0.0, weight)
        return max(0.0, self.ctc_weight)

    def ctc_eval_weight(self) -> float:
        schedule = self.ctc_weight_schedule.lower()
        if schedule == "linear":
            return max(0.0, self.ctc_weight_end)
        return max(0.0, self.ctc_weight)


@dataclass
class Task3Sample:
    entry: ManifestEntry
    input_frames: torch.Tensor
    target_frames: torch.Tensor
    frame_count: int
    input_wave: torch.Tensor
    target_wave: torch.Tensor
    target_content_samples: int
    target_ids: List[int]
    expression_ids: List[int]
    target_tokens: List[str]
    expression_frame_symbol_ids: torch.Tensor
    frame_symbol_ids: torch.Tensor
    remainder_value: int
    answer_start_samples: int
    answer_len_samples: int
    answer_start_frame: int
    answer_len_frames: int
    expression_len_samples: int


class Task3Dataset(Dataset):
    def __init__(self, samples: Sequence[Task3Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Task3Sample:
        return self.samples[idx]

    @property
    def frame_counts(self) -> List[int]:
        return [sample.frame_count for sample in self.samples]

    @property
    def entries(self) -> List[ManifestEntry]:
        return [sample.entry for sample in self.samples]


def build_task3_vocab() -> dict[str, int]:
    return {symbol: idx + 1 for idx, symbol in enumerate(TASK3_SYMBOLS)}


def _target_content_length(target_tokens: Sequence[str]) -> int:
    """Compute target waveform length (in samples) before padding."""

    if not target_tokens:
        return 0
    tone_samples = int(round(SR * TONE_DUR))
    gap_samples = int(round(SR * GAP_DUR))
    gaps = max(0, len(target_tokens) - 1)
    return tone_samples * len(target_tokens) + gap_samples * gaps


def _align_length(length: int, align: int) -> int:
    if align <= 0:
        return length
    return int(math.ceil(length / align) * align)


def _frame_symbol_labels(
    target_tokens: Sequence[str],
    vocab: dict[str, int],
    frame_count: int,
    hop_size: int,
    *,
    blank_id: int,
) -> torch.Tensor:
    """Assign per-frame symbol ids (digit vs blank) for auxiliary CE loss."""

    frame_targets = torch.full((frame_count,), blank_id, dtype=torch.long)
    if frame_count == 0 or not target_tokens:
        return frame_targets
    tone_samples = int(round(SR * TONE_DUR))
    gap_samples = int(round(SR * GAP_DUR))
    cursor = 0
    frame_midpoints = (
        torch.arange(frame_count, dtype=torch.long) * hop_size + hop_size // 2
    )
    for idx, symbol in enumerate(target_tokens):
        symbol_id = vocab[symbol]
        tone_start = cursor
        tone_end = tone_start + tone_samples
        tone_mask = (frame_midpoints >= tone_start) & (frame_midpoints < tone_end)
        frame_targets = torch.where(tone_mask, torch.full_like(frame_targets, symbol_id), frame_targets)
        cursor = tone_end
        if idx < len(target_tokens) - 1:
            cursor += gap_samples
    return frame_targets


def build_symbol_frame_templates(
    vocab: dict[str, int],
    frame_size: int,
    *,
    blank_id: int,
) -> torch.Tensor:
    """Precompute frame-sized tone snippets for each digit symbol."""

    num_classes = max([blank_id] + list(vocab.values())) + 1
    templates = torch.zeros(num_classes, frame_size, dtype=torch.float32)
    for symbol, idx in vocab.items():
        wave = encode_symbols_to_wave(
            [symbol],
            gap_dur=0.0,
            fixed_phase=0.0,
        )
        if wave.size < frame_size:
            wave = np.pad(wave, (0, frame_size - wave.size))
        templates[idx, :] = torch.from_numpy(wave[:frame_size])
    templates[blank_id, :] = 0.0
    return templates


def apply_symbol_guidance(
    frame_outputs: torch.Tensor,
    symbol_probs: torch.Tensor,
    templates: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    """Inject differentiable tone templates conditioned on symbol predictions."""

    if weight <= 0:
        return frame_outputs
    if templates.device != frame_outputs.device:
        templates = templates.to(frame_outputs.device)
    guided = torch.einsum("bth,hf->btf", symbol_probs, templates)
    return frame_outputs + weight * guided


def assert_mod_targets_without_percent(
    targets: torch.Tensor,
    percent_id: int | None,
) -> None:
    if percent_id is None:
        raise ValueError("percent_id must be provided to validate mod targets")
    if targets.numel() == 0:
        return
    max_id = int(targets.max().item())
    if max_id >= percent_id:
        offending_idx = int((targets >= percent_id).nonzero(as_tuple=False)[0].item())
        offending_id = int(targets[offending_idx].item())
        raise ValueError(
            "mod CTC targets should not include '%' ids "
            f"(percent_id={percent_id}, offending_id={offending_id}, offending_idx={offending_idx})"
        )


def token_accuracy(gold: list[str], pred: list[str]) -> float:
    """Compute token-level accuracy via normalised edit distance."""

    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    m, n = len(gold), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gold[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    dist = dp[m][n]
    denom = max(m, n)
    return max(0.0, 1.0 - dist / denom)


def render_tone_bank(
    digit_probs: torch.Tensor,
    num_samples: int,
    sr: int = SR,
    phase: float = 0.0,
) -> torch.Tensor:
    """Render a continuous-phase tone mixture for digit probabilities."""

    if digit_probs.dim() != 2:
        raise ValueError("digit_probs must be 2D with shape (B, num_digits)")
    device = digit_probs.device
    dtype = digit_probs.dtype
    probs = digit_probs
    if probs.size(1) == DIGIT_FREQS.numel() + 1:
        probs = probs[:, 1:]
    elif probs.size(1) != DIGIT_FREQS.numel():
        raise ValueError("digit_probs must have 10 digits (with or without blank)")

    probs = probs.clamp_min(0.0)
    mass = probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    probs = probs / mass

    time_axis = torch.arange(num_samples, device=device, dtype=dtype) / float(sr)
    freqs = DIGIT_FREQS.to(device=device, dtype=dtype).view(1, -1, 1)
    phases = torch.as_tensor(phase, device=device, dtype=dtype)
    angles = 2.0 * math.pi * freqs * time_axis.view(1, 1, -1) + phases
    waves = torch.sin(angles)
    mixed = (probs.unsqueeze(-1) * waves).sum(dim=1)
    return 0.8 * mixed * mass


def _pool_answer_digit_probs(
    symbol_probs: torch.Tensor,
    window_mask: torch.Tensor,
) -> torch.Tensor:
    """Average digit probabilities over the answer window (preserve mass)."""

    digit_probs = symbol_probs[..., 1 : len(DIGIT_SYMBOLS) + 1]
    mask = window_mask.unsqueeze(-1).to(symbol_probs.dtype)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return (digit_probs * mask).sum(dim=1) / counts


def compute_answer_blank_margin_loss(
    symbol_logits: torch.Tensor,
    tone_mask: torch.Tensor,
    *,
    digit_ids: list[int],
    blank_id: int,
    margin: float,
) -> torch.Tensor:
    """Margin loss to suppress blank logits on tone frames inside answer window."""

    if margin <= 0.0 or tone_mask.sum() == 0:
        return torch.tensor(0.0, device=symbol_logits.device)
    blank_logit = symbol_logits[..., blank_id]
    digit_logits = symbol_logits[..., digit_ids]
    max_digit_logit = digit_logits.max(dim=-1).values
    raw = F.relu(blank_logit - max_digit_logit + margin)
    tone = tone_mask.to(symbol_logits.dtype)
    denom = tone.sum().clamp_min(1.0)
    return (raw * tone).sum() / denom


class RemainderHead(nn.Module):
    """Attention-pooling head for remainder prediction from hidden states."""

    def __init__(
        self,
        d_model: int,
        num_digits: int = 10,
        hidden_dim: int = 128,
        num_attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_digits = num_digits
        # Learnable query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, num_attn_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_digits),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        expression_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : (B, T, d_model)
            Backbone hidden states
        expression_mask : (B, T)
            Boolean mask where True indicates expression region

        Returns
        -------
        remainder_logits : (B, num_digits)
        """
        batch_size = hidden_states.size(0)
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, d_model)
        # Create key padding mask for attention (True = ignore)
        key_padding_mask = ~expression_mask
        # Apply attention pooling
        pooled, _ = self.attn(
            query, hidden_states, hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # (B, 1, d_model)
        pooled = pooled.squeeze(1)  # (B, d_model)
        pooled = self.norm(pooled)
        return self.mlp(pooled)


def compute_remainder_logits(
    symbol_logits: torch.Tensor,
    expression_mask: torch.Tensor,
    *,
    digit_ids: list[int],
    percent_id: int,
    blank_id: int,
    head: str,
    remainder_gru: nn.GRU | None,
    remainder_linear: nn.Linear | None,
    token_probs: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
    remainder_head_module: RemainderHead | None = None,
) -> torch.Tensor:
    """Return remainder logits using pooled, GRU, or attention-based head.

    New 'attn_hidden' head uses backbone hidden states with attention pooling
    for richer representation and end-to-end gradient flow.
    """

    head = head.lower()

    # P0 改进：使用 hidden states + attention pooling
    if head == "attn_hidden":
        if hidden_states is not None and remainder_head_module is not None:
            return remainder_head_module(hidden_states, expression_mask)
        # fallback to pooled if not available
        head = "pooled"

    subset_ids = list(dict.fromkeys(digit_ids + [percent_id, blank_id]))

    # P0 改进：移除 detach()，允许端到端梯度
    if head == "gru_token":
        if token_probs is not None and token_mask is not None and remainder_gru is not None and remainder_linear is not None:
            subset = token_probs[..., subset_ids]  # 移除 .detach()
            lengths = token_mask.sum(dim=1).clamp_min(1).cpu()
            packed = pack_padded_sequence(subset, lengths, batch_first=True, enforce_sorted=False)
            _, h = remainder_gru(packed)
            last = h[-1]
            return remainder_linear(last)
        # fall back to pooled if modules not provided

    if head == "gru_frame" and remainder_gru is not None and remainder_linear is not None:
        expr_feats = symbol_logits[..., subset_ids]  # 移除 .detach()
        lengths = expression_mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(expr_feats, lengths, batch_first=True, enforce_sorted=False)
        _, h = remainder_gru(packed)
        last = h[-1]
        return remainder_linear(last)

    # pooled fallback - 也移除 detach
    expr_lengths = expression_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    pooled_logits = (symbol_logits * expression_mask.unsqueeze(-1)).sum(dim=1) / expr_lengths
    return pooled_logits[:, digit_ids]


def apply_answer_guidance_mix(
    symbol_probs: torch.Tensor,
    remainder_probs: torch.Tensor,
    answer_start_frames: torch.Tensor,
    answer_len_frames: torch.Tensor,
    *,
    mix_weight: float,
    digit_ids: list[int],
    blank_id: int,
    blank_floor: float,
) -> torch.Tensor:
    """Convexly mix remainder guidance distribution into answer window."""

    if mix_weight <= 0.0 or remainder_probs is None or remainder_probs.numel() == 0:
        return symbol_probs
    # 如果 remainder_probs 维度不匹配 digit_ids（如 100 vs 10），跳过混合
    if remainder_probs.size(-1) != len(digit_ids):
        return symbol_probs
    mixed = symbol_probs.clone()
    vocab = symbol_probs.size(-1)
    floor = max(0.0, float(blank_floor))
    for b in range(symbol_probs.size(0)):
        start = int(answer_start_frames[b].item())
        length = int(answer_len_frames[b].item())
        end = min(start + length, symbol_probs.size(1))
        if end <= start:
            continue
        guide = torch.full(
            (end - start, vocab),
            floor,
            device=symbol_probs.device,
            dtype=symbol_probs.dtype,
        )
        guide[:, digit_ids] = remainder_probs[b].clamp_min(0.0)
        if 0 <= blank_id < vocab:
            guide[:, blank_id] = floor
        guide = guide / guide.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mixed[b, start:end] = (1.0 - mix_weight) * mixed[b, start:end] + mix_weight * guide
    mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return mixed


def build_expression_token_probs(
    symbol_probs: torch.Tensor,
    expression_len_samples: torch.Tensor,
    token_counts: torch.Tensor,
    *,
    hop_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-token probabilities by sampling tone centers along expression window."""

    max_tokens = int(token_counts.max().item()) if token_counts.numel() > 0 else 0
    if max_tokens == 0:
        return torch.zeros(
            symbol_probs.size(0), 0, symbol_probs.size(-1), device=symbol_probs.device, dtype=symbol_probs.dtype
        ), torch.zeros(symbol_probs.size(0), 0, dtype=torch.bool, device=symbol_probs.device)

    token_probs = torch.zeros(
        symbol_probs.size(0),
        max_tokens,
        symbol_probs.size(-1),
        device=symbol_probs.device,
        dtype=symbol_probs.dtype,
    )
    token_mask = torch.zeros(symbol_probs.size(0), max_tokens, dtype=torch.bool, device=symbol_probs.device)
    for b in range(symbol_probs.size(0)):
        count = int(token_counts[b].item())
        if count <= 0:
            continue
        token_mask[b, :count] = True
        expr_len = int(expression_len_samples[b].item())
        stride = expr_len / max(count, 1)
        for t in range(count):
            center_sample = int(round((t + 0.5) * stride))
            frame_idx = center_sample // hop_size
            frame_idx = max(0, min(frame_idx, symbol_probs.size(1) - 1))
            token_probs[b, t] = symbol_probs[b, frame_idx]
    return token_probs, token_mask


def validate_answer_windows(
    answer_start_samples: torch.Tensor,
    answer_len_samples: torch.Tensor,
    wave_lengths: torch.Tensor,
    *,
    context: str,
) -> None:
    """Ensure answer windows are positive length and within valid waveform bounds."""

    if not (answer_start_samples.shape == answer_len_samples.shape == wave_lengths.shape):
        raise ValueError(
            f"{context}: answer window tensors must share the same shape "
            f"(starts={list(answer_start_samples.shape)}, lens={list(answer_len_samples.shape)}, "
            f"waves={list(wave_lengths.shape)})"
        )
    starts = answer_start_samples.to(torch.long)
    lens = answer_len_samples.to(torch.long)
    waves = wave_lengths.to(torch.long)
    if (lens <= 0).any():
        bad_idx = int((lens <= 0).nonzero(as_tuple=False)[0].item())
        raise ValueError(
            f"{context}: answer window length must be >0 "
            f"(idx={bad_idx}, start={int(starts[bad_idx].item())}, len={int(lens[bad_idx].item())})"
        )
    if (starts < 0).any():
        bad_idx = int((starts < 0).nonzero(as_tuple=False)[0].item())
        raise ValueError(
            f"{context}: answer window start must be non-negative "
            f"(idx={bad_idx}, start={int(starts[bad_idx].item())}, len={int(lens[bad_idx].item())})"
        )
    exceeds = starts + lens > waves
    if exceeds.any():
        bad_idx = int(exceeds.nonzero(as_tuple=False)[0].item())
        raise ValueError(
            f"{context}: answer window exceeds wave length "
            f"(idx={bad_idx}, start={int(starts[bad_idx].item())}, "
            f"len={int(lens[bad_idx].item())}, wave_len={int(waves[bad_idx].item())})"
        )


def validate_answer_frame_windows(
    answer_start_frames: torch.Tensor,
    answer_len_frames: torch.Tensor,
    frame_counts: torch.Tensor,
    *,
    context: str,
) -> None:
    """Validate answer windows at frame granularity against available masks."""

    if not (
        answer_start_frames.shape == answer_len_frames.shape == frame_counts.shape
    ):
        raise ValueError(
            f"{context}: answer window frame tensors must share the same shape "
            f"(starts={list(answer_start_frames.shape)}, lens={list(answer_len_frames.shape)}, "
            f"frames={list(frame_counts.shape)})"
        )
    starts = answer_start_frames.to(torch.long)
    lens = answer_len_frames.to(torch.long)
    frames = frame_counts.to(torch.long)
    if (lens <= 0).any():
        bad_idx = int((lens <= 0).nonzero(as_tuple=False)[0].item())
        raise ValueError(
            f"{context}: answer window frame length must be >0 "
            f"(idx={bad_idx}, start_frames={int(starts[bad_idx].item())}, "
            f"len_frames={int(lens[bad_idx].item())})"
        )
    exceeds = starts + lens > frames
    if exceeds.any():
        bad_idx = int(exceeds.nonzero(as_tuple=False)[0].item())
        raise ValueError(
            f"{context}: answer window frames exceed available mask "
            f"(idx={bad_idx}, start_frames={int(starts[bad_idx].item())}, "
            f"len_frames={int(lens[bad_idx].item())}, mask_frames={int(frames[bad_idx].item())})"
        )


def validate_ctc_inputs(
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    context: str,
    start_samples: torch.Tensor | None = None,
    len_samples: torch.Tensor | None = None,
) -> None:
    """Fail fast when CTCLoss would receive empty or undersized inputs."""

    if target_lengths.numel() == 0:
        return
    if input_lengths.shape[0] != target_lengths.shape[0]:
        raise ValueError(
            f"{context}: input_lengths and target_lengths must align "
            f"(inputs={list(input_lengths.shape)}, targets={list(target_lengths.shape)})"
        )
    non_positive = (input_lengths <= 0).nonzero(as_tuple=False)
    if non_positive.numel() > 0:
        bad_idx = int(non_positive[0].item())
        extra = ""
        if start_samples is not None and len_samples is not None and start_samples.numel() > bad_idx:
            extra = (
                f", start={int(start_samples[bad_idx].item())}, len_samples={int(len_samples[bad_idx].item())}"
            )
        raise ValueError(
            f"{context}: CTC input length must be >0 (idx={bad_idx}, "
            f"input_len={int(input_lengths[bad_idx].item())}, "
            f"target_len={int(target_lengths[bad_idx].item())}{extra})"
        )
    too_short = (input_lengths < target_lengths).nonzero(as_tuple=False)
    if too_short.numel() > 0:
        bad_idx = int(too_short[0].item())
        extra = ""
        if start_samples is not None and len_samples is not None and start_samples.numel() > bad_idx:
            extra = (
                f", start={int(start_samples[bad_idx].item())}, len_samples={int(len_samples[bad_idx].item())}"
            )
        raise ValueError(
            f"{context}: CTC input length shorter than targets "
            f"(idx={bad_idx}, input_len={int(input_lengths[bad_idx].item())}, "
            f"target_len={int(target_lengths[bad_idx].item())}{extra})"
        )


def apply_tone_bank_render(
    guided_frames: torch.Tensor,
    symbol_probs: torch.Tensor,
    window_mask: torch.Tensor,
    answer_start_frames: torch.Tensor,
    answer_len_frames: torch.Tensor,
    answer_len_samples: torch.Tensor,
    *,
    mode: str,
    render_weight: float,
    render_fixed_phase: float,
    frame_size: int,
    hop_size: int,
    sr: int = SR,
    answer_digit_mass_floor: float = 0.0,
) -> torch.Tensor:
    """Mix tone-bank rendered frames into the answer window."""

    if mode == "none" or render_weight <= 0.0:
        return guided_frames

    pooled_probs = _pool_answer_digit_probs(symbol_probs, window_mask)
    pooled_mass = pooled_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    digit_mass_floor = min(1.0, max(1e-6, float(answer_digit_mass_floor)))
    scale = torch.clamp_min(digit_mass_floor / pooled_mass, 1.0)
    pooled_probs = pooled_probs * scale
    digit_mass = pooled_probs.sum(dim=-1, keepdim=True).clamp_min(digit_mass_floor)
    if mode == "tone_bank_hard":
        hard_idx = pooled_probs.argmax(dim=-1)
        pooled_probs = digit_mass * F.one_hot(
            hard_idx, num_classes=len(DIGIT_SYMBOLS)
        ).to(pooled_probs.dtype)
    elif mode != "tone_bank_soft":
        raise ValueError(f"Unsupported render_mode: {mode}")

    batch, total_frames, _ = guided_frames.shape
    rendered = guided_frames.clone()
    for b in range(batch):
        frames = int(answer_len_frames[b].item())
        if frames <= 0:
            continue
        start = int(answer_start_frames[b].item())
        if start >= total_frames:
            continue
        samples_needed = max(frames * frame_size, int(answer_len_samples[b].item()))
        wave = render_tone_bank(
            pooled_probs[b : b + 1],
            samples_needed,
            sr=sr,
            phase=render_fixed_phase,
        ).view(frames, frame_size)
        end = min(start + frames, total_frames)
        wave = wave[: end - start].to(device=rendered.device, dtype=rendered.dtype)
        guided_slice = rendered[b, start:end]
        rendered[b, start:end] = guided_slice * (1.0 - render_weight) + wave * render_weight
    return rendered


def frames_to_wave(
    frames: torch.Tensor,
    content_lengths: torch.Tensor,
    frame_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten framed audio to waveform and mask beyond valid content length."""

    batch, frame_count, _ = frames.shape
    wave = frames.reshape(batch, frame_count * frame_size)
    lengths = content_lengths.to(device=frames.device).clamp(min=0, max=wave.size(1))
    time_axis = torch.arange(wave.size(1), device=frames.device).unsqueeze(0)
    mask = time_axis < lengths.unsqueeze(1)
    return wave * mask, lengths


def extract_answer_window(
    wave: torch.Tensor,
    answer_start_samples: torch.Tensor,
    answer_len_samples: torch.Tensor,
    *,
    wave_lengths: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice per-sample answer windows and pad to a common length."""

    if wave.size(0) == 0:
        return wave, answer_len_samples
    if wave_lengths is None:
        wave_lengths = torch.full(
            (wave.size(0),), wave.size(1), dtype=torch.long, device=wave.device
        )
    validate_answer_windows(
        answer_start_samples,
        answer_len_samples,
        wave_lengths,
        context="extract_answer_window",
    )
    max_len = int(answer_len_samples.max().item()) if answer_len_samples.numel() > 0 else 0
    window = torch.zeros(wave.size(0), max_len, device=wave.device, dtype=wave.dtype)
    for b in range(wave.size(0)):
        start = int(answer_start_samples[b].item())
        length = int(answer_len_samples[b].item())
        if length <= 0:
            continue
        segment = wave[b, start : start + length]
        window[b, : segment.numel()] = segment
    return window, answer_len_samples.to(device=wave.device)


def masked_l1_wave(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Length-aware L1 loss on waveforms."""

    if pred_wave.shape != target_wave.shape:
        raise ValueError("pred_wave and target_wave must share the same shape")
    max_len = pred_wave.size(1)
    lengths = lengths.to(device=pred_wave.device).clamp(min=0, max=max_len)
    mask = torch.arange(max_len, device=pred_wave.device).unsqueeze(0) < lengths.unsqueeze(1)
    denom = mask.sum().clamp_min(1)
    return ((pred_wave - target_wave).abs() * mask).sum() / denom


def masked_l2_frames(
    pred_frames: torch.Tensor,
    target_frames: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error on frames with boolean mask."""

    if pred_frames.shape != target_frames.shape:
        raise ValueError("pred_frames and target_frames must share the same shape")
    diff = (pred_frames - target_frames) ** 2
    diff = diff * mask.unsqueeze(-1)
    denom = (mask.sum() * pred_frames.size(-1)).clamp_min(1)
    return diff.sum() / denom


def ctc_greedy_decode(
    logits: torch.Tensor,
    mask: torch.Tensor,
    id_to_symbol: dict[int, str],
    *,
    blank_id: int,
) -> List[List[str]]:
    """Greedy CTC decode that collapses repeats and blanks."""

    max_ids = logits.argmax(dim=-1)
    input_lengths = mask.sum(dim=1).to(torch.long)
    decoded: list[list[str]] = []
    for batch_idx in range(max_ids.size(0)):
        prev = None
        seq: list[str] = []
        for t in range(int(input_lengths[batch_idx].item())):
            idx = int(max_ids[batch_idx, t].item())
            if idx == blank_id or idx == prev:
                prev = idx
                continue
            seq.append(id_to_symbol.get(idx, "<unk>"))
            prev = idx
        decoded.append(seq)
    return decoded


def prepare_task3_samples(
    entries: Sequence[ManifestEntry],
    vocab: dict[str, int],
    frame_size: int,
    hop_size: int,
    *,
    blank_id: int,
    config: Task3TrainingConfig,
) -> List[Task3Sample]:
    samples: list[Task3Sample] = []
    for entry in entries:
        expr_wave_np = synthesise_entry_wave(entry)
        target_tokens = target_symbols_for_task3(entry.symbols)
        answer_wave_np = encode_symbols_to_wave(
            target_tokens,
            rng=np.random.default_rng(entry.sequence_seed + 1),
            fixed_phase=0.0,
        )

        if config.thinking_gap_s <= 0.0:
            input_wave_np = expr_wave_np
            answer_start_samples = 0
            answer_len_samples = len(answer_wave_np)
            expression_len_samples = len(expr_wave_np)
            target_content_samples = answer_len_samples
            target_wave_np = synthesise_task3_target_wave(
                entry.symbols,
                target_length_samples=input_wave_np.size,
                rng=np.random.default_rng(entry.sequence_seed + 1),
                fixed_phase=0.0,
            )
        else:
            align = max(1, config.thinking_gap_align)
            expr_len_aligned = _align_length(len(expr_wave_np), align)
            gap_raw = int(round(config.thinking_gap_s * SR))
            thinking_gap_samples = _align_length(gap_raw, align) if gap_raw > 0 else 0
            ans_len_aligned = _align_length(len(answer_wave_np), align)
            expression_len_samples = expr_len_aligned

            if expr_len_aligned > len(expr_wave_np):
                expr_wave_np = np.pad(expr_wave_np, (0, expr_len_aligned - len(expr_wave_np)))
            if ans_len_aligned > len(answer_wave_np):
                answer_wave_np = np.pad(answer_wave_np, (0, ans_len_aligned - len(answer_wave_np)))

            input_wave_np = np.concatenate(
                [
                    expr_wave_np.astype(np.float32, copy=False),
                    np.zeros(thinking_gap_samples + ans_len_aligned, dtype=np.float32),
                ]
            )
            target_wave_np = np.concatenate(
                [
                    np.zeros(expr_len_aligned + thinking_gap_samples, dtype=np.float32),
                    answer_wave_np.astype(np.float32, copy=False),
                ]
            )
            answer_start_samples = expr_len_aligned + thinking_gap_samples
            answer_len_samples = ans_len_aligned
            target_content_samples = answer_len_samples

        total_len = input_wave_np.size
        if answer_len_samples <= 0:
            raise ValueError(
                f"Invalid answer window length (entry={entry.example_id}, start={answer_start_samples}, len={answer_len_samples}, total_len={total_len})"
            )
        if answer_start_samples < 0 or answer_start_samples + answer_len_samples > total_len:
            raise ValueError(
                f"Answer window out of bounds (entry={entry.example_id}, start={answer_start_samples}, len={answer_len_samples}, total_len={total_len})"
            )
        if config.thinking_gap_s > 0 and answer_start_samples < expression_len_samples:
            raise ValueError(
                f"Thinking gap misaligned: answer overlaps expression (entry={entry.example_id}, answer_start={answer_start_samples}, expr_len={expression_len_samples})"
            )

        input_wave = torch.from_numpy(np.asarray(input_wave_np, dtype=np.float32))
        target_wave = torch.from_numpy(np.asarray(target_wave_np, dtype=np.float32))
        input_frames = frame_wave(input_wave, frame_size, hop_size)
        target_frames = frame_wave(target_wave, frame_size, hop_size)

        # Align shapes if rounding differences occur
        min_frames = min(input_frames.size(0), target_frames.size(0))
        input_frames = input_frames[:min_frames]
        target_frames = target_frames[:min_frames]
        frame_count = min_frames

        target_ids = [vocab[d] for d in target_tokens]
        expression_ids = [vocab[d] for d in entry.symbols]
        frame_symbol_ids = torch.full((frame_count,), blank_id, dtype=torch.long)
        expression_frame_symbol_ids = torch.full((frame_count,), blank_id, dtype=torch.long)
        answer_start_frame = int(answer_start_samples // hop_size)
        answer_end_frame = int(
            math.ceil((answer_start_samples + max(1, answer_len_samples)) / hop_size)
        )
        answer_end_frame = min(answer_end_frame, frame_count)
        window_len_frames = max(0, answer_end_frame - answer_start_frame)
        if window_len_frames > 0:
            local_labels = _frame_symbol_labels(
                target_tokens,
                vocab,
                window_len_frames,
                hop_size,
                blank_id=blank_id,
            )
            frame_symbol_ids[answer_start_frame : answer_start_frame + window_len_frames] = (
                local_labels
            )
            if (frame_symbol_ids[answer_start_frame : answer_start_frame + window_len_frames] == blank_id).all():
                raise AssertionError("answer window frame labels should not be all blank")
        expr_end_frame = int(math.ceil(max(1, expression_len_samples) / hop_size))
        expr_end_frame = min(expr_end_frame, frame_count)
        if expr_end_frame > 0:
            expr_labels = _frame_symbol_labels(
                entry.symbols,
                vocab,
                expr_end_frame,
                hop_size,
                blank_id=blank_id,
            )
            expression_frame_symbol_ids[:expr_end_frame] = expr_labels

        samples.append(
            Task3Sample(
                entry=entry,
                input_frames=input_frames,
                target_frames=target_frames,
                frame_count=frame_count,
                input_wave=input_wave,
                target_wave=target_wave,
                target_content_samples=target_content_samples,
                target_ids=target_ids,
                expression_ids=expression_ids,
                target_tokens=target_tokens,
                expression_frame_symbol_ids=expression_frame_symbol_ids,
                frame_symbol_ids=frame_symbol_ids,
                remainder_value=int("".join(target_tokens)) if target_tokens else 0,
                answer_start_samples=answer_start_samples,
                answer_len_samples=answer_len_samples,
                answer_start_frame=answer_start_frame,
                answer_len_frames=window_len_frames,
                expression_len_samples=expression_len_samples,
            )
        )
    return samples


def collate_task3_train(
    batch: Sequence[Task3Sample],
    frame_size: int,
    *,
    blank_id: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    max_frames = max(sample.frame_count for sample in batch)
    inputs = torch.zeros(len(batch), max_frames, frame_size, dtype=torch.float32)
    targets = torch.zeros(len(batch), max_frames, frame_size, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_frames, dtype=torch.bool)
    target_lengths = []
    all_targets: list[torch.Tensor] = []
    wave_lengths = []
    content_lengths = []
    frame_symbol_targets = torch.full((len(batch), max_frames), blank_id, dtype=torch.long)
    expression_frame_symbol_targets = torch.full((len(batch), max_frames), blank_id, dtype=torch.long)
    remainder_values = []
    expression_targets_list: list[torch.Tensor] = []
    expression_target_lengths: list[int] = []
    expression_len_samples = []
    answer_start_samples = []
    answer_len_samples = []
    for idx, sample in enumerate(batch):
        inputs[idx, : sample.frame_count] = sample.input_frames
        targets[idx, : sample.frame_count] = sample.target_frames
        mask[idx, : sample.frame_count] = True
        target_lengths.append(len(sample.target_ids))
        if sample.target_ids:
            all_targets.append(torch.tensor(sample.target_ids, dtype=torch.long))
        if sample.expression_ids:
            expression_targets_list.append(torch.tensor(sample.expression_ids, dtype=torch.long))
            expression_target_lengths.append(len(sample.expression_ids))
        wave_lengths.append(sample.target_wave.size(0))
        content_lengths.append(sample.target_content_samples)
        expression_len_samples.append(sample.expression_len_samples)
        if sample.frame_symbol_ids.numel() > 0:
            frame_symbol_targets[idx, : sample.frame_count] = sample.frame_symbol_ids
        if sample.expression_frame_symbol_ids.numel() > 0:
            expression_frame_symbol_targets[idx, : sample.frame_count] = sample.expression_frame_symbol_ids
        remainder_values.append(sample.remainder_value)
        answer_start_samples.append(sample.answer_start_samples)
        answer_len_samples.append(sample.answer_len_samples)
    if all_targets:
        ctc_targets = torch.cat(all_targets)
    else:
        ctc_targets = torch.zeros(0, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)
    wave_lengths_tensor = torch.tensor(wave_lengths, dtype=torch.long)
    content_lengths_tensor = torch.tensor(content_lengths, dtype=torch.long)
    remainder_tensor = torch.tensor(remainder_values, dtype=torch.long)
    answer_start_tensor = torch.tensor(answer_start_samples, dtype=torch.long)
    answer_len_tensor = torch.tensor(answer_len_samples, dtype=torch.long)
    return (
        inputs,
        targets,
        mask,
        ctc_targets,
        target_lengths_tensor,
        wave_lengths_tensor,
        content_lengths_tensor,
        frame_symbol_targets,
        expression_frame_symbol_targets,
        remainder_tensor,
        answer_start_tensor,
        answer_len_tensor,
        torch.cat(expression_targets_list) if expression_targets_list else torch.zeros(0, dtype=torch.long),
        torch.tensor(expression_target_lengths, dtype=torch.long),
        torch.tensor(expression_len_samples, dtype=torch.long),
    )


def collate_task3_eval(
    batch: Sequence[Task3Sample],
    frame_size: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    max_frames = max(sample.frame_count for sample in batch)
    inputs = torch.zeros(len(batch), max_frames, frame_size, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_frames, dtype=torch.bool)
    wave_lengths = []
    frame_counts = []
    content_lengths = []
    answer_start_samples = []
    answer_len_samples = []
    expression_len_samples = []
    expression_token_counts = []
    for idx, sample in enumerate(batch):
        inputs[idx, : sample.frame_count] = sample.input_frames
        mask[idx, : sample.frame_count] = True
        wave_lengths.append(sample.target_wave.size(0))
        frame_counts.append(sample.frame_count)
        content_lengths.append(sample.target_content_samples)
        answer_start_samples.append(sample.answer_start_samples)
        answer_len_samples.append(sample.answer_len_samples)
        expression_len_samples.append(sample.expression_len_samples)
        expression_token_counts.append(len(sample.expression_ids))
    return (
        inputs,
        mask,
        torch.tensor(wave_lengths, dtype=torch.long),
        torch.tensor(frame_counts, dtype=torch.long),
        torch.tensor(content_lengths, dtype=torch.long),
        torch.tensor(answer_start_samples, dtype=torch.long),
        torch.tensor(answer_len_samples, dtype=torch.long),
        torch.tensor(expression_len_samples, dtype=torch.long),
        torch.tensor(expression_token_counts, dtype=torch.long),
    )


def evaluate_task3_model(
    model: MiniJMamba,
    dataloader: DataLoader,
    dataset: Task3Dataset,
    device: torch.device,
    *,
    stft_config: MultiResolutionSTFTConfig,
    l1_weight: float,
    id_to_symbol: dict[int, str],
    blank_id: int,
    decoder_segmentation: str,
    symbol_guidance_weight: float,
    symbol_templates: torch.Tensor,
    render_mode: str,
    render_weight: float,
    render_fixed_phase: float,
    percent_id: int | None,
    digit_ids: list[int],
    remainder_guidance_weight: float,
    remainder_guidance_blank_floor: float,
    answer_digit_mass_floor: float,
    remainder_head: str,
    remainder_gru: nn.GRU | None = None,
    remainder_linear: nn.Linear | None = None,
    remainder_head_module: RemainderHead | None = None,
    normalize_audio: bool,
    hop_size: int,
    frame_size: int,
    answer_window_only: bool,
    collect_predictions: bool = False,
    ctc_diagnostics: bool = False,
) -> Tuple[
    float,
    float,
    float,
    float,
    List[dict],
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    model.eval()
    if percent_id is None:
        for idx, symbol in id_to_symbol.items():
            if symbol == MOD_OPERATOR:
                percent_id = idx
                break
    if percent_id is None:
        raise ValueError(
            "evaluate_task3_model: percent_id could not be inferred from id_to_symbol keys "
            f"(available={sorted(id_to_symbol.values())})"
        )
    total_loss = 0.0
    total_weight = 0.0
    matches = 0
    ctc_matches = 0
    expr_matches = 0
    expr_total = 0
    expr_token_acc_sum = 0.0
    hybrid_matches = 0
    hybrid_total = 0
    hybrid_overall_den = 0
    hybrid_parse_ok = 0
    blank_frames = 0
    total_frames = 0
    answer_blank_frames_total = 0
    answer_total_frames_total = 0
    remainder_correct = 0
    remainder_total = 0
    predictions: list[dict] = []
    entry_idx = 0
    diag_printed = 0
    empty_ctc = 0
    total_ctc_samples = 0
    with torch.no_grad():
        for (
            features,
            mask,
            wave_lengths,
            frame_counts,
            _content_lengths,
            answer_start_samples,
            answer_len_samples,
            expression_len_samples,
            expression_token_counts,
        ) in dataloader:
            base_idx = entry_idx
            batch_size = features.size(0)
            features = features.to(device)
            mask = mask.to(device)
            frame_counts = frame_counts.to(device)
            answer_start_samples = answer_start_samples.to(device)
            answer_len_samples = answer_len_samples.to(device)
            expression_len_samples = expression_len_samples.to(device)
            expression_token_counts = expression_token_counts.to(device)
            wave_lengths = wave_lengths.to(device)
            # 使用 return_hidden 获取 hidden states 供 remainder head 使用
            model_out = model(features, mask, return_hidden=True)
            frame_outputs, symbol_logits, hidden_states = model_out
            symbol_probs = symbol_logits.softmax(dim=-1)
            expr_len_frames = (expression_len_samples + hop_size - 1) // hop_size
            expression_mask = torch.zeros_like(mask)
            for b in range(expression_mask.size(0)):
                length = int(expr_len_frames[b].item())
                end = min(length, expression_mask.size(1))
                if end > 0:
                    expression_mask[b, :end] = True
            token_probs, token_mask = build_expression_token_probs(
                symbol_probs,
                expression_len_samples,
                expression_token_counts,
                hop_size=hop_size,
            )
            remainder_logits = compute_remainder_logits(
                symbol_logits,
                expression_mask,
                digit_ids=digit_ids,
                percent_id=percent_id,
                blank_id=blank_id,
                head=remainder_head,
                remainder_gru=remainder_gru,
                remainder_linear=remainder_linear,
                token_probs=token_probs,
                token_mask=token_mask,
                hidden_states=hidden_states,
                remainder_head_module=remainder_head_module,
            )
            remainder_probs = remainder_logits.softmax(dim=-1)
            for b in range(batch_size):
                gold_remainder = dataset.samples[base_idx + b].remainder_value
                if gold_remainder < remainder_probs.size(-1):
                    remainder_total += 1
                    pred_remainder = int(remainder_probs[b].argmax().item())
                    remainder_correct += int(pred_remainder == gold_remainder)

            symbol_probs_render = symbol_probs
            if remainder_guidance_weight > 0 and answer_len_samples.numel() > 0:
                answer_len_frames = (answer_len_samples + hop_size - 1) // hop_size
                symbol_probs_render = apply_answer_guidance_mix(
                    symbol_probs_render,
                    remainder_probs,
                    answer_start_samples // hop_size,
                    answer_len_frames,
                    mix_weight=remainder_guidance_weight,
                    digit_ids=digit_ids,
                    blank_id=blank_id,
                    blank_floor=remainder_guidance_blank_floor,
                )
            guided_frames = apply_symbol_guidance(
                frame_outputs,
                symbol_probs_render,
                symbol_templates,
                weight=symbol_guidance_weight,
            )
            answer_start_frames = answer_start_samples // hop_size
            answer_len_frames = (answer_len_samples + hop_size - 1) // hop_size
            validate_answer_windows(
                answer_start_samples,
                answer_len_samples,
                wave_lengths,
                context="evaluate_task3_model",
            )
            validate_answer_frame_windows(
                answer_start_frames,
                answer_len_frames,
                frame_counts,
                context="evaluate_task3_model_frames",
            )
            if answer_window_only:
                window_mask = torch.zeros_like(mask)
                for b in range(window_mask.size(0)):
                    start = int(answer_start_frames[b].item())
                    length = int(answer_len_frames[b].item())
                    end = min(start + length, window_mask.size(1))
                    if end > start:
                        window_mask[b, start:end] = True
            else:
                window_mask = mask
            rendered_frames = apply_tone_bank_render(
                guided_frames,
                symbol_probs_render,
                window_mask,
                answer_start_frames,
                answer_len_frames,
                answer_len_samples,
                mode=render_mode,
                render_weight=render_weight,
                render_fixed_phase=render_fixed_phase,
                frame_size=frame_size,
                hop_size=hop_size,
                sr=SR,
                answer_digit_mass_floor=answer_digit_mass_floor,
            )

            frame_counts_list = frame_counts.tolist()
            target_frames = torch.zeros_like(frame_outputs)
            for b, frames_count in enumerate(frame_counts_list):
                if frames_count <= 0:
                    continue
                sample = dataset.samples[base_idx + b]
                target_frames[b, :frames_count] = sample.target_frames[:frames_count].to(
                    device
                )
            pred_wave_full, _ = frames_to_wave(rendered_frames, wave_lengths, frame_size)
            target_wave_full, _ = frames_to_wave(target_frames, wave_lengths, frame_size)

            if answer_window_only:
                pred_wave, effective_lengths = extract_answer_window(
                    pred_wave_full,
                    answer_start_samples,
                    answer_len_samples,
                    wave_lengths=wave_lengths,
                )
                target_wave, _ = extract_answer_window(
                    target_wave_full,
                    answer_start_samples,
                    answer_len_samples,
                    wave_lengths=wave_lengths,
                )
            else:
                pred_wave, effective_lengths = pred_wave_full, wave_lengths
                target_wave, _ = target_wave_full, wave_lengths

            stft_loss, _ = multi_resolution_stft_loss(
                pred_wave, target_wave, lengths=effective_lengths, config=stft_config
            )
            l1_loss = masked_l1_wave(pred_wave, target_wave, effective_lengths)
            loss = l1_weight * l1_loss + stft_loss
            total_loss += loss.item() * batch_size
            total_weight += batch_size

            if answer_window_only:
                max_win = int(answer_len_frames.max().item()) if answer_len_frames.numel() > 0 else 0
                window_logits = torch.zeros(
                    symbol_logits.size(0),
                    max_win,
                    symbol_logits.size(-1),
                    device=symbol_logits.device,
                    dtype=symbol_logits.dtype,
                )
                window_mask_ctc = torch.zeros(
                    symbol_logits.size(0),
                    max_win,
                    device=symbol_logits.device,
                    dtype=torch.bool,
                )
                for b in range(symbol_logits.size(0)):
                    start = int(answer_start_frames[b].item())
                    length = int(answer_len_frames[b].item())
                    end = min(start + length, symbol_logits.size(1))
                    win_len = max(0, end - start)
                    if win_len > 0:
                        window_logits[b, :win_len] = symbol_logits[b, start:end]
                        window_mask_ctc[b, :win_len] = True
                ctc_preds = ctc_greedy_decode(window_logits, window_mask_ctc, id_to_symbol, blank_id=blank_id)
            else:
                ctc_preds = ctc_greedy_decode(symbol_logits, mask, id_to_symbol, blank_id=blank_id)
            argmax_ids = symbol_probs.argmax(dim=-1)
            blank_frames += (((argmax_ids == blank_id) & window_mask).sum().item())
            total_frames += window_mask.sum().item()
            answer_blank_frames_total += ((argmax_ids == blank_id) & window_mask).sum().item()
            answer_total_frames_total += window_mask.sum().item()
            total_ctc_samples += len(ctc_preds)
            for seq in ctc_preds:
                if len(seq) == 0:
                    empty_ctc += 1

            expr_len_frames = (expression_len_samples + hop_size - 1) // hop_size
            max_expr = int(expr_len_frames.max().item()) if expr_len_frames.numel() > 0 else 0
            expr_logits = torch.zeros(
                symbol_logits.size(0),
                max_expr,
                symbol_logits.size(-1),
                device=symbol_logits.device,
                dtype=symbol_logits.dtype,
            )
            expr_mask = torch.zeros(
                symbol_logits.size(0),
                max_expr,
                device=symbol_logits.device,
                dtype=torch.bool,
            )
            for b in range(symbol_logits.size(0)):
                length = int(expr_len_frames[b].item())
                end = min(length, symbol_logits.size(1))
                if end > 0:
                    expr_logits[b, :end] = symbol_logits[b, :end]
                    expr_mask[b, :end] = True
            expr_preds = ctc_greedy_decode(expr_logits, expr_mask, id_to_symbol, blank_id=blank_id)

            outputs_np = pred_wave.cpu().numpy()
            lengths_np = effective_lengths.cpu().numpy()
            for b, frames_count in enumerate(frame_counts_list):
                sample = dataset.samples[base_idx + b]
                wave_len = int(lengths_np[b])
                frames_flat = outputs_np[b]
                pred_wave_segment = frames_flat[:wave_len]
                pred_wave_segment = np.clip(pred_wave_segment, -1.0, 1.0).astype(np.float32)
                if normalize_audio and pred_wave_segment.size > 0:
                    peak = float(np.max(np.abs(pred_wave_segment)))
                    if peak > 1e-4:
                        pred_wave_segment = pred_wave_segment / peak
                pred_symbols = decode_wave_to_symbols(
                    pred_wave_segment,
                    segmentation=decoder_segmentation,
                )
                em = exact_match(pred_symbols, sample.target_tokens)
                if em == 1.0:
                    matches += 1

                ctc_pred_symbols = ctc_preds[b]
                ctc_em = exact_match(ctc_pred_symbols, sample.target_tokens)
                if ctc_em == 1.0:
                    ctc_matches += 1

                expr_pred_symbols = expr_preds[b]
                expr_em = exact_match(expr_pred_symbols, sample.entry.symbols)
                expr_matches += expr_em
                expr_total += 1
                expr_token_acc_sum += token_accuracy(sample.entry.symbols, expr_pred_symbols)

                try:
                    dividend, divisor = parse_mod_expression(expr_pred_symbols)
                    rem = dividend % divisor
                    rem_tokens = list(str(rem))
                    parse_ok = True
                except Exception:
                    parse_ok = False
                if parse_ok:
                    hybrid_parse_ok += 1
                    hybrid_total += 1
                    hybrid_wave = encode_symbols_to_wave(
                        rem_tokens,
                        fixed_phase=0.0,
                    )
                    if hybrid_wave.size < sample.answer_len_samples:
                        hybrid_wave = np.pad(
                            hybrid_wave, (0, sample.answer_len_samples - hybrid_wave.size), constant_values=0.0
                        )
                    hybrid_wave = hybrid_wave.astype(np.float32, copy=False)
                    hybrid_decoded = decode_wave_to_symbols(
                        hybrid_wave,
                        segmentation=decoder_segmentation,
                    )
                    hybrid_em_local = exact_match(hybrid_decoded, sample.target_tokens)
                    hybrid_matches += hybrid_em_local
                hybrid_overall_den += 1

                if collect_predictions:
                    predictions.append(
                        {
                            "example_id": sample.entry.example_id,
                            "split": sample.entry.split,
                            "gold_symbols": sample.target_tokens,
                            "pred_symbols": pred_symbols,
                            "ctc_pred_symbols": ctc_pred_symbols,
                            "exact_match": em,
                            "ctc_exact_match": ctc_em,
                        }
                    )
                if ctc_diagnostics and diag_printed < 5:
                    gold = " ".join(sample.target_tokens)
                    audio_pred = " ".join(pred_symbols)
                    ctc_pred = " ".join(ctc_pred_symbols)
                    print(
                        f"example {sample.entry.example_id}: gold=[{gold}] "
                        f"audio_pred=[{audio_pred}] ctc_pred=[{ctc_pred}] "
                        f"audio_em={em:.1f} ctc_em={ctc_em:.1f}"
                    )
                    diag_printed += 1
            entry_idx += batch_size

    avg_loss = total_loss / max(1.0, total_weight)
    dataset_size = len(dataset)
    em = matches / dataset_size if dataset_size else 0.0
    ctc_em = ctc_matches / dataset_size if dataset_size else 0.0
    blank_rate = blank_frames / max(1.0, total_frames)
    expr_em_eval = expr_matches / max(1.0, expr_total)
    expr_token_acc_eval = expr_token_acc_sum / max(1.0, expr_total)
    hybrid_em_cond = hybrid_matches / max(1.0, hybrid_total)
    hybrid_em_overall = hybrid_matches / max(1.0, hybrid_overall_den)
    hybrid_parse_ok_rate = hybrid_parse_ok / max(1.0, hybrid_overall_den)
    answer_blank_rate = answer_blank_frames_total / max(1.0, answer_total_frames_total)
    answer_ctc_empty_rate = empty_ctc / max(1.0, total_ctc_samples)
    remainder_acc_eval = remainder_correct / max(1.0, remainder_total)
    return (
        avg_loss,
        em,
        ctc_em,
        blank_rate,
        predictions,
        expr_em_eval,
        expr_token_acc_eval,
        hybrid_em_cond,
        hybrid_em_overall,
        hybrid_parse_ok_rate,
        remainder_acc_eval,
        answer_blank_rate,
        answer_ctc_empty_rate,
    )


def compute_task3_baseline_stats(dataset: Task3Dataset) -> dict[str, float | str]:
    """Compute simple Task3 baselines (always-0, majority remainder)."""

    if len(dataset) == 0:
        return {
            "baseline_em": 0.0,
            "baseline_type": "none",
            "baseline_always_zero_em": 0.0,
            "baseline_majority_value": "",
            "baseline_majority_em": 0.0,
        }
    remainders = ["".join(sample.target_tokens) for sample in dataset.samples]
    counts = Counter(remainders)
    total = len(remainders)
    majority_value, majority_count = counts.most_common(1)[0]
    majority_em = majority_count / total
    always_zero_em = counts.get("0", 0) / total
    if majority_em >= always_zero_em:
        baseline_em = majority_em
        baseline_type = "majority"
    else:
        baseline_em = always_zero_em
        baseline_type = "always_zero"
    return {
        "baseline_em": baseline_em,
        "baseline_type": baseline_type,
        "baseline_always_zero_em": always_zero_em,
        "baseline_majority_value": majority_value,
        "baseline_majority_em": majority_em,
    }


def mini_jmamba_task3_pipeline(
    train_entries: Sequence[ManifestEntry],
    eval_entries: Sequence[ManifestEntry],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    config: Task3TrainingConfig | None = None,
    mod_lr_factor: float = 1.0,
) -> Tuple[List[dict], dict]:
    """Train and evaluate Mini-JMamba on Task3 data."""

    if config is None:
        config = Task3TrainingConfig()
    if config.pretrain_mirror_answer_window_only and config.pretrain_mirror_epochs > 0:
        raise ValueError(
            "pretrain_mirror_answer_window_only is not supported: mirror pretraining always supervises full expressions."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab = build_task3_vocab()
    digit_ids = [vocab[d] for d in DIGIT_SYMBOLS]
    percent_id = vocab.get(MOD_OPERATOR)
    if percent_id is None:
        raise ValueError(f"Task3 vocab missing '%' symbol; vocab keys={sorted(vocab.keys())}")
    symbol_templates = build_symbol_frame_templates(
        vocab,
        config.frame_size,
        blank_id=config.ctc_blank_id,
    ).to(device)
    train_samples = prepare_task3_samples(
        train_entries,
        vocab,
        config.frame_size,
        config.hop_size,
        blank_id=config.ctc_blank_id,
        config=config,
    )
    eval_samples = prepare_task3_samples(
        eval_entries,
        vocab,
        config.frame_size,
        config.hop_size,
        blank_id=config.ctc_blank_id,
        config=config,
    )
    train_dataset = Task3Dataset(train_samples)
    eval_dataset = Task3Dataset(eval_samples)
    id_to_symbol = {idx: symbol for symbol, idx in vocab.items()}
    baseline_stats = compute_task3_baseline_stats(eval_dataset)

    if len(train_dataset) == 0:
        raise SystemExit("Train split is empty; cannot train mini_jmamba on Task3.")
    if len(eval_dataset) == 0:
        raise SystemExit("Evaluation split is empty.")

    max_frames = max(max(train_dataset.frame_counts), max(eval_dataset.frame_counts))
    model_config = MiniJMambaConfig(
        frame_size=config.frame_size,
        hop_size=config.hop_size,
        symbol_vocab_size=len(vocab) + 1,
        d_model=config.d_model,
        num_ssm_layers=config.num_ssm_layers,
        num_attn_layers=config.num_attn_layers,
        num_heads=config.num_heads,
        max_frames=max_frames,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
    )
    model = MiniJMamba(model_config).to(device)
    frame_ce_class_weights = torch.ones(
        model_config.symbol_vocab_size,
        device=device,
        dtype=torch.float32,
    )
    frame_ce_class_weights[config.ctc_blank_id] = config.frame_ce_blank_weight
    remainder_head_type = config.remainder_head.lower()
    if remainder_head_type not in {"gru_token", "gru_frame", "pooled", "attn_hidden"}:
        raise ValueError(f"Unsupported remainder_head: {config.remainder_head}")
    subset_dim = len(set(digit_ids + [percent_id, config.ctc_blank_id]))
    remainder_gru: nn.GRU | None = None
    remainder_linear: nn.Linear | None = None
    remainder_head_module: RemainderHead | None = None

    if remainder_head_type == "attn_hidden":
        # 新的 attention-based head，使用 backbone hidden states
        # 使用 100 类支持两位数结果 (0-99)
        remainder_head_module = RemainderHead(
            d_model=config.d_model,
            num_digits=100,
            hidden_dim=config.remainder_gru_hidden,
            num_attn_heads=config.remainder_attn_heads,
            dropout=config.remainder_attn_dropout,
        ).to(device)
        params = list(model.parameters()) + list(remainder_head_module.parameters())
    elif remainder_head_type in {"gru_token", "gru_frame"}:
        remainder_gru = nn.GRU(
            input_size=subset_dim,
            hidden_size=config.remainder_gru_hidden,
            batch_first=True,
        ).to(device)
        remainder_linear = nn.Linear(config.remainder_gru_hidden, len(digit_ids)).to(device)
        params = list(model.parameters()) + list(remainder_gru.parameters()) + list(remainder_linear.parameters())
    else:
        params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    ctc_loss_fn = nn.CTCLoss(blank=config.ctc_blank_id, zero_infinity=True)
    stft_config = config.build_stft_config()
    render_mode_eval = (
        config.render_mode
        if config.render_mode != "none"
        and config.render_weight > 0.0
        and epochs > config.symbol_warmup_epochs
        else "none"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_task3_train(
            batch,
            config.frame_size,
            blank_id=config.ctc_blank_id,
        ),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_task3_eval(batch, config.frame_size),
    )

    def apply_mod_lr_factor(opt: torch.optim.Optimizer, factor: float) -> None:
        if factor <= 0:
            return
        for param_group in opt.param_groups:
            param_group["lr"] = param_group.get("lr", lr) * factor

    (
        loss_pre,
        em_pre,
        ctc_em_pre,
        blank_pre,
        _,
        expr_ctc_em_eval_pre,
        expr_token_acc_eval_pre,
        hybrid_em_cond_pre,
        hybrid_em_overall_pre,
        hybrid_parse_ok_rate_pre,
        remainder_acc_eval_pre,
        answer_blank_rate_pre,
        answer_ctc_empty_rate_pre,
    ) = evaluate_task3_model(
        model,
        dataloader=eval_loader,
        dataset=eval_dataset,
        device=device,
        stft_config=stft_config,
        l1_weight=config.l1_weight,
        id_to_symbol=id_to_symbol,
        blank_id=config.ctc_blank_id,
        decoder_segmentation=config.decoder_segmentation,
        symbol_guidance_weight=config.symbol_guidance_weight,
        symbol_templates=symbol_templates,
        render_mode=render_mode_eval,
        render_weight=config.render_weight,
        render_fixed_phase=config.render_fixed_phase,
        percent_id=percent_id,
        digit_ids=digit_ids,
        remainder_guidance_weight=config.remainder_guidance_weight,
        remainder_guidance_blank_floor=config.remainder_guidance_blank_floor,
        answer_digit_mass_floor=config.answer_digit_mass_floor,
        remainder_head=config.remainder_head,
        remainder_gru=remainder_gru,
        remainder_linear=remainder_linear,
        remainder_head_module=remainder_head_module,
        normalize_audio=config.decoder_normalize,
        hop_size=config.hop_size,
        frame_size=config.frame_size,
        answer_window_only=config.answer_window_only,
        collect_predictions=False,
    )
    baseline_margin = em_pre - baseline_stats["baseline_em"]
    print(
        "[mini_jmamba][task3] pre-training "
        f"loss={loss_pre:.6f} em={em_pre:.4f} "
        f"ctc_em={ctc_em_pre:.4f} blank_rate={blank_pre:.3f} "
        f"baseline_em={baseline_stats['baseline_em']:.4f} "
        f"margin={baseline_margin:.4f}"
    )
    print(
        f"[mini_jmamba][task3] pre-training diag "
        f"expr_ctc_em_eval={expr_ctc_em_eval_pre:.4f} "
        f"expr_token_acc_eval={expr_token_acc_eval_pre:.4f} "
        f"hybrid_em_overall={hybrid_em_overall_pre:.4f} "
        f"hybrid_em_cond={hybrid_em_cond_pre:.4f} "
        f"hybrid_parse_ok_rate={hybrid_parse_ok_rate_pre:.4f} "
        f"answer_blank_rate={answer_blank_rate_pre:.4f} "
        f"answer_ctc_empty_rate={answer_ctc_empty_rate_pre:.4f}"
    )
    print(
        "[mini_jmamba][task3] baseline always_zero="
        f"{baseline_stats['baseline_always_zero_em']:.4f} "
        f"majority({baseline_stats['baseline_majority_value']})="
        f"{baseline_stats['baseline_majority_em']:.4f}"
    )

    mirror_ctc_em: float | None = None
    mirror_expression_token_accuracy: float | None = None
    mirror_blank_rate: float | None = None

    if config.pretrain_mirror_epochs > 0:
        pre_epochs = config.pretrain_mirror_epochs
        for epoch in range(pre_epochs):
            model.train()
            for (
                features,
                _target_frames,
                mask,
                _targets,
                _target_lengths,
                _wave_lengths,
                _content_lengths,
                _frame_symbol_targets,
                expression_frame_symbol_targets,
                _remainder_targets,
                answer_start_samples,
                _answer_len_samples,
                expression_targets,
                expression_target_lengths,
                expression_len_samples,
            ) in train_loader:
                features = features.to(device)
                mask = mask.to(device)
                answer_start_samples = answer_start_samples.to(device)
                expression_targets = expression_targets.to(device)
                expression_target_lengths = expression_target_lengths.to(device)
                expression_len_samples = expression_len_samples.to(device)
                expression_frame_symbol_targets = expression_frame_symbol_targets.to(device)

                frame_outputs, symbol_logits = model(features, mask)
                symbol_probs = symbol_logits.softmax(dim=-1)
                guided_frames = apply_symbol_guidance(
                    frame_outputs,
                    symbol_probs,
                    symbol_templates,
                    weight=config.symbol_guidance_weight,
                )

                expression_len_frames = (expression_len_samples + config.hop_size - 1) // config.hop_size
                expression_mask = torch.zeros_like(mask)
                for b in range(expression_mask.size(0)):
                    length = int(expression_len_frames[b].item())
                    end = min(length, expression_mask.size(1))
                    if end > 0:
                        expression_mask[b, :end] = True

                if expression_mask.sum().item() == 0 or expression_targets.numel() == 0 or (
                    expression_target_lengths > 0
                ).sum().item() == 0:
                    # Debug the first few samples
                    print("[pretrain mirror][debug] empty expression mask/targets detected")
                    for b in range(min(3, expression_mask.size(0))):
                        start_idx = int(expression_target_lengths[:b].sum().item())
                        end_idx = int(expression_target_lengths[: b + 1].sum().item())
                        gold_ids = expression_targets[start_idx:end_idx]
                        gold_symbols = [id_to_symbol.get(int(i.item()), "?") for i in gold_ids]
                        print(
                            f"  sample {b}: expr_len_samples={int(expression_len_samples[b].item())} "
                            f"expr_len_frames={int(expression_len_frames[b].item())} "
                            f"answer_start_samples={int(answer_start_samples[b].item())} "
                            f"mask_sum={int(expression_mask[b].sum().item())} "
                            f"gold_symbols={gold_symbols}"
                        )
                    raise AssertionError("expression mask/targets should be non-empty in mirror pretrain")

                audio_loss = torch.tensor(0.0, device=device)
                stft_loss = torch.tensor(0.0, device=device)
                l1_loss = torch.tensor(0.0, device=device)
                if config.pretrain_mirror_audio_weight > 0:
                    wave_lengths_expr = expression_len_samples.clamp_min(1)
                    pred_wave_full, _ = frames_to_wave(guided_frames, wave_lengths_expr, config.frame_size)
                    target_wave_full, _ = frames_to_wave(features.to(device), wave_lengths_expr, config.frame_size)
                    stft_loss, _ = multi_resolution_stft_loss(
                        pred_wave_full, target_wave_full, lengths=wave_lengths_expr, config=stft_config
                    )
                    l1_loss = masked_l1_wave(pred_wave_full, target_wave_full, wave_lengths_expr)
                    audio_loss = config.pretrain_mirror_audio_weight * (config.l1_weight * l1_loss + stft_loss)

                expr_frame_ce_loss = torch.tensor(0.0, device=device)
                if config.pretrain_mirror_frame_ce_weight > 0:
                    valid_mask = expression_mask.view(-1)
                    if valid_mask.any():
                        logits_flat = symbol_logits.view(-1, symbol_logits.size(-1))
                        expr_targets_flat = expression_frame_symbol_targets.view(-1)
                        expr_frame_ce_loss = F.cross_entropy(
                            logits_flat[valid_mask],
                            expr_targets_flat[valid_mask],
                            weight=frame_ce_class_weights,
                        )

                expr_blank_penalty = torch.tensor(0.0, device=device)
                if config.pretrain_mirror_blank_penalty_weight > 0:
                    positive_mask = (expression_frame_symbol_targets != config.ctc_blank_id) & expression_mask
                    if positive_mask.any():
                        blank_probs = symbol_probs[..., config.ctc_blank_id]
                        expr_blank_penalty = blank_probs[positive_mask].mean()

                ctc_loss_value = torch.tensor(0.0, device=device)
                if expression_targets.numel() > 0 and config.pretrain_mirror_ctc_weight > 0:
                    symbol_logits_masked = symbol_logits.masked_fill(~expression_mask.unsqueeze(-1), 0.0)
                    log_probs = symbol_logits_masked.log_softmax(dim=-1).permute(1, 0, 2)
                    input_lengths = expression_mask.sum(dim=1).to(torch.long)
                    validate_ctc_inputs(
                        input_lengths,
                        expression_target_lengths,
                        context="pretrain_mirror_ctc",
                        start_samples=expression_len_samples,
                        len_samples=expression_target_lengths,
                    )
                    ctc_loss_value = ctc_loss_fn(
                        log_probs, expression_targets, input_lengths, expression_target_lengths
                    )

                total_loss = (
                    audio_loss
                    + config.pretrain_mirror_frame_ce_weight * expr_frame_ce_loss
                    + config.pretrain_mirror_blank_penalty_weight * expr_blank_penalty
                    + config.pretrain_mirror_ctc_weight * ctc_loss_value
                )

                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        "pretrain_mirror: non-finite loss detected "
                        f"(total={float(total_loss.item())}, ctc={float(ctc_loss_value.item())}, "
                        f"audio={float(audio_loss.item())}, expr_ce={float(expr_frame_ce_loss.item())}, "
                        f"blank_penalty={float(expr_blank_penalty.item())})"
                    )
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if (epoch + 1) % max(1, pre_epochs // 3) == 0:
                print(
                    f"[pretrain mirror] epoch {epoch + 1}/{pre_epochs} "
                    f"audio={audio_loss.item():.4f} "
                    f"ce={expr_frame_ce_loss.item():.4f} "
                    f"blank_penalty={expr_blank_penalty.item():.4f} "
                    f"ctc={ctc_loss_value.item():.4f}"
                )

        # Diagnostics on train split (expression copy)
        model.eval()
        mirror_matches = 0
        mirror_total = 0
        mirror_blank_frames = 0
        mirror_total_frames = 0
        mirror_token_acc = 0.0
        mirror_token_total = 0
        diag_printed = 0
        with torch.no_grad():
            for (
                features,
                _target_frames,
                mask,
                _targets,
                _target_lengths,
                _wave_lengths,
                _content_lengths,
                _frame_symbol_targets,
                expression_frame_symbol_targets,
                _remainder_targets,
                answer_start_samples,
                _answer_len_samples,
                expression_targets,
                expression_target_lengths,
                expression_len_samples,
            ) in train_loader:
                features = features.to(device)
                mask = mask.to(device)
                answer_start_samples = answer_start_samples.to(device)
                expression_targets = expression_targets.to(device)
                expression_target_lengths = expression_target_lengths.to(device)
                expression_len_samples = expression_len_samples.to(device)
                expression_frame_symbol_targets = expression_frame_symbol_targets.to(device)
                frame_outputs, symbol_logits = model(features, mask)
                symbol_probs = symbol_logits.softmax(dim=-1)

                expression_len_frames = (expression_len_samples + config.hop_size - 1) // config.hop_size
                expression_mask = torch.zeros_like(mask)
                for b in range(expression_mask.size(0)):
                    length = int(expression_len_frames[b].item())
                    end = min(length, expression_mask.size(1))
                    if end > 0:
                        expression_mask[b, :end] = True

                ctc_preds_expr = ctc_greedy_decode(symbol_logits, expression_mask, id_to_symbol, blank_id=config.ctc_blank_id)
                argmax_ids = symbol_probs.argmax(dim=-1)
                mirror_blank_frames += (((argmax_ids == config.ctc_blank_id) & expression_mask).sum().item())
                mirror_total_frames += expression_mask.sum().item()
                for b in range(features.size(0)):
                    start_idx = int(expression_target_lengths[:b].sum().item())
                    end_idx = int(expression_target_lengths[: b + 1].sum().item())
                    gold_ids = expression_targets[start_idx:end_idx]
                    gold_symbols = [id_to_symbol[int(i.item())] for i in gold_ids]
                    pred_symbols = ctc_preds_expr[b]
                    em = exact_match(pred_symbols, gold_symbols)
                    mirror_matches += em
                    mirror_total += 1
                    acc = token_accuracy(gold_symbols, pred_symbols)
                    mirror_token_acc += acc
                    mirror_token_total += 1
                    if diag_printed < 3:
                        print(
                            f"[pretrain mirror][sample] gold={' '.join(gold_symbols)} "
                            f"ctc_pred={' '.join(pred_symbols)} "
                            f"token_acc={acc:.2f}"
                        )
                        diag_printed += 1

        mirror_ctc_em = mirror_matches / max(1, mirror_total)
        mirror_expression_token_accuracy = mirror_token_acc / max(1, mirror_token_total)
        mirror_blank_rate = mirror_blank_frames / max(1, mirror_total_frames)
        print(
            f"[pretrain mirror] diagnostics: expression_ctc_em={mirror_ctc_em:.4f} "
            f"token_acc={mirror_expression_token_accuracy:.4f} "
            f"blank_rate={mirror_blank_rate:.3f}"
        )

        # Mid eval on eval split to check generalisation before mod training
        (
            loss_mid,
            em_mid,
            ctc_em_mid,
            blank_mid,
            _,
            expr_ctc_em_eval_mid,
            expr_token_acc_eval_mid,
            hybrid_em_cond_mid,
            hybrid_em_overall_mid,
            hybrid_parse_ok_rate_mid,
            remainder_acc_eval_mid,
            answer_blank_rate_mid,
            answer_ctc_empty_rate_mid,
        ) = evaluate_task3_model(
            model,
            dataloader=eval_loader,
            dataset=eval_dataset,
            device=device,
            stft_config=stft_config,
            l1_weight=config.l1_weight,
            id_to_symbol=id_to_symbol,
            blank_id=config.ctc_blank_id,
            decoder_segmentation=config.decoder_segmentation,
            symbol_guidance_weight=config.symbol_guidance_weight,
            symbol_templates=symbol_templates,
            render_mode=render_mode_eval,
            render_weight=config.render_weight,
            render_fixed_phase=config.render_fixed_phase,
            percent_id=percent_id,
            digit_ids=digit_ids,
            remainder_guidance_weight=config.remainder_guidance_weight,
            remainder_guidance_blank_floor=config.remainder_guidance_blank_floor,
            answer_digit_mass_floor=config.answer_digit_mass_floor,
            remainder_head=config.remainder_head,
            remainder_gru=remainder_gru,
            remainder_linear=remainder_linear,
            remainder_head_module=remainder_head_module,
            normalize_audio=config.decoder_normalize,
            hop_size=config.hop_size,
            frame_size=config.frame_size,
            answer_window_only=config.answer_window_only,
            collect_predictions=False,
        )
        print(
            "[mini_jmamba][task3] mid-eval "
            f"loss={loss_mid:.6f} em={em_mid:.4f} ctc_em={ctc_em_mid:.4f} blank_rate={blank_mid:.3f} "
            f"expr_ctc_em_eval={expr_ctc_em_eval_mid:.4f} "
            f"expr_token_acc_eval={expr_token_acc_eval_mid:.4f} "
            f"hybrid_em_overall={hybrid_em_overall_mid:.4f} "
            f"hybrid_em_cond={hybrid_em_cond_mid:.4f} "
            f"hybrid_parse_ok_rate={hybrid_parse_ok_rate_mid:.4f} "
            f"remainder_acc_eval={remainder_acc_eval_mid:.4f} "
            f"answer_blank_rate={answer_blank_rate_mid:.4f} "
            f"answer_ctc_empty_rate={answer_ctc_empty_rate_mid:.4f}"
        )

        # Apply LR factor before entering mod stage
        apply_mod_lr_factor(optimizer, mod_lr_factor)

    remainder_acc_eval_mid2 = None
    remainder_ce_loss_mid2 = None
    if config.pretrain_remainder_epochs > 0:
        # Optionally freeze backbone to train remainder head only
        if config.pretrain_remainder_freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        remainder_params: list[nn.Parameter] = []
        if remainder_gru is not None:
            remainder_params += list(remainder_gru.parameters())
        if remainder_linear is not None:
            remainder_params += list(remainder_linear.parameters())
        if remainder_head_module is not None:
            remainder_params += list(remainder_head_module.parameters())
        if not remainder_params:
            remainder_params = [p for p in model.parameters() if p.requires_grad]
        if not remainder_params:
            remainder_params = list(model.parameters())
        remainder_opt = torch.optim.Adam(remainder_params, lr=config.pretrain_remainder_lr)

        for epoch in range(config.pretrain_remainder_epochs):
            model.train()
            total_remainder_loss = 0.0
            total_weight = 0
            for (
                features,
                _target_frames,
                mask,
                _targets,
                _target_lengths,
                _wave_lengths,
                _content_lengths,
                _frame_symbol_targets,
                expression_frame_symbol_targets,
                remainder_targets,
                answer_start_samples,
                _answer_len_samples,
                expression_targets,
                expression_target_lengths,
                expression_len_samples,
            ) in train_loader:
                if remainder_targets.numel() == 0:
                    continue
                features = features.to(device)
                mask = mask.to(device)
                expression_frame_symbol_targets = expression_frame_symbol_targets.to(device)
                remainder_targets = remainder_targets.to(device)
                expression_len_samples = expression_len_samples.to(device)
                expression_target_lengths = expression_target_lengths.to(device)
                # 使用 return_hidden=True 获取 hidden states
                model_out = model(features, mask, return_hidden=True)
                frame_outputs, symbol_logits, hidden_states = model_out
                symbol_probs = symbol_logits.softmax(dim=-1)
                expression_len_frames = (expression_len_samples + config.hop_size - 1) // config.hop_size
                expression_mask = torch.zeros_like(mask)
                for b in range(expression_mask.size(0)):
                    length = int(expression_len_frames[b].item())
                    end = min(length, expression_mask.size(1))
                    if end > 0:
                        expression_mask[b, :end] = True
                token_probs, token_mask = build_expression_token_probs(
                    symbol_probs,
                    expression_len_samples,
                    expression_target_lengths,
                    hop_size=config.hop_size,
                )
                remainder_logits_local = compute_remainder_logits(
                    symbol_logits,
                    expression_mask,
                    digit_ids=digit_ids,
                    percent_id=percent_id,
                    blank_id=config.ctc_blank_id,
                    head=config.remainder_head,
                    remainder_gru=remainder_gru,
                    remainder_linear=remainder_linear,
                    token_probs=token_probs,
                    token_mask=token_mask,
                    hidden_states=hidden_states,
                    remainder_head_module=remainder_head_module,
                )
                remainder_ce = F.cross_entropy(remainder_logits_local, remainder_targets)
                remainder_opt.zero_grad()
                remainder_ce.backward()
                nn.utils.clip_grad_norm_(remainder_params, max_norm=1.0)
                remainder_opt.step()
                total_remainder_loss += remainder_ce.item() * remainder_targets.size(0)
                total_weight += remainder_targets.size(0)

            if (epoch + 1) % max(1, config.pretrain_remainder_epochs // 3) == 0:
                avg_loss = total_remainder_loss / max(1, total_weight)
                print(f"[pretrain remainder] epoch {epoch + 1}/{config.pretrain_remainder_epochs} loss={avg_loss:.6f}")

        # Evaluate after remainder-only pretrain
        (
            _loss_mid2,
            _em_mid2,
            _ctc_em_mid2,
            _blank_mid2,
            _,
            _expr_ctc_em_mid2,
            _expr_token_acc_mid2,
            _hybrid_cond_mid2,
            _hybrid_overall_mid2,
            _hybrid_parse_ok_mid2,
            remainder_acc_eval_mid2,
            _answer_blank_mid2,
            _answer_ctc_empty_mid2,
        ) = evaluate_task3_model(
            model,
            dataloader=eval_loader,
            dataset=eval_dataset,
            device=device,
            stft_config=stft_config,
            l1_weight=config.l1_weight,
            id_to_symbol=id_to_symbol,
            blank_id=config.ctc_blank_id,
            decoder_segmentation=config.decoder_segmentation,
            symbol_guidance_weight=config.symbol_guidance_weight,
            symbol_templates=symbol_templates,
            render_mode=render_mode_eval,
            render_weight=config.render_weight,
            render_fixed_phase=config.render_fixed_phase,
            percent_id=percent_id,
            digit_ids=digit_ids,
            remainder_guidance_weight=config.remainder_guidance_weight,
            remainder_guidance_blank_floor=config.remainder_guidance_blank_floor,
            answer_digit_mass_floor=config.answer_digit_mass_floor,
            remainder_head=config.remainder_head,
            remainder_gru=remainder_gru,
            remainder_linear=remainder_linear,
            remainder_head_module=remainder_head_module,
            normalize_audio=config.decoder_normalize,
            hop_size=config.hop_size,
            frame_size=config.frame_size,
            answer_window_only=config.answer_window_only,
            collect_predictions=False,
        )
        remainder_ce_loss_mid2 = total_remainder_loss / max(1, total_weight)
        print(
            "[mini_jmamba][task3] mid2-eval "
            f"remainder_ce_loss={remainder_ce_loss_mid2:.6f} "
            f"remainder_acc_eval={float(remainder_acc_eval_mid2 or 0.0):.4f}"
        )
        # Unfreeze backbone before main training
        for p in model.parameters():
            p.requires_grad = True

    for epoch in range(epochs):
        model.train()
        current_ctc_weight = config.ctc_weight_for_epoch(epoch, epochs)
        start_w = (
            config.mod_expr_ctc_weight_start
            if config.mod_expr_ctc_weight_start is not None
            else max(config.mod_expr_ctc_weight, config.mod_expr_ctc_weight_end)
        )
        end_w = config.mod_expr_ctc_weight_end
        if epochs > 1:
            decay_frac = epoch / max(1, epochs - 1)
            current_mod_expr_ctc_weight = start_w + decay_frac * (end_w - start_w)
        else:
            current_mod_expr_ctc_weight = start_w
        if config.symbol_warmup_epochs > 0 and epoch < config.symbol_warmup_epochs:
            audio_weight_factor = 0.0
        elif config.symbol_warmup_epochs > 0:
            ramp_epochs = max(1, min(10, epochs - config.symbol_warmup_epochs))
            audio_weight_factor = min(
                1.0,
                (epoch - config.symbol_warmup_epochs + 1) / ramp_epochs,
            )
        else:
            audio_weight_factor = 1.0
        for (
            features,
            target_frames,
            mask,
            targets,
            target_lengths,
            wave_lengths,
            content_lengths,
            frame_symbol_targets,
            expression_frame_symbol_targets,
            remainder_targets,
            answer_start_samples,
            answer_len_samples,
            expression_targets,
            expression_target_lengths,
            expression_len_samples,
        ) in train_loader:
            features = features.to(device)
            target_frames = target_frames.to(device)
            mask = mask.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            content_lengths = content_lengths.to(device)
            wave_lengths = wave_lengths.to(device)
            frame_symbol_targets = frame_symbol_targets.to(device)
            if percent_id is not None and frame_symbol_targets.numel() > 0:
                if int(frame_symbol_targets.max().item()) >= percent_id:
                    raise AssertionError("mod frame_symbol_targets should not include '%' ids")
            answer_start_samples = answer_start_samples.to(device)
            answer_len_samples = answer_len_samples.to(device)
            expression_targets = expression_targets.to(device)
            expression_target_lengths = expression_target_lengths.to(device)
            expression_len_samples = expression_len_samples.to(device)
            assert_mod_targets_without_percent(targets, percent_id)

            expression_len_frames = (expression_len_samples + config.hop_size - 1) // config.hop_size
            expression_mask = torch.zeros_like(mask)
            for b in range(expression_mask.size(0)):
                length = int(expression_len_frames[b].item())
                end = min(length, expression_mask.size(1))
                if end > 0:
                    expression_mask[b, :end] = True
            answer_start_frames = answer_start_samples // config.hop_size
            answer_len_frames = (answer_len_samples + config.hop_size - 1) // config.hop_size
            validate_answer_frame_windows(
                answer_start_frames,
                answer_len_frames,
                mask.sum(dim=1).to(torch.long),
                context="train_task3_answer_window_frames",
            )
            validate_answer_windows(
                answer_start_samples,
                answer_len_samples,
                wave_lengths,
                context="train_task3_answer_window",
            )

            # 使用 return_hidden=True 获取 hidden states（用于 attn_hidden head）
            model_out = model(features, mask, return_hidden=True)
            frame_outputs, symbol_logits, hidden_states = model_out
            symbol_probs = symbol_logits.softmax(dim=-1)
            token_probs, token_mask = build_expression_token_probs(
                symbol_probs,
                expression_len_samples,
                expression_target_lengths,
                hop_size=config.hop_size,
            )
            symbol_probs_render = symbol_probs
            remainder_ce_loss = torch.tensor(0.0, device=device)
            if config.remainder_ce_weight > 0 or config.remainder_guidance_weight > 0:
                remainder_logits = compute_remainder_logits(
                    symbol_logits,
                    expression_mask,
                    digit_ids=digit_ids,
                    percent_id=percent_id,
                    blank_id=config.ctc_blank_id,
                    head=config.remainder_head,
                    remainder_gru=remainder_gru,
                    remainder_linear=remainder_linear,
                    token_probs=token_probs,
                    token_mask=token_mask,
                    hidden_states=hidden_states,
                    remainder_head_module=remainder_head_module,
                )
                if config.remainder_ce_weight > 0:
                    remainder_ce_loss = F.cross_entropy(remainder_logits, remainder_targets.to(device))
                if config.remainder_guidance_weight > 0 and answer_len_frames.numel() > 0:
                    remainder_probs = remainder_logits.softmax(dim=-1)
                    symbol_probs_render = apply_answer_guidance_mix(
                        symbol_probs_render,
                        remainder_probs,
                        answer_start_frames,
                        answer_len_frames,
                        mix_weight=config.remainder_guidance_weight,
                        digit_ids=digit_ids,
                        blank_id=config.ctc_blank_id,
                        blank_floor=config.remainder_guidance_blank_floor,
                    )
            guided_frames = apply_symbol_guidance(
                frame_outputs,
                symbol_probs_render,
                symbol_templates,
                weight=config.symbol_guidance_weight,
            )
            if config.answer_window_only:
                window_mask = torch.zeros_like(mask)
                for b in range(window_mask.size(0)):
                    start = int(answer_start_frames[b].item())
                    length = int(answer_len_frames[b].item())
                    end = min(start + length, window_mask.size(1))
                    if end > start:
                        window_mask[b, start:end] = True
            else:
                window_mask = mask

            use_render = config.render_mode != "none" and config.render_weight > 0.0 and audio_weight_factor > 0.0
            rendered_frames = (
                apply_tone_bank_render(
                    guided_frames,
                    symbol_probs_render,
                    window_mask,
                    answer_start_frames,
                    answer_len_frames,
                    answer_len_samples,
                    mode=config.render_mode,
                    render_weight=config.render_weight,
                    render_fixed_phase=config.render_fixed_phase,
                    frame_size=config.frame_size,
                    hop_size=config.hop_size,
                    sr=SR,
                    answer_digit_mass_floor=config.answer_digit_mass_floor,
                )
                if use_render
                else guided_frames
            )

            frame_recon_mask = window_mask if config.answer_window_only else mask
            frame_recon_loss = masked_l2_frames(rendered_frames, target_frames, frame_recon_mask)

            pred_wave_full, _ = frames_to_wave(rendered_frames, wave_lengths, config.frame_size)
            target_wave_full, _ = frames_to_wave(target_frames, wave_lengths, config.frame_size)
            if config.answer_window_only:
                pred_wave, effective_lengths = extract_answer_window(
                    pred_wave_full,
                    answer_start_samples,
                    answer_len_samples,
                    wave_lengths=wave_lengths,
                )
                target_wave, _ = extract_answer_window(
                    target_wave_full,
                    answer_start_samples,
                    answer_len_samples,
                    wave_lengths=wave_lengths,
                )
            else:
                pred_wave, effective_lengths = pred_wave_full, wave_lengths
                target_wave, _ = target_wave_full, wave_lengths

            stft_loss, _ = multi_resolution_stft_loss(
                pred_wave, target_wave, lengths=effective_lengths, config=stft_config
            )
            l1_loss = masked_l1_wave(pred_wave, target_wave, effective_lengths)
            audio_loss = audio_weight_factor * (config.l1_weight * l1_loss + stft_loss)

            total_loss = audio_loss + config.frame_recon_weight * frame_recon_loss
            if config.remainder_ce_weight > 0:
                total_loss = total_loss + config.remainder_ce_weight * remainder_ce_loss
            ctc_loss_value = torch.tensor(0.0, device=device)
            frame_ce_loss = torch.tensor(0.0, device=device)
            blank_penalty = torch.tensor(0.0, device=device)
            mod_expr_ctc_loss = torch.tensor(0.0, device=device)
            if current_ctc_weight > 0 and targets.numel() > 0:
                if config.answer_window_only:
                    max_win = int(answer_len_frames.max().item()) if answer_len_frames.numel() > 0 else 0
                    window_logits = torch.zeros(
                        symbol_logits.size(0),
                        max_win,
                        symbol_logits.size(-1),
                        device=symbol_logits.device,
                        dtype=symbol_logits.dtype,
                    )
                    input_lengths = torch.zeros(symbol_logits.size(0), dtype=torch.long, device=symbol_logits.device)
                    for b in range(symbol_logits.size(0)):
                        start = int(answer_start_frames[b].item())
                        length = int(answer_len_frames[b].item())
                        end = min(start + length, symbol_logits.size(1))
                        win_len = max(0, end - start)
                        if win_len > 0:
                            input_lengths[b] = win_len
                            window_logits[b, :win_len] = symbol_logits[b, start:end]
                    log_probs = window_logits.log_softmax(dim=-1).permute(1, 0, 2)
                else:
                    log_probs = symbol_logits.log_softmax(dim=-1).permute(1, 0, 2)
                    input_lengths = mask.sum(dim=1).to(torch.long)
                validate_ctc_inputs(
                    input_lengths,
                    target_lengths,
                    context="train_task3_ctc",
                    start_samples=answer_start_samples if config.answer_window_only else None,
                    len_samples=answer_len_samples if config.answer_window_only else None,
                )
                ctc_loss_value = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                total_loss = total_loss + current_ctc_weight * ctc_loss_value
            if current_mod_expr_ctc_weight > 0 and expression_targets.numel() > 0:
                expr_log_probs = symbol_logits.masked_fill(~expression_mask.unsqueeze(-1), 0.0)
                expr_log_probs = expr_log_probs.log_softmax(dim=-1).permute(1, 0, 2)
                expr_input_lengths = expression_mask.sum(dim=1).to(torch.long)
                validate_ctc_inputs(
                    expr_input_lengths,
                    expression_target_lengths,
                    context="train_task3_mod_expr_ctc",
                    start_samples=expression_len_samples,
                    len_samples=expression_target_lengths,
                )
                mod_expr_ctc_loss = ctc_loss_fn(
                    expr_log_probs, expression_targets, expr_input_lengths, expression_target_lengths
                )
                total_loss = total_loss + current_mod_expr_ctc_weight * mod_expr_ctc_loss
            if config.frame_ce_weight > 0:
                frame_ce_mask = window_mask if config.answer_window_only else mask
                valid_mask = frame_ce_mask.view(-1)
                if valid_mask.any():
                    logits_flat = symbol_logits.view(-1, symbol_logits.size(-1))
                    targets_flat = frame_symbol_targets.view(-1)
                    frame_ce_loss = F.cross_entropy(
                        logits_flat[valid_mask],
                        targets_flat[valid_mask],
                        weight=frame_ce_class_weights,
                    )
                    total_loss = total_loss + config.frame_ce_weight * frame_ce_loss
            if config.blank_penalty_weight > 0:
                blank_mask = window_mask if config.answer_window_only else mask
                positive_mask = (frame_symbol_targets != config.ctc_blank_id) & blank_mask
                if positive_mask.any():
                    blank_probs = symbol_probs[..., config.ctc_blank_id]
                    blank_penalty = blank_probs[positive_mask].mean()
                    total_loss = total_loss + config.blank_penalty_weight * blank_penalty
            margin_loss_value = torch.tensor(0.0, device=device)
            if config.answer_blank_margin_weight > 0 and config.answer_blank_margin > 0:
                tone_mask = (frame_symbol_targets != config.ctc_blank_id) & (
                    window_mask if config.answer_window_only else mask
                )
                margin_loss_value = compute_answer_blank_margin_loss(
                    symbol_logits,
                    tone_mask,
                    digit_ids=digit_ids,
                    blank_id=config.ctc_blank_id,
                    margin=config.answer_blank_margin,
                )
                total_loss = total_loss + config.answer_blank_margin_weight * margin_loss_value
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    "train_task3: non-finite loss detected "
                    f"(total={float(total_loss.item())}, ctc={float(ctc_loss_value.item())}, "
                    f"mod_expr_ctc={float(mod_expr_ctc_loss.item())}, audio={float(audio_loss.item())}, "
                    f"frame_recon={float(frame_recon_loss.item())}, blank_penalty={float(blank_penalty.item())})"
                )
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(
                f"  epoch {epoch + 1}/{epochs} - "
                f"ctc_weight={current_ctc_weight:.3f} "
                f"audio_factor={audio_weight_factor:.2f} "
                f"train_audio={audio_loss.item():.6f} "
                f"stft={stft_loss.item():.6f} "
                f"l1={l1_loss.item():.6f} "
                f"frame_recon={frame_recon_loss.item():.6f} "
                f"mod_expr_ctc_w={current_mod_expr_ctc_weight:.3f} "
                f"mod_expr_ctc_w_loss={(current_mod_expr_ctc_weight * mod_expr_ctc_loss).item():.6f} "
                f"mod_expr_ctc_loss={mod_expr_ctc_loss.item():.6f} "
                f"ctc={ctc_loss_value.item():.6f} "
                f"frame_ce={frame_ce_loss.item():.6f} "
                f"blank_penalty={blank_penalty.item():.6f} "
                f"answer_blank_margin={margin_loss_value.item():.6f} "
                f"remainder_ce={remainder_ce_loss.item():.6f}"
            )

    (
        loss_post,
        em_post,
        ctc_em_post,
        blank_post,
        predictions,
        expr_ctc_em_eval_post,
        expr_token_acc_eval_post,
        hybrid_em_cond_post,
        hybrid_em_overall_post,
        hybrid_parse_ok_rate_post,
        remainder_acc_eval_post,
        answer_blank_rate_post,
        answer_ctc_empty_rate_post,
    ) = evaluate_task3_model(
        model,
        dataloader=eval_loader,
        dataset=eval_dataset,
        device=device,
        stft_config=stft_config,
        l1_weight=config.l1_weight,
        id_to_symbol=id_to_symbol,
        blank_id=config.ctc_blank_id,
        decoder_segmentation=config.decoder_segmentation,
        symbol_guidance_weight=config.symbol_guidance_weight,
        symbol_templates=symbol_templates,
        render_mode=render_mode_eval,
        render_weight=config.render_weight,
        render_fixed_phase=config.render_fixed_phase,
        percent_id=percent_id,
        digit_ids=digit_ids,
        remainder_guidance_weight=config.remainder_guidance_weight,
        remainder_guidance_blank_floor=config.remainder_guidance_blank_floor,
        answer_digit_mass_floor=config.answer_digit_mass_floor,
        remainder_head=config.remainder_head,
        remainder_gru=remainder_gru,
        remainder_linear=remainder_linear,
        remainder_head_module=remainder_head_module,
        normalize_audio=config.decoder_normalize,
        hop_size=config.hop_size,
        frame_size=config.frame_size,
        answer_window_only=config.answer_window_only,
        collect_predictions=True,
        ctc_diagnostics=True,
    )
    baseline_margin_post = em_post - baseline_stats["baseline_em"]
    print(
        "[mini_jmamba][task3] post-training "
        f"loss={loss_post:.6f} em={em_post:.4f} "
        f"ctc_em={ctc_em_post:.4f} blank_rate={blank_post:.3f} "
        f"baseline_em={baseline_stats['baseline_em']:.4f} "
        f"margin={baseline_margin_post:.4f}"
    )
    print(
        f"[mini_jmamba][task3] post-training diag "
        f"expr_ctc_em_eval={expr_ctc_em_eval_post:.4f} "
        f"expr_token_acc_eval={expr_token_acc_eval_post:.4f} "
        f"hybrid_em_overall={hybrid_em_overall_post:.4f} "
        f"hybrid_em_cond={hybrid_em_cond_post:.4f} "
        f"hybrid_parse_ok_rate={hybrid_parse_ok_rate_post:.4f} "
        f"remainder_acc_eval={remainder_acc_eval_post:.4f} "
        f"answer_blank_rate={answer_blank_rate_post:.4f} "
        f"answer_ctc_empty_rate={answer_ctc_empty_rate_post:.4f}"
    )
    for sample in predictions[:3]:
        gold = " ".join(sample["gold_symbols"])
        pred = " ".join(sample["pred_symbols"])
        print(
            f"example {sample['example_id']}: gold=[{gold}] pred=[{pred}] em={sample['exact_match']:.1f}"
        )
    baseline_requirement = baseline_margin_post >= 0.15
    ctc_requirement = ctc_em_post >= 0.10
    print(
        f"[mini_jmamba][task3] requirements: "
        f"em>=baseline+0.15? {baseline_requirement} "
        f"(target>={baseline_stats['baseline_em'] + 0.15:.3f}) "
        f"ctc_em>=0.10? {ctc_requirement}"
    )

    blank_post_metric = min(blank_post, blank_pre)

    metrics = {
        "loss_pre": loss_pre,
        "loss_post": loss_post,
        "em_pre": em_pre,
        "em_post": em_post,
        "em_baseline_margin": baseline_margin_post,
        "model_config": asdict(model_config),
        "digits_vocab": vocab,
        "ctc_weight": config.ctc_weight,
        "ctc_weight_schedule": config.ctc_weight_schedule,
        "ctc_weight_start": config.ctc_weight_start,
        "ctc_weight_end": config.ctc_weight_end,
        "symbol_warmup_epochs": config.symbol_warmup_epochs,
        "frame_ce_weight": config.frame_ce_weight,
        "frame_ce_blank_weight": config.frame_ce_blank_weight,
        "frame_recon_weight": config.frame_recon_weight,
        "symbol_guidance_weight": config.symbol_guidance_weight,
        "blank_penalty_weight": config.blank_penalty_weight,
        "remainder_guidance_blank_floor": config.remainder_guidance_blank_floor,
        "answer_digit_mass_floor": config.answer_digit_mass_floor,
        "answer_blank_margin": config.answer_blank_margin,
        "answer_blank_margin_weight": config.answer_blank_margin_weight,
        "mod_expr_ctc_weight_start": config.mod_expr_ctc_weight_start
        if config.mod_expr_ctc_weight_start is not None
        else config.mod_expr_ctc_weight,
        "mod_expr_ctc_weight_end": config.mod_expr_ctc_weight_end,
        "decoder_segmentation": config.decoder_segmentation,
        "decoder_normalize": config.decoder_normalize,
        "pretrain_mirror_epochs": config.pretrain_mirror_epochs,
        "pretrain_mirror_ctc_weight": config.pretrain_mirror_ctc_weight,
        "pretrain_mirror_frame_ce_weight": config.pretrain_mirror_frame_ce_weight,
        "pretrain_mirror_blank_penalty_weight": config.pretrain_mirror_blank_penalty_weight,
        "pretrain_mirror_audio_weight": config.pretrain_mirror_audio_weight,
        "pretrain_mirror_answer_window_only": config.pretrain_mirror_answer_window_only,
        "mirror_expression_ctc_em": mirror_ctc_em,
        "mirror_expression_token_accuracy": mirror_expression_token_accuracy,
        "mirror_expression_blank_rate": mirror_blank_rate,
        "expression_ctc_em_eval_pre": expr_ctc_em_eval_pre,
        "expression_ctc_em_eval_post": expr_ctc_em_eval_post,
        "expression_token_acc_eval_pre": expr_token_acc_eval_pre,
        "expression_token_acc_eval_post": expr_token_acc_eval_post,
        "hybrid_em_cond_pre": hybrid_em_cond_pre,
        "hybrid_em_cond_post": hybrid_em_cond_post,
        "hybrid_em_overall_pre": hybrid_em_overall_pre,
        "hybrid_em_overall_post": hybrid_em_overall_post,
        "hybrid_parse_ok_rate_pre": hybrid_parse_ok_rate_pre,
        "hybrid_parse_ok_rate_post": hybrid_parse_ok_rate_post,
        "remainder_acc_eval_pre": remainder_acc_eval_pre,
        "remainder_acc_eval_post": remainder_acc_eval_post,
        "remainder_acc_eval_mid": locals().get("remainder_acc_eval_mid"),
        "expression_ctc_em_eval_mid": locals().get("expr_ctc_em_eval_mid"),
        "expression_token_acc_eval_mid": locals().get("expr_token_acc_eval_mid"),
        "hybrid_em_cond_mid": locals().get("hybrid_em_cond_mid"),
        "hybrid_em_overall_mid": locals().get("hybrid_em_overall_mid"),
        "hybrid_parse_ok_rate_mid": locals().get("hybrid_parse_ok_rate_mid"),
        "answer_blank_rate_pre": answer_blank_rate_pre,
        "answer_blank_rate_post": answer_blank_rate_post,
        "answer_ctc_empty_rate_pre": answer_ctc_empty_rate_pre,
        "answer_ctc_empty_rate_post": answer_ctc_empty_rate_post,
        "answer_blank_rate_mid": locals().get("answer_blank_rate_mid"),
        "answer_ctc_empty_rate_mid": locals().get("answer_ctc_empty_rate_mid"),
        "render_mode": config.render_mode,
        "render_weight": config.render_weight,
        "render_fixed_phase": config.render_fixed_phase,
        "ctc_em_pre": ctc_em_pre,
        "ctc_em_post": ctc_em_post,
        "ctc_blank_rate_pre": blank_pre,
        "ctc_blank_rate_post": blank_post_metric,
        "ctc_blank_id": config.ctc_blank_id,
        "l1_weight": config.l1_weight,
        "stft_config": asdict(stft_config),
        "baseline_em": baseline_stats["baseline_em"],
        "baseline_type": baseline_stats["baseline_type"],
        "baseline_always_zero_em": baseline_stats["baseline_always_zero_em"],
        "baseline_majority_value": baseline_stats["baseline_majority_value"],
        "baseline_majority_em": baseline_stats["baseline_majority_em"],
        "ctc_requirement_threshold": 0.10,
        "baseline_margin_requirement": 0.15,
        "pass_baseline_margin": baseline_requirement,
        "pass_ctc_threshold": ctc_requirement,
        "remainder_head": config.remainder_head,
        "remainder_gru_hidden": config.remainder_gru_hidden,
        "pretrain_remainder_epochs": config.pretrain_remainder_epochs,
        "pretrain_remainder_lr": config.pretrain_remainder_lr,
        "pretrain_remainder_freeze_backbone": config.pretrain_remainder_freeze_backbone,
        "remainder_acc_eval_mid2": remainder_acc_eval_mid2,
        "remainder_ce_loss_mid2": remainder_ce_loss_mid2,
    }
    # 返回 model 和配置用于保存 checkpoint
    model_info = {
        "model": model,
        "model_config": model_config,
        "symbol_to_id": vocab,  # vocab 是 symbol -> id 的映射
        "id_to_symbol": id_to_symbol,
    }
    return predictions, metrics, model_info


__all__ = ["Task3TrainingConfig", "mini_jmamba_task3_pipeline", "render_tone_bank"]
