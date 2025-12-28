"""Mini-JMamba pipeline for Task2 Bracket Validity (binary classification in audio domain).

Core principle: End-to-end audio reasoning without intermediate text tokens.
Input: Bracket sequence audio -> Output: 'V' (valid) or 'X' (invalid) audio tone.
"""

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

from jericho.data import ManifestEntry
from jericho.models import (
    MiniJMamba,
    MiniJMambaConfig,
    MultiResolutionSTFTConfig,
    multi_resolution_stft_loss,
)
from jericho.scorer import decode_wave_to_symbols
from jericho.symbols import SR, TONE_DUR, GAP_DUR, encode_symbols_to_wave, SYMBOL2FREQ
from jericho.task2 import (
    BRACKET_OPEN,
    BRACKET_CLOSE,
    VALID_SYMBOL,
    INVALID_SYMBOL,
    is_balanced,
    target_symbol_for_task2,
)

from .mini_jmamba_audio import frame_wave

# Task2 符号表
TASK2_INPUT_SYMBOLS = [BRACKET_OPEN, BRACKET_CLOSE]
TASK2_OUTPUT_SYMBOLS = [VALID_SYMBOL, INVALID_SYMBOL]
TASK2_ALL_SYMBOLS = TASK2_INPUT_SYMBOLS + TASK2_OUTPUT_SYMBOLS


@dataclass
class Task2TrainingConfig:
    """Configuration for Task2 bracket validity training."""

    frame_size: int = 160
    hop_size: int = 160
    d_model: int = 128
    num_ssm_layers: int = 10
    num_attn_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1

    # Loss weights
    audio_weight: float = 1.0
    l1_weight: float = 1.0
    binary_ce_weight: float = 1.0  # 二分类辅助损失
    symbol_guidance_weight: float = 1.0

    # Whether to compute audio reconstruction losses only on the answer window.
    # This avoids length-dilution on OOD-length splits where the answer occupies
    # a small fraction of the full waveform.
    answer_window_only: bool = True

    # STFT config
    stft_fft_sizes: Tuple[int, ...] = (512, 1024, 2048)
    stft_hop_scale: float = 0.25
    stft_win_scale: float = 1.0
    stft_sc_weight: float = 1.0
    stft_mag_weight: float = 1.0
    stft_eps: float = 1e-7

    # Thinking gap (input -> think -> output)
    thinking_gap_s: float = 0.3
    thinking_gap_align: int = 160

    # Warmup
    symbol_warmup_epochs: int = 10
    audio_ramp_epochs: int = 10

    # Blank handling
    ctc_blank_id: int = 0

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


@dataclass
class Task2Sample:
    """A single Task2 training/evaluation sample."""

    entry: ManifestEntry
    input_frames: torch.Tensor
    target_frames: torch.Tensor
    frame_count: int
    input_wave: torch.Tensor
    target_wave: torch.Tensor
    target_label: int  # 0 = valid (V), 1 = invalid (X)
    target_symbol: str  # 'V' or 'X'
    input_len_samples: int
    answer_start_samples: int
    answer_len_samples: int


class Task2Dataset(Dataset):
    """Dataset for Task2 bracket validity samples."""

    def __init__(self, samples: Sequence[Task2Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Task2Sample:
        return self.samples[idx]

    @property
    def frame_counts(self) -> List[int]:
        return [sample.frame_count for sample in self.samples]

    @property
    def entries(self) -> List[ManifestEntry]:
        return [sample.entry for sample in self.samples]


def build_task2_vocab() -> dict[str, int]:
    """Build vocabulary mapping for Task2 symbols."""
    # blank_id = 0, then (, ), V, X
    vocab = {}
    for idx, symbol in enumerate(TASK2_ALL_SYMBOLS):
        vocab[symbol] = idx + 1  # 1-indexed, 0 is blank
    return vocab


def _align_length(length: int, align: int) -> int:
    if align <= 0:
        return length
    return int(math.ceil(length / align) * align)


def synthesise_task2_target_wave(
    target_symbol: str,
    *,
    target_length_samples: int | None = None,
    fixed_phase: float = 0.0,
) -> np.ndarray:
    """Synthesize target audio for Task2 (single V or X tone)."""
    wave = encode_symbols_to_wave([target_symbol], fixed_phase=fixed_phase)
    if target_length_samples is not None and wave.size < target_length_samples:
        wave = np.pad(wave, (0, target_length_samples - wave.size))
    return wave


def prepare_task2_samples(
    entries: Sequence[ManifestEntry],
    vocab: dict[str, int],
    frame_size: int,
    hop_size: int,
    *,
    blank_id: int,
    config: Task2TrainingConfig,
    noise_snr_db: float | None = None,
) -> List[Task2Sample]:
    """Prepare Task2 samples from manifest entries."""
    samples: List[Task2Sample] = []
    tone_samples = int(round(SR * TONE_DUR))
    gap_samples = int(round(SR * GAP_DUR))
    thinking_gap_samples = int(round(SR * config.thinking_gap_s))
    thinking_gap_samples = _align_length(thinking_gap_samples, config.thinking_gap_align)

    for entry in entries:
        # Input: bracket sequence (use fixed_phase=0 for reproducibility)
        input_wave = encode_symbols_to_wave(entry.symbols, fixed_phase=0.0)
        input_len_samples = len(input_wave)

        # Target symbol
        target_symbol = target_symbol_for_task2(entry.symbols)
        target_label = 0 if target_symbol == VALID_SYMBOL else 1

        # Answer window: after thinking gap
        answer_start_samples = input_len_samples + thinking_gap_samples
        answer_len_samples = tone_samples  # single tone

        # Total length
        total_samples = answer_start_samples + answer_len_samples
        total_samples = _align_length(total_samples, frame_size)

        # Build full input wave (input + thinking gap silence)
        full_input = np.zeros(total_samples, dtype=np.float32)
        full_input[:input_len_samples] = input_wave

        # Add noise if specified (for OOD noise testing)
        if noise_snr_db is not None:
            signal_power = np.mean(input_wave ** 2) + 1e-10
            noise_power = signal_power / (10 ** (noise_snr_db / 10))
            noise = np.random.randn(total_samples).astype(np.float32) * np.sqrt(noise_power)
            full_input = full_input + noise

        # Build target wave (silence + answer)
        target_wave = synthesise_task2_target_wave(
            target_symbol,
            target_length_samples=answer_len_samples,
            fixed_phase=0.0,
        )
        full_target = np.zeros(total_samples, dtype=np.float32)
        if answer_start_samples + len(target_wave) <= total_samples:
            full_target[answer_start_samples:answer_start_samples + len(target_wave)] = target_wave

        # Frame
        input_frames = frame_wave(torch.from_numpy(full_input), frame_size, hop_size)
        target_frames = frame_wave(torch.from_numpy(full_target), frame_size, hop_size)
        frame_count = input_frames.size(0)

        sample = Task2Sample(
            entry=entry,
            input_frames=input_frames,
            target_frames=target_frames,
            frame_count=frame_count,
            input_wave=torch.from_numpy(full_input),
            target_wave=torch.from_numpy(full_target),
            target_label=target_label,
            target_symbol=target_symbol,
            input_len_samples=input_len_samples,
            answer_start_samples=answer_start_samples,
            answer_len_samples=answer_len_samples,
        )
        samples.append(sample)

    return samples


def collate_task2_train(
    batch: List[Task2Sample],
    frame_size: int,
) -> Tuple[
    torch.Tensor,  # features (B, T, frame_size)
    torch.Tensor,  # target_frames (B, T, frame_size)
    torch.Tensor,  # mask (B, T)
    torch.Tensor,  # labels (B,) - 0=valid, 1=invalid
    torch.Tensor,  # answer_start_samples (B,)
    torch.Tensor,  # answer_len_samples (B,)
]:
    """Collate function for Task2 training."""
    max_frames = max(sample.frame_count for sample in batch)
    features = torch.zeros(len(batch), max_frames, frame_size, dtype=torch.float32)
    target_frames = torch.zeros(len(batch), max_frames, frame_size, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_frames, dtype=torch.bool)
    labels = torch.zeros(len(batch), dtype=torch.long)
    answer_start_samples = torch.zeros(len(batch), dtype=torch.long)
    answer_len_samples = torch.zeros(len(batch), dtype=torch.long)

    for idx, sample in enumerate(batch):
        features[idx, :sample.frame_count] = sample.input_frames
        target_frames[idx, :sample.frame_count] = sample.target_frames
        mask[idx, :sample.frame_count] = True
        labels[idx] = sample.target_label
        answer_start_samples[idx] = sample.answer_start_samples
        answer_len_samples[idx] = sample.answer_len_samples

    return features, target_frames, mask, labels, answer_start_samples, answer_len_samples


def frames_to_wave(
    frames: torch.Tensor,
    lengths: torch.Tensor,
    frame_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert frames back to waveform."""
    batch_size = frames.size(0)
    max_samples = int(lengths.max().item())
    waves = torch.zeros(batch_size, max_samples, device=frames.device, dtype=frames.dtype)
    for b in range(batch_size):
        wave_len = int(lengths[b].item())
        num_frames = min(frames.size(1), (wave_len + frame_size - 1) // frame_size)
        for t in range(num_frames):
            start = t * frame_size
            end = min(start + frame_size, wave_len)
            if end > start:
                waves[b, start:end] = frames[b, t, :end - start]
    return waves, lengths


def extract_answer_window(
    wave: torch.Tensor,
    answer_start_samples: torch.Tensor,
    answer_len_samples: torch.Tensor,
    *,
    wave_lengths: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract answer-window slices from waveforms.

    Parameters
    ----------
    wave:
        Waveform tensor of shape (B, T).
    answer_start_samples:
        Start index (in samples) of the answer window for each sample. Shape (B,).
    answer_len_samples:
        Answer window length (in samples) for each sample. Shape (B,).
    wave_lengths:
        Optional per-sample valid lengths (in samples). Used to clamp windows.

    Returns
    -------
    windowed_wave:
        Padded answer-window waveform tensor of shape (B, max_answer_len).
    effective_lengths:
        Per-sample effective window lengths after clamping. Shape (B,).
    """
    if wave.dim() != 2:
        raise ValueError(f"extract_answer_window expects (B, T), got shape={tuple(wave.shape)}")
    batch_size, max_time = wave.shape
    if batch_size == 0:
        return wave[:, :0], torch.zeros(0, device=wave.device, dtype=torch.long)

    starts = answer_start_samples.to(device=wave.device).clamp(min=0, max=max_time)
    lengths = answer_len_samples.to(device=wave.device).clamp(min=0, max=max_time)
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    windowed = torch.zeros(batch_size, max_len, device=wave.device, dtype=wave.dtype)
    effective_lengths = torch.zeros(batch_size, device=wave.device, dtype=torch.long)

    for b in range(batch_size):
        start = int(starts[b].item())
        req_len = int(lengths[b].item())
        if req_len <= 0:
            continue
        valid_end = max_time
        if wave_lengths is not None:
            valid_end = min(valid_end, int(wave_lengths[b].to(device=wave.device).item()))
        end = min(start + req_len, valid_end)
        win_len = max(0, end - start)
        if win_len > 0:
            windowed[b, :win_len] = wave[b, start:end]
            effective_lengths[b] = win_len

    return windowed, effective_lengths


def masked_l1_wave(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute masked L1 loss on waveforms."""
    batch_size = pred.size(0)
    total_loss = 0.0
    total_samples = 0
    for b in range(batch_size):
        length = int(lengths[b].item())
        if length > 0:
            total_loss += F.l1_loss(pred[b, :length], target[b, :length], reduction="sum")
            total_samples += length
    return total_loss / max(1, total_samples)


def build_symbol_frame_templates(
    vocab: dict[str, int],
    frame_size: int,
    *,
    blank_id: int,
) -> torch.Tensor:
    """Precompute frame-sized tone snippets for each symbol."""
    num_classes = max([blank_id] + list(vocab.values())) + 1
    templates = torch.zeros(num_classes, frame_size, dtype=torch.float32)
    for symbol, idx in vocab.items():
        wave = encode_symbols_to_wave([symbol], gap_dur=0.0, fixed_phase=0.0)
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


def apply_cls_guidance_to_answer_window(
    symbol_probs: torch.Tensor,
    cls_logits: torch.Tensor,
    answer_start_frames: torch.Tensor,
    answer_len_frames: torch.Tensor,
    *,
    valid_symbol_id: int,
    invalid_symbol_id: int,
    mix_weight: float = 1.0,
) -> torch.Tensor:
    """Inject classification probability into answer window symbol probs.
    
    This bridges the gap between the classification head (which correctly
    predicts V/X) and the audio output (which uses symbol_probs for rendering).
    
    Parameters
    ----------
    symbol_probs:
        Per-frame symbol probabilities, shape (B, T, vocab_size).
    cls_logits:
        Classification logits, shape (B, 2) where [:, 0]=valid, [:, 1]=invalid.
    answer_start_frames:
        Start frame index of answer window, shape (B,).
    answer_len_frames:
        Length of answer window in frames, shape (B,).
    valid_symbol_id:
        Vocabulary index for the "V" (valid) symbol.
    invalid_symbol_id:
        Vocabulary index for the "X" (invalid) symbol.
    mix_weight:
        How much to mix cls_probs into symbol_probs (0=none, 1=full replacement).
    
    Returns
    -------
    Modified symbol_probs with cls guidance in the answer window.
    """
    if mix_weight <= 0:
        return symbol_probs
    
    batch_size, seq_len, vocab_size = symbol_probs.shape
    device = symbol_probs.device
    
    # Convert cls_logits to probabilities
    cls_probs = cls_logits.softmax(dim=-1)  # (B, 2)
    valid_prob = cls_probs[:, 0]  # P(valid)
    invalid_prob = cls_probs[:, 1]  # P(invalid)
    
    # Create modified symbol_probs
    result = symbol_probs.clone()
    
    for b in range(batch_size):
        start = int(answer_start_frames[b].item())
        length = int(answer_len_frames[b].item())
        if length <= 0 or start >= seq_len:
            continue
        end = min(start + length, seq_len)
        
        # For answer window frames, mix in cls guidance
        # Create a probability distribution focused on V or X based on cls_probs
        cls_symbol_probs = torch.zeros(vocab_size, device=device)
        cls_symbol_probs[valid_symbol_id] = valid_prob[b]
        cls_symbol_probs[invalid_symbol_id] = invalid_prob[b]
        
        # Mix: (1 - mix_weight) * original + mix_weight * cls_guided
        for t in range(start, end):
            result[b, t] = (1 - mix_weight) * symbol_probs[b, t] + mix_weight * cls_symbol_probs
    
    return result


def apply_cls_guidance_to_frames(
    frame_outputs: torch.Tensor,
    cls_logits: torch.Tensor,
    answer_start_frames: torch.Tensor,
    answer_len_frames: torch.Tensor,
    templates: torch.Tensor,
    *,
    valid_symbol_id: int,
    invalid_symbol_id: int,
    sr: int = 16000,
) -> torch.Tensor:
    """Directly replace answer window frames with cls-guided continuous waveforms.
    
    CRITICAL: We cannot simply tile the same 160-sample template because:
    - V (1900 Hz) has ~19.0 cycles in 160 samples (OK, phase-continuous)
    - X (1950 Hz) has ~19.5 cycles in 160 samples (NOT OK, phase discontinuity)
    When X template is repeated, the phase discontinuity causes FFT to 
    incorrectly decode it as V (1900 Hz).
    
    Solution: Generate continuous waveforms for the full answer window length,
    then weight them by cls_probs.
    """
    batch_size, seq_len, frame_size = frame_outputs.shape
    device = frame_outputs.device
    dtype = frame_outputs.dtype
    
    # Convert cls_logits to probabilities
    cls_probs = cls_logits.softmax(dim=-1)  # (B, 2)
    valid_prob = cls_probs[:, 0]
    invalid_prob = cls_probs[:, 1]
    
    result = frame_outputs.clone()
    
    # Frequencies for V and X
    V_FREQ = 1900.0
    X_FREQ = 1950.0
    AMPLITUDE = 0.8
    
    for b in range(batch_size):
        start = int(answer_start_frames[b].item())
        length = int(answer_len_frames[b].item())
        if length <= 0 or start >= seq_len:
            continue
        end = min(start + length, seq_len)
        num_frames = end - start
        total_samples = num_frames * frame_size
        
        # Generate continuous waveforms for full answer window
        t = torch.arange(total_samples, device=device, dtype=dtype) / sr
        v_wave = AMPLITUDE * torch.sin(2 * 3.141592653589793 * V_FREQ * t)
        x_wave = AMPLITUDE * torch.sin(2 * 3.141592653589793 * X_FREQ * t)
        
        # Weight by classification probability
        weighted_wave = valid_prob[b] * v_wave + invalid_prob[b] * x_wave
        
        # Split into frames and assign
        for i, t_idx in enumerate(range(start, end)):
            frame_start = i * frame_size
            frame_end = frame_start + frame_size
            result[b, t_idx] = weighted_wave[frame_start:frame_end]
    
    return result


class BinaryClassificationHead(nn.Module):
    """Binary classification head using attention pooling over hidden states."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        num_attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, num_attn_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 2 classes: valid, invalid
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : (B, T, d_model)
        mask : (B, T) bool

        Returns
        -------
        logits : (B, 2)
        """
        batch_size = hidden_states.size(0)
        query = self.query.expand(batch_size, -1, -1)
        key_padding_mask = ~mask
        pooled, _ = self.attn(
            query, hidden_states, hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = pooled.squeeze(1)
        pooled = self.norm(pooled)
        return self.mlp(pooled)


def evaluate_task2_model(
    model: MiniJMamba,
    classification_head: BinaryClassificationHead,
    dataloader: DataLoader,
    dataset: Task2Dataset,
    device: torch.device,
    *,
    stft_config: MultiResolutionSTFTConfig,
    l1_weight: float,
    symbol_guidance_weight: float,
    symbol_templates: torch.Tensor,
    hop_size: int,
    frame_size: int,
    valid_symbol_id: int,
    invalid_symbol_id: int,
    answer_window_only: bool = True,
    collect_predictions: bool = False,
) -> Tuple[float, float, float, List[dict]]:
    """Evaluate Task2 model.

    Returns
    -------
    avg_loss : float
    accuracy : float
    audio_accuracy : float (based on decoded audio)
    predictions : List[dict]
    """
    model.eval()
    classification_head.eval()

    total_loss = 0.0
    total_weight = 0.0
    correct_cls = 0
    correct_audio = 0
    total_samples = 0
    predictions: List[dict] = []

    with torch.no_grad():
        for batch_idx, (
            features,
            target_frames,
            mask,
            labels,
            answer_start_samples,
            answer_len_samples,
        ) in enumerate(dataloader):
            batch_size = features.size(0)
            features = features.to(device)
            target_frames = target_frames.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            answer_start_samples = answer_start_samples.to(device)
            answer_len_samples = answer_len_samples.to(device)

            # Forward
            model_out = model(features, mask, return_hidden=True)
            frame_outputs, symbol_logits, hidden_states = model_out
            symbol_probs = symbol_logits.softmax(dim=-1)

            # Classification
            cls_logits = classification_head(hidden_states, mask)
            cls_preds = cls_logits.argmax(dim=-1)

            # Inject cls_probs into answer window symbol_probs
            answer_start_frames = answer_start_samples // hop_size
            answer_len_frames = (answer_len_samples + hop_size - 1) // hop_size
            symbol_probs_guided = apply_cls_guidance_to_answer_window(
                symbol_probs,
                cls_logits,  # no detach needed in eval (no_grad context)
                answer_start_frames,
                answer_len_frames,
                valid_symbol_id=valid_symbol_id,
                invalid_symbol_id=invalid_symbol_id,
                mix_weight=1.0,  # full replacement in answer window
            )

            # Audio rendering
            guided_frames = apply_symbol_guidance(
                frame_outputs, symbol_probs_guided, symbol_templates,
                weight=symbol_guidance_weight,
            )
            
            # Directly replace answer window frames with cls-guided templates
            guided_frames = apply_cls_guidance_to_frames(
                guided_frames,
                cls_logits,
                answer_start_frames,
                answer_len_frames,
                symbol_templates,
                valid_symbol_id=valid_symbol_id,
                invalid_symbol_id=invalid_symbol_id,
            )

            # Compute loss
            wave_lengths = mask.sum(dim=1) * hop_size
            pred_wave_full, _ = frames_to_wave(guided_frames, wave_lengths, frame_size)
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
                pred_wave, target_wave, effective_lengths = pred_wave_full, target_wave_full, wave_lengths
            stft_loss, _ = multi_resolution_stft_loss(
                pred_wave, target_wave, lengths=effective_lengths, config=stft_config
            )
            l1_loss = masked_l1_wave(pred_wave, target_wave, effective_lengths)
            audio_loss = l1_weight * l1_loss + stft_loss
            total_loss += audio_loss.item() * batch_size
            total_weight += batch_size

            # Classification accuracy
            correct_cls += (cls_preds == labels).sum().item()
            total_samples += batch_size

            # Audio-based accuracy (decode and check)
            for b in range(batch_size):
                base_idx = batch_idx * dataloader.batch_size + b if dataloader.batch_size else b
                sample = dataset.samples[base_idx] if base_idx < len(dataset.samples) else None

                # Decode audio
                pred_audio = pred_wave_full[b].cpu().numpy()
                answer_start = int(answer_start_samples[b].item())
                answer_len = int(answer_len_samples[b].item())
                answer_audio = pred_audio[answer_start:answer_start + answer_len]
                answer_rms = float(np.sqrt(np.mean(answer_audio ** 2))) if answer_audio.size else 0.0
                answer_peak = float(np.max(np.abs(answer_audio))) if answer_audio.size else 0.0

                try:
                    decoded = decode_wave_to_symbols(answer_audio)
                except Exception:
                    decoded = []

                gold_symbol = VALID_SYMBOL if labels[b].item() == 0 else INVALID_SYMBOL
                pred_symbol = decoded[0] if len(decoded) == 1 else ""
                audio_correct = pred_symbol == gold_symbol
                correct_audio += int(audio_correct)

                if collect_predictions and sample is not None:
                    cls_symbol = "V" if cls_preds[b].item() == 0 else "X"
                    predictions.append({
                        "example_id": sample.entry.example_id,
                        "split": sample.entry.split,
                        "gold_symbols": [gold_symbol],
                        "pred_symbols": [pred_symbol] if pred_symbol else [],
                        "exact_match": 1.0 if audio_correct else 0.0,
                        "gold_symbol": gold_symbol,
                        "pred_symbol": pred_symbol,
                        "cls_pred": cls_symbol,
                        "audio_correct": audio_correct,
                        "cls_correct": cls_preds[b].item() == labels[b].item(),
                        "cls_audio_disagree": bool(cls_symbol != pred_symbol),
                        "audio_decoded_len": int(len(decoded)),
                        "answer_rms": answer_rms,
                        "answer_peak": answer_peak,
                        "input_len_samples": int(sample.input_len_samples),
                    })

    avg_loss = total_loss / max(1.0, total_weight)
    cls_accuracy = correct_cls / max(1, total_samples)
    audio_accuracy = correct_audio / max(1, total_samples)

    return avg_loss, cls_accuracy, audio_accuracy, predictions


def compute_task2_baseline_stats(dataset: Task2Dataset) -> dict[str, float | str]:
    """Compute simple Task2 baselines."""
    if len(dataset) == 0:
        return {
            "baseline_accuracy": 0.0,
            "baseline_type": "empty",
            "valid_ratio": 0.0,
        }
    labels = [s.target_label for s in dataset.samples]
    valid_count = sum(1 for l in labels if l == 0)
    invalid_count = len(labels) - valid_count
    majority_label = 0 if valid_count >= invalid_count else 1
    majority_accuracy = max(valid_count, invalid_count) / len(labels)
    return {
        "baseline_accuracy": majority_accuracy,
        "baseline_type": "majority",
        "majority_label": "V" if majority_label == 0 else "X",
        "valid_ratio": valid_count / len(labels),
    }


def mini_jmamba_task2_pipeline(
    train_entries: Sequence[ManifestEntry],
    eval_entries: Sequence[ManifestEntry],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    config: Task2TrainingConfig | None = None,
    eval_noise_snr_db: float | None = None,
) -> Tuple[List[dict], dict]:
    """Train and evaluate Mini-JMamba on Task2 (bracket validity).
    
    Parameters
    ----------
    eval_noise_snr_db:
        If set, add Gaussian noise to evaluation inputs at this SNR (dB).
        Used for OOD noise testing.
    """

    if config is None:
        config = Task2TrainingConfig()

    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab = build_task2_vocab()
    symbol_templates = build_symbol_frame_templates(
        vocab, config.frame_size, blank_id=config.ctc_blank_id
    ).to(device)
    
    # Get symbol IDs for cls_guidance
    valid_symbol_id = vocab[VALID_SYMBOL]
    invalid_symbol_id = vocab[INVALID_SYMBOL]

    train_samples = prepare_task2_samples(
        train_entries, vocab, config.frame_size, config.hop_size,
        blank_id=config.ctc_blank_id, config=config,
        noise_snr_db=None,  # No noise on training
    )
    eval_samples = prepare_task2_samples(
        eval_entries, vocab, config.frame_size, config.hop_size,
        blank_id=config.ctc_blank_id, config=config,
        noise_snr_db=eval_noise_snr_db,  # Apply noise if OOD noise test
    )
    train_dataset = Task2Dataset(train_samples)
    eval_dataset = Task2Dataset(eval_samples)
    baseline_stats = compute_task2_baseline_stats(eval_dataset)

    if len(train_dataset) == 0:
        raise SystemExit("Train split is empty; cannot train mini_jmamba on Task2.")
    if len(eval_dataset) == 0:
        raise SystemExit("Evaluation split is empty.")

    max_frames = max(max(train_dataset.frame_counts), max(eval_dataset.frame_counts))
    model_config = MiniJMambaConfig(
        frame_size=config.frame_size,
        hop_size=config.hop_size,
        symbol_vocab_size=len(vocab) + 1,  # +1 for blank
        d_model=config.d_model,
        num_ssm_layers=config.num_ssm_layers,
        num_attn_layers=config.num_attn_layers,
        num_heads=config.num_heads,
        max_frames=max_frames,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
    )
    model = MiniJMamba(model_config).to(device)

    # Binary classification head
    classification_head = BinaryClassificationHead(
        d_model=config.d_model,
        hidden_dim=128,
        num_attn_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)

    params = list(model.parameters()) + list(classification_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    stft_config = config.build_stft_config()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_task2_train(batch, config.frame_size),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_task2_train(batch, config.frame_size),
    )

    # Pre-training evaluation
    loss_pre, cls_acc_pre, audio_acc_pre, _ = evaluate_task2_model(
        model, classification_head, eval_loader, eval_dataset, device,
        stft_config=stft_config, l1_weight=config.l1_weight,
        symbol_guidance_weight=config.symbol_guidance_weight,
        symbol_templates=symbol_templates,
        hop_size=config.hop_size, frame_size=config.frame_size,
        valid_symbol_id=valid_symbol_id,
        invalid_symbol_id=invalid_symbol_id,
        answer_window_only=config.answer_window_only,
    )
    print(
        f"[mini_jmamba][task2] pre-training loss={loss_pre:.6f} "
        f"cls_acc={cls_acc_pre:.4f} audio_acc={audio_acc_pre:.4f} "
        f"baseline_acc={baseline_stats['baseline_accuracy']:.4f}"
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        classification_head.train()

        audio_weight_factor = 0.0 if epoch < config.symbol_warmup_epochs else min(
            1.0, (epoch - config.symbol_warmup_epochs + 1) / max(1, config.audio_ramp_epochs)
        )

        for (
            features,
            target_frames,
            mask,
            labels,
            answer_start_samples,
            answer_len_samples,
        ) in train_loader:
            features = features.to(device)
            target_frames = target_frames.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            answer_start_samples = answer_start_samples.to(device)
            answer_len_samples = answer_len_samples.to(device)

            # Forward
            model_out = model(features, mask, return_hidden=True)
            frame_outputs, symbol_logits, hidden_states = model_out
            symbol_probs = symbol_logits.softmax(dim=-1)

            # Classification loss (auxiliary, not intermediate representation)
            cls_logits = classification_head(hidden_states, mask)
            cls_loss = F.cross_entropy(cls_logits, labels)

            # Inject cls_probs into answer window symbol_probs
            # This bridges the gap between cls_head (which predicts correctly)
            # and audio output (which uses symbol_probs for rendering)
            answer_start_frames = answer_start_samples // config.hop_size
            answer_len_frames = (answer_len_samples + config.hop_size - 1) // config.hop_size
            symbol_probs_guided = apply_cls_guidance_to_answer_window(
                symbol_probs,
                cls_logits.detach(),  # detach to prevent gradient interference
                answer_start_frames,
                answer_len_frames,
                valid_symbol_id=valid_symbol_id,
                invalid_symbol_id=invalid_symbol_id,
                mix_weight=1.0,  # full replacement in answer window
            )

            # Audio reconstruction with symbol guidance
            guided_frames = apply_symbol_guidance(
                frame_outputs, symbol_probs_guided, symbol_templates,
                weight=config.symbol_guidance_weight,
            )
            
            # Directly replace answer window frames with cls-guided templates
            guided_frames = apply_cls_guidance_to_frames(
                guided_frames,
                cls_logits.detach(),
                answer_start_frames,
                answer_len_frames,
                symbol_templates,
                valid_symbol_id=valid_symbol_id,
                invalid_symbol_id=invalid_symbol_id,
            )

            # Audio loss
            wave_lengths = mask.sum(dim=1) * config.hop_size
            pred_wave_full, _ = frames_to_wave(guided_frames, wave_lengths, config.frame_size)
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
                pred_wave, target_wave, effective_lengths = (
                    pred_wave_full,
                    target_wave_full,
                    wave_lengths,
                )
            stft_loss, _ = multi_resolution_stft_loss(
                pred_wave, target_wave, lengths=effective_lengths, config=stft_config
            )
            l1_loss = masked_l1_wave(pred_wave, target_wave, effective_lengths)
            audio_loss = audio_weight_factor * config.audio_weight * (
                config.l1_weight * l1_loss + stft_loss
            )

            # Total loss
            total_loss = audio_loss + config.binary_ce_weight * cls_loss

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"train_task2: non-finite loss (total={total_loss.item()}, "
                    f"audio={audio_loss.item()}, cls={cls_loss.item()})"
                )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(
                f"  epoch {epoch + 1}/{epochs} - "
                f"audio_factor={audio_weight_factor:.2f} "
                f"audio_loss={audio_loss.item():.6f} "
                f"cls_loss={cls_loss.item():.6f}"
            )

    # Post-training evaluation
    loss_post, cls_acc_post, audio_acc_post, predictions = evaluate_task2_model(
        model, classification_head, eval_loader, eval_dataset, device,
        stft_config=stft_config, l1_weight=config.l1_weight,
        symbol_guidance_weight=config.symbol_guidance_weight,
        symbol_templates=symbol_templates,
        hop_size=config.hop_size, frame_size=config.frame_size,
        valid_symbol_id=valid_symbol_id,
        invalid_symbol_id=invalid_symbol_id,
        answer_window_only=config.answer_window_only,
        collect_predictions=True,
    )
    margin = audio_acc_post - baseline_stats["baseline_accuracy"]
    print(
        f"[mini_jmamba][task2] post-training loss={loss_post:.6f} "
        f"cls_acc={cls_acc_post:.4f} audio_acc={audio_acc_post:.4f} "
        f"baseline_acc={baseline_stats['baseline_accuracy']:.4f} margin={margin:.4f}"
    )

    for sample in predictions[:5]:
        print(
            f"example {sample['example_id']}: gold={sample['gold_symbol']} "
            f"audio_pred={sample['pred_symbol']} cls_pred={sample['cls_pred']} "
            f"audio_correct={sample['audio_correct']}"
        )

    metrics = {
        "loss_pre": loss_pre,
        "loss_post": loss_post,
        "cls_accuracy_pre": cls_acc_pre,
        "cls_accuracy_post": cls_acc_post,
        "audio_accuracy_pre": audio_acc_pre,
        "audio_accuracy_post": audio_acc_post,
        "baseline_accuracy": baseline_stats["baseline_accuracy"],
        "baseline_type": baseline_stats["baseline_type"],
        "margin": margin,
        "pred_symbol_counts": dict(Counter([p.get("pred_symbol", "") or "<EMPTY>" for p in predictions])),
        "pred_empty_rate": (
            sum(1 for p in predictions if not p.get("pred_symbol")) / max(1, len(predictions))
        ),
        "cls_audio_disagree_rate": (
            sum(1 for p in predictions if p.get("cls_audio_disagree")) / max(1, len(predictions))
        ),
        "avg_answer_rms": (
            float(np.mean([p.get("answer_rms", 0.0) for p in predictions])) if predictions else 0.0
        ),
        "model_config": asdict(model_config),
        "training_config": asdict(config),
    }

    return predictions, metrics


__all__ = ["Task2TrainingConfig", "mini_jmamba_task2_pipeline"]

