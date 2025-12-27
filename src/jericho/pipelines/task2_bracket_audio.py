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

            # Audio rendering
            guided_frames = apply_symbol_guidance(
                frame_outputs, symbol_probs, symbol_templates,
                weight=symbol_guidance_weight,
            )

            # Compute loss
            wave_lengths = mask.sum(dim=1) * hop_size
            pred_wave, _ = frames_to_wave(guided_frames, wave_lengths, frame_size)
            target_wave, _ = frames_to_wave(target_frames, wave_lengths, frame_size)
            stft_loss, _ = multi_resolution_stft_loss(
                pred_wave, target_wave, lengths=wave_lengths, config=stft_config
            )
            l1_loss = masked_l1_wave(pred_wave, target_wave, wave_lengths)
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
                pred_audio = pred_wave[b].cpu().numpy()
                answer_start = int(answer_start_samples[b].item())
                answer_len = int(answer_len_samples[b].item())
                answer_audio = pred_audio[answer_start:answer_start + answer_len]

                try:
                    decoded = decode_wave_to_symbols(answer_audio)
                except Exception:
                    decoded = []

                gold_symbol = VALID_SYMBOL if labels[b].item() == 0 else INVALID_SYMBOL
                pred_symbol = decoded[0] if len(decoded) == 1 else ""
                audio_correct = pred_symbol == gold_symbol
                correct_audio += int(audio_correct)

                if collect_predictions and sample is not None:
                    predictions.append({
                        "example_id": sample.entry.example_id,
                        "split": sample.entry.split,
                        "gold_symbols": [gold_symbol],
                        "pred_symbols": [pred_symbol] if pred_symbol else [],
                        "exact_match": 1.0 if audio_correct else 0.0,
                        "gold_symbol": gold_symbol,
                        "pred_symbol": pred_symbol,
                        "cls_pred": "V" if cls_preds[b].item() == 0 else "X",
                        "audio_correct": audio_correct,
                        "cls_correct": cls_preds[b].item() == labels[b].item(),
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
) -> Tuple[List[dict], dict]:
    """Train and evaluate Mini-JMamba on Task2 (bracket validity)."""

    if config is None:
        config = Task2TrainingConfig()

    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab = build_task2_vocab()
    symbol_templates = build_symbol_frame_templates(
        vocab, config.frame_size, blank_id=config.ctc_blank_id
    ).to(device)

    train_samples = prepare_task2_samples(
        train_entries, vocab, config.frame_size, config.hop_size,
        blank_id=config.ctc_blank_id, config=config,
    )
    eval_samples = prepare_task2_samples(
        eval_entries, vocab, config.frame_size, config.hop_size,
        blank_id=config.ctc_blank_id, config=config,
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
            1.0, (epoch - config.symbol_warmup_epochs + 1) / 10
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

            # Audio reconstruction with symbol guidance
            guided_frames = apply_symbol_guidance(
                frame_outputs, symbol_probs, symbol_templates,
                weight=config.symbol_guidance_weight,
            )

            # Audio loss
            wave_lengths = mask.sum(dim=1) * config.hop_size
            pred_wave, _ = frames_to_wave(guided_frames, wave_lengths, config.frame_size)
            target_wave, _ = frames_to_wave(target_frames, wave_lengths, config.frame_size)
            stft_loss, _ = multi_resolution_stft_loss(
                pred_wave, target_wave, lengths=wave_lengths, config=stft_config
            )
            l1_loss = masked_l1_wave(pred_wave, target_wave, wave_lengths)
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
        "model_config": asdict(model_config),
        "training_config": asdict(config),
    }

    return predictions, metrics


__all__ = ["Task2TrainingConfig", "mini_jmamba_task2_pipeline"]

