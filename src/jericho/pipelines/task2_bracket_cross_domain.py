"""Task2 Bracket Cross-Domain Pipeline: IPD → Audio.

Phase 3 core pipeline: Input is IPD (1 kHz MPPM), Output is Audio (16 kHz frequency).
This validates cross-physical-domain reasoning capability.

Reference: docs/phase2-4/phase3_light_to_sound.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..domains.optical_intensity import (
    OpticalConfig,
    OpticalIntensityDomain,
)
from ..symbols import SYMBOL2FREQ, encode_symbols_to_wave, SR, TONE_DUR, GAP_DUR
from ..task2.utils import is_balanced
from ..models.mini_jmamba import SSMLikeBlock, AttentionBlock


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Phase3Config:
    """Configuration for Phase 3 Cross-Domain Task2."""
    
    # Input domain (IPD)
    input_sample_rate: int = 1000
    input_frame_size: int = 10  # = slot_samples
    input_hop_size: int = 10
    
    # Output domain (Audio)
    output_sample_rate: int = 16000
    output_frame_size: int = 160  # 10ms frames at 16kHz
    output_hop_size: int = 160
    output_tone_dur: float = 0.1  # 100ms per symbol
    output_gap_dur: float = 0.0   # No gap for single-symbol output
    
    # Core model
    d_model: int = 128
    num_ssm_layers: int = 10
    num_attn_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    
    # Loss weights
    l1_weight: float = 1.0
    mse_weight: float = 1.0
    binary_ce_weight: float = 1.0  # Auxiliary classification loss
    
    # Thinking gap (IPD domain)
    thinking_gap_s: float = 0.2  # 200ms = 200 IPD samples = 20 frames
    
    # Sequence alignment
    output_num_frames: int = 10  # Fixed output: 10 frames (1 symbol)
    
    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    binary_warmup_epochs: int = 10
    
    # Derived values
    thinking_gap_samples: int = field(init=False)
    thinking_gap_frames: int = field(init=False)
    output_samples_per_symbol: int = field(init=False)
    
    def __post_init__(self):
        self.thinking_gap_samples = int(self.input_sample_rate * self.thinking_gap_s)
        self.thinking_gap_frames = self.thinking_gap_samples // self.input_frame_size
        self.output_samples_per_symbol = int(self.output_sample_rate * self.output_tone_dur)


# =============================================================================
# Audio Domain Helper (simplified, no gap for single symbol)
# =============================================================================

class AudioDomainHelper:
    """Helper for audio encoding/decoding (Phase 1 compatible)."""
    
    def __init__(self, sample_rate: int = 16000, tone_dur: float = 0.1):
        self.sample_rate = sample_rate
        self.tone_dur = tone_dur
        self.samples_per_symbol = int(sample_rate * tone_dur)
    
    def encode(self, symbols: List[str], fixed_phase: float = 0.0) -> np.ndarray:
        """Encode symbols to audio waveform."""
        return encode_symbols_to_wave(
            symbols,
            sr=self.sample_rate,
            tone_dur=self.tone_dur,
            gap_dur=0.0,  # No gap for cross-domain output
            fixed_phase=fixed_phase,
        )
    
    def decode(self, wave: np.ndarray) -> List[str]:
        """Decode audio waveform to symbols using FFT."""
        if wave.size == 0:
            return []
        
        num_symbols = wave.size // self.samples_per_symbol
        if num_symbols == 0:
            return []
        
        decoded = []
        for sym_idx in range(num_symbols):
            start = sym_idx * self.samples_per_symbol
            end = start + self.samples_per_symbol
            segment = wave[start:end]
            
            # FFT to find dominant frequency
            fft = np.fft.rfft(segment)
            freqs = np.fft.rfftfreq(len(segment), 1 / self.sample_rate)
            
            # Find peak (ignore DC)
            magnitude = np.abs(fft)
            if len(magnitude) > 1:
                magnitude[0] = 0  # Ignore DC
            peak_idx = np.argmax(magnitude)
            peak_freq = freqs[peak_idx]
            
            # Match to nearest symbol
            best_symbol = "?"
            best_dist = float("inf")
            for sym, freq in SYMBOL2FREQ.items():
                dist = abs(peak_freq - freq)
                if dist < best_dist:
                    best_dist = dist
                    best_symbol = sym
            
            decoded.append(best_symbol)
        
        return decoded


# =============================================================================
# Dataset
# =============================================================================

@dataclass
class CrossDomainSample:
    """A single cross-domain sample."""
    input_symbols: List[str]
    target_symbols: List[str]
    input_wave: np.ndarray
    target_wave: np.ndarray
    input_frames: np.ndarray  # (seq_in, input_frame_size)
    target_frames: np.ndarray  # (output_num_frames, output_frame_size)
    binary_label: int  # 1 = Valid (V), 0 = Invalid (X)
    example_id: str


class CrossDomainDataset(Dataset):
    """Dataset for cross-domain Task2 (IPD → Audio)."""
    
    def __init__(
        self,
        entries: List[Any],  # ManifestEntry or similar
        ipd_domain: OpticalIntensityDomain,
        audio_helper: AudioDomainHelper,
        config: Phase3Config,
    ):
        self.entries = entries
        self.ipd_domain = ipd_domain
        self.audio_helper = audio_helper
        self.config = config
        
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        
        # Get symbols
        input_symbols = list(entry.symbols)
        is_valid = is_balanced("".join(input_symbols))
        target_symbols = ["V"] if is_valid else ["X"]
        binary_label = 1 if is_valid else 0
        
        # Encode input (IPD domain)
        input_wave = self.ipd_domain.encode(input_symbols)
        
        # Add thinking gap (zeros in IPD domain)
        gap_samples = self.config.thinking_gap_samples
        input_wave_with_gap = np.concatenate([
            input_wave,
            np.zeros(gap_samples, dtype=np.float32),
        ])
        
        # Pad to ensure output frames can be extracted
        # We need at least output_num_frames at the end
        min_input_frames = self.config.output_num_frames
        current_frames = len(input_wave_with_gap) // self.config.input_frame_size
        if current_frames < min_input_frames:
            pad_frames = min_input_frames - current_frames
            pad_samples = pad_frames * self.config.input_frame_size
            input_wave_with_gap = np.concatenate([
                input_wave_with_gap,
                np.zeros(pad_samples, dtype=np.float32),
            ])
        
        # Frame the input
        input_frames = self._frame_wave(
            input_wave_with_gap,
            self.config.input_frame_size,
            self.config.input_hop_size,
        )
        
        # Encode target (Audio domain)
        target_wave = self.audio_helper.encode(target_symbols, fixed_phase=0.0)
        
        # Frame the target
        target_frames = self._frame_wave(
            target_wave,
            self.config.output_frame_size,
            self.config.output_hop_size,
        )
        
        # Ensure target has exactly output_num_frames
        if len(target_frames) < self.config.output_num_frames:
            pad = np.zeros(
                (self.config.output_num_frames - len(target_frames), self.config.output_frame_size),
                dtype=np.float32,
            )
            target_frames = np.concatenate([target_frames, pad], axis=0)
        elif len(target_frames) > self.config.output_num_frames:
            target_frames = target_frames[:self.config.output_num_frames]
        
        return {
            "input_frames": torch.from_numpy(input_frames).float(),
            "target_frames": torch.from_numpy(target_frames).float(),
            "binary_label": torch.tensor(binary_label, dtype=torch.float32),
            "input_symbols": input_symbols,
            "target_symbols": target_symbols,
            "example_id": getattr(entry, "id", str(idx)),
        }
    
    def _frame_wave(
        self,
        wave: np.ndarray,
        frame_size: int,
        hop_size: int,
    ) -> np.ndarray:
        """Frame waveform into overlapping windows."""
        num_frames = (len(wave) - frame_size) // hop_size + 1
        if num_frames <= 0:
            # Pad to at least one frame
            padded = np.zeros(frame_size, dtype=np.float32)
            padded[:len(wave)] = wave
            return padded.reshape(1, -1)
        
        frames = np.zeros((num_frames, frame_size), dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_size
            frames[i] = wave[start:start + frame_size]
        
        return frames


# =============================================================================
# Model
# =============================================================================

class Phase3Model(nn.Module):
    """Cross-domain model: IPD → Audio.
    
    Architecture:
    - encoder_in: Linear projection from IPD frame to d_model
    - core: Stack of SSM and Attention blocks (from MiniJMamba)
    - decoder_out: MLP from d_model to Audio frame
    - cls_head: Binary classification head (auxiliary)
    """
    
    def __init__(self, config: Phase3Config):
        super().__init__()
        self.config = config
        
        # Input encoder (IPD frames → d_model)
        self.encoder_in = nn.Linear(config.input_frame_size, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        
        # Core reasoning backbone: interleaved SSM + Attention blocks
        total_layers = config.num_ssm_layers + config.num_attn_layers
        layers = []
        ssm_count = 0
        attn_count = 0
        
        # Interleave: ratio based on counts
        ssm_interval = max(1, total_layers // max(1, config.num_ssm_layers))
        
        for i in range(total_layers):
            if (i % ssm_interval != 0 or attn_count >= config.num_attn_layers) and ssm_count < config.num_ssm_layers:
                layers.append(SSMLikeBlock(config.d_model, config.dropout))
                ssm_count += 1
            elif attn_count < config.num_attn_layers:
                layers.append(AttentionBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                    use_rope=True,
                ))
                attn_count += 1
            else:
                layers.append(SSMLikeBlock(config.d_model, config.dropout))
                ssm_count += 1
        
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        
        # Output decoder (d_model → Audio frames)
        self.decoder_out = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.output_frame_size),
        )
        
        # Binary classification head (auxiliary)
        self.cls_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )
    
    def forward(
        self,
        input_frames: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Parameters
        ----------
        input_frames : torch.Tensor
            Shape (batch, seq_in, input_frame_size)
        return_hidden : bool
            If True, also return hidden states
            
        Returns
        -------
        output_frames : torch.Tensor
            Shape (batch, output_num_frames, output_frame_size)
        cls_logits : torch.Tensor
            Shape (batch, 1)
        hidden : Optional[torch.Tensor]
            Shape (batch, seq_in, d_model) if return_hidden=True
        """
        # Encode IPD frames
        x = self.encoder_in(input_frames)  # (batch, seq_in, d_model)
        x = self.input_dropout(x)
        
        # Core reasoning through SSM + Attention blocks
        for layer in self.layers:
            x = layer(x, mask=None)
        
        hidden = self.final_norm(x)  # (batch, seq_in, d_model)
        
        # Extract last K frames for output
        K = self.config.output_num_frames
        hidden_out = hidden[:, -K:, :]  # (batch, K, d_model)
        
        # Decode to Audio frames
        output_frames = self.decoder_out(hidden_out)  # (batch, K, output_frame_size)
        
        # Classification from pooled hidden
        pooled = hidden.mean(dim=1)  # (batch, d_model)
        cls_logits = self.cls_head(pooled)  # (batch, 1)
        
        if return_hidden:
            return output_frames, cls_logits, hidden
        return output_frames, cls_logits, None


# =============================================================================
# Training
# =============================================================================

def compute_cross_domain_loss(
    pred_frames: torch.Tensor,
    target_frames: torch.Tensor,
    cls_logits: torch.Tensor,
    binary_labels: torch.Tensor,
    config: Phase3Config,
    epoch: int = 0,
) -> Dict[str, torch.Tensor]:
    """Compute cross-domain loss.
    
    Returns dict with 'loss', 'l1', 'mse', 'bce' keys.
    """
    # Time-domain losses (Audio output)
    l1_loss = F.l1_loss(pred_frames, target_frames)
    mse_loss = F.mse_loss(pred_frames, target_frames)
    
    # Binary CE loss (auxiliary classification)
    bce_loss = F.binary_cross_entropy_with_logits(
        cls_logits.squeeze(-1),
        binary_labels,
    )
    
    # Warmup: gradually increase BCE weight
    bce_weight = config.binary_ce_weight
    if epoch < config.binary_warmup_epochs:
        bce_weight = config.binary_ce_weight * (epoch + 1) / config.binary_warmup_epochs
    
    total = (
        config.l1_weight * l1_loss +
        config.mse_weight * mse_loss +
        bce_weight * bce_loss
    )
    
    return {
        "loss": total,
        "l1": l1_loss,
        "mse": mse_loss,
        "bce": bce_loss,
    }


def train_cross_domain(
    model: Phase3Model,
    train_loader,
    val_loader,
    audio_helper: AudioDomainHelper,
    config: Phase3Config,
    device: torch.device,
    epochs: int,
) -> Dict[str, List[float]]:
    """Train cross-domain model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    history = {"train_loss": [], "val_loss": [], "val_em": []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            input_frames = batch["input_frames"].to(device)
            target_frames = batch["target_frames"].to(device)
            binary_labels = batch["binary_label"].to(device)
            
            optimizer.zero_grad()
            pred_frames, cls_logits, _ = model(input_frames)
            
            losses = compute_cross_domain_loss(
                pred_frames, target_frames, cls_logits, binary_labels,
                config, epoch
            )
            
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(losses["loss"].item())
        
        scheduler.step()
        
        # Validation
        val_metrics = evaluate_cross_domain(model, val_loader, audio_helper, config, device)
        
        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(val_metrics["loss"])
        history["val_em"].append(val_metrics["em"])
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={np.mean(train_losses):.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, "
              f"val_em={val_metrics['em']:.4f}, "
              f"val_cls_acc={val_metrics['cls_acc']:.4f}")
    
    return history


def evaluate_cross_domain(
    model: Phase3Model,
    data_loader,
    audio_helper: AudioDomainHelper,
    config: Phase3Config,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate cross-domain model."""
    model.eval()
    
    all_losses = []
    correct = 0
    total = 0
    cls_correct = 0
    v_count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_frames = batch["input_frames"].to(device)
            target_frames = batch["target_frames"].to(device)
            binary_labels = batch["binary_label"].to(device)
            target_symbols_batch = batch["target_symbols"]
            
            pred_frames, cls_logits, _ = model(input_frames)
            
            losses = compute_cross_domain_loss(
                pred_frames, target_frames, cls_logits, binary_labels,
                config, epoch=config.epochs  # Full weight
            )
            all_losses.append(losses["loss"].item())
            
            # Decode predictions
            pred_frames_np = pred_frames.cpu().numpy()
            
            for i in range(len(pred_frames_np)):
                # Reconstruct waveform from frames
                pred_wave = pred_frames_np[i].flatten()
                pred_symbols = audio_helper.decode(pred_wave)
                
                target_syms = target_symbols_batch[i]
                
                # Exact match
                if pred_symbols == target_syms:
                    correct += 1
                total += 1
                
                # Track V rate
                if pred_symbols and pred_symbols[0] == "V":
                    v_count += 1
            
            # Classification accuracy
            cls_preds = (torch.sigmoid(cls_logits.squeeze(-1)) > 0.5).float()
            cls_correct += (cls_preds == binary_labels).sum().item()
    
    em = correct / total if total > 0 else 0.0
    cls_acc = cls_correct / total if total > 0 else 0.0
    v_rate = v_count / total if total > 0 else 0.0
    
    return {
        "loss": np.mean(all_losses),
        "em": em,
        "cls_acc": cls_acc,
        "v_rate": v_rate,
        "total": total,
        "correct": correct,
    }


# =============================================================================
# Collate function
# =============================================================================

def collate_cross_domain(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for cross-domain dataset."""
    # Find max input length
    max_in_len = max(b["input_frames"].shape[0] for b in batch)
    out_len = batch[0]["target_frames"].shape[0]  # Fixed
    
    input_frame_size = batch[0]["input_frames"].shape[1]
    output_frame_size = batch[0]["target_frames"].shape[1]
    
    # Pad input sequences
    input_frames = torch.zeros(len(batch), max_in_len, input_frame_size)
    target_frames = torch.zeros(len(batch), out_len, output_frame_size)
    binary_labels = torch.zeros(len(batch))
    
    for i, b in enumerate(batch):
        seq_len = b["input_frames"].shape[0]
        input_frames[i, :seq_len] = b["input_frames"]
        target_frames[i] = b["target_frames"]
        binary_labels[i] = b["binary_label"]
    
    return {
        "input_frames": input_frames,
        "target_frames": target_frames,
        "binary_label": binary_labels,
        "input_symbols": [b["input_symbols"] for b in batch],
        "target_symbols": [b["target_symbols"] for b in batch],
        "example_id": [b["example_id"] for b in batch],
    }


__all__ = [
    "Phase3Config",
    "Phase3Model",
    "CrossDomainDataset",
    "CrossDomainSample",
    "AudioDomainHelper",
    "train_cross_domain",
    "evaluate_cross_domain",
    "collate_cross_domain",
    "compute_cross_domain_loss",
]

