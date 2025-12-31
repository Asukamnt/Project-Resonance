"""Task3 Mod Cross-Domain Pipeline: IPD → Audio.

Phase 3 Task3: Arithmetic modulo reasoning across physical domains.
Input: Expression (e.g., "42%7") in IPD encoding
Output: Remainder (e.g., "0") in Audio encoding

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

from ..domains.optical_intensity import OpticalConfig, OpticalIntensityDomain
from ..symbols import SYMBOL2FREQ, encode_symbols_to_wave
from ..models.mini_jmamba import SSMLikeBlock, AttentionBlock


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Task3CrossConfig:
    """Configuration for Task3 Mod Cross-Domain."""
    
    # Input domain (IPD)
    input_sample_rate: int = 1000
    input_frame_size: int = 10
    input_hop_size: int = 10
    
    # Output domain (Audio)
    output_sample_rate: int = 16000
    output_frame_size: int = 160
    output_hop_size: int = 160
    output_tone_dur: float = 0.1
    
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
    
    # Thinking gap
    thinking_gap_s: float = 0.2
    
    # Sequence alignment - Task3 outputs 1-2 digits
    max_output_symbols: int = 2  # Max 2 digits for remainder
    output_num_frames: int = 20  # 2 symbols × 10 frames each
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # Derived
    thinking_gap_samples: int = field(init=False)
    thinking_gap_frames: int = field(init=False)
    
    def __post_init__(self):
        self.thinking_gap_samples = int(self.input_sample_rate * self.thinking_gap_s)
        self.thinking_gap_frames = self.thinking_gap_samples // self.input_frame_size


# =============================================================================
# Audio Helper
# =============================================================================

class AudioDomainHelper:
    """Audio encoding/decoding helper."""
    
    def __init__(self, sample_rate: int = 16000, tone_dur: float = 0.1):
        self.sample_rate = sample_rate
        self.tone_dur = tone_dur
        self.samples_per_symbol = int(sample_rate * tone_dur)
    
    def encode(self, symbols: List[str], fixed_phase: float = 0.0) -> np.ndarray:
        """Encode symbols to audio."""
        return encode_symbols_to_wave(
            symbols,
            sr=self.sample_rate,
            tone_dur=self.tone_dur,
            gap_dur=0.0,
            fixed_phase=fixed_phase,
        )
    
    def decode(self, wave: np.ndarray) -> List[str]:
        """Decode audio to symbols."""
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
            
            fft = np.fft.rfft(segment)
            freqs = np.fft.rfftfreq(len(segment), 1 / self.sample_rate)
            
            magnitude = np.abs(fft)
            if len(magnitude) > 1:
                magnitude[0] = 0
            peak_idx = np.argmax(magnitude)
            peak_freq = freqs[peak_idx]
            
            # Match to digit symbols only
            digit_freqs = {str(i): SYMBOL2FREQ[str(i)] for i in range(10)}
            
            best_symbol = "?"
            best_dist = float("inf")
            for sym, freq in digit_freqs.items():
                dist = abs(peak_freq - freq)
                if dist < best_dist:
                    best_dist = dist
                    best_symbol = sym
            
            decoded.append(best_symbol)
        
        return decoded


# =============================================================================
# Dataset
# =============================================================================

class Task3CrossDataset(Dataset):
    """Dataset for cross-domain Task3 Mod."""
    
    def __init__(
        self,
        entries: List[Any],
        ipd_domain: OpticalIntensityDomain,
        audio_helper: AudioDomainHelper,
        config: Task3CrossConfig,
    ):
        self.entries = entries
        self.ipd_domain = ipd_domain
        self.audio_helper = audio_helper
        self.config = config
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        
        # Parse expression: "42%7" → dividend=42, divisor=7
        input_symbols = list(entry.symbols)
        expr = "".join(input_symbols)
        
        # Compute remainder
        parts = expr.split("%")
        dividend = int(parts[0])
        divisor = int(parts[1])
        remainder = dividend % divisor
        target_symbols = list(str(remainder))  # e.g., "0" → ["0"] or "12" → ["1", "2"]
        
        # Encode input (IPD)
        input_wave = self.ipd_domain.encode(input_symbols)
        
        # Add thinking gap
        gap_samples = self.config.thinking_gap_samples
        input_wave_with_gap = np.concatenate([
            input_wave,
            np.zeros(gap_samples, dtype=np.float32),
        ])
        
        # Pad to ensure output frames
        min_frames = self.config.output_num_frames
        current_frames = len(input_wave_with_gap) // self.config.input_frame_size
        if current_frames < min_frames:
            pad = (min_frames - current_frames) * self.config.input_frame_size
            input_wave_with_gap = np.concatenate([
                input_wave_with_gap,
                np.zeros(pad, dtype=np.float32),
            ])
        
        # Frame input
        input_frames = self._frame_wave(
            input_wave_with_gap,
            self.config.input_frame_size,
            self.config.input_hop_size,
        )
        
        # Encode target (Audio) - pad to max_output_symbols
        padded_target = target_symbols + ["0"] * (self.config.max_output_symbols - len(target_symbols))
        target_wave = self.audio_helper.encode(padded_target, fixed_phase=0.0)
        
        # Frame target
        target_frames = self._frame_wave(
            target_wave,
            self.config.output_frame_size,
            self.config.output_hop_size,
        )
        
        # Ensure fixed output size
        if len(target_frames) < self.config.output_num_frames:
            pad = np.zeros(
                (self.config.output_num_frames - len(target_frames), self.config.output_frame_size),
                dtype=np.float32,
            )
            target_frames = np.concatenate([target_frames, pad], axis=0)
        elif len(target_frames) > self.config.output_num_frames:
            target_frames = target_frames[:self.config.output_num_frames]
        
        # Number of actual output symbols (for loss masking)
        num_target_symbols = len(target_symbols)
        
        return {
            "input_frames": torch.from_numpy(input_frames).float(),
            "target_frames": torch.from_numpy(target_frames).float(),
            "num_target_symbols": num_target_symbols,
            "input_symbols": input_symbols,
            "target_symbols": target_symbols,
            "example_id": getattr(entry, "id", str(idx)),
        }
    
    def _frame_wave(self, wave: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
        num_frames = (len(wave) - frame_size) // hop_size + 1
        if num_frames <= 0:
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

class Task3CrossModel(nn.Module):
    """Cross-domain model for Task3 Mod: IPD → Audio."""
    
    def __init__(self, config: Task3CrossConfig):
        super().__init__()
        self.config = config
        
        # Input encoder
        self.encoder_in = nn.Linear(config.input_frame_size, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        
        # Core layers
        total_layers = config.num_ssm_layers + config.num_attn_layers
        layers = []
        ssm_count = 0
        attn_count = 0
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
        
        # Output decoder
        self.decoder_out = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.output_frame_size),
        )
    
    def forward(self, input_frames: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Returns: output_frames (batch, output_num_frames, output_frame_size)
        """
        x = self.encoder_in(input_frames)
        x = self.input_dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask=None)
        
        hidden = self.final_norm(x)
        
        # Extract last K frames
        K = self.config.output_num_frames
        hidden_out = hidden[:, -K:, :]
        
        output_frames = self.decoder_out(hidden_out)
        return output_frames


# =============================================================================
# Training
# =============================================================================

def train_task3_cross(
    model: Task3CrossModel,
    train_loader,
    val_loader,
    audio_helper: AudioDomainHelper,
    config: Task3CrossConfig,
    device: torch.device,
    epochs: int,
) -> Dict[str, List[float]]:
    """Train Task3 cross-domain model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    history = {"train_loss": [], "val_loss": [], "val_em": []}
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            input_frames = batch["input_frames"].to(device)
            target_frames = batch["target_frames"].to(device)
            
            optimizer.zero_grad()
            pred_frames = model(input_frames)
            
            l1 = F.l1_loss(pred_frames, target_frames)
            mse = F.mse_loss(pred_frames, target_frames)
            loss = config.l1_weight * l1 + config.mse_weight * mse
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # Validation
        val_metrics = evaluate_task3_cross(model, val_loader, audio_helper, config, device)
        
        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(val_metrics["loss"])
        history["val_em"].append(val_metrics["em"])
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"train_loss={np.mean(train_losses):.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, "
                  f"val_em={val_metrics['em']:.4f}")
    
    return history


def evaluate_task3_cross(
    model: Task3CrossModel,
    data_loader,
    audio_helper: AudioDomainHelper,
    config: Task3CrossConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate Task3 cross-domain model."""
    model.eval()
    
    all_losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_frames = batch["input_frames"].to(device)
            target_frames = batch["target_frames"].to(device)
            target_symbols_batch = batch["target_symbols"]
            num_target_batch = batch["num_target_symbols"]
            
            pred_frames = model(input_frames)
            
            l1 = F.l1_loss(pred_frames, target_frames)
            mse = F.mse_loss(pred_frames, target_frames)
            loss = config.l1_weight * l1 + config.mse_weight * mse
            all_losses.append(loss.item())
            
            # Decode predictions
            pred_frames_np = pred_frames.cpu().numpy()
            
            for i in range(len(pred_frames_np)):
                pred_wave = pred_frames_np[i].flatten()
                pred_symbols = audio_helper.decode(pred_wave)
                
                # Truncate to actual target length
                num_target = num_target_batch[i].item() if hasattr(num_target_batch[i], 'item') else num_target_batch[i]
                pred_truncated = pred_symbols[:num_target]
                target_syms = target_symbols_batch[i]
                
                if pred_truncated == target_syms:
                    correct += 1
                total += 1
    
    em = correct / total if total > 0 else 0.0
    
    return {
        "loss": np.mean(all_losses),
        "em": em,
        "total": total,
        "correct": correct,
    }


def collate_task3_cross(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function."""
    max_in_len = max(b["input_frames"].shape[0] for b in batch)
    out_len = batch[0]["target_frames"].shape[0]
    
    input_frame_size = batch[0]["input_frames"].shape[1]
    output_frame_size = batch[0]["target_frames"].shape[1]
    
    input_frames = torch.zeros(len(batch), max_in_len, input_frame_size)
    target_frames = torch.zeros(len(batch), out_len, output_frame_size)
    num_target_symbols = []
    
    for i, b in enumerate(batch):
        seq_len = b["input_frames"].shape[0]
        input_frames[i, :seq_len] = b["input_frames"]
        target_frames[i] = b["target_frames"]
        num_target_symbols.append(b["num_target_symbols"])
    
    return {
        "input_frames": input_frames,
        "target_frames": target_frames,
        "num_target_symbols": num_target_symbols,
        "input_symbols": [b["input_symbols"] for b in batch],
        "target_symbols": [b["target_symbols"] for b in batch],
        "example_id": [b["example_id"] for b in batch],
    }


__all__ = [
    "Task3CrossConfig",
    "Task3CrossModel",
    "Task3CrossDataset",
    "AudioDomainHelper",
    "train_task3_cross",
    "evaluate_task3_cross",
    "collate_task3_cross",
]

