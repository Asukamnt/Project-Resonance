"""Task1 Mirror pipeline for optical domain (Phase2 pilot).

This is the pilot task for Phase2 - validates the basic encode/decode/model
pipeline before tackling logical reasoning tasks (Bracket, Mod).

Key differences from audio:
- Uses MPPM 2-of-10 encoding (not sine wave frequencies)
- Sample rate: 1 kHz (not 16 kHz)
- Loss: L1 + MSE (not STFT)
- Output: softplus + eps (non-negative constraint, docs v0.2+)

Reference: docs/phase2_light_to_light.md Section 3.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from jericho.data import ManifestEntry
from jericho.domains.optical_intensity import (
    OpticalConfig,
    OpticalIntensityDomain,
)
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.scorer import exact_match


@dataclass
class Task1OpticalConfig:
    """Configuration for Task1 Mirror optical training."""
    
    # Domain config (use defaults from OpticalConfig)
    sample_rate: int = 1000
    symbol_duration: float = 0.1
    num_slots: int = 10
    
    # Frame params (aligned to slot structure)
    frame_size: int = 10  # = slot_samples
    hop_size: int = 10    # = slot_samples
    
    # Model config
    d_model: int = 64  # Smaller than audio (simpler task)
    num_ssm_layers: int = 6
    num_attn_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    
    # Loss weights
    l1_weight: float = 1.0
    mse_weight: float = 1.0
    
    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # Diagnostics
    log_blank_rate: bool = True
    
    @property
    def slot_samples(self) -> int:
        return int(self.sample_rate * self.symbol_duration) // self.num_slots
    
    @property
    def samples_per_symbol(self) -> int:
        return int(self.sample_rate * self.symbol_duration)


class Task1OpticalDataset(Dataset):
    """Dataset for Task1 Mirror in optical domain."""
    
    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        domain: OpticalIntensityDomain,
        config: Task1OpticalConfig,
        ood_params: Optional[Dict] = None,
    ) -> None:
        self.entries = list(entries)
        self.domain = domain
        self.config = config
        self.ood_params = ood_params or {}
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        symbols = entry.symbols
        
        # Encode input
        input_wave = self.domain.encode(symbols)
        
        # Apply channel distortions if specified
        if self.ood_params:
            input_wave = self.domain.channel(input_wave, self.ood_params)
        
        # Target is same as input (Mirror task)
        target_wave = self.domain.encode(symbols)
        
        # Frame the waves
        input_frames = frame_optical_wave(
            input_wave,
            self.config.frame_size,
            self.config.hop_size,
        )
        target_frames = frame_optical_wave(
            target_wave,
            self.config.frame_size,
            self.config.hop_size,
        )
        
        return {
            "input_frames": torch.from_numpy(input_frames),
            "target_frames": torch.from_numpy(target_frames),
            "input_wave": torch.from_numpy(input_wave),
            "target_wave": torch.from_numpy(target_wave),
            "symbols": symbols,
            "example_id": entry.example_id,
        }


def frame_optical_wave(
    wave: np.ndarray,
    frame_size: int,
    hop_size: int,
) -> np.ndarray:
    """Frame optical waveform into overlapping windows.
    
    Parameters
    ----------
    wave : np.ndarray
        Input waveform, shape (T,)
    frame_size : int
        Frame size in samples
    hop_size : int
        Hop size in samples
        
    Returns
    -------
    np.ndarray
        Framed waveform, shape (num_frames, frame_size)
    """
    if wave.size == 0:
        return np.zeros((0, frame_size), dtype=np.float32)
    
    # Pad to ensure complete frames
    num_frames = (len(wave) + hop_size - 1) // hop_size
    padded_length = num_frames * hop_size + (frame_size - hop_size)
    
    if len(wave) < padded_length:
        wave = np.pad(wave, (0, padded_length - len(wave)))
    
    frames = []
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        if end <= len(wave):
            frames.append(wave[start:end])
    
    if not frames:
        return np.zeros((1, frame_size), dtype=np.float32)
    
    return np.stack(frames, axis=0).astype(np.float32)


def unframe_optical_wave(
    frames: np.ndarray,
    hop_size: int,
) -> np.ndarray:
    """Reconstruct waveform from frames (overlap-add for hop < frame_size).
    
    For Phase2 v0.1, hop_size == frame_size, so this is just concatenation.
    """
    if frames.size == 0:
        return np.zeros(0, dtype=np.float32)
    
    num_frames, frame_size = frames.shape
    
    if hop_size == frame_size:
        # Simple concatenation
        return frames.flatten().astype(np.float32)
    else:
        # Overlap-add (for future use)
        output_length = (num_frames - 1) * hop_size + frame_size
        output = np.zeros(output_length, dtype=np.float32)
        norm = np.zeros(output_length, dtype=np.float32)
        
        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_size
            output[start:end] += frames[i]
            norm[start:end] += 1.0
        
        norm = np.maximum(norm, 1e-8)
        return (output / norm).astype(np.float32)


class Task1OpticalModel(nn.Module):
    """Task1 Mirror model for optical domain.
    
    Uses MiniJMamba backbone directly (it has input_proj and frame_head).
    Output is constrained to non-negative via softplus + eps.
    """
    
    def __init__(self, config: Task1OpticalConfig) -> None:
        super().__init__()
        self.config = config
        
        # MiniJMamba backbone (already has input_proj and frame_head)
        backbone_config = MiniJMambaConfig(
            frame_size=config.frame_size,
            hop_size=config.hop_size,
            symbol_vocab_size=21,  # Phase2 vocab size (fixed)
            d_model=config.d_model,
            num_ssm_layers=config.num_ssm_layers,
            num_attn_layers=config.num_attn_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
        )
        self.backbone = MiniJMamba(backbone_config)
    
    def forward(
        self, input_frames: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        input_frames : torch.Tensor
            Shape (batch, num_frames, frame_size)
        padding_mask : torch.Tensor, optional
            Bool tensor where True means valid frame. Shape (batch, num_frames)
            
        Returns
        -------
        torch.Tensor
            Output frames, shape (batch, num_frames, frame_size)
            Values in (0, inf) (softplus activation)
        """
        # MiniJMamba returns (frame_outputs, symbol_logits)
        frame_outputs, _ = self.backbone(input_frames, padding_mask)
        
        # Non-negative constraint: softplus + eps (docs v0.2+)
        return F.softplus(frame_outputs) + 1e-3


def compute_optical_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: Task1OpticalConfig,
) -> Dict[str, torch.Tensor]:
    """Compute loss for optical domain (L1 + MSE, no STFT).
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted frames, shape (batch, num_frames, frame_size)
    target : torch.Tensor
        Target frames, same shape
    config : Task1OpticalConfig
        Configuration with loss weights
        
    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with 'loss', 'l1_loss', 'mse_loss'
    """
    # Ensure pred is non-negative (softplus output is already non-neg, clamp upper for safety)
    pred = pred.clamp(0, 1.5)  # Allow some headroom above 1.0
    
    l1_loss = F.l1_loss(pred, target)
    mse_loss = F.mse_loss(pred, target)
    
    total_loss = config.l1_weight * l1_loss + config.mse_weight * mse_loss
    
    return {
        "loss": total_loss,
        "l1_loss": l1_loss,
        "mse_loss": mse_loss,
    }


def evaluate_task1_optical(
    model: Task1OpticalModel,
    dataloader: DataLoader,
    domain: OpticalIntensityDomain,
    config: Task1OpticalConfig,
    device: torch.device,
) -> Dict:
    """Evaluate Task1 Mirror model.
    
    Returns EM score and diagnostic metrics.
    """
    model.eval()
    
    total_em = 0.0
    total_samples = 0
    blank_count = 0
    total_symbols = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_frames = batch["input_frames"].to(device)
            symbols_list = batch["symbols"]
            
            # Forward pass
            pred_frames = model(input_frames)
            
            # Decode predictions
            for i in range(len(symbols_list)):
                pred_frame = pred_frames[i].cpu().numpy()
                pred_wave = unframe_optical_wave(pred_frame, config.hop_size)
                pred_symbols = domain.decode(pred_wave)
                
                target_symbols = list(symbols_list[i])
                
                # Compute EM
                em = exact_match(pred_symbols, target_symbols)
                total_em += em
                total_samples += 1
                
                # Diagnostics
                blank_count += sum(1 for s in pred_symbols if s == "?")
                total_symbols += len(pred_symbols)
    
    return {
        "em": total_em / max(total_samples, 1),
        "num_samples": total_samples,
        "blank_rate": blank_count / max(total_symbols, 1),
    }


def train_task1_optical(
    model: Task1OpticalModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    domain: OpticalIntensityDomain,
    config: Task1OpticalConfig,
    device: torch.device,
    epochs: Optional[int] = None,
) -> Dict:
    """Train Task1 Mirror optical model.
    
    Returns training history.
    """
    epochs = epochs or config.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    history = {
        "train_loss": [],
        "val_em": [],
        "val_blank_rate": [],
    }
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            input_frames = batch["input_frames"].to(device)
            target_frames = batch["target_frames"].to(device)
            
            optimizer.zero_grad()
            pred_frames = model(input_frames)
            
            losses = compute_optical_loss(pred_frames, target_frames, config)
            loss = losses["loss"]
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        history["train_loss"].append(avg_loss)
        
        # Validation
        val_metrics = evaluate_task1_optical(
            model, val_loader, domain, config, device
        )
        history["val_em"].append(val_metrics["em"])
        history["val_blank_rate"].append(val_metrics["blank_rate"])
        
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Val EM: {val_metrics['em']:.4f} - "
            f"Blank: {val_metrics['blank_rate']:.4f}"
        )
    
    return history


__all__ = [
    "Task1OpticalConfig",
    "Task1OpticalDataset",
    "Task1OpticalModel",
    "frame_optical_wave",
    "unframe_optical_wave",
    "compute_optical_loss",
    "evaluate_task1_optical",
    "train_task1_optical",
]

