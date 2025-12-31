"""Task2 Bracket Validity pipeline for optical domain (Phase2 main task).

This is the main logical reasoning task for Phase2 - binary classification
of bracket sequence validity in the optical domain.

Input: Bracket sequence encoded as MPPM waveform
Output: "V" (valid) or "X" (invalid) encoded as MPPM waveform

Key aspects:
- Uses thinking gap between input and output
- answer-window-only loss to avoid OOD-length gradient dilution
- Binary CE auxiliary loss for classification guidance

Reference: docs/phase2_light_to_light.md Section 3.1
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
    OPTICAL_SLOT_MAPPING,
)
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.scorer import exact_match
from jericho.task2 import is_balanced

from .task1_mirror_optical import (
    frame_optical_wave,
    unframe_optical_wave,
)


@dataclass
class Task2OpticalConfig:
    """Configuration for Task2 Bracket optical training."""
    
    # Domain config
    sample_rate: int = 1000
    symbol_duration: float = 0.1
    num_slots: int = 10
    
    # Frame params (aligned to slot structure)
    frame_size: int = 10
    hop_size: int = 10
    
    # Model config
    d_model: int = 128
    num_ssm_layers: int = 10
    num_attn_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    
    # Loss weights
    l1_weight: float = 1.0
    mse_weight: float = 1.0
    binary_ce_weight: float = 1.0  # Binary classification auxiliary
    
    # Answer window only (avoid length dilution)
    answer_window_only: bool = True
    
    # Thinking gap
    thinking_gap_s: float = 0.2  # 200ms gap between input and output
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # Warmup
    binary_warmup_epochs: int = 10
    
    @property
    def slot_samples(self) -> int:
        return int(self.sample_rate * self.symbol_duration) // self.num_slots
    
    @property
    def samples_per_symbol(self) -> int:
        return int(self.sample_rate * self.symbol_duration)
    
    @property
    def thinking_gap_samples(self) -> int:
        samples = int(self.sample_rate * self.thinking_gap_s)
        # Align to hop
        return (samples // self.hop_size) * self.hop_size


class Task2OpticalDataset(Dataset):
    """Dataset for Task2 Bracket in optical domain."""
    
    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        domain: OpticalIntensityDomain,
        config: Task2OpticalConfig,
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
        input_symbols = entry.symbols
        target_symbols = entry.target_symbols if hasattr(entry, 'target_symbols') else None
        
        # Determine target if not provided
        if target_symbols is None:
            is_valid = is_balanced(input_symbols)
            target_symbols = ["V"] if is_valid else ["X"]
        
        # Encode input
        input_wave = self.domain.encode(input_symbols)
        
        # Apply channel distortions
        if self.ood_params:
            input_wave = self.domain.channel(input_wave, self.ood_params)
        
        # Create thinking gap (silence)
        gap_samples = self.config.thinking_gap_samples
        gap = np.zeros(gap_samples, dtype=np.float32)
        
        # Encode target (answer window)
        target_wave = self.domain.encode(target_symbols)
        
        # Concatenate: input + gap + target
        full_input = np.concatenate([input_wave, gap, np.zeros_like(target_wave)])
        full_target = np.concatenate([
            np.zeros_like(input_wave),
            np.zeros(gap_samples, dtype=np.float32),
            target_wave,
        ])
        
        # Frame
        input_frames = frame_optical_wave(
            full_input, self.config.frame_size, self.config.hop_size
        )
        target_frames = frame_optical_wave(
            full_target, self.config.frame_size, self.config.hop_size
        )
        
        # Answer window mask (for answer-window-only loss)
        num_frames = target_frames.shape[0]
        answer_start_sample = len(input_wave) + gap_samples
        answer_start_frame = answer_start_sample // self.config.hop_size
        
        answer_mask = np.zeros(num_frames, dtype=np.float32)
        if answer_start_frame < num_frames:
            answer_mask[answer_start_frame:] = 1.0
        
        # Binary label for CE loss
        binary_label = 1.0 if target_symbols[0] == "V" else 0.0
        
        return {
            "input_frames": torch.from_numpy(input_frames),
            "target_frames": torch.from_numpy(target_frames),
            "answer_mask": torch.from_numpy(answer_mask),
            "binary_label": torch.tensor(binary_label),
            "input_symbols": input_symbols,
            "target_symbols": target_symbols,
            "example_id": entry.example_id,
        }


class Task2OpticalModel(nn.Module):
    """Task2 Bracket model for optical domain.
    
    Similar to Task1 but with:
    - Binary classification head for auxiliary loss
    - Larger model for reasoning capability
    """
    
    def __init__(self, config: Task2OpticalConfig) -> None:
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
        
        # Binary classification head (auxiliary)
        self.cls_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )
    
    def forward(
        self,
        input_frames: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_cls: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Parameters
        ----------
        input_frames : torch.Tensor
            Shape (batch, num_frames, frame_size)
        padding_mask : torch.Tensor, optional
            Bool tensor where True means valid frame. Shape (batch, num_frames)
        return_cls : bool
            If True, also return classification logits
            
        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            - Output frames, shape (batch, num_frames, frame_size), values in [0, 1]
            - Classification logits if return_cls, else None
        """
        # MiniJMamba returns (frame_outputs, symbol_logits, hidden) when return_hidden=True
        frame_outputs, _, hidden = self.backbone(
            input_frames, padding_mask, return_hidden=True
        )
        
        # Apply softplus for non-negative constraint (docs: v0.2+)
        # softplus + eps ensures gradient flow while staying non-negative
        output = F.softplus(frame_outputs) + 1e-3
        
        cls_logits = None
        if return_cls:
            # Pool over sequence for classification
            pooled = hidden.mean(dim=1)  # (batch, d_model)
            cls_logits = self.cls_head(pooled).squeeze(-1)  # (batch,)
        
        return output, cls_logits


def compute_task2_optical_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    answer_mask: torch.Tensor,
    cls_logits: Optional[torch.Tensor],
    binary_label: torch.Tensor,
    config: Task2OpticalConfig,
    epoch: int = 0,
) -> Dict[str, torch.Tensor]:
    """Compute loss for Task2 optical.
    
    Uses answer-window-only loss and optional binary CE.
    """
    # Ensure pred is non-negative (softplus output, clamp upper for safety)
    pred = pred.clamp(0, 1.5)
    
    if config.answer_window_only:
        # Apply answer mask
        mask = answer_mask.unsqueeze(-1)  # (batch, num_frames, 1)
        pred_masked = pred * mask
        target_masked = target * mask
        
        # Compute loss only on answer window
        # Normalize by mask sum to avoid length dilution
        mask_sum = mask.sum() + 1e-8
        l1_loss = (F.l1_loss(pred_masked, target_masked, reduction='sum')) / mask_sum
        mse_loss = (F.mse_loss(pred_masked, target_masked, reduction='sum')) / mask_sum
    else:
        l1_loss = F.l1_loss(pred, target)
        mse_loss = F.mse_loss(pred, target)
    
    total_loss = config.l1_weight * l1_loss + config.mse_weight * mse_loss
    
    # Binary CE loss (with warmup)
    binary_ce_loss = torch.tensor(0.0, device=pred.device)
    if cls_logits is not None and epoch >= config.binary_warmup_epochs:
        binary_ce_loss = F.binary_cross_entropy_with_logits(
            cls_logits, binary_label
        )
        total_loss = total_loss + config.binary_ce_weight * binary_ce_loss
    
    return {
        "loss": total_loss,
        "l1_loss": l1_loss,
        "mse_loss": mse_loss,
        "binary_ce_loss": binary_ce_loss,
    }


def evaluate_task2_optical(
    model: Task2OpticalModel,
    dataloader: DataLoader,
    domain: OpticalIntensityDomain,
    config: Task2OpticalConfig,
    device: torch.device,
) -> Dict:
    """Evaluate Task2 Bracket model.
    
    Returns EM score, classification accuracy, and diagnostics.
    """
    model.eval()
    
    total_em = 0.0
    total_cls_correct = 0.0
    total_samples = 0
    blank_count = 0
    v_count = 0
    x_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_frames = batch["input_frames"].to(device)
            target_symbols_list = batch["target_symbols"]
            binary_labels = batch["binary_label"].to(device)
            
            # Forward pass
            pred_frames, cls_logits = model(input_frames, return_cls=True)
            
            # Classification accuracy
            if cls_logits is not None:
                cls_preds = (torch.sigmoid(cls_logits) > 0.5).float()
                total_cls_correct += (cls_preds == binary_labels).sum().item()
            
            # Decode predictions
            for i in range(len(target_symbols_list)):
                pred_frame = pred_frames[i].cpu().numpy()
                pred_wave = unframe_optical_wave(pred_frame, config.hop_size)
                
                # Extract answer window
                input_len = len(batch["input_symbols"][i]) * config.samples_per_symbol
                gap_len = config.thinking_gap_samples
                answer_start = input_len + gap_len
                
                if answer_start < len(pred_wave):
                    answer_wave = pred_wave[answer_start:]
                    pred_symbols = domain.decode(answer_wave)
                else:
                    pred_symbols = ["?"]
                
                target_symbols = list(target_symbols_list[i])
                
                # Take first predicted symbol for comparison
                pred_first = pred_symbols[0] if pred_symbols else "?"
                target_first = target_symbols[0] if target_symbols else "?"
                
                em = 1.0 if pred_first == target_first else 0.0
                total_em += em
                total_samples += 1
                
                # Diagnostics
                if pred_first == "?":
                    blank_count += 1
                elif pred_first == "V":
                    v_count += 1
                elif pred_first == "X":
                    x_count += 1
    
    return {
        "em": total_em / max(total_samples, 1),
        "cls_acc": total_cls_correct / max(total_samples, 1),
        "num_samples": total_samples,
        "blank_rate": blank_count / max(total_samples, 1),
        "v_rate": v_count / max(total_samples, 1),
        "x_rate": x_count / max(total_samples, 1),
    }


def train_task2_optical(
    model: Task2OpticalModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    domain: OpticalIntensityDomain,
    config: Task2OpticalConfig,
    device: torch.device,
    epochs: Optional[int] = None,
) -> Dict:
    """Train Task2 Bracket optical model."""
    epochs = epochs or config.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    history = {
        "train_loss": [],
        "val_em": [],
        "val_cls_acc": [],
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
            answer_mask = batch["answer_mask"].to(device)
            binary_label = batch["binary_label"].to(device)
            
            optimizer.zero_grad()
            pred_frames, cls_logits = model(input_frames, return_cls=True)
            
            losses = compute_task2_optical_loss(
                pred_frames, target_frames, answer_mask,
                cls_logits, binary_label, config, epoch
            )
            loss = losses["loss"]
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        history["train_loss"].append(avg_loss)
        
        # Validation
        val_metrics = evaluate_task2_optical(
            model, val_loader, domain, config, device
        )
        history["val_em"].append(val_metrics["em"])
        history["val_cls_acc"].append(val_metrics["cls_acc"])
        history["val_blank_rate"].append(val_metrics["blank_rate"])
        
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Val EM: {val_metrics['em']:.4f} - "
            f"CLS: {val_metrics['cls_acc']:.4f} - "
            f"V/X/Blank: {val_metrics['v_rate']:.2f}/{val_metrics['x_rate']:.2f}/{val_metrics['blank_rate']:.2f}"
        )
    
    return history


__all__ = [
    "Task2OpticalConfig",
    "Task2OpticalDataset",
    "Task2OpticalModel",
    "compute_task2_optical_loss",
    "evaluate_task2_optical",
    "train_task2_optical",
]

