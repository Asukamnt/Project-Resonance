#!/usr/bin/env python3
"""
Training script for Turing Machine tasks.

Trains Mini-JMamba on waveform-encoded Turing machine tasks,
with hidden state visualization for internal clock analysis.
"""

import argparse
import json
import sys
import io
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Ensure UTF-8 output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jericho.symbols import encode_symbols_to_wave, SR
from src.jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig


# Extended symbol vocabulary for Turing machine tasks
TM_SYMBOLS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Digits
    'A', 'B', 'C',  # Counter symbols
    '(', ')',  # Brackets
    '+',  # Addition operator
    'I', 'D', 'T', 'H',  # Program instructions
]

# Map symbols to frequencies (spread across audible range)
TM_SYMBOL2FREQ = {s: 220 + i * 50 for i, s in enumerate(TM_SYMBOLS)}
TM_SYMBOL2IDX = {s: i for i, s in enumerate(TM_SYMBOLS)}
TM_IDX2SYMBOL = {i: s for s, i in TM_SYMBOL2IDX.items()}


def encode_tm_symbols_to_wave(
    symbols: List[str],
    sr: int = SR,
    tone_dur: float = 0.1,
    gap_dur: float = 0.02,
) -> np.ndarray:
    """Encode Turing machine symbols to waveform."""
    waves = []
    samples_per_tone = int(sr * tone_dur)
    samples_per_gap = int(sr * gap_dur)
    
    for symbol in symbols:
        if symbol not in TM_SYMBOL2FREQ:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        freq = TM_SYMBOL2FREQ[symbol]
        t = np.linspace(0, tone_dur, samples_per_tone, endpoint=False)
        tone = np.sin(2 * np.pi * freq * t)
        waves.append(tone)
        
        # Add gap
        if samples_per_gap > 0:
            gap = np.zeros(samples_per_gap)
            waves.append(gap)
    
    return np.concatenate(waves).astype(np.float32)


class TuringMachineDataset(Dataset):
    """Dataset for Turing machine tasks."""
    
    def __init__(
        self,
        manifest_path: Path,
        max_input_len: int = 32,
        max_output_len: int = 4,
        frame_size: int = 160,
        hop_size: int = 80,
    ):
        self.manifest_path = manifest_path
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.frame_size = frame_size
        self.hop_size = hop_size
        
        # Load samples
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_symbols = sample['input_symbols']
        output_symbols = sample['output_symbols']
        
        # Encode to waveform
        try:
            wave = encode_tm_symbols_to_wave(input_symbols)
        except ValueError as e:
            print(f"Warning: {e}, skipping sample {idx}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Frame the waveform
        frames = self._frame_wave(wave)
        
        # Encode output as class indices
        output_indices = [TM_SYMBOL2IDX.get(s, 0) for s in output_symbols[:self.max_output_len]]
        
        # Pad output if needed
        while len(output_indices) < self.max_output_len:
            output_indices.append(0)  # Pad with 0
        
        return {
            'frames': torch.tensor(frames, dtype=torch.float32),
            'targets': torch.tensor(output_indices, dtype=torch.long),
            'input_symbols': input_symbols,
            'output_symbols': output_symbols,
            'sample_id': sample['id'],
        }
    
    def _frame_wave(self, wave: np.ndarray) -> np.ndarray:
        """Convert waveform to frames."""
        # Pad to ensure we have enough frames
        pad_len = self.frame_size + (self.max_input_len * 2 - 1) * self.hop_size
        if len(wave) < pad_len:
            wave = np.pad(wave, (0, pad_len - len(wave)))
        
        # Extract frames
        num_frames = min(
            (len(wave) - self.frame_size) // self.hop_size + 1,
            self.max_input_len * 2  # Allow 2 frames per symbol
        )
        
        frames = np.zeros((num_frames, self.frame_size), dtype=np.float32)
        for i in range(num_frames):
            start = i * self.hop_size
            frames[i] = wave[start:start + self.frame_size]
        
        return frames


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    # Find max frame length in batch
    max_frames = max(item['frames'].shape[0] for item in batch)
    
    # Pad frames
    padded_frames = []
    padding_masks = []
    
    for item in batch:
        frames = item['frames']
        pad_len = max_frames - frames.shape[0]
        
        if pad_len > 0:
            frames = F.pad(frames, (0, 0, 0, pad_len))
        
        padded_frames.append(frames)
        
        # Create padding mask (True for valid positions)
        mask = torch.ones(max_frames, dtype=torch.bool)
        mask[-pad_len:] = False if pad_len > 0 else True
        padding_masks.append(mask)
    
    return {
        'frames': torch.stack(padded_frames),
        'padding_mask': torch.stack(padding_masks),
        'targets': torch.stack([item['targets'] for item in batch]),
        'input_symbols': [item['input_symbols'] for item in batch],
        'output_symbols': [item['output_symbols'] for item in batch],
        'sample_ids': [item['sample_id'] for item in batch],
    }


class TuringMachineModel(nn.Module):
    """Model for Turing machine tasks with hidden state extraction."""
    
    def __init__(self, config: MiniJMambaConfig, max_output_len: int = 4):
        super().__init__()
        self.backbone = MiniJMamba(config)
        self.max_output_len = max_output_len
        self.d_model = config.d_model
        
        # Attention pooling for sequence aggregation
        self.pool_query = nn.Parameter(torch.randn(1, 1, config.d_model))
        
        # Multi-step output head (for variable length outputs)
        self.output_heads = nn.ModuleList([
            nn.Linear(config.d_model, len(TM_SYMBOLS))
            for _ in range(max_output_len)
        ])
        
        self.hidden_states_cache = []
    
    def forward(
        self,
        frames: torch.Tensor,
        padding_mask: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional hidden state extraction.
        
        Returns:
            logits: (batch, max_output_len, vocab_size)
            hidden_states: (batch, seq_len, d_model) if return_hidden
        """
        # Get backbone output with hidden states
        # MiniJMamba returns: (frame_outputs, symbol_logits, hidden_x) when return_hidden=True
        frame_outputs, symbol_logits, hidden_x = self.backbone(
            frames, padding_mask, return_hidden=True
        )
        
        # hidden_x is (batch, seq_len, d_model)
        batch_size, seq_len, _ = hidden_x.shape
        
        # Attention pooling: use query to attend to sequence
        # Simplified: just use mean pooling with mask
        if padding_mask is not None:
            # Mask invalid positions
            mask_expanded = padding_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (hidden_x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_x.mean(dim=1)  # (batch, d_model)
        
        # Apply output heads
        outputs = []
        for head in self.output_heads:
            outputs.append(head(pooled))
        
        logits = torch.stack(outputs, dim=1)  # (batch, max_output_len, vocab_size)
        
        if return_hidden:
            self.hidden_states_cache = hidden_x
            return logits, hidden_x
        
        return logits, None


def train_epoch(
    model: TuringMachineModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        frames = batch['frames'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(frames, padding_mask)
        
        # Compute loss for each output position
        loss = 0
        for i in range(model.max_output_len):
            loss += F.cross_entropy(logits[:, i], targets[:, i])
        loss /= model.max_output_len
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * frames.shape[0]
        
        # Compute accuracy (all positions must be correct)
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).all(dim=1).sum().item()
        total_correct += correct
        total_samples += frames.shape[0]
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }


def evaluate(
    model: TuringMachineModel,
    dataloader: DataLoader,
    device: str,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            frames = batch['frames'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            targets = batch['targets'].to(device)
            
            logits, _ = model(frames, padding_mask)
            
            # Compute loss
            loss = 0
            for i in range(model.max_output_len):
                loss += F.cross_entropy(logits[:, i], targets[:, i])
            loss /= model.max_output_len
            
            total_loss += loss.item() * frames.shape[0]
            
            # Compute accuracy
            preds = logits.argmax(dim=-1)
            correct = (preds == targets).all(dim=1).sum().item()
            total_correct += correct
            total_samples += frames.shape[0]
            
            if return_predictions:
                for i in range(preds.shape[0]):
                    pred_symbols = [TM_IDX2SYMBOL[p.item()] for p in preds[i]]
                    predictions.append({
                        'sample_id': batch['sample_ids'][i],
                        'input': batch['input_symbols'][i],
                        'target': batch['output_symbols'][i],
                        'prediction': pred_symbols,
                        'correct': correct,
                    })
    
    result = {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'em': total_correct / total_samples,  # Exact match = accuracy for this task
    }
    
    if return_predictions:
        result['predictions'] = predictions
    
    return result


def extract_hidden_states(
    model: TuringMachineModel,
    dataloader: DataLoader,
    device: str,
    num_samples: int = 10,
) -> List[Dict[str, Any]]:
    """Extract hidden states for visualization."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(results) >= num_samples:
                break
            
            frames = batch['frames'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            logits, hidden_states = model(frames, padding_mask, return_hidden=True)
            
            # hidden_states is (batch, seq_len, d_model)
            for i in range(min(frames.shape[0], num_samples - len(results))):
                # Get full sequence hidden states for this sample
                seq_len = padding_mask[i].sum().item()  # Actual sequence length
                sample_hidden = hidden_states[i, :int(seq_len)].cpu().numpy()  # (seq_len, d_model)
                
                results.append({
                    'sample_id': batch['sample_ids'][i],
                    'input_symbols': batch['input_symbols'][i],
                    'output_symbols': batch['output_symbols'][i],
                    'hidden_states': sample_hidden,  # Now (seq_len, d_model)
                    'prediction': [TM_IDX2SYMBOL[p.item()] for p in logits[i].argmax(dim=-1)],
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Turing Machine Model")
    parser.add_argument("--level", type=int, required=True, help="Task level (1-5)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-ssm-layers", type=int, default=4)
    parser.add_argument("--num-attn-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/turing_machine"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests/turing_machine"))
    parser.add_argument("--save-hidden", action="store_true", help="Save hidden states for visualization")
    parser.add_argument("--max-output-len", type=int, default=2, help="Maximum output length")
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"level{args.level}_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Training Turing Machine Level {args.level}")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output: {run_dir}")
    
    # Load datasets
    train_dataset = TuringMachineDataset(
        args.manifest_dir / f"level{args.level}_train.jsonl",
        max_output_len=args.max_output_len
    )
    val_dataset = TuringMachineDataset(
        args.manifest_dir / f"level{args.level}_val.jsonl",
        max_output_len=args.max_output_len
    )
    iid_test_dataset = TuringMachineDataset(
        args.manifest_dir / f"level{args.level}_iid_test.jsonl",
        max_output_len=args.max_output_len
    )
    ood_test_dataset = TuringMachineDataset(
        args.manifest_dir / f"level{args.level}_ood_test.jsonl",
        max_output_len=args.max_output_len
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    iid_test_loader = DataLoader(
        iid_test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    ood_test_loader = DataLoader(
        ood_test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"IID Test: {len(iid_test_dataset)}, OOD Test: {len(ood_test_dataset)}")
    
    # Create model
    config = MiniJMambaConfig(
        frame_size=160,
        hop_size=80,
        d_model=args.d_model,
        num_ssm_layers=args.num_ssm_layers,
        num_attn_layers=args.num_attn_layers,
        symbol_vocab_size=len(TM_SYMBOLS),
    )
    model = TuringMachineModel(config, max_output_len=args.max_output_len).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0
    history = []
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, args.device)
        val_metrics = evaluate(model, val_loader, args.device)
        
        scheduler.step(val_metrics['accuracy'])
        
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
        })
        
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.2%} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.2%}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_acc': val_metrics['accuracy'],
            }, run_dir / "best.pt")
        
        # Early stopping if perfect
        if val_metrics['accuracy'] >= 0.99:
            print(f"Early stopping at epoch {epoch+1} (val acc >= 99%)")
            break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(run_dir / "best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    iid_metrics = evaluate(model, iid_test_loader, args.device, return_predictions=True)
    ood_metrics = evaluate(model, ood_test_loader, args.device, return_predictions=True)
    
    print(f"IID Test: Acc = {iid_metrics['accuracy']:.2%}, EM = {iid_metrics['em']:.2%}")
    print(f"OOD Test: Acc = {ood_metrics['accuracy']:.2%}, EM = {ood_metrics['em']:.2%}")
    
    # Save results
    results = {
        'level': args.level,
        'seed': args.seed,
        'best_val_acc': best_val_acc,
        'iid_test_acc': iid_metrics['accuracy'],
        'iid_test_em': iid_metrics['em'],
        'ood_test_acc': ood_metrics['accuracy'],
        'ood_test_em': ood_metrics['em'],
        'config': {
            'd_model': args.d_model,
            'num_ssm_layers': args.num_ssm_layers,
            'num_attn_layers': args.num_attn_layers,
        },
        'history': history,
    }
    
    with open(run_dir / "results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save hidden states for visualization
    if args.save_hidden:
        print("\nExtracting hidden states for visualization...")
        hidden_data = extract_hidden_states(model, iid_test_loader, args.device, num_samples=20)
        
        # Save as numpy arrays
        import pickle
        with open(run_dir / "hidden_states.pkl", 'wb') as f:
            pickle.dump(hidden_data, f)
        
        print(f"Saved hidden states for {len(hidden_data)} samples")
    
    print(f"\nResults saved to {run_dir}")
    
    return results


if __name__ == "__main__":
    main()

