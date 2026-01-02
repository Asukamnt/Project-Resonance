#!/usr/bin/env python3
"""
P1: 训练基线模型并比较

在 Task3 Mod 数据上训练 Transformer, LSTM, S4, Hyena 基线，
与 Mini-JMamba 进行公平比较。
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jericho.models.baselines import (
    TransformerBaseline,
    LSTMBaseline,
    S4Baseline,
    HyenaBaseline,
    BaselineConfig,
    count_parameters,
)
from src.jericho.symbols import SYMBOL2FREQ, SR, TONE_DUR


# Symbol vocabulary (0-9 for remainders)
OUTPUT_SYMBOLS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
SYMBOL2IDX = {s: i for i, s in enumerate(OUTPUT_SYMBOLS)}


def encode_symbol_to_wave(symbol: str, duration: float = TONE_DUR, sample_rate: int = SR) -> np.ndarray:
    """Encode a symbol to a waveform using frequency mapping."""
    if symbol in SYMBOL2FREQ:
        freq = SYMBOL2FREQ[symbol]
    else:
        # Use a default frequency for unknown symbols
        freq = 440  # A4
    
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return wave


class Task3ModDataset(Dataset):
    """Simple dataset for Task3 Mod.
    
    Supports two manifest formats:
    1. {"symbols": ["1", "9", "%", "5"], ...}  - symbol list format
    2. {"expression": "1+2%5", "result": 3}   - expression format
    """
    
    def __init__(self, manifest_path: str, frame_size: int = 160, max_samples: int = None, split: str = None):
        self.frame_size = frame_size
        self.samples = []
        
        with open(manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                # Filter by split if specified
                if split is not None and entry.get("split") != split:
                    continue
                self.samples.append(entry)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        entry = self.samples[idx]
        
        # Handle different manifest formats
        if "symbols" in entry:
            # Format: {"symbols": ["1", "9", "2", "9", "%", "5"], ...}
            symbols = entry["symbols"]
            # Parse: digits before % are the number, after % is modulus
            if "%" in symbols:
                mod_idx = symbols.index("%")
                num_str = "".join(symbols[:mod_idx])
                mod_str = "".join(symbols[mod_idx + 1:])
                num = int(num_str) if num_str else 0
                mod = int(mod_str) if mod_str else 1
                result = num % mod
            else:
                result = 0
            input_symbols = symbols
        else:
            # Format: {"expression": "1+2%5", "result": 3}
            expr = entry.get("expression", "")
            result = entry.get("result", 0)
            input_symbols = [c for c in expr if c.isdigit() or c in ['+', '%']]
        
        # Encode input
        if len(input_symbols) == 0:
            input_symbols = ['0']  # Fallback
        input_wave = np.concatenate([encode_symbol_to_wave(s) for s in input_symbols])
        
        # Frame the input
        num_frames = len(input_wave) // self.frame_size
        if num_frames == 0:
            num_frames = 1
            input_wave = np.pad(input_wave, (0, self.frame_size - len(input_wave)))
        
        input_frames = input_wave[:num_frames * self.frame_size].reshape(num_frames, self.frame_size)
        
        # Target: the result (single digit, 0-9)
        target_idx = result % 10  # Ensure 0-9
        
        return {
            "input_frames": torch.from_numpy(input_frames),
            "target_idx": target_idx,
        }


def collate_fn(batch):
    """Collate function for Task3ModDataset."""
    input_frames = pad_sequence([b["input_frames"] for b in batch], batch_first=True)
    input_mask = pad_sequence(
        [torch.ones(b["input_frames"].size(0), dtype=torch.bool) for b in batch], 
        batch_first=True
    )
    target_idx = torch.tensor([b["target_idx"] for b in batch], dtype=torch.long)
    
    return {
        "input_frames": input_frames,
        "input_mask": input_mask,
        "target_idx": target_idx,
    }


def create_baseline(name: str, frame_size: int = 160, symbol_vocab_size: int = 10) -> nn.Module:
    """Create baseline model by name."""
    # Adjust layers for ~1M params each
    model_configs = {
        "transformer": (TransformerBaseline, 6),
        "lstm": (LSTMBaseline, 6),
        "s4": (S4Baseline, 12),
        "hyena": (HyenaBaseline, 8),
    }
    
    if name not in model_configs:
        raise ValueError(f"Unknown model: {name}. Available: {list(model_configs.keys())}")
    
    model_class, num_layers = model_configs[name]
    
    config = BaselineConfig(
        frame_size=frame_size,
        hop_size=frame_size,
        symbol_vocab_size=symbol_vocab_size,
        d_model=128,
        num_layers=num_layers,
        num_heads=4,
        max_frames=256,
        dropout=0.1,
    )
    
    return model_class(config)


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        input_frames = batch["input_frames"].to(device)
        input_mask = batch["input_mask"].to(device)
        target_idx = batch["target_idx"].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        _, symbol_logits = model(input_frames, input_mask)
        
        # Use mean pooled logits for classification
        B, T, C = symbol_logits.shape
        valid_counts = input_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_logits = (symbol_logits * input_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        
        loss = F.cross_entropy(pooled_logits, target_idx)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        preds = pooled_logits.argmax(dim=-1)
        total_correct += (preds == target_idx).sum().item()
        total_samples += len(target_idx)
    
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, accuracy


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            input_frames = batch["input_frames"].to(device)
            input_mask = batch["input_mask"].to(device)
            target_idx = batch["target_idx"].to(device)
            
            _, symbol_logits = model(input_frames, input_mask)
            
            B, T, C = symbol_logits.shape
            valid_counts = input_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled_logits = (symbol_logits * input_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
            
            preds = pooled_logits.argmax(dim=-1)
            total_correct += (preds == target_idx).sum().item()
            total_samples += len(target_idx)
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["transformer", "lstm", "s4", "hyena", "all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-manifest", type=str, default="manifests/task3_disjoint_600.jsonl")
    parser.add_argument("--val-manifest", type=str, default="manifests/task3_disjoint_600.jsonl")
    parser.add_argument("--train-split", type=str, default="train", help="Split to use for training")
    parser.add_argument("--val-split", type=str, default="val", help="Split to use for validation")
    parser.add_argument("--output-dir", type=str, default="runs/baselines")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Determine which models to train
    models_to_train = [args.model] if args.model != "all" else ["transformer", "lstm", "s4", "hyena"]
    
    results = []
    
    for model_name in models_to_train:
        print("\n" + "=" * 60)
        print(f"Training: {model_name}")
        print("=" * 60)
        
        # Create model
        model = create_baseline(model_name)
        model = model.to(device)
        n_params = count_parameters(model)
        print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        
        # Load data
        train_dataset = Task3ModDataset(args.train_manifest, split=args.train_split)
        val_dataset = Task3ModDataset(args.val_manifest, split=args.val_split)
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               collate_fn=collate_fn, num_workers=0)
        
        # Training
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_val_acc = 0
        run_dir = Path(args.output_dir) / f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            scheduler.step()
            
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), run_dir / "best.pt")
        
        # Save results
        result = {
            "model": model_name,
            "params": n_params,
            "best_val_acc": best_val_acc,
            "epochs": args.epochs,
        }
        results.append(result)
        
        print(f"\nBest Val Acc: {best_val_acc*100:.1f}%")
        print(f"Model saved to: {run_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<15} {'Params':>10} {'Val Acc':>10}")
    print("-" * 37)
    for r in results:
        print(f"{r['model']:<15} {r['params']/1e6:.2f}M{' ':>4} {r['best_val_acc']*100:>9.1f}%")
    
    # Save summary
    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
