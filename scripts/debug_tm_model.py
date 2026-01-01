#!/usr/bin/env python3
"""Debug Turing Machine model predictions."""

import json
import sys
import io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.train_turing_machine import (
    TM_SYMBOLS, TuringMachineDataset, TuringMachineModel, collate_fn
)
from src.jericho.models.mini_jmamba import MiniJMambaConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load dataset
    manifest = Path('manifests/turing_machine_v2/level1_train.jsonl')
    dataset = TuringMachineDataset(manifest, max_output_len=1)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=0)
    
    # Create model
    config = MiniJMambaConfig(
        frame_size=160,
        hop_size=80,
        d_model=128,
        num_ssm_layers=4,
        num_attn_layers=2,
        symbol_vocab_size=len(TM_SYMBOLS),
    )
    model = TuringMachineModel(config, max_output_len=1).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get a batch
    batch = next(iter(loader))
    frames = batch['frames'].to(device)
    padding_mask = batch['padding_mask'].to(device)
    targets = batch['targets'].to(device)
    
    print(f"\nBatch info:")
    print(f"  Frames: {frames.shape}")
    print(f"  Padding mask: {padding_mask.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Target values: {targets.squeeze()}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, hidden = model(frames, padding_mask, return_hidden=True)
    
    print(f"\nModel output:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Hidden shape: {hidden.shape}")
    
    # Check logits distribution
    probs = F.softmax(logits.squeeze(1), dim=-1)
    print(f"\nPrediction probabilities (first 3 samples):")
    for i in range(3):
        pred = logits[i, 0].argmax().item()
        target = targets[i, 0].item()
        print(f"  Sample {i}: target={target} ({TM_SYMBOLS[target]}), pred={pred} ({TM_SYMBOLS[pred]})")
        print(f"    Probs: {probs[i, :10].tolist()}")  # First 10 symbols
    
    # Check loss
    loss = F.cross_entropy(logits[:, 0], targets[:, 0])
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Expected loss (uniform): {torch.log(torch.tensor(len(TM_SYMBOLS), dtype=torch.float)).item():.4f}")
    
    # Train for a few steps
    print("\n" + "=" * 60)
    print("Quick training test (10 steps)")
    print("=" * 60)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for step in range(10):
        batch = next(iter(loader))
        frames = batch['frames'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(frames, padding_mask)
        loss = F.cross_entropy(logits[:, 0], targets[:, 0])
        loss.backward()
        optimizer.step()
        
        preds = logits[:, 0].argmax(dim=-1)
        acc = (preds == targets[:, 0]).float().mean().item()
        
        print(f"  Step {step+1}: Loss={loss.item():.4f}, Acc={acc:.2%}")


if __name__ == "__main__":
    main()

