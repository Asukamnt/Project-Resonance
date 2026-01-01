#!/usr/bin/env python3
"""Debug Turing Machine training loop."""

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
    TM_SYMBOLS, TuringMachineDataset, TuringMachineModel, collate_fn, train_epoch
)
from src.jericho.models.mini_jmamba import MiniJMambaConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load dataset
    manifest = Path('manifests/turing_machine_v2/level1_train.jsonl')
    dataset = TuringMachineDataset(manifest, max_output_len=1)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"Training for 10 epochs...")
    for epoch in range(10):
        metrics = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2%}")
    
    # Check predictions
    print("\nChecking predictions on a batch:")
    model.eval()
    batch = next(iter(loader))
    frames = batch['frames'].to(device)
    padding_mask = batch['padding_mask'].to(device)
    targets = batch['targets'].to(device)
    
    with torch.no_grad():
        logits, _ = model(frames, padding_mask)
    
    preds = logits[:, 0].argmax(dim=-1)
    print(f"Targets: {targets[:10, 0].tolist()}")
    print(f"Preds:   {preds[:10].tolist()}")
    print(f"Correct: {(preds[:10] == targets[:10, 0]).tolist()}")


if __name__ == "__main__":
    main()

