#!/usr/bin/env python3
"""Debug Turing Machine data and encoding."""

import json
import sys
import io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_turing_machine import (
    TM_SYMBOLS, TM_SYMBOL2IDX, encode_tm_symbols_to_wave, TuringMachineDataset, collate_fn
)
from torch.utils.data import DataLoader
import torch
import numpy as np

def main():
    # Check sample data
    manifest = Path('manifests/turing_machine_v2/level1_train.jsonl')
    print("=" * 60)
    print("Checking Level 1 Data")
    print("=" * 60)
    
    with open(manifest, 'r', encoding='utf-8') as f:
        samples = [json.loads(l) for l in f.readlines()[:5]]
    
    print("\nSample data:")
    for i, s in enumerate(samples):
        inp = s['input_symbols']
        out = s['output_symbols']
        print(f"  {i}: Input: {inp} (len={len(inp)}) -> Output: {out}")
    
    # Check symbol vocabulary
    print(f"\nSymbol vocabulary: {TM_SYMBOLS}")
    print(f"Vocab size: {len(TM_SYMBOLS)}")
    
    # Check waveform encoding
    print("\nWaveform encoding test:")
    test_symbols = ['A', 'A', 'A']
    wave = encode_tm_symbols_to_wave(test_symbols)
    print(f"  Input: {test_symbols}")
    print(f"  Wave shape: {wave.shape}")
    print(f"  Wave stats: min={wave.min():.3f}, max={wave.max():.3f}, std={wave.std():.3f}")
    
    # Check dataset
    print("\nDataset test:")
    dataset = TuringMachineDataset(manifest, max_output_len=1)
    print(f"  Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"  Sample frames shape: {sample['frames'].shape}")
    print(f"  Sample targets: {sample['targets']}")
    print(f"  Sample targets shape: {sample['targets'].shape}")
    
    # Check dataloader
    print("\nDataLoader test:")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"  Batch frames shape: {batch['frames'].shape}")
    print(f"  Batch targets shape: {batch['targets'].shape}")
    print(f"  Batch targets: {batch['targets']}")
    
    # Check target distribution
    print("\nTarget distribution:")
    all_targets = []
    for sample in dataset:
        all_targets.extend(sample['targets'].tolist())
    
    from collections import Counter
    dist = Counter(all_targets)
    for idx, count in sorted(dist.items()):
        symbol = TM_SYMBOLS[idx] if idx < len(TM_SYMBOLS) else '?'
        print(f"  {idx} ({symbol}): {count}")


if __name__ == "__main__":
    main()

