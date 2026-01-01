#!/usr/bin/env python3
"""
TSAE Task2 (Bracket) Verification

验证 TSAE 是否在 Task2 Bracket 任务中也存在。

用法:
    python scripts/tsae_task2_verify.py --device cuda
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from jericho.data import read_manifest, synthesise_entry_wave
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.scorer import decode_wave_to_symbols
from jericho.task2 import is_balanced


def time_stretch(wave: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return wave
    original_len = len(wave)
    new_len = int(original_len * factor)
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, new_len)
    stretched = np.interp(x_new, x_old, wave).astype(np.float32)
    if len(stretched) > original_len:
        return stretched[:original_len]
    return np.pad(stretched, (0, original_len - len(stretched)))


def evaluate_task2_tsae(entries: list, device: str, limit: int = 100) -> dict:
    """Evaluate TSAE for Task2 using oracle (no model, just decoder)"""
    
    results = {}
    
    for factor in [0.95, 1.00, 1.05]:
        correct = 0
        total = 0
        
        for entry in entries[:limit]:
            # Get target
            balanced = is_balanced(entry.symbols)
            target = ["V"] if balanced else ["X"]
            
            # Synthesize and stretch
            wave = synthesise_entry_wave(entry)
            wave = time_stretch(wave, factor)
            
            # Decode
            decoded = decode_wave_to_symbols(wave)
            
            # For Task2, we care about the last symbol (answer)
            if decoded and decoded[-1:] == target:
                correct += 1
            
            total += 1
        
        results[f"{factor:.2f}x"] = correct / total if total > 0 else 0
    
    delta = results.get("1.05x", 0) - results.get("0.95x", 0)
    results["delta_1.05_0.95"] = delta
    results["tsae_present"] = abs(delta) > 0.01
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/task2_full.jsonl"))
    parser.add_argument("--split", type=str, default="iid_test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("reports/tsae_task2.json"))
    args = parser.parse_args()
    
    # Try to find manifest
    if not args.manifest.exists():
        alt = Path("manifests/task2_tiny.jsonl")
        if alt.exists():
            args.manifest = alt
    
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        return
    
    entries = [e for e in read_manifest(args.manifest) if e.split == args.split]
    print(f"Manifest: {args.manifest}")
    print(f"Samples: {len(entries)}")
    
    print()
    print("=" * 60)
    print("TSAE TASK2 (BRACKET) VERIFICATION")
    print("=" * 60)
    print("Testing decoder-level TSAE (no model, pure FFT decode)")
    print()
    
    results = evaluate_task2_tsae(entries, "cpu", args.limit)
    
    print(f"{'Factor':<12} {'EM':<10}")
    print("-" * 22)
    for factor in ["0.95x", "1.00x", "1.05x"]:
        em = results.get(factor, 0)
        print(f"{factor:<12} {em:.1%}")
    
    delta = results.get("delta_1.05_0.95", 0)
    print("-" * 22)
    print(f"Delta (1.05x - 0.95x): {delta:+.1%}")
    print()
    
    if results.get("tsae_present", False):
        print("[INFO] TSAE-like asymmetry detected in Task2 decoder")
    else:
        print("[INFO] No significant TSAE in Task2 decoder")
    
    # Save
    output = {
        "experiment": "tsae_task2",
        "timestamp": datetime.now().isoformat(),
        "manifest": str(args.manifest),
        "results": results,
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

