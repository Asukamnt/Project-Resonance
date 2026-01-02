#!/usr/bin/env python3
"""
P1: 现代基线比较实验

比较 Mini-JMamba 与同参数量级的现代架构：
- Transformer (6 layers)
- LSTM (4 layers bidirectional)
- S4 (8 layers, simplified diagonal)
- Hyena (6 layers, implicit long conv)

所有模型约 1M 参数，公平比较。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig
from src.jericho.models.baselines import (
    TransformerBaseline,
    LSTMBaseline,
    S4Baseline,
    HyenaBaseline,
    BaselineConfig,
    count_parameters,
    create_comparable_configs,
)


def create_mini_jmamba_config(frame_size: int = 160, symbol_vocab_size: int = 12):
    """Create Mini-JMamba config for comparison."""
    return MiniJMambaConfig(
        frame_size=frame_size,
        hop_size=frame_size,
        symbol_vocab_size=symbol_vocab_size,
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
        num_heads=4,
        max_frames=256,
        dropout=0.1,
    )


def benchmark_model(model: nn.Module, name: str, device: torch.device, batch_size: int = 32, seq_len: int = 100):
    """Benchmark model inference speed and parameter count."""
    model = model.to(device)
    model.eval()
    
    # Count parameters
    n_params = count_parameters(model)
    
    # Create dummy input
    frames = torch.randn(batch_size, seq_len, 160, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(frames, mask)
    
    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    n_runs = 10
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(frames, mask)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time = elapsed / n_runs * 1000  # ms
    
    return {
        "name": name,
        "params": n_params,
        "params_str": f"{n_params / 1e6:.2f}M",
        "inference_time_ms": avg_time,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline model comparison")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--output", type=str, default="reports/baseline_comparison.json")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    print("=" * 60)
    
    # Create all models
    frame_size = 160
    symbol_vocab_size = 12
    
    models = {}
    
    # Mini-JMamba
    jmamba_config = create_mini_jmamba_config(frame_size, symbol_vocab_size)
    models["mini_jmamba"] = MiniJMamba(jmamba_config)
    
    # Baselines
    baselines = create_comparable_configs(frame_size, symbol_vocab_size)
    for name, (model_class, config) in baselines.items():
        models[name] = model_class(config)
    
    # Benchmark all models
    results = []
    print(f"\n{'Model':<15} {'Params':>10} {'Inference (ms)':>15}")
    print("-" * 42)
    
    for name, model in models.items():
        try:
            result = benchmark_model(
                model, name, device,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )
            results.append(result)
            print(f"{name:<15} {result['params_str']:>10} {result['inference_time_ms']:>15.2f}")
        except Exception as e:
            print(f"{name:<15} ERROR: {e}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_path}")
    
    # Summary
    print("\n[Summary]")
    jmamba_time = next((r["inference_time_ms"] for r in results if r["name"] == "mini_jmamba"), None)
    if jmamba_time:
        for r in results:
            if r["name"] != "mini_jmamba":
                speedup = r["inference_time_ms"] / jmamba_time
                print(f"  Mini-JMamba vs {r['name']}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")


if __name__ == "__main__":
    main()

