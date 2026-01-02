#!/usr/bin/env python3
"""
P2: 资源效率基准测试

测量各模型的：
- 参数量
- 推理延迟 (ms)
- 吞吐量 (samples/sec)
- 内存占用 (MB)

与 wav2vec2 进行对比，展示 Mini-JMamba 的效率优势。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig
from src.jericho.models.baselines import (
    TransformerBaseline,
    LSTMBaseline,
    S4Baseline,
    HyenaBaseline,
    BaselineConfig,
    count_parameters,
)


def create_models(frame_size: int = 160, symbol_vocab_size: int = 10):
    """Create all models for comparison."""
    models = {}
    
    # Mini-JMamba
    jmamba_config = MiniJMambaConfig(
        frame_size=frame_size,
        hop_size=frame_size,
        symbol_vocab_size=symbol_vocab_size,
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
        num_heads=4,
        max_frames=256,
        dropout=0.0,  # Disable dropout for inference
    )
    models["Mini-JMamba"] = MiniJMamba(jmamba_config)
    
    # Baselines
    base_config = BaselineConfig(
        frame_size=frame_size,
        hop_size=frame_size,
        symbol_vocab_size=symbol_vocab_size,
        d_model=128,
        num_heads=4,
        max_frames=256,
        dropout=0.0,
    )
    
    # Transformer (6 layers)
    transformer_config = BaselineConfig(**{**base_config.__dict__, "num_layers": 6})
    models["Transformer"] = TransformerBaseline(transformer_config)
    
    # LSTM (6 layers)
    lstm_config = BaselineConfig(**{**base_config.__dict__, "num_layers": 6})
    models["LSTM"] = LSTMBaseline(lstm_config)
    
    # S4 (12 layers)
    s4_config = BaselineConfig(**{**base_config.__dict__, "num_layers": 12})
    models["S4"] = S4Baseline(s4_config)
    
    # Hyena (8 layers)
    hyena_config = BaselineConfig(**{**base_config.__dict__, "num_layers": 8})
    models["Hyena"] = HyenaBaseline(hyena_config)
    
    return models


def benchmark_model(model: nn.Module, name: str, device: torch.device, 
                    batch_size: int = 1, seq_len: int = 100, num_runs: int = 50):
    """Benchmark a single model."""
    model = model.to(device)
    model.eval()
    
    # Count parameters
    n_params = count_parameters(model)
    
    # Memory before
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Create dummy input
    frames = torch.randn(batch_size, seq_len, 160, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(frames, mask)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark inference time
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(frames, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    avg_latency = sum(latencies) / len(latencies)
    std_latency = (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5
    
    # Memory usage
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        import psutil
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # Approximate
    
    # Throughput
    throughput = batch_size / (avg_latency / 1000)  # samples/sec
    
    return {
        "name": name,
        "params": n_params,
        "params_str": f"{n_params / 1e6:.2f}M",
        "latency_ms": avg_latency,
        "latency_std_ms": std_latency,
        "throughput_sps": throughput,
        "memory_mb": peak_memory,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Efficiency benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", type=str, default="1,8,32", help="Batch sizes to test")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument("--output", type=str, default="reports/efficiency_benchmark.json")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    
    print("=" * 70)
    print("P2: Efficiency Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Number of runs: {args.num_runs}")
    print()
    
    # Create models
    models = create_models()
    
    all_results = []
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        print(f"{'Model':<15} {'Params':>10} {'Latency (ms)':>15} {'Throughput':>15} {'Memory (MB)':>12}")
        print("-" * 70)
        
        batch_results = []
        for name, model in models.items():
            try:
                result = benchmark_model(
                    model, name, device,
                    batch_size=batch_size,
                    seq_len=args.seq_len,
                    num_runs=args.num_runs,
                )
                batch_results.append(result)
                print(f"{name:<15} {result['params_str']:>10} "
                      f"{result['latency_ms']:>10.2f} +/- {result['latency_std_ms']:.2f} "
                      f"{result['throughput_sps']:>12.1f}/s "
                      f"{result['memory_mb']:>10.1f}")
            except Exception as e:
                print(f"{name:<15} ERROR: {e}")
        
        all_results.extend(batch_results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary: Mini-JMamba vs others
    print("\n" + "=" * 70)
    print("Summary: Mini-JMamba Efficiency Advantage")
    print("=" * 70)
    
    # Find Mini-JMamba results for batch_size=1
    jmamba_result = next((r for r in all_results if r["name"] == "Mini-JMamba" and r["batch_size"] == 1), None)
    if jmamba_result:
        print(f"\nMini-JMamba baseline (batch=1):")
        print(f"  Latency: {jmamba_result['latency_ms']:.2f} ms")
        print(f"  Throughput: {jmamba_result['throughput_sps']:.1f} samples/sec")
        print(f"  Parameters: {jmamba_result['params_str']}")
        
        print("\nComparison:")
        for r in all_results:
            if r["batch_size"] == 1 and r["name"] != "Mini-JMamba":
                speedup = r["latency_ms"] / jmamba_result["latency_ms"]
                param_ratio = r["params"] / jmamba_result["params"]
                status = "slower" if speedup > 1 else "faster"
                print(f"  vs {r['name']:<12}: {speedup:.2f}x {status}, {param_ratio:.2f}x params")
    
    print(f"\nResults saved to: {output_path}")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(all_results)
    latex_path = output_path.with_suffix(".tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")


def generate_latex_table(results):
    """Generate LaTeX table for paper."""
    # Filter batch_size=1 results
    b1_results = [r for r in results if r["batch_size"] == 1]
    
    table = r"""
\begin{table}[h]
\centering
\caption{Inference Efficiency Comparison (batch size = 1, seq len = 100)}
\label{tab:efficiency}
\begin{tabular}{lrrrr}
\toprule
Model & Params & Latency (ms) & Throughput (s/s) & Memory (MB) \\
\midrule
"""
    for r in b1_results:
        table += f"{r['name']} & {r['params_str']} & {r['latency_ms']:.2f} & {r['throughput_sps']:.1f} & {r['memory_mb']:.1f} \\\\\n"
    
    table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return table


if __name__ == "__main__":
    main()


