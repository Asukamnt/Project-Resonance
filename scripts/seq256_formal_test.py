#!/usr/bin/env python3
"""
SeqLen >= 256 正式压力测试

使用正式 manifest 数据测试超长序列下的模型表现。

Author: Jericho Team
Date: 2026-01-02
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig
from jericho.symbols import encode_symbols_to_wave, SYMBOL2FREQ, SR
from jericho.data.manifest import read_manifest


def load_model(checkpoint_path: str, device: str):
    """加载模型"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = ckpt["config"]
    state_dict = ckpt["model_state_dict"]
    
    config = MiniJMambaConfig(
        frame_size=saved_config.get("frame_size", 160),
        hop_size=saved_config.get("hop_size", 160),
        symbol_vocab_size=saved_config.get("symbol_vocab_size", 12),
        d_model=saved_config.get("d_model", 128),
        num_ssm_layers=saved_config.get("num_ssm_layers", 10),
        num_attn_layers=saved_config.get("num_attn_layers", 2),
        max_frames=512,  # 增加以支持更长序列
        use_rope=saved_config.get("use_rope", True),
    )
    
    model = MiniJMamba(config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, config


def generate_long_expression(num_symbols: int, seed: int) -> str:
    """生成指定长度的 Mod 表达式（仅使用 0-9 和 %）"""
    np.random.seed(seed)
    
    # 生成形如 "A1%B1A2%B2..." 的连续表达式
    # 只使用支持的符号：0-9 和 %
    symbols = []
    remaining = num_symbols
    
    while remaining > 0:
        # 生成一个子表达式 A%B（不用 +）
        a = np.random.randint(10, 99)  # 2位数
        b = np.random.randint(2, 9)    # 1位除数
        expr = f"{a}%{b}"
        
        if remaining >= len(expr):
            symbols.extend(list(expr))
            remaining -= len(expr)
        else:
            # 用数字填充剩余
            for _ in range(remaining):
                symbols.append(str(np.random.randint(0, 9)))
            remaining = 0
    
    return "".join(symbols)


def test_sequence_length(
    model: MiniJMamba,
    config: MiniJMambaConfig,
    seq_len: int,
    keep_ratio: float,
    num_samples: int,
    device: str,
) -> dict:
    """测试特定序列长度"""
    
    probs = []
    entropies = []
    norms = []
    
    for sample_idx in range(num_samples):
        # 生成表达式
        expr = generate_long_expression(seq_len, seed=42 + sample_idx)
        symbols = list(expr)
        
        # 编码为波形
        wave = encode_symbols_to_wave(symbols, tone_dur=0.01, sr=SR)
        
        # 转换为帧
        frame_size = config.frame_size
        num_frames = len(wave) // frame_size
        if num_frames == 0:
            continue
        wave = wave[:num_frames * frame_size]
        frames = wave.reshape(1, num_frames, frame_size)
        frames = torch.tensor(frames, dtype=torch.float32).to(device)
        padding_mask = torch.ones(1, num_frames, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            # 前向传播（带选择性修剪）
            x = model.input_proj(frames)
            x = model.dropout(x)
            
            for layer in model.layers:
                x = layer(x, padding_mask)
                
                if keep_ratio < 1.0:
                    k = int(x.shape[-1] * keep_ratio)
                    if k > 0:
                        sorted_vals, _ = torch.sort(x.abs(), dim=-1, descending=True)
                        threshold = sorted_vals[:, :, k-1:k]
                        mask = (x.abs() >= threshold).float()
                        x = x * mask
            
            # 记录隐状态范数
            norm = torch.norm(x, p=2).item()
            norms.append(norm)
            
            # 输出
            x_norm = model.final_norm(x)
            symbol_logits = model.symbol_head(x_norm)
            
            probs_sample = torch.softmax(symbol_logits, dim=-1)
            max_prob = probs_sample.max(dim=-1).values.mean().item()
            probs.append(max_prob)
            
            entropy = -(probs_sample * torch.log(probs_sample + 1e-10)).sum(dim=-1).mean().item()
            entropies.append(entropy)
    
    return {
        "seq_len": seq_len,
        "keep_ratio": keep_ratio,
        "mean_prob": np.mean(probs),
        "std_prob": np.std(probs),
        "mean_norm": np.mean(norms),
        "mean_entropy": np.mean(entropies),
        "n_samples": len(probs),
    }


def run_test(
    checkpoint_path: str,
    seq_lengths: list,
    keep_ratios: list,
    num_samples: int,
    device: str,
    output_dir: Path,
):
    """运行测试"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)
    
    results = []
    
    for seq_len in seq_lengths:
        for keep_ratio in keep_ratios:
            print(f"\nSeqLen={seq_len}, k={keep_ratio}")
            result = test_sequence_length(
                model, config, seq_len, keep_ratio, num_samples, device
            )
            results.append(result)
            print(f"  Prob: {result['mean_prob']:.4f} +/- {result['std_prob']:.4f}")
            print(f"  Norm: {result['mean_norm']:.1f}")
    
    # 保存结果
    output_file = output_dir / "seq256_formal_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": checkpoint_path,
            "seq_lengths": seq_lengths,
            "keep_ratios": keep_ratios,
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # 生成图
    plot_results(results, seq_lengths, keep_ratios, output_dir)
    
    return results


def plot_results(results: list, seq_lengths: list, keep_ratios: list, output_dir: Path):
    """生成图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {1.0: '#2ecc71', 0.7: '#3498db', 0.5: '#e74c3c'}
    
    # (a) Confidence
    ax = axes[0]
    for k in keep_ratios:
        k_results = [r for r in results if r["keep_ratio"] == k]
        x = [r["seq_len"] for r in k_results]
        y = [r["mean_prob"] for r in k_results]
        yerr = [r["std_prob"] for r in k_results]
        ax.errorbar(x, y, yerr=yerr, fmt='o-', color=colors.get(k, 'gray'),
                   label=f'k={k}', capsize=3)
    
    ax.set_xlabel('Sequence Length (symbols)')
    ax.set_ylabel('Classification Confidence')
    ax.set_title('(a) Confidence vs Sequence Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Delta vs baseline
    ax = axes[1]
    baseline_by_len = {r["seq_len"]: r["mean_prob"] 
                       for r in results if r["keep_ratio"] == 1.0}
    
    for k in [k for k in keep_ratios if k < 1.0]:
        k_results = [r for r in results if r["keep_ratio"] == k]
        x = [r["seq_len"] for r in k_results]
        y = [(r["mean_prob"] - baseline_by_len.get(r["seq_len"], 0)) * 100 
             for r in k_results]
        ax.plot(x, y, 'o-', color=colors.get(k, 'gray'), label=f'k={k}', linewidth=2)
        
        for xi, yi in zip(x, y):
            ax.annotate(f'{yi:+.1f}pp', xy=(xi, yi), xytext=(5, 5),
                       textcoords='offset points', fontsize=9, color=colors.get(k, 'gray'))
    
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.axvline(x=64, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Sequence Length (symbols)')
    ax.set_ylabel('Delta Confidence vs Baseline (pp)')
    ax.set_title('(b) Pruning Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "seq256_formal_plot.png"
    fig.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="SeqLen >= 256 Formal Test")
    parser.add_argument("--checkpoint", type=str,
                       default="artifacts/checkpoints/mod_best_em0.75.pt")
    parser.add_argument("--seq-lengths", type=str, default="32,64,128,192,256")
    parser.add_argument("--keep-ratios", type=str, default="1.0,0.7,0.5")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="reports/seq256_formal")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
    
    run_test(
        checkpoint_path=args.checkpoint,
        seq_lengths=seq_lengths,
        keep_ratios=keep_ratios,
        num_samples=args.num_samples,
        device=device,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

