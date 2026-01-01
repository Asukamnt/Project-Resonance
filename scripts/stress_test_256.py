#!/usr/bin/env python3
"""
SeqLen=256 压力测试

测试超长序列（256符号）下的模型表现和修剪效果。

目的：
1. 验证状态累积在超长序列下是否成为更严重的瓶颈
2. 验证 k=0.7 修剪在更长序列下效果是否更明显
3. 为第二篇论文收集数据

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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig
from jericho.symbols import encode_symbols_to_wave, SYMBOL2FREQ, SR


class SelectivePruningModel(nn.Module):
    """带选择性修剪的模型包装器"""
    
    def __init__(self, model: MiniJMamba, keep_ratio: float = 1.0):
        super().__init__()
        self.model = model
        self.keep_ratio = keep_ratio
    
    def forward(self, frames: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        # 获取模型配置 - 使用 input_proj 而非 frame_encoder
        x = self.model.input_proj(frames)
        x = self.model.dropout(x)
        
        for layer in self.model.layers:
            x = layer(x, padding_mask)
            
            # 应用选择性修剪
            if self.keep_ratio < 1.0:
                # 计算每个通道的 L2 范数
                channel_norms = torch.norm(x, p=2, dim=-1, keepdim=True)  # (batch, seq_len, 1)
                
                # 确定阈值（按通道维度）
                k = int(x.shape[-1] * self.keep_ratio)
                if k > 0:
                    # 对每个位置独立计算阈值
                    sorted_norms, _ = torch.sort(x.abs(), dim=-1, descending=True)
                    threshold = sorted_norms[:, :, k-1:k]  # (batch, seq_len, 1)
                    mask = (x.abs() >= threshold).float()
                    x = x * mask
        
        # 输出层
        x = self.model.final_norm(x)
        output = self.model.frame_head(x)
        symbol_logits = self.model.symbol_head(x)
        
        return output, symbol_logits


def generate_long_sequence(num_symbols: int, device: str) -> tuple:
    """生成指定长度的测试序列"""
    
    # 生成随机符号序列（数字 + 运算符）
    digits = list("0123456789")
    operators = ["%"]
    
    symbols = []
    for i in range(num_symbols):
        if i % 3 == 2:  # 每三个符号一个运算符
            symbols.append(np.random.choice(operators))
        else:
            symbols.append(np.random.choice(digits))
    
    # 编码为波形
    wave = encode_symbols_to_wave(symbols, tone_dur=0.01, sr=SR)
    
    # 转换为帧
    frame_size = 160  # 10ms at 16kHz
    num_frames = len(wave) // frame_size
    wave = wave[:num_frames * frame_size]
    frames = wave.reshape(num_frames, frame_size)
    frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    
    return frames, symbols


def run_stress_test(
    checkpoint_path: str,
    sequence_lengths: list,
    keep_ratios: list,
    num_samples: int,
    seeds: list,
    device: str,
    output_dir: Path
):
    """运行压力测试"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 从 checkpoint 提取配置和状态
    if "config" in ckpt:
        # 新格式 checkpoint - 直接使用保存的配置
        saved_config = ckpt["config"]
        state_dict = ckpt["model_state_dict"]
        config = MiniJMambaConfig(
            frame_size=saved_config.get("frame_size", 160),
            hop_size=saved_config.get("hop_size", 160),
            symbol_vocab_size=saved_config.get("symbol_vocab_size", 12),  # 使用保存的大小
            d_model=saved_config.get("d_model", 128),
            num_ssm_layers=saved_config.get("num_ssm_layers", 10),
            num_attn_layers=saved_config.get("num_attn_layers", 2),
            max_frames=saved_config.get("max_frames", 256),
            use_rope=saved_config.get("use_rope", True),
        )
    else:
        # 旧格式 checkpoint
        state_dict = ckpt.get("model", ckpt)
        d_model = state_dict["frame_encoder.0.weight"].shape[0]
        symbol_vocab_size = state_dict["symbol_head.weight"].shape[0]
        config = MiniJMambaConfig(
            frame_size=160,
            hop_size=160,
            symbol_vocab_size=symbol_vocab_size,
            d_model=d_model,
            num_ssm_layers=10,
            num_attn_layers=2,
        )
    
    results = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        for seq_len in sequence_lengths:
            for keep_ratio in keep_ratios:
                print(f"\nSeed={seed}, SeqLen={seq_len}, k={keep_ratio}")
                
                # 创建模型
                base_model = MiniJMamba(config).to(device)
                base_model.load_state_dict(state_dict, strict=False)
                base_model.eval()
                
                model = SelectivePruningModel(base_model, keep_ratio).to(device)
                
                probs = []
                norms = []
                entropies = []
                
                for _ in range(num_samples):
                    frames, _ = generate_long_sequence(seq_len, device)
                    padding_mask = torch.zeros(1, frames.shape[1], dtype=torch.bool, device=device)
                    
                    with torch.no_grad():
                        _, symbol_logits = model(frames, padding_mask)
                        
                        # 计算分类置信度
                        probs_sample = torch.softmax(symbol_logits, dim=-1)
                        max_prob = probs_sample.max(dim=-1).values.mean().item()
                        probs.append(max_prob)
                        
                        # 计算熵
                        entropy = -(probs_sample * torch.log(probs_sample + 1e-10)).sum(dim=-1).mean().item()
                        entropies.append(entropy)
                        
                        # 计算隐状态范数（通过模型中间层）
                        x = base_model.input_proj(frames)
                        x = base_model.dropout(x)
                        for layer in base_model.layers:
                            x = layer(x, padding_mask)
                        norm = torch.norm(x, p=2).item()
                        norms.append(norm)
                
                result = {
                    "seed": seed,
                    "seq_len": seq_len,
                    "keep_ratio": keep_ratio,
                    "mean_prob": np.mean(probs),
                    "std_prob": np.std(probs),
                    "mean_norm": np.mean(norms),
                    "mean_entropy": np.mean(entropies),
                    "n_samples": num_samples,
                }
                results.append(result)
                
                print(f"  Prob: {result['mean_prob']:.4f} +/- {result['std_prob']:.4f}")
                print(f"  Norm: {result['mean_norm']:.1f}")
                print(f"  Entropy: {result['mean_entropy']:.4f}")
    
    # 保存结果
    output_file = output_dir / "stress_test_256_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": str(np.datetime64("now")),
            "checkpoint": checkpoint_path,
            "sequence_lengths": sequence_lengths,
            "keep_ratios": keep_ratios,
            "seeds": seeds,
            "experiments": results,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # 生成可视化
    plot_results(results, sequence_lengths, keep_ratios, output_dir)
    
    return results


def plot_results(results: list, seq_lens: list, keep_ratios: list, output_dir: Path):
    """生成可视化图"""
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'figure.dpi': 150,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    colors = {1.0: '#2ecc71', 0.7: '#3498db', 0.5: '#e74c3c'}
    
    # 按 keep_ratio 分组
    for k in keep_ratios:
        k_results = [r for r in results if r["keep_ratio"] == k]
        
        # 按 seq_len 聚合（跨 seeds 平均）
        by_len = {}
        for r in k_results:
            sl = r["seq_len"]
            if sl not in by_len:
                by_len[sl] = {"probs": [], "norms": [], "entropies": []}
            by_len[sl]["probs"].append(r["mean_prob"])
            by_len[sl]["norms"].append(r["mean_norm"])
            by_len[sl]["entropies"].append(r["mean_entropy"])
        
        x = sorted(by_len.keys())
        y_prob = [np.mean(by_len[sl]["probs"]) for sl in x]
        y_norm = [np.mean(by_len[sl]["norms"]) for sl in x]
        y_entropy = [np.mean(by_len[sl]["entropies"]) for sl in x]
        
        # (a) Confidence
        axes[0].plot(x, y_prob, 'o-', color=colors.get(k, 'gray'), label=f'k={k}')
        
        # (b) Norm
        axes[1].plot(x, y_norm, 'o-', color=colors.get(k, 'gray'), label=f'k={k}')
        
        # (c) Entropy
        axes[2].plot(x, y_entropy, 'o-', color=colors.get(k, 'gray'), label=f'k={k}')
    
    axes[0].set_xlabel('Sequence Length (symbols)')
    axes[0].set_ylabel('Classification Confidence')
    axes[0].set_title('(a) Confidence vs Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Sequence Length (symbols)')
    axes[1].set_ylabel('Hidden State L2 Norm')
    axes[1].set_title('(b) State Accumulation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Sequence Length (symbols)')
    axes[2].set_ylabel('Prediction Entropy')
    axes[2].set_title('(c) Uncertainty')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "stress_test_256_plot.png"
    fig.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {plot_path}")
    
    # 生成 Delta 图
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 计算相对于 baseline 的 delta
    baseline_by_len = {}
    for r in results:
        if r["keep_ratio"] == 1.0:
            sl = r["seq_len"]
            if sl not in baseline_by_len:
                baseline_by_len[sl] = []
            baseline_by_len[sl].append(r["mean_prob"])
    
    for sl in baseline_by_len:
        baseline_by_len[sl] = np.mean(baseline_by_len[sl])
    
    for k in [0.7, 0.5]:
        k_results = [r for r in results if r["keep_ratio"] == k]
        
        by_len = {}
        for r in k_results:
            sl = r["seq_len"]
            if sl not in by_len:
                by_len[sl] = []
            by_len[sl].append(r["mean_prob"])
        
        x = sorted(by_len.keys())
        y_delta = [(np.mean(by_len[sl]) - baseline_by_len.get(sl, 0)) * 100 for sl in x]
        
        ax.plot(x, y_delta, 'o-', color=colors.get(k, 'gray'), label=f'k={k}', linewidth=2, markersize=8)
        
        # 标注数值
        for xi, yi in zip(x, y_delta):
            ax.annotate(f'{yi:+.1f}pp', xy=(xi, yi), xytext=(5, 5),
                       textcoords='offset points', fontsize=9, color=colors.get(k, 'gray'))
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=64, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax.fill_between([64, 300], -15, 10, alpha=0.1, color='green')
    
    ax.set_xlabel('Sequence Length (symbols)', fontsize=12)
    ax.set_ylabel('Delta Confidence vs Baseline (pp)', fontsize=12)
    ax.set_title('Pruning Effect at Extended Sequence Lengths', fontsize=13)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    delta_path = output_dir / "stress_test_256_delta.png"
    fig.savefig(delta_path, dpi=300)
    plt.close()
    
    print(f"Delta plot saved to {delta_path}")


def main():
    parser = argparse.ArgumentParser(description="SeqLen=256 Stress Test")
    parser.add_argument("--checkpoint", type=str, 
                       default="artifacts/checkpoints/mod_best_em0.75.pt")
    parser.add_argument("--sequence-lengths", type=str, default="32,64,128,192,256")
    parser.add_argument("--keep-ratios", type=str, default="1.0,0.7,0.5")
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--output-dir", type=str, default="reports/stress_test_256")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    seq_lens = [int(x) for x in args.sequence_lengths.split(",")]
    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    
    run_stress_test(
        checkpoint_path=args.checkpoint,
        sequence_lengths=seq_lens,
        keep_ratios=keep_ratios,
        num_samples=args.num_samples,
        seeds=seeds,
        device=device,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

