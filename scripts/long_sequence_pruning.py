#!/usr/bin/env python3
"""
超长序列修剪实验 (Long Sequence Pruning)

核心假设：
- 对于短序列（32符号），修剪不必要
- 对于超长序列（100+符号），状态累积可能过载，修剪变得有益

实验设计：
- 序列长度：32, 64, 96, 128, 160 符号
- 修剪策略：无修剪、选择性修剪（k=0.5, 0.7）
- 观察：性能随序列长度的变化趋势

Author: Jericho Team
Date: 2026-01-02
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig, SSMLikeBlock
from jericho.symbols import SR, SYMBOL2FREQ, encode_symbols_to_wave


class SelectivePruningWrapper(nn.Module):
    """选择性修剪包装器"""
    
    def __init__(self, base_model: MiniJMamba, keep_ratio: float = 1.0):
        super().__init__()
        self.model = base_model
        self.config = base_model.config
        self.keep_ratio = keep_ratio
        
    def forward(
        self,
        frames: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        *,
        return_hidden: bool = False,
    ):
        batch_size, seq_len, feat_dim = frames.shape
        x = self.model.input_proj(frames)
        
        if self.model.pos_emb is not None:
            positions = torch.arange(seq_len, device=frames.device)
            positions = positions.clamp(max=self.config.max_frames - 1)
            x = x + self.model.pos_emb(positions).unsqueeze(0)
        
        x = self.model.dropout(x)
        
        for layer in self.model.layers:
            x = layer(x, padding_mask)
            
            # 选择性修剪
            if self.keep_ratio < 1.0:
                x = self._selective_prune(x)
        
        x = self.model.final_norm(x)
        frame_outputs = self.model.frame_head(x)
        symbol_logits = self.model.symbol_head(x)
        
        if padding_mask is not None:
            frame_outputs = frame_outputs.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
            symbol_logits = symbol_logits.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        
        if return_hidden:
            return frame_outputs, symbol_logits, x
        return frame_outputs, symbol_logits
    
    def _selective_prune(self, h: torch.Tensor) -> torch.Tensor:
        """选择性修剪：保留强通道"""
        B, T, D = h.shape
        
        channel_importance = torch.norm(h, dim=(0, 1))  # (D,)
        k = int(D * self.keep_ratio)
        if k == 0:
            k = 1
        threshold = torch.topk(channel_importance, k).values[-1]
        
        mask = (channel_importance >= threshold).float().unsqueeze(0).unsqueeze(0)
        return h * mask


def generate_long_sequence(
    num_symbols: int,
    frame_size: int,
    device: str,
    seed: int = 42,
) -> tuple:
    """生成长序列测试数据"""
    np.random.seed(seed)
    symbols = list(SYMBOL2FREQ.keys())[:10]
    
    symbol_seq = [symbols[np.random.randint(0, len(symbols))] for _ in range(num_symbols)]
    wave = encode_symbols_to_wave(symbol_seq, tone_dur=0.1, gap_dur=0.0)
    
    num_frames = len(wave) // frame_size
    if num_frames == 0:
        raise ValueError(f"Sequence too short: {len(wave)} samples, frame_size={frame_size}")
    
    frames = wave[:num_frames * frame_size].reshape(num_frames, frame_size)
    frames_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
    padding_mask = torch.ones(1, num_frames, dtype=torch.bool)
    
    return frames_tensor.to(device), padding_mask.to(device), symbol_seq


def run_experiment(
    checkpoint_path: str,
    sequence_lengths: List[int] = [32, 64, 96, 128, 160],
    keep_ratios: List[float] = [1.0, 0.7, 0.5],
    num_samples: int = 20,
    frame_size: int = 160,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seeds: List[int] = [42, 123, 456],
) -> Dict[str, Any]:
    """运行超长序列实验"""
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Sequence lengths: {sequence_lengths}")
    print(f"Keep ratios: {keep_ratios}")
    print(f"Device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 增加 max_frames 以支持长序列
    max_frames = max(sequence_lengths) * 20  # 足够长
    
    config = MiniJMambaConfig(
        frame_size=frame_size,
        hop_size=frame_size,
        symbol_vocab_size=12,
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
        max_frames=max_frames,
    )
    
    base_model = MiniJMamba(config).to(device)
    if 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        base_model.load_state_dict(checkpoint, strict=False)
    base_model.eval()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        "sequence_lengths": sequence_lengths,
        "keep_ratios": keep_ratios,
        "seeds": seeds,
        "experiments": [],
    }
    
    for seq_len in sequence_lengths:
        print(f"\n=== Sequence Length = {seq_len} symbols ===")
        
        for keep_ratio in keep_ratios:
            all_probs = []
            all_norms = []
            all_entropies = []
            
            for seed in seeds:
                # 创建包装器
                if keep_ratio < 1.0:
                    model = SelectivePruningWrapper(base_model, keep_ratio).to(device)
                else:
                    model = base_model
                model.eval()
                
                seed_probs = []
                seed_norms = []
                seed_entropies = []
                
                for sample_idx in range(num_samples):
                    try:
                        frames, mask, symbols = generate_long_sequence(
                            seq_len, frame_size, device, seed=seed * 1000 + sample_idx
                        )
                        
                        with torch.no_grad():
                            _, logits, hidden = model(frames, mask, return_hidden=True)
                            
                            # 计算指标
                            probs = torch.softmax(logits, dim=-1)
                            max_prob = probs.max(dim=-1).values.mean().item()
                            
                            hidden_norm = torch.norm(hidden).item()
                            
                            # 熵
                            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
                            
                            seed_probs.append(max_prob)
                            seed_norms.append(hidden_norm)
                            seed_entropies.append(entropy)
                            
                    except Exception as e:
                        print(f"    Error at seq_len={seq_len}, seed={seed}, sample={sample_idx}: {e}")
                        continue
                
                if seed_probs:
                    all_probs.extend(seed_probs)
                    all_norms.extend(seed_norms)
                    all_entropies.extend(seed_entropies)
            
            if all_probs:
                mean_prob = np.mean(all_probs)
                std_prob = np.std(all_probs)
                mean_norm = np.mean(all_norms)
                mean_entropy = np.mean(all_entropies)
                
                keep_str = f"keep={keep_ratio}" if keep_ratio < 1.0 else "baseline"
                print(f"  {keep_str}: prob={mean_prob:.3f}+/-{std_prob:.3f}, norm={mean_norm:.1f}, entropy={mean_entropy:.3f}")
                
                results["experiments"].append({
                    "seq_len": seq_len,
                    "keep_ratio": keep_ratio,
                    "mean_prob": mean_prob,
                    "std_prob": std_prob,
                    "mean_norm": mean_norm,
                    "mean_entropy": mean_entropy,
                    "n_samples": len(all_probs),
                })
    
    return results


def print_summary(results: Dict[str, Any]):
    """打印实验总结"""
    print("\n" + "="*70)
    print("LONG SEQUENCE PRUNING EXPERIMENT SUMMARY")
    print("="*70)
    
    # 按序列长度分组
    by_length = {}
    for exp in results["experiments"]:
        seq_len = exp["seq_len"]
        if seq_len not in by_length:
            by_length[seq_len] = {}
        by_length[seq_len][exp["keep_ratio"]] = exp
    
    print("\n| Seq Len | Baseline | keep=0.7 | keep=0.5 | Best |")
    print("|---------|----------|----------|----------|------|")
    
    for seq_len in sorted(by_length.keys()):
        row = f"| {seq_len:7d} |"
        
        baseline = by_length[seq_len].get(1.0, {}).get("mean_prob", 0)
        k07 = by_length[seq_len].get(0.7, {}).get("mean_prob", 0)
        k05 = by_length[seq_len].get(0.5, {}).get("mean_prob", 0)
        
        row += f" {baseline:.3f}    |"
        
        # 与 baseline 比较
        delta_07 = k07 - baseline if baseline > 0 else 0
        delta_05 = k05 - baseline if baseline > 0 else 0
        
        row += f" {k07:.3f} ({delta_07*100:+.1f}pp) |"
        row += f" {k05:.3f} ({delta_05*100:+.1f}pp) |"
        
        # 最佳
        best_keep = max([(1.0, baseline), (0.7, k07), (0.5, k05)], key=lambda x: x[1])
        row += f" k={best_keep[0]} |"
        
        print(row)
    
    # 趋势分析
    print("\n" + "="*70)
    print("TREND ANALYSIS")
    print("="*70)
    
    # 检查是否随序列长度增加，修剪变得更有益
    print("\nDelta (pruned - baseline) by sequence length:")
    print("| Seq Len | delta(k=0.7) | delta(k=0.5) |")
    print("|---------|--------------|--------------|")
    
    for seq_len in sorted(by_length.keys()):
        baseline = by_length[seq_len].get(1.0, {}).get("mean_prob", 0)
        k07 = by_length[seq_len].get(0.7, {}).get("mean_prob", 0)
        k05 = by_length[seq_len].get(0.5, {}).get("mean_prob", 0)
        
        delta_07 = (k07 - baseline) * 100 if baseline > 0 else 0
        delta_05 = (k05 - baseline) * 100 if baseline > 0 else 0
        
        trend_07 = "[+]" if delta_07 > 1 else "[-]" if delta_07 < -1 else "[=]"
        trend_05 = "[+]" if delta_05 > 1 else "[-]" if delta_05 < -1 else "[=]"
        
        print(f"| {seq_len:7d} | {delta_07:+.1f}pp {trend_07}   | {delta_05:+.1f}pp {trend_05}   |")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Long Sequence Pruning Experiment')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sequence-lengths', type=str, default='32,64,96,128')
    parser.add_argument('--keep-ratios', type=str, default='1.0,0.7,0.5')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--seeds', type=str, default='42,123,456')
    parser.add_argument('--output-dir', type=str, default='reports/long_sequence')
    
    args = parser.parse_args()
    
    sequence_lengths = [int(x) for x in args.sequence_lengths.split(',')]
    keep_ratios = [float(x) for x in args.keep_ratios.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_experiment(
        checkpoint_path=args.checkpoint,
        sequence_lengths=sequence_lengths,
        keep_ratios=keep_ratios,
        num_samples=args.num_samples,
        seeds=seeds,
    )
    
    # 保存结果
    with open(output_dir / 'long_sequence_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_summary(results)
    
    # 绘图
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 按 keep_ratio 分组
    by_keep = {}
    for exp in results["experiments"]:
        k = exp["keep_ratio"]
        if k not in by_keep:
            by_keep[k] = {"seq_lens": [], "probs": [], "norms": [], "entropies": []}
        by_keep[k]["seq_lens"].append(exp["seq_len"])
        by_keep[k]["probs"].append(exp["mean_prob"])
        by_keep[k]["norms"].append(exp["mean_norm"])
        by_keep[k]["entropies"].append(exp["mean_entropy"])
    
    colors = {1.0: '#2ecc71', 0.7: '#3498db', 0.5: '#e74c3c'}
    labels = {1.0: 'Baseline', 0.7: 'keep=0.7', 0.5: 'keep=0.5'}
    
    # 置信度
    ax = axes[0]
    for k, data in sorted(by_keep.items(), reverse=True):
        sorted_pairs = sorted(zip(data["seq_lens"], data["probs"]))
        x, y = zip(*sorted_pairs)
        ax.plot(x, y, 'o-', color=colors.get(k, 'gray'), label=labels.get(k, f'k={k}'))
    ax.set_xlabel('Sequence Length (symbols)')
    ax.set_ylabel('Max Probability')
    ax.set_title('Classification Confidence vs Sequence Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 隐状态范数
    ax = axes[1]
    for k, data in sorted(by_keep.items(), reverse=True):
        sorted_pairs = sorted(zip(data["seq_lens"], data["norms"]))
        x, y = zip(*sorted_pairs)
        ax.plot(x, y, 'o-', color=colors.get(k, 'gray'), label=labels.get(k, f'k={k}'))
    ax.set_xlabel('Sequence Length (symbols)')
    ax.set_ylabel('Hidden State Norm')
    ax.set_title('State Accumulation vs Sequence Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Delta (pruned - baseline)
    ax = axes[2]
    if 1.0 in by_keep:
        baseline_by_len = dict(zip(by_keep[1.0]["seq_lens"], by_keep[1.0]["probs"]))
        
        for k, data in sorted(by_keep.items(), reverse=True):
            if k == 1.0:
                continue
            deltas = []
            seq_lens = []
            for sl, prob in zip(data["seq_lens"], data["probs"]):
                if sl in baseline_by_len:
                    deltas.append((prob - baseline_by_len[sl]) * 100)
                    seq_lens.append(sl)
            
            if seq_lens:
                sorted_pairs = sorted(zip(seq_lens, deltas))
                x, y = zip(*sorted_pairs)
                ax.plot(x, y, 'o-', color=colors.get(k, 'gray'), label=labels.get(k, f'k={k}'))
    
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Sequence Length (symbols)')
    ax.set_ylabel('Delta vs Baseline (pp)')
    ax.set_title('Pruning Benefit vs Sequence Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'long_sequence_results.png', dpi=150)
    plt.close()
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()

