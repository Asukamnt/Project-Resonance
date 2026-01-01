#!/usr/bin/env python3
"""
选择性突触修剪实验 (Selective Synaptic Pruning)

核心思想：模拟 Tononi 的突触稳态假说 (SHY)
- 不是均匀衰减所有状态
- 而是：强信号保留，弱信号（噪声）清除
- 这才是真正的"睡眠"机制

实验设计：
1. 计算隐状态每个维度的"强度"（L2 范数）
2. 根据强度排序，保留前 K% 最强的
3. 弱信号乘以 decay factor（或直接归零）
4. 对比不同保留比例的效果

Author: Jericho Team
Date: 2026-01-02
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig, SSMLikeBlock
from jericho.symbols import SR, SYMBOL2FREQ, encode_symbols_to_wave


class SelectivePruningSSMBlock(nn.Module):
    """带选择性修剪的 SSM Block"""
    
    def __init__(
        self,
        original_block: SSMLikeBlock,
        prune_period: int = 10,
        keep_ratio: float = 0.7,      # 保留前 70% 最强的
        weak_decay: float = 0.1,      # 弱信号衰减到 10%
        prune_mode: str = "channel",  # "channel" 或 "spatial"
    ):
        super().__init__()
        self.block = original_block
        self.prune_period = prune_period
        self.keep_ratio = keep_ratio
        self.weak_decay = weak_decay
        self.prune_mode = prune_mode
        self.frame_counter = 0
    
    def reset_counter(self):
        self.frame_counter = 0
    
    def selective_prune(self, h: torch.Tensor) -> torch.Tensor:
        """选择性修剪：保留强信号，衰减弱信号
        
        Args:
            h: 隐状态 (B, T, D)
        
        Returns:
            修剪后的隐状态
        """
        if self.prune_mode == "channel":
            # 按通道（维度）修剪：计算每个维度在所有位置的平均强度
            # shape: (D,)
            strength = h.abs().mean(dim=(0, 1))
            threshold = torch.quantile(strength, 1 - self.keep_ratio)
            
            # 强通道保留，弱通道衰减
            mask = (strength > threshold).float()  # (D,)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            
            h_pruned = h * mask + h * (1 - mask) * self.weak_decay
            
        elif self.prune_mode == "spatial":
            # 按位置修剪：计算每个位置的强度
            # shape: (B, T)
            strength = h.norm(dim=-1)
            
            # 对每个 batch 单独计算阈值
            threshold = torch.quantile(strength, 1 - self.keep_ratio, dim=-1, keepdim=True)
            
            mask = (strength > threshold).float().unsqueeze(-1)  # (B, T, 1)
            h_pruned = h * mask + h * (1 - mask) * self.weak_decay
            
        elif self.prune_mode == "element":
            # 按元素修剪：每个元素独立
            strength = h.abs()
            threshold = torch.quantile(strength.flatten(), 1 - self.keep_ratio)
            
            mask = (strength > threshold).float()
            h_pruned = h * mask + h * (1 - mask) * self.weak_decay
            
        else:
            raise ValueError(f"Unknown prune_mode: {self.prune_mode}")
        
        return h_pruned
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # 正常 SSM 前向
        out = self.block(x, mask)
        
        batch_size, seq_len, d_model = out.shape
        self.frame_counter += seq_len
        
        # 周期性修剪
        if self.frame_counter >= self.prune_period:
            out = self.selective_prune(out)
            self.frame_counter = 0
        
        return out


class SelectivePruningMiniJMamba(nn.Module):
    """带选择性修剪的 MiniJMamba"""
    
    def __init__(
        self,
        base_model: MiniJMamba,
        prune_period: int = 10,
        keep_ratio: float = 0.7,
        weak_decay: float = 0.1,
        prune_mode: str = "channel",
    ):
        super().__init__()
        self.config = base_model.config
        self.prune_period = prune_period
        self.keep_ratio = keep_ratio
        self.weak_decay = weak_decay
        self.prune_mode = prune_mode
        
        # 复制组件
        self.input_proj = base_model.input_proj
        self.pos_emb = base_model.pos_emb
        self.dropout = base_model.dropout
        self.final_norm = base_model.final_norm
        self.frame_head = base_model.frame_head
        self.symbol_head = base_model.symbol_head
        
        # 包装 SSM 层
        self.layers = nn.ModuleList()
        self.pruning_layers = []
        
        for layer in base_model.layers:
            if isinstance(layer, SSMLikeBlock):
                wrapper = SelectivePruningSSMBlock(
                    layer, prune_period, keep_ratio, weak_decay, prune_mode
                )
                self.layers.append(wrapper)
                self.pruning_layers.append(wrapper)
            else:
                self.layers.append(layer)
    
    def reset_all_counters(self):
        for layer in self.pruning_layers:
            layer.reset_counter()
    
    def forward(
        self,
        frames: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        *,
        return_hidden: bool = False,
    ):
        self.reset_all_counters()
        
        batch_size, seq_len, feat_dim = frames.shape
        x = self.input_proj(frames)
        
        if self.pos_emb is not None:
            positions = torch.arange(seq_len, device=frames.device)
            positions = positions.clamp(max=self.config.max_frames - 1)
            x = x + self.pos_emb(positions).unsqueeze(0)
        
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, padding_mask)
        
        x = self.final_norm(x)
        frame_outputs = self.frame_head(x)
        symbol_logits = self.symbol_head(x)
        
        if padding_mask is not None:
            frame_outputs = frame_outputs.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
            symbol_logits = symbol_logits.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        
        if return_hidden:
            return frame_outputs, symbol_logits, x
        return frame_outputs, symbol_logits


def generate_sequence(num_symbols: int) -> torch.Tensor:
    symbols = list(SYMBOL2FREQ.keys())[:10]
    symbol_seq = [symbols[i % len(symbols)] for i in range(num_symbols)]
    wave = encode_symbols_to_wave(symbol_seq, tone_dur=0.1, gap_dur=0.0)
    return torch.tensor(wave, dtype=torch.float32)


def measure_model(
    model: nn.Module,
    num_symbols: int,
    frame_size: int,
    device: str,
) -> Dict[str, float]:
    wave = generate_sequence(num_symbols)
    num_frames = len(wave) // frame_size
    if num_frames == 0:
        return {"hidden_norm": 0, "entropy": 0, "max_prob": 0, "sparsity": 0}
    
    frames = wave[:num_frames * frame_size].reshape(1, num_frames, frame_size).to(device)
    padding_mask = torch.ones(1, num_frames, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        output, logits, hidden = model(frames, padding_mask, return_hidden=True)
        
        hidden_norm = hidden.norm().item()
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        max_prob = probs.max(dim=-1).values.mean().item()
        
        # 稀疏度：接近零的元素比例
        sparsity = (hidden.abs() < 0.01).float().mean().item()
    
    return {
        "hidden_norm": hidden_norm,
        "entropy": entropy,
        "max_prob": max_prob,
        "sparsity": sparsity,
    }


def run_experiment(
    checkpoint_path: str,
    max_symbols: int = 64,
    keep_ratios: List[float] = [0.3, 0.5, 0.7, 0.9, 1.0],
    weak_decays: List[float] = [0.0, 0.1],
    prune_mode: str = "channel",
    prune_period: int = 10,
    frame_size: int = 160,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Prune mode: {prune_mode}")
    print(f"Keep ratios: {keep_ratios}")
    print(f"Weak decays: {weak_decays}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = MiniJMambaConfig(
        frame_size=frame_size,
        hop_size=frame_size,
        symbol_vocab_size=12,
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
    )
    
    base_model = MiniJMamba(config).to(device)
    if 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        base_model.load_state_dict(checkpoint, strict=False)
    base_model.eval()
    
    results = {
        "prune_mode": prune_mode,
        "prune_period": prune_period,
        "sequence_lengths": list(range(5, max_symbols + 1, 5)),
        "experiments": [],
    }
    
    # 测试所有组合
    for keep_ratio in keep_ratios:
        for weak_decay in weak_decays:
            if keep_ratio == 1.0 and weak_decay > 0:
                continue  # 跳过无意义组合
            
            exp_name = f"keep={keep_ratio}_decay={weak_decay}"
            print(f"\n--- Testing {exp_name} ---")
            
            if keep_ratio == 1.0:
                model = base_model
            else:
                model = SelectivePruningMiniJMamba(
                    base_model, prune_period, keep_ratio, weak_decay, prune_mode
                ).to(device)
                model.eval()
            
            exp_data = {
                "name": exp_name,
                "keep_ratio": keep_ratio,
                "weak_decay": weak_decay,
                "hidden_norms": [],
                "entropies": [],
                "max_probs": [],
                "sparsities": [],
            }
            
            for num_symbols in results["sequence_lengths"]:
                metrics = measure_model(model, num_symbols, frame_size, device)
                exp_data["hidden_norms"].append(metrics["hidden_norm"])
                exp_data["entropies"].append(metrics["entropy"])
                exp_data["max_probs"].append(metrics["max_prob"])
                exp_data["sparsities"].append(metrics["sparsity"])
            
            final_norm = exp_data["hidden_norms"][-1]
            final_prob = exp_data["max_probs"][-1]
            print(f"  Final: norm={final_norm:.1f}, max_prob={final_prob:.3f}")
            
            results["experiments"].append(exp_data)
    
    return results


def plot_results(results: Dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = results["experiments"]
    lengths = results["sequence_lengths"]
    
    # 找到基线（keep=1.0）
    baseline = next(e for e in experiments if e["keep_ratio"] == 1.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Selective Synaptic Pruning Experiment ({results["prune_mode"]} mode)', 
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
    
    # 1. 隐状态范数
    ax1 = axes[0, 0]
    for i, exp in enumerate(experiments):
        style = '--' if exp["keep_ratio"] == 1.0 else '-'
        lw = 2.5 if exp["keep_ratio"] == 1.0 else 1.5
        ax1.plot(lengths, exp["hidden_norms"], style, color=colors[i], 
                linewidth=lw, marker='o', markersize=3, label=exp["name"])
    ax1.set_xlabel('Sequence Length (symbols)')
    ax1.set_ylabel('Hidden State Norm')
    ax1.set_title('Hidden Norm (lower = less accumulation)')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 分类置信度
    ax2 = axes[0, 1]
    for i, exp in enumerate(experiments):
        style = '--' if exp["keep_ratio"] == 1.0 else '-'
        lw = 2.5 if exp["keep_ratio"] == 1.0 else 1.5
        ax2.plot(lengths, exp["max_probs"], style, color=colors[i], 
                linewidth=lw, marker='s', markersize=3, label=exp["name"])
    ax2.set_xlabel('Sequence Length (symbols)')
    ax2.set_ylabel('Max Probability')
    ax2.set_title('Classification Confidence (higher = better)')
    ax2.legend(fontsize=7, loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # 3. 范数相对变化
    ax3 = axes[1, 0]
    baseline_norms = baseline["hidden_norms"]
    for i, exp in enumerate(experiments):
        if exp["keep_ratio"] == 1.0:
            continue
        ratios = [e / b if b > 0 else 1 for e, b in zip(exp["hidden_norms"], baseline_norms)]
        ax3.plot(lengths, ratios, '-', color=colors[i], linewidth=1.5, 
                marker='^', markersize=3, label=exp["name"])
    ax3.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
    ax3.set_xlabel('Sequence Length (symbols)')
    ax3.set_ylabel('Norm Ratio (vs baseline)')
    ax3.set_title('Norm Reduction (< 1.0 = improvement)')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # 4. 置信度相对变化
    ax4 = axes[1, 1]
    baseline_probs = baseline["max_probs"]
    for i, exp in enumerate(experiments):
        if exp["keep_ratio"] == 1.0:
            continue
        ratios = [e / b if b > 0 else 1 for e, b in zip(exp["max_probs"], baseline_probs)]
        ax4.plot(lengths, ratios, '-', color=colors[i], linewidth=1.5, 
                marker='d', markersize=3, label=exp["name"])
    ax4.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
    ax4.set_xlabel('Sequence Length (symbols)')
    ax4.set_ylabel('Confidence Ratio (vs baseline)')
    ax4.set_title('Confidence Change (> 1.0 = improvement)')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'selective_pruning_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {output_dir / 'selective_pruning_results.png'}")


def main():
    parser = argparse.ArgumentParser(description='Selective Synaptic Pruning Experiment')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--max-symbols', type=int, default=64)
    parser.add_argument('--prune-period', type=int, default=10)
    parser.add_argument('--prune-mode', type=str, default='channel',
                        choices=['channel', 'spatial', 'element'])
    parser.add_argument('--keep-ratios', type=str, default='0.3,0.5,0.7,0.9,1.0')
    parser.add_argument('--weak-decays', type=str, default='0.0,0.1')
    parser.add_argument('--output-dir', type=str, default='reports/selective_pruning')
    
    args = parser.parse_args()
    
    keep_ratios = [float(x) for x in args.keep_ratios.split(',')]
    weak_decays = [float(x) for x in args.weak_decays.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_experiment(
        checkpoint_path=args.checkpoint,
        max_symbols=args.max_symbols,
        keep_ratios=keep_ratios,
        weak_decays=weak_decays,
        prune_mode=args.prune_mode,
        prune_period=args.prune_period,
    )
    
    # 保存
    with open(output_dir / 'selective_pruning_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # 绘图
    plot_results(results, output_dir)
    
    # 摘要
    print("\n" + "="*70)
    print("SELECTIVE SYNAPTIC PRUNING EXPERIMENT SUMMARY")
    print("="*70)
    
    baseline = next(e for e in results["experiments"] if e["keep_ratio"] == 1.0)
    baseline_norm = baseline["hidden_norms"][-1]
    baseline_prob = baseline["max_probs"][-1]
    
    print(f"Baseline (no pruning): norm={baseline_norm:.1f}, confidence={baseline_prob:.3f}")
    print()
    
    best_exp = None
    best_score = 0
    
    for exp in results["experiments"]:
        if exp["keep_ratio"] == 1.0:
            continue
        
        final_norm = exp["hidden_norms"][-1]
        final_prob = exp["max_probs"][-1]
        
        # 评分：置信度保持度 * 范数减少度
        norm_reduction = baseline_norm / final_norm if final_norm > 0 else 0
        prob_retention = final_prob / baseline_prob if baseline_prob > 0 else 0
        score = prob_retention * norm_reduction
        
        print(f"  {exp['name']}: norm={final_norm:.1f} ({final_norm/baseline_norm:.2f}x), "
              f"confidence={final_prob:.3f} ({prob_retention:.2f}x), score={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_exp = exp
    
    if best_exp:
        print(f"\nBest configuration: {best_exp['name']} (score={best_score:.3f})")
        
        if best_score > 1.1:
            print("[OK] Selective pruning IMPROVES performance!")
        elif best_score > 0.95:
            print("[OK] Selective pruning maintains performance with reduced state.")
        else:
            print("[--] Selective pruning still degrades performance.")


if __name__ == '__main__':
    main()

