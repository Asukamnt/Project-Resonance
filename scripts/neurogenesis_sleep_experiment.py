#!/usr/bin/env python3
"""
神经增生 + 睡眠修剪实验 (Neurogenesis + Sleep Pruning)

核心假设：
- 清醒期：动态增加隐状态容量来处理新信息
- 睡眠期：选择性修剪回原始容量，保留重要信息

生物学对应：
- 海马体神经发生 (Hippocampal Neurogenesis)
- 睡眠期突触稳态 (Synaptic Homeostasis during Sleep)

实验设计：
1. 基线：固定容量 d_model=128
2. 增生：扩展到 d_model=192 或 256
3. 修剪：选择性压缩回 128
4. 对比：增生+修剪 vs 纯修剪 vs 基线

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


class DynamicCapacityWrapper(nn.Module):
    """
    动态容量包装器：支持增生和修剪
    
    增生：将 d_model 维的隐状态扩展到 expanded_dim
    修剪：选择性地将 expanded_dim 压缩回 d_model
    """
    
    def __init__(
        self,
        base_model: MiniJMamba,
        expansion_ratio: float = 1.5,  # 扩展比例
    ):
        super().__init__()
        self.config = base_model.config
        self.d_model = self.config.d_model
        self.expanded_dim = int(self.d_model * expansion_ratio)
        self.expansion_ratio = expansion_ratio
        
        # 复制基础模型组件
        self.input_proj = base_model.input_proj
        self.pos_emb = base_model.pos_emb
        self.dropout = base_model.dropout
        self.final_norm = base_model.final_norm
        self.frame_head = base_model.frame_head
        self.symbol_head = base_model.symbol_head
        self.layers = base_model.layers
        
        # 增生投影：d_model -> expanded_dim
        self.expand_proj = nn.Linear(self.d_model, self.expanded_dim)
        nn.init.eye_(self.expand_proj.weight[:self.d_model, :])  # 保持原始信息
        nn.init.zeros_(self.expand_proj.bias)
        
        # 修剪投影：expanded_dim -> d_model
        self.prune_proj = nn.Linear(self.expanded_dim, self.d_model)
        nn.init.eye_(self.prune_proj.weight[:, :self.d_model])  # 保持原始信息
        nn.init.zeros_(self.prune_proj.bias)
        
        # 当前模式
        self.mode = "baseline"  # baseline, expanded, pruned
        
        # 记录增生后的隐状态（用于分析）
        self.expanded_states = []
        
    def set_mode(self, mode: str):
        """设置运行模式"""
        assert mode in ["baseline", "expanded", "pruned", "expand_then_prune"]
        self.mode = mode
        
    def selective_prune(self, h: torch.Tensor, keep_ratio: float = 0.5) -> torch.Tensor:
        """
        选择性修剪：保留强通道，压缩回原始维度
        
        Args:
            h: (B, T, expanded_dim)
            keep_ratio: 保留比例（相对于 d_model）
        
        Returns:
            pruned_h: (B, T, d_model)
        """
        B, T, D = h.shape
        
        # 计算每个通道的重要性（L2 范数）
        channel_importance = torch.norm(h, dim=(0, 1))  # (expanded_dim,)
        
        # 选择 top-k 通道
        k = int(self.d_model * keep_ratio)
        top_indices = torch.topk(channel_importance, k).indices
        
        # 提取重要通道
        h_selected = h[:, :, top_indices]  # (B, T, k)
        
        # 填充到 d_model（用零或插值）
        h_pruned = torch.zeros(B, T, self.d_model, device=h.device, dtype=h.dtype)
        h_pruned[:, :, :k] = h_selected
        
        return h_pruned
    
    def forward(
        self,
        frames: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        *,
        return_hidden: bool = False,
        keep_ratio: float = 0.5,
    ):
        batch_size, seq_len, feat_dim = frames.shape
        x = self.input_proj(frames)
        
        if self.pos_emb is not None:
            positions = torch.arange(seq_len, device=frames.device)
            positions = positions.clamp(max=self.config.max_frames - 1)
            x = x + self.pos_emb(positions).unsqueeze(0)
        
        x = self.dropout(x)
        
        # 根据模式处理
        if self.mode == "baseline":
            # 正常处理
            for layer in self.layers:
                x = layer(x, padding_mask)
                
        elif self.mode == "expanded":
            # 增生模式：扩展维度
            x = self.expand_proj(x)  # (B, T, expanded_dim)
            self.expanded_states = []
            
            for layer in self.layers:
                # 需要调整层的输入/输出维度...
                # 这里简化处理：用投影
                x_down = self.prune_proj(x)  # 临时降维
                x_down = layer(x_down, padding_mask)
                x = self.expand_proj(x_down)  # 再升维
                self.expanded_states.append(x.detach().cpu())
            
            # 最后降维回去
            x = self.prune_proj(x)
            
        elif self.mode == "pruned":
            # 纯修剪模式（之前的实验）
            for layer in self.layers:
                x = layer(x, padding_mask)
                # 选择性修剪
                x = self._selective_prune_in_place(x, keep_ratio)
                
        elif self.mode == "expand_then_prune":
            # 完整周期：先增生，再修剪
            x = self.expand_proj(x)  # 增生
            self.expanded_states = []
            
            for i, layer in enumerate(self.layers):
                x_down = self.prune_proj(x)
                x_down = layer(x_down, padding_mask)
                x = self.expand_proj(x_down)
                self.expanded_states.append(x.detach().cpu())
                
                # 每隔几层做一次选择性修剪（模拟"微睡眠"）
                if (i + 1) % 4 == 0:
                    x = self._selective_prune_expanded(x, keep_ratio)
            
            # 最终修剪回原始维度
            x = self.prune_proj(x)
        
        x = self.final_norm(x)
        frame_outputs = self.frame_head(x)
        symbol_logits = self.symbol_head(x)
        
        if padding_mask is not None:
            frame_outputs = frame_outputs.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
            symbol_logits = symbol_logits.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        
        if return_hidden:
            return frame_outputs, symbol_logits, x
        return frame_outputs, symbol_logits
    
    def _selective_prune_in_place(self, h: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """在 d_model 维度内做选择性修剪"""
        B, T, D = h.shape
        
        channel_importance = torch.norm(h, dim=(0, 1))
        k = int(D * keep_ratio)
        threshold = torch.topk(channel_importance, k).values[-1]
        
        mask = (channel_importance >= threshold).float().unsqueeze(0).unsqueeze(0)
        return h * mask
    
    def _selective_prune_expanded(self, h: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        """在扩展维度内做选择性修剪"""
        B, T, D = h.shape
        
        channel_importance = torch.norm(h, dim=(0, 1))
        k = int(D * keep_ratio)
        threshold = torch.topk(channel_importance, k).values[-1]
        
        mask = (channel_importance >= threshold).float().unsqueeze(0).unsqueeze(0)
        return h * mask


def generate_test_batch(
    batch_size: int,
    num_symbols: int,
    frame_size: int,
    device: str,
    seed: int = 42,
) -> tuple:
    """生成测试数据"""
    np.random.seed(seed)
    symbols = list(SYMBOL2FREQ.keys())[:10]
    
    all_frames = []
    
    for i in range(batch_size):
        symbol_seq = [symbols[np.random.randint(0, len(symbols))] for _ in range(num_symbols)]
        wave = encode_symbols_to_wave(symbol_seq, tone_dur=0.1, gap_dur=0.0)
        
        num_frames = len(wave) // frame_size
        if num_frames == 0:
            continue
        
        frames = wave[:num_frames * frame_size].reshape(num_frames, frame_size)
        all_frames.append(torch.tensor(frames, dtype=torch.float32))
    
    max_len = max(f.shape[0] for f in all_frames)
    padded_frames = torch.zeros(batch_size, max_len, frame_size)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, f in enumerate(all_frames):
        padded_frames[i, :f.shape[0]] = f
        padding_mask[i, :f.shape[0]] = True
    
    return padded_frames.to(device), padding_mask.to(device)


def run_experiment(
    checkpoint_path: str,
    expansion_ratios: List[float] = [1.0, 1.5, 2.0],
    keep_ratios: List[float] = [0.3, 0.5, 0.7, 1.0],
    num_samples: int = 50,
    num_symbols: int = 32,
    frame_size: int = 160,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seeds: List[int] = [42, 123, 456],
) -> Dict[str, Any]:
    """运行完整实验"""
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Expansion ratios: {expansion_ratios}")
    print(f"Keep ratios: {keep_ratios}")
    print(f"Device: {device}")
    
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
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        "expansion_ratios": expansion_ratios,
        "keep_ratios": keep_ratios,
        "seeds": seeds,
        "experiments": [],
    }
    
    # 基线测试
    print("\n=== Baseline (no modification) ===")
    baseline_probs = []
    for seed in seeds:
        frames, mask = generate_test_batch(num_samples, num_symbols, frame_size, device, seed)
        with torch.no_grad():
            _, logits, _ = base_model(frames, mask, return_hidden=True)
            probs = torch.softmax(logits, dim=-1)
            max_prob = probs.max(dim=-1).values.mean().item()
            baseline_probs.append(max_prob)
            print(f"  Seed {seed}: max_prob = {max_prob:.3f}")
    
    baseline_mean = np.mean(baseline_probs)
    baseline_std = np.std(baseline_probs)
    print(f"  Baseline: {baseline_mean:.3f} +/- {baseline_std:.3f}")
    
    results["baseline"] = {
        "mean": baseline_mean,
        "std": baseline_std,
        "all": baseline_probs,
    }
    
    # 测试不同模式
    modes = ["pruned", "expanded", "expand_then_prune"]
    
    for expansion_ratio in expansion_ratios:
        if expansion_ratio == 1.0:
            continue  # 跳过，这就是基线
            
        print(f"\n=== Expansion Ratio = {expansion_ratio} ===")
        
        for keep_ratio in keep_ratios:
            print(f"\n  Keep ratio = {keep_ratio}")
            
            for mode in modes:
                mode_probs = []
                
                for seed in seeds:
                    # 每次创建新的包装器
                    wrapper = DynamicCapacityWrapper(base_model, expansion_ratio).to(device)
                    wrapper.set_mode(mode)
                    wrapper.eval()
                    
                    frames, mask = generate_test_batch(num_samples, num_symbols, frame_size, device, seed)
                    
                    with torch.no_grad():
                        _, logits, _ = wrapper(frames, mask, return_hidden=True, keep_ratio=keep_ratio)
                        probs = torch.softmax(logits, dim=-1)
                        max_prob = probs.max(dim=-1).values.mean().item()
                        mode_probs.append(max_prob)
                
                mean_prob = np.mean(mode_probs)
                std_prob = np.std(mode_probs)
                delta = mean_prob - baseline_mean
                
                print(f"    {mode}: {mean_prob:.3f} +/- {std_prob:.3f} (delta = {delta*100:+.1f}pp)")
                
                results["experiments"].append({
                    "expansion_ratio": expansion_ratio,
                    "keep_ratio": keep_ratio,
                    "mode": mode,
                    "mean_prob": mean_prob,
                    "std_prob": std_prob,
                    "delta": delta,
                    "all_probs": mode_probs,
                })
    
    return results


def print_summary(results: Dict[str, Any]):
    """打印实验总结"""
    print("\n" + "="*70)
    print("NEUROGENESIS + SLEEP PRUNING EXPERIMENT SUMMARY")
    print("="*70)
    
    baseline = results["baseline"]["mean"]
    print(f"\nBaseline: {baseline:.3f}")
    
    print("\n| Expansion | Keep | Mode              | max_prob | Delta    |")
    print("|-----------|------|-------------------|----------|----------|")
    
    for exp in results["experiments"]:
        delta_str = f"{exp['delta']*100:+.1f}pp"
        if exp["delta"] > 0.01:
            delta_str += " [+]"
        elif exp["delta"] < -0.01:
            delta_str += " [-]"
        
        print(f"| {exp['expansion_ratio']:.1f}       | {exp['keep_ratio']:.1f}  | {exp['mode']:<17} | {exp['mean_prob']:.3f}    | {delta_str:<8} |")
    
    # 找最佳配置
    best_exp = max(results["experiments"], key=lambda x: x["mean_prob"])
    print(f"\n[BEST] {best_exp['mode']} with expansion={best_exp['expansion_ratio']}, keep={best_exp['keep_ratio']}")
    print(f"       max_prob = {best_exp['mean_prob']:.3f} (delta = {best_exp['delta']*100:+.1f}pp)")
    
    # 结论
    print("\n" + "="*70)
    if best_exp["delta"] > 0.01:
        print("[OK] Neurogenesis + Pruning IMPROVES performance!")
    elif best_exp["delta"] > -0.01:
        print("[==] No significant difference from baseline.")
    else:
        print("[--] Neurogenesis + Pruning does not help.")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Neurogenesis + Sleep Pruning Experiment')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--expansion-ratios', type=str, default='1.5,2.0')
    parser.add_argument('--keep-ratios', type=str, default='0.3,0.5,0.7,1.0')
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--num-symbols', type=int, default=32)
    parser.add_argument('--seeds', type=str, default='42,123,456')
    parser.add_argument('--output-dir', type=str, default='reports/neurogenesis_sleep')
    
    args = parser.parse_args()
    
    expansion_ratios = [float(x) for x in args.expansion_ratios.split(',')]
    keep_ratios = [float(x) for x in args.keep_ratios.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_experiment(
        checkpoint_path=args.checkpoint,
        expansion_ratios=expansion_ratios,
        keep_ratios=keep_ratios,
        num_samples=args.num_samples,
        num_symbols=args.num_symbols,
        seeds=seeds,
    )
    
    # 保存结果
    with open(output_dir / 'neurogenesis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_summary(results)
    
    # 绘图
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    modes = ["pruned", "expanded", "expand_then_prune"]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    baseline = results["baseline"]["mean"]
    
    for ax_idx, mode in enumerate(modes):
        ax = axes[ax_idx]
        
        for exp_ratio in set(e["expansion_ratio"] for e in results["experiments"]):
            keep_vals = []
            prob_vals = []
            
            for exp in results["experiments"]:
                if exp["mode"] == mode and exp["expansion_ratio"] == exp_ratio:
                    keep_vals.append(exp["keep_ratio"])
                    prob_vals.append(exp["mean_prob"])
            
            if keep_vals:
                ax.plot(keep_vals, prob_vals, 'o-', label=f'exp={exp_ratio}')
        
        ax.axhline(y=baseline, color='gray', linestyle='--', label='Baseline')
        ax.set_xlabel('Keep Ratio')
        ax.set_ylabel('Max Probability')
        ax.set_title(f'Mode: {mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'neurogenesis_results.png', dpi=150)
    plt.close()
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()

