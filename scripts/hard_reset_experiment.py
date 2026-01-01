#!/usr/bin/env python3
"""
硬重置机制实验：验证周期性状态衰减是否能改善长序列稳定性

核心思想：不是插入静默帧，而是直接在 SSM 层输出后乘以 decay factor，
实现真正的"状态压缩"。

实验设计：
1. 包装 MiniJMamba，在每个 reset_period 帧后应用状态衰减
2. 对比不同 decay_factor 的效果
3. 测量隐状态范数和分类质量

Author: Jericho Team
Date: 2026-01-02
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig, SSMLikeBlock, AttentionBlock
from jericho.symbols import SR, SYMBOL2FREQ, encode_symbols_to_wave


class HardResetSSMBlock(nn.Module):
    """带硬重置的 SSM Block 包装器"""
    
    def __init__(
        self, 
        original_block: SSMLikeBlock,
        reset_period: int = 10,
        decay_factor: float = 0.1,
    ):
        super().__init__()
        self.block = original_block
        self.reset_period = reset_period
        self.decay_factor = decay_factor
        self.frame_counter = 0
    
    def reset_counter(self):
        self.frame_counter = 0
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # 对每一帧应用 SSM，并在周期性位置应用衰减
        outputs = []
        for t in range(seq_len):
            frame = x[:, t:t+1, :]  # (B, 1, D)
            out = self.block(frame, None if mask is None else mask[:, t:t+1])
            
            self.frame_counter += 1
            if self.frame_counter % self.reset_period == 0:
                # 硬重置：状态衰减
                out = out * self.decay_factor
            
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)


class HardResetMiniJMamba(nn.Module):
    """带硬重置机制的 MiniJMamba"""
    
    def __init__(
        self,
        base_model: MiniJMamba,
        reset_period: int = 10,
        decay_factor: float = 0.1,
    ):
        super().__init__()
        self.config = base_model.config
        self.reset_period = reset_period
        self.decay_factor = decay_factor
        
        # 复制基础模型的组件
        self.input_proj = base_model.input_proj
        self.pos_emb = base_model.pos_emb
        self.dropout = base_model.dropout
        self.final_norm = base_model.final_norm
        self.frame_head = base_model.frame_head
        self.symbol_head = base_model.symbol_head
        
        # 包装 SSM 层（保留 Attention 层不变）
        self.layers = nn.ModuleList()
        self.ssm_wrappers = []
        
        for layer in base_model.layers:
            if isinstance(layer, SSMLikeBlock):
                wrapper = HardResetSSMBlock(layer, reset_period, decay_factor)
                self.layers.append(wrapper)
                self.ssm_wrappers.append(wrapper)
            else:
                self.layers.append(layer)
    
    def reset_all_counters(self):
        for wrapper in self.ssm_wrappers:
            wrapper.reset_counter()
    
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
    """生成测试序列"""
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
    """测量模型指标"""
    wave = generate_sequence(num_symbols)
    num_frames = len(wave) // frame_size
    if num_frames == 0:
        return {"hidden_norm": 0, "entropy": 0, "max_prob": 0}
    
    frames = wave[:num_frames * frame_size].reshape(1, num_frames, frame_size).to(device)
    padding_mask = torch.ones(1, num_frames, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        output, logits, hidden = model(frames, padding_mask, return_hidden=True)
        
        hidden_norm = hidden.norm().item()
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        max_prob = probs.max(dim=-1).values.mean().item()
    
    return {"hidden_norm": hidden_norm, "entropy": entropy, "max_prob": max_prob}


def run_experiment(
    checkpoint_path: str,
    max_symbols: int = 64,
    decay_factors: List[float] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    reset_period: int = 10,
    frame_size: int = 160,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """运行硬重置实验"""
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Testing decay factors: {decay_factors}")
    print(f"Reset period: {reset_period} frames")
    
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
        "reset_period": reset_period,
        "decay_factors": decay_factors,
        "sequence_lengths": list(range(5, max_symbols + 1, 5)),
        "by_decay": {},
    }
    
    for decay in decay_factors:
        print(f"\n--- Testing decay_factor = {decay} ---")
        
        if decay == 1.0:
            # decay=1.0 等于无衰减，直接用原模型
            model = base_model
        else:
            model = HardResetMiniJMamba(base_model, reset_period, decay).to(device)
            model.eval()
        
        results["by_decay"][str(decay)] = {
            "hidden_norms": [],
            "entropies": [],
            "max_probs": [],
        }
        
        for num_symbols in results["sequence_lengths"]:
            metrics = measure_model(model, num_symbols, frame_size, device)
            results["by_decay"][str(decay)]["hidden_norms"].append(metrics["hidden_norm"])
            results["by_decay"][str(decay)]["entropies"].append(metrics["entropy"])
            results["by_decay"][str(decay)]["max_probs"].append(metrics["max_prob"])
            
            if num_symbols % 20 == 0:
                print(f"  {num_symbols} symbols: norm={metrics['hidden_norm']:.1f}, "
                      f"entropy={metrics['entropy']:.3f}, max_prob={metrics['max_prob']:.3f}")
    
    return results


def plot_results(results: Dict[str, Any], output_dir: Path):
    """绘制结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Hard Reset Experiment (reset_period={results["reset_period"]})', 
                 fontsize=14, fontweight='bold')
    
    lengths = results["sequence_lengths"]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results["decay_factors"])))
    
    # 隐状态范数
    ax1 = axes[0]
    for i, decay in enumerate(results["decay_factors"]):
        data = results["by_decay"][str(decay)]
        label = f'decay={decay}' if decay != 1.0 else 'No Reset (decay=1.0)'
        linestyle = '--' if decay == 1.0 else '-'
        linewidth = 2.5 if decay == 1.0 else 1.5
        ax1.plot(lengths, data["hidden_norms"], linestyle, color=colors[i], 
                linewidth=linewidth, marker='o', markersize=3, label=label)
    ax1.set_xlabel('Sequence Length (symbols)')
    ax1.set_ylabel('Hidden State Norm')
    ax1.set_title('Hidden Norm vs Decay Factor')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 范数比率（相对于 decay=1.0）
    ax2 = axes[1]
    baseline_norms = results["by_decay"]["1.0"]["hidden_norms"]
    for i, decay in enumerate(results["decay_factors"]):
        if decay == 1.0:
            continue
        data = results["by_decay"][str(decay)]
        ratios = [d / b if b > 0 else 0 for d, b in zip(data["hidden_norms"], baseline_norms)]
        ax2.plot(lengths, ratios, '-', color=colors[i], linewidth=1.5, 
                marker='s', markersize=3, label=f'decay={decay}')
    ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Sequence Length (symbols)')
    ax2.set_ylabel('Norm Ratio (vs No Reset)')
    ax2.set_title('Norm Reduction Ratio')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 分类置信度
    ax3 = axes[2]
    for i, decay in enumerate(results["decay_factors"]):
        data = results["by_decay"][str(decay)]
        label = f'decay={decay}' if decay != 1.0 else 'No Reset'
        linestyle = '--' if decay == 1.0 else '-'
        ax3.plot(lengths, data["max_probs"], linestyle, color=colors[i], 
                linewidth=1.5, marker='^', markersize=3, label=label)
    ax3.set_xlabel('Sequence Length (symbols)')
    ax3.set_ylabel('Max Probability')
    ax3.set_title('Classification Confidence')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hard_reset_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {output_dir / 'hard_reset_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Hard Reset Experiment')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--max-symbols', type=int, default=64)
    parser.add_argument('--reset-period', type=int, default=10)
    parser.add_argument('--decay-factors', type=str, default='0.0,0.1,0.3,0.5,0.7,0.9,1.0',
                        help='Comma-separated decay factors to test')
    parser.add_argument('--output-dir', type=str, default='reports/hard_reset')
    
    args = parser.parse_args()
    
    decay_factors = [float(x) for x in args.decay_factors.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_experiment(
        checkpoint_path=args.checkpoint,
        max_symbols=args.max_symbols,
        decay_factors=decay_factors,
        reset_period=args.reset_period,
    )
    
    # 保存结果
    with open(output_dir / 'hard_reset_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # 绘图
    plot_results(results, output_dir)
    
    # 摘要
    print("\n" + "="*70)
    print("HARD RESET EXPERIMENT SUMMARY")
    print("="*70)
    
    baseline_final = results["by_decay"]["1.0"]["hidden_norms"][-1]
    print(f"Baseline (no reset) final norm: {baseline_final:.1f}")
    
    best_decay = None
    best_reduction = 1.0
    
    for decay in results["decay_factors"]:
        if decay == 1.0:
            continue
        final_norm = results["by_decay"][str(decay)]["hidden_norms"][-1]
        reduction = final_norm / baseline_final
        print(f"  decay={decay}: final_norm={final_norm:.1f}, reduction={reduction:.3f}")
        
        if reduction < best_reduction:
            best_reduction = reduction
            best_decay = decay
    
    print(f"\nBest decay factor: {best_decay} (reduction={best_reduction:.3f})")
    
    if best_reduction < 0.5:
        print("[OK] Hard reset SIGNIFICANTLY reduces hidden state accumulation!")
    elif best_reduction < 0.9:
        print("[OK] Hard reset moderately reduces hidden state accumulation.")
    else:
        print("[--] Hard reset has minimal effect.")


if __name__ == '__main__':
    main()

