#!/usr/bin/env python3
"""
Reset Token 实验：验证周期性重置是否能改善长序列稳定性

核心假设：在序列中周期性插入"重置"静默段，可以让 SSM 状态回归稳态，
从而改善长序列的分类性能。

Author: Jericho Team
Date: 2026-01-02
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig
from jericho.symbols import SR, SYMBOL2FREQ, encode_symbols_to_wave
from jericho.scorer import decode_wave_to_symbols


def generate_long_sequence(num_symbols: int, tone_dur: float = 0.1) -> Tuple[torch.Tensor, List[str]]:
    """生成长序列波形"""
    symbols = list(SYMBOL2FREQ.keys())[:10]
    symbol_seq = [symbols[i % len(symbols)] for i in range(num_symbols)]
    full_wave = encode_symbols_to_wave(symbol_seq, tone_dur=tone_dur, gap_dur=0.0)
    return torch.tensor(full_wave, dtype=torch.float32), symbol_seq


def inject_reset_tokens(
    wave: torch.Tensor, 
    frame_size: int,
    reset_period: int = 10,
    reset_frames: int = 3,
) -> torch.Tensor:
    """在波形中周期性插入静默重置帧
    
    Args:
        wave: 原始波形
        frame_size: 帧大小
        reset_period: 每隔多少帧插入一次重置
        reset_frames: 每次插入的静默帧数
    
    Returns:
        带有重置帧的波形
    """
    num_frames = len(wave) // frame_size
    frames = wave[:num_frames * frame_size].reshape(num_frames, frame_size)
    
    new_frames = []
    for i, frame in enumerate(frames):
        new_frames.append(frame)
        if (i + 1) % reset_period == 0:
            # 插入静默帧（零值）
            for _ in range(reset_frames):
                new_frames.append(torch.zeros(frame_size))
    
    return torch.stack(new_frames).reshape(-1)


def compute_classification_quality(
    model: MiniJMamba,
    wave: torch.Tensor,
    frame_size: int,
    device: str,
) -> Dict[str, float]:
    """计算分类质量指标"""
    num_frames = len(wave) // frame_size
    if num_frames == 0:
        return {"hidden_norm": 0, "logits_entropy": 0, "max_prob": 0}
    
    frames = wave[:num_frames * frame_size].reshape(1, num_frames, frame_size).to(device)
    padding_mask = torch.ones(1, num_frames, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        output, symbol_logits, hidden = model(frames, padding_mask, return_hidden=True)
        
        # 隐状态范数
        hidden_norm = hidden.norm().item()
        
        # 分类熵（低熵 = 高置信度）
        probs = torch.softmax(symbol_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        
        # 最大概率（高 = 更确定）
        max_prob = probs.max(dim=-1).values.mean().item()
        
    return {
        "hidden_norm": hidden_norm,
        "logits_entropy": entropy,
        "max_prob": max_prob,
    }


def run_experiment(
    checkpoint_path: str,
    max_symbols: int = 64,
    reset_period: int = 10,
    reset_frames: int = 3,
    frame_size: int = 160,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """运行 Reset Token 实验"""
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Reset period: {reset_period} frames, Reset duration: {reset_frames} frames")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = MiniJMambaConfig(
        frame_size=frame_size,
        hop_size=frame_size,
        symbol_vocab_size=12,
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
    )
    
    model = MiniJMamba(config).to(device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    results = {
        "reset_period": reset_period,
        "reset_frames": reset_frames,
        "no_reset": {"sequence_lengths": [], "hidden_norms": [], "entropies": [], "max_probs": []},
        "with_reset": {"sequence_lengths": [], "hidden_norms": [], "entropies": [], "max_probs": []},
    }
    
    print(f"\nComparing with and without reset tokens...")
    
    for num_symbols in range(5, max_symbols + 1, 5):
        wave, symbols = generate_long_sequence(num_symbols)
        
        # 无重置
        metrics_no_reset = compute_classification_quality(model, wave, frame_size, device)
        results["no_reset"]["sequence_lengths"].append(num_symbols)
        results["no_reset"]["hidden_norms"].append(metrics_no_reset["hidden_norm"])
        results["no_reset"]["entropies"].append(metrics_no_reset["logits_entropy"])
        results["no_reset"]["max_probs"].append(metrics_no_reset["max_prob"])
        
        # 有重置
        wave_with_reset = inject_reset_tokens(wave, frame_size, reset_period, reset_frames)
        metrics_with_reset = compute_classification_quality(model, wave_with_reset, frame_size, device)
        results["with_reset"]["sequence_lengths"].append(num_symbols)
        results["with_reset"]["hidden_norms"].append(metrics_with_reset["hidden_norm"])
        results["with_reset"]["entropies"].append(metrics_with_reset["logits_entropy"])
        results["with_reset"]["max_probs"].append(metrics_with_reset["max_prob"])
        
        print(f"  {num_symbols} symbols: no_reset_norm={metrics_no_reset['hidden_norm']:.1f}, "
              f"with_reset_norm={metrics_with_reset['hidden_norm']:.1f}, "
              f"ratio={metrics_with_reset['hidden_norm']/metrics_no_reset['hidden_norm']:.2f}")
    
    return results


def plot_comparison(results: Dict[str, Any], output_dir: Path):
    """绘制对比图"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Reset Token Experiment (period={results["reset_period"]}, frames={results["reset_frames"]})', 
                 fontsize=14, fontweight='bold')
    
    lengths = results["no_reset"]["sequence_lengths"]
    
    # 隐状态范数
    ax1 = axes[0]
    ax1.plot(lengths, results["no_reset"]["hidden_norms"], 'r-', linewidth=2, marker='o', label='No Reset')
    ax1.plot(lengths, results["with_reset"]["hidden_norms"], 'g-', linewidth=2, marker='s', label='With Reset')
    ax1.set_xlabel('Sequence Length (symbols)')
    ax1.set_ylabel('Hidden State Norm')
    ax1.set_title('Hidden State Norm Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 分类熵
    ax2 = axes[1]
    ax2.plot(lengths, results["no_reset"]["entropies"], 'r-', linewidth=2, marker='o', label='No Reset')
    ax2.plot(lengths, results["with_reset"]["entropies"], 'g-', linewidth=2, marker='s', label='With Reset')
    ax2.set_xlabel('Sequence Length (symbols)')
    ax2.set_ylabel('Logits Entropy (lower = more confident)')
    ax2.set_title('Classification Entropy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 最大概率
    ax3 = axes[2]
    ax3.plot(lengths, results["no_reset"]["max_probs"], 'r-', linewidth=2, marker='o', label='No Reset')
    ax3.plot(lengths, results["with_reset"]["max_probs"], 'g-', linewidth=2, marker='s', label='With Reset')
    ax3.set_xlabel('Sequence Length (symbols)')
    ax3.set_ylabel('Max Probability (higher = more confident)')
    ax3.set_title('Classification Confidence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reset_token_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {output_dir / 'reset_token_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Reset Token Experiment')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--max-symbols', type=int, default=64)
    parser.add_argument('--reset-period', type=int, default=10,
                        help='Insert reset every N frames')
    parser.add_argument('--reset-frames', type=int, default=3,
                        help='Number of silence frames per reset')
    parser.add_argument('--output-dir', type=str, default='reports/reset_token')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_experiment(
        checkpoint_path=args.checkpoint,
        max_symbols=args.max_symbols,
        reset_period=args.reset_period,
        reset_frames=args.reset_frames,
    )
    
    # 保存结果
    with open(output_dir / 'reset_token_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # 绘图
    plot_comparison(results, output_dir)
    
    # 摘要
    print("\n" + "="*60)
    print("RESET TOKEN EXPERIMENT SUMMARY")
    print("="*60)
    
    no_reset_norms = results["no_reset"]["hidden_norms"]
    with_reset_norms = results["with_reset"]["hidden_norms"]
    
    avg_ratio = np.mean([w/n for w, n in zip(with_reset_norms, no_reset_norms)])
    print(f"Average norm ratio (with_reset / no_reset): {avg_ratio:.3f}")
    
    if avg_ratio < 0.9:
        print("[OK] Reset tokens REDUCE hidden state accumulation!")
    elif avg_ratio > 1.1:
        print("[!] Reset tokens INCREASE hidden state (unexpected)")
    else:
        print("[--] Reset tokens have minimal effect on norms")


if __name__ == '__main__':
    main()

