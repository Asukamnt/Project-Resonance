#!/usr/bin/env python3
"""
诊断 SSM 稳定半径实验（简化版）

核心假设：SSM 递归系统存在"稳定半径"，超过后相位误差累积导致流形破碎。

Author: Jericho Team
Date: 2026-01-02
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig
from jericho.symbols import SR, SYMBOL2FREQ, encode_symbols_to_wave


def generate_synthetic_sequence(num_symbols: int, tone_dur: float = 0.1) -> torch.Tensor:
    """生成指定长度的合成波形序列"""
    symbols = list(SYMBOL2FREQ.keys())[:10]  # 使用数字 0-9
    symbol_seq = [symbols[i % len(symbols)] for i in range(num_symbols)]
    full_wave = encode_symbols_to_wave(symbol_seq, tone_dur=tone_dur, gap_dur=0.0)
    return torch.tensor(full_wave, dtype=torch.float32)


def analyze_stability(
    checkpoint_path: str,
    max_symbols: int = 64,
    frame_size: int = 160,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """分析 SSM 稳定性"""
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Device: {device}")
    
    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 构建默认配置
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
        'sequence_lengths': [],
        'output_norms': [],
        'hidden_norms': [],
    }
    
    print(f"\nAnalyzing stability for sequence lengths 1 to {max_symbols}...")
    
    for num_symbols in range(1, max_symbols + 1):
        # 生成序列
        wave = generate_synthetic_sequence(num_symbols)
        
        # 分帧
        num_frames = len(wave) // frame_size
        if num_frames == 0:
            continue
            
        frames = wave[:num_frames * frame_size].reshape(1, num_frames, frame_size)
        frames = frames.to(device)
        
        # 前向传播
        with torch.no_grad():
            # 创建 padding_mask (全 True，表示所有帧都有效)
            padding_mask = torch.ones(1, num_frames, dtype=torch.bool, device=device)
            output, symbol_logits, hidden = model(frames, padding_mask, return_hidden=True)
            
            # 计算输出范数
            output_norm = output.norm().item()
            
            # 计算隐状态范数
            hidden_norm = hidden.norm().item() if hidden is not None else 0.0
            
            results['sequence_lengths'].append(num_symbols)
            results['output_norms'].append(output_norm)
            results['hidden_norms'].append(hidden_norm)
        
        if num_symbols % 10 == 0:
            print(f"  Processed {num_symbols}/{max_symbols} symbols, output_norm={output_norm:.2f}, hidden_norm={hidden_norm:.2f}")
    
    # 计算增长率
    if len(results['output_norms']) > 1:
        results['output_growth_rates'] = []
        for i in range(1, len(results['output_norms'])):
            prev = results['output_norms'][i-1]
            curr = results['output_norms'][i]
            growth = (curr - prev) / prev if prev > 0 else 0
            results['output_growth_rates'].append(growth)
        
        # 找拐点
        growth_rates = np.array(results['output_growth_rates'])
        growth_diff = np.diff(growth_rates)
        if len(growth_diff) > 0:
            inflection_idx = np.argmax(np.abs(growth_diff)) + 2
            if inflection_idx < len(results['sequence_lengths']):
                results['inflection_point'] = results['sequence_lengths'][inflection_idx]
            else:
                results['inflection_point'] = None
        else:
            results['inflection_point'] = None
    
    return results


def plot_stability_analysis(results: Dict[str, Any], output_dir: Path):
    """绘制稳定性分析图"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SSM Stability Radius Analysis', fontsize=16, fontweight='bold')
    
    lengths = results['sequence_lengths']
    
    # 1. 输出范数 vs 序列长度
    ax1 = axes[0, 0]
    ax1.plot(lengths, results['output_norms'], 'b-', linewidth=2, marker='o', markersize=3)
    if results.get('inflection_point'):
        ax1.axvline(x=results['inflection_point'], color='r', linestyle='--', 
                    linewidth=2, label=f'Inflection @ {results["inflection_point"]}')
        ax1.legend()
    ax1.set_xlabel('Sequence Length (symbols)', fontsize=12)
    ax1.set_ylabel('Output Norm', fontsize=12)
    ax1.set_title('Output Norm vs Sequence Length', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 2. 隐状态范数
    ax2 = axes[0, 1]
    ax2.plot(lengths, results['hidden_norms'], 'g-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Sequence Length (symbols)', fontsize=12)
    ax2.set_ylabel('Hidden State Norm', fontsize=12)
    ax2.set_title('Hidden State Norm vs Sequence Length', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 3. 增长率
    ax3 = axes[1, 0]
    if 'output_growth_rates' in results and results['output_growth_rates']:
        ax3.plot(lengths[2:], results['output_growth_rates'][1:], 'r-', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Sequence Length (symbols)', fontsize=12)
    ax3.set_ylabel('Output Norm Growth Rate', fontsize=12)
    ax3.set_title('Norm Growth Rate (stability indicator)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. 对数刻度 + 区域标注
    ax4 = axes[1, 1]
    ax4.semilogy(lengths, results['output_norms'], 'b-', linewidth=2)
    ax4.semilogy(lengths, results['hidden_norms'], 'g--', linewidth=2, label='Hidden')
    
    # 标注稳定/不稳定区域
    if results.get('inflection_point'):
        ip = results['inflection_point']
        ax4.axvline(x=ip, color='r', linestyle=':', linewidth=2)
        ax4.text(ip/2, max(results['output_norms'])*0.8, 'STABLE', 
                 fontsize=12, ha='center', color='green', fontweight='bold')
        ax4.text((ip + max(lengths))/2, max(results['output_norms'])*0.8, 'UNSTABLE', 
                 fontsize=12, ha='center', color='red', fontweight='bold')
    
    ax4.set_xlabel('Sequence Length (symbols)', fontsize=12)
    ax4.set_ylabel('Norm (log scale)', fontsize=12)
    ax4.set_title('Log-Scale Analysis with Stability Regions', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ssm_stability_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {output_dir / 'ssm_stability_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose SSM Stability Radius')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--max-symbols', type=int, default=64,
                        help='Maximum sequence length to test')
    parser.add_argument('--frame-size', type=int, default=160,
                        help='Frame size for waveform processing')
    parser.add_argument('--output-dir', type=str, default='reports/ssm_stability',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行分析
    results = analyze_stability(
        checkpoint_path=args.checkpoint,
        max_symbols=args.max_symbols,
        frame_size=args.frame_size,
        device=device
    )
    
    # 保存结果
    results_file = output_dir / 'stability_analysis.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {results_file}")
    
    # 绘图
    plot_stability_analysis(results, output_dir)
    
    # 打印摘要
    print("\n" + "="*60)
    print("SSM STABILITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Sequence lengths tested: {min(results['sequence_lengths'])} - {max(results['sequence_lengths'])}")
    print(f"Output norm range: {min(results['output_norms']):.4f} - {max(results['output_norms']):.4f}")
    print(f"Hidden norm range: {min(results['hidden_norms']):.4f} - {max(results['hidden_norms']):.4f}")
    
    if results.get('inflection_point'):
        print(f"\n[!] INFLECTION POINT DETECTED at {results['inflection_point']} symbols")
        print(f"    This may indicate the STABILITY RADIUS of the SSM.")
    else:
        print("\n[OK] No clear inflection point detected in tested range.")
    
    # 计算线性拟合斜率
    if len(results['sequence_lengths']) > 2:
        z = np.polyfit(results['sequence_lengths'], results['output_norms'], 1)
        print(f"\nLinear fit slope: {z[0]:.6f} (norm per symbol)")
        if z[0] > 0.1:
            print("  → Positive slope suggests norm accumulation (potential instability)")
        elif z[0] < -0.1:
            print("  → Negative slope suggests norm decay (stable but may forget)")
        else:
            print("  → Near-zero slope suggests good stability")


if __name__ == '__main__':
    main()
