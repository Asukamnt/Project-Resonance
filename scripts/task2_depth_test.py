#!/usr/bin/env python3
"""
Task2-Bracket 深度测试

测试不同序列深度（嵌套层数）对模型性能的影响。

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
from jericho.task2 import is_balanced


def load_model(checkpoint_path: str, device: str):
    """加载模型"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = ckpt["config"]
    state_dict = ckpt["model_state_dict"]
    
    config = MiniJMambaConfig(
        frame_size=saved_config.get("frame_size", 160),
        hop_size=saved_config.get("hop_size", 160),
        symbol_vocab_size=saved_config.get("symbol_vocab_size", 2),  # V, X
        d_model=saved_config.get("d_model", 128),
        num_ssm_layers=saved_config.get("num_ssm_layers", 10),
        num_attn_layers=saved_config.get("num_attn_layers", 2),
        max_frames=512,
        use_rope=saved_config.get("use_rope", True),
    )
    
    model = MiniJMamba(config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, config


def generate_bracket_sequence(depth: int, balanced: bool, seed: int) -> list:
    """生成指定深度的括号序列"""
    np.random.seed(seed)
    
    if balanced:
        # 生成平衡括号：嵌套到指定深度
        # 策略：随机选择嵌套或并列
        symbols = []
        current_depth = 0
        target_depth = depth
        
        while current_depth < target_depth:
            symbols.append("(")
            current_depth += 1
        
        # 随机添加更多括号对
        extra_pairs = np.random.randint(0, depth // 2 + 1)
        for _ in range(extra_pairs):
            idx = np.random.randint(0, len(symbols) + 1)
            symbols.insert(idx, "(")
            symbols.insert(idx + 1, ")")
        
        # 添加闭合括号
        for _ in range(current_depth):
            symbols.append(")")
        
    else:
        # 生成不平衡括号
        symbols = []
        opens = np.random.randint(depth, depth * 2)
        closes = np.random.randint(depth, depth * 2)
        
        for _ in range(opens):
            symbols.append("(")
        for _ in range(closes):
            symbols.append(")")
        
        np.random.shuffle(symbols)
        
        # 确保真的不平衡
        if is_balanced(symbols):
            symbols.append("(")
    
    return symbols


def test_depth(
    model: MiniJMamba,
    config: MiniJMambaConfig,
    depth: int,
    num_samples: int,
    device: str,
) -> dict:
    """测试特定深度"""
    
    correct = 0
    total = 0
    
    for i in range(num_samples):
        # 一半平衡，一半不平衡
        balanced = (i % 2 == 0)
        symbols = generate_bracket_sequence(depth, balanced, seed=42 + i)
        expected = is_balanced(symbols)
        
        # 编码为波形
        try:
            wave = encode_symbols_to_wave(symbols, tone_dur=0.01, sr=SR)
        except ValueError:
            continue
        
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
            logits, symbol_logits, _ = model(frames, padding_mask, return_hidden=True)
            # 取最后一个位置的预测
            pred_class = symbol_logits[0, -1, :].argmax().item()
            # 假设 0=V(平衡), 1=X(不平衡)
            predicted_balanced = (pred_class == 0)
            
            if predicted_balanced == expected:
                correct += 1
            total += 1
    
    return {
        "depth": depth,
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
    }


def run_test(
    checkpoint_path: str,
    depths: list,
    num_samples: int,
    device: str,
    output_dir: Path,
):
    """运行测试"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)
    
    results = []
    
    for depth in depths:
        print(f"\nDepth={depth}")
        result = test_depth(model, config, depth, num_samples, device)
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
    
    # 保存结果
    output_file = output_dir / "task2_depth_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": checkpoint_path,
            "depths": depths,
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # 生成图
    plot_results(results, output_dir)
    
    return results


def plot_results(results: list, output_dir: Path):
    """生成图"""
    
    depths = [r["depth"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(depths, accuracies, 'o-', color='#3498db', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random baseline')
    
    for d, acc in zip(depths, accuracies):
        ax.annotate(f'{acc:.1%}', xy=(d, acc), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=10)
    
    ax.set_xlabel('Bracket Depth (nesting level)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Task2-Bracket: Performance vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    plot_path = output_dir / "task2_depth_plot.png"
    fig.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Task2-Bracket Depth Test")
    parser.add_argument("--checkpoint", type=str,
                       default="runs/mini_jmamba_bracket_20260101-201407/bracket_seed42_epoch30.pt")
    parser.add_argument("--depths", type=str, default="4,8,16,32,64")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="reports/task2_depth")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    depths = [int(x) for x in args.depths.split(",")]
    
    run_test(
        checkpoint_path=args.checkpoint,
        depths=depths,
        num_samples=args.num_samples,
        device=device,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

