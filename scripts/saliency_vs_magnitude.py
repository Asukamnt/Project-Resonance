#!/usr/bin/env python3
"""
Saliency-based Gate vs Magnitude Top-k 对比实验

比较两种选择性修剪策略：
1. Magnitude-based: 按 L2 范数排序，保留 top-k
2. Saliency-based: 按梯度敏感度排序，保留 top-k

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
from jericho.symbols import SYMBOL2FREQ, SR
from jericho.data.manifest import read_manifest


class MagnitudePruningModel(nn.Module):
    """基于幅值的选择性修剪"""
    
    def __init__(self, model: MiniJMamba, keep_ratio: float = 1.0):
        super().__init__()
        self.model = model
        self.keep_ratio = keep_ratio
    
    def forward(self, frames: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        x = self.model.input_proj(frames)
        x = self.model.dropout(x)
        
        for layer in self.model.layers:
            x = layer(x, padding_mask)
            
            if self.keep_ratio < 1.0:
                # 按幅值排序，保留 top-k
                k = int(x.shape[-1] * self.keep_ratio)
                if k > 0:
                    sorted_vals, _ = torch.sort(x.abs(), dim=-1, descending=True)
                    threshold = sorted_vals[:, :, k-1:k]
                    mask = (x.abs() >= threshold).float()
                    x = x * mask
        
        x = self.model.final_norm(x)
        symbol_logits = self.model.symbol_head(x)
        return symbol_logits


class SaliencyPruningModel(nn.Module):
    """基于梯度敏感度的选择性修剪"""
    
    def __init__(self, model: MiniJMamba, keep_ratio: float = 1.0):
        super().__init__()
        self.model = model
        self.keep_ratio = keep_ratio
        self.saliency_scores = None
    
    def compute_saliency(self, frames: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """计算每个隐藏维度的梯度敏感度"""
        frames = frames.clone().requires_grad_(True)
        
        x = self.model.input_proj(frames)
        x = self.model.dropout(x)
        
        saliency_per_layer = []
        for layer in self.model.layers:
            x = layer(x, padding_mask)
            x.retain_grad()
            saliency_per_layer.append(x)
        
        x_final = self.model.final_norm(x)
        symbol_logits = self.model.symbol_head(x_final)
        
        # 对输出求梯度
        loss = symbol_logits.sum()
        loss.backward(retain_graph=True)
        
        # 收集每层的梯度敏感度
        saliency = []
        for h in saliency_per_layer:
            if h.grad is not None:
                # 梯度 × 激活值 = 敏感度
                sal = (h.grad * h).abs().mean(dim=(0, 1))  # (d_model,)
                saliency.append(sal)
        
        if saliency:
            self.saliency_scores = torch.stack(saliency).mean(dim=0)  # 平均跨层
        else:
            self.saliency_scores = None
        
        return symbol_logits.detach()
    
    def forward(self, frames: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        x = self.model.input_proj(frames)
        x = self.model.dropout(x)
        
        for layer in self.model.layers:
            x = layer(x, padding_mask)
            
            if self.keep_ratio < 1.0 and self.saliency_scores is not None:
                # 按敏感度排序，保留 top-k
                k = int(x.shape[-1] * self.keep_ratio)
                if k > 0:
                    _, top_indices = torch.topk(self.saliency_scores, k, largest=True)
                    mask = torch.zeros(x.shape[-1], device=x.device)
                    mask[top_indices] = 1.0
                    x = x * mask.view(1, 1, -1)
        
        x = self.model.final_norm(x)
        symbol_logits = self.model.symbol_head(x)
        return symbol_logits


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
        max_frames=saved_config.get("max_frames", 256),
        use_rope=saved_config.get("use_rope", True),
    )
    
    model = MiniJMamba(config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, config


def generate_test_data(manifest_path: str, num_samples: int, config, device: str):
    """从 manifest 加载测试数据"""
    from jericho.symbols import encode_symbols_to_wave
    
    entries = list(read_manifest(Path(manifest_path)))[:num_samples]
    
    frames_list = []
    for entry in entries:
        symbols = list(entry.symbols)
        wave = encode_symbols_to_wave(symbols, tone_dur=0.01, sr=SR)
        
        frame_size = config.frame_size
        num_frames = len(wave) // frame_size
        if num_frames == 0:
            continue
        wave = wave[:num_frames * frame_size]
        frames = wave.reshape(num_frames, frame_size)
        frames_list.append(torch.tensor(frames, dtype=torch.float32))
    
    # Pad to same length
    max_len = max(f.shape[0] for f in frames_list)
    padded = []
    masks = []
    for f in frames_list:
        pad_len = max_len - f.shape[0]
        padded.append(torch.nn.functional.pad(f, (0, 0, 0, pad_len)))
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:f.shape[0]] = True
        masks.append(mask)
    
    frames = torch.stack(padded).to(device)
    masks = torch.stack(masks).to(device)
    
    return frames, masks


def run_comparison(
    checkpoint_path: str,
    manifest_path: str,
    keep_ratios: list,
    num_samples: int,
    device: str,
    output_dir: Path
):
    """运行对比实验"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"Loading model from {checkpoint_path}")
    base_model, config = load_model(checkpoint_path, device)
    
    # 加载测试数据
    print(f"Loading {num_samples} samples from {manifest_path}")
    frames, masks = generate_test_data(manifest_path, num_samples, config, device)
    print(f"Loaded {frames.shape[0]} samples, max_len={frames.shape[1]}")
    
    results = []
    
    for keep_ratio in keep_ratios:
        print(f"\n=== keep_ratio = {keep_ratio} ===")
        
        # Magnitude-based
        mag_model = MagnitudePruningModel(base_model, keep_ratio)
        with torch.no_grad():
            mag_logits = mag_model(frames, masks)
            mag_probs = torch.softmax(mag_logits, dim=-1)
            mag_max_prob = mag_probs.max(dim=-1).values.mean().item()
            mag_entropy = -(mag_probs * torch.log(mag_probs + 1e-10)).sum(dim=-1).mean().item()
        
        print(f"  Magnitude: prob={mag_max_prob:.4f}, entropy={mag_entropy:.4f}")
        
        # Saliency-based
        sal_model = SaliencyPruningModel(base_model, keep_ratio)
        
        # 先计算 saliency（需要梯度）
        base_model.train()  # 临时开启训练模式以允许梯度
        sal_model.compute_saliency(frames[:1], masks[:1])  # 用一个样本计算 saliency
        base_model.eval()
        
        with torch.no_grad():
            sal_logits = sal_model(frames, masks)
            sal_probs = torch.softmax(sal_logits, dim=-1)
            sal_max_prob = sal_probs.max(dim=-1).values.mean().item()
            sal_entropy = -(sal_probs * torch.log(sal_probs + 1e-10)).sum(dim=-1).mean().item()
        
        print(f"  Saliency:  prob={sal_max_prob:.4f}, entropy={sal_entropy:.4f}")
        
        results.append({
            "keep_ratio": keep_ratio,
            "magnitude_prob": mag_max_prob,
            "magnitude_entropy": mag_entropy,
            "saliency_prob": sal_max_prob,
            "saliency_entropy": sal_entropy,
            "delta_prob": sal_max_prob - mag_max_prob,
        })
    
    # 保存结果
    output_file = output_dir / "saliency_vs_magnitude.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": checkpoint_path,
            "manifest": manifest_path,
            "num_samples": num_samples,
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # 生成对比图
    plot_comparison(results, output_dir)
    
    return results


def plot_comparison(results: list, output_dir: Path):
    """生成对比图"""
    
    keep_ratios = [r["keep_ratio"] for r in results]
    mag_probs = [r["magnitude_prob"] for r in results]
    sal_probs = [r["saliency_prob"] for r in results]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = range(len(keep_ratios))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], mag_probs, width, label='Magnitude-based', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], sal_probs, width, label='Saliency-based', color='#e74c3c')
    
    ax.set_xlabel('Keep Ratio (k)')
    ax.set_ylabel('Classification Confidence')
    ax.set_title('Magnitude vs Saliency Pruning')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in keep_ratios])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标注
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    plot_path = output_dir / "saliency_vs_magnitude.png"
    fig.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Saliency vs Magnitude Pruning")
    parser.add_argument("--checkpoint", type=str, 
                       default="artifacts/checkpoints/mod_best_em0.75.pt")
    parser.add_argument("--manifest", type=str,
                       default="manifests/task3.jsonl")
    parser.add_argument("--keep-ratios", type=str, default="1.0,0.7,0.5,0.3")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="reports/saliency_comparison")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
    
    run_comparison(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        keep_ratios=keep_ratios,
        num_samples=args.num_samples,
        device=device,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

