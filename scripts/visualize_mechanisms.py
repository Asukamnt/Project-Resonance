#!/usr/bin/env python3
"""
机制可视化脚本

产出：
1. SSM 隐状态轨迹（PCA/t-SNE）
2. Attention 热力图

用法:
    python scripts/visualize_mechanisms.py --checkpoint artifacts/checkpoints/task3_mod_best.pt
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.symbols import SYMBOL2FREQ


class AttentionExtractor:
    """Hook-based attention weight extractor"""
    
    def __init__(self, model: nn.Module):
        self.attentions = []
        self.hooks = []
        self._register_hooks(model)
    
    def _register_hooks(self, model: nn.Module):
        """Register forward hooks on attention layers"""
        for name, module in model.named_modules():
            if 'AttentionBlock' in type(module).__name__:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
    
    def _hook_fn(self, name: str):
        def hook(module, input, output):
            # 我们需要在forward中计算attention weights
            # 由于原forward不返回weights，我们重新计算
            x = input[0]
            batch_size, seq_len, _ = x.shape
            y = module.norm1(x)
            
            q = module.q_proj(y).view(batch_size, seq_len, module.num_heads, module.head_dim)
            k = module.k_proj(y).view(batch_size, seq_len, module.num_heads, module.head_dim)
            
            if module.rope is not None:
                cos, sin = module.rope(seq_len, x.device)
                from jericho.models.mini_jmamba import apply_rotary_pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / module.scale
            attn_weights = torch.softmax(attn_weights, dim=-1)
            
            self.attentions.append({
                'name': name,
                'weights': attn_weights.detach().cpu().numpy()
            })
        return hook
    
    def clear(self):
        self.attentions = []
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class HiddenStateExtractor:
    """Hook-based hidden state extractor for each layer"""
    
    def __init__(self, model: nn.Module):
        self.states = []
        self.hooks = []
        self._register_hooks(model)
    
    def _register_hooks(self, model: nn.Module):
        for idx, layer in enumerate(model.layers):
            hook = layer.register_forward_hook(self._hook_fn(idx, type(layer).__name__))
            self.hooks.append(hook)
    
    def _hook_fn(self, layer_idx: int, layer_type: str):
        def hook(module, input, output):
            self.states.append({
                'layer_idx': layer_idx,
                'layer_type': layer_type,
                'hidden': output.detach().cpu().numpy()
            })
        return hook
    
    def clear(self):
        self.states = []
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def load_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[MiniJMamba, dict]:
    """加载模型和配置"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 尝试从checkpoint恢复配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = MiniJMambaConfig(**config_dict)
    else:
        # 默认配置
        config = MiniJMambaConfig(
            d_model=128,
            num_ssm_layers=10,
            num_attn_layers=2,
            num_heads=4,
        )
    
    model = MiniJMamba(config)
    
    # 加载权重
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    return model, checkpoint


def visualize_attention(attentions: List[dict], symbols: List[str], save_path: str):
    """可视化 Attention 热力图"""
    n_layers = len(attentions)
    if n_layers == 0:
        print("No attention layers found")
        return
    
    # 每个attention layer一张图
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]
    
    for idx, attn_data in enumerate(attentions):
        weights = attn_data['weights']  # (B, H, T, T)
        # 取第一个样本，对所有头取平均
        avg_weights = weights[0].mean(axis=0)  # (T, T)
        
        ax = axes[idx]
        im = ax.imshow(avg_weights, cmap='viridis', aspect='auto')
        ax.set_title(f"Attention Layer {idx + 1}\n({attn_data['name']})")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        
        # 添加符号标注（如果序列不太长）
        if len(symbols) <= 20:
            ax.set_xticks(range(0, len(symbols), 2))
            ax.set_xticklabels(symbols[::2], fontsize=6, rotation=45)
            ax.set_yticks(range(0, len(symbols), 2))
            ax.set_yticklabels(symbols[::2], fontsize=6)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention heatmap saved: {save_path}")


def visualize_hidden_trajectory(states: List[dict], symbols: List[str], save_path: str):
    """可视化隐状态轨迹（PCA降维）"""
    from sklearn.decomposition import PCA
    
    # 收集所有层的最后隐状态
    layer_names = []
    layer_states = []
    
    for state in states:
        hidden = state['hidden'][0]  # (T, D) - 取第一个样本
        layer_names.append(f"L{state['layer_idx']}_{state['layer_type'][:3]}")
        layer_states.append(hidden)
    
    # 选择最后一层做时间轨迹
    final_hidden = layer_states[-1]  # (T, D)
    
    # PCA降维到2D
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(final_hidden)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：时间轨迹
    ax1 = axes[0]
    scatter = ax1.scatter(
        trajectory_2d[:, 0], trajectory_2d[:, 1],
        c=range(len(trajectory_2d)),
        cmap='coolwarm',
        s=50,
        alpha=0.7
    )
    
    # 连接轨迹
    ax1.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'k-', alpha=0.3, linewidth=0.5)
    
    # 标注起点和终点
    ax1.annotate('START', trajectory_2d[0], fontsize=8, color='blue')
    ax1.annotate('END', trajectory_2d[-1], fontsize=8, color='red')
    
    # 标注一些关键符号位置
    if len(symbols) <= 30:
        for i, sym in enumerate(symbols):
            if sym in ['%', '=', ';'] or i % 5 == 0:
                ax1.annotate(
                    f"{sym}@{i}",
                    trajectory_2d[i],
                    fontsize=6,
                    alpha=0.7
                )
    
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_title("Hidden State Trajectory (Final Layer)")
    plt.colorbar(scatter, ax=ax1, label="Time Step")
    
    # 右图：层间隐状态变化
    ax2 = axes[1]
    
    # 计算每层隐状态的范数变化
    layer_norms = []
    for hidden in layer_states:
        norm = np.linalg.norm(hidden, axis=1).mean()
        layer_norms.append(norm)
    
    colors = ['#2ecc71' if 'SSM' in name else '#e74c3c' for name in layer_names]
    bars = ax2.bar(range(len(layer_norms)), layer_norms, color=colors, alpha=0.7)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Mean Hidden State Norm")
    ax2.set_title("Hidden State Magnitude Across Layers")
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(layer_names, rotation=45, fontsize=7)
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='SSM Layer'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Attention Layer')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Hidden trajectory saved: {save_path}")


def run_visualization(
    checkpoint_path: str,
    manifest_path: str,
    output_dir: str,
    sample_idx: int = 0,
    device: str = 'cpu'
):
    """运行完整的可视化流程"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {checkpoint_path}...")
    model, checkpoint = load_model(checkpoint_path, device)
    
    print(f"Loading manifest from {manifest_path}...")
    entries = read_manifest(manifest_path)
    
    # 筛选 iid_test 样本
    test_entries = [e for e in entries if e.split == 'iid_test']
    if not test_entries:
        test_entries = entries[:10]
    
    entry = test_entries[min(sample_idx, len(test_entries) - 1)]
    print(f"Using sample: {entry.symbols}")
    
    # 合成波形
    wave = synthesise_entry_wave(entry)
    
    # 准备输入 - 手动帧化
    frame_size = 160
    hop_size = 160
    num_frames = (len(wave) - frame_size) // hop_size + 1
    if num_frames <= 0:
        frames = np.zeros((1, frame_size), dtype=np.float32)
        frames[0, :len(wave)] = wave
    else:
        frames = np.zeros((num_frames, frame_size), dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_size
            frames[i] = wave[start:start + frame_size]
    
    frames_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.ones(1, frames_tensor.shape[1], dtype=torch.bool, device=device)
    
    # 注册提取器
    attn_extractor = AttentionExtractor(model)
    hidden_extractor = HiddenStateExtractor(model)
    
    # 前向传播
    with torch.no_grad():
        model(frames_tensor, mask)
    
    # 提取符号用于标注
    symbols = list(entry.symbols)
    # 由于帧化可能改变长度，需要对齐
    n_frames = frames_tensor.shape[1]
    if len(symbols) < n_frames:
        # 扩展符号列表（每个符号对应多帧）
        frames_per_symbol = n_frames // len(symbols)
        expanded_symbols = []
        for s in symbols:
            expanded_symbols.extend([s] * frames_per_symbol)
        # 补齐剩余帧
        while len(expanded_symbols) < n_frames:
            expanded_symbols.append('')
        symbols = expanded_symbols
    
    # 可视化 Attention
    if attn_extractor.attentions:
        visualize_attention(
            attn_extractor.attentions,
            symbols,
            str(output_path / "attention_heatmap.png")
        )
    
    # 可视化隐状态轨迹
    if hidden_extractor.states:
        visualize_hidden_trajectory(
            hidden_extractor.states,
            symbols,
            str(output_path / "hidden_trajectory.png")
        )
    
    # 清理
    attn_extractor.remove_hooks()
    hidden_extractor.remove_hooks()
    
    # 保存元信息
    meta = {
        'checkpoint': checkpoint_path,
        'manifest': manifest_path,
        'sample_symbols': entry.symbols,
        'n_frames': n_frames,
        'n_attention_layers': len(attn_extractor.attentions),
        'n_total_layers': len(hidden_extractor.states),
    }
    with open(output_path / "visualization_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nVisualization complete! Output: {output_path}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Visualize model mechanisms")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/task3_mod_best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifests/task3_tiny_disjoint.jsonl",
        help="Path to manifest file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/mechanism_viz",
        help="Output directory"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index to visualize"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda)"
    )
    
    args = parser.parse_args()
    
    # 设置UTF-8输出
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    run_visualization(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output,
        sample_idx=args.sample,
        device=args.device
    )


if __name__ == "__main__":
    main()

