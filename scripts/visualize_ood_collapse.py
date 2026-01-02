#!/usr/bin/env python3
"""
OOD 长度外推崩溃可视化

对比 IID 长度 vs OOD 长度时的隐状态变化，揭示崩溃机制。

用法:
    python scripts/visualize_ood_collapse.py --checkpoint runs/ood_length_decay/mini_jmamba_seed42/mod_seed42_epoch200.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.models import MiniJMamba, MiniJMambaConfig


class HiddenStateExtractor:
    """提取每层隐状态"""
    
    def __init__(self, model: nn.Module):
        self.states = []
        self.hooks = []
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


def load_model(checkpoint_path: str, device: str = 'cpu') -> MiniJMamba:
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        config = MiniJMambaConfig(**checkpoint['config'])
    else:
        config = MiniJMambaConfig(d_model=128, num_ssm_layers=10, num_attn_layers=2, num_heads=4)
    
    model = MiniJMamba(config)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def synthesize_mod_expression(dividend: int, divisor: int) -> Tuple[np.ndarray, str, str]:
    """合成取模表达式的波形"""
    from jericho.symbols import encode_symbols_to_wave
    
    expression = f"{dividend}%{divisor}"
    remainder = str(dividend % divisor)
    
    # 合成输入波形（expression 是字符串，需要转为字符列表）
    wave = encode_symbols_to_wave(list(expression), sr=16000)
    
    return wave, expression, remainder


def extract_hidden_states(model: MiniJMamba, wave: np.ndarray, device: str) -> List[dict]:
    """提取隐状态"""
    frame_size = 160
    hop_size = 160
    
    num_frames = max(1, (len(wave) - frame_size) // hop_size + 1)
    frames = np.zeros((num_frames, frame_size), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_size
        end = min(start + frame_size, len(wave))
        frames[i, :end-start] = wave[start:end]
    
    frames_tensor = torch.tensor(frames).unsqueeze(0).to(device)
    mask = torch.ones(1, num_frames, dtype=torch.bool, device=device)
    
    extractor = HiddenStateExtractor(model)
    
    with torch.no_grad():
        model(frames_tensor, mask)
    
    states = extractor.states
    extractor.remove_hooks()
    
    return states


def visualize_ood_collapse(
    checkpoint_path: str,
    output_dir: str = "reports/ood_collapse_viz",
    device: str = "cpu"
):
    """可视化 OOD 崩溃"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    
    # 定义测试用例
    test_cases = [
        # IID: 2位数 % 1位数 = 1位数
        {"dividend": 29, "divisor": 7, "label": "IID (29%7=1)"},
        {"dividend": 85, "divisor": 9, "label": "IID (85%9=4)"},
        # OOD digits: 3位数 % 1位数 = 1位数 (输入更长，输出维度相同)
        {"dividend": 456, "divisor": 7, "label": "OOD_digits (456%7=1)"},
        {"dividend": 789, "divisor": 9, "label": "OOD_digits (789%9=6)"},
        # OOD length: 4位数 % 2位数 = 可能2位数 (输入更长，输出维度可能增加)
        {"dividend": 1234, "divisor": 17, "label": "OOD_len (1234%17=10)"},
        {"dividend": 5678, "divisor": 89, "label": "OOD_len (5678%89=75)"},
    ]
    
    all_trajectories = []
    all_norms = []
    
    for case in test_cases:
        print(f"\nProcessing: {case['label']}")
        wave, expr, remainder = synthesize_mod_expression(case['dividend'], case['divisor'])
        print(f"  Expression: {expr} = {remainder}, wave_len: {len(wave)}")
        
        states = extract_hidden_states(model, wave, device)
        
        # 提取最后一层隐状态
        final_hidden = states[-1]['hidden'][0]  # (T, D)
        
        # 计算每层的范数
        layer_norms = [np.linalg.norm(s['hidden'][0], axis=1).mean() for s in states]
        
        all_trajectories.append({
            'label': case['label'],
            'hidden': final_hidden,
            'n_frames': final_hidden.shape[0],
            'expression': expr,
            'remainder': remainder,
        })
        all_norms.append({
            'label': case['label'],
            'norms': layer_norms,
        })
    
    # ========== 图1: 隐状态轨迹对比 (PCA) ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 收集所有隐状态用于全局 PCA
    all_hidden = np.vstack([t['hidden'] for t in all_trajectories])
    pca = PCA(n_components=2)
    pca.fit(all_hidden)
    
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    
    for idx, (traj, color) in enumerate(zip(all_trajectories, colors)):
        ax = axes[idx]
        hidden_2d = pca.transform(traj['hidden'])
        
        # 绘制轨迹
        ax.plot(hidden_2d[:, 0], hidden_2d[:, 1], '-', color=color, alpha=0.5, linewidth=1)
        scatter = ax.scatter(
            hidden_2d[:, 0], hidden_2d[:, 1],
            c=range(len(hidden_2d)),
            cmap='coolwarm',
            s=30,
            alpha=0.8
        )
        
        # 标注起点和终点
        ax.scatter(hidden_2d[0, 0], hidden_2d[0, 1], c='blue', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(hidden_2d[-1, 0], hidden_2d[-1, 1], c='red', s=100, marker='X', label='End', zorder=5)
        
        ax.set_title(f"{traj['label']}\n{traj['expression']}={traj['remainder']}", fontsize=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Hidden State Trajectories: IID vs OOD", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "trajectory_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}/trajectory_comparison.png")
    
    # ========== 图2: 层间范数变化 ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(all_norms[0]['norms']))
    width = 0.12
    
    for idx, (norm_data, color) in enumerate(zip(all_norms, colors)):
        offset = (idx - len(all_norms)/2 + 0.5) * width
        bars = ax.bar(x + offset, norm_data['norms'], width, label=norm_data['label'], color=color, alpha=0.8)
    
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Hidden State Norm")
    ax.set_title("Hidden State Magnitude Across Layers: IID vs OOD")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in range(len(x))], fontsize=8)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / "layer_norms_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}/layer_norms_comparison.png")
    
    # ========== 图3: 终点位置分布 ==========
    fig, ax = plt.subplots(figsize=(10, 8))
    
    endpoints = []
    for traj in all_trajectories:
        hidden_2d = pca.transform(traj['hidden'])
        endpoints.append(hidden_2d[-1])
    endpoints = np.array(endpoints)
    
    # IID vs OOD 分组
    iid_mask = np.array([0, 1])  # 前两个是 IID
    ood_digits_mask = np.array([2, 3])  # 中间两个是 OOD digits
    ood_len_mask = np.array([4, 5])  # 后两个是 OOD length
    
    ax.scatter(endpoints[iid_mask, 0], endpoints[iid_mask, 1], 
               c='#2ecc71', s=200, marker='o', label='IID', edgecolors='black', linewidths=2)
    ax.scatter(endpoints[ood_digits_mask, 0], endpoints[ood_digits_mask, 1], 
               c='#f39c12', s=200, marker='s', label='OOD (digits)', edgecolors='black', linewidths=2)
    ax.scatter(endpoints[ood_len_mask, 0], endpoints[ood_len_mask, 1], 
               c='#e74c3c', s=200, marker='^', label='OOD (length)', edgecolors='black', linewidths=2)
    
    # 标注每个点
    for idx, traj in enumerate(all_trajectories):
        ax.annotate(
            f"{traj['remainder']}",
            endpoints[idx],
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=12,
            fontweight='bold'
        )
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.set_title("Final Hidden State Positions: Where Does OOD End Up?", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "endpoint_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}/endpoint_distribution.png")
    
    # ========== 图4: 时序范数演化 ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, (traj, color) in enumerate(zip(all_trajectories, colors)):
        hidden = traj['hidden']
        frame_norms = np.linalg.norm(hidden, axis=1)
        
        # 归一化时间轴到 [0, 1]
        t = np.linspace(0, 1, len(frame_norms))
        ax.plot(t, frame_norms, '-', color=color, linewidth=2, label=traj['label'], alpha=0.8)
    
    ax.set_xlabel("Normalized Time (0=Start, 1=End)", fontsize=12)
    ax.set_ylabel("Hidden State Norm", fontsize=12)
    ax.set_title("Hidden State Norm Evolution Over Time", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "temporal_norm_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}/temporal_norm_evolution.png")
    
    # 保存元数据
    meta = {
        'checkpoint': checkpoint_path,
        'test_cases': test_cases,
        'pca_variance_ratio': pca.explained_variance_ratio_.tolist(),
    }
    with open(output_path / "ood_collapse_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n[OK] OOD collapse visualization complete!")
    print(f"   Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize OOD length collapse")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/ood_length_decay/mini_jmamba_seed42/mod_seed42_epoch200.pt",
        help="Model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/ood_collapse_viz",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    args = parser.parse_args()
    visualize_ood_collapse(args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()

