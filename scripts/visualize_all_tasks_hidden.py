#!/usr/bin/env python3
"""
多任务 + 跨域隐状态可视化（含动画）

生成：
1. 各任务隐状态轨迹对比
2. 跨域迁移隐状态对比
3. **动态 GIF 动画**展示隐状态随时间移动

用法:
    python scripts/visualize_all_tasks_hidden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.symbols import encode_symbols_to_wave


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
    
    # 默认配置
    default_config = {
        'frame_size': 160,
        'hop_size': 160,
        'symbol_vocab_size': 12,
        'd_model': 128,
        'num_ssm_layers': 10,
        'num_attn_layers': 2,
        'num_heads': 4,
        'max_frames': 115,
        'dropout': 0.1,
        'attn_dropout': 0.1,
        'use_rope': True,
        'use_learnable_pos': False,
    }
    
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        # 合并：checkpoint 的值覆盖默认值
        import inspect
        valid_params = set(inspect.signature(MiniJMambaConfig.__init__).parameters.keys()) - {'self'}
        merged_config = {**default_config}
        for k, v in config_dict.items():
            if k in valid_params:
                merged_config[k] = v
        config = MiniJMambaConfig(**{k: v for k, v in merged_config.items() if k in valid_params})
    else:
        config = MiniJMambaConfig(**default_config)
    
    model = MiniJMamba(config)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def synthesize_wave(symbols: str, domain: str = 'audio') -> np.ndarray:
    """合成波形"""
    sym_list = list(symbols)
    
    if domain == 'audio':
        wave = encode_symbols_to_wave(sym_list, sr=16000)
    elif domain == 'ipd':
        # IPD 域使用不同频率范围
        from jericho.symbols import encode_symbols_to_wave as encode_audio
        wave = encode_audio(sym_list, sr=16000)
        # 模拟 IPD 编码（简化：使用不同的频率映射）
    else:
        wave = encode_symbols_to_wave(sym_list, sr=16000)
    
    return wave


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


def create_trajectory_animation(
    trajectories: List[dict],
    pca: PCA,
    output_path: str,
    fps: int = 10
):
    """创建隐状态轨迹动画 GIF"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 颜色方案
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    # 预计算所有 2D 轨迹
    trajectories_2d = []
    max_len = 0
    for traj in trajectories:
        hidden_2d = pca.transform(traj['hidden'])
        trajectories_2d.append(hidden_2d)
        max_len = max(max_len, len(hidden_2d))
    
    # 计算边界
    all_points = np.vstack(trajectories_2d)
    x_min, x_max = all_points[:, 0].min() - 5, all_points[:, 0].max() + 5
    y_min, y_max = all_points[:, 1].min() - 5, all_points[:, 1].max() + 5
    
    # 初始化绑定元素
    lines = []
    points = []
    trails = []
    
    for i, (traj, color) in enumerate(zip(trajectories, colors)):
        # 轨迹线（淡色）
        line, = ax.plot([], [], '-', color=color, alpha=0.3, linewidth=1)
        lines.append(line)
        # 尾迹（渐变）
        trail, = ax.plot([], [], '-', color=color, alpha=0.6, linewidth=2)
        trails.append(trail)
        # 当前点
        point, = ax.plot([], [], 'o', color=color, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
        points.append(point)
    
    # 时间标签
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14, 
                        verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.set_title("Hidden State Trajectory Animation", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 图例
    for i, traj in enumerate(trajectories):
        ax.plot([], [], 'o-', color=colors[i], label=traj['label'], markersize=8)
    ax.legend(loc='upper right', fontsize=9)
    
    def init():
        for line, trail, point in zip(lines, trails, points):
            line.set_data([], [])
            trail.set_data([], [])
            point.set_data([], [])
        time_text.set_text('')
        return lines + trails + points + [time_text]
    
    def animate(frame):
        for i, (traj_2d, line, trail, point) in enumerate(zip(trajectories_2d, lines, trails, points)):
            n = len(traj_2d)
            # 当前帧对应的时间步（归一化到各轨迹长度）
            t = int(frame / max_len * n)
            t = min(t, n - 1)
            
            # 完整轨迹（淡色）
            line.set_data(traj_2d[:t+1, 0], traj_2d[:t+1, 1])
            
            # 尾迹（最近 5 帧）
            trail_start = max(0, t - 5)
            trail.set_data(traj_2d[trail_start:t+1, 0], traj_2d[trail_start:t+1, 1])
            
            # 当前点
            point.set_data([traj_2d[t, 0]], [traj_2d[t, 1]])
        
        progress = frame / max_len * 100
        time_text.set_text(f'Progress: {progress:.0f}%')
        
        return lines + trails + points + [time_text]
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=max_len, interval=1000//fps, blit=True
    )
    
    # 保存 GIF
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Animation saved: {output_path}")


def visualize_all_tasks(
    output_dir: str = "reports/all_tasks_hidden_viz",
    device: str = "cuda"
):
    """可视化所有任务的隐状态"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========== 任务定义 ==========
    tasks = [
        # Task3: Mod (取模) - IID vs OOD 对比
        {
            "name": "Task3_Mod_IID",
            "samples": [
                {"symbols": "29%7", "label": "29%7=1"},
                {"symbols": "85%9", "label": "85%9=4"},
                {"symbols": "17%3", "label": "17%3=2"},
                {"symbols": "42%5", "label": "42%5=2"},
            ],
            "checkpoint": "runs/ood_length_decay/mini_jmamba_seed42/mod_seed42_epoch200.pt",
        },
        {
            "name": "Task3_Mod_OOD_digits",
            "samples": [
                {"symbols": "456%7", "label": "456%7=1"},
                {"symbols": "789%9", "label": "789%9=6"},
                {"symbols": "234%5", "label": "234%5=4"},
            ],
            "checkpoint": "runs/ood_length_decay/mini_jmamba_seed42/mod_seed42_epoch200.pt",
        },
        {
            "name": "Task3_Mod_OOD_length",
            "samples": [
                {"symbols": "1234%17", "label": "1234%17=10"},
                {"symbols": "5678%89", "label": "5678%89=71"},
                {"symbols": "9999%99", "label": "9999%99=0"},
            ],
            "checkpoint": "runs/ood_length_decay/mini_jmamba_seed42/mod_seed42_epoch200.pt",
        },
    ]
    
    all_trajectories = []
    
    for task in tasks:
        ckpt_path = Path(task["checkpoint"])
        if not ckpt_path.exists():
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue
        
        print(f"\n=== {task['name']} ===")
        model = load_model(str(ckpt_path), device)
        
        for sample in task["samples"]:
            print(f"  Processing: {sample['label']}")
            wave = synthesize_wave(sample["symbols"])
            states = extract_hidden_states(model, wave, device)
            
            final_hidden = states[-1]['hidden'][0]  # (T, D)
            all_trajectories.append({
                'task': task['name'],
                'label': sample['label'],
                'hidden': final_hidden,
                'n_frames': final_hidden.shape[0],
            })
    
    if not all_trajectories:
        print("No trajectories collected!")
        return
    
    # ========== 全局 PCA ==========
    all_hidden = np.vstack([t['hidden'] for t in all_trajectories])
    pca = PCA(n_components=2)
    pca.fit(all_hidden)
    
    # ========== 静态图：按任务分组 ==========
    task_names = list(set(t['task'] for t in all_trajectories))
    n_tasks = len(task_names)
    
    fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 6))
    if n_tasks == 1:
        axes = [axes]
    
    task_colors = {
        'Task3_Mod_IID': '#2ecc71',         # 绿色 - IID
        'Task3_Mod_OOD_digits': '#f39c12',  # 橙色 - OOD digits
        'Task3_Mod_OOD_length': '#e74c3c',  # 红色 - OOD length (崩溃)
    }
    
    for ax, task_name in zip(axes, task_names):
        task_trajs = [t for t in all_trajectories if t['task'] == task_name]
        base_color = task_colors.get(task_name, '#95a5a6')
        
        for i, traj in enumerate(task_trajs):
            hidden_2d = pca.transform(traj['hidden'])
            shade = 0.5 + 0.5 * i / max(1, len(task_trajs) - 1)
            color = matplotlib.colors.to_rgba(base_color, shade)
            
            ax.plot(hidden_2d[:, 0], hidden_2d[:, 1], '-', color=color, alpha=0.5, linewidth=1)
            scatter = ax.scatter(
                hidden_2d[:, 0], hidden_2d[:, 1],
                c=range(len(hidden_2d)),
                cmap='coolwarm',
                s=20,
                alpha=0.7
            )
            ax.scatter(hidden_2d[0, 0], hidden_2d[0, 1], c='blue', s=80, marker='o', zorder=5)
            ax.scatter(hidden_2d[-1, 0], hidden_2d[-1, 1], c='red', s=80, marker='X', zorder=5)
            
            # 标注
            label_text = traj['label'].split(': ')[-1] if ': ' in traj['label'] else traj['label']
            ax.annotate(label_text, hidden_2d[-1], fontsize=8, alpha=0.8)
        
        ax.set_title(task_name, fontsize=12, fontweight='bold')
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Hidden State Trajectories by Task", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "all_tasks_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}/all_tasks_trajectories.png")
    
    # ========== 动画 GIF ==========
    print("\nGenerating animation...")
    create_trajectory_animation(
        all_trajectories,
        pca,
        str(output_path / "trajectory_animation.gif"),
        fps=8
    )
    
    # ========== 任务间对比（所有轨迹叠加）==========
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for traj in all_trajectories:
        hidden_2d = pca.transform(traj['hidden'])
        color = task_colors.get(traj['task'], '#95a5a6')
        
        ax.plot(hidden_2d[:, 0], hidden_2d[:, 1], '-', color=color, alpha=0.4, linewidth=1)
        ax.scatter(hidden_2d[-1, 0], hidden_2d[-1, 1], c=color, s=100, marker='X', 
                   edgecolors='black', linewidths=1, zorder=5)
    
    # 图例
    for task_name, color in task_colors.items():
        ax.plot([], [], 'o-', color=color, label=task_name, markersize=8)
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.set_title("All Tasks Hidden State Comparison", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "all_tasks_overlay.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}/all_tasks_overlay.png")
    
    # 保存元数据
    meta = {
        'tasks': [t['name'] for t in tasks],
        'n_trajectories': len(all_trajectories),
        'pca_variance_ratio': pca.explained_variance_ratio_.tolist(),
    }
    with open(output_path / "visualization_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n[OK] All tasks visualization complete!")
    print(f"     Output: {output_path}")


def visualize_cross_domain(
    output_dir: str = "reports/cross_domain_hidden_viz",
    device: str = "cuda"
):
    """可视化跨域迁移的隐状态"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找跨域 checkpoint
    cross_domain_ckpts = list(Path("runs").glob("*transfer*/*.pt")) + \
                         list(Path("runs").glob("*cross*/*.pt")) + \
                         list(Path("artifacts/checkpoints").glob("*ipd*.pt"))
    
    # 使用相同的模型，测试不同域的输入
    ckpt_path = "runs/ood_length_decay/mini_jmamba_seed42/mod_seed42_epoch200.pt"
    
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    print(f"Loading model: {ckpt_path}")
    model = load_model(ckpt_path, device)
    
    # 测试用例：相同表达式，不同"域"的模拟
    test_cases = [
        {"symbols": "29%7", "domain": "audio", "label": "Audio: 29%7"},
        {"symbols": "29%7", "domain": "audio_slow", "label": "Audio Slow: 29%7"},
        {"symbols": "85%9", "domain": "audio", "label": "Audio: 85%9"},
        {"symbols": "85%9", "domain": "audio_slow", "label": "Audio Slow: 85%9"},
    ]
    
    all_trajectories = []
    
    for case in test_cases:
        print(f"Processing: {case['label']}")
        
        # 合成波形（模拟不同域通过不同采样率）
        sym_list = list(case["symbols"])
        if case["domain"] == "audio":
            wave = encode_symbols_to_wave(sym_list, sr=16000)
        elif case["domain"] == "audio_slow":
            # 模拟"慢速"域：拉伸波形
            wave = encode_symbols_to_wave(sym_list, sr=16000)
            # 简单的时间拉伸（重采样模拟）
            wave = np.interp(
                np.linspace(0, len(wave), int(len(wave) * 1.2)),
                np.arange(len(wave)),
                wave
            ).astype(np.float32)
        else:
            wave = encode_symbols_to_wave(sym_list, sr=16000)
        
        states = extract_hidden_states(model, wave, device)
        final_hidden = states[-1]['hidden'][0]
        
        all_trajectories.append({
            'label': case['label'],
            'domain': case['domain'],
            'hidden': final_hidden,
            'n_frames': final_hidden.shape[0],
        })
    
    # PCA
    all_hidden = np.vstack([t['hidden'] for t in all_trajectories])
    pca = PCA(n_components=2)
    pca.fit(all_hidden)
    
    # 静态图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    domain_colors = {'audio': '#2ecc71', 'audio_slow': '#e74c3c'}
    
    for traj in all_trajectories:
        hidden_2d = pca.transform(traj['hidden'])
        color = domain_colors.get(traj['domain'], '#95a5a6')
        
        ax.plot(hidden_2d[:, 0], hidden_2d[:, 1], '-', color=color, alpha=0.5, linewidth=1.5)
        ax.scatter(
            hidden_2d[:, 0], hidden_2d[:, 1],
            c=range(len(hidden_2d)),
            cmap='coolwarm',
            s=30,
            alpha=0.7
        )
        ax.scatter(hidden_2d[0, 0], hidden_2d[0, 1], c='blue', s=100, marker='o', zorder=5)
        ax.scatter(hidden_2d[-1, 0], hidden_2d[-1, 1], c='red', s=100, marker='X', zorder=5)
        ax.annotate(traj['label'], hidden_2d[-1], fontsize=9)
    
    for domain, color in domain_colors.items():
        ax.plot([], [], 'o-', color=color, label=domain, markersize=8)
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.set_title("Cross-Domain Hidden State Comparison", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "cross_domain_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}/cross_domain_trajectories.png")
    
    # 动画
    print("\nGenerating cross-domain animation...")
    create_trajectory_animation(
        all_trajectories,
        pca,
        str(output_path / "cross_domain_animation.gif"),
        fps=8
    )
    
    print(f"\n[OK] Cross-domain visualization complete!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--all-tasks", action="store_true", help="Visualize all tasks")
    parser.add_argument("--cross-domain", action="store_true", help="Visualize cross-domain")
    parser.add_argument("--all", action="store_true", help="Run all visualizations")
    args = parser.parse_args()
    
    if args.all or (not args.all_tasks and not args.cross_domain):
        args.all_tasks = True
        args.cross_domain = True
    
    if args.all_tasks:
        visualize_all_tasks(device=args.device)
    
    if args.cross_domain:
        visualize_cross_domain(device=args.device)


if __name__ == "__main__":
    main()

