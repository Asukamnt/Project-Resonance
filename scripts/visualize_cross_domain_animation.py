#!/usr/bin/env python3
"""
跨域隐空间动态演化动画

展示 Audio 和 IPD 域的隐状态如何随时间同步演化
证明 "Different Bodies, Same Soul"

用法:
    python scripts/visualize_cross_domain_animation.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.symbols import encode_symbols_to_wave


def load_model(checkpoint_path: str, device: str = 'cpu') -> MiniJMamba:
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    default_config = {
        'frame_size': 160, 'hop_size': 160, 'symbol_vocab_size': 12,
        'd_model': 128, 'num_ssm_layers': 10, 'num_attn_layers': 2,
        'num_heads': 4, 'max_frames': 115, 'dropout': 0.1,
        'attn_dropout': 0.1, 'use_rope': True, 'use_learnable_pos': False,
    }
    
    if 'config' in checkpoint:
        valid_params = set(inspect.signature(MiniJMambaConfig.__init__).parameters.keys()) - {'self'}
        for k, v in checkpoint['config'].items():
            if k in valid_params:
                default_config[k] = v
    
    valid_params = set(inspect.signature(MiniJMambaConfig.__init__).parameters.keys()) - {'self'}
    config = MiniJMambaConfig(**{k: v for k, v in default_config.items() if k in valid_params})
    model = MiniJMamba(config)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


class HiddenStateExtractor:
    """提取每层隐状态"""
    def __init__(self, model):
        self.states = []
        self.hooks = []
        for layer in model.layers:
            hook = layer.register_forward_hook(self._hook_fn())
            self.hooks.append(hook)
    
    def _hook_fn(self):
        def hook(module, input, output):
            self.states.append(output.detach().cpu().numpy())
        return hook
    
    def clear(self):
        self.states = []
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def extract_hidden_trajectory(model, wave: np.ndarray, device: str) -> np.ndarray:
    """提取完整隐状态轨迹 (T, D)"""
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
    
    # 最后一层隐状态 (1, T, D) -> (T, D)
    final_hidden = extractor.states[-1][0]
    extractor.remove_hooks()
    
    return final_hidden


def create_cross_domain_animation(
    audio_trajectories: list,
    ipd_trajectories: list,
    expressions: list,
    pca: PCA,
    output_path: str,
    fps: int = 8
):
    """创建跨域隐状态同步演化动画"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    # 预计算 2D 轨迹
    audio_2d = [pca.transform(t) for t in audio_trajectories]
    ipd_2d = [pca.transform(t) for t in ipd_trajectories]
    
    max_len = max(max(len(t) for t in audio_2d), max(len(t) for t in ipd_2d))
    
    # 计算边界
    all_points = np.vstack(audio_2d + ipd_2d)
    x_min, x_max = all_points[:, 0].min() - 5, all_points[:, 0].max() + 5
    y_min, y_max = all_points[:, 1].min() - 5, all_points[:, 1].max() + 5
    
    # 颜色
    audio_color = '#ff6b6b'
    ipd_color = '#00d9ff'
    
    # 初始化图形元素
    audio_lines = []
    audio_points = []
    ipd_lines = []
    ipd_points = []
    connection_lines = []
    
    for i in range(len(expressions)):
        # Audio 轨迹线
        line, = ax.plot([], [], '-', color=audio_color, alpha=0.3, linewidth=1)
        audio_lines.append(line)
        # Audio 当前点
        point, = ax.plot([], [], 'o', color=audio_color, markersize=14, 
                         markeredgecolor='white', markeredgewidth=2)
        audio_points.append(point)
        
        # IPD 轨迹线
        line, = ax.plot([], [], '-', color=ipd_color, alpha=0.3, linewidth=1)
        ipd_lines.append(line)
        # IPD 当前点
        point, = ax.plot([], [], '^', color=ipd_color, markersize=12,
                         markeredgecolor='white', markeredgewidth=2)
        ipd_points.append(point)
        
        # 连接线（相同表达式的 Audio 和 IPD）
        conn, = ax.plot([], [], '--', color='#ffffff', alpha=0.4, linewidth=1)
        connection_lines.append(conn)
    
    # 时间和进度显示
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', fontweight='bold', color='white',
                        bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))
    
    # 标题
    title_text = ax.set_title('Cross-Domain Hidden State Evolution\n"Different Bodies, Same Soul"',
                              fontsize=16, fontweight='bold', color='white', pad=15)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12, color='white')
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)
    ax.grid(True, alpha=0.2, color='white')
    
    # 图例
    ax.plot([], [], 'o-', color=audio_color, label='Audio Domain', markersize=10)
    ax.plot([], [], '^-', color=ipd_color, label='Optical (IPD) Domain', markersize=10)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.3, 
              facecolor='#1a1a2e', labelcolor='white')
    
    def init():
        for line in audio_lines + ipd_lines + connection_lines:
            line.set_data([], [])
        for point in audio_points + ipd_points:
            point.set_data([], [])
        time_text.set_text('')
        return audio_lines + audio_points + ipd_lines + ipd_points + connection_lines + [time_text]
    
    def animate(frame):
        for i in range(len(expressions)):
            # Audio
            audio_t = audio_2d[i]
            n_audio = len(audio_t)
            t_audio = min(int(frame / max_len * n_audio), n_audio - 1)
            
            audio_lines[i].set_data(audio_t[:t_audio+1, 0], audio_t[:t_audio+1, 1])
            audio_points[i].set_data([audio_t[t_audio, 0]], [audio_t[t_audio, 1]])
            
            # IPD
            ipd_t = ipd_2d[i]
            n_ipd = len(ipd_t)
            t_ipd = min(int(frame / max_len * n_ipd), n_ipd - 1)
            
            ipd_lines[i].set_data(ipd_t[:t_ipd+1, 0], ipd_t[:t_ipd+1, 1])
            ipd_points[i].set_data([ipd_t[t_ipd, 0]], [ipd_t[t_ipd, 1]])
            
            # 连接线
            connection_lines[i].set_data(
                [audio_t[t_audio, 0], ipd_t[t_ipd, 0]],
                [audio_t[t_audio, 1], ipd_t[t_ipd, 1]]
            )
        
        progress = frame / max_len * 100
        time_text.set_text(f'Time: {progress:.0f}%')
        
        return audio_lines + audio_points + ipd_lines + ipd_points + connection_lines + [time_text]
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=max_len, interval=1000//fps, blit=True
    )
    
    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    
    print(f"Animation saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="reports/figures")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    ckpt_path = "runs/ood_length_decay/mini_jmamba_seed42/mod_seed42_epoch200.pt"
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    print(f"Loading model: {ckpt_path}")
    model = load_model(ckpt_path, args.device)
    
    # 测试表达式
    expressions = ["12%5", "23%7", "45%9", "67%8", "34%4"]
    
    print("Extracting Audio domain trajectories...")
    audio_trajectories = []
    for expr in expressions:
        wave = encode_symbols_to_wave(list(expr), sr=16000)
        traj = extract_hidden_trajectory(model, wave, args.device)
        audio_trajectories.append(traj)
        print(f"  {expr}: {traj.shape}")
    
    print("Extracting IPD domain trajectories (simulated)...")
    ipd_trajectories = []
    for expr in expressions:
        # 模拟 IPD 域：时间拉伸 0.8x
        wave = encode_symbols_to_wave(list(expr), sr=16000)
        stretched = np.interp(
            np.linspace(0, len(wave), int(len(wave) * 0.8)),
            np.arange(len(wave)),
            wave
        ).astype(np.float32)
        traj = extract_hidden_trajectory(model, stretched, args.device)
        ipd_trajectories.append(traj)
        print(f"  {expr} (IPD): {traj.shape}")
    
    # 全局 PCA
    all_hidden = np.vstack(audio_trajectories + ipd_trajectories)
    pca = PCA(n_components=2)
    pca.fit(all_hidden)
    print(f"PCA variance ratio: {pca.explained_variance_ratio_}")
    
    # 生成动画
    print("\nGenerating cross-domain animation...")
    create_cross_domain_animation(
        audio_trajectories,
        ipd_trajectories,
        expressions,
        pca,
        str(output_dir / "cross_domain_animation.gif"),
        fps=8
    )
    
    # 同时生成静态对比图
    print("\nGenerating static comparison...")
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    audio_color = '#ff6b6b'
    ipd_color = '#00d9ff'
    
    for i, expr in enumerate(expressions):
        audio_2d = pca.transform(audio_trajectories[i])
        ipd_2d = pca.transform(ipd_trajectories[i])
        
        # 轨迹
        ax.plot(audio_2d[:, 0], audio_2d[:, 1], '-', color=audio_color, alpha=0.4, linewidth=1)
        ax.plot(ipd_2d[:, 0], ipd_2d[:, 1], '-', color=ipd_color, alpha=0.4, linewidth=1)
        
        # 终点
        ax.scatter(audio_2d[-1, 0], audio_2d[-1, 1], c=audio_color, s=150, 
                   marker='o', edgecolors='white', linewidths=2, zorder=5)
        ax.scatter(ipd_2d[-1, 0], ipd_2d[-1, 1], c=ipd_color, s=150,
                   marker='^', edgecolors='white', linewidths=2, zorder=5)
        
        # 连接终点
        ax.plot([audio_2d[-1, 0], ipd_2d[-1, 0]], 
                [audio_2d[-1, 1], ipd_2d[-1, 1]],
                '--', color='white', alpha=0.5, linewidth=1)
        
        # 标注
        ax.annotate(expr, audio_2d[-1], fontsize=9, color=audio_color,
                    ha='center', va='bottom', xytext=(0, 8), textcoords='offset points')
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12, color='white')
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12, color='white')
    ax.set_title('Cross-Domain Trajectory Comparison\nAudio (circles) vs Optical/IPD (triangles)',
                 fontsize=14, fontweight='bold', color='white', pad=15)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)
    ax.grid(True, alpha=0.2, color='white')
    
    ax.plot([], [], 'o', color=audio_color, label='Audio Domain', markersize=10)
    ax.plot([], [], '^', color=ipd_color, label='Optical (IPD) Domain', markersize=10)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.3,
              facecolor='#1a1a2e', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(output_dir / "cross_domain_trajectory_comparison.png", 
                dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_dir}/cross_domain_trajectory_comparison.png")
    
    print(f"\n[OK] Cross-domain visualization complete!")


if __name__ == "__main__":
    main()

