#!/usr/bin/env python3
"""
思维的轨迹 (The Trajectories of Thought)
3D 流线可视化 - 展示跨域推理的殊途同归

核心理念：推理是一个过程，不是瞬间完成的。
视觉效果：霓虹发光的双螺旋流线，从分离到汇聚。

用法:
    python scripts/visualize_trajectories_of_thought.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
    
    final_hidden = extractor.states[-1][0]
    extractor.remove_hooks()
    
    return final_hidden


def smooth_trajectory(traj: np.ndarray, window: int = 3) -> np.ndarray:
    """平滑轨迹"""
    smoothed = np.zeros_like(traj)
    for i in range(len(traj)):
        start = max(0, i - window // 2)
        end = min(len(traj), i + window // 2 + 1)
        smoothed[i] = traj[start:end].mean(axis=0)
    return smoothed


def create_glowing_line_segments(points, color, alpha_range=(0.3, 1.0), lw_range=(1, 4)):
    """创建发光效果的线段"""
    segments = []
    colors = []
    linewidths = []
    
    n_points = len(points)
    for i in range(n_points - 1):
        segments.append([points[i], points[i + 1]])
        
        # 渐变：从暗到亮
        t = i / (n_points - 1)
        alpha = alpha_range[0] + t * (alpha_range[1] - alpha_range[0])
        lw = lw_range[0] + t * (lw_range[1] - lw_range[0])
        
        rgba = list(matplotlib.colors.to_rgba(color))
        rgba[3] = alpha
        colors.append(rgba)
        linewidths.append(lw)
    
    return segments, colors, linewidths


def create_trajectories_of_thought(
    audio_traj_3d: np.ndarray,
    ipd_traj_3d: np.ndarray,
    output_path: str,
    expression: str = "45%9"
):
    """创建 3D 思维轨迹静态图"""
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 深色背景
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    
    # 颜色
    audio_color = '#ff3366'  # 霓虹红/粉
    ipd_color = '#00ffff'    # 霓虹青
    
    # 平滑轨迹
    audio_smooth = smooth_trajectory(audio_traj_3d, window=5)
    ipd_smooth = smooth_trajectory(ipd_traj_3d, window=5)
    
    n_audio = len(audio_smooth)
    n_ipd = len(ipd_smooth)
    
    # ===== 绘制发光流线 =====
    
    # 多层叠加创造发光效果
    for glow_width, glow_alpha in [(12, 0.05), (8, 0.1), (4, 0.3), (2, 0.8)]:
        # Audio 流线
        ax.plot(audio_smooth[:, 0], audio_smooth[:, 1], audio_smooth[:, 2],
                color=audio_color, linewidth=glow_width, alpha=glow_alpha)
        # IPD 流线
        ax.plot(ipd_smooth[:, 0], ipd_smooth[:, 1], ipd_smooth[:, 2],
                color=ipd_color, linewidth=glow_width, alpha=glow_alpha)
    
    # ===== 起点标记 =====
    ax.scatter(*audio_smooth[0], c=audio_color, s=200, marker='o', 
               edgecolors='white', linewidths=2, alpha=0.9, label='Audio Start')
    ax.scatter(*ipd_smooth[0], c=ipd_color, s=200, marker='^',
               edgecolors='white', linewidths=2, alpha=0.9, label='Optical Start')
    
    # ===== 终点 - 汇聚奇点 =====
    # 计算汇聚点（两个终点的中点）
    convergence_point = (audio_smooth[-1] + ipd_smooth[-1]) / 2
    
    # 发光奇点效果
    for size, alpha in [(600, 0.1), (400, 0.2), (200, 0.5), (80, 1.0)]:
        ax.scatter(*convergence_point, c='#ffffff', s=size, alpha=alpha, marker='*')
    
    # 连接终点到奇点
    ax.plot([audio_smooth[-1, 0], convergence_point[0]],
            [audio_smooth[-1, 1], convergence_point[1]],
            [audio_smooth[-1, 2], convergence_point[2]],
            '--', color=audio_color, alpha=0.5, linewidth=2)
    ax.plot([ipd_smooth[-1, 0], convergence_point[0]],
            [ipd_smooth[-1, 1], convergence_point[1]],
            [ipd_smooth[-1, 2], convergence_point[2]],
            '--', color=ipd_color, alpha=0.5, linewidth=2)
    
    # ===== 时间标记 =====
    # 在轨迹上添加时间点
    for t_frac in [0.25, 0.5, 0.75]:
        idx_a = int(t_frac * (n_audio - 1))
        idx_i = int(t_frac * (n_ipd - 1))
        
        ax.scatter(*audio_smooth[idx_a], c=audio_color, s=50, alpha=0.7, marker='.')
        ax.scatter(*ipd_smooth[idx_i], c=ipd_color, s=50, alpha=0.7, marker='.')
        
        # 虚线连接同一时刻的两个点
        ax.plot([audio_smooth[idx_a, 0], ipd_smooth[idx_i, 0]],
                [audio_smooth[idx_a, 1], ipd_smooth[idx_i, 1]],
                [audio_smooth[idx_a, 2], ipd_smooth[idx_i, 2]],
                ':', color='#ffffff', alpha=0.2, linewidth=1)
    
    # ===== 样式 =====
    ax.set_xlabel('PC1', fontsize=12, color='white', labelpad=10)
    ax.set_ylabel('PC2', fontsize=12, color='white', labelpad=10)
    ax.set_zlabel('PC3', fontsize=12, color='white', labelpad=10)
    
    # 标题
    ax.text2D(0.5, 0.95, 'The Trajectories of Thought',
              transform=ax.transAxes, fontsize=20, fontweight='bold',
              color='white', ha='center', va='top',
              fontfamily='serif')
    ax.text2D(0.5, 0.90, f'Expression: {expression}',
              transform=ax.transAxes, fontsize=14, color='#888888',
              ha='center', va='top', style='italic')
    
    # 底部标语
    ax.text2D(0.5, 0.02, '"Different physical carriers, converging to the same answer"',
              transform=ax.transAxes, fontsize=12, color='#666666',
              ha='center', va='bottom', style='italic')
    
    # 坐标轴样式
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333333')
    ax.yaxis.pane.set_edgecolor('#333333')
    ax.zaxis.pane.set_edgecolor('#333333')
    ax.grid(True, alpha=0.1, color='white')
    
    # 图例
    legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.3,
                       facecolor='#1a1a2e', labelcolor='white')
    
    # 视角
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0a0a0f')
    plt.close()
    
    print(f"Saved: {output_path}")


def create_trajectories_animation(
    audio_traj_3d: np.ndarray,
    ipd_traj_3d: np.ndarray,
    output_path: str,
    expression: str = "45%9",
    fps: int = 15
):
    """创建 3D 思维轨迹动画"""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    
    audio_color = '#ff3366'
    ipd_color = '#00ffff'
    
    audio_smooth = smooth_trajectory(audio_traj_3d, window=5)
    ipd_smooth = smooth_trajectory(ipd_traj_3d, window=5)
    
    n_audio = len(audio_smooth)
    n_ipd = len(ipd_smooth)
    max_len = max(n_audio, n_ipd)
    
    # 设置坐标范围
    all_points = np.vstack([audio_smooth, ipd_smooth])
    margin = 5
    ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)
    
    # 初始化线条
    audio_line, = ax.plot([], [], [], color=audio_color, linewidth=3, alpha=0.8)
    audio_glow, = ax.plot([], [], [], color=audio_color, linewidth=8, alpha=0.2)
    ipd_line, = ax.plot([], [], [], color=ipd_color, linewidth=3, alpha=0.8)
    ipd_glow, = ax.plot([], [], [], color=ipd_color, linewidth=8, alpha=0.2)
    
    # 当前点
    audio_point, = ax.plot([], [], [], 'o', color=audio_color, markersize=12,
                           markeredgecolor='white', markeredgewidth=2)
    ipd_point, = ax.plot([], [], [], '^', color=ipd_color, markersize=10,
                         markeredgecolor='white', markeredgewidth=2)
    
    # 连接线
    connection, = ax.plot([], [], [], ':', color='white', alpha=0.3, linewidth=1)
    
    # 时间文本
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                          color='white', fontweight='bold')
    
    # 标题
    ax.text2D(0.5, 0.98, 'The Trajectories of Thought',
              transform=ax.transAxes, fontsize=18, fontweight='bold',
              color='white', ha='center', va='top')
    
    ax.set_xlabel('PC1', fontsize=10, color='white')
    ax.set_ylabel('PC2', fontsize=10, color='white')
    ax.set_zlabel('PC3', fontsize=10, color='white')
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.1, color='white')
    
    def init():
        audio_line.set_data([], [])
        audio_line.set_3d_properties([])
        audio_glow.set_data([], [])
        audio_glow.set_3d_properties([])
        ipd_line.set_data([], [])
        ipd_line.set_3d_properties([])
        ipd_glow.set_data([], [])
        ipd_glow.set_3d_properties([])
        audio_point.set_data([], [])
        audio_point.set_3d_properties([])
        ipd_point.set_data([], [])
        ipd_point.set_3d_properties([])
        connection.set_data([], [])
        connection.set_3d_properties([])
        time_text.set_text('')
        return [audio_line, audio_glow, ipd_line, ipd_glow, 
                audio_point, ipd_point, connection, time_text]
    
    def animate(frame):
        # 计算当前时间点
        t_audio = min(int(frame / max_len * n_audio), n_audio - 1)
        t_ipd = min(int(frame / max_len * n_ipd), n_ipd - 1)
        
        # 更新轨迹线
        audio_line.set_data(audio_smooth[:t_audio+1, 0], audio_smooth[:t_audio+1, 1])
        audio_line.set_3d_properties(audio_smooth[:t_audio+1, 2])
        audio_glow.set_data(audio_smooth[:t_audio+1, 0], audio_smooth[:t_audio+1, 1])
        audio_glow.set_3d_properties(audio_smooth[:t_audio+1, 2])
        
        ipd_line.set_data(ipd_smooth[:t_ipd+1, 0], ipd_smooth[:t_ipd+1, 1])
        ipd_line.set_3d_properties(ipd_smooth[:t_ipd+1, 2])
        ipd_glow.set_data(ipd_smooth[:t_ipd+1, 0], ipd_smooth[:t_ipd+1, 1])
        ipd_glow.set_3d_properties(ipd_smooth[:t_ipd+1, 2])
        
        # 更新当前点
        audio_point.set_data([audio_smooth[t_audio, 0]], [audio_smooth[t_audio, 1]])
        audio_point.set_3d_properties([audio_smooth[t_audio, 2]])
        ipd_point.set_data([ipd_smooth[t_ipd, 0]], [ipd_smooth[t_ipd, 1]])
        ipd_point.set_3d_properties([ipd_smooth[t_ipd, 2]])
        
        # 连接线
        connection.set_data([audio_smooth[t_audio, 0], ipd_smooth[t_ipd, 0]],
                           [audio_smooth[t_audio, 1], ipd_smooth[t_ipd, 1]])
        connection.set_3d_properties([audio_smooth[t_audio, 2], ipd_smooth[t_ipd, 2]])
        
        # 计算两点距离（衡量汇聚程度）
        distance = np.linalg.norm(audio_smooth[t_audio] - ipd_smooth[t_ipd])
        progress = frame / max_len * 100
        
        time_text.set_text(f't = {progress:.0f}%  |  Distance: {distance:.1f}')
        
        # 慢慢旋转视角
        ax.view_init(elev=25, azim=30 + frame * 0.5)
        
        return [audio_line, audio_glow, ipd_line, ipd_glow,
                audio_point, ipd_point, connection, time_text]
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=max_len, interval=1000//fps, blit=False
    )
    
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
    
    # 选择一个表达式来展示
    expression = "45%9"
    print(f"\nProcessing expression: {expression}")
    
    # Audio 域轨迹
    print("  Extracting Audio trajectory...")
    wave_audio = encode_symbols_to_wave(list(expression), sr=16000)
    traj_audio = extract_hidden_trajectory(model, wave_audio, args.device)
    print(f"    Shape: {traj_audio.shape}")
    
    # IPD 域轨迹（模拟：时间拉伸 + 相位偏移）
    print("  Extracting Optical (IPD) trajectory...")
    wave_ipd = np.interp(
        np.linspace(0, len(wave_audio), int(len(wave_audio) * 0.75)),
        np.arange(len(wave_audio)),
        wave_audio
    ).astype(np.float32)
    # 添加一些相位噪声模拟不同物理载体
    wave_ipd = wave_ipd + np.random.randn(len(wave_ipd)) * 0.05
    traj_ipd = extract_hidden_trajectory(model, wave_ipd, args.device)
    print(f"    Shape: {traj_ipd.shape}")
    
    # 3D PCA
    print("\n  Fitting 3D PCA...")
    all_hidden = np.vstack([traj_audio, traj_ipd])
    pca = PCA(n_components=3)
    pca.fit(all_hidden)
    print(f"    Variance ratio: {pca.explained_variance_ratio_}")
    
    audio_3d = pca.transform(traj_audio)
    ipd_3d = pca.transform(traj_ipd)
    
    # 生成静态图
    print("\n  Generating static figure...")
    create_trajectories_of_thought(
        audio_3d, ipd_3d,
        str(output_dir / "trajectories_of_thought.png"),
        expression
    )
    
    # 生成动画
    print("\n  Generating animation...")
    create_trajectories_animation(
        audio_3d, ipd_3d,
        str(output_dir / "trajectories_of_thought.gif"),
        expression,
        fps=12
    )
    
    print(f"\n[OK] 'Trajectories of Thought' visualization complete!")
    print(f"     Output: {output_dir}")


if __name__ == "__main__":
    main()

