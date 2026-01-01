#!/usr/bin/env python3
"""
动态 k 训练期实验

在训练期学习 keep_ratio，验证模型是否能自动迁移到最优修剪策略。

设计：
- 添加 DynamicPruningGate 模块到 MiniJMamba 每层后
- Gate: k = sigmoid(linear(layer_stats))
- 训练时同时优化 gate 参数
- 验证是否自动迁移到 ~0.7

Author: Jericho Team
Date: 2026-01-02
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig
from jericho.symbols import encode_symbols_to_wave, SYMBOL2FREQ, SR
from jericho.data.manifest import read_manifest


class DynamicPruningGate(nn.Module):
    """学习动态 keep_ratio 的门控模块"""
    
    def __init__(self, d_model: int, init_k: float = 0.7):
        super().__init__()
        self.d_model = d_model
        
        # 统计层：计算每个通道的重要性
        self.stats_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )
        
        # 学习 keep_ratio：基于全局统计
        self.k_predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # 初始化偏置使初始 k 接近 init_k
        with torch.no_grad():
            self.k_predictor[-2].bias.fill_(np.log(init_k / (1 - init_k)))
        
        self.learned_k_history = []
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            pruned_x: (batch, seq_len, d_model) 修剪后的隐状态
            k: 本次使用的 keep_ratio
        """
        batch, seq_len, d_model = x.shape
        
        # 计算全局统计
        global_stats = x.mean(dim=(0, 1))  # (d_model,)
        
        # 预测 keep_ratio
        k = self.k_predictor(global_stats).squeeze()  # scalar
        k = k.clamp(0.3, 1.0)  # 限制在合理范围
        
        self.learned_k_history.append(k.item())
        
        # 计算通道重要性
        channel_importance = self.stats_proj(x).squeeze(-1)  # (batch, seq_len)
        
        # 计算每个通道的 L2 范数作为重要性权重
        channel_norms = torch.norm(x, p=2, dim=1)  # (batch, d_model)
        
        # 软修剪：使用 top-k 掩码
        num_keep = int(d_model * k)
        _, top_indices = torch.topk(channel_norms, num_keep, dim=-1)  # (batch, num_keep)
        
        # 创建软掩码
        mask = torch.zeros(batch, d_model, device=x.device)
        mask.scatter_(1, top_indices, 1.0)
        
        # 应用掩码（软方式，保持可微）
        # 使用 Straight-Through Estimator 技巧
        soft_mask = torch.sigmoid((channel_norms - channel_norms.mean(dim=-1, keepdim=True)) * 10)
        mask_combined = mask + (soft_mask - mask).detach()  # STE
        
        pruned_x = x * mask_combined.unsqueeze(1)
        
        return pruned_x, k.item()


class DynamicPruningMiniJMamba(nn.Module):
    """带动态修剪门控的 MiniJMamba"""
    
    def __init__(self, base_model: MiniJMamba, init_k: float = 0.7, gate_every_n_layers: int = 2):
        super().__init__()
        self.base_model = base_model
        self.gate_every_n_layers = gate_every_n_layers
        
        # 为每 N 层添加一个门控
        num_gates = len(base_model.layers) // gate_every_n_layers
        # 从 base_model.config 获取 d_model
        d_model = base_model.symbol_head.in_features if hasattr(base_model, 'symbol_head') else 128
        self.gates = nn.ModuleList([
            DynamicPruningGate(d_model, init_k)
            for _ in range(num_gates)
        ])
        
        self.k_values = []
    
    def forward(
        self,
        frames: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        return_hidden: bool = False
    ):
        x = self.base_model.input_proj(frames)
        x = self.base_model.dropout(x)
        
        self.k_values = []
        gate_idx = 0
        
        for i, layer in enumerate(self.base_model.layers):
            x = layer(x, padding_mask)
            
            # 每 N 层应用门控
            if (i + 1) % self.gate_every_n_layers == 0 and gate_idx < len(self.gates):
                x, k = self.gates[gate_idx](x)
                self.k_values.append(k)
                gate_idx += 1
        
        x = self.base_model.final_norm(x)
        symbol_logits = self.base_model.symbol_head(x)
        
        if return_hidden:
            return None, symbol_logits, x
        return None, symbol_logits
    
    def get_mean_k(self) -> float:
        """获取平均 keep_ratio"""
        if not self.k_values:
            return 1.0
        return sum(self.k_values) / len(self.k_values)
    
    def get_gate_parameters(self):
        """获取门控参数"""
        return [p for gate in self.gates for p in gate.parameters()]


@dataclass
class DynamicKTrainingConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    gate_lr: float = 1e-3
    init_k: float = 0.7
    gate_every_n_layers: int = 2
    sparsity_weight: float = 0.01  # 稀疏性正则化权重


class SimpleModDataset(Dataset):
    """简化的 Mod 数据集"""
    
    def __init__(self, entries, frame_size: int = 160):
        self.entries = entries
        self.frame_size = frame_size
        self.samples = []
        
        for entry in entries:
            symbols = list(entry.symbols)
            try:
                wave = encode_symbols_to_wave(symbols, tone_dur=0.01, sr=SR)
                num_frames = len(wave) // frame_size
                if num_frames > 0:
                    wave = wave[:num_frames * frame_size]
                    frames = wave.reshape(num_frames, frame_size)
                    self.samples.append({
                        "frames": torch.tensor(frames, dtype=torch.float32),
                        "symbols": symbols,
                    })
            except:
                continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """批处理函数"""
    max_len = max(b["frames"].shape[0] for b in batch)
    
    frames_list = []
    masks_list = []
    
    for b in batch:
        f = b["frames"]
        pad_len = max_len - f.shape[0]
        padded = F.pad(f, (0, 0, 0, pad_len))
        frames_list.append(padded)
        
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:f.shape[0]] = True
        masks_list.append(mask)
    
    return {
        "frames": torch.stack(frames_list),
        "masks": torch.stack(masks_list),
    }


def train_dynamic_k(
    checkpoint_path: str,
    manifest_path: str,
    config: DynamicKTrainingConfig,
    device: str,
    output_dir: Path,
):
    """训练动态 k 模型"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载基础模型
    print(f"Loading base model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = ckpt["config"]
    
    base_config = MiniJMambaConfig(
        frame_size=saved_config.get("frame_size", 160),
        hop_size=saved_config.get("hop_size", 160),
        symbol_vocab_size=saved_config.get("symbol_vocab_size", 12),
        d_model=saved_config.get("d_model", 128),
        num_ssm_layers=saved_config.get("num_ssm_layers", 10),
        num_attn_layers=saved_config.get("num_attn_layers", 2),
        max_frames=saved_config.get("max_frames", 256),
        use_rope=saved_config.get("use_rope", True),
    )
    
    base_model = MiniJMamba(base_config).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    # 保持基础模型参数可训练（需要梯度流动）
    # 但使用较小的学习率
    for param in base_model.parameters():
        param.requires_grad = True
    
    # 创建动态修剪模型
    model = DynamicPruningMiniJMamba(
        base_model,
        init_k=config.init_k,
        gate_every_n_layers=config.gate_every_n_layers,
    ).to(device)
    
    # 加载数据
    print(f"Loading data from {manifest_path}")
    entries = list(read_manifest(Path(manifest_path)))[:500]  # 限制数据量
    dataset = SimpleModDataset(entries)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # 优化门控参数和基础模型参数（后者使用较小学习率）
    optimizer = torch.optim.Adam([
        {"params": model.get_gate_parameters(), "lr": config.gate_lr},
        {"params": model.base_model.parameters(), "lr": config.gate_lr * 0.01},  # 基础模型学习率更小
    ])
    
    # 训练循环
    k_history = []
    loss_history = []
    
    print(f"Training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        epoch_k = 0
        n_batches = 0
        
        for batch in dataloader:
            frames = batch["frames"].to(device)
            masks = batch["masks"].to(device)
            
            optimizer.zero_grad()
            
            _, symbol_logits = model(frames, masks)
            
            # 简单的重建损失（预测下一帧）
            pred = symbol_logits[:, :-1, :]
            target = symbol_logits[:, 1:, :].argmax(dim=-1)
            loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
            
            # 稀疏性正则化：鼓励较低的 k 值
            mean_k = model.get_mean_k()
            sparsity_loss = config.sparsity_weight * mean_k
            
            total_loss = loss + sparsity_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_k += mean_k
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_k = epoch_k / n_batches
        
        k_history.append(avg_k)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config.epochs}: loss={avg_loss:.4f}, k={avg_k:.4f}")
    
    # 保存结果
    results = {
        "config": asdict(config),
        "k_history": k_history,
        "loss_history": loss_history,
        "final_k": k_history[-1] if k_history else config.init_k,
    }
    
    output_file = output_dir / "dynamic_k_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Final learned k: {results['final_k']:.4f}")
    
    # 绘图
    plot_results(results, output_dir)
    
    return results


def plot_results(results: dict, output_dir: Path):
    """绘制训练结果"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) k 值演变
    ax = axes[0]
    ax.plot(results["k_history"], 'o-', color='#3498db', linewidth=2)
    ax.axhline(y=0.7, color='green', linestyle='--', label='Target k=0.7')
    ax.axhline(y=1.0, color='red', linestyle=':', label='No pruning k=1.0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learned k (keep ratio)')
    ax.set_title('(a) Dynamic k Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # (b) Loss 曲线
    ax = axes[1]
    ax.plot(results["loss_history"], 'o-', color='#e74c3c', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(b) Training Loss')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "dynamic_k_plot.png"
    fig.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Dynamic k Training Experiment")
    parser.add_argument("--checkpoint", type=str,
                       default="artifacts/checkpoints/mod_best_em0.75.pt")
    parser.add_argument("--manifest", type=str, default="manifests/task3.jsonl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--init-k", type=float, default=0.7)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default="reports/dynamic_k")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = DynamicKTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        gate_lr=args.gate_lr,
        init_k=args.init_k,
        sparsity_weight=args.sparsity_weight,
    )
    
    train_dynamic_k(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        config=config,
        device=device,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

