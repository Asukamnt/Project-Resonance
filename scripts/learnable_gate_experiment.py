#!/usr/bin/env python3
"""
可学习 Gate 实验 (Learnable Synaptic Gate)

核心思想：让模型自己学习哪些通道需要保留
- 不再手动设置 keep_ratio
- 使用 Sigmoid Gate 自适应决定每个通道的重要性
- Gate 参数通过反向传播学习

设计：
1. SigmoidGate: 每个通道一个可学习参数 → sigmoid → 软掩码
2. TopKGate: 学习一个全局阈值，动态选择 top-k
3. AttentionGate: 用 attention 机制决定通道重要性

Author: Jericho Team
Date: 2026-01-02
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig, SSMLikeBlock
from jericho.symbols import SR, SYMBOL2FREQ, encode_symbols_to_wave


class SigmoidGate(nn.Module):
    """Sigmoid Gate: 每个通道一个可学习的重要性分数"""
    
    def __init__(self, d_model: int, init_bias: float = 2.0):
        super().__init__()
        # 初始化为正值，使得初始时大部分通道被保留
        self.gate_logits = nn.Parameter(torch.ones(d_model) * init_bias)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, D)
        Returns:
            gated_h: (B, T, D)
        """
        gate = torch.sigmoid(self.gate_logits)  # (D,)
        return h * gate.unsqueeze(0).unsqueeze(0)  # (B, T, D)
    
    def get_keep_ratio(self) -> float:
        """返回当前的有效 keep_ratio（gate > 0.5 的比例）"""
        with torch.no_grad():
            gate = torch.sigmoid(self.gate_logits)
            return (gate > 0.5).float().mean().item()
    
    def get_gate_distribution(self) -> Dict[str, float]:
        """返回 gate 分布统计"""
        with torch.no_grad():
            gate = torch.sigmoid(self.gate_logits)
            return {
                "mean": gate.mean().item(),
                "std": gate.std().item(),
                "min": gate.min().item(),
                "max": gate.max().item(),
                "keep_ratio": (gate > 0.5).float().mean().item(),
            }


class LearnableGateMiniJMamba(nn.Module):
    """带可学习 Gate 的 MiniJMamba"""
    
    def __init__(
        self,
        base_model: MiniJMamba,
        gate_type: str = "sigmoid",
        gate_layers: str = "ssm",  # "ssm", "attn", "all"
        init_bias: float = 2.0,
    ):
        super().__init__()
        self.config = base_model.config
        self.gate_type = gate_type
        self.gate_layers = gate_layers
        
        # 复制组件
        self.input_proj = base_model.input_proj
        self.pos_emb = base_model.pos_emb
        self.dropout = base_model.dropout
        self.final_norm = base_model.final_norm
        self.frame_head = base_model.frame_head
        self.symbol_head = base_model.symbol_head
        self.layers = base_model.layers
        
        # 为指定层添加 Gate
        self.gates = nn.ModuleDict()
        for i, layer in enumerate(self.layers):
            is_ssm = isinstance(layer, SSMLikeBlock)
            if (gate_layers == "ssm" and is_ssm) or \
               (gate_layers == "attn" and not is_ssm) or \
               (gate_layers == "all"):
                self.gates[str(i)] = SigmoidGate(self.config.d_model, init_bias)
    
    def forward(
        self,
        frames: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        *,
        return_hidden: bool = False,
    ):
        batch_size, seq_len, feat_dim = frames.shape
        x = self.input_proj(frames)
        
        if self.pos_emb is not None:
            positions = torch.arange(seq_len, device=frames.device)
            positions = positions.clamp(max=self.config.max_frames - 1)
            x = x + self.pos_emb(positions).unsqueeze(0)
        
        x = self.dropout(x)
        
        for i, layer in enumerate(self.layers):
            x = layer(x, padding_mask)
            # 应用 Gate（如果存在）
            if str(i) in self.gates:
                x = self.gates[str(i)](x)
        
        x = self.final_norm(x)
        frame_outputs = self.frame_head(x)
        symbol_logits = self.symbol_head(x)
        
        if padding_mask is not None:
            frame_outputs = frame_outputs.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
            symbol_logits = symbol_logits.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        
        if return_hidden:
            return frame_outputs, symbol_logits, x
        return frame_outputs, symbol_logits
    
    def get_all_gate_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有 Gate 的统计信息"""
        stats = {}
        for name, gate in self.gates.items():
            stats[f"layer_{name}"] = gate.get_gate_distribution()
        return stats
    
    def get_average_keep_ratio(self) -> float:
        """获取平均 keep_ratio"""
        if not self.gates:
            return 1.0
        ratios = [g.get_keep_ratio() for g in self.gates.values()]
        return np.mean(ratios)


def generate_training_batch(
    batch_size: int,
    num_symbols: int,
    frame_size: int,
    device: str,
    seed_offset: int = 0,
) -> tuple:
    """生成训练 batch"""
    symbols = list(SYMBOL2FREQ.keys())[:10]  # 0-9
    
    all_frames = []
    all_targets = []
    
    for i in range(batch_size):
        np.random.seed(seed_offset * 1000 + i)
        symbol_seq = [symbols[np.random.randint(0, len(symbols))] for _ in range(num_symbols)]
        wave = encode_symbols_to_wave(symbol_seq, tone_dur=0.1, gap_dur=0.0)
        
        num_frames = len(wave) // frame_size
        if num_frames == 0:
            continue
        
        frames = wave[:num_frames * frame_size].reshape(num_frames, frame_size)
        all_frames.append(torch.tensor(frames, dtype=torch.float32))
        
        # 目标：符号序列的 one-hot
        target = torch.zeros(num_frames, len(symbols))
        frames_per_symbol = num_frames // num_symbols
        for j, sym in enumerate(symbol_seq):
            sym_idx = symbols.index(sym)
            start = j * frames_per_symbol
            end = min((j + 1) * frames_per_symbol, num_frames)
            target[start:end, sym_idx] = 1.0
        all_targets.append(target)
    
    # Pad to same length
    max_len = max(f.shape[0] for f in all_frames)
    padded_frames = torch.zeros(batch_size, max_len, frame_size)
    padded_targets = torch.zeros(batch_size, max_len, len(symbols))
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, (f, t) in enumerate(zip(all_frames, all_targets)):
        padded_frames[i, :f.shape[0]] = f
        padded_targets[i, :t.shape[0]] = t
        padding_mask[i, :f.shape[0]] = True
    
    return padded_frames.to(device), padded_targets.to(device), padding_mask.to(device)


def train_gate(
    checkpoint_path: str,
    num_epochs: int = 20,
    batch_size: int = 16,
    lr_gate: float = 0.01,
    lr_model: float = 1e-5,
    freeze_model: bool = True,
    gate_layers: str = "ssm",
    init_bias: float = 2.0,
    sparsity_weight: float = 0.01,  # 稀疏性正则化
    num_symbols: int = 16,
    frame_size: int = 160,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """训练可学习 Gate"""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Gate layers: {gate_layers}")
    print(f"Freeze model: {freeze_model}")
    print(f"Sparsity weight: {sparsity_weight}")
    print(f"Device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = MiniJMambaConfig(
        frame_size=frame_size,
        hop_size=frame_size,
        symbol_vocab_size=12,
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
    )
    
    base_model = MiniJMamba(config).to(device)
    if 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        base_model.load_state_dict(checkpoint, strict=False)
    
    # 创建带 Gate 的模型
    model = LearnableGateMiniJMamba(
        base_model, 
        gate_type="sigmoid",
        gate_layers=gate_layers,
        init_bias=init_bias,
    ).to(device)
    
    # 优化器：分离 gate 和 model 参数
    gate_params = list(model.gates.parameters())
    model_params = [p for n, p in model.named_parameters() if "gates" not in n]
    
    if freeze_model:
        for p in model_params:
            p.requires_grad = False
        optimizer = torch.optim.Adam(gate_params, lr=lr_gate)
    else:
        optimizer = torch.optim.Adam([
            {"params": gate_params, "lr": lr_gate},
            {"params": model_params, "lr": lr_model},
        ])
    
    # 训练记录
    history = {
        "epochs": [],
        "losses": [],
        "keep_ratios": [],
        "max_probs": [],
        "gate_stats": [],
    }
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Initial keep_ratio: {model.get_average_keep_ratio():.3f}")
    
    for epoch in range(num_epochs):
        model.train()
        
        # 生成训练数据
        frames, targets, mask = generate_training_batch(
            batch_size, num_symbols, frame_size, device, seed_offset=epoch
        )
        
        optimizer.zero_grad()
        
        # 前向传播
        output, logits, hidden = model(frames, mask, return_hidden=True)
        
        # 损失：分类 + 稀疏性正则化
        probs = torch.softmax(logits, dim=-1)
        
        # 分类损失（使用 target 作为软标签）
        cls_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1, targets.size(-1)).argmax(dim=-1),
            reduction='none'
        )
        cls_loss = (cls_loss * mask.view(-1).float()).mean()
        
        # 稀疏性正则化：鼓励 gate 学习稀疏模式
        sparsity_loss = 0.0
        for gate in model.gates.values():
            gate_vals = torch.sigmoid(gate.gate_logits)
            # 使用熵作为稀疏性度量：鼓励 gate 接近 0 或 1
            entropy = -(gate_vals * torch.log(gate_vals + 1e-8) + 
                       (1 - gate_vals) * torch.log(1 - gate_vals + 1e-8))
            sparsity_loss += entropy.mean()
        sparsity_loss /= len(model.gates)
        
        total_loss = cls_loss + sparsity_weight * sparsity_loss
        
        total_loss.backward()
        optimizer.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            eval_frames, eval_targets, eval_mask = generate_training_batch(
                32, num_symbols, frame_size, device, seed_offset=epoch + 1000
            )
            _, eval_logits, _ = model(eval_frames, eval_mask, return_hidden=True)
            eval_probs = torch.softmax(eval_logits, dim=-1)
            max_prob = eval_probs.max(dim=-1).values.mean().item()
        
        keep_ratio = model.get_average_keep_ratio()
        gate_stats = model.get_all_gate_stats()
        
        history["epochs"].append(epoch)
        history["losses"].append(total_loss.item())
        history["keep_ratios"].append(keep_ratio)
        history["max_probs"].append(max_prob)
        history["gate_stats"].append(gate_stats)
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}: loss={total_loss.item():.4f}, "
                  f"keep_ratio={keep_ratio:.3f}, max_prob={max_prob:.3f}")
    
    # 最终评估
    print("\n" + "="*60)
    print("FINAL GATE DISTRIBUTION")
    print("="*60)
    
    final_stats = model.get_all_gate_stats()
    for layer_name, stats in final_stats.items():
        print(f"{layer_name}: keep_ratio={stats['keep_ratio']:.3f}, "
              f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    return {
        "config": {
            "checkpoint": checkpoint_path,
            "num_epochs": num_epochs,
            "gate_layers": gate_layers,
            "init_bias": init_bias,
            "sparsity_weight": sparsity_weight,
            "freeze_model": freeze_model,
            "seed": seed,
        },
        "history": history,
        "final_stats": final_stats,
        "final_keep_ratio": model.get_average_keep_ratio(),
    }


def main():
    parser = argparse.ArgumentParser(description='Learnable Gate Experiment')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr-gate', type=float, default=0.01)
    parser.add_argument('--lr-model', type=float, default=1e-5)
    parser.add_argument('--freeze-model', action='store_true', default=True)
    parser.add_argument('--no-freeze-model', action='store_false', dest='freeze_model')
    parser.add_argument('--gate-layers', type=str, default='ssm', choices=['ssm', 'attn', 'all'])
    parser.add_argument('--init-bias', type=float, default=2.0)
    parser.add_argument('--sparsity-weight', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='reports/learnable_gate')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = train_gate(
        checkpoint_path=args.checkpoint,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr_gate=args.lr_gate,
        lr_model=args.lr_model,
        freeze_model=args.freeze_model,
        gate_layers=args.gate_layers,
        init_bias=args.init_bias,
        sparsity_weight=args.sparsity_weight,
        seed=args.seed,
    )
    
    # 保存结果
    with open(output_dir / 'learnable_gate_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 绘制学习曲线
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = results["history"]["epochs"]
    
    axes[0].plot(epochs, results["history"]["losses"])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, results["history"]["keep_ratios"])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Keep Ratio')
    axes[1].set_title('Learned Keep Ratio')
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='k=0.5')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, results["history"]["max_probs"])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Max Probability')
    axes[2].set_title('Classification Confidence')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learnable_gate_training.png', dpi=150)
    plt.close()
    
    print(f"\nResults saved to {output_dir}")
    print(f"Final learned keep_ratio: {results['final_keep_ratio']:.3f}")


if __name__ == '__main__':
    main()

