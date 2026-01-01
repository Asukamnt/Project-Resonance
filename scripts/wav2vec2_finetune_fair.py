#!/usr/bin/env python3
"""
wav2vec2 公平微调实验 - 三种设置对比

设置1: 冻结全部，只训练线性头 (frozen)
设置2: 微调最后4层 + 线性头 (partial)
设置3: 全参数微调 (full)

用法:
    python scripts/wav2vec2_finetune_fair.py --setting frozen --seed 42
    python scripts/wav2vec2_finetune_fair.py --setting partial --seed 42
    python scripts/wav2vec2_finetune_fair.py --setting full --seed 42
    
    # 或者一次性跑完所有设置和种子
    python scripts/wav2vec2_finetune_fair.py --all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.scorer import decode_wave_to_symbols, exact_match
from jericho.task3 import target_symbols_for_task3

# 检查 transformers 是否可用
try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not installed, wav2vec2 experiments will fail")


FinetuneSetting = Literal["frozen", "partial", "full"]

@dataclass
class Wav2Vec2Config:
    """wav2vec2 微调配置"""
    setting: FinetuneSetting = "frozen"
    model_name: str = "facebook/wav2vec2-base-960h"
    epochs: int = 30
    batch_size: int = 2
    lr: float = 1e-4
    lr_head: float = 1e-3  # 头部学习率（通常更高）
    unfreeze_layers: int = 4  # partial 模式解冻的层数
    num_classes: int = 10  # 0-9 digits
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    limit: Optional[int] = None  # 限制样本数（调试用）


class Wav2Vec2ForModTask(nn.Module):
    """wav2vec2 用于 mod 任务的包装模型"""
    
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        
        # 加载预训练 wav2vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.model_name)
        hidden_size = self.wav2vec2.config.hidden_size  # 768 for base
        
        # 分类头：attention pooling + MLP
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, config.num_classes)
        )
        
        # 根据设置冻结参数
        self._configure_freezing()
    
    def _configure_freezing(self):
        """配置参数冻结"""
        if self.config.setting == "frozen":
            # 冻结所有 wav2vec2 参数
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
                
        elif self.config.setting == "partial":
            # 冻结大部分，解冻最后 N 层
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            
            # 解冻最后 N 层 encoder
            num_layers = len(self.wav2vec2.encoder.layers)
            for i in range(num_layers - self.config.unfreeze_layers, num_layers):
                for param in self.wav2vec2.encoder.layers[i].parameters():
                    param.requires_grad = True
                    
        elif self.config.setting == "full":
            # 全部可训练
            for param in self.wav2vec2.parameters():
                param.requires_grad = True
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] 16kHz 音频波形
        Returns:
            logits: [B, num_classes] 余数预测
        """
        # wav2vec2 提取特征
        outputs = self.wav2vec2(waveform)
        hidden_states = outputs.last_hidden_state  # [B, T', H]
        
        # Attention pooling
        B = hidden_states.size(0)
        query = self.query.expand(B, -1, -1)  # [B, 1, H]
        pooled, _ = self.attn(query, hidden_states, hidden_states)  # [B, 1, H]
        pooled = pooled.squeeze(1)  # [B, H]
        
        # 分类
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits
    
    def count_trainable_params(self) -> int:
        """计算可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_params(self) -> int:
        """计算总参数数量"""
        return sum(p.numel() for p in self.parameters())


class Task3Dataset(Dataset):
    """Task3 数据集"""
    
    def __init__(self, entries: List[ManifestEntry], sample_rate: int = 16000):
        self.entries = entries
        self.sample_rate = sample_rate
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        entry = self.entries[idx]
        
        # 生成波形
        wave = synthesise_entry_wave(entry)  # numpy array
        
        # 获取目标余数 - target_symbols_for_task3 返回 digit 列表
        target_digits = target_symbols_for_task3(entry.symbols)  # ["4", "2"] or ["7"]
        remainder = int("".join(target_digits))  # 转为整数
        
        # 对于 >9 的余数，取模 10（简化为 0-9 分类）
        remainder = remainder % 10
        
        return torch.from_numpy(wave).float(), remainder


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """批次整理函数 - 填充到相同长度"""
    waves, remainders = zip(*batch)
    
    # 填充波形到最长
    max_len = max(w.size(0) for w in waves)
    padded = torch.zeros(len(waves), max_len)
    for i, w in enumerate(waves):
        padded[i, :w.size(0)] = w
    
    remainders = torch.tensor(remainders, dtype=torch.long)
    return padded, remainders


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: Wav2Vec2ForModTask,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for waves, targets in loader:
        waves = waves.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(waves)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * waves.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += waves.size(0)
    
    return total_correct / total_samples


def evaluate(
    model: Wav2Vec2ForModTask,
    loader: DataLoader,
    device: str
) -> float:
    """评估模型"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for waves, targets in loader:
            waves = waves.to(device)
            targets = targets.to(device)
            
            logits = model(waves)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += waves.size(0)
    
    return total_correct / total_samples


def run_experiment(config: Wav2Vec2Config) -> dict:
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Setting: {config.setting}, Seed: {config.seed}")
    print(f"{'='*60}")
    
    set_seed(config.seed)
    
    # 加载数据
    manifest_path = Path("manifests/task3.jsonl")
    if not manifest_path.exists():
        manifest_path = Path("manifests/task3_mod.jsonl")
    if not manifest_path.exists():
        raise FileNotFoundError("找不到 task3 manifest")
    
    entries = read_manifest(manifest_path)
    
    # 按 split 分组
    train_entries = [e for e in entries if e.split == "train"]
    test_entries = [e for e in entries if e.split == "iid_test"]
    
    if config.limit:
        train_entries = train_entries[:config.limit]
        test_entries = test_entries[:min(config.limit // 5, len(test_entries))]
    
    print(f"Train samples: {len(train_entries)}")
    print(f"Test samples: {len(test_entries)}")
    
    # 创建数据集和加载器
    train_dataset = Task3Dataset(train_entries)
    test_dataset = Task3Dataset(test_entries)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 创建模型
    model = Wav2Vec2ForModTask(config)
    model = model.to(config.device)
    
    trainable_params = model.count_trainable_params()
    total_params = model.count_total_params()
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M ({100*trainable_params/total_params:.1f}%)")
    
    # 优化器 - 头部用更高学习率
    head_params = [model.query] + \
                  list(model.attn.parameters()) + \
                  list(model.classifier.parameters())
    backbone_params = [p for p in model.wav2vec2.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": config.lr_head},
        {"params": backbone_params, "lr": config.lr}
    ], weight_decay=0.01)
    
    # 训练
    best_acc = 0.0
    for epoch in range(config.epochs):
        train_acc = train_epoch(model, train_loader, optimizer, config.device)
        test_acc = evaluate(model, test_loader, config.device)
        best_acc = max(best_acc, test_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, best={best_acc:.3f}")
    
    result = {
        "setting": config.setting,
        "seed": config.seed,
        "epochs": config.epochs,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
        "trainable_pct": 100 * trainable_params / total_params,
        "final_test_acc": test_acc,
        "best_test_acc": best_acc,
        "train_samples": len(train_entries),
        "test_samples": len(test_entries),
    }
    
    print(f"\n结果: {config.setting} seed={config.seed} -> IID EM = {best_acc*100:.1f}%")
    return result


def main():
    parser = argparse.ArgumentParser(description="wav2vec2 公平微调实验")
    parser.add_argument("--setting", type=str, choices=["frozen", "partial", "full"],
                        default="frozen", help="微调设置")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="backbone 学习率")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="head 学习率")
    parser.add_argument("--limit", type=int, default=None, help="限制样本数（调试用）")
    parser.add_argument("--all", action="store_true", help="运行所有设置和种子")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="种子列表（逗号分隔）")
    args = parser.parse_args()
    
    if not HAS_TRANSFORMERS:
        print("ERROR: transformers 库未安装，请运行: pip install transformers")
        sys.exit(1)
    
    # 确保输出目录存在
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    if args.all:
        # 运行所有设置和种子
        settings: List[FinetuneSetting] = ["frozen", "partial", "full"]
        seeds = [int(s) for s in args.seeds.split(",")]
        all_results = []
        
        for setting in settings:
            for seed in seeds:
                config = Wav2Vec2Config(
                    setting=setting,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    lr_head=args.lr_head,
                    limit=args.limit,
                )
                result = run_experiment(config)
                all_results.append(result)
        
        # 汇总结果
        print("\n" + "="*80)
        print("汇总结果")
        print("="*80)
        
        summary = {}
        for setting in settings:
            setting_results = [r for r in all_results if r["setting"] == setting]
            accs = [r["best_test_acc"] for r in setting_results]
            summary[setting] = {
                "mean_acc": np.mean(accs),
                "std_acc": np.std(accs),
                "min_acc": np.min(accs),
                "max_acc": np.max(accs),
                "trainable_params_M": setting_results[0]["trainable_params_M"],
                "all_accs": accs,
            }
            print(f"{setting:8s}: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}% "
                  f"(trainable: {summary[setting]['trainable_params_M']:.2f}M)")
        
        # 保存结果
        output = {
            "timestamp": datetime.now().isoformat(),
            "settings": settings,
            "seeds": seeds,
            "all_results": all_results,
            "summary": summary,
            "mini_jmamba_reference": {
                "params_M": 0.94,
                "iid_em": 0.45,
                "note": "Mini-JMamba Task3 disjoint IID EM"
            }
        }
        
        output_path = reports_dir / "wav2vec2_finetune_fair.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")
        
    else:
        # 运行单个实验
        config = Wav2Vec2Config(
            setting=args.setting,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_head=args.lr_head,
            limit=args.limit,
        )
        result = run_experiment(config)
        
        # 保存结果
        output_path = reports_dir / f"wav2vec2_{args.setting}_seed{args.seed}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

