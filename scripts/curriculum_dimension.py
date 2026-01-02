#!/usr/bin/env python3
"""
P3: 维度泛化实验 - Curriculum Learning

目标：通过渐进式扩展输出维度，提升 OOD 泛化能力
策略：10 → 20 → 30 类逐步训练

核心思想：
1. 先在简单任务上学习（mod 10，输出 0-9）
2. 逐步扩展到更大的输出空间
3. 观察 OOD length 性能是否提升
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig


class CurriculumModDataset(Dataset):
    """可变 mod 范围的模运算数据集"""
    
    def __init__(self, num_samples: int, seq_len_range: tuple, mod_range: tuple,
                 frame_size: int = 160, sample_rate: int = 16000):
        self.num_samples = num_samples
        self.seq_len_range = seq_len_range
        self.mod_range = mod_range  # (min_mod, max_mod)
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        
        # 预生成数据
        self.data = []
        for _ in range(num_samples):
            seq_len = random.randint(*seq_len_range)
            mod = random.randint(*mod_range)
            
            # 生成随机数字序列
            nums = [random.randint(0, 9) for _ in range(seq_len)]
            result = sum(nums) % mod
            
            # 生成伪音频帧（简化：用随机 + 编码信息）
            frames = torch.randn(seq_len, frame_size)
            # 在帧中嵌入数字信息（前 10 维编码数字）
            for i, num in enumerate(nums):
                frames[i, num] += 2.0  # 增强对应数字的信号
            # 最后一帧编码 mod 值
            frames[-1, 10 + mod] += 2.0
            
            self.data.append({
                'frames': frames,
                'target': result,
                'mod': mod,
                'nums': nums,
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['frames'], item['target'], item['mod']


def collate_fn(batch):
    """动态 padding"""
    frames_list, targets, mods = zip(*batch)
    max_len = max(f.size(0) for f in frames_list)
    
    batch_size = len(frames_list)
    frame_size = frames_list[0].size(1)
    
    padded = torch.zeros(batch_size, max_len, frame_size)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, frames in enumerate(frames_list):
        seq_len = frames.size(0)
        padded[i, :seq_len] = frames
        masks[i, :seq_len] = True
    
    return padded, masks, torch.tensor(targets), torch.tensor(mods)


def train_stage(model, train_loader, val_loader, optimizer, device, 
                epochs: int, stage_name: str, max_output_class: int):
    """训练一个阶段"""
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for frames, masks, targets, mods in train_loader:
            frames, masks, targets = frames.to(device), masks.to(device), targets.to(device)
            
            optimizer.zero_grad()
            _, logits = model(frames, masks)  # logits: (B, T, vocab_size)
            
            # 序列聚合：使用 mean pooling over valid positions
            logits_masked = logits * masks.unsqueeze(-1).float()
            logits_sum = logits_masked.sum(dim=1)  # (B, vocab_size)
            lengths = masks.sum(dim=1, keepdim=True).clamp(min=1)
            logits = logits_sum / lengths  # (B, vocab_size)
            
            # 只使用当前阶段的输出维度
            logits = logits[:, :max_output_class]
            
            # 过滤超出范围的目标
            valid_mask = targets < max_output_class
            if valid_mask.sum() == 0:
                continue
            
            loss = criterion(logits[valid_mask], targets[valid_mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds[valid_mask] == targets[valid_mask]).sum().item()
            total += valid_mask.sum().item()
        
        train_acc = correct / max(total, 1)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for frames, masks, targets, mods in val_loader:
                frames, masks, targets = frames.to(device), masks.to(device), targets.to(device)
                _, logits = model(frames, masks)  # (B, T, vocab_size)
                
                # Mean pooling
                logits_masked = logits * masks.unsqueeze(-1).float()
                logits_sum = logits_masked.sum(dim=1)
                lengths = masks.sum(dim=1, keepdim=True).clamp(min=1)
                logits = logits_sum / lengths  # (B, vocab_size)
                logits = logits[:, :max_output_class]
                
                valid_mask = targets < max_output_class
                if valid_mask.sum() == 0:
                    continue
                
                preds = logits.argmax(dim=-1)
                val_correct += (preds[valid_mask] == targets[valid_mask]).sum().item()
                val_total += valid_mask.sum().item()
        
        val_acc = val_correct / max(val_total, 1)
        best_acc = max(best_acc, val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"  [{stage_name}] Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
    
    return best_acc


def evaluate_ood(model, device, seq_len_range: tuple, mod_range: tuple, 
                 max_output_class: int, num_samples: int = 200):
    """评估 OOD 性能"""
    dataset = CurriculumModDataset(num_samples, seq_len_range, mod_range)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, masks, targets, mods in loader:
            frames, masks, targets = frames.to(device), masks.to(device), targets.to(device)
            _, logits = model(frames, masks)  # (B, T, vocab_size)
            
            # Mean pooling
            logits_masked = logits * masks.unsqueeze(-1).float()
            logits_sum = logits_masked.sum(dim=1)
            lengths = masks.sum(dim=1, keepdim=True).clamp(min=1)
            logits = logits_sum / lengths  # (B, vocab_size)
            logits = logits[:, :max_output_class]
            
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Curriculum dimension generalization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-per-stage", type=int, default=20)
    parser.add_argument("--output", type=str, default="reports/curriculum_dimension.json")
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    print("=" * 60)
    print("P3: Curriculum Dimension Generalization")
    print("=" * 60)
    print(f"Device: {device}")
    
    # 模型配置：最大支持 30 类输出
    config = MiniJMambaConfig(
        frame_size=160,
        hop_size=80,
        symbol_vocab_size=30,  # 最大输出维度
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
        num_heads=4,
        max_frames=256,
    )
    model = MiniJMamba(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    results = {
        "stages": [],
        "ood_results": [],
    }
    
    # Curriculum 训练策略
    stages = [
        {"name": "Stage1: mod 2-10", "mod_range": (2, 10), "max_output": 10, "seq_len": (3, 8)},
        {"name": "Stage2: mod 2-20", "mod_range": (2, 20), "max_output": 20, "seq_len": (3, 10)},
        {"name": "Stage3: mod 2-30", "mod_range": (2, 30), "max_output": 30, "seq_len": (3, 12)},
    ]
    
    for stage in stages:
        print(f"\n--- {stage['name']} ---")
        
        # 创建数据集
        train_ds = CurriculumModDataset(1000, stage['seq_len'], stage['mod_range'])
        val_ds = CurriculumModDataset(200, stage['seq_len'], stage['mod_range'])
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)
        
        best_acc = train_stage(
            model, train_loader, val_loader, optimizer, device,
            epochs=args.epochs_per_stage,
            stage_name=stage['name'],
            max_output_class=stage['max_output'],
        )
        
        results["stages"].append({
            "name": stage['name'],
            "best_val_acc": best_acc,
            "mod_range": stage['mod_range'],
            "max_output": stage['max_output'],
        })
        print(f"  Best val acc: {best_acc:.3f}")
    
    # OOD 评估
    print("\n--- OOD Evaluation ---")
    ood_tests = [
        {"name": "IID (len 3-12)", "seq_len": (3, 12), "mod_range": (2, 30)},
        {"name": "OOD length (len 15-25)", "seq_len": (15, 25), "mod_range": (2, 30)},
        {"name": "OOD extreme (len 30-50)", "seq_len": (30, 50), "mod_range": (2, 30)},
    ]
    
    for test in ood_tests:
        acc = evaluate_ood(model, device, test['seq_len'], test['mod_range'], max_output_class=30)
        results["ood_results"].append({
            "name": test['name'],
            "accuracy": acc,
        })
        print(f"  {test['name']}: {acc:.3f}")
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # 对比：直接训练 30 类（无 curriculum）
    print("\n--- Ablation: No Curriculum (direct 30-class) ---")
    model_nocur = MiniJMamba(config).to(device)
    optimizer_nocur = optim.AdamW(model_nocur.parameters(), lr=1e-3)
    
    train_ds_nocur = CurriculumModDataset(1000, (3, 12), (2, 30))
    val_ds_nocur = CurriculumModDataset(200, (3, 12), (2, 30))
    train_loader_nocur = DataLoader(train_ds_nocur, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader_nocur = DataLoader(val_ds_nocur, batch_size=32, collate_fn=collate_fn)
    
    nocur_acc = train_stage(
        model_nocur, train_loader_nocur, val_loader_nocur, optimizer_nocur, device,
        epochs=args.epochs_per_stage * 3,  # 同样总 epoch 数
        stage_name="No Curriculum",
        max_output_class=30,
    )
    
    print(f"\nNo-curriculum best val acc: {nocur_acc:.3f}")
    
    # OOD for no-curriculum
    nocur_ood = evaluate_ood(model_nocur, device, (15, 25), (2, 30), max_output_class=30)
    print(f"No-curriculum OOD length acc: {nocur_ood:.3f}")
    
    results["ablation_no_curriculum"] = {
        "best_val_acc": nocur_acc,
        "ood_length_acc": nocur_ood,
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Curriculum OOD length: {results['ood_results'][1]['accuracy']:.3f}")
    print(f"No-curriculum OOD length: {nocur_ood:.3f}")
    improvement = results['ood_results'][1]['accuracy'] - nocur_ood
    print(f"Improvement: {improvement:+.3f}")


if __name__ == "__main__":
    main()

