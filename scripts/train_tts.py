#!/usr/bin/env python3
"""
在 TTS 真实数据上训练 Mini-JMamba

验证模型在真实语音（Edge-TTS 生成）上的性能
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig


class TTSDataset(Dataset):
    """TTS 数据集"""
    
    def __init__(self, manifest_path: Path, frame_size: int = 160, hop_size: int = 80):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.samples = []
        
        with open(manifest_path) as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载输入音频
        input_wav, sr = sf.read(sample["input_audio"])
        input_wav = torch.from_numpy(input_wav).float()
        # soundfile 返回 (samples,) 或 (samples, channels)
        if input_wav.dim() == 2:
            input_wav = input_wav[:, 0]  # 取第一个声道
        
        # 分帧
        num_frames = (len(input_wav) - self.frame_size) // self.hop_size + 1
        if num_frames < 1:
            # Pad if too short
            input_wav = torch.nn.functional.pad(input_wav, (0, self.frame_size - len(input_wav)))
            num_frames = 1
        
        frames = []
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            if end <= len(input_wav):
                frames.append(input_wav[start:end])
        
        frames = torch.stack(frames)  # (num_frames, frame_size)
        
        # 目标是模运算结果
        target = sample["result"]
        
        return frames, target


def collate_fn(batch):
    """动态 padding"""
    frames_list, targets = zip(*batch)
    max_len = max(f.size(0) for f in frames_list)
    
    batch_size = len(frames_list)
    frame_size = frames_list[0].size(1)
    
    padded = torch.zeros(batch_size, max_len, frame_size)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, frames in enumerate(frames_list):
        seq_len = frames.size(0)
        padded[i, :seq_len] = frames
        masks[i, :seq_len] = True
    
    return padded, masks, torch.tensor(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=str, default="manifests/tts_task3_v3")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    print("=" * 60)
    print("Training Mini-JMamba on TTS Data")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Manifest: {args.manifest_dir}")
    
    # 数据集
    manifest_dir = Path(args.manifest_dir)
    train_ds = TTSDataset(manifest_dir / "train.jsonl")
    val_ds = TTSDataset(manifest_dir / "val.jsonl")
    test_ds = TTSDataset(manifest_dir / "test.jsonl")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # 模型
    config = MiniJMambaConfig(
        frame_size=160,
        hop_size=80,
        symbol_vocab_size=10,  # 0-9
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
        num_heads=4,
        max_frames=512,  # TTS 音频较长
    )
    model = MiniJMamba(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for frames, masks, targets in train_loader:
            frames, masks, targets = frames.to(device), masks.to(device), targets.to(device)
            
            optimizer.zero_grad()
            _, logits = model(frames, masks)  # (B, T, vocab_size)
            
            # Mean pooling
            logits_masked = logits * masks.unsqueeze(-1).float()
            logits_sum = logits_masked.sum(dim=1)
            lengths = masks.sum(dim=1, keepdim=True).clamp(min=1)
            logits = logits_sum / lengths  # (B, vocab_size)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        
        train_acc = correct / total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, masks, targets in val_loader:
                frames, masks, targets = frames.to(device), masks.to(device), targets.to(device)
                _, logits = model(frames, masks)
                
                logits_masked = logits * masks.unsqueeze(-1).float()
                logits_sum = logits_masked.sum(dim=1)
                lengths = masks.sum(dim=1, keepdim=True).clamp(min=1)
                logits = logits_sum / lengths
                
                preds = logits.argmax(dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save(model.state_dict(), "runs/tts_best.pt")
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, best={best_val_acc:.3f}")
    
    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for frames, masks, targets in test_loader:
            frames, masks, targets = frames.to(device), masks.to(device), targets.to(device)
            _, logits = model(frames, masks)
            
            logits_masked = logits * masks.unsqueeze(-1).float()
            logits_sum = logits_masked.sum(dim=1)
            lengths = masks.sum(dim=1, keepdim=True).clamp(min=1)
            logits = logits_sum / lengths
            
            preds = logits.argmax(dim=-1)
            test_correct += (preds == targets).sum().item()
            test_total += targets.size(0)
    
    test_acc = test_correct / test_total
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_val_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    
    # Save results
    results = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs": args.epochs,
        "train_samples": len(train_ds),
        "device": str(device),
    }
    
    with open("reports/tts_training.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to reports/tts_training.json")


if __name__ == "__main__":
    main()

