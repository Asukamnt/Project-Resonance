#!/usr/bin/env python3
"""
真实数据验证实验：使用 Google Speech Commands 数据集

任务：Mirror（听到数字，输出相同数字）
数据：真实人声录制的 0-9 数字
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
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig


DIGIT_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}


class SpeechCommandsDataset(Dataset):
    """Google Speech Commands 数字子集"""
    
    def __init__(self, data_dir: Path, split: str = "train", 
                 frame_size: int = 160, hop_size: int = 80,
                 max_samples_per_digit: int = 2500):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.samples = []
        
        for digit_name, digit_val in DIGIT_MAP.items():
            digit_dir = data_dir / digit_name
            if not digit_dir.exists():
                continue
            
            wav_files = list(digit_dir.glob("*.wav"))
            
            # 划分 train/val/test
            random.seed(42)
            random.shuffle(wav_files)
            
            n = len(wav_files)
            if split == "train":
                files = wav_files[:int(n * 0.7)]
            elif split == "val":
                files = wav_files[int(n * 0.7):int(n * 0.85)]
            else:  # test
                files = wav_files[int(n * 0.85):]
            
            # 限制每个数字的样本数
            files = files[:max_samples_per_digit]
            
            for f in files:
                self.samples.append({
                    "path": f,
                    "label": digit_val,
                    "digit_name": digit_name,
                })
        
        print(f"  {split}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载音频
        waveform, sr = torchaudio.load(sample["path"])
        waveform = waveform.squeeze(0)  # (samples,)
        
        # 重采样到 16kHz
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        
        # 分帧
        num_frames = (len(waveform) - self.frame_size) // self.hop_size + 1
        if num_frames < 1:
            waveform = torch.nn.functional.pad(waveform, (0, self.frame_size - len(waveform)))
            num_frames = 1
        
        frames = []
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            if end <= len(waveform):
                frames.append(waveform[start:end])
        
        if len(frames) == 0:
            frames = [waveform[:self.frame_size]]
        
        frames = torch.stack(frames)  # (num_frames, frame_size)
        
        return frames, sample["label"]


def collate_fn(batch):
    """动态 padding"""
    frames_list, labels = zip(*batch)
    max_len = max(f.size(0) for f in frames_list)
    
    batch_size = len(frames_list)
    frame_size = frames_list[0].size(1)
    
    padded = torch.zeros(batch_size, max_len, frame_size)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, frames in enumerate(frames_list):
        seq_len = frames.size(0)
        padded[i, :seq_len] = frames
        masks[i, :seq_len] = True
    
    return padded, masks, torch.tensor(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/speech_commands_digits")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix for output files, e.g. '_seed123'")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    
    print("=" * 60)
    print("Real Speech Validation: Google Speech Commands")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory {data_dir} not found!")
        print("Run scripts/download_speech_commands.py first.")
        return
    
    # 数据集
    print("\nLoading datasets...")
    train_ds = SpeechCommandsDataset(data_dir, "train")
    val_ds = SpeechCommandsDataset(data_dir, "val")
    test_ds = SpeechCommandsDataset(data_dir, "test")
    
    if len(train_ds) == 0:
        print("ERROR: No training samples found!")
        return
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # 模型
    config = MiniJMambaConfig(
        frame_size=160,
        hop_size=80,
        symbol_vocab_size=10,  # 0-9
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
        num_heads=4,
        max_frames=256,
    )
    model = MiniJMamba(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params / 1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for frames, masks, labels in train_loader:
            frames, masks, labels = frames.to(device), masks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            _, logits = model(frames, masks)  # (B, T, vocab_size)
            
            # Mean pooling
            logits_masked = logits * masks.unsqueeze(-1).float()
            logits_sum = logits_masked.sum(dim=1)
            lengths = masks.sum(dim=1, keepdim=True).clamp(min=1)
            logits = logits_sum / lengths  # (B, vocab_size)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, masks, labels in val_loader:
                frames, masks, labels = frames.to(device), masks.to(device), labels.to(device)
                _, logits = model(frames, masks)
                
                logits_masked = logits * masks.unsqueeze(-1).float()
                logits_sum = logits_masked.sum(dim=1)
                lengths = masks.sum(dim=1, keepdim=True).clamp(min=1)
                logits = logits_sum / lengths
                
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            suffix = args.output_suffix or ""
            torch.save(model.state_dict(), f"runs/real_speech_best{suffix}.pt")
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, best={best_val_acc:.3f}")
    
    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for frames, masks, labels in test_loader:
            frames, masks, labels = frames.to(device), masks.to(device), labels.to(device)
            _, logits = model(frames, masks)
            
            logits_masked = logits * masks.unsqueeze(-1).float()
            logits_sum = logits_masked.sum(dim=1)
            lengths = masks.sum(dim=1, keepdim=True).clamp(min=1)
            logits = logits_sum / lengths
            
            preds = logits.argmax(dim=-1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    print("\n" + "=" * 60)
    print("Results: Real Speech (Google Speech Commands)")
    print("=" * 60)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Best Val Accuracy: {best_val_acc:.1%}")
    print(f"Test Accuracy: {test_acc:.1%}")
    
    # 保存结果
    results = {
        "dataset": "Google Speech Commands (digits 0-9)",
        "task": "Mirror (digit recognition)",
        "seed": args.seed,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs": args.epochs,
        "device": str(device),
    }
    
    suffix = args.output_suffix or ""
    Path("reports").mkdir(exist_ok=True)
    output_file = f"reports/real_speech_validation{suffix}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # 判断是否成功
    if test_acc > 0.5:
        print(f"\n[OK] SUCCESS: Real speech validation passed ({test_acc:.1%} > 50%)")
    else:
        print(f"\n[!] WARNING: Real speech validation low ({test_acc:.1%})")


if __name__ == "__main__":
    main()

