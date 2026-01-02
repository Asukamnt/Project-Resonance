#!/usr/bin/env python3
"""
评估真实人声 checkpoint（不训练）
用于补充因意外中断导致缺失的验证结果
"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models import MiniJMamba, MiniJMambaConfig
from torch.utils.data import DataLoader, Dataset
import torchaudio


class SpeechCommandsDigits(Dataset):
    """Google Speech Commands digits subset"""
    
    def __init__(self, root: str, split: str = "training", sr: int = 16000, target_length: int = 16000):
        self.sr = sr
        self.target_length = target_length
        self.digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        
        self.samples = []
        root_path = Path(root)
        
        for digit_idx, digit in enumerate(self.digits):
            digit_dir = root_path / digit
            if not digit_dir.exists():
                continue
            
            files = sorted(list(digit_dir.glob("*.wav")))
            n_files = len(files)
            
            if split == "training":
                files = files[:int(n_files * 0.7)]
            elif split == "validation":
                files = files[int(n_files * 0.7):int(n_files * 0.85)]
            else:  # test
                files = files[int(n_files * 0.85):]
            
            for f in files:
                self.samples.append((str(f), digit_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = torchaudio.load(path)
        
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        
        waveform = waveform.mean(dim=0)
        
        if len(waveform) < self.target_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_length - len(waveform)))
        else:
            waveform = waveform[:self.target_length]
        
        return waveform, label


def evaluate_checkpoint(checkpoint_path: str, data_root: str, device: str = "cuda"):
    """评估单个 checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 配置
    config = MiniJMambaConfig(
        frame_size=160,
        hop_size=160,
        symbol_vocab_size=10,
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
        num_heads=4,
        max_frames=100,
        dropout=0.1,
        attn_dropout=0.1,
        use_rope=True,
        use_learnable_pos=False,
    )
    
    model = MiniJMamba(config)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    
    # 数据
    test_ds = SpeechCommandsDigits(data_root, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Test samples: {len(test_ds)}")
    
    # 评估
    correct = 0
    total = 0
    
    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            # 分帧
            frame_size = 160
            hop_size = 160
            batch_size = waveforms.size(0)
            
            frames_list = []
            for wave in waveforms:
                num_frames = (len(wave) - frame_size) // hop_size + 1
                frames = torch.zeros(num_frames, frame_size, device=device)
                for i in range(num_frames):
                    start = i * hop_size
                    frames[i] = wave[start:start+frame_size]
                frames_list.append(frames)
            
            max_frames = max(f.size(0) for f in frames_list)
            padded_frames = torch.zeros(batch_size, max_frames, frame_size, device=device)
            mask = torch.zeros(batch_size, max_frames, dtype=torch.bool, device=device)
            
            for i, frames in enumerate(frames_list):
                padded_frames[i, :frames.size(0)] = frames
                mask[i, :frames.size(0)] = True
            
            # 推理
            logits, _ = model(padded_frames, mask)
            
            # 取最后一帧的预测
            last_indices = mask.sum(dim=1) - 1
            batch_logits = torch.zeros(batch_size, logits.size(-1), device=device)
            for i in range(batch_size):
                batch_logits[i] = logits[i, last_indices[i]]
            
            preds = batch_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/SpeechCommands/speech_commands_v0.02")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    if args.seed:
        torch.manual_seed(args.seed)
    
    accuracy = evaluate_checkpoint(args.checkpoint, args.data_root, args.device)
    
    # 保存结果
    if args.output:
        output_path = args.output
    else:
        ckpt_name = Path(args.checkpoint).stem
        output_path = f"reports/eval_{ckpt_name}.json"
    
    result = {
        "checkpoint": args.checkpoint,
        "test_acc": accuracy,
        "seed": args.seed,
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

