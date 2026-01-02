#!/usr/bin/env python3
"""
下载 Google Speech Commands 数据集中的数字部分

用于真实数据验证实验
"""

import os
from pathlib import Path

import torch
import torchaudio

# 目标目录
OUTPUT_DIR = Path("data/speech_commands_digits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数字词汇
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def main():
    print("=" * 60)
    print("Downloading Google Speech Commands Dataset")
    print("=" * 60)
    
    # 下载数据集
    print("Downloading... (this may take a while)")
    dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root="data/",
        download=True,
        subset="training",
    )
    
    print(f"Total samples: {len(dataset)}")
    
    # 统计数字样本
    digit_counts = {d: 0 for d in DIGITS}
    digit_samples = {d: [] for d in DIGITS}
    
    for i in range(len(dataset)):
        waveform, sample_rate, label, speaker_id, utterance_number = dataset[i]
        if label in DIGITS:
            digit_counts[label] += 1
            if len(digit_samples[label]) < 2500:  # 每个数字最多 2500 个样本（全部）
                digit_samples[label].append({
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                    "label": label,
                    "speaker_id": speaker_id,
                    "index": i,
                })
    
    print("\nDigit counts:")
    for d, count in digit_counts.items():
        print(f"  {d}: {count}")
    
    # 保存数字样本
    total_saved = 0
    for digit, samples in digit_samples.items():
        digit_dir = OUTPUT_DIR / digit
        digit_dir.mkdir(exist_ok=True)
        
        for j, sample in enumerate(samples):
            wav_path = digit_dir / f"{j:04d}.wav"
            torchaudio.save(str(wav_path), sample["waveform"], sample["sample_rate"])
            total_saved += 1
    
    print(f"\nSaved {total_saved} samples to {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()

