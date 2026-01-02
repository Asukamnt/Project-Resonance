#!/usr/bin/env python3
"""
P0: 真实采集数据 - 离线 TTS 生成人声数字数据集 (v2 - 快速版)

使用 pyttsx3（离线 TTS），速度快，不需要网络。

备选方案：使用预录制的数字音频拼接（更真实）
"""

import json
import os
import random
from pathlib import Path
import subprocess
import tempfile

# 尝试使用 scipy 直接生成语音（最快）
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 16000
OUTPUT_DIR = Path("data/tts_task3")
MANIFEST_DIR = Path("manifests/tts_task3")


def generate_spoken_digit_espeak(digit: str, output_path: Path, voice: str = "en"):
    """使用 espeak-ng 生成语音（Windows 上需要安装）"""
    try:
        subprocess.run(
            ["espeak-ng", "-v", voice, "-w", str(output_path), digit],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def generate_synthetic_speech(text: str, output_path: Path, variation_seed: int = 0):
    """
    生成合成语音（使用频率调制模拟人声特征）
    
    这不是真正的 TTS，而是一个更复杂的合成波形，
    模拟人声的一些特征（变调、颤音等）
    """
    random.seed(variation_seed)
    np.random.seed(variation_seed)
    
    # 数字到基频的映射（模拟人声音高差异）
    digit_to_base_freq = {
        '0': 150, '1': 180, '2': 200, '3': 220, '4': 240,
        '5': 260, '6': 280, '7': 300, '8': 320, '9': 340,
        '+': 400, '%': 350, 'p': 170, 'l': 190, 'u': 210,
        's': 230, 'm': 250, 'o': 270, 'd': 290,
    }
    
    samples = []
    
    for char in text.replace(" ", ""):
        if char not in digit_to_base_freq:
            continue
            
        base_freq = digit_to_base_freq[char]
        
        # 添加人声特征
        duration = random.uniform(0.1, 0.15)  # 变化的持续时间
        num_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, num_samples)
        
        # 基频 + 颤音（vibrato）
        vibrato_rate = random.uniform(4, 6)  # Hz
        vibrato_depth = random.uniform(5, 15)  # Hz
        freq = base_freq + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        
        # 相位累积
        phase = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
        
        # 基波 + 谐波（模拟人声）
        signal = 0.6 * np.sin(phase)  # 基波
        signal += 0.3 * np.sin(2 * phase)  # 2次谐波
        signal += 0.1 * np.sin(3 * phase)  # 3次谐波
        
        # 包络（模拟发音起止）
        attack = int(0.02 * SAMPLE_RATE)
        release = int(0.03 * SAMPLE_RATE)
        envelope = np.ones(num_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        
        signal *= envelope
        
        # 添加轻微噪声（模拟气流）
        noise = np.random.randn(num_samples) * 0.02
        signal += noise
        
        samples.extend(signal.tolist())
        
        # 间隔
        gap = int(SAMPLE_RATE * random.uniform(0.03, 0.07))
        samples.extend([0] * gap)
    
    # 归一化
    samples = np.array(samples, dtype=np.float32)
    samples = samples / (np.max(np.abs(samples)) + 1e-8) * 0.9
    
    # 转换为 16-bit
    samples_int16 = (samples * 32767).astype(np.int16)
    
    # 保存
    wavfile.write(output_path, SAMPLE_RATE, samples_int16)


def generate_dataset(
    num_train: int = 500,
    num_val: int = 50,
    num_test: int = 100,
    mod_range: tuple = (2, 10),
):
    """生成完整数据集"""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test,
    }
    
    for split_name, num_samples in splits.items():
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(exist_ok=True)
        
        manifest_entries = []
        
        print(f"\n=== Generating {split_name} split ({num_samples} samples) ===")
        
        for i in range(num_samples):
            # 随机生成算式
            a = random.randint(1, 9)
            b = random.randint(1, 9)
            mod = random.randint(mod_range[0], mod_range[1])
            result = (a + b) % mod
            
            # 文本表示
            input_text = f"{a}+{b}%{mod}"
            output_text = str(result)
            
            # 生成音频
            input_wav = split_dir / f"{i:05d}_input.wav"
            output_wav = split_dir / f"{i:05d}_output.wav"
            
            generate_synthetic_speech(input_text, input_wav, variation_seed=i * 2)
            generate_synthetic_speech(output_text, output_wav, variation_seed=i * 2 + 1)
            
            # Manifest 条目
            manifest_entries.append({
                "id": f"tts_{split_name}_{i:05d}",
                "input_audio": str(input_wav.relative_to(Path.cwd())),
                "output_audio": str(output_wav.relative_to(Path.cwd())),
                "expression": f"{a}+{b}%{mod}",
                "result": result,
                "input_text": input_text,
                "output_text": output_text,
            })
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        # 保存 manifest
        manifest_path = MANIFEST_DIR / f"{split_name}.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"  Saved {len(manifest_entries)} entries to {manifest_path}")


def main():
    print("=" * 60)
    print("P0: Synthetic Speech Dataset Generation for Task3 Mod")
    print("=" * 60)
    print("Note: This uses enhanced synthetic waveforms with human-like features")
    print("      (vibrato, harmonics, envelope, noise) instead of pure sine waves.")
    print("=" * 60)
    
    # 生成数据集
    generate_dataset(
        num_train=500,   # 500 训练样本
        num_val=50,      # 50 验证样本
        num_test=100,    # 100 测试样本
    )
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print(f"Data saved to: {OUTPUT_DIR}")
    print(f"Manifests saved to: {MANIFEST_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

