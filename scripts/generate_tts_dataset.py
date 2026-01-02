#!/usr/bin/env python3
"""
P0: 真实采集数据 - TTS 生成人声数字数据集

目标：用 TTS 生成 Task3 Mod 的真实人声版本，验证模型在真实语音上的推理能力。

数据格式：
- 输入：人声念出 "3 + 5 % 7" 
- 输出：人声念出 "1"（结果）

使用 edge-tts（微软 Azure TTS，高质量免费）
"""

import asyncio
import json
import os
import random
from pathlib import Path

import edge_tts
import torchaudio
import torch

# 配置
SAMPLE_RATE = 16000
OUTPUT_DIR = Path("data/tts_task3")
MANIFEST_DIR = Path("manifests/tts_task3")

# TTS 语音选项（多样性）
VOICES = [
    "en-US-GuyNeural",      # 美式男声
    "en-US-JennyNeural",    # 美式女声
    "en-GB-RyanNeural",     # 英式男声
    "en-GB-SoniaNeural",    # 英式女声
]

# 语速变化
RATES = ["-10%", "+0%", "+10%"]


async def generate_tts_audio(text: str, voice: str, rate: str, output_path: Path):
    """生成单个 TTS 音频"""
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(str(output_path))


def text_to_speech_expression(a: int, b: int, mod: int) -> tuple[str, str]:
    """将算式转换为语音文本"""
    # 输入：念出算式
    input_text = f"{a} plus {b} modulo {mod}"
    # 输出：念出结果
    result = (a + b) % mod
    output_text = str(result)
    return input_text, output_text, result


async def generate_dataset(
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
            
            input_text, output_text, result = text_to_speech_expression(a, b, mod)
            
            # 随机选择语音和语速
            voice = random.choice(VOICES)
            rate = random.choice(RATES)
            
            # 生成输入音频
            input_path = split_dir / f"{i:05d}_input.mp3"
            output_path = split_dir / f"{i:05d}_output.mp3"
            
            try:
                await generate_tts_audio(input_text, voice, rate, input_path)
                await generate_tts_audio(output_text, voice, rate, output_path)
            except Exception as e:
                print(f"  Error generating sample {i}: {e}")
                continue
            
            # 转换为 wav（统一格式）
            input_wav = split_dir / f"{i:05d}_input.wav"
            output_wav = split_dir / f"{i:05d}_output.wav"
            
            try:
                # 加载并转换
                waveform, sr = torchaudio.load(input_path)
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                    waveform = resampler(waveform)
                # 转单声道
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                torchaudio.save(input_wav, waveform, SAMPLE_RATE)
                
                waveform, sr = torchaudio.load(output_path)
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                torchaudio.save(output_wav, waveform, SAMPLE_RATE)
                
                # 删除 mp3
                input_path.unlink()
                output_path.unlink()
                
            except Exception as e:
                print(f"  Error converting sample {i}: {e}")
                continue
            
            # Manifest 条目
            manifest_entries.append({
                "id": f"tts_{split_name}_{i:05d}",
                "input_audio": str(input_wav.relative_to(OUTPUT_DIR.parent.parent)),
                "output_audio": str(output_wav.relative_to(OUTPUT_DIR.parent.parent)),
                "expression": f"{a}+{b}%{mod}",
                "result": result,
                "input_text": input_text,
                "output_text": output_text,
                "voice": voice,
                "rate": rate,
            })
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        # 保存 manifest
        manifest_path = MANIFEST_DIR / f"{split_name}.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"  Saved {len(manifest_entries)} entries to {manifest_path}")


async def main():
    print("=" * 60)
    print("P0: TTS Dataset Generation for Task3 Mod")
    print("=" * 60)
    
    # 检查 edge-tts 是否安装
    try:
        import edge_tts
    except ImportError:
        print("Installing edge-tts...")
        os.system("pip install edge-tts")
        import edge_tts
    
    # 生成数据集
    await generate_dataset(
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
    asyncio.run(main())

