#!/usr/bin/env python3
"""
P0: 真实 TTS 数据 - 使用 ffmpeg 直接转换 (v3)

直接用 subprocess 调用 ffmpeg 转换 mp3 到 wav。
"""

import asyncio
import json
import os
import random
import subprocess
from pathlib import Path

try:
    import edge_tts
except ImportError:
    os.system("pip install edge-tts -q")
    import edge_tts

SAMPLE_RATE = 16000
OUTPUT_DIR = Path("data/tts_task3_v3")
MANIFEST_DIR = Path("manifests/tts_task3_v3")

VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
]


async def generate_audio(text: str, voice: str, output_path: Path):
    """生成单个音频"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))


def convert_mp3_to_wav_ffmpeg(mp3_path: Path, wav_path: Path, target_sr: int = 16000):
    """使用 ffmpeg 将 mp3 转换为 wav"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", str(target_sr), "-ac", "1", str(wav_path)],
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg and add to PATH.")
        return False


async def process_sample(i: int, split_dir: Path, split_name: str):
    """处理单个样本"""
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    mod = random.randint(2, 9)
    result = (a + b) % mod
    
    input_text = f"{a} plus {b} mod {mod}"
    output_text = str(result)
    
    voice = random.choice(VOICES)
    
    input_mp3 = split_dir / f"{i:04d}_input.mp3"
    output_mp3 = split_dir / f"{i:04d}_output.mp3"
    input_wav = split_dir / f"{i:04d}_input.wav"
    output_wav = split_dir / f"{i:04d}_output.wav"
    
    try:
        await generate_audio(input_text, voice, input_mp3)
        await generate_audio(output_text, voice, output_mp3)
        
        # 使用 ffmpeg 转换
        if not convert_mp3_to_wav_ffmpeg(input_mp3, input_wav, SAMPLE_RATE):
            return None
        if not convert_mp3_to_wav_ffmpeg(output_mp3, output_wav, SAMPLE_RATE):
            return None
        
        # 删除 mp3
        input_mp3.unlink()
        output_mp3.unlink()
        
        return {
            "id": f"tts_{split_name}_{i:04d}",
            "input_audio": str(input_wav),
            "output_audio": str(output_wav),
            "expression": f"{a}+{b}%{mod}",
            "result": result,
            "voice": voice,
        }
    except Exception as e:
        print(f"  Error sample {i}: {e}")
        return None


async def main():
    print("=" * 60)
    print("P0: TTS Dataset with ffmpeg (100 samples)")
    print("=" * 60)
    
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("ffmpeg found!")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg not found. Please install ffmpeg.")
        print("  Windows: winget install ffmpeg")
        print("  Or download from https://ffmpeg.org/download.html")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    
    splits = {"train": 70, "val": 15, "test": 15}
    
    for split_name, num in splits.items():
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"\n{split_name}: generating {num} samples...")
        
        entries = []
        for i in range(num):
            entry = await process_sample(i, split_dir, split_name)
            if entry:
                entries.append(entry)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{num} done")
        
        manifest_path = MANIFEST_DIR / f"{split_name}.jsonl"
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"  Saved {len(entries)} to {manifest_path}")
    
    print("\n" + "=" * 60)
    print("Done! Data in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


