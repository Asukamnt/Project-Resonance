#!/usr/bin/env python3
"""
P0: 真实 TTS 数据 - 使用 pydub 转换 mp3 (v2)

edge-tts 生成的 mp3 文件需要转换为 wav。
使用 pydub + ffmpeg 进行转换。
"""

import asyncio
import json
import os
import random
from pathlib import Path

try:
    import edge_tts
except ImportError:
    os.system("pip install edge-tts -q")
    import edge_tts

try:
    from pydub import AudioSegment
except ImportError:
    os.system("pip install pydub -q")
    from pydub import AudioSegment

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 16000
OUTPUT_DIR = Path("data/tts_task3_v2")
MANIFEST_DIR = Path("manifests/tts_task3_v2")

VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
]


async def generate_audio(text: str, voice: str, output_path: Path):
    """生成单个音频"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path, target_sr: int = 16000):
    """使用 pydub 将 mp3 转换为 wav"""
    audio = AudioSegment.from_mp3(str(mp3_path))
    audio = audio.set_frame_rate(target_sr)
    audio = audio.set_channels(1)  # 单声道
    audio.export(str(wav_path), format="wav")
    return True


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
        
        # 使用 pydub 转换
        convert_mp3_to_wav(input_mp3, input_wav, SAMPLE_RATE)
        convert_mp3_to_wav(output_mp3, output_wav, SAMPLE_RATE)
        
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
    print("P0: Small TTS Dataset with pydub (100 samples)")
    print("=" * 60)
    
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

