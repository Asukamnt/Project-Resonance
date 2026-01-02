#!/usr/bin/env python3
"""
Demo 视频录制脚本

创建一个可视化界面，展示：
1. 输入波形
2. 模型推理过程
3. 预测结果和置信度

用法：
  python scripts/demo_video_script.py

输出：
  - 终端动画演示
  - 可保存为 GIF/视频
"""

import time
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig

DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def clear_screen():
    print("\033[2J\033[H", end="")


def print_waveform_ascii(waveform, width=60):
    """打印 ASCII 波形图"""
    if len(waveform) == 0:
        return
    
    # 下采样
    step = max(1, len(waveform) // width)
    samples = waveform[::step][:width]
    
    # 归一化到 0-10
    if samples.max() - samples.min() > 0:
        normalized = ((samples - samples.min()) / (samples.max() - samples.min()) * 10).int()
    else:
        normalized = torch.zeros_like(samples).int()
    
    # 打印
    for level in range(10, -1, -1):
        line = ""
        for val in normalized:
            if val >= level:
                line += "#"
            else:
                line += " "
        if level == 5:
            print(f"  {line} -")
        else:
            print(f"  {line}")


def print_confidence_bar(probs, highlight_idx):
    """打印置信度条形图"""
    print("\n  Confidence Distribution:")
    print("  " + "-" * 50)
    
    for i, name in enumerate(DIGIT_NAMES):
        prob = probs[i]
        bar_len = int(prob * 40)
        bar = "#" * bar_len
        
        if i == highlight_idx:
            print(f"  \033[92m{name:5s} | {bar:<40s} {prob:5.1%} <-\033[0m")
        else:
            print(f"  {name:5s} | {bar:<40s} {prob:5.1%}")
    
    print("  " + "-" * 50)


def demo_single_file(model, audio_path, device, delay=0.5):
    """演示单个文件的推理过程"""
    clear_screen()
    
    # 标题
    print("\n" + "=" * 60)
    print("   [*] Waveform Reasoning Demo - Mini-JMamba")
    print("=" * 60)
    print(f"\n  Model: Mini-JMamba (0.94M params)")
    print(f"  Task: Digit Recognition (0-9)")
    print(f"  Data: Real Human Voice (Google Speech Commands)")
    
    # 加载音频
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.squeeze(0)
    
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    
    # 从文件名提取真实标签
    true_label = Path(audio_path).parent.name
    
    print(f"\n  Input: {Path(audio_path).name}")
    print(f"  True Label: {true_label.upper()}")
    print(f"  Duration: {len(waveform)/sr*1000:.0f} ms")
    
    # 显示波形
    print("\n  Input Waveform:")
    print_waveform_ascii(waveform)
    
    time.sleep(delay)
    
    # 推理动画
    print("\n  Processing", end="", flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print(" Done!")
    
    # 分帧
    frame_size = 160
    hop_size = 80
    num_frames = (len(waveform) - frame_size) // hop_size + 1
    
    frames = []
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        if end <= len(waveform):
            frames.append(waveform[start:end])
    
    if len(frames) == 0:
        frames = [waveform[:frame_size]]
    
    frames = torch.stack(frames).unsqueeze(0).to(device)
    mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
    
    # 推理
    with torch.no_grad():
        _, logits = model(frames, mask)
        
        logits_masked = logits * mask.unsqueeze(-1).float()
        logits_sum = logits_masked.sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        logits = logits_sum / lengths
        
        probs = torch.softmax(logits, dim=-1)[0].cpu()
        pred = probs.argmax().item()
    
    # 显示结果
    print_confidence_bar(probs, pred)
    
    # 最终结果
    is_correct = DIGIT_NAMES[pred] == true_label
    status = "CORRECT" if is_correct else "WRONG"
    color = "\033[92m" if is_correct else "\033[91m"
    
    print(f"\n  {color}{'-' * 50}\033[0m")
    print(f"  {color}  PREDICTION: {DIGIT_NAMES[pred].upper()}\033[0m")
    print(f"  {color}  CONFIDENCE: {probs[pred]:.1%}\033[0m")
    print(f"  {color}  STATUS: {status}\033[0m")
    print(f"  {color}{'-' * 50}\033[0m")
    
    return is_correct


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Auto mode, no interaction")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between samples (sec)")
    args = parser.parse_args()
    
    print("Loading model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = MiniJMambaConfig(
        frame_size=160,
        hop_size=80,
        symbol_vocab_size=10,
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
        num_heads=4,
        max_frames=256,
    )
    model = MiniJMamba(config).to(device)
    model.load_state_dict(torch.load("runs/real_speech_best.pt", map_location=device))
    model.eval()
    
    # 收集示例
    data_dir = Path("data/speech_commands_digits")
    examples = []
    
    for digit_name in DIGIT_NAMES:
        digit_dir = data_dir / digit_name
        if digit_dir.exists():
            wav_files = list(digit_dir.glob("*.wav"))[:3]  # 每个数字 3 个
            examples.extend(wav_files)
    
    # 随机选择 10 个
    import random
    random.seed(42)
    random.shuffle(examples)
    examples = examples[:10]
    
    print(f"\nReady! Will demo {len(examples)} samples.")
    
    if not args.auto:
        print("Press Enter to start...")
        input()
    
    correct = 0
    total = len(examples)
    
    for i, audio_path in enumerate(examples):
        if demo_single_file(model, str(audio_path), device, delay=0.3):
            correct += 1
        
        print(f"\n  [{i+1}/{total}] Running accuracy: {correct}/{i+1} = {correct/(i+1):.1%}")
        
        if args.auto:
            time.sleep(args.delay)
        else:
            print("\n  Press Enter for next sample (or 'q' to quit)...")
            user_input = input()
            if user_input.lower() == 'q':
                break
    
    # 最终汇总
    clear_screen()
    print("\n" + "=" * 60)
    print("   Demo Complete!")
    print("=" * 60)
    print(f"\n  Total Samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {correct/total:.1%}")
    print("\n  Mini-JMamba successfully demonstrates")
    print("  waveform reasoning on real human speech!")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
