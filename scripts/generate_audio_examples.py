#!/usr/bin/env python3
"""
生成音频示例供人工听取验证

对真实语音数据运行 Mini-JMamba，保存输入和预测结果
"""

import json
import random
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig

DIGIT_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def main():
    print("=" * 60)
    print("Generating Audio Examples for Human Listening")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载模型
    model_path = Path("runs/real_speech_best.pt")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run scripts/train_real_speech.py first.")
        return
    
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded.")
    
    # 输出目录
    output_dir = Path("artifacts/audio_examples_real")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集测试样本
    data_dir = Path("data/speech_commands_digits")
    samples = []
    
    for digit_name, digit_val in DIGIT_MAP.items():
        digit_dir = data_dir / digit_name
        if not digit_dir.exists():
            continue
        
        wav_files = list(digit_dir.glob("*.wav"))
        # 取最后 15 个作为测试集
        test_files = wav_files[int(len(wav_files) * 0.85):][:5]  # 每个数字 5 个
        
        for f in test_files:
            samples.append({
                "path": f,
                "label": digit_val,
                "digit_name": digit_name,
            })
    
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:20]  # 只取 20 个示例
    
    print(f"\nProcessing {len(samples)} samples...")
    
    results = []
    correct = 0
    
    for i, sample in enumerate(samples):
        # 加载音频
        waveform, sr = torchaudio.load(sample["path"])
        waveform = waveform.squeeze(0)
        
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        
        # 分帧
        frame_size = 160
        hop_size = 80
        num_frames = (len(waveform) - frame_size) // hop_size + 1
        
        frames = []
        for j in range(num_frames):
            start = j * hop_size
            end = start + frame_size
            if end <= len(waveform):
                frames.append(waveform[start:end])
        
        if len(frames) == 0:
            frames = [waveform[:frame_size]]
        
        frames = torch.stack(frames).unsqueeze(0).to(device)  # (1, T, frame_size)
        mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
        
        # 推理
        with torch.no_grad():
            _, logits = model(frames, mask)
            
            # Mean pooling
            logits_masked = logits * mask.unsqueeze(-1).float()
            logits_sum = logits_masked.sum(dim=1)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            logits = logits_sum / lengths
            
            pred = logits.argmax(dim=-1).item()
            probs = torch.softmax(logits, dim=-1)[0]
        
        is_correct = pred == sample["label"]
        if is_correct:
            correct += 1
        
        # 保存音频
        output_filename = f"{i:02d}_{sample['digit_name']}_pred{DIGIT_NAMES[pred]}_{'OK' if is_correct else 'WRONG'}.wav"
        output_path = output_dir / output_filename
        torchaudio.save(str(output_path), waveform.unsqueeze(0), sr)
        
        results.append({
            "id": i,
            "file": output_filename,
            "true_label": sample["digit_name"],
            "true_label_id": sample["label"],
            "predicted": DIGIT_NAMES[pred],
            "predicted_id": pred,
            "correct": is_correct,
            "confidence": probs[pred].item(),
            "all_probs": {DIGIT_NAMES[k]: probs[k].item() for k in range(10)},
        })
        
        status = "OK" if is_correct else "WRONG"
        print(f"  [{i+1:2d}] {sample['digit_name']:5s} -> {DIGIT_NAMES[pred]:5s} [{status}] (conf: {probs[pred]:.2f})")
    
    # 保存索引
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump({
            "total": len(results),
            "correct": correct,
            "accuracy": correct / len(results),
            "samples": results,
        }, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print(f"Results: {correct}/{len(results)} = {correct/len(results):.1%}")
    print(f"Audio files saved to: {output_dir}")
    print(f"Index saved to: {index_path}")
    print("=" * 60)
    
    # 打印混淆情况
    print("\nConfusion examples (incorrect predictions):")
    for r in results:
        if not r["correct"]:
            print(f"  {r['file']}")
            print(f"    True: {r['true_label']}, Pred: {r['predicted']} (conf: {r['confidence']:.2f})")


if __name__ == "__main__":
    main()

