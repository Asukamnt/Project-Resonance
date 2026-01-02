#!/usr/bin/env python3
"""
实时推理演示脚本

功能：
1. 加载训练好的模型
2. 对指定音频文件进行推理
3. 显示预测结果和置信度
4. 可选：播放音频

用法：
  python scripts/demo_realtime.py --audio path/to/audio.wav
  python scripts/demo_realtime.py --interactive  # 交互模式
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig

DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


class WaveformReasoningDemo:
    def __init__(self, model_path: str = "runs/real_speech_best.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
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
        self.model = MiniJMamba(config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.frame_size = 160
        self.hop_size = 80
        
        print(f"Model loaded on {self.device}")
        print(f"Parameters: 0.94M")
    
    def preprocess(self, waveform: torch.Tensor, sr: int) -> tuple:
        """预处理音频"""
        # 重采样到 16kHz
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        
        if waveform.dim() == 2:
            waveform = waveform[0]  # 取第一个声道
        
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
        
        frames = torch.stack(frames).unsqueeze(0).to(self.device)
        mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=self.device)
        
        return frames, mask
    
    def predict(self, audio_path: str) -> dict:
        """对音频文件进行推理"""
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        frames, mask = self.preprocess(waveform, sr)
        
        # 推理
        with torch.no_grad():
            _, logits = self.model(frames, mask)
            
            # Mean pooling
            logits_masked = logits * mask.unsqueeze(-1).float()
            logits_sum = logits_masked.sum(dim=1)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            logits = logits_sum / lengths
            
            probs = torch.softmax(logits, dim=-1)[0]
            pred = probs.argmax().item()
        
        return {
            "prediction": DIGIT_NAMES[pred],
            "prediction_id": pred,
            "confidence": probs[pred].item(),
            "all_probs": {DIGIT_NAMES[i]: probs[i].item() for i in range(10)},
            "duration_ms": len(waveform[0]) / sr * 1000,
            "num_frames": frames.size(1),
        }
    
    def demo_single(self, audio_path: str):
        """演示单个音频"""
        print(f"\n{'='*50}")
        print(f"Input: {audio_path}")
        print(f"{'='*50}")
        
        result = self.predict(audio_path)
        
        print(f"\nPrediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Duration: {result['duration_ms']:.0f} ms")
        print(f"Frames: {result['num_frames']}")
        
        print(f"\nTop-3 probabilities:")
        sorted_probs = sorted(result['all_probs'].items(), key=lambda x: -x[1])[:3]
        for name, prob in sorted_probs:
            bar = "#" * int(prob * 30)
            print(f"  {name:5s}: {prob:.1%} {bar}")
        
        return result
    
    def demo_batch(self, audio_dir: str, limit: int = 10):
        """批量演示"""
        audio_dir = Path(audio_dir)
        wav_files = list(audio_dir.rglob("*.wav"))[:limit]
        
        print(f"\nProcessing {len(wav_files)} files from {audio_dir}...")
        
        for f in wav_files:
            self.demo_single(str(f))


def main():
    parser = argparse.ArgumentParser(description="Waveform Reasoning Demo")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--dir", type=str, help="Directory of audio files")
    parser.add_argument("--model", type=str, default="runs/real_speech_best.pt")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    
    print("=" * 60)
    print("   Waveform Reasoning Demo - Mini-JMamba")
    print("=" * 60)
    
    demo = WaveformReasoningDemo(args.model)
    
    if args.audio:
        demo.demo_single(args.audio)
    elif args.dir:
        demo.demo_batch(args.dir, args.limit)
    else:
        # 默认：演示一些示例
        examples_dir = Path("artifacts/audio_examples_real")
        if examples_dir.exists():
            demo.demo_batch(str(examples_dir), limit=5)
        else:
            print("\nNo audio specified. Use --audio or --dir")
            print("Example:")
            print("  python scripts/demo_realtime.py --audio data/speech_commands_digits/zero/0001.wav")


if __name__ == "__main__":
    main()

