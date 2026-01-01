#!/usr/bin/env python3
"""
真实硬件/信道验证脚本

三种方案：
A) 硬件录放：扬声器播放 → 话筒录音 → 评测
B) 信道模拟：添加 RIR（房间脉冲响应）+ AWGN 噪声
C) 参考 TTS：使用 TTS digits 结果作为真实语音参考

用法:
    # 方案B（模拟信道，无需硬件）
    python scripts/real_hardware_test.py --mode simulate --snr 20
    
    # 方案A（硬件录放）
    python scripts/real_hardware_test.py --mode hardware --play-device 0 --record-device 1
    
    # 查看设备列表
    python scripts/real_hardware_test.py --list-devices
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.scorer import decode_wave_to_symbols, exact_match

# Audio 采样率（标准 16kHz）
SAMPLE_RATE = 16000

# 可选依赖
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    from scipy import signal
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class ChannelConfig:
    """信道配置"""
    snr_db: float = 20.0           # 信噪比
    reverb_decay: float = 0.3      # 混响衰减系数
    reverb_delay_ms: float = 50.0  # 混响延迟（毫秒）
    lowpass_hz: float = 7000.0     # 低通滤波频率
    highpass_hz: float = 100.0     # 高通滤波频率


def add_awgn(wave: np.ndarray, snr_db: float) -> np.ndarray:
    """添加加性高斯白噪声"""
    signal_power = np.mean(wave ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(wave)) * np.sqrt(noise_power)
    return wave + noise


def add_simple_reverb(wave: np.ndarray, decay: float, delay_ms: float, sr: int) -> np.ndarray:
    """添加简单混响（单次反射）"""
    delay_samples = int(delay_ms * sr / 1000)
    if delay_samples >= len(wave):
        return wave
    
    output = wave.copy()
    output[delay_samples:] += decay * wave[:-delay_samples]
    return output


def apply_bandpass(wave: np.ndarray, low_hz: float, high_hz: float, sr: int) -> np.ndarray:
    """应用带通滤波器（模拟话筒频率响应）"""
    if not HAS_SCIPY:
        return wave
    
    nyq = sr / 2
    low = max(low_hz / nyq, 0.01)
    high = min(high_hz / nyq, 0.99)
    
    if low >= high:
        return wave
    
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, wave)


def simulate_channel(wave: np.ndarray, config: ChannelConfig, sr: int = SAMPLE_RATE) -> np.ndarray:
    """模拟真实信道（混响 + 滤波 + 噪声）"""
    # 1. 混响
    wave = add_simple_reverb(wave, config.reverb_decay, config.reverb_delay_ms, sr)
    
    # 2. 带通滤波（话筒频率响应）
    wave = apply_bandpass(wave, config.highpass_hz, config.lowpass_hz, sr)
    
    # 3. 加噪声
    wave = add_awgn(wave, config.snr_db)
    
    # 4. 归一化
    wave = wave / (np.max(np.abs(wave)) + 1e-8)
    
    return wave


def play_and_record(
    wave: np.ndarray,
    play_device: int,
    record_device: int,
    sr: int = SAMPLE_RATE,
    extra_duration: float = 0.5
) -> np.ndarray:
    """播放并录制（需要 sounddevice）"""
    if not HAS_SOUNDDEVICE:
        raise RuntimeError("sounddevice 未安装，请运行: pip install sounddevice")
    
    # 计算录制时长（播放时长 + 额外余量）
    play_duration = len(wave) / sr
    record_duration = play_duration + extra_duration
    record_samples = int(record_duration * sr)
    
    # 同时播放和录制
    recorded = sd.playrec(
        wave.astype(np.float32),
        samplerate=sr,
        channels=1,
        device=(play_device, record_device),
        blocking=True
    )
    
    return recorded.flatten()


def evaluate_with_channel(
    entries: List[ManifestEntry],
    channel_fn,
    limit: Optional[int] = None
) -> dict:
    """使用指定信道函数评估"""
    if limit:
        entries = entries[:limit]
    
    correct = 0
    total = 0
    results = []
    
    for entry in entries:
        # 生成原始波形
        original_wave = synthesise_entry_wave(entry)
        
        # 通过信道
        channel_wave = channel_fn(original_wave)
        
        # 解码
        decoded = decode_wave_to_symbols(channel_wave)
        
        # 评估
        target = entry.symbols
        em = exact_match(decoded, target)
        
        if em:
            correct += 1
        total += 1
        
        results.append({
            "id": entry.example_id,
            "target": target,
            "decoded": decoded,
            "em": em,
        })
    
    return {
        "total": total,
        "correct": correct,
        "em": correct / total if total > 0 else 0,
        "samples": results[:10],  # 只保存前10个样本
    }


def list_audio_devices():
    """列出可用的音频设备"""
    if not HAS_SOUNDDEVICE:
        print("sounddevice 未安装，请运行: pip install sounddevice")
        return
    
    print("\n可用的音频设备：")
    print(sd.query_devices())


def run_simulation_test(
    manifest_path: Path,
    config: ChannelConfig,
    limit: Optional[int] = None,
    split: str = "iid_test"
) -> dict:
    """运行信道模拟测试"""
    print(f"\n{'='*60}")
    print("信道模拟测试")
    print(f"{'='*60}")
    print(f"SNR: {config.snr_db} dB")
    print(f"混响: decay={config.reverb_decay}, delay={config.reverb_delay_ms}ms")
    print(f"带通: {config.highpass_hz}-{config.lowpass_hz} Hz")
    
    # 加载数据
    entries = read_manifest(manifest_path)
    entries = [e for e in entries if e.split == split]
    print(f"样本数: {len(entries)}")
    
    # 定义信道函数
    def channel_fn(wave):
        return simulate_channel(wave, config)
    
    # 评估
    result = evaluate_with_channel(entries, channel_fn, limit)
    
    print(f"\n结果: EM = {result['em']*100:.1f}% ({result['correct']}/{result['total']})")
    
    return {
        "mode": "simulate",
        "config": {
            "snr_db": config.snr_db,
            "reverb_decay": config.reverb_decay,
            "reverb_delay_ms": config.reverb_delay_ms,
            "lowpass_hz": config.lowpass_hz,
            "highpass_hz": config.highpass_hz,
        },
        "split": split,
        **result,
    }


def run_hardware_test(
    manifest_path: Path,
    play_device: int,
    record_device: int,
    limit: Optional[int] = None,
    split: str = "iid_test"
) -> dict:
    """运行硬件录放测试"""
    if not HAS_SOUNDDEVICE:
        raise RuntimeError("sounddevice 未安装")
    
    print(f"\n{'='*60}")
    print("硬件录放测试")
    print(f"{'='*60}")
    print(f"播放设备: {play_device}")
    print(f"录音设备: {record_device}")
    
    # 加载数据
    entries = read_manifest(manifest_path)
    entries = [e for e in entries if e.split == split]
    print(f"样本数: {len(entries)}")
    
    # 定义信道函数
    def channel_fn(wave):
        return play_and_record(wave, play_device, record_device)
    
    # 评估
    result = evaluate_with_channel(entries, channel_fn, limit)
    
    print(f"\n结果: EM = {result['em']*100:.1f}% ({result['correct']}/{result['total']})")
    
    return {
        "mode": "hardware",
        "play_device": play_device,
        "record_device": record_device,
        "split": split,
        **result,
    }


def run_snr_sweep(
    manifest_path: Path,
    snr_values: List[float],
    limit: Optional[int] = None,
    split: str = "iid_test"
) -> dict:
    """SNR 扫描测试"""
    print(f"\n{'='*60}")
    print("SNR 扫描测试")
    print(f"{'='*60}")
    print(f"SNR 值: {snr_values}")
    
    results = []
    for snr in snr_values:
        config = ChannelConfig(snr_db=snr)
        result = run_simulation_test(manifest_path, config, limit, split)
        results.append({
            "snr_db": snr,
            "em": result["em"],
        })
        print(f"SNR={snr:5.1f} dB -> EM={result['em']*100:.1f}%")
    
    return {
        "mode": "snr_sweep",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="真实硬件/信道验证")
    parser.add_argument("--mode", type=str, 
                        choices=["simulate", "hardware", "snr_sweep"],
                        default="simulate", help="测试模式")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest 路径（默认自动查找 task1）")
    parser.add_argument("--split", type=str, default="iid_test",
                        help="评估的 split")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制样本数（调试用）")
    
    # 信道模拟参数
    parser.add_argument("--snr", type=float, default=20.0,
                        help="信噪比 (dB)")
    parser.add_argument("--reverb-decay", type=float, default=0.3,
                        help="混响衰减系数")
    parser.add_argument("--reverb-delay", type=float, default=50.0,
                        help="混响延迟 (ms)")
    
    # 硬件参数
    parser.add_argument("--play-device", type=int, default=0,
                        help="播放设备 ID")
    parser.add_argument("--record-device", type=int, default=0,
                        help="录音设备 ID")
    parser.add_argument("--list-devices", action="store_true",
                        help="列出可用的音频设备")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    # 查找 manifest
    if args.manifest is None:
        for path in [Path("manifests/task1.jsonl"), Path("manifests/task1_mirror.jsonl")]:
            if path.exists():
                args.manifest = path
                break
        if args.manifest is None:
            print("ERROR: 找不到 task1 manifest")
            sys.exit(1)
    
    print(f"Manifest: {args.manifest}")
    
    # 确保输出目录存在
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # 运行测试
    if args.mode == "simulate":
        config = ChannelConfig(
            snr_db=args.snr,
            reverb_decay=args.reverb_decay,
            reverb_delay_ms=args.reverb_delay,
        )
        result = run_simulation_test(args.manifest, config, args.limit, args.split)
        
    elif args.mode == "hardware":
        result = run_hardware_test(
            args.manifest, 
            args.play_device, 
            args.record_device,
            args.limit,
            args.split
        )
        
    elif args.mode == "snr_sweep":
        snr_values = [40, 30, 20, 15, 10, 5, 0]
        result = run_snr_sweep(args.manifest, snr_values, args.limit, args.split)
    
    # 保存结果
    result["timestamp"] = datetime.now().isoformat()
    result["manifest"] = str(args.manifest)
    
    output_path = reports_dir / f"real_channel_{args.mode}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

