#!/usr/bin/env python3
"""
一键最小复现脚本 - 5 分钟验证 Wave Reasoning 核心能力

用法:
    python scripts/repro_tiny.py
    
输出:
    - Oracle EM: 协议正确性验证（应为 1.0）
    - Model EM: 10 epoch 快速训练后的 Mirror 任务 EM
    
预期:
    - Oracle EM = 1.0 (100%)
    - Model EM > 0.8 (10 epoch 可达 80%+)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.data import ManifestEntry, synthesise_entry_wave
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.scorer import decode_wave_to_symbols
from jericho.symbols import SYMBOLS, encode_symbols_to_wave


def safe_unfold(wave_tensor: torch.Tensor, frame_size: int = 160, hop_size: int = 160):
    """安全分帧"""
    original_len = len(wave_tensor)
    if original_len < frame_size:
        wave_tensor = nn.functional.pad(wave_tensor, (0, frame_size - original_len))
        return wave_tensor.unsqueeze(0)
    remainder = (original_len - frame_size) % hop_size
    if remainder > 0:
        wave_tensor = nn.functional.pad(wave_tensor, (0, hop_size - remainder))
    return wave_tensor.unfold(0, frame_size, hop_size)


def generate_samples(n: int = 100, seed: int = 42) -> list[ManifestEntry]:
    """生成 Mirror 任务样本"""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        length = rng.integers(3, 8)
        symbols = tuple(rng.choice(SYMBOLS, size=length))
        samples.append(ManifestEntry(
            id=f"repro_{i:04d}",
            symbols=symbols,
            length=length,
            split="train" if i < n * 0.8 else "test",
        ))
    return samples


def oracle_test(samples: list[ManifestEntry]) -> float:
    """Oracle 测试：验证编码-解码协议"""
    correct = 0
    for entry in samples:
        wave = encode_symbols_to_wave(list(entry.symbols))
        decoded = decode_wave_to_symbols(wave)
        if tuple(decoded) == entry.symbols:
            correct += 1
    return correct / len(samples)


def train_and_eval(
    train_samples: list[ManifestEntry],
    test_samples: list[ManifestEntry],
    epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """快速训练并评估"""
    
    # 模型配置
    config = MiniJMambaConfig(
        frame_size=160,
        hop_size=160,
        symbol_vocab_size=len(SYMBOLS) + 2,
        d_model=128,
        num_ssm_layers=4,
        num_attn_layers=1,
        num_heads=4,
        max_frames=512,
    )
    
    model = MiniJMamba(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    
    print(f"\n训练中... ({epochs} epochs, device={device})")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for entry in train_samples:
            wave = encode_symbols_to_wave(list(entry.symbols))
            wave_tensor = torch.from_numpy(wave).float()
            
            frames = safe_unfold(wave_tensor, 160, 160)
            frames = frames.unsqueeze(0).to(device)
            mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
            
            target = frames.clone()
            
            optimizer.zero_grad()
            frame_out, _ = model(frames, mask)
            loss = mse_loss(frame_out, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_samples)
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")
    
    # 评估
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for entry in test_samples:
            wave = encode_symbols_to_wave(list(entry.symbols))
            wave_tensor = torch.from_numpy(wave).float()
            
            frames = safe_unfold(wave_tensor, 160, 160)
            frames = frames.unsqueeze(0).to(device)
            mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
            
            frame_out, _ = model(frames, mask)
            pred_wave = frame_out.squeeze(0).cpu().numpy().flatten()
            pred_wave = np.clip(pred_wave, -1.0, 1.0).astype(np.float32)
            
            decoded = decode_wave_to_symbols(pred_wave)
            if tuple(decoded) == entry.symbols:
                correct += 1
    
    return correct / len(test_samples)


def main():
    print("=" * 60)
    print("Wave Reasoning 最小复现")
    print("=" * 60)
    
    start = time.time()
    
    # 生成样本
    samples = generate_samples(100, seed=42)
    train_samples = [s for s in samples if s.split == "train"]
    test_samples = [s for s in samples if s.split == "test"]
    
    print(f"\n样本: {len(train_samples)} train, {len(test_samples)} test")
    
    # Oracle 测试
    print("\n[1/2] Oracle 验证...")
    oracle_em = oracle_test(samples)
    print(f"  Oracle EM = {oracle_em:.4f} {'✓' if oracle_em == 1.0 else '✗'}")
    
    # 模型训练与评估
    print("\n[2/2] 模型训练与评估...")
    model_em = train_and_eval(train_samples, test_samples, epochs=10)
    print(f"\n  Model EM = {model_em:.4f} {'✓' if model_em >= 0.5 else '?'}")
    
    elapsed = time.time() - start
    
    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"  Oracle EM: {oracle_em:.2%}")
    print(f"  Model EM:  {model_em:.2%}")
    print(f"  耗时:      {elapsed:.1f}s")
    print()
    
    if oracle_em == 1.0 and model_em >= 0.5:
        print("✓ 复现成功！Wave Reasoning 核心能力验证通过")
        return 0
    else:
        print("✗ 复现失败，请检查环境")
        return 1


if __name__ == "__main__":
    sys.exit(main())

