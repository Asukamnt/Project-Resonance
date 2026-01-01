#!/usr/bin/env python3
"""
TSAE Frequency Range Dependency Test

验证核心预测：TSAE 最优校准点随任务频率范围可预测地移动。

如果成立 → TSAE 从"发现"升级为"规律"
如果不成立 → TSAE 可能只是训练偶然

实验设计：
- 用不同频率范围生成波形
- 测试 time-stretch 响应
- 看最优点是否移动

用法:
    python scripts/tsae_frequency_range_test.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jericho.scorer import decode_wave_to_symbols


def time_stretch(wave: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return wave
    original_len = len(wave)
    new_len = int(original_len * factor)
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, new_len)
    stretched = np.interp(x_new, x_old, wave).astype(np.float32)
    if len(stretched) > original_len:
        return stretched[:original_len]
    return np.pad(stretched, (0, original_len - len(stretched)))


def generate_tone(freq: float, duration: float, sr: int = 16000, phase: float = 0.0) -> np.ndarray:
    """Generate a pure sine wave at given frequency"""
    t = np.arange(int(duration * sr)) / sr
    return (0.8 * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def test_frequency_range(
    base_freq: float,
    symbol_duration: float = 0.05,
    sr: int = 16000,
    n_samples: int = 100,
) -> dict:
    """Test TSAE at a specific frequency range"""
    
    stretch_factors = [0.92, 0.95, 0.98, 1.00, 1.02, 1.05, 1.08, 1.10]
    results = {}
    
    for factor in stretch_factors:
        correct = 0
        total = 0
        
        for _ in range(n_samples):
            # Generate a simple test tone
            wave = generate_tone(base_freq, symbol_duration, sr)
            wave = time_stretch(wave, factor)
            
            # Decode
            decoded = decode_wave_to_symbols(wave)
            
            # Check if any symbol was decoded (simple test)
            if decoded:
                correct += 1
            total += 1
        
        results[f"{factor:.2f}x"] = correct / total if total > 0 else 0
    
    # Find optimal stretch
    best_factor = max(results.keys(), key=lambda k: results[k])
    results["optimal_stretch"] = float(best_factor.replace("x", ""))
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/tsae_frequency_range.json"))
    args = parser.parse_args()
    
    # Test different frequency ranges
    # Using frequencies similar to our symbol encoding
    frequency_ranges = {
        "low_freq": 300,      # Low frequency (like digit 0-2)
        "mid_freq": 600,      # Mid frequency (like digit 5-7)
        "high_freq": 1000,    # High frequency (like brackets/%)
        "very_high": 1500,    # Very high (like V/X in Task2)
    }
    
    results = {
        "experiment": "tsae_frequency_range",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Optimal stretch point shifts predictably with frequency range",
        "ranges": {},
    }
    
    print("=" * 70)
    print("TSAE FREQUENCY RANGE DEPENDENCY TEST")
    print("=" * 70)
    print()
    print("Hypothesis: If internal time base exists, optimal stretch should")
    print("shift predictably as frequency range changes.")
    print()
    print("-" * 70)
    print(f"{'Freq Range':<15} {'Base Freq':<12} {'Optimal Stretch':<15} {'Decode Rate @ 1.0x'}")
    print("-" * 70)
    
    optimal_stretches = []
    base_freqs = []
    
    for name, freq in frequency_ranges.items():
        r = test_frequency_range(freq, n_samples=50)
        results["ranges"][name] = {
            "base_freq": freq,
            "optimal_stretch": r["optimal_stretch"],
            "decode_rates": {k: v for k, v in r.items() if k != "optimal_stretch"},
        }
        
        decode_at_1 = r.get("1.00x", 0)
        opt = r["optimal_stretch"]
        print(f"{name:<15} {freq:<12} {opt:.2f}x           {decode_at_1:.1%}")
        
        optimal_stretches.append(opt)
        base_freqs.append(freq)
    
    print("-" * 70)
    
    # Analyze trend
    print()
    print("ANALYSIS")
    print("-" * 70)
    
    # Check if there's a correlation between frequency and optimal stretch
    if len(set(optimal_stretches)) > 1:
        # Calculate correlation
        mean_freq = np.mean(base_freqs)
        mean_stretch = np.mean(optimal_stretches)
        
        numerator = sum((f - mean_freq) * (s - mean_stretch) for f, s in zip(base_freqs, optimal_stretches))
        denom_freq = sum((f - mean_freq) ** 2 for f in base_freqs) ** 0.5
        denom_stretch = sum((s - mean_stretch) ** 2 for s in optimal_stretches) ** 0.5
        
        if denom_freq > 0 and denom_stretch > 0:
            correlation = numerator / (denom_freq * denom_stretch)
        else:
            correlation = 0
        
        results["correlation"] = correlation
        print(f"Correlation (freq vs optimal_stretch): {correlation:.3f}")
        
        if abs(correlation) > 0.5:
            print("[PASS] Strong correlation detected!")
            print("       -> TSAE is FREQUENCY-DEPENDENT (supports internal time base)")
            results["conclusion"] = "frequency_dependent"
        else:
            print("[INFO] Weak correlation")
            print("       -> TSAE may be more complex or task-specific")
            results["conclusion"] = "weak_or_complex"
    else:
        print("[INFO] No variation in optimal stretch detected")
        results["conclusion"] = "no_variation"
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

