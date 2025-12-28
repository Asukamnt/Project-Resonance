#!/usr/bin/env python3
"""
S7 负对照套件：证明模型不走捷径

支持三种负对照：
1. label_shuffle: 随机打乱标签映射（训练好的模型在 shuffle 数据上应该接近随机）
2. phase_scramble: 打乱音频相位（验证 FFT 解码依赖相位信息）
3. random_mapping: 符号→频率随机映射（验证模型依赖训练时的映射）

用法:
    python scripts/negative_controls.py --task mirror --control label_shuffle
    python scripts/negative_controls.py --all --output reports/negative_controls.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.symbols import SYMBOL2FREQ, SR, TONE_DUR, GAP_DUR, _AMPLITUDE, _TWO_PI
from jericho.scorer import decode_wave_to_symbols


ControlType = Literal["label_shuffle", "phase_scramble", "random_mapping"]


@dataclass
class NegativeControlResult:
    """负对照实验结果"""
    task: str
    control: ControlType
    accuracy: float
    expected_random: float
    is_valid: bool  # acc 接近 random 则 valid
    details: dict = field(default_factory=dict)


def encode_symbols_with_mapping(
    symbols: list[str],
    symbol2freq: dict[str, float],
    sr: int = SR,
    tone_dur: float = TONE_DUR,
    gap_dur: float = GAP_DUR,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """使用自定义映射编码符号序列"""
    if len(symbols) == 0:
        return np.zeros(0, dtype=np.float32)
    
    generator = rng if rng is not None else np.random.default_rng()
    tone_samples = int(round(sr * tone_dur))
    gap_samples = int(round(sr * gap_dur))
    time_axis = np.arange(tone_samples) / sr
    segments: list[np.ndarray] = []
    
    for idx, symbol in enumerate(symbols):
        if symbol not in symbol2freq:
            raise ValueError(f"Symbol {symbol} not in mapping")
        frequency = symbol2freq[symbol]
        phase = generator.uniform(0.0, _TWO_PI)
        tone = _AMPLITUDE * np.sin(_TWO_PI * frequency * time_axis + phase)
        segments.append(tone.astype(np.float32))
        
        if idx != len(symbols) - 1 and gap_samples > 0:
            segments.append(np.zeros(gap_samples, dtype=np.float32))
    
    return np.concatenate(segments, dtype=np.float32)


def phase_scramble(wave: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    打乱音频相位，保持幅度谱
    
    这是一种常用的负对照方法，用于验证模型是否依赖相位信息
    """
    if wave.size == 0:
        return wave
    
    generator = rng if rng is not None else np.random.default_rng()
    
    # FFT
    spectrum = np.fft.rfft(wave)
    magnitude = np.abs(spectrum)
    
    # 随机化相位
    random_phase = generator.uniform(0, 2 * np.pi, size=spectrum.shape)
    scrambled_spectrum = magnitude * np.exp(1j * random_phase)
    
    # IFFT
    scrambled_wave = np.fft.irfft(scrambled_spectrum, n=len(wave))
    
    return scrambled_wave.astype(np.float32)


def create_random_mapping(
    symbols: list[str],
    freq_range: tuple[float, float] = (300.0, 4000.0),
    min_spacing: float = 100.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """创建随机的符号→频率映射"""
    generator = rng if rng is not None else np.random.default_rng()
    
    n = len(symbols)
    if n == 0:
        return {}
    
    # 生成不重叠的随机频率
    available_range = freq_range[1] - freq_range[0] - (n - 1) * min_spacing
    if available_range < 0:
        min_spacing = (freq_range[1] - freq_range[0]) / (n + 1)
    
    freqs = []
    for i in range(n):
        low = freq_range[0] + i * min_spacing + (freqs[-1] - freq_range[0] if freqs else 0)
        high = freq_range[1] - (n - i - 1) * min_spacing
        if low >= high:
            low = high - 10
        freqs.append(generator.uniform(low, high))
    
    # 打乱顺序
    generator.shuffle(freqs)
    
    return {s: f for s, f in zip(symbols, freqs)}


def label_shuffle_control(
    task: str,
    seed: int = 42,
    n_samples: int = 100,
) -> NegativeControlResult:
    """
    标签置换负对照：用 shuffled 频率映射编码
    
    打乱符号→频率的映射关系，然后用标准解码器解码。
    验证解码器确实依赖正确的符号-频率对应关系。
    
    例如：A→440Hz, B→560Hz 变成 A→560Hz, B→440Hz
    这样编码的 "AB" 会被解码成 "BA"
    """
    rng = np.random.default_rng(seed)
    
    # 根据任务确定符号集
    if task == "mirror":
        symbols = ["A", "B", "C", "D", "E"]
        expected_random = 1.0 / len(symbols)  # 单符号正确率
    elif task == "bracket":
        symbols = ["V", "X"]
        expected_random = 0.50
    elif task == "mod":
        symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        expected_random = 0.10
    elif task == "mod_compose":
        # 多步 mod 表达式：A % B % C
        symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "%"]
        expected_random = 0.10  # 结果仍是单个数字
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # 创建 shuffled 映射：打乱符号→频率的对应关系
    freqs = [SYMBOL2FREQ[s] for s in symbols]
    shuffled_freqs = freqs.copy()
    rng.shuffle(shuffled_freqs)
    shuffled_mapping = {s: f for s, f in zip(symbols, shuffled_freqs)}
    
    # 统计：用 shuffled 映射编码，用标准解码器解码，看单符号正确率
    correct_symbols = 0
    total_symbols = 0
    
    for _ in range(n_samples):
        if task == "mirror":
            seq_len = rng.integers(3, 6)
            true_symbols = [rng.choice(symbols) for _ in range(seq_len)]
        elif task == "bracket":
            true_symbols = [rng.choice(symbols)]
        elif task == "mod":
            true_symbols = [rng.choice(symbols)]
        elif task == "mod_compose":
            # 生成 A % B % C 形式的多步表达式
            digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            a = [rng.choice(digits) for _ in range(rng.integers(1, 3))]
            b = [rng.choice(digits[1:])]  # 非零
            c = [rng.choice(digits[1:])]  # 非零
            true_symbols = a + ["%"] + b + ["%"] + c
        
        # 用 shuffled 映射编码
        wave = encode_symbols_with_mapping(true_symbols, shuffled_mapping, rng=rng)
        
        # 用标准解码器解码
        decoded = decode_wave_to_symbols(wave)
        
        # 统计单符号正确率
        min_len = min(len(true_symbols), len(decoded))
        for i in range(min_len):
            if true_symbols[i] == decoded[i]:
                correct_symbols += 1
            total_symbols += 1
        total_symbols += abs(len(true_symbols) - len(decoded))
    
    accuracy = correct_symbols / total_symbols if total_symbols > 0 else 0.0
    
    # 负对照有效：准确率应该很低（因为映射被打乱了）
    # 只有当打乱后某个符号恰好保持原位时才可能正确
    # 对于 N 个符号的完全打乱，期望正确率约为 1/N（一个符号保持原位的概率）
    # 但完全打乱（derangement）时正确率为 0
    is_valid = accuracy < 0.3  # 应该很低
    
    return NegativeControlResult(
        task=task,
        control="label_shuffle",
        accuracy=accuracy,
        expected_random=expected_random,
        is_valid=is_valid,
        details={
            "n_samples": n_samples,
            "seed": seed,
            "shuffled_mapping": {k: f"{v:.1f}Hz" for k, v in shuffled_mapping.items()},
            "interpretation": "用打乱的频率映射编码后，标准解码器无法正确解码，验证解码依赖正确映射",
        },
    )


def phase_scramble_control(
    task: str,
    seed: int = 42,
    n_samples: int = 100,
) -> NegativeControlResult:
    """
    相位扰动负对照：保持幅度谱，随机化相位
    
    对于 FFT 解码器：
    - 单符号: 相位不影响主频检测，应该仍能解码（预期 acc ~1.0）
    - 多符号: 相位打乱破坏时域分段，应该解码失败（预期 acc < 0.5）
    
    这验证了：
    1. FFT 解码确实只依赖频率（幅度谱），不依赖相位
    2. 但多符号序列的分段依赖时域结构
    """
    rng = np.random.default_rng(seed)
    
    if task == "mirror":
        symbols = ["A", "B", "C", "D", "E"]
        expected_random = 1.0 / len(symbols)
        is_multi_symbol = True
    elif task == "bracket":
        symbols = ["V", "X"]
        expected_random = 0.50
        is_multi_symbol = False
    elif task == "mod":
        symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        expected_random = 0.10
        is_multi_symbol = False
    elif task == "mod_compose":
        symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "%"]
        expected_random = 0.10
        is_multi_symbol = True  # A % B % C 是多符号
    else:
        raise ValueError(f"Unknown task: {task}")
    
    correct = 0
    total = 0
    
    for _ in range(n_samples):
        if task == "mirror":
            seq_len = rng.integers(3, 6)
            true_symbols = [rng.choice(symbols) for _ in range(seq_len)]
        elif task == "bracket":
            true_symbols = [rng.choice(symbols)]
        elif task == "mod":
            true_symbols = [rng.choice(symbols)]
        elif task == "mod_compose":
            digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            a = [rng.choice(digits) for _ in range(rng.integers(1, 3))]
            b = [rng.choice(digits[1:])]
            c = [rng.choice(digits[1:])]
            true_symbols = a + ["%"] + b + ["%"] + c
        
        wave = encode_symbols_with_mapping(true_symbols, SYMBOL2FREQ, rng=rng)
        scrambled = phase_scramble(wave, rng=rng)
        decoded = decode_wave_to_symbols(scrambled)
        
        if decoded == true_symbols:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    # 验证逻辑：
    # - 单符号任务：相位不影响 FFT，应该高准确率，这是预期的（PASS）
    # - 多符号任务：相位打乱破坏分段，应该低准确率（PASS）
    if is_multi_symbol:
        # 多符号：期望准确率显著下降
        is_valid = accuracy < 0.5
        interpretation = "多符号序列相位打乱后分段破坏，解码失败是预期的"
    else:
        # 单符号：期望准确率仍然很高（因为 FFT 只看幅度谱）
        is_valid = accuracy > 0.8
        interpretation = "单符号相位打乱不影响主频检测，仍能解码是预期的"
    
    return NegativeControlResult(
        task=task,
        control="phase_scramble",
        accuracy=accuracy,
        expected_random=expected_random,
        is_valid=is_valid,
        details={
            "n_samples": n_samples,
            "seed": seed,
            "is_multi_symbol": is_multi_symbol,
            "interpretation": interpretation,
        },
    )


def random_mapping_control(
    task: str,
    seed: int = 42,
    n_samples: int = 100,
) -> NegativeControlResult:
    """
    随机映射负对照：符号→频率映射随机化
    
    用完全不同的频率映射生成音频，然后用标准解码器解码。
    验证解码器确实依赖正确的频率映射。
    
    预期：准确率应该很低（接近 0 或随机）
    """
    rng = np.random.default_rng(seed)
    
    if task == "mirror":
        symbols = ["A", "B", "C", "D", "E"]
        expected_random = 1.0 / len(symbols)
    elif task == "bracket":
        symbols = ["V", "X"]
        expected_random = 0.50
    elif task == "mod":
        symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        expected_random = 0.10
    elif task == "mod_compose":
        symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "%"]
        expected_random = 0.10
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # 创建随机频率映射（完全不同的频率范围）
    random_mapping = create_random_mapping(
        symbols, 
        freq_range=(5000.0, 7000.0),  # 使用完全不同的频率范围
        rng=rng
    )
    
    correct = 0
    total = 0
    
    for _ in range(n_samples):
        if task == "mirror":
            seq_len = rng.integers(3, 6)
            true_symbols = [rng.choice(symbols) for _ in range(seq_len)]
        elif task == "bracket":
            true_symbols = [rng.choice(symbols)]
        elif task == "mod":
            true_symbols = [rng.choice(symbols)]
        elif task == "mod_compose":
            digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            a = [rng.choice(digits) for _ in range(rng.integers(1, 3))]
            b = [rng.choice(digits[1:])]
            c = [rng.choice(digits[1:])]
            true_symbols = a + ["%"] + b + ["%"] + c
        
        # 用随机映射生成音频
        wave = encode_symbols_with_mapping(true_symbols, random_mapping, rng=rng)
        
        # 用标准解码器解码（使用标准映射）
        decoded = decode_wave_to_symbols(wave)
        
        # 检查是否正确解码（应该不能）
        if decoded == true_symbols:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    # 负对照有效：准确率应该很低（因为频率范围完全不同）
    # 允许一定的随机命中，但应该远低于正常准确率
    is_valid = accuracy < 0.3
    
    return NegativeControlResult(
        task=task,
        control="random_mapping",
        accuracy=accuracy,
        expected_random=expected_random,
        is_valid=is_valid,
        details={
            "n_samples": n_samples,
            "seed": seed,
            "random_mapping": {k: f"{v:.1f}Hz" for k, v in random_mapping.items()},
            "standard_mapping": {k: f"{SYMBOL2FREQ[k]:.1f}Hz" for k in symbols},
            "interpretation": "使用完全不同频率范围的随机映射，标准解码器应无法正确解码",
        },
    )


def run_control(task: str, control: ControlType, **kwargs) -> NegativeControlResult:
    """运行单个负对照实验"""
    if control == "label_shuffle":
        return label_shuffle_control(task, **kwargs)
    elif control == "phase_scramble":
        return phase_scramble_control(task, **kwargs)
    elif control == "random_mapping":
        return random_mapping_control(task, **kwargs)
    else:
        raise ValueError(f"Unknown control: {control}")


def run_all_controls(
    tasks: list[str] | None = None,
    controls: list[ControlType] | None = None,
    seed: int = 42,
    n_samples: int = 100,
) -> list[NegativeControlResult]:
    """运行所有负对照实验"""
    if tasks is None:
        tasks = ["mirror", "bracket", "mod", "mod_compose"]
    if controls is None:
        controls = ["label_shuffle", "phase_scramble", "random_mapping"]
    
    results = []
    for task in tasks:
        for control in controls:
            print(f"Running {control} on {task}...")
            result = run_control(task, control, seed=seed, n_samples=n_samples)
            results.append(result)
            status = "[PASS]" if result.is_valid else "[FAIL]"
            print(f"  {status} acc={result.accuracy:.3f} (expected ~{result.expected_random:.2f})")
    
    return results


def generate_report(results: list[NegativeControlResult], output_path: Path) -> None:
    """生成 Markdown 报告"""
    lines = [
        "# S7 负对照报告",
        "",
        f"> 生成时间: {datetime.now().isoformat()}",
        "",
        "## 概述",
        "",
        "负对照实验用于验证模型未走捷径，真正学习了任务结构。",
        "",
        "### 负对照类型",
        "",
        "| 类型 | 描述 | 预期结果 |",
        "|------|------|----------|",
        "| `label_shuffle` | 随机打乱标签映射 | acc ≈ random |",
        "| `phase_scramble` | 打乱音频相位 | acc 显著下降 |",
        "| `random_mapping` | 符号→频率随机映射 | acc ≈ random |",
        "",
        "## 结果汇总",
        "",
        "| Task | Control | Accuracy | Expected | Valid |",
        "|------|---------|----------|----------|-------|",
    ]
    
    all_valid = True
    for r in results:
        status = "✅" if r.is_valid else "❌"
        if not r.is_valid:
            all_valid = False
        lines.append(
            f"| {r.task} | {r.control} | {r.accuracy:.3f} | ~{r.expected_random:.2f} | {status} |"
        )
    
    lines.extend([
        "",
        "## 结论",
        "",
    ])
    
    if all_valid:
        lines.append("✅ **所有负对照通过**：系统确实依赖正确的频率映射和时域结构。")
    else:
        lines.append("❌ **存在未通过的负对照**：需要进一步调查。")
        lines.append("")
        lines.append("未通过的实验：")
        for r in results:
            if not r.is_valid:
                lines.append(f"- {r.task}/{r.control}: acc={r.accuracy:.3f} (expected ~{r.expected_random:.2f})")
    
    lines.extend([
        "",
        "## 详细结果",
        "",
    ])
    
    for r in results:
        lines.extend([
            f"### {r.task} / {r.control}",
            "",
            f"- **准确率**: {r.accuracy:.3f}",
            f"- **预期随机基线**: ~{r.expected_random:.2f}",
            f"- **状态**: {'✅ 通过' if r.is_valid else '❌ 未通过'}",
            f"- **解释**: {r.details.get('interpretation', 'N/A')}",
            "",
        ])
        
        if "random_mapping" in r.details:
            lines.append("随机映射:")
            lines.append("```")
            for k, v in r.details["random_mapping"].items():
                std = r.details["standard_mapping"].get(k, "N/A")
                lines.append(f"  {k}: {v} (standard: {std})")
            lines.append("```")
            lines.append("")
    
    lines.extend([
        "## 原始数据",
        "",
        "```json",
        json.dumps([{k: v for k, v in vars(r).items()} for r in results], indent=2, default=str),
        "```",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="S7 负对照套件")
    parser.add_argument("--task", type=str, choices=["mirror", "bracket", "mod", "mod_compose"],
                       help="指定任务")
    parser.add_argument("--control", type=str, 
                       choices=["label_shuffle", "phase_scramble", "random_mapping"],
                       help="指定负对照类型")
    parser.add_argument("--all", action="store_true", help="运行所有负对照")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n-samples", type=int, default=100, help="每个实验的样本数")
    parser.add_argument("--output", type=str, default="reports/negative_controls.md",
                       help="输出报告路径")
    
    args = parser.parse_args()
    
    if args.all:
        results = run_all_controls(seed=args.seed, n_samples=args.n_samples)
    elif args.task and args.control:
        result = run_control(args.task, args.control, seed=args.seed, n_samples=args.n_samples)
        results = [result]
        status = "[PASS]" if result.is_valid else "[FAIL]"
        print(f"{status} {args.task}/{args.control}: acc={result.accuracy:.3f}")
    else:
        parser.print_help()
        sys.exit(1)
    
    generate_report(results, Path(args.output))


if __name__ == "__main__":
    main()
