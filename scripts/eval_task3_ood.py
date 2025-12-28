#!/usr/bin/env python3
"""Task3 多轴 OOD 评测脚本

验收目标：
1. ood_digits: 模型在更大数字上的泛化
2. ood_compose: 模型在多步组合上的泛化
3. ood_length: 模型在更长表达式上的泛化

用法:
    python scripts/eval_task3_ood.py --manifest manifests/task3_multistep.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.scorer import decode_wave_to_symbols, exact_match
from jericho.task3 import (
    target_symbols_for_task3,
    count_mod_steps,
    synthesise_task3_target_wave,
)


@dataclass
class OODResult:
    """单个 split 的评测结果"""
    split: str
    total: int
    correct: int
    em: float
    avg_steps: float  # 平均 mod 步数
    avg_length: float  # 平均表达式长度


def evaluate_split(
    entries: list[ManifestEntry],
    method: str = "oracle",
) -> OODResult:
    """评估单个 split"""
    correct = 0
    total = 0
    total_steps = 0
    total_length = 0
    
    for entry in entries:
        try:
            # Oracle 预测
            oracle_target = target_symbols_for_task3(entry.symbols)
            
            # 生成目标波形并解码（模拟完整流程）
            target_wave = synthesise_task3_target_wave(entry.symbols)
            decoded = decode_wave_to_symbols(target_wave)
            
            if decoded == oracle_target:
                correct += 1
            
            total_steps += count_mod_steps(entry.symbols)
            total_length += len(entry.symbols)
            total += 1
        except Exception as e:
            print(f"  Error on {entry.example_id}: {e}")
            total += 1
    
    return OODResult(
        split=entries[0].split if entries else "unknown",
        total=total,
        correct=correct,
        em=correct / total if total > 0 else 0.0,
        avg_steps=total_steps / total if total > 0 else 0.0,
        avg_length=total_length / total if total > 0 else 0.0,
    )


def run_ood_evaluation(manifest_path: Path) -> list[OODResult]:
    """运行多轴 OOD 评测"""
    entries = list(read_manifest(manifest_path))
    
    # 按 split 分组
    splits_data: dict[str, list[ManifestEntry]] = {}
    for entry in entries:
        splits_data.setdefault(entry.split, []).append(entry)
    
    results = []
    
    # 定义 OOD 评测顺序
    ood_splits = ["iid_test", "ood_digits", "ood_compose", "ood_length"]
    
    for split in ood_splits:
        if split not in splits_data:
            continue
        
        print(f"Evaluating {split}...")
        result = evaluate_split(splits_data[split])
        results.append(result)
        
        print(f"  EM={result.em:.4f} ({result.correct}/{result.total})")
        print(f"  avg_steps={result.avg_steps:.2f}, avg_length={result.avg_length:.1f}")
    
    return results


def generate_report(results: list[OODResult], output_path: Path) -> None:
    """生成 OOD 评测报告"""
    lines = [
        "# Task3 多轴 OOD 评测报告",
        "",
        f"> 生成时间: {datetime.now().isoformat()}",
        "",
        "## 验收目标",
        "",
        "| 目标 | 指标 | 状态 |",
        "|------|------|------|",
    ]
    
    # 检查验收目标
    iid_result = next((r for r in results if r.split == "iid_test"), None)
    ood_digits_result = next((r for r in results if r.split == "ood_digits"), None)
    ood_compose_result = next((r for r in results if r.split == "ood_compose"), None)
    
    # 目标 1: Oracle 在所有 split 上 EM=1.0
    all_oracle_pass = all(r.em == 1.0 for r in results)
    lines.append(
        f"| Oracle EM=1.0 | {'全部通过' if all_oracle_pass else '存在失败'} | {'✅' if all_oracle_pass else '❌'} |"
    )
    
    # 目标 2: 至少两轴 OOD
    ood_axes = sum(1 for r in results if r.split.startswith("ood_"))
    lines.append(
        f"| 至少两轴 OOD | {ood_axes} 轴 | {'✅' if ood_axes >= 2 else '❌'} |"
    )
    
    # 目标 3: ood_compose 有多步表达式
    compose_has_multistep = ood_compose_result and ood_compose_result.avg_steps > 1.5
    avg_steps_str = f"{ood_compose_result.avg_steps:.2f}" if ood_compose_result else "0.00"
    compose_status = "PASS" if compose_has_multistep else "FAIL"
    lines.append(
        f"| 多步组合 | avg_steps={avg_steps_str} | {compose_status} |"
    )
    
    lines.extend([
        "",
        "## 结果汇总",
        "",
        "| Split | EM | Correct/Total | Avg Steps | Avg Length |",
        "|-------|-----|---------------|-----------|------------|",
    ])
    
    for r in results:
        status = "✅" if r.em == 1.0 else "⚠️" if r.em >= 0.8 else "❌"
        lines.append(
            f"| {r.split} | {r.em:.4f} {status} | {r.correct}/{r.total} | {r.avg_steps:.2f} | {r.avg_length:.1f} |"
        )
    
    lines.extend([
        "",
        "## OOD 轴说明",
        "",
        "| 轴 | 描述 |",
        "|-----|------|",
        "| ood_digits | 更大的数字范围（3-4位）|",
        "| ood_compose | 多步组合 (A%B%C) |",
        "| ood_length | 更长的表达式 |",
        "",
        "## 结论",
        "",
    ])
    
    if all_oracle_pass and ood_axes >= 2 and compose_has_multistep:
        lines.append("✅ **所有验收目标通过**")
    else:
        lines.append("⚠️ **存在未通过的验收目标**")
        if not all_oracle_pass:
            lines.append("- Oracle 未在所有 split 上达到 EM=1.0")
        if ood_axes < 2:
            lines.append("- OOD 轴数量不足")
        if not compose_has_multistep:
            lines.append("- 多步组合验证不充分")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已生成: {output_path}")
    
    # 同时生成 CSV
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "em", "correct", "total", "avg_steps", "avg_length"])
        for r in results:
            writer.writerow([r.split, f"{r.em:.4f}", r.correct, r.total, f"{r.avg_steps:.2f}", f"{r.avg_length:.1f}"])
    print(f"CSV 已生成: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Task3 多轴 OOD 评测")
    parser.add_argument("--manifest", type=Path, default=Path("manifests/task3_multistep.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/task3_ood.md"))
    
    args = parser.parse_args()
    
    if not args.manifest.exists():
        print(f"Error: Manifest not found: {args.manifest}")
        sys.exit(1)
    
    print(f"Running Task3 OOD evaluation on: {args.manifest}")
    print("=" * 60)
    
    results = run_ood_evaluation(args.manifest)
    generate_report(results, args.output)
    
    print("=" * 60)
    all_pass = all(r.em == 1.0 for r in results)
    if all_pass:
        print("All OOD evaluations PASSED")
    else:
        print("Some OOD evaluations FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

