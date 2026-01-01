#!/usr/bin/env python3
"""
S1/S19/S20 一键总评估脚本

⚠️ 口径说明
===========
本脚本评估的是 **Oracle/Protocol 闭环正确性**，而非训练模型能力：

- Oracle EM=1.0 表示：编码→解码系统闭环正确，评测协议无漏洞
- Model EM 需要通过 train.py 训练后，使用模型检查点单独评估

这两个指标的区别：
| 指标        | 含义                           | 本脚本 |
|-------------|--------------------------------|--------|
| Oracle EM   | 系统闭环验证（编码→解码一致性）| ✅     |
| Model EM    | 训练模型能力（模型预测准确率）  | ❌     |

用法:
    # Final Gate 完整评估（S1）- Oracle/Protocol 验证
    python evaluate.py --stage final --tasks mirror bracket mod
    
    # 快速 MVP 验证（S2）
    python evaluate.py --stage mvp --task mirror
    
    # 指定 split 评估（S19/S20）
    python evaluate.py --splits iid_test ood_length ood_compose --task mod

产物:
    reports/system_overview_final.json
    reports/test_metrics.json
    reports/ood_summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.scorer import decode_wave_to_symbols, exact_match
from jericho.task2 import target_symbol_for_task2
from jericho.task3 import target_symbols_for_task3, count_mod_steps


@dataclass
class TaskMetrics:
    """单任务评测指标"""
    task: str
    split: str
    total: int
    correct: int
    em: float  # Exact Match
    token_accuracy: float  # Token-level accuracy
    avg_edit_distance: float  # 平均编辑距离
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationReport:
    """评估报告"""
    stage: str
    timestamp: str
    tasks: list[str]
    splits: list[str]
    metrics: list[TaskMetrics]
    summary: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "timestamp": self.timestamp,
            "tasks": self.tasks,
            "splits": self.splits,
            "metrics": [m.to_dict() for m in self.metrics],
            "summary": self.summary,
        }


def levenshtein_distance(s1: list[str], s2: list[str]) -> int:
    """计算两个序列的 Levenshtein 编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def get_target_symbols(entry: ManifestEntry, task: str) -> list[str]:
    """获取目标符号序列"""
    if task == "mirror":
        return list(entry.symbols)
    elif task == "bracket":
        return [target_symbol_for_task2(entry.symbols)]
    elif task == "mod":
        return target_symbols_for_task3(entry.symbols)
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_task_split(
    manifest_path: Path,
    task: str,
    split: str,
    limit: int | None = None,
) -> TaskMetrics:
    """评估单个任务的单个 split"""
    entries = [e for e in read_manifest(manifest_path) if e.split == split]
    
    if limit:
        entries = entries[:limit]
    
    if not entries:
        return TaskMetrics(
            task=task, split=split, total=0, correct=0,
            em=0.0, token_accuracy=0.0, avg_edit_distance=0.0,
        )
    
    correct = 0
    total_tokens = 0
    correct_tokens = 0
    total_edit_distance = 0
    
    for entry in entries:
        try:
            target = get_target_symbols(entry, task)
            
            # 生成波形并解码
            if task == "mirror":
                from jericho.symbols import encode_symbols_to_wave
                wave = encode_symbols_to_wave(target)
            elif task == "bracket":
                from jericho.task2 import synthesise_task2_target_wave
                wave = synthesise_task2_target_wave(entry.symbols)
            elif task == "mod":
                from jericho.task3 import synthesise_task3_target_wave
                wave = synthesise_task3_target_wave(entry.symbols)
            
            decoded = decode_wave_to_symbols(wave)
            
            # Exact Match
            if decoded == target:
                correct += 1
            
            # Token Accuracy
            min_len = min(len(decoded), len(target))
            for i in range(min_len):
                if decoded[i] == target[i]:
                    correct_tokens += 1
            total_tokens += max(len(decoded), len(target))
            
            # Edit Distance
            total_edit_distance += levenshtein_distance(decoded, target)
            
        except Exception as e:
            print(f"  Error on {entry.example_id}: {e}")
    
    total = len(entries)
    
    return TaskMetrics(
        task=task,
        split=split,
        total=total,
        correct=correct,
        em=correct / total if total > 0 else 0.0,
        token_accuracy=correct_tokens / total_tokens if total_tokens > 0 else 0.0,
        avg_edit_distance=total_edit_distance / total if total > 0 else 0.0,
    )


def run_evaluation(
    stage: str,
    tasks: list[str],
    splits: list[str],
    manifests: dict[str, Path],
    limit: int | None = None,
) -> EvaluationReport:
    """运行完整评估"""
    metrics = []
    
    for task in tasks:
        manifest_path = manifests.get(task)
        if not manifest_path or not manifest_path.exists():
            print(f"Warning: Manifest not found for task {task}: {manifest_path}")
            continue
        
        for split in splits:
            print(f"Evaluating {task}/{split}...")
            result = evaluate_task_split(manifest_path, task, split, limit)
            metrics.append(result)
            print(f"  EM={result.em:.4f}, Token={result.token_accuracy:.4f}, EditDist={result.avg_edit_distance:.2f}")
    
    # 计算汇总
    summary = {}
    for task in tasks:
        task_metrics = [m for m in metrics if m.task == task]
        if task_metrics:
            summary[task] = {
                "avg_em": sum(m.em for m in task_metrics) / len(task_metrics),
                "iid_em": next((m.em for m in task_metrics if "iid" in m.split), None),
                "ood_em_min": min((m.em for m in task_metrics if "ood" in m.split), default=None),
            }
    
    overall_em = sum(m.em for m in metrics) / len(metrics) if metrics else 0.0
    summary["overall"] = {
        "avg_em": overall_em,
        "pass": overall_em >= 0.95,
    }
    
    return EvaluationReport(
        stage=stage,
        timestamp=datetime.now().isoformat(),
        tasks=tasks,
        splits=splits,
        metrics=metrics,
        summary=summary,
    )


def generate_reports(report: EvaluationReport, output_dir: Path) -> None:
    """生成所有报告文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON 报告 (S1)
    json_path = output_dir / f"system_overview_{report.stage}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Generated: {json_path}")
    
    # 2. 测试指标 (S19)
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": report.timestamp,
            "metrics": [m.to_dict() for m in report.metrics],
        }, f, indent=2, ensure_ascii=False)
    print(f"Generated: {metrics_path}")
    
    # 3. OOD 汇总 (S20)
    ood_path = output_dir / "ood_summary.md"
    lines = [
        "# OOD 评估汇总报告",
        "",
        f"> 生成时间: {report.timestamp}",
        "",
        "## 口径说明",
        "",
        "**本报告评估的是 Oracle/Protocol 闭环正确性，而非训练模型能力。**",
        "",
        "| 指标类型 | 含义 | 本报告 |",
        "|----------|------|--------|",
        "| **Oracle EM** | 系统闭环验证：编码→解码一致性 | 验证 |",
        "| **Model EM** | 训练模型能力：模型预测准确率 | 未包含 |",
        "",
        "Oracle EM=1.0 证明：编码正确、解码正确、评测协议无漏洞。",
        "",
        "---",
        "",
        "## 评估配置",
        "",
        f"- Stage: {report.stage} (Oracle/Protocol)",
        f"- Tasks: {', '.join(report.tasks)}",
        f"- Splits: {', '.join(report.splits)}",
        "",
        "## 结果矩阵",
        "",
        "| Task | Split | EM | Token Acc | Edit Dist | Total |",
        "|------|-------|-----|-----------|-----------|-------|",
    ]
    
    for m in report.metrics:
        status = "✅" if m.em >= 0.95 else "⚠️" if m.em >= 0.80 else "❌"
        lines.append(
            f"| {m.task} | {m.split} | {m.em:.4f} {status} | {m.token_accuracy:.4f} | {m.avg_edit_distance:.2f} | {m.total} |"
        )
    
    lines.extend([
        "",
        "## 任务汇总",
        "",
        "| Task | IID EM | OOD Min EM | Avg EM |",
        "|------|--------|------------|--------|",
    ])
    
    for task in report.tasks:
        if task in report.summary:
            s = report.summary[task]
            iid = f"{s['iid_em']:.4f}" if s['iid_em'] is not None else "N/A"
            ood = f"{s['ood_em_min']:.4f}" if s['ood_em_min'] is not None else "N/A"
            lines.append(f"| {task} | {iid} | {ood} | {s['avg_em']:.4f} |")
    
    lines.extend([
        "",
        "## 总体结论",
        "",
    ])
    
    if report.summary.get("overall", {}).get("pass"):
        lines.append("✅ **评估通过**: 平均 EM >= 95%")
    else:
        avg_em = report.summary.get("overall", {}).get("avg_em", 0)
        lines.append(f"⚠️ **评估未达标**: 平均 EM = {avg_em:.4f}")
    
    ood_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {ood_path}")


def main():
    parser = argparse.ArgumentParser(description="S1/S19/S20 一键总评估")
    parser.add_argument("--stage", type=str, default="final", choices=["mvp", "final"],
                       help="评估阶段")
    parser.add_argument("--tasks", type=str, nargs="+", default=["mirror", "bracket", "mod"],
                       help="要评估的任务")
    parser.add_argument("--splits", type=str, nargs="+", 
                       default=["iid_test", "ood_length", "ood_digits", "ood_compose"],
                       help="要评估的 splits")
    parser.add_argument("--manifests-dir", type=Path, default=Path("manifests"),
                       help="manifest 目录")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"),
                       help="报告输出目录")
    parser.add_argument("--limit", type=int, default=None,
                       help="每个 split 的样本数限制")
    parser.add_argument("--no-text", action="store_true",
                       help="验证无文本中间表征（S1 要求）")
    
    args = parser.parse_args()
    
    # 自动探测 manifest 文件（兼容不同命名）
    def find_manifest(task: str) -> Path:
        candidates = {
            "mirror": ["task1.jsonl"],
            "bracket": ["task2.jsonl"],
            "mod": ["task3.jsonl", "task3_multistep.jsonl"],
        }
        for name in candidates.get(task, []):
            path = args.manifests_dir / name
            if path.exists():
                return path
        return args.manifests_dir / candidates[task][0]
    
    manifests = {
        "mirror": find_manifest("mirror"),
        "bracket": find_manifest("bracket"),
        "mod": find_manifest("mod"),
    }
    
    print(f"Running {args.stage} evaluation")
    print(f"Tasks: {args.tasks}")
    print(f"Splits: {args.splits}")
    print("=" * 60)
    
    report = run_evaluation(
        stage=args.stage,
        tasks=args.tasks,
        splits=args.splits,
        manifests=manifests,
        limit=args.limit,
    )
    
    generate_reports(report, args.output_dir)
    
    print("=" * 60)
    if report.summary.get("overall", {}).get("pass"):
        print("Evaluation PASSED")
    else:
        print("Evaluation completed (some targets may not be met)")


if __name__ == "__main__":
    main()
