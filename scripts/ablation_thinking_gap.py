#!/usr/bin/env python3
"""
Thinking Gap Ablation Study

测试不同 thinking_gap_s 对 Task3 Mod 性能的影响。
生成：reports/ablation_thinking_gap.json + 控制台总结

用法:
    python scripts/ablation_thinking_gap.py --device cuda --epochs 30
    python scripts/ablation_thinking_gap.py --device cpu --epochs 10 --limit 50  # 快速测试
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 确保能 import jericho
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from jericho.data import read_manifest
from jericho.pipelines.task3_mod_audio import (
    Task3TrainingConfig,
    mini_jmamba_task3_pipeline,
)


def run_ablation(
    thinking_gaps: list[float],
    manifest_path: Path,
    epochs: int,
    device: str,
    seed: int,
    limit: int | None,
) -> list[dict]:
    """运行 thinking gap ablation"""
    
    results = []
    
    for gap in thinking_gaps:
        print(f"\n{'='*60}")
        print(f"Testing thinking_gap_s = {gap}")
        print(f"{'='*60}")
        
        # 读取数据
        entries = list(read_manifest(manifest_path))
        train_entries = [e for e in entries if e.split == "iid_train"]
        eval_entries = [e for e in entries if e.split == "iid_test"]
        
        if limit:
            train_entries = train_entries[:limit]
            eval_entries = eval_entries[:min(limit // 2, len(eval_entries))]
        
        if not train_entries or not eval_entries:
            print(f"  Skipping: no data")
            continue
        
        # 配置
        config = Task3TrainingConfig(
            thinking_gap_s=gap,
            thinking_gap_align=160,
            pretrain_mirror_epochs=5,  # 快速预训练
        )
        
        # 训练
        try:
            metrics_history, final_state = mini_jmamba_task3_pipeline(
                train_entries=train_entries,
                eval_entries=eval_entries,
                seed=seed,
                epochs=epochs,
                batch_size=16,
                lr=1e-3,
                device=torch.device(device),
                config=config,
            )
            
            # 提取最终指标
            if metrics_history:
                final_metrics = metrics_history[-1]
                best_em = max(m.get("eval_mod_em", 0) for m in metrics_history)
            else:
                final_metrics = {}
                best_em = 0.0
            
            result = {
                "thinking_gap_s": gap,
                "final_em": final_metrics.get("eval_mod_em", 0),
                "best_em": best_em,
                "final_loss": final_metrics.get("eval_loss", float("inf")),
                "epochs": epochs,
                "train_samples": len(train_entries),
                "eval_samples": len(eval_entries),
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            result = {
                "thinking_gap_s": gap,
                "error": str(e),
            }
        
        results.append(result)
        print(f"  Result: {result}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Thinking Gap Ablation")
    parser.add_argument(
        "--gaps",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        help="Thinking gap values to test (seconds)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/task3.jsonl"),
        help="Manifest file path",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per config")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (for quick test)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ablation_thinking_gap.json"),
        help="Output JSON path",
    )
    
    args = parser.parse_args()
    
    # 检查 manifest
    if not args.manifest.exists():
        # 尝试备选
        alt = Path("manifests/task3_multistep.jsonl")
        if alt.exists():
            args.manifest = alt
        else:
            print(f"Error: manifest not found: {args.manifest}")
            sys.exit(1)
    
    print(f"Manifest: {args.manifest}")
    print(f"Thinking gaps to test: {args.gaps}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    
    # 运行 ablation
    results = run_ablation(
        thinking_gaps=args.gaps,
        manifest_path=args.manifest,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        limit=args.limit,
    )
    
    # 保存结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "experiment": "thinking_gap_ablation",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "manifest": str(args.manifest),
            "epochs": args.epochs,
            "device": args.device,
            "seed": args.seed,
            "limit": args.limit,
        },
        "results": results,
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Gap (s)':<10} {'Final EM':<12} {'Best EM':<12}")
    print("-" * 34)
    for r in results:
        if "error" in r:
            print(f"{r['thinking_gap_s']:<10} ERROR: {r['error'][:30]}")
        else:
            print(f"{r['thinking_gap_s']:<10} {r['final_em']:<12.4f} {r['best_em']:<12.4f}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

