#!/usr/bin/env python3
"""
Architecture Ablation Study

对比不同架构配置对 Task3 Mod 性能的影响：
- Mini-JMamba (SSM + Attention hybrid) - 默认
- Pure SSM (no attention)
- Pure Attention (no SSM)

用法:
    python scripts/ablation_architecture.py --device cuda --epochs 30
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from jericho.data import read_manifest
from jericho.pipelines.task3_mod_audio import (
    Task3TrainingConfig,
    mini_jmamba_task3_pipeline,
)


def run_ablation(
    configs: dict[str, dict],
    manifest_path: Path,
    epochs: int,
    device: str,
    seed: int,
    limit: int | None,
) -> list[dict]:
    """运行架构 ablation"""
    
    results = []
    
    for name, arch_config in configs.items():
        print(f"\n{'='*60}")
        print(f"Testing architecture: {name}")
        print(f"Config: {arch_config}")
        print(f"{'='*60}")
        
        # 读取数据
        entries = list(read_manifest(manifest_path))
        # 支持不同的 split 命名
        train_entries = [e for e in entries if e.split in ("train", "iid_train")]
        eval_entries = [e for e in entries if e.split in ("iid_test", "val")]
        
        if limit:
            train_entries = train_entries[:limit]
            eval_entries = eval_entries[:min(limit // 2, len(eval_entries))]
        
        if not train_entries or not eval_entries:
            print(f"  Skipping: no data")
            continue
        
        # 配置
        config = Task3TrainingConfig(
            num_ssm_layers=arch_config.get("num_ssm_layers", 10),
            num_attn_layers=arch_config.get("num_attn_layers", 2),
            pretrain_mirror_epochs=5,
        )
        
        # 训练
        try:
            predictions, metrics, model_info = mini_jmamba_task3_pipeline(
                train_entries=train_entries,
                eval_entries=eval_entries,
                seed=seed,
                epochs=epochs,
                batch_size=16,
                lr=1e-3,
                device=torch.device(device),
                config=config,
            )
            
            # 提取指标
            final_em = metrics.get("eval_em", metrics.get("em", 0))
            best_em = final_em
            
            result = {
                "architecture": name,
                "num_ssm_layers": arch_config.get("num_ssm_layers", 10),
                "num_attn_layers": arch_config.get("num_attn_layers", 2),
                "final_em": final_em,
                "best_em": best_em,
                "final_loss": metrics.get("eval_loss", metrics.get("loss", float("inf"))),
                "epochs": epochs,
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "architecture": name,
                "error": str(e),
            }
        
        results.append(result)
        print(f"  Result: {result}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Architecture Ablation")
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
        default=Path("reports/ablation_architecture.json"),
        help="Output JSON path",
    )
    
    args = parser.parse_args()
    
    # 检查 manifest
    if not args.manifest.exists():
        alt = Path("manifests/task3_multistep.jsonl")
        if alt.exists():
            args.manifest = alt
        else:
            print(f"Error: manifest not found: {args.manifest}")
            sys.exit(1)
    
    # 定义架构配置
    configs = {
        "Mini-JMamba (10 SSM + 2 Attn)": {"num_ssm_layers": 10, "num_attn_layers": 2},
        "Pure SSM (12 SSM + 0 Attn)": {"num_ssm_layers": 12, "num_attn_layers": 0},
        "More Attention (8 SSM + 4 Attn)": {"num_ssm_layers": 8, "num_attn_layers": 4},
        "Balanced (6 SSM + 6 Attn)": {"num_ssm_layers": 6, "num_attn_layers": 6},
    }
    
    print(f"Manifest: {args.manifest}")
    print(f"Architectures to test: {list(configs.keys())}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    
    # 运行 ablation
    results = run_ablation(
        configs=configs,
        manifest_path=args.manifest,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        limit=args.limit,
    )
    
    # 保存结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "experiment": "architecture_ablation",
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
    print(f"{'Architecture':<35} {'Final EM':<12} {'Best EM':<12}")
    print("-" * 59)
    for r in results:
        if "error" in r:
            print(f"{r['architecture']:<35} ERROR")
        else:
            print(f"{r['architecture']:<35} {r['final_em']:<12.4f} {r['best_em']:<12.4f}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

