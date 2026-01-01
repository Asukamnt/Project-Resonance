#!/usr/bin/env python3
"""
选择性突触修剪参数扫描实验 (Selective Synaptic Pruning Sweep)

实验目的：
1. 验证 keep_ratio 的最优值
2. 生成 10-seed 统计数据
3. 计算 95% CI 和 p-value

Author: Jericho Team
Date: 2026-01-02
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig, SSMLikeBlock
from jericho.symbols import SR, SYMBOL2FREQ, encode_symbols_to_wave


class SelectivePruningWrapper(nn.Module):
    """选择性修剪包装器"""
    
    def __init__(
        self,
        model: MiniJMamba,
        keep_ratio: float = 0.5,
        prune_mode: str = "channel",  # "channel", "spatial", "element"
    ):
        super().__init__()
        self.model = model
        self.keep_ratio = keep_ratio
        self.prune_mode = prune_mode
        
        # Hook 存储
        self._hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册 SSM 层的 hook"""
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, SSMLikeBlock):
                hook = layer.register_forward_hook(self._prune_hook)
                self._hooks.append(hook)
    
    def _prune_hook(self, module, input, output):
        """在 SSM 层输出后进行选择性修剪"""
        if self.keep_ratio >= 1.0:
            return output
        
        h = output  # (B, T, D)
        
        if self.prune_mode == "channel":
            # 按通道修剪：计算每个通道的平均能量
            strength = h.abs().mean(dim=(0, 1))  # (D,)
            threshold = torch.quantile(strength, 1 - self.keep_ratio)
            mask = (strength > threshold).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            return h * mask
        
        elif self.prune_mode == "element":
            # 按元素修剪
            strength = h.abs()
            threshold = torch.quantile(strength.flatten(), 1 - self.keep_ratio)
            mask = (strength > threshold).float()
            return h * mask
        
        return output
    
    def forward(self, frames, padding_mask, *, return_hidden=False):
        return self.model(frames, padding_mask, return_hidden=return_hidden)
    
    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def generate_test_sequence(num_symbols: int, seed: int) -> torch.Tensor:
    """生成测试序列"""
    np.random.seed(seed)
    symbols = list(SYMBOL2FREQ.keys())[:10]  # 0-9
    symbol_seq = [symbols[np.random.randint(0, len(symbols))] for _ in range(num_symbols)]
    wave = encode_symbols_to_wave(symbol_seq, tone_dur=0.1, gap_dur=0.0)
    return torch.tensor(wave, dtype=torch.float32)


def evaluate_model(
    model: nn.Module,
    num_samples: int,
    num_symbols: int,
    frame_size: int,
    device: str,
    seed_offset: int = 0,
) -> Dict[str, float]:
    """评估模型性能"""
    model.eval()
    
    total_entropy = 0.0
    total_max_prob = 0.0
    total_hidden_norm = 0.0
    
    for i in range(num_samples):
        wave = generate_test_sequence(num_symbols, seed=seed_offset * 1000 + i)
        num_frames = len(wave) // frame_size
        if num_frames == 0:
            continue
        
        frames = wave[:num_frames * frame_size].reshape(1, num_frames, frame_size).to(device)
        padding_mask = torch.ones(1, num_frames, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            output, logits, hidden = model(frames, padding_mask, return_hidden=True)
            
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            max_prob = probs.max(dim=-1).values.mean().item()
            hidden_norm = hidden.norm().item()
            
            total_entropy += entropy
            total_max_prob += max_prob
            total_hidden_norm += hidden_norm
    
    return {
        "entropy": total_entropy / num_samples,
        "max_prob": total_max_prob / num_samples,
        "hidden_norm": total_hidden_norm / num_samples,
    }


def run_sweep(
    checkpoint_path: str,
    keep_ratios: List[float],
    seeds: List[int],
    num_samples: int = 50,
    num_symbols: int = 32,
    frame_size: int = 160,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """运行参数扫描"""
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Keep ratios: {keep_ratios}")
    print(f"Seeds: {seeds}")
    print(f"Device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = MiniJMambaConfig(
        frame_size=frame_size,
        hop_size=frame_size,
        symbol_vocab_size=12,
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
    )
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        "keep_ratios": keep_ratios,
        "seeds": seeds,
        "num_samples": num_samples,
        "num_symbols": num_symbols,
        "data": [],
    }
    
    # Baseline (no pruning)
    print("\n=== Baseline (no pruning) ===")
    baseline_results = []
    for seed in seeds:
        torch.manual_seed(seed)
        base_model = MiniJMamba(config).to(device)
        if 'model_state_dict' in checkpoint:
            base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            base_model.load_state_dict(checkpoint, strict=False)
        base_model.eval()
        
        metrics = evaluate_model(base_model, num_samples, num_symbols, frame_size, device, seed)
        baseline_results.append(metrics)
        print(f"  Seed {seed}: max_prob={metrics['max_prob']:.3f}, entropy={metrics['entropy']:.3f}")
    
    baseline_max_probs = [r["max_prob"] for r in baseline_results]
    results["baseline"] = {
        "mean_max_prob": np.mean(baseline_max_probs),
        "std_max_prob": np.std(baseline_max_probs),
        "all_results": baseline_results,
    }
    
    # Sweep keep_ratios
    for keep_ratio in keep_ratios:
        print(f"\n=== keep_ratio = {keep_ratio} ===")
        ratio_results = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            base_model = MiniJMamba(config).to(device)
            if 'model_state_dict' in checkpoint:
                base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                base_model.load_state_dict(checkpoint, strict=False)
            
            model = SelectivePruningWrapper(base_model, keep_ratio=keep_ratio)
            model = model.to(device)
            model.eval()
            
            metrics = evaluate_model(model, num_samples, num_symbols, frame_size, device, seed)
            ratio_results.append(metrics)
            
            model.remove_hooks()
            print(f"  Seed {seed}: max_prob={metrics['max_prob']:.3f}, entropy={metrics['entropy']:.3f}")
        
        max_probs = [r["max_prob"] for r in ratio_results]
        
        # 计算相对于 baseline 的提升
        delta_probs = [p - b for p, b in zip(max_probs, baseline_max_probs)]
        
        # Bootstrap CI
        bootstrap_deltas = []
        for _ in range(1000):
            indices = np.random.choice(len(delta_probs), len(delta_probs), replace=True)
            bootstrap_deltas.append(np.mean([delta_probs[i] for i in indices]))
        
        ci_lower = np.percentile(bootstrap_deltas, 2.5)
        ci_upper = np.percentile(bootstrap_deltas, 97.5)
        
        # t-test
        t_stat, p_value = stats.ttest_rel(max_probs, baseline_max_probs)
        
        results["data"].append({
            "keep_ratio": keep_ratio,
            "mean_max_prob": np.mean(max_probs),
            "std_max_prob": np.std(max_probs),
            "mean_delta": np.mean(delta_probs),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "significant": ci_lower > 0 or ci_upper < 0,
            "all_results": ratio_results,
        })
        
        print(f"  Mean Δ: {np.mean(delta_probs)*100:.1f}pp, 95% CI: [{ci_lower*100:.1f}, {ci_upper*100:.1f}]pp, p={p_value:.4f}")
    
    return results


def print_summary(results: Dict[str, Any]):
    """打印结果摘要"""
    print("\n" + "="*70)
    print("SELECTIVE PRUNING SWEEP SUMMARY")
    print("="*70)
    
    baseline = results["baseline"]
    print(f"\nBaseline: max_prob = {baseline['mean_max_prob']:.3f} +/- {baseline['std_max_prob']:.3f}")
    
    print("\n| keep_ratio | mean_max_prob | Δ vs baseline | 95% CI | p-value | sig? |")
    print("|------------|---------------|---------------|--------|---------|------|")
    
    best_config = None
    best_delta = -float('inf')
    
    for d in results["data"]:
        sig = "[+]" if d["significant"] and d["mean_delta"] > 0 else "[-]" if d["significant"] else "[ ]"
        print(f"| {d['keep_ratio']:.1f} | {d['mean_max_prob']:.3f} | {d['mean_delta']*100:+.1f}pp | [{d['ci_lower']*100:.1f}, {d['ci_upper']*100:.1f}] | {d['p_value']:.4f} | {sig} |")
        
        if d["mean_delta"] > best_delta:
            best_delta = d["mean_delta"]
            best_config = d["keep_ratio"]
    
    print(f"\n[BEST] Best configuration: keep_ratio = {best_config} (Delta = {best_delta*100:+.1f}pp)")
    
    if best_delta > 0:
        print("\n[OK] Selective pruning IMPROVES performance!")
    else:
        print("\n[--] Selective pruning does not improve performance.")


def save_csv(results: Dict[str, Any], output_path: Path):
    """保存 CSV 格式结果"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["keep_ratio", "seed", "max_prob", "entropy", "hidden_norm"])
        
        # Baseline
        for i, r in enumerate(results["baseline"]["all_results"]):
            writer.writerow([1.0, results["seeds"][i], r["max_prob"], r["entropy"], r["hidden_norm"]])
        
        # Pruned
        for d in results["data"]:
            for i, r in enumerate(d["all_results"]):
                writer.writerow([d["keep_ratio"], results["seeds"][i], r["max_prob"], r["entropy"], r["hidden_norm"]])


def main():
    parser = argparse.ArgumentParser(description='Selective Pruning Parameter Sweep')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--keep-ratios', type=str, default='0.3,0.5,0.7')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1000,1001,1002,1003,1004,1005')
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--num-symbols', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='reports/pruning_sweep')
    
    args = parser.parse_args()
    
    keep_ratios = [float(x) for x in args.keep_ratios.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_sweep(
        checkpoint_path=args.checkpoint,
        keep_ratios=keep_ratios,
        seeds=seeds,
        num_samples=args.num_samples,
        num_symbols=args.num_symbols,
    )
    
    # 保存结果
    with open(output_dir / 'sweep_results.json', 'w', encoding='utf-8') as f:
        # 移除不可序列化的内容
        save_results = {k: v for k, v in results.items()}
        json.dump(save_results, f, indent=2, default=str)
    
    save_csv(results, output_dir / 'sweep_results.csv')
    
    print_summary(results)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()

