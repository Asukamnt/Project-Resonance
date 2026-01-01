#!/usr/bin/env python3
"""
Dropout 对照实验

目的：证明"选择性修剪"的效果不是简单的正则化
- 如果 Dropout 也能提升性能 → 可能只是正则化效果
- 如果 Dropout 无效而选择性修剪有效 → 证明"选择性"是关键

Author: Jericho Team
Date: 2026-01-02
"""

import argparse
import json
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


class DropoutWrapper(nn.Module):
    """Dropout 包装器（作为对照）"""
    
    def __init__(self, model: MiniJMamba, dropout_rate: float = 0.5):
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        
        self._hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        for layer in self.model.layers:
            if isinstance(layer, SSMLikeBlock):
                hook = layer.register_forward_hook(self._dropout_hook)
                self._hooks.append(hook)
    
    def _dropout_hook(self, module, input, output):
        # 注意：eval 模式下 Dropout 不生效，所以我们手动应用
        if self.dropout_rate > 0:
            mask = torch.rand_like(output) > self.dropout_rate
            return output * mask / (1 - self.dropout_rate)
        return output
    
    def forward(self, frames, padding_mask, *, return_hidden=False):
        return self.model(frames, padding_mask, return_hidden=return_hidden)
    
    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class SelectivePruningWrapper(nn.Module):
    """选择性修剪包装器"""
    
    def __init__(self, model: MiniJMamba, keep_ratio: float = 0.5):
        super().__init__()
        self.model = model
        self.keep_ratio = keep_ratio
        
        self._hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        for layer in self.model.layers:
            if isinstance(layer, SSMLikeBlock):
                hook = layer.register_forward_hook(self._prune_hook)
                self._hooks.append(hook)
    
    def _prune_hook(self, module, input, output):
        if self.keep_ratio >= 1.0:
            return output
        
        h = output
        strength = h.abs().mean(dim=(0, 1))
        threshold = torch.quantile(strength, 1 - self.keep_ratio)
        mask = (strength > threshold).float().unsqueeze(0).unsqueeze(0)
        return h * mask
    
    def forward(self, frames, padding_mask, *, return_hidden=False):
        return self.model(frames, padding_mask, return_hidden=return_hidden)
    
    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def generate_test_sequence(num_symbols: int, seed: int) -> torch.Tensor:
    np.random.seed(seed)
    symbols = list(SYMBOL2FREQ.keys())[:10]
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
    # 不用 eval() 因为我们要让 Dropout 生效
    
    total_max_prob = 0.0
    total_entropy = 0.0
    
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
            
            total_max_prob += max_prob
            total_entropy += entropy
    
    return {
        "max_prob": total_max_prob / num_samples,
        "entropy": total_entropy / num_samples,
    }


def run_control_experiment(
    checkpoint_path: str,
    rates: List[float],
    seeds: List[int],
    num_samples: int = 50,
    num_symbols: int = 32,
    frame_size: int = 160,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Rates: {rates}")
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
        "rates": rates,
        "seeds": seeds,
        "baseline": {},
        "dropout": [],
        "selective": [],
    }
    
    # Baseline
    print("\n=== Baseline (no modification) ===")
    baseline_probs = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = MiniJMamba(config).to(device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        metrics = evaluate_model(model, num_samples, num_symbols, frame_size, device, seed)
        baseline_probs.append(metrics["max_prob"])
        print(f"  Seed {seed}: max_prob={metrics['max_prob']:.3f}")
    
    results["baseline"]["mean"] = np.mean(baseline_probs)
    results["baseline"]["std"] = np.std(baseline_probs)
    results["baseline"]["all"] = baseline_probs
    
    # Dropout 对照
    for rate in rates:
        print(f"\n=== Dropout rate = {rate} ===")
        dropout_probs = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            base_model = MiniJMamba(config).to(device)
            if 'model_state_dict' in checkpoint:
                base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                base_model.load_state_dict(checkpoint, strict=False)
            
            model = DropoutWrapper(base_model, dropout_rate=rate).to(device)
            
            metrics = evaluate_model(model, num_samples, num_symbols, frame_size, device, seed)
            dropout_probs.append(metrics["max_prob"])
            
            model.remove_hooks()
            print(f"  Seed {seed}: max_prob={metrics['max_prob']:.3f}")
        
        delta = [d - b for d, b in zip(dropout_probs, baseline_probs)]
        t_stat, p_value = stats.ttest_rel(dropout_probs, baseline_probs)
        
        results["dropout"].append({
            "rate": rate,
            "mean": np.mean(dropout_probs),
            "std": np.std(dropout_probs),
            "mean_delta": np.mean(delta),
            "p_value": p_value,
            "all": dropout_probs,
        })
        print(f"  Mean Δ: {np.mean(delta)*100:+.1f}pp, p={p_value:.4f}")
    
    # 选择性修剪
    for rate in rates:
        keep_ratio = 1 - rate  # 等效 keep_ratio
        print(f"\n=== Selective Pruning keep={keep_ratio} (drop {rate*100}%) ===")
        prune_probs = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            base_model = MiniJMamba(config).to(device)
            if 'model_state_dict' in checkpoint:
                base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                base_model.load_state_dict(checkpoint, strict=False)
            
            model = SelectivePruningWrapper(base_model, keep_ratio=keep_ratio).to(device)
            model.eval()
            
            metrics = evaluate_model(model, num_samples, num_symbols, frame_size, device, seed)
            prune_probs.append(metrics["max_prob"])
            
            model.remove_hooks()
            print(f"  Seed {seed}: max_prob={metrics['max_prob']:.3f}")
        
        delta = [p - b for p, b in zip(prune_probs, baseline_probs)]
        t_stat, p_value = stats.ttest_rel(prune_probs, baseline_probs)
        
        results["selective"].append({
            "keep_ratio": keep_ratio,
            "mean": np.mean(prune_probs),
            "std": np.std(prune_probs),
            "mean_delta": np.mean(delta),
            "p_value": p_value,
            "all": prune_probs,
        })
        print(f"  Mean Δ: {np.mean(delta)*100:+.1f}pp, p={p_value:.4f}")
    
    return results


def print_summary(results: Dict[str, Any]):
    print("\n" + "="*70)
    print("DROPOUT vs SELECTIVE PRUNING CONTROL EXPERIMENT")
    print("="*70)
    
    print(f"\nBaseline: max_prob = {results['baseline']['mean']:.3f}")
    
    print("\n| Method | Rate/Keep | Δ vs baseline | p-value | Better? |")
    print("|--------|-----------|---------------|---------|---------|")
    
    for d in results["dropout"]:
        better = "[+]" if d["mean_delta"] > 0.01 and d["p_value"] < 0.05 else "[-]"
        print(f"| Dropout | {d['rate']:.1f} | {d['mean_delta']*100:+.1f}pp | {d['p_value']:.4f} | {better} |")
    
    for s in results["selective"]:
        better = "[+]" if s["mean_delta"] > 0.01 and s["p_value"] < 0.05 else "[-]"
        print(f"| Selective | keep={s['keep_ratio']:.1f} | {s['mean_delta']*100:+.1f}pp | {s['p_value']:.4f} | {better} |")
    
    # 判断
    dropout_helps = any(d["mean_delta"] > 0.01 and d["p_value"] < 0.05 for d in results["dropout"])
    selective_helps = any(s["mean_delta"] > 0.01 and s["p_value"] < 0.05 for s in results["selective"])
    
    print("\n" + "="*70)
    if selective_helps and not dropout_helps:
        print("[OK] CONCLUSION: Selectivity is the key, not just regularization!")
        print("     Dropout does NOT help, but Selective Pruning DOES.")
    elif selective_helps and dropout_helps:
        print("[??] CONCLUSION: Both help. May need further investigation.")
    elif not selective_helps and not dropout_helps:
        print("[--] CONCLUSION: Neither method helps at these rates.")
    else:
        print("[??] CONCLUSION: Only Dropout helps. Selective pruning may not be the key factor.")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Dropout Control Experiment')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--rates', type=str, default='0.3,0.5,0.7')
    parser.add_argument('--seeds', type=str, default='42,123,456')
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--num-symbols', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='reports/dropout_control')
    
    args = parser.parse_args()
    
    rates = [float(x) for x in args.rates.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_control_experiment(
        checkpoint_path=args.checkpoint,
        rates=rates,
        seeds=seeds,
        num_samples=args.num_samples,
        num_symbols=args.num_symbols,
    )
    
    with open(output_dir / 'control_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_summary(results)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()

