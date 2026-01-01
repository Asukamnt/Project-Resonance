#!/usr/bin/env python3
"""
TSAE Multi-Checkpoint Verification

验证 TSAE 是否在多个不同 checkpoint 中一致存在。

用法:
    python scripts/tsae_multi_checkpoint.py --device cuda
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from jericho.data import read_manifest, synthesise_entry_wave
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.task3 import target_symbols_for_task3


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


def evaluate_tsae(checkpoint_path: Path, entries: list, device: str, limit: int = 100) -> dict:
    """Evaluate TSAE for a single checkpoint"""
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = checkpoint.get("model_config") or checkpoint.get("config", {})
    
    if not model_cfg:
        return {"error": "no model config"}
    
    config = MiniJMambaConfig(**model_cfg)
    model = MiniJMamba(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    hop_size = config.hop_size
    frame_size = config.frame_size
    id_to_symbol = checkpoint.get("id_to_symbol", {})
    
    SR = 16000
    SYMBOL_DURATION = 0.05
    thinking_gap_s = 0.5
    
    results = {}
    
    for factor in [0.95, 1.00, 1.05]:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for entry in entries[:limit]:
                target = target_symbols_for_task3(entry.symbols)
                
                input_wave = synthesise_entry_wave(entry)
                input_wave = time_stretch(input_wave, factor)
                
                wave_tensor = torch.from_numpy(input_wave).float()
                n_frames = (len(wave_tensor) - frame_size) // hop_size + 1
                if n_frames < 1:
                    continue
                
                frames = wave_tensor.unfold(0, frame_size, hop_size).unsqueeze(0).to(device)
                mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
                
                _, symbol_logits = model(frames, mask)
                
                if symbol_logits is not None:
                    expr_len_samples = int(len(entry.symbols) * SYMBOL_DURATION * SR)
                    align = hop_size
                    expr_len_aligned = ((expr_len_samples + align - 1) // align) * align
                    thinking_gap_samples = int(round(thinking_gap_s * SR))
                    thinking_gap_aligned = ((thinking_gap_samples + align - 1) // align) * align
                    answer_start_frame = (expr_len_aligned + thinking_gap_aligned) // hop_size
                    
                    probs = symbol_logits.softmax(dim=-1).squeeze(0)
                    window_probs = probs[answer_start_frame:] if answer_start_frame < probs.size(0) else probs[-1:]
                    
                    pred_ids = window_probs.argmax(dim=-1).cpu().tolist()
                    decoded = []
                    prev_id = None
                    for idx in pred_ids:
                        if idx != 0 and idx != prev_id:
                            symbol = id_to_symbol.get(idx, str(idx))
                            decoded.append(symbol)
                        prev_id = idx
                    
                    if decoded == target:
                        correct += 1
                
                total += 1
        
        results[f"{factor:.2f}x"] = correct / total if total > 0 else 0
    
    # Compute TSAE delta
    delta = results.get("1.05x", 0) - results.get("0.95x", 0)
    results["delta_1.05_0.95"] = delta
    results["tsae_present"] = delta > 0.01
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/task3.jsonl"))
    parser.add_argument("--split", type=str, default="iid_test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, default=Path("reports/tsae_multi_checkpoint.json"))
    args = parser.parse_args()
    
    # Find checkpoints
    checkpoints = [
        Path("artifacts/checkpoints/mod_best_em0.75.pt"),
        Path("runs/best_200ep/mod_seed123_epoch200.pt"),
        Path("runs/best_400ep/mod_seed123_epoch400.pt"),
        Path("runs/best_seed42/mod_seed42_epoch200.pt"),
        Path("runs/disjoint_tiny_200ep/mod_seed123_epoch200.pt"),
    ]
    
    checkpoints = [c for c in checkpoints if c.exists()]
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    # Load manifest
    entries = [e for e in read_manifest(args.manifest) if e.split == args.split]
    print(f"Samples: {len(entries)}, Checkpoints: {len(checkpoints)}")
    
    results = {
        "experiment": "tsae_multi_checkpoint",
        "timestamp": datetime.now().isoformat(),
        "checkpoints": {},
    }
    
    print()
    print("=" * 80)
    print("TSAE MULTI-CHECKPOINT VERIFICATION")
    print("=" * 80)
    print(f"{'Checkpoint':<45} {'0.95x':<8} {'1.00x':<8} {'1.05x':<8} {'Delta':<8} {'TSAE?'}")
    print("-" * 80)
    
    tsae_count = 0
    
    for ckpt in checkpoints:
        name = str(ckpt.relative_to(Path(".")))[:40]
        try:
            r = evaluate_tsae(ckpt, entries, args.device, args.limit)
            results["checkpoints"][str(ckpt)] = r
            
            em_095 = r.get("0.95x", 0)
            em_100 = r.get("1.00x", 0)
            em_105 = r.get("1.05x", 0)
            delta = r.get("delta_1.05_0.95", 0)
            tsae = "YES" if r.get("tsae_present", False) else "no"
            
            if r.get("tsae_present", False):
                tsae_count += 1
            
            print(f"{name:<45} {em_095:.1%}    {em_100:.1%}    {em_105:.1%}    {delta:+.1%}   {tsae}")
        except Exception as e:
            print(f"{name:<45} ERROR: {e}")
            results["checkpoints"][str(ckpt)] = {"error": str(e)}
    
    print("-" * 80)
    print(f"TSAE present in {tsae_count}/{len(checkpoints)} checkpoints")
    
    results["summary"] = {
        "total_checkpoints": len(checkpoints),
        "tsae_present_count": tsae_count,
        "tsae_reproducible": tsae_count >= len(checkpoints) * 0.8,
    }
    
    if tsae_count >= len(checkpoints) * 0.8:
        print("\n[PASS] TSAE is REPRODUCIBLE across checkpoints!")
    else:
        print("\n[WARN] TSAE is NOT consistently reproducible")
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

