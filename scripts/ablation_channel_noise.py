#!/usr/bin/env python3
"""
Channel Noise Robustness Test

测试模型对不同信道扰动的鲁棒性：
- AWGN (不同 SNR)
- 随机相位偏移
- 轻微 time stretch

用法:
    python scripts/ablation_channel_noise.py --checkpoint runs/mod_best.pt --device cuda
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.scorer import decode_wave_to_symbols
from jericho.task3 import target_symbols_for_task3


# ============================================================
# Channel Perturbations
# ============================================================

def add_awgn(wave: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Additive White Gaussian Noise at specified SNR"""
    signal_power = np.mean(wave ** 2)
    if signal_power < 1e-10:
        return wave
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*wave.shape) * np.sqrt(noise_power)
    return (wave + noise).astype(np.float32)


def random_phase_offset(wave: np.ndarray, max_offset: float = np.pi) -> np.ndarray:
    """Apply random phase offset (simple approximation via shift)"""
    # Approximate phase shift via time shift
    sr = 16000
    max_shift = int(max_offset / (2 * np.pi) * sr * 0.01)  # ~1% of a cycle
    if max_shift < 1:
        return wave
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        return np.concatenate([np.zeros(shift, dtype=np.float32), wave[:-shift]])
    elif shift < 0:
        return np.concatenate([wave[-shift:], np.zeros(-shift, dtype=np.float32)])
    return wave


def time_stretch(wave: np.ndarray, factor: float) -> np.ndarray:
    """Simple time stretch via linear interpolation"""
    if abs(factor - 1.0) < 1e-6:
        return wave
    original_len = len(wave)
    new_len = int(original_len * factor)
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, new_len)
    stretched = np.interp(x_new, x_old, wave).astype(np.float32)
    # Pad or truncate to original length
    if len(stretched) > original_len:
        return stretched[:original_len]
    else:
        return np.pad(stretched, (0, original_len - len(stretched)))


# ============================================================
# Evaluation
# ============================================================

def evaluate_with_perturbation(
    model: MiniJMamba,
    entries: list[ManifestEntry],
    perturbation_fn: Callable[[np.ndarray], np.ndarray],
    device: str,
    checkpoint: dict,
) -> dict:
    """Evaluate model with a specific perturbation applied to inputs"""
    
    model.eval()
    correct = 0
    total = 0
    
    hop_size = model.config.hop_size
    frame_size = model.config.frame_size
    id_to_symbol = checkpoint.get("id_to_symbol", {})
    thinking_gap_s = checkpoint.get("thinking_gap_s", 0.5)
    
    with torch.no_grad():
        for entry in entries:
            target = target_symbols_for_task3(entry.symbols)
            
            # Generate and perturb input wave
            input_wave = synthesise_entry_wave(entry)
            input_wave = perturbation_fn(input_wave)
            
            # Frame it
            wave_tensor = torch.from_numpy(input_wave).float()
            n_frames = (len(wave_tensor) - frame_size) // hop_size + 1
            if n_frames < 1:
                continue
            
            frames = wave_tensor.unfold(0, frame_size, hop_size).unsqueeze(0).to(device)
            mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
            
            # Inference
            frame_out, symbol_logits = model(frames, mask)
            
            # Decode (answer window only for Task3)
            if symbol_logits is not None:
                SR = 16000
                SYMBOL_DURATION = 0.05
                expr_len_samples = int(len(entry.symbols) * SYMBOL_DURATION * SR)
                align = hop_size
                expr_len_aligned = ((expr_len_samples + align - 1) // align) * align
                thinking_gap_samples = int(round(thinking_gap_s * SR))
                thinking_gap_aligned = ((thinking_gap_samples + align - 1) // align) * align if thinking_gap_samples > 0 else 0
                answer_start_samples = expr_len_aligned + thinking_gap_aligned
                answer_start_frame = answer_start_samples // hop_size
                
                probs = symbol_logits.softmax(dim=-1).squeeze(0)
                if answer_start_frame < probs.size(0):
                    window_probs = probs[answer_start_frame:]
                else:
                    window_probs = probs[-1:]
                
                pred_ids = window_probs.argmax(dim=-1).cpu().tolist()
                decoded = []
                prev_id = None
                for idx in pred_ids:
                    if idx != 0 and idx != prev_id:
                        symbol = id_to_symbol.get(idx, str(idx))
                        decoded.append(symbol)
                    prev_id = idx
            else:
                pred_wave = frame_out.squeeze(0).cpu().numpy().flatten()
                pred_wave = np.clip(pred_wave, -1.0, 1.0).astype(np.float32)
                decoded = decode_wave_to_symbols(pred_wave)
            
            if decoded == target:
                correct += 1
            total += 1
    
    em = correct / total if total > 0 else 0.0
    return {"correct": correct, "total": total, "em": em}


def main():
    parser = argparse.ArgumentParser(description="Channel Noise Robustness Test")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/task3.jsonl"),
        help="Manifest file",
    )
    parser.add_argument("--split", type=str, default="iid_test", help="Split to evaluate")
    parser.add_argument("--limit", type=int, default=100, help="Max samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ablation_channel_noise.json"),
        help="Output JSON path",
    )
    
    args = parser.parse_args()
    
    # Load checkpoint
    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = MiniJMambaConfig(**checkpoint.get("model_config", {}))
    model = MiniJMamba(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()
    
    # Load manifest
    if not args.manifest.exists():
        alt = Path("manifests/task3_multistep.jsonl")
        if alt.exists():
            args.manifest = alt
        else:
            print(f"Error: manifest not found")
            sys.exit(1)
    
    entries = [e for e in read_manifest(args.manifest) if e.split == args.split]
    if args.limit:
        entries = entries[:args.limit]
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Manifest: {args.manifest}")
    print(f"Samples: {len(entries)}")
    
    # Define perturbations
    perturbations = {
        "clean": lambda w: w,
        "awgn_30db": lambda w: add_awgn(w, 30),
        "awgn_20db": lambda w: add_awgn(w, 20),
        "awgn_10db": lambda w: add_awgn(w, 10),
        "awgn_5db": lambda w: add_awgn(w, 5),
        "phase_offset": lambda w: random_phase_offset(w, np.pi / 4),
        "time_stretch_0.95": lambda w: time_stretch(w, 0.95),
        "time_stretch_1.05": lambda w: time_stretch(w, 1.05),
    }
    
    results = {}
    print(f"\n{'='*60}")
    print("Running channel perturbation tests...")
    print(f"{'='*60}")
    
    for name, fn in perturbations.items():
        print(f"  {name}...", end=" ", flush=True)
        np.random.seed(42)  # Reproducible noise
        result = evaluate_with_perturbation(model, entries, fn, args.device, checkpoint)
        results[name] = result
        print(f"EM = {result['em']:.4f}")
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "channel_noise_robustness",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "checkpoint": str(args.checkpoint),
            "manifest": str(args.manifest),
            "split": args.split,
            "limit": args.limit,
        },
        "results": results,
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Perturbation':<25} {'EM':<10}")
    print("-" * 35)
    for name, r in results.items():
        print(f"{name:<25} {r['em']:.4f}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

