#!/usr/bin/env python3
"""
TSAE Hybrid Verification (Simplified)

用简化的 Hybrid 逻辑复核 TSAE 效应：
1. CTC 解码表达式
2. 用表达式计算余数（规则执行器）
3. 比较与 gold 的一致性

用法:
    python scripts/tsae_hybrid_verify.py --device cuda
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.scorer import decode_wave_to_symbols
from jericho.task3 import target_symbols_for_task3


def time_stretch(wave: np.ndarray, factor: float) -> np.ndarray:
    """Simple time stretch via linear interpolation"""
    if abs(factor - 1.0) < 1e-6:
        return wave
    original_len = len(wave)
    new_len = int(original_len * factor)
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, new_len)
    stretched = np.interp(x_new, x_old, wave).astype(np.float32)
    if len(stretched) > original_len:
        return stretched[:original_len]
    else:
        return np.pad(stretched, (0, original_len - len(stretched)))


def parse_and_compute_mod(symbols: list[str]) -> list[str] | None:
    """Parse expression and compute modulo result"""
    try:
        expr_str = "".join(symbols)
        # Match pattern: number % number
        match = re.match(r"(\d+)%(\d+)", expr_str)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            if b != 0:
                result = a % b
                return list(str(result))
        return None
    except:
        return None


def evaluate_hybrid(
    model: MiniJMamba,
    entries: list[ManifestEntry],
    device: str,
    perturbation_fn,
    checkpoint: dict,
    limit: int = 100,
) -> dict:
    """Evaluate using hybrid approach: CTC decode → rule executor"""
    
    model.eval()
    
    hop_size = model.config.hop_size
    frame_size = model.config.frame_size
    id_to_symbol = checkpoint.get("id_to_symbol", {})
    thinking_gap_s = 0.5
    SR = 16000
    SYMBOL_DURATION = 0.05
    
    ctc_correct = 0
    hybrid_correct = 0
    hybrid_parse_ok = 0
    audio_correct = 0
    total = 0
    
    with torch.no_grad():
        for entry in entries[:limit]:
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
            
            # === CTC Decode (answer window only) ===
            ctc_pred = []
            if symbol_logits is not None:
                expr_len_samples = int(len(entry.symbols) * SYMBOL_DURATION * SR)
                align = hop_size
                expr_len_aligned = ((expr_len_samples + align - 1) // align) * align
                thinking_gap_samples = int(round(thinking_gap_s * SR))
                thinking_gap_aligned = ((thinking_gap_samples + align - 1) // align) * align
                answer_start_frame = (expr_len_aligned + thinking_gap_aligned) // hop_size
                
                probs = symbol_logits.softmax(dim=-1).squeeze(0)
                if answer_start_frame < probs.size(0):
                    window_probs = probs[answer_start_frame:]
                else:
                    window_probs = probs[-1:]
                
                pred_ids = window_probs.argmax(dim=-1).cpu().tolist()
                prev_id = None
                for idx in pred_ids:
                    if idx != 0 and idx != prev_id:
                        symbol = id_to_symbol.get(idx, str(idx))
                        ctc_pred.append(symbol)
                    prev_id = idx
            
            # === Hybrid: CTC decode expression → compute mod ===
            # Decode full sequence for expression
            full_probs = symbol_logits.softmax(dim=-1).squeeze(0) if symbol_logits is not None else None
            expr_pred = []
            if full_probs is not None:
                expr_end_frame = (expr_len_aligned) // hop_size
                expr_probs = full_probs[:expr_end_frame] if expr_end_frame < full_probs.size(0) else full_probs
                pred_ids = expr_probs.argmax(dim=-1).cpu().tolist()
                prev_id = None
                for idx in pred_ids:
                    if idx != 0 and idx != prev_id:
                        symbol = id_to_symbol.get(idx, str(idx))
                        expr_pred.append(symbol)
                    prev_id = idx
            
            # Try to compute hybrid result
            hybrid_result = parse_and_compute_mod(expr_pred)
            if hybrid_result is not None:
                hybrid_parse_ok += 1
                if hybrid_result == target:
                    hybrid_correct += 1
            
            # === Audio decode ===
            pred_wave = frame_out.squeeze(0).cpu().numpy().flatten()
            pred_wave = np.clip(pred_wave, -1.0, 1.0).astype(np.float32)
            audio_pred = decode_wave_to_symbols(pred_wave)
            
            # Check matches
            if ctc_pred == target:
                ctc_correct += 1
            if audio_pred == target:
                audio_correct += 1
            
            total += 1
    
    return {
        "ctc_em": ctc_correct / total if total > 0 else 0,
        "hybrid_em": hybrid_correct / total if total > 0 else 0,
        "hybrid_parse_rate": hybrid_parse_ok / total if total > 0 else 0,
        "audio_em": audio_correct / total if total > 0 else 0,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="TSAE Hybrid Verification")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/mod_best_em0.75.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/task3.jsonl"))
    parser.add_argument("--split", type=str, default="iid_test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, default=Path("reports/tsae_hybrid_verify.json"))
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = checkpoint.get("model_config") or checkpoint.get("config", {})
    config = MiniJMambaConfig(**model_cfg)
    model = MiniJMamba(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()
    
    # Load manifest
    if not args.manifest.exists():
        alt = Path("manifests/task3_multistep.jsonl")
        if alt.exists():
            args.manifest = alt
    
    entries = [e for e in read_manifest(args.manifest) if e.split == args.split]
    print(f"Samples: {len(entries)}")
    
    # Test conditions
    conditions = {
        "clean": lambda w: w,
        "stretch_0.95": lambda w: time_stretch(w, 0.95),
        "stretch_1.00": lambda w: w,
        "stretch_1.05": lambda w: time_stretch(w, 1.05),
    }
    
    results = {
        "experiment": "tsae_hybrid_verify",
        "timestamp": datetime.now().isoformat(),
        "conditions": {},
    }
    
    print()
    print("=" * 75)
    print("TSAE HYBRID VERIFICATION")
    print("=" * 75)
    print(f"{'Condition':<15} {'Audio EM':<12} {'CTC EM':<12} {'Hybrid EM':<12} {'Parse Rate':<12}")
    print("-" * 75)
    
    for name, fn in conditions.items():
        r = evaluate_hybrid(model, entries, args.device, fn, checkpoint, args.limit)
        results["conditions"][name] = r
        print(f"{name:<15} {r['audio_em']:.2%}       {r['ctc_em']:.2%}       {r['hybrid_em']:.2%}       {r['hybrid_parse_rate']:.2%}")
    
    # Analysis
    print()
    print("=" * 75)
    print("TSAE CONCLUSION")
    print("=" * 75)
    
    r_095 = results["conditions"]["stretch_0.95"]
    r_105 = results["conditions"]["stretch_1.05"]
    
    delta_ctc = r_105["ctc_em"] - r_095["ctc_em"]
    delta_hybrid = r_105["hybrid_em"] - r_095["hybrid_em"]
    delta_audio = r_105["audio_em"] - r_095["audio_em"]
    
    print(f"Δ CTC EM (1.05x - 0.95x):    {delta_ctc:+.2%}")
    print(f"Δ Hybrid EM (1.05x - 0.95x): {delta_hybrid:+.2%}")
    print(f"Δ Audio EM (1.05x - 0.95x):  {delta_audio:+.2%}")
    print()
    
    # Determine conclusion
    if delta_ctc > 0.01 and delta_hybrid > 0.01:
        conclusion = "model_intrinsic"
        print("[PASS] TSAE persists across all decoding methods -> MODEL-INTRINSIC EFFECT")
    elif delta_ctc > 0.01 and abs(delta_hybrid) < 0.01:
        conclusion = "ctc_specific"
        print("[WARN] TSAE only in CTC, not Hybrid -> CTC-DECODER SPECIFIC")
    elif delta_ctc > 0.01 and delta_hybrid < -0.01:
        conclusion = "complex"
        print("[INFO] TSAE reverses in Hybrid -> COMPLEX INTERACTION")
    else:
        conclusion = "weak_or_none"
        print("[?] TSAE weak or absent in this setting")
    
    results["conclusion"] = conclusion
    results["deltas"] = {
        "ctc_em": delta_ctc,
        "hybrid_em": delta_hybrid,
        "audio_em": delta_audio,
    }
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
