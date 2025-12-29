#!/usr/bin/env python3
"""
生成发布 artifacts：checkpoint + 音频示例

用法:
    python scripts/generate_artifacts.py --task mirror --epochs 20
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.scorer import decode_wave_to_symbols
from jericho.symbols import SR, encode_symbols_to_wave
from jericho.task2 import target_symbol_for_task2, synthesise_task2_target_wave
from jericho.task3 import target_symbols_for_task3, synthesise_task3_target_wave


def train_and_save_checkpoint(
    manifest_path: Path,
    task: str,
    output_dir: Path,
    epochs: int = 30,
    device: str = "cuda",
    seed: int = 42,
    limit: int = 200,
) -> tuple[MiniJMamba, dict]:
    """训练模型并保存 checkpoint"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    entries = list(read_manifest(manifest_path))
    train_entries = [e for e in entries if e.split == "train"][:limit]
    
    # 词表
    symbols = set()
    for e in entries:
        symbols.update(e.symbols)
    symbol_to_id = {s: i + 1 for i, s in enumerate(sorted(symbols))}
    vocab_size = len(symbol_to_id) + 1
    
    # 模型配置
    config = MiniJMambaConfig(
        frame_size=160,
        hop_size=160,
        symbol_vocab_size=vocab_size,
        d_model=128,
        num_ssm_layers=10,
        num_attn_layers=2,
        num_heads=4,
        max_frames=256,
        dropout=0.1,
        attn_dropout=0.1,
        use_rope=True,
        use_learnable_pos=False,
    )
    
    model = MiniJMamba(config)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    
    print(f"Training {task} model for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for entry in train_entries:
            input_wave = synthesise_entry_wave(entry)
            input_tensor = torch.from_numpy(input_wave).float()
            
            input_frames = input_tensor.unfold(0, 160, 160)
            if input_frames.size(0) == 0:
                continue
            
            # 根据任务获取正确的目标
            if task == "mirror":
                target_symbols = list(entry.symbols)
                target_wave = encode_symbols_to_wave(target_symbols)
            elif task == "bracket":
                target_symbols = [target_symbol_for_task2(entry.symbols)]
                target_wave = synthesise_task2_target_wave(entry.symbols)
            elif task == "mod":
                target_symbols = target_symbols_for_task3(entry.symbols)
                target_wave = synthesise_task3_target_wave(entry.symbols)
            else:
                target_symbols = list(entry.symbols)
                target_wave = input_wave
            
            target_tensor = torch.from_numpy(target_wave).float()
            target_frames = target_tensor.unfold(0, 160, 160)
            
            # 对齐长度（取较短的）
            min_len = min(input_frames.size(0), target_frames.size(0))
            if min_len == 0:
                continue
            
            input_frames = input_frames[:min_len].unsqueeze(0).to(device)
            target_frames = target_frames[:min_len].unsqueeze(0).to(device)
            mask = torch.ones(1, min_len, dtype=torch.bool, device=device)
            
            frame_out, symbol_logits = model(input_frames, mask)
            
            # 音频损失：对齐目标波形
            audio_loss = ((frame_out - target_frames) ** 2).mean()
            
            # CTC 损失：对齐目标符号
            target_ids = [symbol_to_id.get(s, 0) for s in target_symbols]
            if target_ids:
                log_probs = symbol_logits.log_softmax(dim=-1).permute(1, 0, 2)
                input_lengths = torch.tensor([min_len], device=device)
                target_lengths = torch.tensor([len(target_ids)], device=device)
                targets = torch.tensor(target_ids, device=device)
                ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                loss = audio_loss + 0.3 * ctc_loss
            else:
                loss = audio_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_entries)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
    
    # 保存 checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{task}_demo_seed{seed}_epoch{epochs}.pt"
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "frame_size": config.frame_size,
            "hop_size": config.hop_size,
            "symbol_vocab_size": config.symbol_vocab_size,
            "d_model": config.d_model,
            "num_ssm_layers": config.num_ssm_layers,
            "num_attn_layers": config.num_attn_layers,
            "num_heads": config.num_heads,
            "max_frames": config.max_frames,
            "use_rope": config.use_rope,
        },
        "symbol_to_id": symbol_to_id,
        "task": task,
        "epochs": epochs,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    return model, checkpoint


def generate_audio_examples(
    model: MiniJMamba,
    manifest_path: Path,
    task: str,
    output_dir: Path,
    device: str = "cuda",
    n_examples: int = 5,
) -> list[dict]:
    """生成音频示例"""
    entries = list(read_manifest(manifest_path))
    
    # 选择不同 split 的示例
    examples = []
    for split in ["iid_test", "ood_length", "ood_compose", "ood_digits"]:
        split_entries = [e for e in entries if e.split == split][:n_examples]
        examples.extend(split_entries)
    
    if not examples:
        examples = entries[:n_examples * 2]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    
    model.eval()
    with torch.no_grad():
        for i, entry in enumerate(examples[:n_examples * 2]):
            # 生成输入波形
            input_wave = synthesise_entry_wave(entry)
            
            # 获取目标
            if task == "mirror":
                target = list(entry.symbols)
                target_wave = encode_symbols_to_wave(target)
            elif task == "bracket":
                target = [target_symbol_for_task2(entry.symbols)]
                target_wave = synthesise_task2_target_wave(entry.symbols)
            elif task == "mod":
                target = target_symbols_for_task3(entry.symbols)
                target_wave = synthesise_task3_target_wave(entry.symbols)
            
            # 模型推理
            wave_tensor = torch.from_numpy(input_wave).float()
            frames = wave_tensor.unfold(0, 160, 160)
            if frames.size(0) == 0:
                continue
            
            frames = frames.unsqueeze(0).to(device)
            mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
            
            frame_out, _ = model(frames, mask)
            pred_wave = frame_out.squeeze(0).cpu().numpy().flatten()
            pred_wave = np.clip(pred_wave, -1.0, 1.0).astype(np.float32)
            
            # 解码
            decoded = decode_wave_to_symbols(pred_wave)
            is_correct = decoded == target
            
            # 保存音频
            prefix = f"{task}_{entry.split}_{i:03d}"
            
            input_path = output_dir / f"{prefix}_input.wav"
            target_path = output_dir / f"{prefix}_target.wav"
            output_path = output_dir / f"{prefix}_output.wav"
            
            # 归一化到 int16
            def save_wav(path: Path, audio: np.ndarray):
                audio = np.clip(audio, -1.0, 1.0)
                audio_int16 = (audio * 32767).astype(np.int16)
                with wave.open(str(path), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(SR)
                    wf.writeframes(audio_int16.tobytes())
            
            save_wav(input_path, input_wave)
            save_wav(target_path, target_wave)
            save_wav(output_path, pred_wave)
            
            result = {
                "example_id": entry.example_id,
                "split": entry.split,
                "input_symbols": list(entry.symbols),
                "target_symbols": target,
                "predicted_symbols": decoded,
                "correct": is_correct,
                "files": {
                    "input": str(input_path.name),
                    "target": str(target_path.name),
                    "output": str(output_path.name),
                },
            }
            results.append(result)
            
            status = "OK" if is_correct else "FAIL"
            print(f"  [{status}] {prefix}: {entry.symbols[:5]}... -> {decoded}")
    
    # 保存索引
    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "examples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved index: {index_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="生成发布 artifacts")
    parser.add_argument("--task", type=str, default="mirror", choices=["mirror", "bracket", "mod"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifests-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    
    args = parser.parse_args()
    
    manifests = {
        "mirror": args.manifests_dir / "task1.jsonl",
        "bracket": args.manifests_dir / "task2.jsonl",
        "mod": args.manifests_dir / "task3_multistep.jsonl",
    }
    
    manifest_path = manifests[args.task]
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        sys.exit(1)
    
    print(f"Generating artifacts for task: {args.task}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # 1. 训练并保存 checkpoint
    print("\n[1/2] Training and saving checkpoint...")
    model, checkpoint = train_and_save_checkpoint(
        manifest_path=manifest_path,
        task=args.task,
        output_dir=args.output_dir / "checkpoints",
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )
    
    # 2. 生成音频示例
    print("\n[2/2] Generating audio examples...")
    examples = generate_audio_examples(
        model=model,
        manifest_path=manifest_path,
        task=args.task,
        output_dir=args.output_dir / "audio_examples",
        device=args.device,
    )
    
    print("\n" + "=" * 60)
    correct = sum(1 for e in examples if e["correct"])
    print(f"Audio examples: {correct}/{len(examples)} correct")
    print("Artifacts generation complete!")


if __name__ == "__main__":
    main()

