#!/usr/bin/env python3
"""
增生 + 睡眠再训练实验 - 使用固定 Top-k

不使用 L0 正则化，直接使用固定的 Top-k 选择。
这确保 keep_ratio 严格等于设定值。

Author: Jericho Team
Date: 2026-01-02
"""

import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig
from jericho.training.replay_buffer import ReplayBuffer, WakeSleepScheduler
from jericho.symbols import encode_symbols_to_wave, SR
from jericho.data.manifest import read_manifest


class TopKPruningGate(nn.Module):
    """固定 Top-k 选择门控"""
    
    def __init__(self, dim: int, keep_ratio: float = 0.7):
        super().__init__()
        self.dim = dim
        self.keep_ratio = keep_ratio
        self.k = max(1, int(dim * keep_ratio))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            pruned_x: (batch, seq_len, dim)
        """
        # 计算每个通道在整个 batch 上的平均 L2 范数
        channel_importance = torch.norm(x, p=2, dim=(0, 1))  # (dim,)
        
        # 选择 top-k 重要的通道
        _, top_indices = torch.topk(channel_importance, self.k)
        
        # 创建掩码
        mask = torch.zeros(self.dim, device=x.device)
        mask[top_indices] = 1.0
        
        # 应用掩码
        return x * mask.view(1, 1, -1)


class TopKMiniJMamba(nn.Module):
    """带 Top-k 门控的 MiniJMamba"""
    
    def __init__(
        self,
        base_model: MiniJMamba,
        keep_ratio: float = 0.7,
        gate_every_n_layers: int = 2,
    ):
        super().__init__()
        self.base_model = base_model
        self.keep_ratio = keep_ratio
        self.gate_every_n_layers = gate_every_n_layers
        
        d_model = base_model.symbol_head.in_features
        num_gates = len(base_model.layers) // gate_every_n_layers
        self.gates = nn.ModuleList([
            TopKPruningGate(d_model, keep_ratio=keep_ratio)
            for _ in range(num_gates)
        ])
        
        self.gates_enabled = True
    
    def forward(
        self,
        frames: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.base_model.input_proj(frames)
        x = self.base_model.dropout(x)
        
        gate_idx = 0
        for i, layer in enumerate(self.base_model.layers):
            x = layer(x, padding_mask)
            
            if self.gates_enabled and (i + 1) % self.gate_every_n_layers == 0:
                if gate_idx < len(self.gates):
                    x = self.gates[gate_idx](x)
                    gate_idx += 1
        
        x = self.base_model.final_norm(x)
        return self.base_model.symbol_head(x)


class SimpleModDataset(Dataset):
    def __init__(self, entries, frame_size: int = 160, max_samples: int = None):
        self.entries = entries[:max_samples] if max_samples else entries
        self.frame_size = frame_size
        self.samples = []
        
        for entry in self.entries:
            symbols = list(entry.symbols)
            try:
                wave = encode_symbols_to_wave(symbols, tone_dur=0.01, sr=SR)
                num_frames = len(wave) // frame_size
                if num_frames > 0:
                    wave = wave[:num_frames * frame_size]
                    frames = wave.reshape(num_frames, frame_size)
                    self.samples.append({
                        "frames": torch.tensor(frames, dtype=torch.float32),
                        "symbols": symbols,
                    })
            except:
                continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    max_len = max(b["frames"].shape[0] for b in batch)
    frames_list = []
    masks_list = []
    
    for b in batch:
        f = b["frames"]
        pad_len = max_len - f.shape[0]
        padded = F.pad(f, (0, 0, 0, pad_len))
        frames_list.append(padded)
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:f.shape[0]] = True
        masks_list.append(mask)
    
    return {"frames": torch.stack(frames_list), "masks": torch.stack(masks_list)}


def run_experiment(
    group: str,
    keep_ratio: float,
    checkpoint_path: str,
    manifest_path: str,
    device: str,
    seed: int,
    epochs: int = 30,
    widen_factor: float = 1.5,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Group {group}: keep_ratio={keep_ratio} (seed={seed})")
    print(f"{'='*60}")
    
    # 加载模型配置
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = ckpt["config"]
    
    d_model_base = saved_config.get("d_model", 128)
    d_model = int(d_model_base * widen_factor) if group != 'A' else d_model_base
    
    model_config = MiniJMambaConfig(
        frame_size=saved_config.get("frame_size", 160),
        hop_size=saved_config.get("hop_size", 160),
        symbol_vocab_size=saved_config.get("symbol_vocab_size", 12),
        d_model=d_model,
        num_ssm_layers=saved_config.get("num_ssm_layers", 10),
        num_attn_layers=saved_config.get("num_attn_layers", 2),
        max_frames=saved_config.get("max_frames", 256),
        use_rope=saved_config.get("use_rope", True),
    )
    
    base_model = MiniJMamba(model_config).to(device)
    
    if group == 'A':
        base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model = base_model
    elif group == 'B':
        model = base_model  # 增生但无修剪
    else:  # C, D
        model = TopKMiniJMamba(base_model, keep_ratio=keep_ratio).to(device)
    
    # 数据
    entries = list(read_manifest(Path(manifest_path)))[:1000]
    dataset = SimpleModDataset(entries)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = WakeSleepScheduler(cycle_epochs=5, warmup_epochs=5)
    buffer = ReplayBuffer(capacity=5000)
    
    history = {"loss": [], "phase": []}
    
    for epoch in range(epochs):
        model.train()
        phase = scheduler.get_phase(epoch) if group == 'D' else ("wake" if group != 'C' else "sleep_prune")
        
        epoch_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            frames = batch["frames"].to(device)
            masks = batch["masks"].to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, TopKMiniJMamba):
                symbol_logits = model(frames, masks)
            else:
                _, symbol_logits, _ = model(frames, masks, return_hidden=True)
            
            pred = symbol_logits[:, :-1, :]
            target = symbol_logits[:, 1:, :].argmax(dim=-1)
            loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
            
            if phase != "sleep_prune" or group != 'D':
                loss.backward()
                optimizer.step()
            
            if group == 'D' and phase == "wake":
                buffer.add(batch, loss.item(), epoch * len(dataloader) + n_batches)
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)
        history["phase"].append(phase)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, phase={phase}")
    
    # 评估
    model.eval()
    final_losses = []
    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device)
            masks = batch["masks"].to(device)
            
            if isinstance(model, TopKMiniJMamba):
                symbol_logits = model(frames, masks)
            else:
                _, symbol_logits, _ = model(frames, masks, return_hidden=True)
            
            pred = symbol_logits[:, :-1, :]
            target = symbol_logits[:, 1:, :].argmax(dim=-1)
            loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
            final_losses.append(loss.item())
    
    final_loss = np.mean(final_losses)
    
    result = {
        "group": group,
        "seed": seed,
        "keep_ratio": keep_ratio,
        "d_model": d_model,
        "final_loss": final_loss,
        "history": history,
    }
    
    print(f"  Final: loss={final_loss:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Neurogenesis + Top-k Pruning")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/mod_best_em0.75.pt")
    parser.add_argument("--manifest", type=str, default="manifests/task3.jsonl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--keep-ratio", type=float, default=0.7)
    parser.add_argument("--widen-factor", type=float, default=1.5)
    parser.add_argument("--output-dir", type=str, default="reports/neurogenesis_topk")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = []
    
    for group in ['A', 'B', 'C', 'D']:
        for seed in seeds:
            result = run_experiment(
                group=group,
                keep_ratio=args.keep_ratio,
                checkpoint_path=args.checkpoint,
                manifest_path=args.manifest,
                device=device,
                seed=seed,
                epochs=args.epochs,
                widen_factor=args.widen_factor,
            )
            all_results.append(result)
    
    # 保存
    output_file = output_dir / "topk_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for group in ['A', 'B', 'C', 'D']:
        group_results = [r for r in all_results if r["group"] == group]
        losses = [r["final_loss"] for r in group_results]
        print(f"Group {group}: loss={np.mean(losses):.4f}±{np.std(losses):.4f}")
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

