#!/usr/bin/env python3
"""
S22 消融套件：证明关键组件必要性

5 个核心消融：
1. no_attention: 删除 2 层 Attention（设置 num_attn_layers=0）
2. no_rope: 换回 learnable 绝对位置编码
3. no_ctc: 关闭 CTC 辅助损失
4. no_curriculum: 直接混合训练（不使用预训练阶段）
5. ssm_only: 仅使用 SSM 层（等同于 no_attention）

用法:
    python experiments/run_ablations.py --suite core5 --task mirror --epochs 10
    python experiments/run_ablations.py --ablation no_rope --task mirror
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.scorer import decode_wave_to_symbols, exact_match


def safe_unfold(wave_tensor: torch.Tensor, frame_size: int = 160, hop_size: int = 160) -> tuple[torch.Tensor, int]:
    """安全分帧：对波形进行 padding 确保不丢失尾部数据。"""
    original_len = len(wave_tensor)
    
    if original_len < frame_size:
        wave_tensor = torch.nn.functional.pad(wave_tensor, (0, frame_size - original_len))
        return wave_tensor.unsqueeze(0), 1
    
    remainder = (original_len - frame_size) % hop_size
    if remainder > 0:
        pad_len = hop_size - remainder
        wave_tensor = torch.nn.functional.pad(wave_tensor, (0, pad_len))
    
    frames = wave_tensor.unfold(0, frame_size, hop_size)
    valid_frames = (original_len - frame_size) // hop_size + 1
    
    return frames, valid_frames


@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    description: str
    # 模型配置修改
    num_attn_layers: int | None = None  # 覆盖 attention 层数
    use_rope: bool | None = None  # 是否使用 RoPE
    use_learnable_pos: bool | None = None  # 是否使用 learnable 位置编码
    # 训练配置修改
    ctc_weight: float | None = None  # CTC 损失权重
    pretrain_epochs: int | None = None  # 预训练 epochs（用于 no_curriculum）


# 核心 5 消融配置
CORE_ABLATIONS: dict[str, AblationConfig] = {
    "baseline": AblationConfig(
        name="baseline",
        description="完整模型（对照组）",
    ),
    "no_attention": AblationConfig(
        name="no_attention",
        description="删除 2 层 Attention，仅用 SSM",
        num_attn_layers=0,
    ),
    "no_rope": AblationConfig(
        name="no_rope",
        description="使用 learnable 绝对位置编码替代 RoPE",
        use_rope=False,
        use_learnable_pos=True,
    ),
    "no_ctc": AblationConfig(
        name="no_ctc",
        description="关闭 CTC 辅助损失",
        ctc_weight=0.0,
    ),
    "no_curriculum": AblationConfig(
        name="no_curriculum",
        description="不使用课程学习，直接混合训练",
        pretrain_epochs=0,
    ),
}


@dataclass
class AblationResult:
    """消融实验结果"""
    name: str
    task: str
    split: str
    em: float
    loss: float
    config_changes: dict = field(default_factory=dict)


def build_model_with_ablation(
    base_config: MiniJMambaConfig,
    ablation: AblationConfig,
) -> MiniJMamba:
    """根据消融配置构建模型"""
    # 复制配置
    config_dict = {
        "frame_size": base_config.frame_size,
        "hop_size": base_config.hop_size,
        "symbol_vocab_size": base_config.symbol_vocab_size,
        "d_model": base_config.d_model,
        "num_ssm_layers": base_config.num_ssm_layers,
        "num_attn_layers": base_config.num_attn_layers,
        "num_heads": base_config.num_heads,
        "max_frames": base_config.max_frames,
        "dropout": base_config.dropout,
        "attn_dropout": base_config.attn_dropout,
        "use_rope": base_config.use_rope,
        "use_learnable_pos": base_config.use_learnable_pos,
    }
    
    # 应用消融配置
    if ablation.num_attn_layers is not None:
        config_dict["num_attn_layers"] = ablation.num_attn_layers
        # 如果删除了 attention，增加 SSM 层以保持总层数
        if ablation.num_attn_layers == 0:
            config_dict["num_ssm_layers"] = base_config.num_ssm_layers + base_config.num_attn_layers
    
    if ablation.use_rope is not None:
        config_dict["use_rope"] = ablation.use_rope
    
    if ablation.use_learnable_pos is not None:
        config_dict["use_learnable_pos"] = ablation.use_learnable_pos
    
    config = MiniJMambaConfig(**config_dict)
    return MiniJMamba(config)


def run_quick_ablation(
    ablation: AblationConfig,
    manifest_path: Path,
    task: str = "mirror",
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 42,
    device: str = "cpu",
    limit: int | None = 50,
) -> AblationResult:
    """运行单个消融实验（快速版本，用于验证）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 读取数据
    entries = list(read_manifest(manifest_path))
    if limit:
        entries = entries[:limit]
    
    # 分割数据
    train_entries = [e for e in entries if e.split == "train"][:limit // 2 if limit else None]
    test_entries = [e for e in entries if e.split in ("val", "iid_test")][:limit // 2 if limit else None]
    
    if not train_entries:
        train_entries = entries[:len(entries) // 2]
    if not test_entries:
        test_entries = entries[len(entries) // 2:]
    
    # 构建词表
    symbols = set()
    for e in train_entries + test_entries:
        symbols.update(e.symbols)
    symbol_to_id = {s: i + 1 for i, s in enumerate(sorted(symbols))}
    vocab_size = len(symbol_to_id) + 1  # +1 for blank
    
    # 构建模型
    base_config = MiniJMambaConfig(
        frame_size=160,
        hop_size=160,
        symbol_vocab_size=vocab_size,
        d_model=64,  # 使用较小的模型加快消融
        num_ssm_layers=4,
        num_attn_layers=1,
        num_heads=2,
        max_frames=256,
        dropout=0.1,
        attn_dropout=0.1,
        use_rope=True,
        use_learnable_pos=False,
    )
    
    model = build_model_with_ablation(base_config, ablation)
    model = model.to(device)
    
    # 准备数据（使用 safe_unfold 避免丢失尾部数据）
    def prepare_sample(entry: ManifestEntry):
        wave = synthesise_entry_wave(entry)
        wave = torch.from_numpy(wave).float()
        frames, _ = safe_unfold(wave, 160, 160)
        return frames, [symbol_to_id.get(s, 0) for s in entry.symbols]
    
    train_data = [prepare_sample(e) for e in train_entries]
    test_data = [(prepare_sample(e), e) for e in test_entries]
    
    # 简化训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctc_weight = ablation.ctc_weight if ablation.ctc_weight is not None else 0.3
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for frames, target_ids in train_data:
            frames = frames.unsqueeze(0).to(device)
            mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
            
            frame_out, symbol_logits = model(frames, mask)
            
            # MSE 损失
            audio_loss = ((frame_out - frames) ** 2).mean()
            
            # CTC 损失
            if ctc_weight > 0 and target_ids:
                log_probs = symbol_logits.log_softmax(dim=-1).permute(1, 0, 2)
                input_lengths = torch.tensor([frames.size(1)], device=device)
                target_lengths = torch.tensor([len(target_ids)], device=device)
                targets = torch.tensor(target_ids, device=device)
                ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                loss = audio_loss + ctc_weight * ctc_loss
            else:
                loss = audio_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # 评估
    model.eval()
    correct = 0
    total = 0
    total_eval_loss = 0.0
    
    with torch.no_grad():
        for (frames, target_ids), entry in test_data:
            frames = frames.unsqueeze(0).to(device)
            mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
            
            frame_out, _ = model(frames, mask)
            
            # 评估损失
            eval_loss = ((frame_out - frames) ** 2).mean()
            total_eval_loss += eval_loss.item()
            
            # 解码
            pred_wave = frame_out.squeeze(0).cpu().numpy().flatten()
            pred_wave = np.clip(pred_wave, -1.0, 1.0).astype(np.float32)
            pred_symbols = decode_wave_to_symbols(pred_wave)
            
            if pred_symbols == list(entry.symbols):
                correct += 1
            total += 1
    
    em = correct / total if total > 0 else 0.0
    avg_loss = total_eval_loss / total if total > 0 else 0.0
    
    return AblationResult(
        name=ablation.name,
        task=task,
        split="test",
        em=em,
        loss=avg_loss,
        config_changes={
            "num_attn_layers": ablation.num_attn_layers,
            "use_rope": ablation.use_rope,
            "ctc_weight": ablation.ctc_weight,
        },
    )


def run_suite(
    suite: str = "core5",
    manifest_path: Path = Path("manifests/task1.jsonl"),
    task: str = "mirror",
    epochs: int = 10,
    device: str = "cpu",
    limit: int = 50,
) -> list[AblationResult]:
    """运行消融套件"""
    if suite == "core5":
        ablations = CORE_ABLATIONS
    else:
        raise ValueError(f"Unknown suite: {suite}")
    
    results = []
    baseline_em = None
    
    for name, ablation in ablations.items():
        print(f"Running ablation: {name} - {ablation.description}")
        
        result = run_quick_ablation(
            ablation=ablation,
            manifest_path=manifest_path,
            task=task,
            epochs=epochs,
            device=device,
            limit=limit,
        )
        
        if name == "baseline":
            baseline_em = result.em
        
        delta = result.em - baseline_em if baseline_em is not None else 0.0
        print(f"  EM={result.em:.3f} (delta={delta:+.3f}), loss={result.loss:.4f}")
        
        results.append(result)
    
    return results


def generate_report(results: list[AblationResult], output_path: Path) -> None:
    """生成消融报告"""
    # CSV 报告
    csv_path = output_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ablation", "task", "split", "em", "loss", "delta_em"])
        
        baseline_em = next((r.em for r in results if r.name == "baseline"), 0.0)
        
        for r in results:
            delta = r.em - baseline_em
            writer.writerow([r.name, r.task, r.split, f"{r.em:.4f}", f"{r.loss:.4f}", f"{delta:+.4f}"])
    
    print(f"CSV 报告已生成: {csv_path}")
    
    # Markdown 报告
    md_path = output_path.with_suffix(".md")
    
    lines = [
        "# S22 消融实验报告",
        "",
        f"> 生成时间: {datetime.now().isoformat()}",
        "",
        "## 消融配置",
        "",
        "| 消融 | 描述 |",
        "|------|------|",
    ]
    
    for name, cfg in CORE_ABLATIONS.items():
        lines.append(f"| {name} | {cfg.description} |")
    
    lines.extend([
        "",
        "## 结果汇总",
        "",
        "| 消融 | EM | Delta | Loss |",
        "|------|-----|-------|------|",
    ])
    
    baseline_em = next((r.em for r in results if r.name == "baseline"), 0.0)
    
    for r in results:
        delta = r.em - baseline_em
        status = "✅" if delta >= -0.05 else "⚠️" if delta >= -0.15 else "❌"
        lines.append(f"| {r.name} | {r.em:.3f} | {delta:+.3f} {status} | {r.loss:.4f} |")
    
    lines.extend([
        "",
        "## 关键发现",
        "",
    ])
    
    # 分析关键发现
    for r in results:
        if r.name == "baseline":
            continue
        delta = r.em - baseline_em
        if delta < -0.1:
            lines.append(f"- **{r.name}** 导致显著性能下降 ({delta:+.3f})，证明该组件必要")
        elif delta < -0.05:
            lines.append(f"- **{r.name}** 导致轻微性能下降 ({delta:+.3f})")
        else:
            lines.append(f"- **{r.name}** 对性能影响较小 ({delta:+.3f})")
    
    lines.extend([
        "",
        "## 结论",
        "",
        "TODO: 根据实际结果填写",
    ])
    
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown 报告已生成: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="S22 消融套件")
    parser.add_argument("--suite", type=str, default="core5", choices=["core5"])
    parser.add_argument("--ablation", type=str, choices=list(CORE_ABLATIONS.keys()))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/task1.jsonl"))
    parser.add_argument("--task", type=str, default="mirror", choices=["mirror", "bracket", "mod"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=50, help="限制样本数（加快测试）")
    parser.add_argument("--report", type=str, default="reports/ablations")
    
    args = parser.parse_args()
    
    if args.ablation:
        # 运行单个消融
        ablation = CORE_ABLATIONS[args.ablation]
        result = run_quick_ablation(
            ablation=ablation,
            manifest_path=args.manifest,
            task=args.task,
            epochs=args.epochs,
            device=args.device,
            limit=args.limit,
        )
        print(f"Result: {args.ablation}")
        print(f"  EM={result.em:.3f}, loss={result.loss:.4f}")
    else:
        # 运行完整套件
        results = run_suite(
            suite=args.suite,
            manifest_path=args.manifest,
            task=args.task,
            epochs=args.epochs,
            device=args.device,
            limit=args.limit,
        )
        generate_report(results, Path(args.report))


if __name__ == "__main__":
    main()
