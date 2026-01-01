#!/usr/bin/env python3
"""
模型评测总表脚本 - 统一输出 Oracle + Model 结果

用法:
    # 完整评测（Oracle + Model）
    python evaluate_model.py --checkpoint runs/latest.pt --tasks mirror bracket mod
    
    # 仅 Oracle 验证
    python evaluate_model.py --oracle-only --tasks mirror
    
    # 消融 OOD 曲线
    python evaluate_model.py --ablation-ood --splits iid_test ood_length

产物:
    reports/eval_summary.md        # 总表（Oracle + Model）
    reports/eval_summary.json      # JSON 格式
    reports/ablation_ood.md        # 消融 OOD 曲线（如果运行消融）
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.models import MiniJMamba, MiniJMambaConfig
from jericho.scorer import decode_wave_to_symbols
from jericho.task2 import target_symbol_for_task2, synthesise_task2_target_wave
from jericho.task3 import target_symbols_for_task3, synthesise_task3_target_wave
from jericho.symbols import encode_symbols_to_wave


@dataclass
class EvalResult:
    """单条评测结果"""
    task: str
    split: str
    eval_type: str  # "oracle" or "model"
    total: int
    correct: int
    em: float
    token_accuracy: float
    
    def to_dict(self) -> dict:
        return asdict(self)


def levenshtein_distance(s1: list[str], s2: list[str]) -> int:
    """Levenshtein 编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def safe_unfold(wave_tensor: torch.Tensor, frame_size: int = 160, hop_size: int = 160) -> tuple[torch.Tensor, int]:
    """
    安全分帧：对波形进行 padding 确保不丢失尾部数据。
    
    Returns:
        frames: 分帧后的张量 (num_frames, frame_size)
        valid_frames: 有效帧数（不含 padding 帧）
    """
    original_len = len(wave_tensor)
    
    # 计算需要的 padding
    if original_len < frame_size:
        # 波形太短，padding 到 frame_size
        wave_tensor = torch.nn.functional.pad(wave_tensor, (0, frame_size - original_len))
        return wave_tensor.unsqueeze(0), 1
    
    # 计算最后不够一个 hop 的部分
    remainder = (original_len - frame_size) % hop_size
    if remainder > 0:
        pad_len = hop_size - remainder
        wave_tensor = torch.nn.functional.pad(wave_tensor, (0, pad_len))
    
    frames = wave_tensor.unfold(0, frame_size, hop_size)
    
    # 计算有效帧数（原始数据覆盖的帧）
    valid_frames = (original_len - frame_size) // hop_size + 1
    
    return frames, valid_frames


def get_target_symbols(entry: ManifestEntry, task: str) -> list[str]:
    """获取目标符号"""
    if task == "mirror":
        return list(entry.symbols)
    elif task == "bracket":
        return [target_symbol_for_task2(entry.symbols)]
    elif task == "mod":
        return target_symbols_for_task3(entry.symbols)
    else:
        raise ValueError(f"Unknown task: {task}")


def get_target_wave(entry: ManifestEntry, task: str) -> np.ndarray:
    """获取目标波形"""
    if task == "mirror":
        return encode_symbols_to_wave(list(entry.symbols))
    elif task == "bracket":
        return synthesise_task2_target_wave(entry.symbols)
    elif task == "mod":
        return synthesise_task3_target_wave(entry.symbols)
    else:
        raise ValueError(f"Unknown task: {task}")


def eval_oracle(
    manifest_path: Path,
    task: str,
    split: str,
    limit: int | None = None,
) -> EvalResult:
    """Oracle/Protocol 评测"""
    entries = [e for e in read_manifest(manifest_path) if e.split == split]
    if limit:
        entries = entries[:limit]
    
    if not entries:
        return EvalResult(task=task, split=split, eval_type="oracle",
                         total=0, correct=0, em=0.0, token_accuracy=0.0)
    
    correct = 0
    total_tokens = 0
    correct_tokens = 0
    
    for entry in entries:
        target = get_target_symbols(entry, task)
        wave = get_target_wave(entry, task)
        decoded = decode_wave_to_symbols(wave)
        
        if decoded == target:
            correct += 1
        
        min_len = min(len(decoded), len(target))
        for i in range(min_len):
            if decoded[i] == target[i]:
                correct_tokens += 1
        total_tokens += max(len(decoded), len(target))
    
    total = len(entries)
    return EvalResult(
        task=task, split=split, eval_type="oracle",
        total=total, correct=correct,
        em=correct / total if total > 0 else 0.0,
        token_accuracy=correct_tokens / total_tokens if total_tokens > 0 else 0.0,
    )


def eval_model(
    model: MiniJMamba,
    manifest_path: Path,
    task: str,
    split: str,
    device: str = "cpu",
    limit: int | None = None,
    checkpoint: dict | None = None,
) -> EvalResult:
    """训练模型评测"""
    entries = [e for e in read_manifest(manifest_path) if e.split == split]
    if limit:
        entries = entries[:limit]
    
    if not entries:
        return EvalResult(task=task, split=split, eval_type="model",
                         total=0, correct=0, em=0.0, token_accuracy=0.0)
    
    model.eval()
    correct = 0
    total_tokens = 0
    correct_tokens = 0
    
    with torch.no_grad():
        for entry in entries:
            target = get_target_symbols(entry, task)
            
            # 生成输入波形
            input_wave = synthesise_entry_wave(entry)
            
            # 转换为帧
            wave_tensor = torch.from_numpy(input_wave).float()
            frame_size = model.config.frame_size
            hop_size = model.config.hop_size
            
            # 分帧（使用 safe_unfold 避免丢失尾部数据）
            frames, valid_frames = safe_unfold(wave_tensor, frame_size, hop_size)
            
            frames = frames.unsqueeze(0).to(device)
            mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
            
            # 模型推理
            frame_out, symbol_logits = model(frames, mask)
            
            # 解码：mod 任务用 CTC 解码（只在答案窗口内，与训练一致）
            if task == "mod" and symbol_logits is not None:
                # 计算答案窗口起始帧（与 task3_mod_audio.py 保持一致）
                # 表达式长度（samples）= 符号数 * symbol_duration
                SR = 16000
                SYMBOL_DURATION = 0.05  # 50ms per symbol
                expr_len_samples = int(len(entry.symbols) * SYMBOL_DURATION * SR)
                # 对齐到 hop_size
                align = hop_size
                expr_len_aligned = ((expr_len_samples + align - 1) // align) * align
                # thinking gap（默认 0.5s）
                thinking_gap_s = checkpoint.get("thinking_gap_s", 0.5)
                thinking_gap_samples = int(round(thinking_gap_s * SR))
                thinking_gap_aligned = ((thinking_gap_samples + align - 1) // align) * align if thinking_gap_samples > 0 else 0
                # 答案窗口起始帧
                answer_start_samples = expr_len_aligned + thinking_gap_aligned
                answer_start_frame = answer_start_samples // hop_size
                
                # 只在答案窗口内做 CTC 解码
                probs = symbol_logits.softmax(dim=-1).squeeze(0)
                # 截取答案窗口部分
                if answer_start_frame < probs.size(0):
                    window_probs = probs[answer_start_frame:]
                else:
                    window_probs = probs[-1:]  # fallback
                
                pred_ids = window_probs.argmax(dim=-1).cpu().tolist()
                # 简单 CTC 解码：去重 + 去 blank(0)
                decoded = []
                prev_id = None
                id_to_symbol = checkpoint.get("id_to_symbol", {})
                for idx in pred_ids:
                    if idx != 0 and idx != prev_id:  # 0 是 blank
                        symbol = id_to_symbol.get(idx, str(idx))
                        decoded.append(symbol)
                    prev_id = idx
            else:
                # 音频解码
                pred_wave = frame_out.squeeze(0).cpu().numpy().flatten()
                pred_wave = np.clip(pred_wave, -1.0, 1.0).astype(np.float32)
                decoded = decode_wave_to_symbols(pred_wave)
            
            if decoded == target:
                correct += 1
            
            min_len = min(len(decoded), len(target))
            for i in range(min_len):
                if decoded[i] == target[i]:
                    correct_tokens += 1
            total_tokens += max(len(decoded), len(target))
    
    total = len(entries)
    return EvalResult(
        task=task, split=split, eval_type="model",
        total=total, correct=correct,
        em=correct / total if total > 0 else 0.0,
        token_accuracy=correct_tokens / total_tokens if total_tokens > 0 else 0.0,
    )


def run_ablation_ood(
    manifest_path: Path,
    task: str = "mirror",
    splits: list[str] = None,
    epochs: int = 15,
    device: str = "cpu",
    limit: int = 50,
    seed: int = 42,
    extreme_ood: bool = True,  # 使用更极端的 OOD 设置
) -> dict[str, list[EvalResult]]:
    """运行消融 OOD 评测"""
    from experiments.run_ablations import CORE_ABLATIONS, build_model_with_ablation
    
    if splits is None:
        splits = ["iid_test", "ood_length"]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 读取数据
    entries = list(read_manifest(manifest_path))
    
    # 构建词表
    symbols = set()
    for e in entries:
        symbols.update(e.symbols)
    symbol_to_id = {s: i + 1 for i, s in enumerate(sorted(symbols))}
    vocab_size = len(symbol_to_id) + 1
    
    # 基础配置
    base_config = MiniJMambaConfig(
        frame_size=160,
        hop_size=160,
        symbol_vocab_size=vocab_size,
        d_model=64,
        num_ssm_layers=4,
        num_attn_layers=1,
        num_heads=2,
        max_frames=256,
        dropout=0.1,
        attn_dropout=0.1,
        use_rope=True,
        use_learnable_pos=False,
    )
    
    results: dict[str, list[EvalResult]] = {}
    
    for ablation_name, ablation in CORE_ABLATIONS.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_name} - {ablation.description}")
        print(f"{'='*60}")
        
        # 构建模型
        model = build_model_with_ablation(base_config, ablation)
        model = model.to(device)
        
        # 快速训练
        train_entries = [e for e in entries if e.split == "train"][:limit * 2]
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ctc_weight = ablation.ctc_weight if ablation.ctc_weight is not None else 0.3
        ctc_loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for entry in train_entries:
                wave = synthesise_entry_wave(entry)
                wave_tensor = torch.from_numpy(wave).float()
                
                # 使用 safe_unfold 避免丢失尾部数据
                frames, valid_frames = safe_unfold(wave_tensor, 160, 160)
                if frames.size(0) == 0:
                    continue
                
                frames = frames.unsqueeze(0).to(device)
                mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
                
                frame_out, symbol_logits = model(frames, mask)
                
                audio_loss = ((frame_out - frames) ** 2).mean()
                
                target_ids = [symbol_to_id.get(s, 0) for s in entry.symbols]
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
        
        # 评测各 split
        ablation_results = []
        for split in splits:
            result = eval_model(model, manifest_path, task, split, device, limit, None)
            result = EvalResult(
                task=result.task,
                split=result.split,
                eval_type=ablation_name,  # 使用 ablation 名称作为 eval_type
                total=result.total,
                correct=result.correct,
                em=result.em,
                token_accuracy=result.token_accuracy,
            )
            ablation_results.append(result)
            print(f"  {split}: EM={result.em:.4f}, Token={result.token_accuracy:.4f}")
        
        results[ablation_name] = ablation_results
    
    return results


def generate_summary_report(
    oracle_results: list[EvalResult],
    model_results: list[EvalResult] | None,
    output_path: Path,
) -> None:
    """生成总表报告"""
    lines = [
        "# 评测总表",
        "",
        f"> 生成时间: {datetime.now().isoformat()}",
        "",
        "## 口径说明",
        "",
        "| 指标 | 定义 |",
        "|------|------|",
        "| **Oracle EM** | 系统闭环验证：目标波形→FFT解码→符号 == 原始符号 |",
        "| **Model EM** | 训练模型能力：输入→模型→FFT解码→符号 == 目标符号 |",
        "",
        "---",
        "",
        "## Oracle/Protocol 验证",
        "",
        "| Task | Split | Oracle EM | Token Acc | Total |",
        "|------|-------|-----------|-----------|-------|",
    ]
    
    for r in oracle_results:
        status = "✅" if r.em >= 0.95 else "⚠️"
        lines.append(f"| {r.task} | {r.split} | {r.em:.4f} {status} | {r.token_accuracy:.4f} | {r.total} |")
    
    if model_results:
        lines.extend([
            "",
            "---",
            "",
            "## Model 能力（训练后）",
            "",
            "| Task | Split | Model EM | Token Acc | Total |",
            "|------|-------|----------|-----------|-------|",
        ])
        
        for r in model_results:
            status = "✅" if r.em >= 0.95 else "⚠️" if r.em >= 0.80 else "❌"
            lines.append(f"| {r.task} | {r.split} | {r.em:.4f} {status} | {r.token_accuracy:.4f} | {r.total} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 对比汇总",
        "",
    ])
    
    if model_results:
        # 按 task+split 对比
        oracle_map = {(r.task, r.split): r for r in oracle_results}
        model_map = {(r.task, r.split): r for r in model_results}
        
        lines.extend([
            "| Task | Split | Oracle EM | Model EM | Gap |",
            "|------|-------|-----------|----------|-----|",
        ])
        
        for key in oracle_map:
            o = oracle_map[key]
            m = model_map.get(key)
            if m:
                gap = o.em - m.em
                lines.append(f"| {key[0]} | {key[1]} | {o.em:.4f} | {m.em:.4f} | {gap:+.4f} |")
    else:
        lines.append("*Model 结果未提供*")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {output_path}")
    
    # JSON 版本
    json_path = output_path.with_suffix(".json")
    data = {
        "timestamp": datetime.now().isoformat(),
        "oracle_results": [r.to_dict() for r in oracle_results],
        "model_results": [r.to_dict() for r in model_results] if model_results else None,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Generated: {json_path}")


def generate_ablation_ood_report(
    results: dict[str, list[EvalResult]],
    output_path: Path,
) -> None:
    """生成消融 OOD 曲线报告"""
    lines = [
        "# S22 消融 OOD 曲线报告",
        "",
        f"> 生成时间: {datetime.now().isoformat()}",
        "",
        "## 说明",
        "",
        "本报告展示各消融配置在 IID 和 OOD 上的性能差异，用于证明关键组件的必要性。",
        "",
        "---",
        "",
        "## 结果矩阵",
        "",
    ]
    
    # 获取所有 splits
    all_splits = set()
    for ablation_results in results.values():
        for r in ablation_results:
            all_splits.add(r.split)
    splits = sorted(all_splits)
    
    # 表头
    header = "| Ablation |"
    separator = "|----------|"
    for split in splits:
        header += f" {split} EM |"
        separator += "----------|"
    header += " IID→OOD Gap |"
    separator += "------------|"
    
    lines.extend([header, separator])
    
    # 数据行
    baseline_iid = None
    for ablation_name, ablation_results in results.items():
        row = f"| {ablation_name} |"
        iid_em = None
        ood_em = None
        
        for split in splits:
            r = next((x for x in ablation_results if x.split == split), None)
            if r:
                row += f" {r.em:.4f} |"
                if "iid" in split:
                    iid_em = r.em
                elif "ood" in split:
                    ood_em = r.em
            else:
                row += " N/A |"
        
        # 计算 gap
        if iid_em is not None and ood_em is not None:
            gap = iid_em - ood_em
            gap_status = "✅" if gap < 0.1 else "⚠️" if gap < 0.2 else "❌"
            row += f" {gap:+.4f} {gap_status} |"
        else:
            row += " N/A |"
        
        lines.append(row)
        
        if ablation_name == "baseline" and iid_em:
            baseline_iid = iid_em
    
    lines.extend([
        "",
        "---",
        "",
        "## 关键发现",
        "",
    ])
    
    # 分析
    for ablation_name, ablation_results in results.items():
        if ablation_name == "baseline":
            continue
        
        iid_r = next((r for r in ablation_results if "iid" in r.split), None)
        ood_r = next((r for r in ablation_results if "ood" in r.split), None)
        
        if iid_r and ood_r:
            iid_em = iid_r.em
            ood_em = ood_r.em
            gap = iid_em - ood_em
            
            baseline_results = results.get("baseline", [])
            baseline_ood = next((r for r in baseline_results if "ood" in r.split), None)
            baseline_ood_em = baseline_ood.em if baseline_ood else 0
            
            ood_delta = ood_em - baseline_ood_em
            
            if ood_delta < -0.1:
                lines.append(f"- **{ablation_name}**: OOD 性能显著下降 ({ood_delta:+.3f})，证明该组件对泛化必要")
            elif ood_delta < -0.05:
                lines.append(f"- **{ablation_name}**: OOD 性能轻微下降 ({ood_delta:+.3f})")
            else:
                lines.append(f"- **{ablation_name}**: OOD 性能无显著变化 ({ood_delta:+.3f})")
    
    lines.extend([
        "",
        "---",
        "",
        "## 结论",
        "",
        "TODO: 根据实际结果填写组件必要性结论",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="模型评测总表")
    parser.add_argument("--checkpoint", type=Path, default=None,
                       help="模型检查点路径（单任务）")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                       help="检查点目录（自动匹配 {task}_*.pt 或 *_{task}.pt）")
    parser.add_argument("--tasks", type=str, nargs="+", default=["mirror", "bracket", "mod"],
                       help="要评估的任务")
    parser.add_argument("--splits", type=str, nargs="+",
                       default=["iid_test", "ood_length"],
                       help="要评估的 splits")
    parser.add_argument("--manifests-dir", type=Path, default=Path("manifests"),
                       help="manifest 目录")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"),
                       help="输出目录")
    parser.add_argument("--limit", type=int, default=50,
                       help="每个 split 的样本数限制")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--oracle-only", action="store_true",
                       help="仅运行 Oracle 验证")
    parser.add_argument("--ablation-ood", action="store_true",
                       help="运行消融 OOD 评测")
    parser.add_argument("--ablation-epochs", type=int, default=15,
                       help="消融训练 epochs")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 自动探测 manifest 文件（兼容不同命名）
    def find_manifest(task: str) -> Path:
        candidates = {
            "mirror": ["task1.jsonl"],
            "bracket": ["task2.jsonl"],
            "mod": ["task3.jsonl", "task3_multistep.jsonl"],  # 优先 task3.jsonl
        }
        for name in candidates.get(task, []):
            path = args.manifests_dir / name
            if path.exists():
                return path
        # 返回第一个候选（即使不存在，后续会报错）
        return args.manifests_dir / candidates[task][0]
    
    manifests = {
        "mirror": find_manifest("mirror"),
        "bracket": find_manifest("bracket"),
        "mod": find_manifest("mod"),
    }
    
    print(f"Tasks: {args.tasks}")
    print(f"Splits: {args.splits}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # 1. Oracle 评测
    print("\n[1/3] Oracle/Protocol 评测...")
    oracle_results = []
    for task in args.tasks:
        manifest_path = manifests.get(task)
        if not manifest_path or not manifest_path.exists():
            print(f"  Skip {task}: manifest not found")
            continue
        
        for split in args.splits:
            result = eval_oracle(manifest_path, task, split, args.limit)
            oracle_results.append(result)
            print(f"  {task}/{split}: Oracle EM={result.em:.4f}")
    
    model_results = []
    
    # 辅助函数：查找任务对应的 checkpoint
    def find_checkpoint_for_task(task: str) -> Path | None:
        if args.checkpoint_dir and args.checkpoint_dir.exists():
            # 尝试多种命名模式
            patterns = [
                f"{task}_*.pt",
                f"*_{task}.pt",
                f"{task}.pt",
                f"*{task}*.pt",
            ]
            for pattern in patterns:
                matches = list(args.checkpoint_dir.glob(pattern))
                if matches:
                    # 返回最新的（按修改时间）
                    return max(matches, key=lambda p: p.stat().st_mtime)
        return None
    
    # 辅助函数：加载 checkpoint 并返回模型
    def load_model_from_checkpoint(ckpt_path: Path) -> tuple[MiniJMamba, dict, str]:
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        config_dict = checkpoint["config"]
        
        config = MiniJMambaConfig(
            frame_size=config_dict["frame_size"],
            hop_size=config_dict["hop_size"],
            symbol_vocab_size=config_dict["symbol_vocab_size"],
            d_model=config_dict["d_model"],
            num_ssm_layers=config_dict["num_ssm_layers"],
            num_attn_layers=config_dict["num_attn_layers"],
            num_heads=config_dict["num_heads"],
            max_frames=config_dict["max_frames"],
            use_rope=config_dict.get("use_rope", True),
            use_learnable_pos=config_dict.get("use_learnable_pos", False),
        )
        
        model = MiniJMamba(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(args.device)
        
        task = checkpoint.get("task", "mirror")
        return model, checkpoint, task
    
    # 2. Model 评测
    if args.checkpoint and args.checkpoint.exists():
        # 单 checkpoint 模式
        print(f"\n[2/4] Model 评测 (checkpoint: {args.checkpoint})...")
        model, checkpoint, task = load_model_from_checkpoint(args.checkpoint)
        manifest_path = manifests.get(task)
        
        if manifest_path and manifest_path.exists():
            for split in args.splits:
                result = eval_model(model, manifest_path, task, split, args.device, args.limit, checkpoint)
                model_results.append(result)
                print(f"  {task}/{split}: Model EM={result.em:.4f}")
                
    elif args.checkpoint_dir and args.checkpoint_dir.exists():
        # 多任务自动匹配模式
        print(f"\n[2/4] Model 评测 (checkpoint-dir: {args.checkpoint_dir})...")
        
        for task in args.tasks:
            ckpt_path = find_checkpoint_for_task(task)
            if not ckpt_path:
                print(f"  Skip {task}: no checkpoint found")
                continue
            
            print(f"  Loading: {ckpt_path.name}")
            model, checkpoint, _ = load_model_from_checkpoint(ckpt_path)
            manifest_path = manifests.get(task)
            
            if manifest_path and manifest_path.exists():
                for split in args.splits:
                    result = eval_model(model, manifest_path, task, split, args.device, args.limit, checkpoint)
                    model_results.append(result)
                    print(f"  {task}/{split}: Model EM={result.em:.4f}")
                    
    elif not args.oracle_only:
        print("\n[2/4] Model 评测跳过 (无 checkpoint 或 checkpoint-dir)")
    
    # 3. 消融 OOD（如果请求）
    if args.ablation_ood:
        print("\n[3/4] 消融 OOD 评测...")
        # 使用 Task1 mirror 进行消融，因为 mirror 任务训练目标最直接
        ablation_manifest = manifests["mirror"]
        ablation_splits = ["iid_test", "ood_length"]
        ablation_task = "mirror"
        
        ablation_results = run_ablation_ood(
            manifest_path=ablation_manifest,
            task=ablation_task,
            splits=ablation_splits,
            epochs=args.ablation_epochs,
            device=args.device,
            limit=args.limit,
            seed=args.seed,
        )
        generate_ablation_ood_report(
            ablation_results,
            args.output_dir / "ablation_ood.md",
        )
    
    # 4. 生成总表
    print("\n[4/4] 生成总表...")
    generate_summary_report(
        oracle_results,
        model_results,
        args.output_dir / "eval_summary.md",
    )
    
    print("\n" + "=" * 60)
    print("评测完成!")


if __name__ == "__main__":
    main()

